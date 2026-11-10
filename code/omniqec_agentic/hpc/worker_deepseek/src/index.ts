/**
 * NSCC -> LLM gateway (Cloudflare Worker).
 *
 * NSCC's firewall resets the TLS handshake to every AI API host from every
 * node, but workers.dev is reachable (verified 2026-08-18). This Worker is the
 * whole bridge: it forwards the two endpoints a campaign needs straight to the
 * provider and streams the response back.
 *
 * One deployment serves several providers, selected by a path segment after
 * the token. The bare (provider-less) path stays pinned to UPSTREAM_ORIGIN so
 * an already-configured DEEPSEEK_BASE_URL keeps working byte for byte.
 *
 * It deliberately holds NO credential. The caller's Authorization header is
 * passed through untouched, so the DeepSeek key never leaves the cluster's
 * private env file and a leaked Worker URL cannot spend anyone's quota.
 *
 * GATEWAY_TOKEN (optional secret) requires the URL to start with a secret path
 * segment. It rides inside DEEPSEEK_BASE_URL, so enabling it needs no client
 * code change at all.
 */

interface Env {
  UPSTREAM_ORIGIN: string; // default upstream for the provider-less path
  UPSTREAM_DEEPSEEK?: string;
  UPSTREAM_OPENAI?: string;
  UPSTREAM_CLIPROXY?: string;
  UPSTREAM_OPENROUTER?: string;
  GATEWAY_TOKEN?: string;
  // JSON object mapping a requested model name to the name the upstream
  // expects, e.g. {"gpt-5.6-sol":"openai/gpt-5.6-sol"}. A resumed campaign
  // must keep requesting the model name frozen in its protocol snapshot
  // (required_model — the strict audit compares every recorded call against
  // it), but OpenRouter only accepts vendor-prefixed ids. Translating at the
  // edge keeps the on-cluster record on the frozen name while the response's
  // `model` field still reports what the upstream actually served, so the
  // substitution stays visible in served_model rather than being hidden.
  // Applies ONLY on the /openrouter path — every other upstream accepts the
  // frozen name directly, and a global rewrite would break their live runs.
  MODEL_MAP?: string;
  // Per-caller policies (SECRET, JSON). Lets one CLIProxyAPI be shared with
  // people who must only reach SOME of its credentials: CLIProxyAPI's own
  // downstream api-keys cannot be scoped to models or credentials, so the
  // scoping happens here. Each caller gets its own path token and bearer;
  // the Worker checks the requested model against `models` (glob patterns)
  // and swaps the bearer for an internal proxy key the caller never sees.
  //   {"callers":[{"path":"<path token>","bearer":"<caller key>",
  //                "key":"<internal proxy key>","provider":"cliproxy",
  //                "models":["claude-*"]}]}
  // Callers outside this table (the GATEWAY_TOKEN path) are untouched.
  CALLER_POLICY?: string;
}

interface CallerPolicy {
  path: string;
  bearer: string;
  key: string;
  provider: string;
  models: string[];
}

function parseCallers(raw: string | undefined): CallerPolicy[] {
  try {
    const parsed = JSON.parse((raw || "").trim() || "{}") as { callers?: CallerPolicy[] };
    return Array.isArray(parsed.callers) ? parsed.callers : [];
  } catch {
    return [];
  }
}

// Glob match with `*` only (claude-* / gpt-5.6-*), case-sensitive.
function modelAllowed(model: string, patterns: string[]): boolean {
  return patterns.some((p) => {
    const re = new RegExp("^" + p.split("*").map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join(".*") + "$");
    return re.test(model);
  });
}

// Provider segment -> env var holding its origin. Adding a provider is one
// entry here plus one var in wrangler.jsonc; nothing else changes.
const UPSTREAM_VARS: Record<string, keyof Env> = {
  deepseek: "UPSTREAM_DEEPSEEK",
  openai: "UPSTREAM_OPENAI",
  // A CLIProxyAPI instance behind an ephemeral tunnel. Its URL changes every
  // time the tunnel restarts, so this var is meant to be rewritten at deploy
  // time (`wrangler deploy --var UPSTREAM_CLIPROXY:https://...`) rather than
  // committed — see hpc/start_cliproxy_tunnel.sh.
  cliproxy: "UPSTREAM_CLIPROXY",
  openrouter: "UPSTREAM_OPENROUTER",
};

// Exactly what a campaign calls. An allow-list rather than a prefix match, so
// the Worker cannot be repurposed as a general-purpose open relay.
const ALLOWED_PATHS = new Set(["/v1/chat/completions", "/v1/models"]);

function json(body: unknown, status: number): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const incoming = new URL(request.url);
    let pathname = incoming.pathname;

    // A scoped caller's own path token takes precedence over GATEWAY_TOKEN;
    // its policy is enforced further down once the provider/path are known.
    let caller: CallerPolicy | undefined;
    for (const c of parseCallers(env.CALLER_POLICY)) {
      if (c.path && pathname.startsWith(`/${c.path}/`)) {
        caller = c;
        pathname = pathname.slice(c.path.length + 1);
        break;
      }
    }

    const token = (env.GATEWAY_TOKEN || "").trim();
    if (!caller && token) {
      const prefix = `/${token}`;
      if (!pathname.startsWith(`${prefix}/`)) {
        return new Response("Not Found", { status: 404 });
      }
      pathname = pathname.slice(prefix.length);
    }

    // A leading provider segment picks the upstream; without one the request
    // goes to UPSTREAM_ORIGIN, which is what every existing client sends.
    let origin = env.UPSTREAM_ORIGIN;
    const segments = pathname.split("/");
    const provider = segments[1] || "";
    if (provider in UPSTREAM_VARS) {
      const configured = (env[UPSTREAM_VARS[provider]] as string | undefined) || "";
      if (!configured) {
        return json({ error: `upstream not configured for '${provider}'` }, 503);
      }
      origin = configured;
      pathname = "/" + segments.slice(2).join("/");
    }

    if (!ALLOWED_PATHS.has(pathname)) {
      return new Response("Not Found", { status: 404 });
    }

    // Fail here rather than forwarding an anonymous request upstream: the
    // error is clearer, and an unauthenticated caller never reaches DeepSeek.
    if (!request.headers.get("Authorization")) {
      return json({ error: "missing Authorization header" }, 401);
    }

    // Keep whatever path the upstream origin already carries. DeepSeek and
    // OpenAI are bare hosts, so this is a no-op for them — but OpenRouter's
    // API lives under /api, and assigning `pathname` outright would have
    // dropped that prefix and sent every call to openrouter.ai/v1/... (404).
    const target = new URL(origin);
    target.pathname = target.pathname.replace(/\/+$/, "") + pathname;
    target.search = incoming.search;

    // Buffer the request body instead of forwarding the stream. Streaming a
    // body makes the runtime send it chunked, and DeepSeek sits behind
    // CloudFront + ELB, which stalls on a chunked request body until the
    // connection is dropped — every POST died at exactly 30s in production
    // while GET worked and local `wrangler dev` (which buffers) looked fine.
    // Campaign payloads are a few KB, so buffering costs nothing.
    const hasBody = request.method !== "GET" && request.method !== "HEAD";
    let body: BodyInit | undefined = hasBody
      ? await request.arrayBuffer()
      : undefined;

    // Scoped caller: right provider, right bearer, allowed model; then the
    // bearer is swapped for the internal proxy key.
    if (caller) {
      if (provider !== caller.provider) {
        return new Response("Not Found", { status: 404 });
      }
      if (request.headers.get("Authorization") !== `Bearer ${caller.bearer}`) {
        return json({ error: "invalid api key" }, 401);
      }
      if (hasBody && pathname === "/v1/chat/completions") {
        let model = "";
        try {
          model = String(JSON.parse(new TextDecoder().decode(body as ArrayBuffer))?.model ?? "");
        } catch {
          return json({ error: "request body is not JSON" }, 400);
        }
        if (!modelAllowed(model, caller.models)) {
          return json(
            { error: { message: `model '${model}' is not available to this key`, type: "permission_error", code: "model_not_allowed" } },
            403,
          );
        }
      }
    }

    // Optional model-name translation (see MODEL_MAP above). A string body is
    // still sent with an explicit Content-Length, so the chunked-POST stall
    // this buffering exists to avoid cannot come back through this path.
    // Scoped to the openrouter path: the other upstreams accept the frozen
    // model name as-is, and rewriting it there would break a live campaign
    // the moment this map is deployed.
    const mapRaw = (env.MODEL_MAP || "").trim();
    if (
      mapRaw &&
      hasBody &&
      provider === "openrouter" &&
      pathname === "/v1/chat/completions"
    ) {
      try {
        const map = JSON.parse(mapRaw) as Record<string, string>;
        const payload = JSON.parse(new TextDecoder().decode(body as ArrayBuffer));
        const mapped = map[payload?.model];
        if (typeof mapped === "string" && mapped) {
          payload.model = mapped;
          body = JSON.stringify(payload);
        }
      } catch {
        // Not JSON, or no model field: forward the original buffer untouched.
      }
    }

    const outboundHeaders = new Headers(request.headers);
    outboundHeaders.delete("host");
    outboundHeaders.delete("content-length"); // recomputed from the buffer
    outboundHeaders.delete("accept-encoding"); // let the runtime negotiate
    if (caller) {
      // The caller's key never reaches the proxy; the internal one does.
      outboundHeaders.set("Authorization", `Bearer ${caller.key}`);
    }

    // 2026-09-11: the edge answers HTTP 530 "error code: 1016" (origin DNS
    // error) on roughly one request in four to the Tailscale Funnel origin,
    // and occasionally 52x. Those never reached the origin, so the request is
    // safe to repeat; three quick retries turn a 25% failure rate into well
    // under 1% and spare the campaign a strict-turn restart per hiccup.
    // Everything else (4xx, 5xx from the provider itself) passes through.
    const RETRY_DELAYS_MS = [500, 2000, 5000];
    const isEdgeOriginError = (r: Response) =>
      r.status === 530 || (r.status >= 520 && r.status <= 527);
    let upstream: Response | undefined;
    let lastError: unknown = undefined;
    for (let attempt = 0; attempt <= RETRY_DELAYS_MS.length; attempt++) {
      try {
        upstream = await fetch(target.toString(), {
          method: request.method,
          headers: outboundHeaders,
          body,
        });
        lastError = undefined;
        if (!isEdgeOriginError(upstream)) break;
      } catch (error) {
        lastError = error;
        upstream = undefined;
      }
      if (attempt < RETRY_DELAYS_MS.length) {
        await new Promise((resolve) => setTimeout(resolve, RETRY_DELAYS_MS[attempt]));
      }
    }
    if (upstream === undefined) {
      // Surfaced as 502 so a campaign log distinguishes "gateway could not
      // reach the provider" from "the provider rejected the request".
      return json(
        { error: "upstream fetch failed", detail: String(lastError) },
        502,
      );
    }
    // The runtime already decoded the upstream body, but the upstream's
    // Content-Encoding header would still claim it is br/gzip — a client then
    // tries to decompress plain JSON and reads garbage. Drop the framing
    // headers and let the edge re-encode for the caller.
    const headers = new Headers(upstream.headers);
    headers.delete("content-encoding");
    headers.delete("content-length");
    // A scoped caller only sees the models it may call.
    if (caller && pathname === "/v1/models" && upstream.status === 200) {
      try {
        const listing = (await upstream.json()) as { data?: Array<{ id?: string }> };
        if (Array.isArray(listing.data)) {
          listing.data = listing.data.filter((m) => modelAllowed(String(m?.id ?? ""), caller!.models));
        }
        return new Response(JSON.stringify(listing), { status: 200, headers: { "content-type": "application/json" } });
      } catch {
        return json({ error: "upstream model list was not JSON" }, 502);
      }
    }
    return new Response(upstream.body, {
      status: upstream.status,
      statusText: upstream.statusText,
      headers,
    });
  },
} satisfies ExportedHandler<Env>;

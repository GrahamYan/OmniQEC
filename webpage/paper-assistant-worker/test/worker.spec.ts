import { env, exports } from "cloudflare:workers";
import { describe, expect, it, vi } from "vitest";
import { handleChat, type ModelRunner } from "../src/index";
import {
  MAX_HISTORY_CHARS,
  MAX_HISTORY_ITEMS,
  cleanHistory,
  retrieveContext,
} from "../src/retrieval";

const allowedOrigin = "http://127.0.0.1:8000";

function request(
  path: string,
  init: RequestInit = {},
  origin: string | null = allowedOrigin,
): Request {
  const headers = new Headers(init.headers);
  if (origin) headers.set("Origin", origin);
  return new Request(`https://assistant.test${path}`, { ...init, headers });
}

function chatRequest(message: string, history: unknown[] = []): Request {
  return request("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history }),
  });
}

describe("retrieval and history utilities", () => {
  it("ranks the slow loop first for a circuit-level evaluation query", () => {
    const chunks = retrieveContext(
      "How does the slow loop compile circuits and evaluate decoder performance?",
    );
    expect(chunks[0]?.id).toBe("Slow Loop");
    expect(chunks.length).toBeLessThanOrEqual(5);
  });

  it("keeps only six valid, trimmed, bounded history turns", () => {
    const raw = Array.from({ length: 9 }, (_, index) => ({
      role: index % 2 ? "assistant" : "user",
      content: ` turn ${index} ${"x".repeat(1_000)} `,
    }));
    const history = cleanHistory(raw);
    expect(history).toHaveLength(MAX_HISTORY_ITEMS);
    expect(history[0]?.content.startsWith("turn 3")).toBe(true);
    expect(history.every((turn) => turn.content.length <= MAX_HISTORY_CHARS)).toBe(true);
  });
});

describe("OmniQEC assistant Worker", () => {
  it("returns health and the allowed CORS origin", async () => {
    const response = await exports.default.fetch(request("/health"));
    expect(response.status).toBe(200);
    expect(response.headers.get("Access-Control-Allow-Origin")).toBe(allowedOrigin);
    await expect(response.json()).resolves.toMatchObject({
      status: "ok",
      mode: "mock",
    });
  });

  it("answers an English grounded question with source labels", async () => {
    const response = await exports.default.fetch(
      chatRequest("What happens in the fast loop?"),
    );
    const result = (await response.json()) as {
      answer: string;
      sources: string[];
      model: string;
    };
    expect(response.status).toBe(200);
    expect(result.answer).toContain("According to the current preview");
    expect(result.sources[0]).toBe("Fast Loop");
    expect(result.model).toBe("local-grounded-mock");
  });

  it("answers a performance question in English", async () => {
    const response = await exports.default.fetch(
      chatRequest("What is the best result at p=0.002?"),
    );
    const result = (await response.json()) as { answer: string; sources: string[] };
    expect(result.answer).toContain("According to the current preview");
    expect(result.answer).toContain("29.23");
    expect(result.sources).toContain("Physical-error Sweep");
  });

  it("rejects empty and overlong messages", async () => {
    const empty = await exports.default.fetch(chatRequest("   "));
    const long = await exports.default.fetch(chatRequest("x".repeat(901)));
    expect(empty.status).toBe(400);
    expect(long.status).toBe(400);
  });

  it("does not invent an answer when no manuscript section matches", async () => {
    const response = await exports.default.fetch(
      chatRequest("Who won the 1974 football championship?"),
    );
    const result = (await response.json()) as { answer: string; sources: string[] };
    expect(result.answer).toContain("does not provide");
    expect(result.sources).toEqual([]);
  });

  it("does not invent unpublished code details", async () => {
    const response = await exports.default.fetch(
      chatRequest("Show me the GitHub source code and repository structure."),
    );
    const result = (await response.json()) as { answer: string };
    expect(result.answer).toContain("formal version");
  });

  it("rejects a disallowed origin and accepts a valid preflight", async () => {
    const rejected = await exports.default.fetch(
      request("/health", {}, "https://untrusted.example"),
    );
    const preflight = await exports.default.fetch(
      request("/chat", { method: "OPTIONS" }),
    );
    expect(rejected.status).toBe(403);
    expect(rejected.headers.get("Access-Control-Allow-Origin")).toBeNull();
    expect(preflight.status).toBe(204);
    expect(preflight.headers.get("Access-Control-Allow-Methods")).toContain("POST");
  });

  it("returns a generic 502 when the model fails", async () => {
    const failingRunner: ModelRunner = vi.fn().mockRejectedValue(new Error("secret"));
    const productionEnv = {
      ...env,
      ASSISTANT_MODE: "workers-ai",
      MODEL: "@cf/zai-org/glm-4.7-flash",
    } as Env;
    const response = await handleChat(
      chatRequest("What is OmniQEC?"),
      productionEnv,
      failingRunner,
    );
    const result = (await response.json()) as { error: string };
    expect(response.status).toBe(502);
    expect(result.error).not.toContain("secret");
  });
});

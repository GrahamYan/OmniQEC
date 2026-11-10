import type { KnowledgeChunk } from "./knowledge";
import {
  MAX_MESSAGE_CHARS,
  cleanHistory,
  isCodeQuestion,
  retrieveContext,
  type ChatTurn,
} from "./retrieval";

interface ChatRequestBody {
  message?: unknown;
  history?: unknown;
}

interface AssistantResponse {
  answer: string;
  sources: string[];
  model: string;
}

interface ModelMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export type ModelRunner = (
  model: string,
  input: { messages: ModelMessage[]; max_tokens: number; temperature: number },
) => Promise<unknown>;

const JSON_HEADERS = {
  "Content-Type": "application/json; charset=utf-8",
  "Cache-Control": "no-store",
};

function getAllowedOrigins(env: Env): Set<string> {
  return new Set(
    env.ALLOWED_ORIGINS.split(",")
      .map((origin) => origin.trim())
      .filter(Boolean),
  );
}

function getRequestOrigin(request: Request): string | null {
  return request.headers.get("Origin");
}

function corsHeaders(origin: string | null, env: Env): HeadersInit {
  if (!origin || !getAllowedOrigins(env).has(origin)) return {};
  return {
    "Access-Control-Allow-Origin": origin,
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Max-Age": "86400",
    Vary: "Origin",
  };
}

function jsonResponse(
  payload: unknown,
  status: number,
  request: Request,
  env: Env,
): Response {
  return Response.json(payload, {
    status,
    headers: {
      ...JSON_HEADERS,
      ...corsHeaders(getRequestOrigin(request), env),
    },
  });
}

function rejectDisallowedOrigin(request: Request, env: Env): Response | null {
  const origin = getRequestOrigin(request);
  if (!origin || getAllowedOrigins(env).has(origin)) return null;
  return jsonResponse({ error: "Origin is not allowed." }, 403, request, env);
}

function createContext(chunks: KnowledgeChunk[]): string {
  return chunks
    .map((chunk, index) => `[${index + 1}] ${chunk.id}\n${chunk.text}`)
    .join("\n\n");
}

function createSystemPrompt(chunks: KnowledgeChunk[]): string {
  const context = chunks.length
    ? createContext(chunks)
    : "No manuscript section matched this question.";

  return `You are the OmniQEC paper reading assistant.
Always answer in English.
Use only the supplied manuscript excerpts. Do not add outside facts or infer unpublished details.
Be concise and technically accurate. Keep the answer under about 180 words.
If the excerpts do not contain the answer, say that the current preview manuscript does not provide it.
If asked for source code, repository layout, or implementation details, state that the source code will be released with the formal version.

MANUSCRIPT EXCERPTS
${context}`;
}

function buildMessages(
  message: string,
  history: ChatTurn[],
  chunks: KnowledgeChunk[],
): ModelMessage[] {
  return [
    { role: "system", content: createSystemPrompt(chunks) },
    ...history.map((turn) => ({ role: turn.role, content: turn.content })),
    { role: "user", content: message },
  ];
}

function extractModelText(value: unknown): string {
  if (typeof value === "string") return value.trim();
  if (!value || typeof value !== "object") return "";

  const record = value as Record<string, unknown>;
  if (typeof record.response === "string") return record.response.trim();
  if (typeof record.result === "string") return record.result.trim();

  const choices = record.choices;
  if (Array.isArray(choices)) {
    const first = choices[0];
    if (first && typeof first === "object") {
      const message = (first as Record<string, unknown>).message;
      if (message && typeof message === "object") {
        const content = (message as Record<string, unknown>).content;
        if (typeof content === "string") return content.trim();
      }
    }
  }
  return "";
}

function mockAnswer(message: string, chunks: KnowledgeChunk[]): string {
  if (isCodeQuestion(message)) {
    return "The current preview does not release source code, repository structure, or implementation details. These will be made available with the formal version.";
  }

  if (!chunks.length) {
    return "The current preview manuscript does not provide information directly answering this question. You can ask about OmniQEC's workflow, code families, performance results, ablations, or cost.";
  }

  const selected = chunks.slice(0, 2);
  const body = selected.map((chunk) => chunk.text).join(" ");
  return `According to the current preview, ${body}`;
}

function defaultModelRunner(env: Env): ModelRunner {
  return async (model, input) => env.AI.run(model, input);
}

export async function handleChat(
  request: Request,
  env: Env,
  modelRunner: ModelRunner = defaultModelRunner(env),
): Promise<Response> {
  let body: ChatRequestBody;
  try {
    body = (await request.json()) as ChatRequestBody;
  } catch {
    return jsonResponse({ error: "Request body must be valid JSON." }, 400, request, env);
  }

  const message = typeof body.message === "string" ? body.message.trim() : "";
  if (!message) {
    return jsonResponse({ error: "Message must not be empty." }, 400, request, env);
  }
  if (message.length > MAX_MESSAGE_CHARS) {
    return jsonResponse(
      { error: `Message must be ${MAX_MESSAGE_CHARS} characters or fewer.` },
      400,
      request,
      env,
    );
  }

  const history = cleanHistory(body.history);
  const chunks = retrieveContext(message);
  const sources = chunks.map((chunk) => chunk.id);

  if (String(env.ASSISTANT_MODE) === "mock") {
    const response: AssistantResponse = {
      answer: mockAnswer(message, chunks),
      sources,
      model: "local-grounded-mock",
    };
    return jsonResponse(response, 200, request, env);
  }

  try {
    const result = await modelRunner(env.MODEL, {
      messages: buildMessages(message, history, chunks),
      max_tokens: 260,
      temperature: 0.15,
    });
    const answer = extractModelText(result);
    if (!answer) throw new Error("Model returned an empty response.");

    const response: AssistantResponse = {
      answer,
      sources,
      model: env.MODEL,
    };
    return jsonResponse(response, 200, request, env);
  } catch (error) {
    console.error("OmniQEC model invocation failed", {
      name: error instanceof Error ? error.name : "UnknownError",
      message: error instanceof Error ? error.message : "Unknown model failure",
    });
    return jsonResponse(
      { error: "The reading assistant could not complete this request." },
      502,
      request,
      env,
    );
  }
}

export async function handleRequest(request: Request, env: Env): Promise<Response> {
  const rejected = rejectDisallowedOrigin(request, env);
  if (rejected) return rejected;

  const url = new URL(request.url);

  if (request.method === "OPTIONS") {
    return new Response(null, {
      status: 204,
      headers: corsHeaders(getRequestOrigin(request), env),
    });
  }

  if (url.pathname === "/health" && request.method === "GET") {
    return jsonResponse(
      {
        status: "ok",
        service: "omniqec-paper-assistant",
        mode: String(env.ASSISTANT_MODE) === "mock" ? "mock" : "workers-ai",
      },
      200,
      request,
      env,
    );
  }

  if (url.pathname === "/chat" && request.method === "POST") {
    return handleChat(request, env);
  }

  return jsonResponse({ error: "Not found." }, 404, request, env);
}

export default {
  fetch: handleRequest,
} satisfies ExportedHandler<Env>;

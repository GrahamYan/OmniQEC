import { knowledgeChunks, type KnowledgeChunk } from "./knowledge";

export const MAX_MESSAGE_CHARS = 900;
export const MAX_HISTORY_ITEMS = 6;
export const MAX_HISTORY_CHARS = 800;
export const MAX_CONTEXT_CHUNKS = 5;

export interface ChatTurn {
  role: "user" | "assistant";
  content: string;
}

const termPattern = /[\p{L}\p{N}_.+\-[\]]+/gu;
const ignoredTerms = new Set([
  "a",
  "an",
  "and",
  "are",
  "can",
  "do",
  "does",
  "for",
  "from",
  "how",
  "i",
  "in",
  "is",
  "it",
  "me",
  "of",
  "on",
  "or",
  "please",
  "tell",
  "that",
  "the",
  "this",
  "to",
  "what",
  "why",
  "with",
  "you",
]);

function normalize(value: string): string {
  return value.normalize("NFKC").toLocaleLowerCase();
}

function tokenize(value: string): string[] {
  return Array.from(normalize(value).matchAll(termPattern), (match) => match[0]).filter(
    (term) => term.length > 1 && !ignoredTerms.has(term),
  );
}

function phraseScore(query: string, chunk: KnowledgeChunk): number {
  const searchable = normalize(
    [chunk.id, chunk.text, ...chunk.keywords].join(" "),
  );
  const exactTerms = new Set(
    Array.from(query.matchAll(termPattern), (match) => match[0]),
  );
  let score = 0;

  for (const keyword of chunk.keywords) {
    const normalizedKeyword = normalize(keyword);
    const matched =
      normalizedKeyword.length <= 2
        ? exactTerms.has(normalizedKeyword)
        : query.includes(normalizedKeyword);
    if (matched) {
      score += normalizedKeyword.includes(" ") ? 12 : 8;
    }
  }

  for (const term of tokenize(query)) {
    if (searchable.includes(term)) {
      score += term.length >= 6 ? 4 : 2;
    }
  }

  if (query.includes(normalize(chunk.id))) score += 14;
  return score;
}

export function retrieveContext(
  message: string,
  limit = MAX_CONTEXT_CHUNKS,
): KnowledgeChunk[] {
  const query = normalize(message.trim());
  if (!query) return [];

  return knowledgeChunks
    .map((chunk, index) => ({
      chunk,
      index,
      score: phraseScore(query, chunk),
    }))
    .filter(({ score }) => score > 0)
    .sort((left, right) => right.score - left.score || left.index - right.index)
    .slice(0, Math.max(0, limit))
    .map(({ chunk }) => chunk);
}

export function cleanHistory(value: unknown): ChatTurn[] {
  if (!Array.isArray(value)) return [];

  return value
    .filter(
      (turn): turn is Record<string, unknown> =>
        typeof turn === "object" && turn !== null,
    )
    .map((turn) => ({
      role: turn.role,
      content: typeof turn.content === "string" ? turn.content.trim() : "",
    }))
    .filter(
      (turn): turn is ChatTurn =>
        (turn.role === "user" || turn.role === "assistant") &&
        turn.content.length > 0,
    )
    .slice(-MAX_HISTORY_ITEMS)
    .map((turn) => ({
      ...turn,
      content: turn.content.slice(0, MAX_HISTORY_CHARS),
    }));
}

export function isCodeQuestion(value: string): boolean {
  return /(?:source\s*code|repository|github|implementation|codebase)/iu.test(
    value,
  );
}

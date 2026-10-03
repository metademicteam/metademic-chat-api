/**
 * Metademic Chat API — a provider-agnostic, streaming LLM/VLM proxy for the
 * /chat page on the Metademic Research Lab site.
 *
 * The underlying LLM/VLM is entirely environment-driven — swapping providers
 * (or pointing at your own hosted LLM/VLM) only requires changing environment
 * variables in the Vercel project settings, never code:
 *
 *   LLM_API_KEY     required — API key for the chosen provider
 *   LLM_BASE_URL    optional — OpenAI-compatible base URL
 *                              (default: https://api.together.xyz/v1)
 *   LLM_MODEL       optional — default chat model id
 *                              (default: deepseek-ai/DeepSeek-V4-Flash-0731)
 *   VLM_MODEL       optional — model used when images are attached
 *   ALLOWED_MODELS  optional — CSV of model ids clients may select
 *   CHAT_SYSTEM_PROMPT optional — overrides the built-in assistant prompt
 *   ALLOWED_ORIGINS optional — CSV of CORS origins (default: local dev only)
 *   SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY — persist chat history
 */
import type { IncomingMessage, ServerResponse } from "node:http";
import OpenAI from "openai";

export type ChatRole = "user" | "assistant" | "system";

export interface ChatMessageIn {
  role: ChatRole;
  content: string;
}

export interface ChatBody {
  sessionId?: string;
  messages: ChatMessageIn[];
  model?: string;
  effort?: "Low" | "Medium" | "Max Effort" | null;
  images?: string[];
  attachments?: { name?: string }[];
}

export const DEFAULT_BASE_URL = "https://api.together.xyz/v1";
export const DEFAULT_LLM_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731";
export const DEFAULT_VLM_MODEL = "meta-llama/Llama-4-Scout-17B-16E-Instruct";
export const DEFAULT_SYSTEM_PROMPT = [
  "You are the MetaDemic Lab assistant, embedded on the Metademic Research Lab website.",
  "You answer questions about the laboratory, its RACoN project",
  "(resource-aware coordination of distributed machines), and general research topics.",
  "Be concise, accurate and friendly. Use markdown for structure.",
  "Answer in the language the visitor writes in.",
].join(" ");

export const MAX_MESSAGES = 24;
export const HISTORY_TURNS = 12;
export const MAX_MESSAGE_CHARS = 8000;
export const MAX_IMAGES = 4;
export const MAX_IMAGE_CHARS = 3_000_000;
export const MAX_REQUEST_CHARS = 4_000_000;
export const RATE_MAX_REQUESTS = 20;
export const RATE_WINDOW_MS = 600_000;

export const EFFORT_TEMPERATURE: Record<string, number> = {
  Low: 0.2,
  Medium: 0.65,
  "Max Effort": 0.95,
};

export function env(name: string, fallback = ""): string {
  const value = process.env[name];
  return value && value.trim() !== "" ? value : fallback;
}

/* ------------------------------------------------------------------ */
/* CORS                                                                */
/* ------------------------------------------------------------------ */

export function allowedOrigins(): string[] {
  return env("ALLOWED_ORIGINS", "http://localhost:5173")
    .split(",")
    .map((o) => o.trim())
    .filter(Boolean);
}

export function applyCors(req: IncomingMessage, res: ServerResponse): boolean {
  const origin = req.headers.origin;
  if (!origin) return true; // same-origin / server-to-server calls
  const allowed = allowedOrigins();
  if (allowed.includes(origin)) {
    res.setHeader("Access-Control-Allow-Origin", origin);
    res.setHeader("Vary", "Origin");
    res.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Content-Type");
    res.setHeader("Access-Control-Max-Age", "600");
    return true;
  }
  return false;
}

/* ------------------------------------------------------------------ */
/* Rate limiting (best effort, per function instance)                  */
/* ------------------------------------------------------------------ */

const hits = new Map<string, number[]>();

export function clientIp(req: IncomingMessage): string {
  const forwarded = req.headers["x-forwarded-for"];
  if (typeof forwarded === "string" && forwarded.length > 0) {
    return forwarded.split(",")[0].trim();
  }
  return req.socket.remoteAddress ?? "unknown";
}

export function rateLimited(ip: string): boolean {
  const now = Date.now();
  const recent = hits.get(ip) ?? [];
  while (recent.length && now - recent[0] > RATE_WINDOW_MS) recent.shift();
  if (recent.length >= RATE_MAX_REQUESTS) return true;
  recent.push(now);
  hits.set(ip, recent);
  return false;
}

/* ------------------------------------------------------------------ */
/* Model resolution                                                    */
/* ------------------------------------------------------------------ */

export function modelAllowlist(): string[] {
  const raw = env("ALLOWED_MODELS");
  if (raw.trim() !== "") {
    return raw.split(",").map((m) => m.trim()).filter(Boolean);
  }
  const models = [env("LLM_MODEL", DEFAULT_LLM_MODEL)];
  const vlm = env("VLM_MODEL", DEFAULT_VLM_MODEL);
  if (vlm && !models.includes(vlm)) models.push(vlm);
  return models;
}

export function resolveModel(
  requested: string | undefined,
  hasImages: boolean,
): string {
  const vlm = env("VLM_MODEL", DEFAULT_VLM_MODEL);
  const chosen = hasImages ? vlm : (requested ?? env("LLM_MODEL", DEFAULT_LLM_MODEL));
  if (!modelAllowlist().includes(chosen)) {
    throw new Error(`Model is not available on this deployment: ${chosen}`);
  }
  return chosen;
}

export function prettyLabel(modelId: string): string {
  const tail = modelId.split("/").pop() ?? modelId;
  const stripped = tail.replace(/-\d{4}$/, "");
  const words = stripped.split(/[-_ ]+/).filter(Boolean);
  if (words.length === 0) return modelId;
  return words
    .map((w) => (w === w.toUpperCase() || /\d/.test(w) ? w.toUpperCase() : w[0].toUpperCase() + w.slice(1)))
    .join(" ");
}

/* ------------------------------------------------------------------ */
/* SSE helpers                                                         */
/* ------------------------------------------------------------------ */

export function sseEvent(payload: { delta?: string; error?: string }): string {
  return `data: ${JSON.stringify(payload)}\n\n`;
}

/* ------------------------------------------------------------------ */
/* Provider client                                                     */
/* ------------------------------------------------------------------ */

let cachedClient: OpenAI | null = null;
let cachedClientKey: string | null = null;

export function getClient(): OpenAI {
  const key = env("LLM_API_KEY");
  if (!key) throw new Error("The chat service is not configured (LLM_API_KEY missing).");
  const baseUrl = env("LLM_BASE_URL", DEFAULT_BASE_URL);
  const cacheKey = `${baseUrl}\u0000${key}`;
  if (cachedClient && cachedClientKey === cacheKey) return cachedClient;
  cachedClient = new OpenAI({ apiKey: key, baseURL: baseUrl });
  cachedClientKey = cacheKey;
  return cachedClient;
}

/* ------------------------------------------------------------------ */
/* Chat history persistence (Supabase, service role)                   */
/* ------------------------------------------------------------------ */

export async function persistConversation(
  sessionId: string | undefined,
  userText: string,
  assistantText: string,
  model: string,
  attachments: { name: string }[],
): Promise<void> {
  const base = env("SUPABASE_URL").replace(/\/$/, "");
  const key = env("SUPABASE_SERVICE_ROLE_KEY");
  if (!base || !key || !sessionId) return;
  const rows = [
    {
      session_id: sessionId,
      role: "user",
      content: userText.slice(0, MAX_MESSAGE_CHARS),
      model,
      attachments: attachments.slice(0, MAX_IMAGES),
    },
    {
      session_id: sessionId,
      role: "assistant",
      content: assistantText,
      model,
      attachments: [],
    },
  ];
  try {
    await fetch(`${base}/rest/v1/chat_messages`, {
      method: "POST",
      headers: {
        apikey: key,
        Authorization: `Bearer ${key}`,
        "Content-Type": "application/json",
        Prefer: "return=minimal",
      },
      body: JSON.stringify(rows),
    });
  } catch (error) {
    console.error("chat history persist failed:", error);
  }
}

import type { IncomingMessage, ServerResponse } from "node:http";
import {
  DEFAULT_BASE_URL,
  DEFAULT_LLM_MODEL,
  DEFAULT_VLM_MODEL,
  applyCors,
  env,
  modelAllowlist,
  prettyLabel,
} from "./_shared.js";

export default async function handler(
  req: IncomingMessage,
  res: ServerResponse,
): Promise<void> {
  if (!applyCors(req, res)) {
    res.statusCode = 403;
    res.end("Origin not allowed");
    return;
  }
  if (req.method === "OPTIONS") {
    res.statusCode = 204;
    res.end();
    return;
  }
  if (req.method !== "GET") {
    res.statusCode = 405;
    res.end("Use GET for /api/models.");
    return;
  }
  const vlm = env("VLM_MODEL", DEFAULT_VLM_MODEL);
  const models: { id: string; label: string }[] = [];
  for (const id of modelAllowlist()) {
    if (!models.some((m) => m.id === id)) {
      models.push({ id, label: prettyLabel(id) });
    }
  }
  res.statusCode = 200;
  res.setHeader("Content-Type", "application/json");
  res.end(
    JSON.stringify({
      models,
      vlm: { id: vlm, label: prettyLabel(vlm) },
      provider: env("LLM_BASE_URL", DEFAULT_BASE_URL),
      defaults: {
        chat: env("LLM_MODEL", DEFAULT_LLM_MODEL),
      },
    }),
  );
}

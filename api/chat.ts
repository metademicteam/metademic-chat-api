import type { IncomingMessage, ServerResponse } from "node:http";
import {
  DEFAULT_SYSTEM_PROMPT,
  DEFAULT_VLM_MODEL,
  EFFORT_TEMPERATURE,
  HISTORY_TURNS,
  MAX_IMAGES,
  MAX_IMAGE_CHARS,
  MAX_MESSAGES,
  MAX_MESSAGE_CHARS,
  MAX_REQUEST_CHARS,
  applyCors,
  clientIp,
  env,
  getProviderConfig,
  persistConversation,
  rateLimited,
  resolveModel,
  sendJson,
  sseEvent,
  streamCompletion,
  type ChatBody,
  type ChatMessageIn,
  type ContentPart,
  type ProviderMessage,
} from "./_shared.js";

type NodeRequest = IncomingMessage;

interface ChatStorageMessage extends ChatMessageIn {
  images?: string[];
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    req.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > MAX_REQUEST_CHARS) {
        reject(new Error("This message is too large to send."));
        req.destroy();
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => resolve(Buffer.concat(chunks).toString("utf8")));
    req.on("error", (error) => reject(error));
  });
}

function buildPayloadMessages(
  messages: ChatStorageMessage[],
): ProviderMessage[] {
  const trimmed = messages.slice(-HISTORY_TURNS);
  const payload: ProviderMessage[] = [
    {
      role: "system",
      content:
        env("CHAT_SYSTEM_PROMPT").trim() !== ""
          ? env("CHAT_SYSTEM_PROMPT")
          : DEFAULT_SYSTEM_PROMPT,
    },
  ];
  for (const message of trimmed) {
    const isLastUserWithImages =
      message.role === "user" &&
      message === trimmed[trimmed.length - 1] &&
      (message.images?.length ?? 0) > 0;
    if (isLastUserWithImages) {
      const parts: ContentPart[] = [
        { type: "text", text: message.content || "Describe the attached image." },
      ];
      for (const image of message.images ?? []) {
        parts.push({ type: "image_url", image_url: { url: image } });
      }
      payload.push({ role: "user", content: parts });
    } else {
      payload.push({ role: message.role, content: message.content });
    }
  }
  return payload;
}

function redirectImagesToLastUser(messages: ChatBody["messages"], images: string[]): ChatStorageMessage[] {
  const converted: ChatStorageMessage[] = messages.map((m) => ({ ...m }));
  if (images.length === 0) return converted;
  const lastUser = [...converted].reverse().find((m) => m.role === "user");
  if (lastUser) lastUser.images = images;
  return converted;
}

export default async function handler(
  req: NodeRequest,
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
  if (req.method !== "POST") {
    sendJson(res, 405, { error: "Use POST for /api/chat." });
    return;
  }

  if (!env("LLM_API_KEY")) {
    sendJson(res, 503, {
      error: "The chat service is not configured (LLM_API_KEY missing).",
    });
    return;
  }
  if (rateLimited(clientIp(req))) {
    sendJson(res, 429, {
      error: "Too many requests — please wait a moment and try again.",
    });
    return;
  }

  let body: ChatBody;
  try {
    body = JSON.parse(await readBody(req)) as ChatBody;
  } catch (error) {
    sendJson(res, 400, {
      error:
        error instanceof Error && error.message.length < 200
          ? error.message
          : "Invalid request body.",
    });
    return;
  }
  if (!Array.isArray(body.messages) || body.messages.length === 0) {
    sendJson(res, 400, { error: "No messages were provided." });
    return;
  }
  if (body.messages.length > MAX_MESSAGES) {
    sendJson(res, 413, { error: "The conversation is too long to send." });
    return;
  }
  for (const message of body.messages) {
    if (
      typeof message.content !== "string" ||
      message.content.length > MAX_MESSAGE_CHARS
    ) {
      sendJson(res, 413, {
        error: "A message in the history is too long to send.",
      });
      return;
    }
  }
  const images = (body.images ?? []).filter(
    (image) => typeof image === "string" && image.startsWith("data:image/"),
  );
  if ((body.images ?? []).length > MAX_IMAGES) {
    sendJson(res, 413, {
      error: `At most ${MAX_IMAGES} images can be attached per message.`,
    });
    return;
  }
  for (const image of images) {
    if (image.length > MAX_IMAGE_CHARS) {
      sendJson(res, 413, {
        error: "An attached image is too large (try a smaller one).",
      });
      return;
    }
  }

  let model: string;
  try {
    model = resolveModel(
      typeof body.model === "string" ? body.model : undefined,
      images.length > 0,
    );
  } catch (error) {
    sendJson(res, 400, {
      error: error instanceof Error ? error.message : "Model not available.",
    });
    return;
  }

  const payloadMessages = buildPayloadMessages(
    redirectImagesToLastUser(body.messages, images),
  );
  const temperature = EFFORT_TEMPERATURE[body.effort ?? "Medium"] ?? 0.65;
  const lastUser = [...body.messages].reverse().find((m) => m.role === "user");
  const userText = lastUser?.content ?? "";
  const attachments = (body.attachments ?? [])
    .filter((a): a is { name: string } => Boolean(a && typeof a.name === "string"))
    .map((a) => ({ name: a.name.slice(0, 120) }));

  res.statusCode = 200;
  res.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");

  const controller = new AbortController();
  req.on("close", () => controller.abort());

  let collected = "";
  const write = (frame: string) => {
    if (!res.writableEnded) res.write(frame);
  };

  const provider = getProviderConfig();
  try {
    for await (const delta of streamCompletion({
      baseUrl: provider.baseUrl,
      apiKey: provider.apiKey,
      model,
      messages: payloadMessages,
      temperature,
      signal: controller.signal,
    })) {
      collected += delta;
      write(sseEvent({ delta }));
    }
  } catch (error) {
    if (!controller.signal.aborted) {
      console.error(
        "chat generation failed:",
        error instanceof Error ? error.message : error,
      );
      write(sseEvent({ error: "The assistant could not answer. Please try again." }));
    }
  }

  write("data: [DONE]\n\n");
  await persistConversation(
    typeof body.sessionId === "string" ? body.sessionId : undefined,
    userText,
    collected,
    model,
    attachments,
  );
  res.end();
}

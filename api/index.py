"""Metademic Chat API — a provider-agnostic, streaming LLM/VLM proxy.

Serves the /chat page on the Metademic Research Lab site. Deployed to Vercel as
a separate project (metademic-chat-api) on the Vercel Python runtime.

The underlying LLM/VLM is entirely environment-driven — swapping providers (or
pointing at your own hosted LLM/VLM) only requires changing environment
variables in the Vercel project settings, never code:

    LLM_API_KEY     required — API key for the chosen provider
    LLM_BASE_URL    optional — OpenAI-compatible base URL
                               (default: https://api.together.xyz/v1)
    LLM_MODEL       optional — default chat model id
                               (default: deepseek-ai/DeepSeek-V4-Flash-0731)
    VLM_MODEL       optional — model used when images are attached
                               (default: meta-llama/Llama-4-Scout-17B-16E-Instruct)
    ALLOWED_MODELS  optional — CSV of model ids clients may select from
    CHAT_SYSTEM_PROMPT  optional — overrides the built-in assistant prompt
    ALLOWED_ORIGINS optional — CSV of CORS origins (default: local dev only)
    SUPABASE_URL    optional — Supabase project URL (for chat history)
    SUPABASE_SERVICE_ROLE_KEY  optional — service key used to persist history

Endpoints:
    GET /              → liveness + provider status
    GET /api/models    → selectable chat models + the vision model
    POST /api/chat     → SSE stream of {"delta": "..."} frames, ends with [DONE]

Streaming format (Server-Sent Events):
    data: {"delta": "partial text"}   (repeated)
    data: {"error": "message"}        (only if generation fails mid-stream)
    data: [DONE]
"""

import json
import os
import re
import sys
import time
from collections import defaultdict, deque
from typing import Any, Literal

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from openai import OpenAI
from pydantic import BaseModel, Field

DEFAULT_BASE_URL = "https://api.together.xyz/v1"
DEFAULT_LLM_MODEL = "deepseek-ai/DeepSeek-V4-Flash-0731"
DEFAULT_VLM_MODEL = "meta-llama/Llama-4-Scout-17B-16E-Instruct"
DEFAULT_SYSTEM_PROMPT = (
    "You are the MetaDemic Lab assistant, embedded on the Metademic Research "
    "Lab website. You answer questions about the laboratory, its RACoN project "
    "(resource-aware coordination of distributed machines), and general "
    "research topics. Be concise, accurate and friendly. Use markdown for "
    "structure. Answer in the language the visitor writes in."
)

MAX_MESSAGES = 24
HISTORY_TURNS = 12
MAX_MESSAGE_CHARS = 8000
MAX_IMAGES = 4
MAX_IMAGE_CHARS = 3_000_000
MAX_REQUEST_CHARS = 4_000_000

RATE_MAX_REQUESTS = 20
RATE_WINDOW_SECONDS = 600.0

EFFORT_TEMPERATURE = {"Low": 0.2, "Medium": 0.65, "Max Effort": 0.95}

app = FastAPI(title="Metademic Chat API")


def env(name: str, default: str = "") -> str:
    value = os.environ.get(name, "")
    return value if value else default


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in env(
            "ALLOWED_ORIGINS", "http://localhost:5173"
        ).split(",")
        if origin.strip()
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_client: OpenAI | None = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = env("LLM_API_KEY")
        if not api_key:
            raise RuntimeError("LLM_API_KEY is not configured")
        _client = OpenAI(
            api_key=api_key, base_url=env("LLM_BASE_URL", DEFAULT_BASE_URL)
        )
    return _client


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


_hits: dict[str, deque[float]] = defaultdict(deque)


def rate_limited(ip: str) -> bool:
    now = time.time()
    recent = _hits[ip]
    while recent and now - recent[0] > RATE_WINDOW_SECONDS:
        recent.popleft()
    if len(recent) >= RATE_MAX_REQUESTS:
        return True
    recent.append(now)
    return False


class ChatMessageIn(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    sessionId: str | None = Field(default=None, max_length=80)
    messages: list[ChatMessageIn]
    model: str | None = Field(default=None, max_length=200)
    effort: Literal["Low", "Medium", "Max Effort"] | None = None
    images: list[str] = Field(default_factory=list)
    attachments: list[dict[str, Any]] = Field(default_factory=list)


def model_allowlist() -> list[str]:
    raw = env("ALLOWED_MODELS")
    if raw:
        return [model_id.strip() for model_id in raw.split(",") if model_id.strip()]
    models: list[str] = [env("LLM_MODEL", DEFAULT_LLM_MODEL)]
    vlm = env("VLM_MODEL", DEFAULT_VLM_MODEL)
    if vlm and vlm not in models:
        models.append(vlm)
    return models


def resolve_model(requested: str | None, has_images: bool) -> str:
    base = env("LLM_MODEL", DEFAULT_LLM_MODEL)
    vlm = env("VLM_MODEL", DEFAULT_VLM_MODEL)
    chosen = vlm if has_images else (requested or base)
    if chosen not in model_allowlist():
        raise ValueError(f"Model is not available on this deployment: {chosen}")
    return chosen


def pretty_label(model_id: str) -> str:
    tail = model_id.split("/")[-1]
    tail = re.sub(r"-\d{4}$", "", tail)
    words: list[str] = []
    for word in re.split(r"[-_ ]", tail):
        if not word:
            continue
        if word.isupper() or any(ch.isdigit() for ch in word):
            words.append(word.upper())
        else:
            words.append(word[0].upper() + word[1:])
    return " ".join(words) if words else model_id


def sse_event(payload: dict[str, str]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def build_payload_messages(
    messages: list[ChatMessageIn], images: list[str]
) -> list[dict[str, Any]]:
    trimmed = messages[-HISTORY_TURNS:]
    payload: list[dict[str, Any]] = [
        {"role": "system", "content": env("CHAT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)}
    ]
    for message in trimmed:
        if (
            message.role == "user"
            and message is trimmed[-1]
            and images
        ):
            parts: list[dict[str, Any]] = [
                {"type": "text", "text": message.content or "Describe the attached image."}
            ]
            for image_url in images:
                parts.append(
                    {"type": "image_url", "image_url": {"url": image_url}}
                )
            payload.append({"role": "user", "content": parts})
        else:
            payload.append({"role": message.role, "content": message.content})
    return payload


def persist_conversation(
    session_id: str | None,
    user_text: str,
    assistant_text: str,
    model: str,
    attachments: list[dict[str, Any]],
) -> None:
    base = env("SUPABASE_URL").rstrip("/")
    key = env("SUPABASE_SERVICE_ROLE_KEY")
    if not (base and key and session_id):
        return
    rows = [
        {
            "session_id": session_id,
            "role": "user",
            "content": user_text[: MAX_MESSAGE_CHARS],
            "model": model,
            "attachments": attachments[:MAX_IMAGES],
        },
        {
            "session_id": session_id,
            "role": "assistant",
            "content": assistant_text,
            "model": model,
            "attachments": [],
        },
    ]
    try:
        with httpx.Client(timeout=10.0) as http:
            http.post(
                f"{base}/rest/v1/chat_messages",
                json=rows,
                headers={
                    "apikey": key,
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "Prefer": "return=minimal",
                },
            )
    except Exception as exc:
        print(f"chat history persist failed: {exc}", file=sys.stderr)


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "status": "ok",
        "provider": env("LLM_BASE_URL", DEFAULT_BASE_URL),
        "model": env("LLM_MODEL", DEFAULT_LLM_MODEL),
    }


@app.get("/api/models")
def list_models() -> dict[str, Any]:
    vlm = env("VLM_MODEL", DEFAULT_VLM_MODEL)
    models: list[dict[str, str]] = []
    for model_id in model_allowlist():
        if model_id not in [m["id"] for m in models]:
            models.append({"id": model_id, "label": pretty_label(model_id)})
    return {
        "models": models,
        "vlm": {"id": vlm, "label": pretty_label(vlm)},
        "provider": env("LLM_BASE_URL", DEFAULT_BASE_URL),
    }


@app.post("/api/chat")
def api_chat(body: ChatRequest, request: Request) -> Any:
    if not env("LLM_API_KEY"):
        return JSONResponse(
            status_code=503,
            content={"error": "The chat service is not configured (LLM_API_KEY missing)."},
        )
    if rate_limited(client_ip(request)):
        return JSONResponse(
            status_code=429,
            content={"error": "Too many requests — please wait a moment and try again."},
        )
    if not body.messages:
        return JSONResponse(status_code=400, content={"error": "No messages were provided."})
    total_chars = sum(len(m.content) for m in body.messages)
    total_chars += sum(len(i) for i in body.images)
    if total_chars > MAX_REQUEST_CHARS:
        return JSONResponse(
            status_code=413,
            content={"error": "This message is too large to send."},
        )
    for message in body.messages:
        if len(message.content) > MAX_MESSAGE_CHARS:
            return JSONResponse(
                status_code=413,
                content={"error": "A message in the history is too long to send."},
            )
    if len(body.images) > MAX_IMAGES:
        return JSONResponse(
            status_code=413,
            content={"error": f"At most {MAX_IMAGES} images can be attached per message."},
        )
    for image in body.images:
        if not image.startswith("data:image/"):
            return JSONResponse(
                status_code=400, content={"error": "Attachments must be images."}
            )
        if len(image) > MAX_IMAGE_CHARS:
            return JSONResponse(
                status_code=413,
                content={"error": "An attached image is too large (try a smaller one)."},
            )

    try:
        model = resolve_model(body.model, bool(body.images))
        payload = build_payload_messages(body.messages, body.images)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})

    last_user = next(
        (m for m in reversed(body.messages) if m.role == "user"), None
    )
    user_text = last_user.content if last_user else ""
    temperature = EFFORT_TEMPERATURE.get(body.effort or "Medium", 0.65)
    session_id = body.sessionId
    safe_attachments = [
        {"name": str(a.get("name", ""))[:120]}
        for a in body.attachments
        if isinstance(a, dict)
    ]

    def stream():
        collected: list[str] = []
        try:
            completion = get_client().chat.completions.create(
                model=model,
                messages=payload,
                temperature=temperature,
                stream=True,
            )
            for chunk in completion:
                delta = None
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta.content
                if not delta:
                    continue
                collected.append(delta)
                yield sse_event({"delta": str(delta)})
        except Exception as exc:
            error_name = getattr(exc, "__class__", type(exc)).__name__
            print(f"chat generation failed ({error_name}): {exc}", file=sys.stderr)
            yield sse_event({"error": "The assistant could not answer. Please try again."})
        yield "data: [DONE]\n\n"
        persist_conversation(
            session_id, user_text, "".join(collected), model, safe_attachments
        )

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

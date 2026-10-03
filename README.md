# Metademic Chat API

Provider-agnostic streaming LLM/VLM proxy that powers the `/chat` page on the
Metademic Research Lab site. Runs on the Vercel Python runtime as the
`metademic-chat-api` project.

## Endpoints

- `GET /` — liveness and current provider/model
- `GET /api/models` — selectable chat models plus the vision model
- `POST /api/chat` — streaming chat (SSE): `data: {"delta": "…"}` frames,
  ending with `data: [DONE]`; a `data: {"error": "…"}` frame is emitted if
  generation fails

## Configuration (environment only — no code changes to swap providers)

| Variable                   | Default                                          | Purpose                                   |
| -------------------------- | ------------------------------------------------ | ----------------------------------------- |
| `LLM_API_KEY`              | —                                                | Provider API key (required)               |
| `LLM_BASE_URL`             | `https://api.together.xyz/v1`                    | Any OpenAI-compatible endpoint            |
| `LLM_MODEL`                | `deepseek-ai/DeepSeek-V4-Flash-0731`             | Default chat model                        |
| `VLM_MODEL`                | `meta-llama/Llama-4-Scout-17B-16E-Instruct`      | Model used when images are attached       |
| `ALLOWED_MODELS`           | LLM + VLM models                                 | CSV allowlist exposed via `/api/models`   |
| `CHAT_SYSTEM_PROMPT`       | built-in lab assistant prompt                    | Override the assistant persona            |
| `ALLOWED_ORIGINS`          | `http://localhost:5173`                          | CSV of origins allowed to call this API   |
| `SUPABASE_URL`             | —                                                | Supabase project URL (chat history)       |
| `SUPABASE_SERVICE_ROLE_KEY`| —                                                | Service key used to persist chat history  |

Because the request layer speaks the OpenAI-compatible chat-completions
protocol, moving to your own LLM/VLM later only means updating
`LLM_API_KEY` (and optionally `LLM_BASE_URL` / model ids) in the Vercel
project settings.

## Local development

```bash
cd chat-api
cp .env.example .env   # fill in the values
pip install -r requirements.txt
uvicorn api.index:app --reload
```

## Deploy

```bash
vercel deploy --prod   # from this directory (project: metademic-chat-api)
```

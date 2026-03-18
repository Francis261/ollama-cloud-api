# Ollama Cloud OpenAI-Compatible Adapter (Dyad Provider)

This repository contains a production-ready Go server that exposes an OpenAI-compatible API and forwards requests to **Ollama Cloud**.

It is designed to be used as a **Dyad custom model provider**.

## Features

- Listens on `PORT` or defaults to `11434`
- OpenAI-compatible routes:
  - `POST /v1/chat/completions`
  - `POST /v1/completions`
  - `GET /v1/models`
- Maps incoming model aliases to Ollama Cloud model tags (for example `gpt-oss` → `gpt-oss:120b`)
- Forwards generation requests to:
  - `https://ollama.com/api/generate`
- Uses Bearer auth from:
  - `OLLAMA_API_KEY`
- Returns OpenAI-style JSON responses that Dyad can consume
- Supports non-streaming and optional streaming (`stream: true` via SSE for chat completions)

## Requirements

- Go 1.22+
- `OLLAMA_API_KEY` environment variable set

## Run locally

```bash
export OLLAMA_API_KEY="your_ollama_cloud_api_key"
export PORT=11434

go run .
```

Server starts on `http://localhost:11434` by default.

## Build

```bash
go build .
```

## API examples

### 1) Chat Completions

```bash
curl -s http://localhost:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"gpt-oss",
    "messages":[
      {"role":"system","content":"You are concise."},
      {"role":"user","content":"Explain what an API adapter is in one sentence."}
    ],
    "stream":false
  }'
```

### 2) Completions (alias endpoint)

```bash
curl -s http://localhost:11434/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model":"gpt-oss",
    "prompt":"Write one sentence about Golang.",
    "stream":false
  }'
```

### 3) List Models

```bash
curl -s http://localhost:11434/v1/models
```

## Environment variables

- `OLLAMA_API_KEY` (required): API key for Ollama Cloud
- `PORT` (optional): HTTP port (default `11434`)

## Deployment note (Vercel / Go build)

This repo includes a `go.mod` file so builders that run `go build .` at repo root can detect the module and compile correctly.

## Render + Dyad setup checklist (important)

If Dyad shows an error like:

`getaddrinfo ENOTFOUND <host>`

that is a **DNS/hostname configuration issue**, not a model response issue.

Use this checklist:

1. In Render, copy the exact generated service URL, for example:
   - `https://ollama-cloud-api.onrender.com`
2. In Dyad custom provider base URL, paste the full URL including `https://`.
3. Do **not** remove dots or edit hostname segments.
   - Correct: `ollama-cloud-api.onrender.com`
   - Incorrect: `ollama-cloud-api-onrender.com`
4. Keep OpenAI-compatible pathing:
   - Dyad should call `/v1/models` and `/v1/chat/completions`.
5. Ensure Render env var is set:
   - `OLLAMA_API_KEY=...`
6. Test your deployed URL directly:

```bash
curl -i https://YOUR-RENDER-URL.onrender.com/healthz
curl -i https://YOUR-RENDER-URL.onrender.com/v1/models
```

### Extra compatibility notes

- This server accepts CORS preflight (`OPTIONS`) and sets permissive CORS headers.
- It also supports aliases without `/v1`:
  - `/models`
  - `/chat/completions`
  - `/completions`

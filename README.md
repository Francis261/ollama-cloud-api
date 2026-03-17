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

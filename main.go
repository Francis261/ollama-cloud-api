package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

const (
	defaultPort     = "11434"
	ollamaGenerate  = "https://ollama.com/api/generate"
	defaultReqLimit = int64(2 << 20) // 2MB
)

// message represents an OpenAI chat message.
type message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// chatCompletionRequest supports the subset Dyad/OpenAI typically sends.
type chatCompletionRequest struct {
	Model       string    `json:"model"`
	Messages    []message `json:"messages"`
	Prompt      string    `json:"prompt"`
	Stream      bool      `json:"stream"`
	MaxTokens   *int      `json:"max_tokens,omitempty"`
	Temperature *float64  `json:"temperature,omitempty"`
	TopP        *float64  `json:"top_p,omitempty"`
	Stop        any       `json:"stop,omitempty"`
}

// ollamaGenerateRequest mirrors Ollama Cloud generate payload.
type ollamaGenerateRequest struct {
	Model   string `json:"model"`
	Prompt  string `json:"prompt"`
	Stream  bool   `json:"stream"`
	Options struct {
		Temperature *float64 `json:"temperature,omitempty"`
		TopP        *float64 `json:"top_p,omitempty"`
		NumPredict  *int     `json:"num_predict,omitempty"`
		Stop        any      `json:"stop,omitempty"`
	} `json:"options,omitempty"`
}

// ollamaGenerateResponse is a minimal response model for non-streaming calls.
type ollamaGenerateResponse struct {
	Model    string `json:"model"`
	Response string `json:"response"`
	Done     bool   `json:"done"`
	Error    string `json:"error,omitempty"`
}

// openAIErrorResponse is the OpenAI-style error envelope.
type openAIErrorResponse struct {
	Error struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code,omitempty"`
	} `json:"error"`
}

// chatCompletionResponse is OpenAI-compatible /v1/chat/completions output.
type chatCompletionResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int `json:"index"`
		FinishReason any `json:"finish_reason"`
		Message      struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

// completionResponse is OpenAI-compatible /v1/completions output.
type completionResponse struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	Model   string `json:"model"`
	Choices []struct {
		Index        int    `json:"index"`
		Text         string `json:"text"`
		FinishReason any    `json:"finish_reason"`
	} `json:"choices"`
}

// modelsResponse is OpenAI-compatible /v1/models output.
type modelsResponse struct {
	Object string      `json:"object"`
	Data   []modelMeta `json:"data"`
}

type modelMeta struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	OwnedBy string `json:"owned_by"`
}

type server struct {
	apiKey      string
	httpClient  *http.Client
	modelMap    map[string]string
	knownModels []string
}

func main() {
	log.SetFlags(log.LstdFlags | log.LUTC | log.Lshortfile)

	apiKey := strings.TrimSpace(os.Getenv("OLLAMA_API_KEY"))
	if apiKey == "" {
		log.Fatal("missing required env var OLLAMA_API_KEY")
	}

	srv := &server{
		apiKey: apiKey,
		httpClient: &http.Client{
			Timeout: 120 * time.Second,
		},
		modelMap: map[string]string{
			"gpt-oss":      "gpt-oss:120b",
			"gpt-oss:120b": "gpt-oss:120b",
			"llama3.3":     "llama3.3:70b",
			"llama3.1":     "llama3.1:8b",
		},
		knownModels: []string{"gpt-oss", "gpt-oss:120b", "llama3.3", "llama3.1"},
	}

	mux := http.NewServeMux()
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			writeError(w, http.StatusNotFound, "invalid_request_error", "route not found")
			return
		}
		if r.Method != http.MethodGet {
			writeError(w, http.StatusMethodNotAllowed, "invalid_request_error", "method not allowed")
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"service":"ollama-cloud-openai-adapter","status":"ok","routes":["/v1/models","/v1/chat/completions","/v1/completions"]}`))
	})
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"status":"ok"}`))
	})
	mux.HandleFunc("/v1/models", srv.handleModels)
	mux.HandleFunc("/v1/chat/completions", srv.handleChatCompletions)
	mux.HandleFunc("/v1/completions", srv.handleCompletions)
	// Compatibility aliases for clients that omit /v1.
	mux.HandleFunc("/models", srv.handleModels)
	mux.HandleFunc("/chat/completions", srv.handleChatCompletions)
	mux.HandleFunc("/completions", srv.handleCompletions)

	port := strings.TrimSpace(os.Getenv("PORT"))
	if port == "" {
		port = defaultPort
	}
	if _, err := strconv.Atoi(port); err != nil {
		log.Fatalf("invalid PORT %q: %v", port, err)
	}

	addr := ":" + port
	log.Printf("starting OpenAI-compatible adapter on %s", addr)
	if err := http.ListenAndServe(addr, loggingMiddleware(mux)); err != nil {
		log.Fatalf("server failed: %v", err)
	}
}

func (s *server) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeError(w, http.StatusMethodNotAllowed, "invalid_request_error", "method not allowed")
		return
	}

	resp := modelsResponse{Object: "list"}
	for _, id := range s.knownModels {
		resp.Data = append(resp.Data, modelMeta{ID: id, Object: "model", OwnedBy: "ollama-cloud"})
	}
	writeJSON(w, http.StatusOK, resp)
}

func (s *server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "invalid_request_error", "method not allowed")
		return
	}

	req, err := decodeRequest(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "model is required")
		return
	}
	if len(req.Messages) == 0 && strings.TrimSpace(req.Prompt) == "" {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "messages or prompt is required")
		return
	}

	prompt := buildPrompt(req.Messages, req.Prompt)
	ollamaModel := s.mapModel(req.Model)

	if req.Stream {
		s.streamChatResponse(w, r.Context(), req.Model, ollamaModel, prompt, req)
		return
	}

	text, err := s.generate(r.Context(), ollamaModel, prompt, false, req)
	if err != nil {
		httpCode, msg := classifyErr(err)
		writeError(w, httpCode, "server_error", msg)
		return
	}

	resp := chatCompletionResponse{
		ID:      "chatcmpl-" + newID(),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
	}
	choice := struct {
		Index        int `json:"index"`
		FinishReason any `json:"finish_reason"`
		Message      struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
	}{Index: 0, FinishReason: "stop"}
	choice.Message.Role = "assistant"
	choice.Message.Content = text
	resp.Choices = []struct {
		Index        int `json:"index"`
		FinishReason any `json:"finish_reason"`
		Message      struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
	}{choice}

	writeJSON(w, http.StatusOK, resp)
}

func (s *server) handleCompletions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "invalid_request_error", "method not allowed")
		return
	}

	req, err := decodeRequest(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}
	if req.Model == "" {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "model is required")
		return
	}
	if strings.TrimSpace(req.Prompt) == "" && len(req.Messages) == 0 {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "prompt or messages is required")
		return
	}

	prompt := buildPrompt(req.Messages, req.Prompt)
	ollamaModel := s.mapModel(req.Model)
	text, err := s.generate(r.Context(), ollamaModel, prompt, false, req)
	if err != nil {
		httpCode, msg := classifyErr(err)
		writeError(w, httpCode, "server_error", msg)
		return
	}

	resp := completionResponse{
		ID:      "cmpl-" + newID(),
		Object:  "text_completion",
		Created: time.Now().Unix(),
		Model:   req.Model,
	}
	resp.Choices = append(resp.Choices, struct {
		Index        int    `json:"index"`
		Text         string `json:"text"`
		FinishReason any    `json:"finish_reason"`
	}{Index: 0, Text: text, FinishReason: "stop"})
	writeJSON(w, http.StatusOK, resp)
}

func decodeRequest(r *http.Request) (chatCompletionRequest, error) {
	var req chatCompletionRequest
	dec := json.NewDecoder(io.LimitReader(r.Body, defaultReqLimit))
	if err := dec.Decode(&req); err != nil {
		return req, fmt.Errorf("invalid JSON body: %w", err)
	}
	return req, nil
}

func buildPrompt(messages []message, fallbackPrompt string) string {
	if len(messages) == 0 {
		return strings.TrimSpace(fallbackPrompt)
	}
	var b strings.Builder
	for _, m := range messages {
		role := strings.ToUpper(strings.TrimSpace(m.Role))
		if role == "" {
			role = "USER"
		}
		content := strings.TrimSpace(m.Content)
		if content == "" {
			continue
		}
		b.WriteString(role)
		b.WriteString(": ")
		b.WriteString(content)
		b.WriteString("\n")
	}
	prompt := strings.TrimSpace(b.String())
	if prompt == "" {
		return strings.TrimSpace(fallbackPrompt)
	}
	return prompt
}

func (s *server) mapModel(requested string) string {
	if mapped, ok := s.modelMap[requested]; ok {
		return mapped
	}
	return requested
}

func (s *server) generate(ctx context.Context, model, prompt string, stream bool, req chatCompletionRequest) (string, error) {
	payload := ollamaGenerateRequest{
		Model:  model,
		Prompt: prompt,
		Stream: stream,
	}
	payload.Options.Temperature = req.Temperature
	payload.Options.TopP = req.TopP
	payload.Options.NumPredict = req.MaxTokens
	payload.Options.Stop = req.Stop

	body, err := json.Marshal(payload)
	if err != nil {
		return "", fmt.Errorf("failed to marshal ollama request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, ollamaGenerate, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("failed to create upstream request: %w", err)
	}
	httpReq.Header.Set("Authorization", "Bearer "+s.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")

	res, err := s.httpClient.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("upstream request failed: %w", err)
	}
	defer res.Body.Close()

	if res.StatusCode < 200 || res.StatusCode > 299 {
		raw, _ := io.ReadAll(io.LimitReader(res.Body, 16<<10))
		return "", fmt.Errorf("upstream error status=%d body=%s", res.StatusCode, strings.TrimSpace(string(raw)))
	}

	if stream {
		return "", nil
	}

	var upstream ollamaGenerateResponse
	if err := json.NewDecoder(io.LimitReader(res.Body, 2<<20)).Decode(&upstream); err != nil {
		return "", fmt.Errorf("failed to decode upstream response: %w", err)
	}
	if upstream.Error != "" {
		return "", errors.New(upstream.Error)
	}
	return upstream.Response, nil
}

func (s *server) streamChatResponse(w http.ResponseWriter, ctx context.Context, requestedModel, model, prompt string, req chatCompletionRequest) {
	payload := ollamaGenerateRequest{Model: model, Prompt: prompt, Stream: true}
	payload.Options.Temperature = req.Temperature
	payload.Options.TopP = req.TopP
	payload.Options.NumPredict = req.MaxTokens
	payload.Options.Stop = req.Stop

	body, err := json.Marshal(payload)
	if err != nil {
		writeError(w, http.StatusInternalServerError, "server_error", "failed to marshal upstream request")
		return
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, ollamaGenerate, bytes.NewReader(body))
	if err != nil {
		writeError(w, http.StatusInternalServerError, "server_error", "failed to create upstream request")
		return
	}
	httpReq.Header.Set("Authorization", "Bearer "+s.apiKey)
	httpReq.Header.Set("Content-Type", "application/json")

	res, err := s.httpClient.Do(httpReq)
	if err != nil {
		writeError(w, http.StatusBadGateway, "server_error", fmt.Sprintf("upstream request failed: %v", err))
		return
	}
	defer res.Body.Close()

	if res.StatusCode < 200 || res.StatusCode > 299 {
		raw, _ := io.ReadAll(io.LimitReader(res.Body, 16<<10))
		writeError(w, http.StatusBadGateway, "server_error", fmt.Sprintf("upstream error status=%d body=%s", res.StatusCode, strings.TrimSpace(string(raw))))
		return
	}

	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "server_error", "streaming unsupported by server")
		return
	}

	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	scanner := bufio.NewScanner(res.Body)
	buf := make([]byte, 0, 1024)
	scanner.Buffer(buf, 2<<20)

	id := "chatcmpl-" + newID()
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		var chunk ollamaGenerateResponse
		if err := json.Unmarshal([]byte(line), &chunk); err != nil {
			log.Printf("stream parse warning: %v line=%q", err, line)
			continue
		}
		if chunk.Error != "" {
			log.Printf("upstream stream error: %s", chunk.Error)
			break
		}
		openAIChunk := map[string]any{
			"id":      id,
			"object":  "chat.completion.chunk",
			"created": time.Now().Unix(),
			"model":   requestedModel,
			"choices": []map[string]any{{
				"index": 0,
				"delta": map[string]any{"content": chunk.Response},
			}},
		}
		if chunk.Done {
			openAIChunk["choices"] = []map[string]any{{
				"index":         0,
				"delta":         map[string]any{},
				"finish_reason": "stop",
			}}
		}
		data, _ := json.Marshal(openAIChunk)
		_, _ = fmt.Fprintf(w, "data: %s\n\n", data)
		flusher.Flush()
		if chunk.Done {
			break
		}
	}

	if err := scanner.Err(); err != nil {
		log.Printf("stream read error: %v", err)
	}
	_, _ = io.WriteString(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func loggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodOptions {
			setCORSHeaders(w)
			w.WriteHeader(http.StatusNoContent)
			return
		}

		setCORSHeaders(w)
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s from=%s took=%s", r.Method, r.URL.Path, r.RemoteAddr, time.Since(start))
	})
}

func setCORSHeaders(w http.ResponseWriter) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Headers", "Authorization, Content-Type")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
}

func classifyErr(err error) (int, string) {
	msg := err.Error()
	switch {
	case strings.Contains(msg, "invalid JSON"):
		return http.StatusBadRequest, msg
	case strings.Contains(msg, "upstream error status=401"), strings.Contains(msg, "upstream error status=403"):
		return http.StatusUnauthorized, "upstream auth failed: check OLLAMA_API_KEY"
	case strings.Contains(msg, "upstream error status=404"):
		return http.StatusBadRequest, "model not found upstream"
	case strings.Contains(msg, "upstream error status=429"):
		return http.StatusTooManyRequests, "upstream rate limit reached"
	default:
		return http.StatusBadGateway, msg
	}
}

func writeError(w http.ResponseWriter, status int, kind, msg string) {
	resp := openAIErrorResponse{}
	resp.Error.Type = kind
	resp.Error.Message = msg
	resp.Error.Code = strconv.Itoa(status)
	writeJSON(w, status, resp)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("failed to encode JSON response: %v", err)
	}
}

func newID() string {
	return strconv.FormatInt(time.Now().UnixNano(), 36)
}

/*
Example usage:

1) Start server:
   OLLAMA_API_KEY=your_key_here PORT=11434 go run main.go

2) Chat completions:
   curl -s http://localhost:11434/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{
       "model":"gpt-oss",
       "messages":[
         {"role":"system","content":"You are concise."},
         {"role":"user","content":"Explain what an API adapter is in one sentence."}
       ],
       "stream":false
     }' | jq .

3) Legacy completions alias:
   curl -s http://localhost:11434/v1/completions \
     -H 'Content-Type: application/json' \
     -d '{
       "model":"gpt-oss",
       "prompt":"Write one sentence about Golang.",
       "stream":false
     }' | jq .

4) List models:
   curl -s http://localhost:11434/v1/models | jq .
*/

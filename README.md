# Gemini Streaming Server

A minimal Express server providing:

- `/gemini` — non-streaming text generation proxy for Gemini
- `/gemini/stream` — Server-Sent Events (SSE) streaming proxy for Gemini
- `/gemini/limit` — rate limit status endpoint (simple in-memory limiter)
- `/tts` and `/tts/stream` — optional OpenAI TTS endpoints to synthesize responses

## Run locally

```bash
npm install
npm run dev   # or: npm start
```

The server listens on `PORT` (defaults to `3000` locally; Render will set this automatically).

## Environment

Copy `.env.example` to `.env` and set values:

```
GEMINI_API_KEY=your_google_gemini_api_key
# or
# GOOGLE_GEMINI_API_KEY=your_google_gemini_api_key

# Optional for TTS
OPENAI_TTS_API_KEY=your_openai_key
```

## Deploy to Render

1. Create a new Web Service in Render.
2. Connect the GitHub repo containing this `server/` code as the root.
3. Settings:
   - Build Command: `npm install`
   - Start Command: `npm start`
   - Environment: add `GEMINI_API_KEY` (or `GOOGLE_GEMINI_API_KEY`) and optionally `OPENAI_TTS_API_KEY`.

> Note: Render automatically provides the `PORT` env var, and this app reads it in `server/server.js`.

## Endpoints

- `POST /gemini` — `{ prompt, model? }` → `{ text }`
- `POST /gemini/stream` — `{ prompt, model? }` → `text/event-stream` chunks; ends with `[DONE]`
- `GET  /gemini/limit` — current limiter state
- `POST /tts` — synthesize audio (OpenAI)
- `POST /tts/stream` and `GET /tts/stream` — stream audio (OpenAI)


import express from "express";
import fetch from "node-fetch";
import dotenv from "dotenv";
import cors from 'cors';
import https from 'https';
import { generateContent, streamGenerateContent } from './gemini.js'

// ! Load environment variables from .env
dotenv.config();

const app = express();
app.use(express.json({ limit: '1mb' }));

// Use permissive defaults (handles preflight automatically)
app.use(cors());

// Reuse TLS connections to OpenAI to reduce latency
const openAiAgent = new https.Agent({ keepAlive: true })

// ! Simple in-memory rate limiter: 5 requests per minute across all Gemini routes
const WINDOW_MS = 60_000, MAX_REQ = 10; // ! 60s window, max 10 hits
let requestTimes = []; // rolling timestamps (ms)
let limitReached = false; // ! true when rate limit is currently exceeded

function geminiRateLimiter(req, res, next) {
  const now = Date.now();
  requestTimes = requestTimes.filter(t => now - t < WINDOW_MS); // drop old entries
  limitReached = requestTimes.length >= MAX_REQ; // ! set flag for observability
  if (limitReached) {
    // ! Rate limit hit: return 429 with no response body
    return res.status(429).end();
  }
  requestTimes.push(now);
  next();
}

// Lightweight status endpoint to expose current Gemini rate limit state for the frontend
app.get('/gemini/limit', (req, res) => {
  const now = Date.now()
  // Refresh rolling window
  requestTimes = requestTimes.filter(t => now - t < WINDOW_MS)
  const reached = requestTimes.length >= MAX_REQ
  limitReached = reached
  const remaining = Math.max(0, MAX_REQ - requestTimes.length)
  let msRemaining = 0
  if (reached && requestTimes.length) {
    // Time until the oldest request drops out of the rolling window
    msRemaining = Math.max(0, WINDOW_MS - (now - requestTimes[0]))
  }
  const secondsRemaining = Math.ceil(msRemaining / 1000)
  res.json({ limit: reached ? 1 : 0, remaining, windowMs: WINDOW_MS, secondsRemaining })
})

// One-time (per client decision) warm-up of OpenAI TTS path to reduce first-byte latency
app.post('/tts/warm', async (req, res) => {
  if (!process.env.OPENAI_TTS_API_KEY) {
    return res.status(500).json({ error: 'Server misconfigured: missing OPENAI_TTS_API_KEY' })
  }
  try {
    const response = await fetch('https://api.openai.com/v1/audio/speech', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.OPENAI_TTS_API_KEY}`,
        'Content-Type': 'application/json',
      },
      // ultra-short input to prime model and connection
      body: JSON.stringify({ model: 'gpt-4o-mini-tts', voice: 'alloy', input: 'ok', format: 'mp3' }),
      agent: openAiAgent,
    })
    if (!response.ok) {
      // Do not propagate upstream error details; treat as best-effort warm
      return res.status(204).end()
    }
    // Drain minimal bytes without buffering entire body
    const reader = response.body?.getReader?.()
    if (reader) {
      try { await reader.read() } catch {}
    }
  } catch {}
  // Always 204 to avoid client waiting
  return res.status(204).end()
})

// Lightweight ping for client preconnect
app.get('/ping', (req, res) => res.status(204).end())
// Proxy endpoint for Gemini text generation (kept for popup functionality)
app.post('/gemini', geminiRateLimiter, async (req, res) => {
  try {
    const { prompt, model } = req.body || {}
    if (!prompt || typeof prompt !== 'string') {
      return res.status(400).json({ error: 'Missing prompt' })
    }
    // Retry on temporary overload conditions
    const maxAttempts = 3
    const baseDelay = 200 // ms
    let lastErr
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        const text = await generateContent(prompt, { model })
        return res.json({ text })
      } catch (e) {
        lastErr = e
        const msg = String(e?.message || '')
        const overloaded = /\b(429|resource\s*exhausted|quota|unavailable|503)\b/i.test(msg)
        const shouldRetry = overloaded && attempt < maxAttempts
        if (shouldRetry) {
          const delay = baseDelay * Math.pow(2, attempt - 1)
          await new Promise(r => setTimeout(r, delay))
          continue
        }
        break
      }
    }
    console.warn('[Gemini] Failing gracefully after retries:', lastErr?.message || lastErr)
    // Do not crash the client: reply with a friendly fallback text
    return res.status(200).json({ text: 'Gemini is overloaded please try again later' })
  } catch (e) {
    console.error('[Gemini] Server error:', e)
    const status = (typeof e?.message === 'string' && e.message.includes('tokens max')) ? 400 : 500
    // Graceful fallback even on unexpected errors
    if (status >= 500) {
      return res.status(200).json({ text: 'Gemini is overloaded please try again later' })
    }
    res.status(status).json({ error: e.message || 'Gemini request failed' })
  }
})

// Silent variant: logs and returns 204 No Content
app.post('/gemini/silent', geminiRateLimiter, async (req, res) => {
  try {
    const { prompt, model } = req.body || {}
    if (!prompt || typeof prompt !== 'string') {
      console.error('[Gemini] Missing prompt')
      return res.status(400).end()
    }
    const text = await generateContent(prompt, { model })
    console.log('[Gemini] Response:', text)
    res.status(204).end()
  } catch (e) {
    console.error('[Gemini] Server error:', e)
    const status = (typeof e?.message === 'string' && e.message.includes('tokens max')) ? 400 : 500
    res.status(status).end()
  }
})

// Streaming proxy endpoint using Server-Sent Events (SSE) (kept for popup functionality)
app.post('/gemini/stream', geminiRateLimiter, async (req, res) => {
  try {
    const { prompt, model } = req.body || {}
    if (!prompt || typeof prompt !== 'string') {
      return res.status(400).json({ error: 'Missing prompt' })
    }
    // Set SSE headers
    res.setHeader('Content-Type', 'text/event-stream')
    res.setHeader('Cache-Control', 'no-cache, no-transform')
    res.setHeader('Connection', 'keep-alive')
    // CORS for SSE
    res.setHeader('Access-Control-Allow-Origin', '*')

    // Flush headers
    res.flushHeaders?.()

    // Heartbeat to keep connection alive
    const hb = setInterval(() => {
      try { res.write(': ping\n\n') } catch {}
    }, 15000)

    const upstream = await streamGenerateContent(prompt, { model })
    const bodyStream = upstream.body
    if (!bodyStream) {
      throw new Error('No upstream body')
    }

    const decoder = new TextDecoder('utf-8')
    let buffer = ''

    const flushDone = () => {
      clearInterval(hb)
      try { res.write('event: done\n') } catch {}
      try { res.write('data: "[DONE]"\n\n') } catch {}
      res.end()
    }

    const emitTextsFromObj = (obj) => {
      const candidates = obj?.candidates
      if (!Array.isArray(candidates)) return
      for (const c of candidates) {
        const parts = c?.content?.parts
        if (Array.isArray(parts)) {
          for (const p of parts) {
            const t = p?.text
            if (typeof t === 'string' && t.length) {
              res.write(`data: ${JSON.stringify(t)}\n\n`)
            }
          }
        }
      }
    }

    const handleText = (textChunk) => {
      buffer += textChunk
      // Prefer parsing complete SSE frames separated by double newlines
      let frameIdx
      while ((frameIdx = buffer.indexOf('\n\n')) !== -1) {
        const frame = buffer.slice(0, frameIdx)
        buffer = buffer.slice(frameIdx + 2)
        // A frame may contain multiple lines like "data: ..."
        for (const rawLine of frame.split(/\r?\n/)) {
          let line = rawLine.trim()
          if (!line || line.startsWith(':') || line.toLowerCase().startsWith('event:')) continue
          if (line.toLowerCase().startsWith('data:')) line = line.slice(5).trim()
          if (!line) continue
          try {
            const obj = JSON.parse(line)
            emitTextsFromObj(obj)
          } catch {
            // ignore partials
          }
        }
      }

      // Also handle newline-delimited JSON objects in case upstream isn't SSE-framed
      let idx
      while ((idx = buffer.indexOf('\n')) !== -1) {
        const line0 = buffer.slice(0, idx)
        buffer = buffer.slice(idx + 1)
        let line = line0.trim()
        if (!line || line.startsWith(':') || line.toLowerCase().startsWith('event:')) continue
        if (line.toLowerCase().startsWith('data:')) line = line.slice(5).trim()
        if (!line) continue
        try {
          const obj = JSON.parse(line)
          emitTextsFromObj(obj)
        } catch {
          // ignore
        }
      }
    }

    // Web ReadableStream path
    if (typeof bodyStream.getReader === 'function') {
      const reader = bodyStream.getReader()
      ;(async () => {
        try {
          while (true) {
            const { value, done } = await reader.read()
            if (done) break
            handleText(decoder.decode(value, { stream: true }))
          }
          if (buffer.trim()) {
            handleText('\n')
          }
          flushDone()
        } catch (err) {
          clearInterval(hb)
          console.error('[Gemini] stream upstream error:', err)
          try { res.write(`event: error\ndata: ${JSON.stringify(err?.message || 'stream error')}\n\n`) } catch {}
          res.end()
        }
      })()
    } else if (typeof bodyStream.on === 'function') {
      // Node.js Readable path
      bodyStream.on('data', (chunk) => {
        const text = Buffer.isBuffer(chunk) ? decoder.decode(chunk, { stream: true }) : String(chunk)
        handleText(text)
      })
      bodyStream.on('end', () => {
        if (buffer.trim()) {
          handleText('\n')
        }
        flushDone()
      })
      bodyStream.on('error', (err) => {
        clearInterval(hb)
        console.error('[Gemini] stream upstream error:', err)
        try { res.write(`event: error\ndata: ${JSON.stringify(err?.message || 'stream error')}\n\n`) } catch {}
        res.end()
      })
    } else if (bodyStream[Symbol.asyncIterator]) {
      // Async iterator fallback
      ;(async () => {
        try {
          for await (const chunk of bodyStream) {
            const text = Buffer.isBuffer(chunk) ? decoder.decode(chunk, { stream: true }) : String(chunk)
            handleText(text)
          }
          if (buffer.trim()) {
            handleText('\n')
          }
          flushDone()
        } catch (err) {
          clearInterval(hb)
          console.error('[Gemini] stream upstream error:', err)
          try { res.write(`event: error\ndata: ${JSON.stringify(err?.message || 'stream error')}\n\n`) } catch {}
          res.end()
        }
      })()
    } else {
      throw new Error('Unknown upstream body stream type')
    }
  } catch (e) {
    console.error('[Gemini] stream setup error:', e)
    if (!res.headersSent) {
      res.status(500).json({ error: e.message || 'Gemini stream failed' })
    } else {
      try { res.write(`event: error\ndata: ${JSON.stringify(e?.message || 'stream error')}\n\n`) } catch {}
      res.end()
    }
  }
})

// Silent streaming variant: consumes stream, logs chunks, 204 No Content
app.post('/gemini/stream/silent', geminiRateLimiter, async (req, res) => {
  try {
    const { prompt, model } = req.body || {}
    if (!prompt || typeof prompt !== 'string') {
      console.error('[Gemini] Missing prompt')
      return res.status(400).end()
    }
    const upstream = await streamGenerateContent(prompt, { model })
    const bodyStream = upstream.body
    if (!bodyStream) throw new Error('No upstream body')

    const decoder = new TextDecoder('utf-8')
    let buffer = ''

    const emitTextsFromObj = (obj) => {
      const candidates = obj?.candidates
      if (!Array.isArray(candidates)) return
      for (const c of candidates) {
        const parts = c?.content?.parts
        if (Array.isArray(parts)) {
          for (const p of parts) {
            const t = p?.text
            if (typeof t === 'string' && t.length) {
              console.log('[Gemini][stream] chunk:', t)
            }
          }
        }
      }
    }

    const handleText = (textChunk) => {
      buffer += textChunk
      let frameIdx
      while ((frameIdx = buffer.indexOf('\n\n')) !== -1) {
        const frame = buffer.slice(0, frameIdx)
        buffer = buffer.slice(frameIdx + 2)
        for (const rawLine of frame.split(/\r?\n/)) {
          let line = rawLine.trim()
          if (!line || line.startsWith(':') || line.toLowerCase().startsWith('event:')) continue
          if (line.toLowerCase().startsWith('data:')) line = line.slice(5).trim()
          if (!line) continue
          try { emitTextsFromObj(JSON.parse(line)) } catch {}
        }
      }

      let idx
      while ((idx = buffer.indexOf('\n')) !== -1) {
        const line0 = buffer.slice(0, idx)
        buffer = buffer.slice(idx + 1)
        let line = line0.trim()
        if (!line || line.startsWith(':') || line.toLowerCase().startsWith('event:')) continue
        if (line.toLowerCase().startsWith('data:')) line = line.slice(5).trim()
        if (!line) continue
        try { emitTextsFromObj(JSON.parse(line)) } catch {}
      }
    }

    if (typeof bodyStream.getReader === 'function') {
      const reader = bodyStream.getReader()
      try {
        while (true) {
          const { value, done } = await reader.read()
          if (done) break
          handleText(decoder.decode(value, { stream: true }))
        }
        if (buffer.trim()) handleText('\n')
        res.status(204).end()
      } catch (err) {
        console.error('[Gemini] stream upstream error:', err)
        res.status(500).end()
      }
    } else if (typeof bodyStream.on === 'function') {
      bodyStream.on('data', (chunk) => {
        const text = Buffer.isBuffer(chunk) ? decoder.decode(chunk, { stream: true }) : String(chunk)
        handleText(text)
      })
      bodyStream.on('end', () => {
        if (buffer.trim()) handleText('\n')
        res.status(204).end()
      })
      bodyStream.on('error', (err) => {
        console.error('[Gemini] stream upstream error:', err)
        res.status(500).end()
      })
    } else if (bodyStream[Symbol.asyncIterator]) {
      try {
        for await (const chunk of bodyStream) {
          const text = Buffer.isBuffer(chunk) ? decoder.decode(chunk, { stream: true }) : String(chunk)
          handleText(text)
        }
        if (buffer.trim()) handleText('\n')
        res.status(204).end()
      } catch (err) {
        console.error('[Gemini] stream upstream error:', err)
        res.status(500).end()
      }
    } else {
      throw new Error('Unknown upstream body stream type')
    }
  } catch (e) {
    console.error('[Gemini] stream setup error:', e)
    if (!res.headersSent) {
      res.status(500).end()
    }
  }
})

app.post("/tts", async (req, res) => {
  const { text, voice = "alloy", format = "wav", model = "gpt-4o-mini-tts" } = req.body;

  if (!text) return res.status(400).json({ error: "No text provided" });

  if (!process.env.OPENAI_TTS_API_KEY) {
    console.error("[TTS] Missing OPENAI_TTS_API_KEY in server/.env");
    return res.status(500).json({ error: "Server misconfigured: missing OPENAI_TTS_API_KEY" });
  }

  try {
    const response = await fetch("https://api.openai.com/v1/audio/speech", {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${process.env.OPENAI_TTS_API_KEY}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ model, voice, input: text, format }),
      // Reuse connection
      agent: openAiAgent,
    });

    if (!response.ok) {
      let detail = ''
      try {
        const json = await response.json()
        detail = json?.error?.message || JSON.stringify(json)
      } catch {
        try { detail = await response.text() } catch {}
      }
      console.error(`[TTS] OpenAI error ${response.status} ${response.statusText}: ${detail}`)
      return res.status(response.status).json({ error: detail || 'OpenAI TTS request failed' });
    }

    const arrayBuffer = await response.arrayBuffer();
    const buffer = Buffer.from(arrayBuffer);

    res.set({
      "Content-Type": `audio/${format}`,
      "Content-Length": buffer.length,
    });
    res.send(buffer);
  } catch (e) {
    console.error('[TTS] Server error:', e)
    res.status(500).json({ error: e.message });
  }
});

// ! Streaming TTS endpoint: progressively pipe OpenAI audio to the client for low latency
app.post('/tts/stream', async (req, res) => {
  const { text, voice = 'alloy', format = 'mp3', model = 'gpt-4o-mini-tts' } = req.body || {}
  if (!text) return res.status(400).json({ error: 'No text provided' })
  if (!process.env.OPENAI_TTS_API_KEY) {
    console.error('[TTS] Missing OPENAI_TTS_API_KEY in server/.env')
    return res.status(500).json({ error: 'Server misconfigured: missing OPENAI_TTS_API_KEY' })
  }
  try {
    const upstream = await fetch('https://api.openai.com/v1/audio/speech', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.OPENAI_TTS_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ model, voice, input: text, format }),
      // Reuse connection
      agent: openAiAgent,
    })
    if (!upstream.ok) {
      let detail = ''
      try {
        const json = await upstream.json()
        detail = json?.error?.message || JSON.stringify(json)
      } catch {
        try { detail = await upstream.text() } catch {}
      }
      console.error(`[TTS] OpenAI stream error ${upstream.status} ${upstream.statusText}: ${detail}`)
      return res.status(upstream.status).json({ error: detail || 'OpenAI TTS request failed' })
    }

    // ! Stream headers; rely on chunked transfer encoding (no Content-Length)
    res.setHeader('Content-Type', `audio/${format}`)
    res.setHeader('Cache-Control', 'no-cache, no-transform')
    res.setHeader('Connection', 'keep-alive')
    res.setHeader('Access-Control-Allow-Origin', '*')

    const body = upstream.body
    if (!body) {
      return res.status(502).json({ error: 'No upstream body' })
    }

    // Node readable or Web stream
    if (typeof body.pipe === 'function') {
      body.pipe(res)
      body.on('error', (err) => {
        console.error('[TTS] stream upstream error:', err)
        try { res.end() } catch {}
      })
      body.on('end', () => {
        try { res.end() } catch {}
      })
    } else if (typeof body.getReader === 'function') {
      const reader = body.getReader()
      ;(async () => {
        try {
          while (true) {
            const { value, done } = await reader.read()
            if (done) break
            if (value) res.write(Buffer.from(value))
          }
        } catch (err) {
          console.error('[TTS] stream upstream error:', err)
        } finally {
          try { res.end() } catch {}
        }
      })()
    } else if (body[Symbol.asyncIterator]) {
      ;(async () => {
        try {
          for await (const chunk of body) {
            res.write(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk))
          }
        } catch (err) {
          console.error('[TTS] stream upstream error:', err)
        } finally {
          try { res.end() } catch {}
        }
      })()
    } else {
      return res.status(502).json({ error: 'Unknown upstream stream type' })
    }
  } catch (e) {
    console.error('[TTS] Server stream error:', e)
    if (!res.headersSent) return res.status(500).json({ error: e.message || 'TTS stream failed' })
    try { res.end() } catch {}
  }
})

// GET variant to allow direct <audio src> streaming with query params
app.get('/tts/stream', async (req, res) => {
  const { text, voice = 'alloy', format = 'mp3', model = 'gpt-4o-mini-tts' } = req.query || {}
  if (!text) return res.status(400).json({ error: 'No text provided' })
  if (!process.env.OPENAI_TTS_API_KEY) {
    console.error('[TTS] Missing OPENAI_TTS_API_KEY in server/.env')
    return res.status(500).json({ error: 'Server misconfigured: missing OPENAI_TTS_API_KEY' })
  }
  try {
    const upstream = await fetch('https://api.openai.com/v1/audio/speech', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.OPENAI_TTS_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ model, voice, input: text, format }),
      agent: openAiAgent,
    })
    if (!upstream.ok) {
      let detail = ''
      try {
        const json = await upstream.json()
        detail = json?.error?.message || JSON.stringify(json)
      } catch {
        try { detail = await upstream.text() } catch {}
      }
      console.error(`[TTS] OpenAI stream error ${upstream.status} ${upstream.statusText}: ${detail}`)
      return res.status(upstream.status).json({ error: detail || 'OpenAI TTS request failed' })
    }

    res.setHeader('Content-Type', `audio/${format}`)
    res.setHeader('Cache-Control', 'no-cache, no-transform')
    res.setHeader('Connection', 'keep-alive')
    res.setHeader('Access-Control-Allow-Origin', '*')

    const body = upstream.body
    if (!body) return res.status(502).json({ error: 'No upstream body' })

    if (typeof body.pipe === 'function') {
      body.pipe(res)
      body.on('error', (err) => { console.error('[TTS][GET] stream upstream error:', err); try { res.end() } catch {} })
      body.on('end', () => { try { res.end() } catch {} })
    } else if (typeof body.getReader === 'function') {
      const reader = body.getReader()
      ;(async () => {
        try {
          while (true) {
            const { value, done } = await reader.read()
            if (done) break
            if (value) res.write(Buffer.from(value))
          }
        } catch (err) {
          console.error('[TTS][GET] stream upstream error:', err)
        } finally {
          try { res.end() } catch {}
        }
      })()
    } else if (body[Symbol.asyncIterator]) {
      ;(async () => {
        try {
          for await (const chunk of body) {
            res.write(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk))
          }
        } catch (err) {
          console.error('[TTS][GET] stream upstream error:', err)
        } finally {
          try { res.end() } catch {}
        }
      })()
    } else {
      return res.status(502).json({ error: 'Unknown upstream stream type' })
    }
  } catch (e) {
    console.error('[TTS][GET] Server stream error:', e)
    if (!res.headersSent) return res.status(500).json({ error: e.message || 'TTS stream failed' })
    try { res.end() } catch {}
  }
})

const port = process.env.PORT || 3000;
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});

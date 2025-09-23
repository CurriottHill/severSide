import express from "express";
import fetch from "node-fetch";
import dotenv from "dotenv";
import cors from 'cors';
import https from 'https';
import { generateContent, streamGenerateContent } from './gemini.js'
import path from 'path';

// ! Load environment variables from .env
dotenv.config();

import pkg from 'pg';
const { Pool } = pkg;
import Stripe from 'stripe';

const stripe = new Stripe(process.env.STRIPE_SECRET_KEY);
const pool = new Pool({
  host: process.env.DB_HOST,
  user: process.env.DB_USER || 'postgres',
  password: process.env.DB_PASS || '',
  database: process.env.DB_NAME || 'mydb',
  port: process.env.DB_PORT || 5432,
});

// Test DB connection
pool.on('connect', () => console.log('Connected to PostgreSQL'));
pool.on('error', (err) => console.error('PostgreSQL error:', err));

// Ensure premium column exists
(async () => {
  try {
    await pool.query(`
      ALTER TABLE users ADD COLUMN IF NOT EXISTS premium BOOLEAN DEFAULT FALSE;
    `);
    console.log('Ensured premium column exists');
  } catch (err) {
    console.error('Error adding premium column:', err);
  }
})();

// Auth middleware
const authenticate = async (req, res, next) => {
  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Unauthorized' });
  }
  const token = authHeader.substring(7);
  try {
    const result = await pool.query('SELECT u.id, u.email, u.premium FROM tokens t JOIN users u ON t.user_id = u.id WHERE t.token = $1 AND (t.expired_at IS NULL OR t.expired_at > NOW())', [token]);
    if (result.rows.length === 0) {
      return res.status(401).json({ error: 'Invalid token' });
    }
    req.user = result.rows[0];
    next();
  } catch (err) {
    console.error('Auth error:', err);
    res.status(500).json({ error: 'Authentication failed' });
  }
};

// Create Stripe checkout session
app.post('/create-checkout-session', authenticate, async (req, res) => {
  try {
    const session = await stripe.checkout.sessions.create({
      payment_method_types: ['card'],
      line_items: [{
        price: process.env.STRIPE_PRICE_ID,
        quantity: 1,
      }],
      mode: 'payment',
      success_url: `${req.protocol}://${req.get('host')}/payment-success?session_id={CHECKOUT_SESSION_ID}`,
      cancel_url: `${req.protocol}://${req.get('host')}/payment-cancel`,
      metadata: {
        user_id: req.user.id.toString(),
      },
    });
    res.json({ url: session.url });
  } catch (error) {
    console.error('Checkout session error:', error);
    res.status(500).json({ error: 'Failed to create checkout session' });
  }
});

// Payment success page
app.get('/payment-success', (req, res) => {
  res.send('<h1>Payment successful! You now have unlimited access. Please log in.</h1>');
});

// Payment cancel page
app.get('/payment-cancel', (req, res) => {
  res.send('<h1>Payment cancelled.</h1>');
});

// Stripe webhook
app.post('/webhook', express.raw({ type: 'application/json' }), async (req, res) => {
  const sig = req.headers['stripe-signature'];
  let event;

  try {
    event = stripe.webhooks.constructEvent(req.body, sig, process.env.STRIPE_WEBHOOK_SECRET);
  } catch (err) {
    console.error('Webhook signature verification failed:', err.message);
    return res.status(400).send(`Webhook Error: ${err.message}`);
  }

  if (event.type === 'checkout.session.completed') {
    const session = event.data.object;
    const userId = session.metadata.user_id;
    try {
      await pool.query('UPDATE users SET premium = TRUE WHERE id = $1', [userId]);
      console.log(`User ${userId} upgraded to premium`);
    } catch (err) {
      console.error('Error updating user premium:', err);
    }
  }

  res.json({ received: true });
});

const app = express();
app.use(express.json({ limit: '1mb' }));
app.use("/privacy", express.static(path.join(process.cwd(), "privacy.html")));

// Use permissive defaults (handles preflight automatically)
app.use(cors());
// Explicit global preflight handler (Express 5 path-to-regexp does not accept bare '*')
app.use((req, res, next) => {
  res.header('Access-Control-Allow-Origin', '*')
  res.header('Access-Control-Allow-Methods', 'GET,POST,PUT,PATCH,DELETE,OPTIONS')
  res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
  if (req.method === 'OPTIONS') {
    return res.status(204).end()
  }
  next()
})

// Reuse TLS connections to OpenAI to reduce latency
const openAiAgent = new https.Agent({ keepAlive: true })

// ! Simple in-memory rate limiter: 5 requests per minute per user, unlimited for premium
const WINDOW_MS = 60_000, MAX_REQ = 10; // ! 60s window, max 10 hits per user
let userRequestTimes = new Map(); // user_id => rolling timestamps (ms)

function geminiRateLimiter(req, res, next) {
  const userId = req.user.id;
  if (req.user.premium) {
    return next(); // No limit for premium
  }
  const now = Date.now();
  if (!userRequestTimes.has(userId)) {
    userRequestTimes.set(userId, []);
  }
  const times = userRequestTimes.get(userId);
  times.push(now);
  times.splice(0, times.length - MAX_REQ); // Keep only last MAX_REQ
  const recent = times.filter(t => now - t < WINDOW_MS);
  userRequestTimes.set(userId, recent);
  if (recent.length >= MAX_REQ) {
    return res.status(429).json({ error: 'Quota exceeded. Please upgrade to premium.', buy: true });
  }
  next();
}

// Lightweight status endpoint to expose current Gemini rate limit state for the frontend
app.get('/gemini/limit', authenticate, (req, res) => {
  const userId = req.user.id;
  if (req.user.premium) {
    return res.json({ limit: 0, remaining: Infinity, windowMs: WINDOW_MS, secondsRemaining: 0 });
  }
  const now = Date.now();
  const times = userRequestTimes.get(userId) || [];
  const recent = times.filter(t => now - t < WINDOW_MS);
  const reached = recent.length >= MAX_REQ;
  const remaining = Math.max(0, MAX_REQ - recent.length);
  let msRemaining = 0;
  if (reached && recent.length) {
    msRemaining = Math.max(0, WINDOW_MS - (now - recent[0]));
  }
  const secondsRemaining = Math.ceil(msRemaining / 1000);
  res.json({ limit: reached ? 1 : 0, remaining, windowMs: WINDOW_MS, secondsRemaining });
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
app.post('/gemini', authenticate, geminiRateLimiter, async (req, res) => {
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
app.post('/gemini/silent', authenticate, geminiRateLimiter, async (req, res) => {
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
app.post('/gemini/stream', authenticate, geminiRateLimiter, async (req, res) => {
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
app.post('/gemini/stream/silent', authenticate, geminiRateLimiter, async (req, res) => {
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

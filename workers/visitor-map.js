/**
 * Visitor Map Worker — ru00ys-lab.com
 * Deploy to Cloudflare Workers at: api.ru00ys-lab.com
 */

const OWNER_COOKIE = 'ru00y-owner'
const STATE_COOKIE = 'ru00y-oauth-state'

function randomState() {
  const bytes = new Uint8Array(16)
  if (globalThis.crypto?.getRandomValues) {
    globalThis.crypto.getRandomValues(bytes)
    return Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('')
  }

  return Math.random().toString(36).slice(2) + Date.now().toString(36)
}

function cookieStr(name, value, maxAge = 31536000) {
  return `${name}=${value}; HttpOnly; Secure; SameSite=Lax; Path=/; Max-Age=${maxAge}`
}

function cookieValue(cookie, name) {
  const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const match = cookie.match(new RegExp(`(?:^|;\\s*)${escaped}=([^;]*)`))
  return match?.[1] || ''
}

// Known bot/crawler User-Agent patterns to filter out
const BOT_PATTERNS = [
  /bot/i, /crawler/i, /spider/i, /scraper/i, /curl/i, /wget/i,
  /headless/i, /selenium/i, /puppeteer/i, /playwright/i,
  /googlebot/i, /bingbot/i, /slurp/i, /duckduckbot/i, /baiduspider/i,
  /yandex/i, /sogou/i, /facebookexternalhit/i, /twitterbot/i,
  /gptbot/i, /chatgpt/i, /ccbot/i, /anthropic/i, /claude/i,
  /bytespider/i, /petalbot/i, /ahrefs/i, /semrush/i, /rogerbot/i,
  /dotbot/i, /mj12bot/i, /barkrowler/i, /BLEXBot/i,
]

function isBot(request) {
  const ua = request.headers.get('User-Agent') || ''
  return BOT_PATTERNS.some(p => p.test(ua))
}

function redirect(location, cookies = []) {
  const headers = new Headers({ Location: location })
  for (const cookie of cookies) headers.append('Set-Cookie', cookie)
  return new Response(null, { status: 302, headers })
}

async function hmac(data, secret) {
  const enc = new TextEncoder()
  const key = await crypto.subtle.importKey('raw', enc.encode(secret), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign'])
  const sig = await crypto.subtle.sign('HMAC', key, enc.encode(data))
  const bytes = new Uint8Array(sig)
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('')
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url)
    const secret = env.COOKIE_SECRET
    function corsHeaders(methods = 'GET, OPTIONS') {
      return {
        'Access-Control-Allow-Origin': 'https://ru00ys-lab.com',
        'Access-Control-Allow-Methods': methods,
        'Access-Control-Allow-Credentials': 'true',
      }
    }
    if (request.method === 'OPTIONS') {
      let methods = 'GET, OPTIONS'
      const allowHeaders = []
      if (url.pathname === '/subscribe') { methods = 'GET, POST, OPTIONS'; allowHeaders.push('Content-Type') }
      if (url.pathname === '/subscribers') { methods = 'GET, DELETE, OPTIONS'; allowHeaders.push('Content-Type') }
      const h = corsHeaders(methods)
      if (allowHeaders.length) h['Access-Control-Allow-Headers'] = allowHeaders.join(', ')
      return new Response(null, { headers: h })
    }

    // /login → redirect to GitHub
    if (url.pathname === '/login') {
      const state = randomState()
      const qs = new URLSearchParams({
        client_id: env.GITHUB_CLIENT_ID,
        redirect_uri: 'https://api.ru00ys-lab.com/callback',
        state,
        scope: 'read:user',
      }).toString()
      return redirect('https://github.com/login/oauth/authorize?' + qs, [
        cookieStr(STATE_COOKIE, state, 600),
      ])
    }

    // /callback → GitHub OAuth callback
    if (url.pathname === '/callback') {
      const code = url.searchParams.get('code')
      const state = url.searchParams.get('state')
      const cookie = request.headers.get('Cookie') || ''
      const savedState = cookieValue(cookie, STATE_COOKIE)

      if (!code || !state || !savedState || savedState !== state) {
        return new Response('Invalid state', { status: 403 })
      }

      const tok = await fetch('https://github.com/login/oauth/access_token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({
          client_id: env.GITHUB_CLIENT_ID,
          client_secret: env.GITHUB_CLIENT_SECRET,
          code,
        }),
      })
      const data = await tok.json()
      if (!data.access_token) return new Response('Auth failed', { status: 403 })

      const userRes = await fetch('https://api.github.com/user', {
        headers: { Authorization: 'Bearer ' + data.access_token, 'User-Agent': 'ru00y-lab' },
      })
      const user = await userRes.json()
      const ownerId = parseInt(env.OWNER_GITHUB_ID || '0')

      if (user.id !== ownerId) {
        return redirect('https://ru00ys-lab.com/?auth=denied')
      }

      const sig = await hmac(String(user.id), secret)
      const val = user.id + ':' + sig
      return redirect('https://ru00ys-lab.com/connect', [
        cookieStr(OWNER_COOKIE, val),
        cookieStr(STATE_COOKIE, '', 0),
      ])
    }

    // /me — check owner
    if (url.pathname === '/me') {
      const cookie = request.headers.get('Cookie') || ''
      const [uid, sig] = cookieValue(cookie, OWNER_COOKIE).split(':')
      const ok = Boolean(uid && sig && (await hmac(uid, secret)) === sig)
      return new Response(JSON.stringify({ ok }), { headers: { ...corsHeaders(), 'Content-Type': 'application/json' } })
    }

    // /track — record (skip bots + owner)
    if (url.pathname === '/track') {
      if (isBot(request)) return new Response('ok', { headers: corsHeaders() })
      const cookie = request.headers.get('Cookie') || ''
      const [uid, sig] = cookieValue(cookie, OWNER_COOKIE).split(':')
      if (uid && sig && (await hmac(uid, secret)) === sig) {
        return new Response('ok', { headers: corsHeaders() })
      }

      const cf = request.cf || {}
      const ip = request.headers.get('cf-connecting-ip') || 'unknown'
      const today = new Date().toISOString().slice(0, 10)
      const entry = {
        ip, city: cf.city || null, region: cf.region || null, country: cf.country || null,
        lat: cf.latitude || null, lon: cf.longitude || null, date: today, time: Date.now(),
      }
      if (!entry.country) return new Response('ok', { headers: corsHeaders() })

      let raw = await env.VISITORS.get('data')
      let visitors = raw ? JSON.parse(raw) : []
      visitors = visitors.filter(v => !(v.ip === ip && v.date === today))
      visitors.push(entry)
      if (visitors.length > 500) visitors = visitors.slice(-500)

      const rounded = visitors.map(v => ({
        lat: v.lat ? Math.round(v.lat * 100) / 100 : null,
        lon: v.lon ? Math.round(v.lon * 100) / 100 : null,
        city: v.city, region: v.region, country: v.country,
      }))
      await env.VISITORS.put('data', JSON.stringify(visitors))
      await env.VISITORS.put('public', JSON.stringify(rounded))
      return new Response('ok', { headers: corsHeaders() })
    }

    // /subscribe — mirror email to KV for owner dashboard
    if (url.pathname === '/subscribe' && request.method === 'POST') {
      try {
        const { email } = await request.json()
        if (!email || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)) {
          return new Response(JSON.stringify({ ok: false, error: 'Invalid email' }),
            { status: 400, headers: { ...corsHeaders('POST, OPTIONS'), 'Content-Type': 'application/json' } })
        }
        let raw = await env.VISITORS.get('subscribers')
        let subs = raw ? JSON.parse(raw) : []
        if (!subs.find(s => s.email === email)) {
          subs.push({ email, time: Date.now() })
          if (subs.length > 1000) subs = subs.slice(-1000)
          await env.VISITORS.put('subscribers', JSON.stringify(subs))
        }
        return new Response(JSON.stringify({ ok: true }),
          { headers: { ...corsHeaders('POST, OPTIONS'), 'Content-Type': 'application/json' } })
      } catch {
        return new Response(JSON.stringify({ ok: false }), { status: 400,
          headers: { ...corsHeaders('POST, OPTIONS'), 'Content-Type': 'application/json' } })
      }
    }

    // /data
    if (url.pathname === '/data') {
      const raw = await env.VISITORS.get('public')
      if (!raw) return new Response('[]', { headers: { ...corsHeaders(), 'Content-Type': 'application/json' } })
      return new Response(raw, { headers: { ...corsHeaders(), 'Content-Type': 'application/json' } })
    }

    // /subscribers — owner-only: GET list, DELETE by email
    if (url.pathname === '/subscribers') {
      const cookie = request.headers.get('Cookie') || ''
      const [uid, sig] = cookieValue(cookie, OWNER_COOKIE).split(':')
      const isOwner = Boolean(uid && sig && (await hmac(uid, secret)) === sig)
      if (!isOwner) {
        return new Response(JSON.stringify({ ok: false }),
          { status: 403, headers: { ...corsHeaders(), 'Content-Type': 'application/json' } })
      }

      if (request.method === 'DELETE') {
        try {
          const { email } = await request.json()
          let raw = await env.VISITORS.get('subscribers')
          let subs = raw ? JSON.parse(raw) : []
          subs = subs.filter(s => s.email !== email)
          await env.VISITORS.put('subscribers', JSON.stringify(subs))
          return new Response(JSON.stringify({ ok: true, count: subs.length }),
            { headers: { ...corsHeaders('DELETE, OPTIONS'), 'Content-Type': 'application/json' } })
        } catch {
          return new Response(JSON.stringify({ ok: false }),
            { status: 400, headers: { ...corsHeaders(), 'Content-Type': 'application/json' } })
        }
      }

      const raw = await env.VISITORS.get('subscribers')
      const subs = raw ? JSON.parse(raw) : []
      return new Response(JSON.stringify({ ok: true, count: subs.length, subscribers: subs }),
        { headers: { ...corsHeaders(), 'Content-Type': 'application/json' } })
    }

    return new Response('Not found', { status: 404, headers: corsHeaders() })
  },
}

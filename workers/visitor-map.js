/**
 * Visitor Map Worker — ru00ys-lab.com
 * Deploy to Cloudflare Workers at: api.ru00ys-lab.com
 *
 * GET  /track     — record visitor (skip if owner cookie present)
 * GET  /data      — return deduplicated GeoJSON
 * GET  /login     — redirect to GitHub OAuth
 * GET  /callback  — GitHub OAuth callback, set owner cookie
 * GET  /me        — check if logged in
 */

const OWNER_GITHUB_ID = 'your-github-id' // replace with your numeric GitHub ID
const COOKIE_NAME = 'ru00y-owner'
const COOKIE_SECRET = 'replace-with-random-secret' // Worker secret: COOKIE_SECRET

async function sign(data) {
  const enc = new TextEncoder()
  const key = await crypto.subtle.importKey('raw', enc.encode(COOKIE_SECRET), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign'])
  const sig = await crypto.subtle.sign('HMAC', key, enc.encode(data))
  return btoa(String.fromCharCode(...new Uint8Array(sig)))
}

async function verify(data, sig) {
  return sig === await sign(data)
}

function cookieHeader(value, maxAge = 31536000) {
  return `${COOKIE_NAME}=${value}; HttpOnly; Secure; SameSite=Lax; Path=/; Max-Age=${maxAge}`
}

const CORS = {
  'Access-Control-Allow-Origin': 'https://ru00ys-lab.com',
  'Access-Control-Allow-Methods': 'GET, OPTIONS',
  'Access-Control-Allow-Credentials': 'true',
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url)
    COOKIE_SECRET = env.COOKIE_SECRET || COOKIE_SECRET

    // --- /login: redirect to GitHub ---
    if (url.pathname === '/login') {
      const state = crypto.randomUUID()
      const params = new URLSearchParams({
        client_id: env.GITHUB_CLIENT_ID,
        redirect_uri: 'https://api.ru00ys-lab.com/callback',
        state,
        scope: 'read:user',
      })
      const res = Response.redirect('https://github.com/login/oauth/authorize?' + params, 302)
      res.headers.set('Set-Cookie', cookieHeader('state:' + state, 600))
      return res
    }

    // --- /callback: GitHub OAuth callback ---
    if (url.pathname === '/callback') {
      const code = url.searchParams.get('code')
      const state = url.searchParams.get('state')
      const cookie = request.headers.get('Cookie') || ''

      // Verify state
      if (!code || !cookie.includes(COOKIE_NAME + '=state:' + state)) {
        return new Response('Invalid state', { status: 403 })
      }

      // Exchange code for token
      const tokenRes = await fetch('https://github.com/login/oauth/access_token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({
          client_id: env.GITHUB_CLIENT_ID,
          client_secret: env.GITHUB_CLIENT_SECRET,
          code,
        }),
      })
      const tokenData = await tokenRes.json()
      if (!tokenData.access_token) {
        return new Response('Auth failed', { status: 403 })
      }

      // Get user info
      const userRes = await fetch('https://api.github.com/user', {
        headers: { Authorization: 'Bearer ' + tokenData.access_token, 'User-Agent': 'ru00y-lab' },
      })
      const user = await userRes.json()

      if (user.id !== env.OWNER_GITHUB_ID) {
        // not the owner — redirect back without cookie
        return Response.redirect('https://ru00ys-lab.com/?auth=denied', 302)
      }

      // Owner authenticated — set signed cookie
      const value = user.id + ':' + (await sign(String(user.id)))
      const res = Response.redirect('https://ru00ys-lab.com/?auth=ok', 302)
      res.headers.set('Set-Cookie', cookieHeader(value))
      // Also clear state cookie
      res.headers.append('Set-Cookie', cookieHeader('state:', 0))
      return res
    }

    // --- /me: check auth status ---
    if (url.pathname === '/me') {
      const cookie = request.headers.get('Cookie') || ''
      const match = cookie.match(new RegExp(COOKIE_NAME + '=([^;]+)'))
      if (!match) return new Response('{"ok":false}', { headers: { ...CORS, 'Content-Type': 'application/json' } })

      const [uid, sig] = match[1].split(':')
      if (uid && sig && (await verify(uid, sig))) {
        return new Response('{"ok":true}', { headers: { ...CORS, 'Content-Type': 'application/json' } })
      }
      return new Response('{"ok":false}', { headers: { ...CORS, 'Content-Type': 'application/json' } })
    }

    // --- /track: record visitor (skip if owner) ---
    if (url.pathname === '/track') {
      // Check owner cookie
      const cookie = request.headers.get('Cookie') || ''
      const match = cookie.match(new RegExp(COOKIE_NAME + '=([^;]+)'))
      if (match) {
        const [uid, sig] = match[1].split(':')
        if (uid && sig && (await verify(uid, sig))) {
          return new Response('ok', { headers: CORS }) // owner — skip
        }
      }

      const cf = request.cf || {}
      const ip = request.headers.get('cf-connecting-ip') || 'unknown'
      const now = Date.now()
      const today = new Date().toISOString().slice(0, 10)
      const entry = {
        ip, city: cf.city || null, region: cf.region || null, country: cf.country || null,
        lat: cf.latitude || null, lon: cf.longitude || null, date: today, time: now,
      }

      if (!entry.country) {
        return new Response('ok', { headers: CORS })
      }

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

      return new Response('ok', { headers: CORS })
    }

    // --- /data ---
    if (url.pathname === '/data') {
      const raw = await env.VISITORS.get('public')
      if (!raw) return new Response('[]', { headers: { ...CORS, 'Content-Type': 'application/json' } })
      return new Response(raw, { headers: { ...CORS, 'Content-Type': 'application/json' } })
    }

    if (request.method === 'OPTIONS') return new Response(null, { headers: CORS })

    return new Response('Not found', { status: 404, headers: CORS })
  },
}

/**
 * Visitor Map Worker — ru00ys-lab.com
 * Deploy to Cloudflare Workers at: api.ru00ys-lab.com
 */

function cookieHeader(value, maxAge = 31536000) {
  const name = 'ru00y-owner'
  return `${name}=${value}; HttpOnly; Secure; SameSite=Lax; Path=/; Max-Age=${maxAge}`
}

async function hmacSign(data, secret) {
  const enc = new TextEncoder()
  const key = await crypto.subtle.importKey('raw', enc.encode(secret), { name: 'HMAC', hash: 'SHA-256' }, false, ['sign'])
  const sig = await crypto.subtle.sign('HMAC', key, enc.encode(data))
  return btoa(Array.from(new Uint8Array(sig)).map(b => String.fromCharCode(b)).join(''))
}

async function verify(data, sig, secret) {
  return sig === await hmacSign(data, secret)
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url)
    const secret = env.COOKIE_SECRET || 'fallback-secret'
    const cors = {
      'Access-Control-Allow-Origin': 'https://ru00ys-lab.com',
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Credentials': 'true',
    }
    if (request.method === 'OPTIONS') return new Response(null, { headers: cors })

    // /login — redirect to GitHub
    if (url.pathname === '/login') {
      const state = crypto.randomUUID()
      const params = new URLSearchParams({
        client_id: env.GITHUB_CLIENT_ID || '',
        redirect_uri: 'https://api.ru00ys-lab.com/callback',
        state,
        scope: 'read:user',
      })
      const res = Response.redirect('https://github.com/login/oauth/authorize?' + params.toString(), 302)
      res.headers.set('Set-Cookie', cookieHeader('state:' + state, 600))
      return res
    }

    // /callback — handle GitHub OAuth
    if (url.pathname === '/callback') {
      const code = url.searchParams.get('code')
      const state = url.searchParams.get('state')
      const cookieStr = request.headers.get('Cookie') || ''

      if (!code || !cookieStr.includes('ru00y-owner=state:' + state)) {
        return new Response('Invalid state', { status: 403 })
      }

      const tokenRes = await fetch('https://github.com/login/oauth/access_token', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({
          client_id: env.GITHUB_CLIENT_ID,
          client_secret: env.GITHUB_CLIENT_SECRET,
          code,
        }),
      })
      const data = await tokenRes.json()
      if (!data.access_token) return new Response('Auth failed', { status: 403 })

      const userRes = await fetch('https://api.github.com/user', {
        headers: { Authorization: 'Bearer ' + data.access_token, 'User-Agent': 'ru00y-lab' },
      })
      const user = await userRes.json()
      const ownerId = parseInt(env.OWNER_GITHUB_ID || '0')

      if (user.id !== ownerId) {
        return Response.redirect('https://ru00ys-lab.com/?auth=denied', 302)
      }

      const sig = await hmacSign(String(user.id), secret)
      const value = user.id + ':' + sig
      const res = Response.redirect('https://ru00ys-lab.com/?auth=ok', 302)
      res.headers.set('Set-Cookie', cookieHeader(value))
      res.headers.append('Set-Cookie', cookieHeader('state:', 0))
      return res
    }

    // /me — check login
    if (url.pathname === '/me') {
      const cookieStr = request.headers.get('Cookie') || ''
      const match = cookieStr.match(/ru00y-owner=([^;]+)/)
      if (!match) return new Response('{"ok":false}', { headers: { ...cors, 'Content-Type': 'application/json' } })
      const [uid, sig] = match[1].split(':')
      if (uid && sig && (await verify(uid, sig, secret))) {
        return new Response('{"ok":true}', { headers: { ...cors, 'Content-Type': 'application/json' } })
      }
      return new Response('{"ok":false}', { headers: { ...cors, 'Content-Type': 'application/json' } })
    }

    // /track — record visitor (skip owner)
    if (url.pathname === '/track') {
      const cookieStr = request.headers.get('Cookie') || ''
      const match = cookieStr.match(/ru00y-owner=([^;]+)/)
      if (match) {
        const [uid, sig] = match[1].split(':')
        if (uid && sig && (await verify(uid, sig, secret))) {
          return new Response('ok', { headers: cors })
        }
      }

      const cf = request.cf || {}
      const ip = request.headers.get('cf-connecting-ip') || 'unknown'
      const today = new Date().toISOString().slice(0, 10)
      const entry = {
        ip, city: cf.city || null, region: cf.region || null, country: cf.country || null,
        lat: cf.latitude || null, lon: cf.longitude || null, date: today, time: Date.now(),
      }
      if (!entry.country) return new Response('ok', { headers: cors })

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
      return new Response('ok', { headers: cors })
    }

    // /data
    if (url.pathname === '/data') {
      const raw = await env.VISITORS.get('public')
      if (!raw) return new Response('[]', { headers: { ...cors, 'Content-Type': 'application/json' } })
      return new Response(raw, { headers: { ...cors, 'Content-Type': 'application/json' } })
    }

    return new Response('Not found', { status: 404, headers: cors })
  },
}

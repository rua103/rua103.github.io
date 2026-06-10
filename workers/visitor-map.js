/**
 * Visitor Map Worker — ru00ys-lab.com
 * Deploy to Cloudflare Workers at: api.ru00ys-lab.com
 *
 * GET /track  — record visitor (skip if owner cookie present)
 * GET /data   — return deduplicated GeoJSON
 * GET /owner  — set owner cookie (requires secret key in URL)
 * GET /me     — check if logged in
 */

function cookieHeader(value, maxAge = 31536000) {
  return `ru00y-owner=${value}; HttpOnly; Secure; SameSite=Lax; Path=/; Max-Age=${maxAge}`
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url)
    const cors = {
      'Access-Control-Allow-Origin': 'https://ru00ys-lab.com',
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
      'Access-Control-Allow-Credentials': 'true',
    }
    if (request.method === 'OPTIONS') return new Response(null, { headers: cors })

    // /owner?key=xxx — set owner cookie
    if (url.pathname === '/owner') {
      const key = url.searchParams.get('key') || ''
      const expected = env.OWNER_KEY || 'ru00y-my-secret-key'
      if (key !== expected) {
        return new Response('Wrong key', { status: 403, headers: cors })
      }
      const res = new Response('You are now the owner. Your visits will not be tracked.', { headers: cors })
      res.headers.set('Set-Cookie', cookieHeader('1'))
      return res
    }

    // /me — check owner
    if (url.pathname === '/me') {
      const cookie = request.headers.get('Cookie') || ''
      const ok = cookie.includes('ru00y-owner=1')
      return new Response(JSON.stringify({ ok }), { headers: { ...cors, 'Content-Type': 'application/json' } })
    }

    // /track — record visitor (skip owner)
    if (url.pathname === '/track') {
      const cookie = request.headers.get('Cookie') || ''
      if (cookie.includes('ru00y-owner=1')) {
        return new Response('ok', { headers: cors })
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

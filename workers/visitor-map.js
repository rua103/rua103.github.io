/**
 * Visitor Map Worker for ru00ys-lab.com
 * Deploy to Cloudflare Workers at: api.ru00ys-lab.com
 *
 * GET  /track  — records visitor location (called by client JS on page load)
 * GET  /data   — returns deduplicated visitor locations as GeoJSON
 */

export default {
  async fetch(request, env) {
    const url = new URL(request.url)

    // CORS headers
    const cors = {
      'Access-Control-Allow-Origin': 'https://ru00ys-lab.com',
      'Access-Control-Allow-Methods': 'GET, OPTIONS',
    }
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: cors })
    }

    // --- /track: record visitor ---
    if (url.pathname === '/track') {
      const cf = request.cf || {}
      const ip = request.headers.get('cf-connecting-ip') || 'unknown'
      const entry = {
        ip,
        city: cf.city || null,
        region: cf.region || null,
        country: cf.country || null,
        lat: cf.latitude || null,
        lon: cf.longitude || null,
        colo: cf.colo || null,
        time: Date.now(),
      }

      // Skip if no geo data
      if (!entry.country) {
        return new Response('ok', { headers: cors })
      }

      // Load existing visitors
      let raw = await env.VISITORS.get('data')
      let visitors = raw ? JSON.parse(raw) : []

      // Dedup by IP — keep latest entry, remove old one
      visitors = visitors.filter(v => v.ip !== ip)
      visitors.push(entry)

      // Keep last 500 visitors max
      if (visitors.length > 500) {
        visitors = visitors.slice(-500)
      }

      // Round coordinates for privacy
      const rounded = visitors.map(v => ({
        lat: v.lat ? Math.round(v.lat * 100) / 100 : null,
        lon: v.lon ? Math.round(v.lon * 100) / 100 : null,
        city: v.city,
        region: v.region,
        country: v.country,
      }))

      await env.VISITORS.put('data', JSON.stringify(visitors))
      await env.VISITORS.put('public', JSON.stringify(rounded))

      return new Response('ok', { headers: cors })
    }

    // --- /data: return visitor locations ---
    if (url.pathname === '/data') {
      const raw = await env.VISITORS.get('public')
      if (!raw) {
        return new Response('[]', {
          headers: { ...cors, 'Content-Type': 'application/json' },
        })
      }
      return new Response(raw, {
        headers: { ...cors, 'Content-Type': 'application/json' },
      })
    }

    // --- 404 ---
    return new Response('Not found', { status: 404, headers: cors })
  },
}

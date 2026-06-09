import type { CardListData, Config, IntegrationUserConfig, ThemeUserConfig } from 'astro-pure/types'

export const theme: ThemeUserConfig = {
  title: "Ru00y's Lab",
  author: 'Ru00y',
  description: '♪ A VLM researcher & Mio fan — bass-driven, code-powered.',
  favicon: '/favicon/favicon-32x32.png',
  socialCard: '/images/social-card.png',
  locale: {
    lang: 'en-US',
    attrs: 'en_US',
    dateLocale: 'en-US',
    dateOptions: {
      day: 'numeric',
      month: 'short',
      year: 'numeric'
    }
  },
  logo: {
    src: '/src/assets/avatar.png',
    alt: "Ru00y's avatar"
  },

  titleDelimiter: '♪',
  prerender: true,
  npmCDN: 'https://cdn.jsdelivr.net/npm',

  head: [],
  customCss: [],

  header: {
    menu: [
      { title: 'Blog', link: '/blog' },
      { title: 'Projects', link: '/projects' },
      { title: 'Publications', link: '/publications' },
      { title: 'About', link: '/about' }
    ]
  },

  footer: {
    year: `♪ ${new Date().getFullYear()}`,
    links: [],
    credits: false,
    social: [
      { icon: 'github', label: 'GitHub', href: 'https://github.com/rua103' },
      { icon: 'rss', label: 'RSS', href: '/rss.xml' }
    ]
  },

  content: {
    externalLinks: {
      content: ' ↗',
      properties: { style: 'user-select:none' }
    },
    blogPageSize: 8,
    share: ['weibo', 'x', 'bluesky']
  }
}

export const integ: IntegrationUserConfig = {
  links: {
    logbook: [
      { date: '2026-06-09', content: '♪ Blog launched! — Ru00y\'s Lab is born.' },
    ],
    applyTip: [
      { name: 'Name', val: theme.title! },
      { name: 'Desc', val: theme.description || 'Null' },
      { name: 'Link', val: 'https://rua103.github.io/' },
      { name: 'Avatar', val: 'https://rua103.github.io/favicon/favicon.ico' }
    ],
    cacheAvatar: false
  },
  pagefind: false,
  quote: {
    server: 'https://dummyjson.com/quotes/random',
    target: `(data) => (data.quote.length > 80 ? \`\${data.quote.slice(0, 80)}...\` : data.quote || 'Error')`
  },
  typography: {
    class: 'prose text-base',
    blockquoteStyle: 'italic',
    inlineCodeBlockStyle: 'modern'
  },
  mediumZoom: {
    enable: true,
    selector: '.prose .zoomable',
    options: {
      className: 'zoomable'
    }
  },
  waline: {
    enable: false,
    server: '',
    showMeta: false,
    emoji: [],
    additionalConfigs: {}
  }
}

export const terms: CardListData = {
  title: 'Terms content',
  list: []
}

const config = { ...theme, integ } as Config
export default config

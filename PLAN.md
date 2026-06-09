# Blog Creation Plan: Ru00y's Lab

> Mio Akiyama Fan Shrine × Tech Blog — A Fusion of Japanese Otaku Aesthetics & Clean Technical Blogging

## Context

The user loves [arthals.ink](https://arthals.ink/)'s blog layout skeleton and is a devoted Akiyama Mio (K-ON!) fan. The goal is to build a personal blog that feels like a **Japanese character support site (キャラクター応援サイト)** while maintaining the readability of a technical blog.

Built from scratch at `d:\blog`. Hosts personal intro, research papers, project showcases, and tech articles.

---

## Design Philosophy: Skeleton vs Skin

```
+-----------------------------------------------+
|  Skeleton ← arthals.ink                       |
|  · Centered max-width · Card layout · Responsive |
|  · Dark/Light mode · RSS/SEO · MDX rendering  |
+-----------------------------------------------+
|  Skin ← Japanese Mio Fan Shrine Aesthetics     |
|  · Mio-blue palette · Sparkle FX · Music notes |
|  · Soft rounded corners · Glow effects         |
|  · Character elements · Japanese iconography   |
+-----------------------------------------------+
```

**Core principle**: Clean, readable layout (arthals.ink), but colors, decorations, and interactions come from fan shrine aesthetics. Curated, not cluttered.

---

## Reference Analysis

### arthals.ink (Skeleton)

| Aspect | Detail |
|--------|--------|
| **Stack** | Astro v5.12 + "Astro Theme Pure" theme |
| **Layout** | Centered single-column max-w-[70rem], sticky header with glass effect, Section: title | content (side-by-side on desktop) |
| **Color System** | HSL CSS variables (shadcn/ui style), highlight `#659EB9` |
| **Cards** | `rounded-2xl border` + hover `bg-muted`, monospace dates, animated arrow on hover |
| **Interactions** | Dark mode 3-state toggle, sticky nav show/hide, mobile hamburger, medium-zoom |

### Japanese Otaku Fan Shrines (Skin)

**Mio Akiyama Character Colors (推しカラー)**:
- Deep navy `#1B3A5C` — her hair color / K-ON! uniform skirt
- Steel blue `#5B9BD5` — her eyes / Light Music Club theme
- Pale blue-white `#D6EAF8` / `#E8F4F8` — uniform shirt / highlights
- Optional accent: Bass sunburst `#C17A3E` (warm contrast)

**Fan Shrine Core Visual Elements**:
1. **Sparkle (キラキラ)**: CSS twinkling star animations around avatar and section titles
2. **Music notes**: ♪♩♫ — Mio is a bassist, notes are her core symbol
3. **Floating petals/notes**: Lightweight canvas CSS particle fall (Hero area only)
4. **Oshi marks**: Fan support markers (uchiwa/penlight concept)
5. **Deco text**: Gradient text, text-shadow glow effects
6. **Japanese emoji flourishes**: ✦ ₊˚✧ ♪ ♩ ♡
7. **Status indicator**: "Currently pushing" online status (adapt arthals.ink's quote widget)
8. **Soft rounded corners**: Softer, larger radius than arthals.ink's `rounded-2xl`

---

## Fusion Design

### 1. Color Palette (replacing arthals.ink HSL tokens)

#### Dark Mode (default/primary — "Mio's Night")

```css
.dark {
  --background: 220 25% 6%;        /* deep blue-black */
  --foreground: 210 40% 98%;       /* blue-white text */
  --card: 220 25% 9%;
  --card-foreground: 210 40% 98%;
  --primary: 207 65% 55%;          /* #5B9BD5 Mio Blue */
  --primary-foreground: 0 0% 98%;
  --secondary: 220 20% 14%;
  --muted: 220 20% 12%;
  --muted-foreground: 215 20% 70%;
  --accent: 207 50% 40%;
  --border: 220 20% 18%;
  --radius: 0.75rem;
  --highlight: 207 65% 55%;
  --sparkle: 200 80% 75%;          /* soft blue glow */
}
```

#### Light Mode ("Mio's Day")

```css
:root {
  --background: 210 40% 98%;
  --foreground: 220 30% 15%;
  --card: 0 0% 100%;
  --card-foreground: 220 30% 15%;
  --primary: 207 60% 40%;
  --primary-foreground: 0 0% 100%;
  --secondary: 210 30% 94%;
  --muted: 210 25% 92%;
  --muted-foreground: 215 15% 45%;
  --accent: 207 45% 45%;
  --border: 210 20% 85%;
  --highlight: 207 60% 42%;
  --sparkle: 200 70% 55%;
}
```

### 2. Typography

| Role | Font | Why |
|------|------|-----|
| **Headings** | M PLUS Rounded 1c (Google Fonts) | Japanese-designed, rounded, friendly |
| **Body** | Inter / Noto Sans | Clean English readability |
| **Code** | JetBrains Mono | Modern monospace |
| **Decorative** | Optional: LXGW WenKai | Chinese/Japanese decorative text |

### 3. Decoration System (curated CSS effects)

#### 3a. Hero Petal/Note Fall (lightweight Canvas)
- Runs only in Hero area (~400px height above avatar)
- Semi-transparent blue petals/music notes, very slow descent
- Uses `requestAnimationFrame`, ~20 particles max
- Subtle glow in dark mode

#### 3b. Sparkle Stars (CSS only)
- Around avatar and section titles
- CSS `@keyframes twinkle`: `scale(0) → scale(1) → scale(0)` + `opacity`
- 4-5 absolutely-positioned `<span>` elements
- Staggered `animation-delay` for organic feel

#### 3c. Music Note Decorations ♪
- Section title prefix notes (emoji/CSS only, no images)
- Note badge next to post dates
- Music symbols in footer

#### 3d. Hover Glow Effects
- Links: `text-shadow: 0 0 8px var(--sparkle)` on hover
- Cards: border-color → primary + soft `box-shadow` glow
- Tags/buttons: slight `translateY(-1px)` + glow

#### 3e. Custom Scrollbar
```css
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: hsl(var(--background)); }
::-webkit-scrollbar-thumb {
  background: hsl(var(--primary) / 0.3);
  border-radius: 4px;
}
```

### 4. Homepage Layout

```
+---------------------------------------------------+
| [Sticky Header] Blog · Projects · About · Links   |
|                 Dark/Light Toggle                  |
+---------------------------------------------------+
|                                                   |
|   ✦ Floating petals/notes area ✦                  |
|   [Round avatar + blue glow ring]                 |
|   Ru00y's Lab                                     |
|   ♪ VLM Researcher & Mio Fan ♪                    |
|   GitHub                                          |
|                                                   |
+---------------- About ----------------------------+
|  [Mio blue accent bar]                            |
|  "Don't say lazy — Akiyama Mio"                   |
|  Bio (EN) + research interests + otaku identity   |
|  [More about me →]                                |
+---------------- Posts ♪ --------------------------+
|  [Post Cards] Date + Title with note icon         |
|  +-------------------------------------------+    |
|  | ♪ xv6 OS Lab Part 8                   →  |    |
|  +-------------------------------------------+    |
+---------------- Publications ✦ ------------------+
|  +-------------------------------------------+    |
|  | ★ DegEval: Benchmarking ... [ECCV 2026]  |    |
|  +-------------------------------------------+    |
+---------------- Projects -------------------------+
|  [Project Cards] 2-column grid                    |
|  Emoji icon + title + description + tags          |
|  hover: border glow + slight lift                 |
+---------------------------------------------------+
+---------------- Skills ♩ -------------------------+
|  [Tag cloud] hover: primary color + glow          |
+---------------- Footer ---------------------------+
|  ♪ After School Tea Time is the best! ♪           |
|  © 2024-2026 Ru00y · KyoAni Shines Forever       |
|  [GitHub] [Email] [RSS]                           |
|  ✦ Thanks for visiting! See you again~ ✦          |
+---------------------------------------------------+
```

### 5. Component Design

#### PostCard
- **Base**: arthals.ink's `rounded-2xl border bg-background px-5 py-2.5`
- **Anime touch**:
  - ♪ emoji before date
  - Hover: border → primary + `box-shadow: 0 0 15px hsl(var(--primary)/0.15)`
  - Arrow animation preserved, in primary color
  - Left accent bar (3px primary) appears on hover only

#### ProjectCard
- **Base**: Rounded card + emoji/icon + title + description + tags
- **Anime touch**:
  - Custom emoji per project (🔥🛡️📝🧠)
  - Hover: `translateY(-3px)` + border glow
  - Tags: `bg-primary/10 text-primary rounded-full`

#### SkillTag
- **Base**: arthals.ink's `bg-muted border rounded-xl px-2 py-1`
- **Anime touch**:
  - Hover: `bg-primary/15 border-primary text-primary`
  - `transition-all duration-200`
  - Optional small decoration prefix

#### Header
- **Base**: arthals.ink's sticky header
- **Anime touch**:
  - Blog name: gradient text (primary → sparkle)
  - Scrolled glass effect: `backdrop-blur` + semi-transparent blue-black bg
  - Nav items: underline animation on hover (primary glow)

#### ThemeToggle
- **Base**: arthals.ink's 3-state toggle
- **Anime touch**:
  - Sun/Moon/System icons with emoji or custom SVG
  - 300ms rotate animation on toggle

### 6. Mio Elements (Copyright-safe)

| Element | Approach | Risk |
|---------|----------|------|
| **Quotes** | "Don't say lazy", "Embarrassing lines are forbidden!" | ✅ Fair use |
| **Color palette** | #1B3A5C / #5B9BD5 — character colors | ✅ Colors not copyrightable |
| **Bass/music** | 🎸 emoji, music notes — Mio is a bassist | ✅ Generic |
| **Song titles** | "Don't say lazy", "Fuwa Fuwa Time" text references | ✅ Not protected |
| **Club concepts** | "After School Tea Time", "Light Music Club" | ✅ Generic |
| **Character art** | User-drawn or AI-generated blue-haired avatar | ✅ Original |
| **Official images** | **DO NOT USE** K-ON! screenshots or official art | ⛔ Copyright violation |

**Recommendation**: Convey "Mio-ness" through colors, emoji, music notes, and quotes. Use an AI-generated + self-drawn blue-haired anime avatar.

---

## Implementation Steps

### Step 1: Project Init
- Run `npm create astro@latest` in `d:\blog`
- Choose: Empty project + TypeScript
- Install UnoCSS (`@unocss/astro`) — what arthals.ink uses
- Manual build (not using Astro Theme Pure — customization needs are too high)
- Init git, link to `https://github.com/rua103/rua103.github.io`

### Step 2: Design System
- Create `src/styles/global.css`: CSS custom properties (dark & light)
- Configure UnoCSS with custom theme
- Import fonts: M PLUS Rounded 1c, Inter, JetBrains Mono
- Create `src/styles/animations.css`: sparkle, glow, float keyframes
- Create `src/styles/components.css`: card, tag, button styles

### Step 3: Layout Components
- `src/layouts/BaseLayout.astro` — HTML shell + SEO meta + global styles
- `src/components/Header.astro` — Sticky nav + glass effect + theme toggle
- `src/components/Footer.astro` — KyoAni tribute + copyright + social links
- `src/components/ThemeToggle.astro` — Dark/Light/System 3-state
- `src/components/SparkleEffect.astro` — CSS-only sparkle stars
- `src/components/PetalCanvas.astro` — Lightweight canvas petal fall (homepage only)

### Step 4: Homepage (`src/pages/index.astro`)
- Hero: avatar + name + tagline + petal canvas
- About section: bio + Mio quote
- Posts section: latest post cards with music notes
- Publications section: paper card
- Projects section: 2-column project card grid
- ~~Education section~~ (removed — user doesn't want school info)
- Skills section: tag cloud
- Footer

### Step 5: Sub-pages
- `/blog` — post list with pagination
- `/blog/[slug]` — post detail (Shiki code highlighting, medium-zoom images)
- `/projects` — project showcase
- `/about` — detailed personal page (more otaku design space)
- `/links` — friends/links (optional)
- `/rss.xml` — RSS feed

### Step 6: Initial Content
- At least 3 initial posts: self-intro, ECCV paper breakdown, one project write-up
- Convert GitHub README projects to blog content
- Configure SEO (title, description, OG image, sitemap)

### Step 7: Deploy
- Create `rua103/rua103.github.io` GitHub repo
- Configure GitHub Actions: push → build → deploy to Pages
- Setup custom domain (if any)
- Verify: `npm run build` → visit `rua103.github.io`

---

## File Tree

```
d:\blog\
├── astro.config.mjs              Astro config + UnoCSS integration
├── uno.config.ts                 UnoCSS custom theme
├── tsconfig.json
├── package.json
├── .github\workflows\deploy.yml  GitHub Actions deploy script
├── public\
│   ├── favicon\
│   │   └── favicon.ico
│   └── images\
│       └── avatar.webp
├── src\
│   ├── styles\
│   │   ├── global.css            CSS variables + design tokens
│   │   ├── animations.css        Sparkle, glow, float keyframes
│   │   └── components.css        Card, tag, button styles
│   ├── layouts\
│   │   └── BaseLayout.astro      HTML shell
│   ├── components\
│   │   ├── Header.astro          Sticky nav
│   │   ├── Footer.astro          KyoAni tribute footer
│   │   ├── ThemeToggle.astro     Dark/light toggle
│   │   ├── PostCard.astro        Blog post card
│   │   ├── ProjectCard.astro     Project card
│   │   ├── SkillTag.astro        Skill tag badge
│   │   ├── SparkleEffect.astro   CSS sparkle stars
│   │   └── PetalCanvas.astro     Falling petals canvas
│   ├── content\
│   │   └── blog\                 Markdown posts
│   │       ├── 01-hello-world.md
│   │       ├── 02-degeval-paper.md
│   │       └── 03-disaster-detection.md
│   └── pages\
│       ├── index.astro           Homepage
│       ├── blog\
│       │   ├── index.astro       Blog list
│       │   └── [slug].astro      Post detail
│       ├── projects.astro        Projects page
│       ├── about.astro           About page
│       └── rss.xml.js            RSS feed
```

---

## Confirmed Decisions

| Decision | Choice | Notes |
|----------|--------|-------|
| **Blog Name** | **Ru00y's Lab** | Otaku lab aesthetic |
| **Domain** | `rua103.github.io` | Default GitHub Pages domain |
| **Comments** | None | Pure static |
| **Anime Level** | Full immersion | Curated Mio fan shrine aesthetic |
| **Language** | English | All UI and content in English |

---

## Verification

1. `npm run dev` → browser at localhost:4321
2. Verify homepage: Hero petal fall, Avatar, About/Posts/Projects/Skills sections
3. Verify dark/light toggle + system follow
4. Verify mobile responsive (hamburger menu + stacked cards)
5. Verify post detail: code highlighting, medium-zoom images
6. Verify animation performance (DevTools FPS > 50)
7. `npm run build` exits cleanly
8. GitHub Pages deploy verified

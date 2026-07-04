# CLAUDE.md

## Preferences

- **No emoji** unless explicitly asked. The author prefers plain text for blog content.
- Blog post tone should be friendly, personal, and conversational — more like talking to a reader than a formal paper.
- Blog tags should be lowercase, single words or short phrases (e.g. `meta`, `blogging`, `vlm`).

## KaTeX formatting rules

The blog uses `remark-math` + `rehype-katex` to render math. Two pitfalls break rendering:

1. **`aligned` blocks: no `\\[2pt]` spacing.** Use bare `\\` for line breaks. The optional spacing parameter `[2pt]` causes the `&` alignment marker on subsequent lines to be parsed as an illegal character (`Expected 'EOF', got '&'`).

2. **`\S` is not supported.** KaTeX has no `\S` command. Use `\text{\textsection}` or just write "sec." instead.

Working `aligned` format:
```
$$
\begin{aligned}
x &= a + b \\
y &= c + d
\end{aligned}
$$
```
Three requirements: `$$` on separate lines from `\begin{aligned}`, `\\` without optional spacing, `&=` for alignment.

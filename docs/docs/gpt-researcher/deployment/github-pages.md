---
sidebar_position: 20
---

# Deploying to GitHub Pages

GitHub Pages can only serve static files, so you **cannot** deploy the new
Next.js frontend there. Instead, use the legacy static UI living in
`frontend/` (the one that powers `index.html`, `styles.css`, etc.).

We ship a helper script that prepares a clean bundle for Pages without
touching the dynamic Next.js app:

```bash
npm run prepare:github-pages
```

The script copies the static assets into `docs/github-pages-static/` (you can
override the target directory by passing a path argument). Point GitHub Pages
to that folder or push its contents to your `gh-pages` branch. No `next export`
step is required or supported.

For full-featured deployments (dynamic routes, streaming chat, API handlers),
deploy `frontend/nextjs` to a platform that runs Node.js (Vercel, Netlify,
Docker, etc.). Those environments should use `npm run build && npm run start`
and **not** attempt static export.

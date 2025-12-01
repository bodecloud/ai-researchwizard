#!/usr/bin/env bash

set -euo pipefail

TARGET_DIR="${1:-docs/github-pages-static}"

echo "Preparing static frontend bundle for GitHub Pages in '${TARGET_DIR}'..."

rm -rf "${TARGET_DIR}"
mkdir -p "${TARGET_DIR}"

cp -R frontend/. "${TARGET_DIR}"
rm -rf "${TARGET_DIR}/nextjs"

cat <<'EOF' > "${TARGET_DIR}/README.md"
# GPT Researcher – Static Frontend Bundle

This directory contains the legacy static frontend that can be hosted on
GitHub Pages or any other static-only hosting platform. It contains **no**
Next.js build artefacts; the interactive SPA lives at `frontend/nextjs`
and must be deployed on a platform that supports Node.js or serverless
execution.

The bundle was generated via `npm run prepare:github-pages`.
EOF

echo "Static frontend bundle ready."

#!/usr/bin/env bash
# Deploy the static site to Vercel and print the live URL.
# One-time setup: run `vercel login` first (try: `! vercel login` in the Claude prompt).
set -euo pipefail

cd "$(dirname "$0")/site"
vercel --prod

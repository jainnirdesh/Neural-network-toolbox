#!/usr/bin/env bash

set -euo pipefail

if command -v npm >/dev/null 2>&1; then
  npm install
  npm run build
else
  echo "npm is not available; skipping frontend build."
fi
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PORT_CONFIG="${MEETING_AI_PORT_CONFIG:-../config/ports.env}"
if [ -f "$PORT_CONFIG" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$PORT_CONFIG"
  set +a
fi

npm install
npm run dev -- --host "${FRONTEND_HOST:-0.0.0.0}" --port "${PORT:-${FRONTEND_PORT:-5173}}"

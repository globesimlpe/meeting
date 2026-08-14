#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate meeting-ai

PORT_CONFIG="${MEETING_AI_PORT_CONFIG:-../config/ports.env}"
if [ -f "$PORT_CONFIG" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$PORT_CONFIG"
  set +a
fi

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

pip install -r requirements.txt
uvicorn app.main:app --host "${BACKEND_HOST:-0.0.0.0}" --port "${PORT:-${BACKEND_PORT:-8001}}"

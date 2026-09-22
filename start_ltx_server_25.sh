#!/usr/bin/env bash
# Launch the single-path LTX-2.5 text-to-video inference server.
#
# Usage:
#   ./start_ltx_server_25.sh
#
# Defaults: 127.0.0.1:8002, no API token (loopback). Set LTX_HOST=0.0.0.0 and
# LTX_API_TOKEN for LAN access. Runs one video per job on a single XPU via
# run_t2v_25_xpu.py (the 2.3 server stays on 8001).
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# --- defaults ---
export LTX_HOST="${LTX_HOST:-127.0.0.1}"
export LTX_PORT="${LTX_PORT:-8002}"
export LTX_API_TOKEN="${LTX_API_TOKEN:-}"
export LTX_MULTI_MODE=1
export LTX_OUTPUT_DIR="${LTX_OUTPUT_DIR:-outputs/ltx-server-25}"
export LTX_DB="${LTX_DB:-outputs/ltx-server-25/jobs.sqlite3}"

# --- env ---
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "Starting LTX-2.5 Video Server at ${LTX_HOST}:${LTX_PORT}"
echo "  API token: ${LTX_API_TOKEN:-<none (loopback only)>}"
echo "  Output dir: ${LTX_OUTPUT_DIR}"

exec "$HERE/.venv/bin/python" "$HERE/ltx_server_25.py"

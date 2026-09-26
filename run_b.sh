#!/usr/bin/env bash
# Launch LTX-2.3 text-to-video Run B: 1024x1024, 121 frames ~5.0s clip.
# Usage:
#   ./run_b.sh                      # default (1024x1024, 121 frames)
#   ./run_b.sh "your prompt here"
#   LTX_PROFILE=b70 ./run_b.sh      # single-B70 layout (all roles on xpu:0)
#
# Output: output/output_1024.mp4.
# Devices default from LTX_PROFILE (b60dual: transformer xpu:1, VAE/Gemma xpu:0).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

# --- python (project venv; override with LTX_PYTHON) ---
PYTHON="${LTX_PYTHON:-$HERE/.venv/bin/python}"
[ -x "$PYTHON" ] || { echo "missing $PYTHON; run 'bash setup_env.sh' or set LTX_PYTHON" >&2; exit 1; }

export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LTX_PROFILE="${LTX_PROFILE:-b60dual}"
export LTX_WIDTH="${LTX_WIDTH:-1024}"
export LTX_HEIGHT="${LTX_HEIGHT:-1024}"
export LTX_FRAMES="${LTX_FRAMES:-121}"

if [[ $# -ge 1 ]]; then
    export LTX_PROMPT="$1"
fi

exec "$PYTHON" -u "$HERE/run_t2v_xpu_perf.py"

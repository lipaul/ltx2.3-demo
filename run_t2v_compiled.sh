#!/usr/bin/env bash
# Single clip with torch.compile on the transformer blocks (LTX_COMPILE=1).
#
# torch.compile on this dual-GPU host needs an Intel toolchain that matches the
# SYCL runtime torch bundles (libsycl.so.8 / libur_loader 0.12).  The login
# shell's oneAPI 2026.1 setvars environment provides libsycl.so.9, which makes
# triton JIT against the wrong runtime ("Backends mismatch" / undefined
# urDeviceWaitExp).  This wrapper runs the harness with a minimal environment
# and an 2025.x icpx that ships the matching libsycl.so.8.
#
# First run compiles the blocks (~9 s extra); later runs reuse the cache in
# .torch_cache/ (shapes are fixed).  Override the usual LTX_* generation vars.
#
# Layout: torch.compile is only viable on the single-card / oneAPI-host path, so
# this wrapper defaults to LTX_PROFILE=b70 (all roles on xpu:0). Override
# LTX_PROFILE / LTX_TDEV / LTX_CDEV / LTX_GEMMA_DEVICE to change it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"
[ -x "$VENV/bin/python" ] || { echo "missing $VENV/bin/python; run bash setup_env.sh" >&2; exit 1; }

# icpx whose root provides libsycl.so.8 (the one torch links against).
ONEAPI_ROOT="${ONEAPI_ROOT:-/opt/intel/oneapi}"
ICX_BIN=""
for d in "$ONEAPI_ROOT"/compiler/*/bin; do
    [ -x "$d/icpx" ] || continue
    root="$(dirname "$d")"
    [ -e "$root/lib/libsycl.so.8" ] && ICX_BIN="$d"
done
[ -n "$ICX_BIN" ] || { echo "no oneAPI compiler with libsycl.so.8 found under $ONEAPI_ROOT/compiler (set ONEAPI_ROOT)" >&2; exit 1; }

CACHE="${LTX_TORCH_CACHE_DIR:-$HERE/.torch_cache}"
mkdir -p "$CACHE/inductor" "$CACHE/triton"

args=(
    env -i HOME="$HOME" PATH="$ICX_BIN:/usr/bin:/bin" LD_LIBRARY_PATH="$VENV/lib"
    TRITON_DEFAULT_BACKEND=intel LTX_COMPILE=1
    TORCHINDUCTOR_CACHE_DIR="$CACHE/inductor" TRITON_CACHE_DIR="$CACHE/triton"
    HF_HUB_OFFLINE=1
    LTX_PROFILE="${LTX_PROFILE:-b70}"
    LTX_TDEV="${LTX_TDEV:-0}" LTX_CDEV="${LTX_CDEV:-0}"
    LTX_GEMMA_DEVICE="${LTX_GEMMA_DEVICE:-xpu:0}" LTX_GEMMA_OFFLOAD="${LTX_GEMMA_OFFLOAD:-cpu}"
    LTX_WIDTH="${LTX_WIDTH:-1024}" LTX_HEIGHT="${LTX_HEIGHT:-1024}" LTX_FRAMES="${LTX_FRAMES:-121}"
    LTX_OUTPUT_PATH="${LTX_OUTPUT_PATH:-$HERE/output/output_1024.mp4}"
)
[ -n "${LTX_PROMPT:-}" ] && args+=(LTX_PROMPT="$LTX_PROMPT")

echo "torch.compile run: icpx=$ICX_BIN cache=$CACHE"
exec "${args[@]}" "$VENV/bin/python" -u "$HERE/run_t2v_xpu_perf.py"

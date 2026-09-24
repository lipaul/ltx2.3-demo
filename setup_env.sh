#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")"

echo "--- [1/4] Checking out LTX-2 ---"
# Pinned revision: bump deliberately, after re-verifying the harness against the
# new code (upstream APIs drift between releases). Override with LTX_REV=...
LTX_REPO="${LTX_REPO:-https://github.com/Lightricks/LTX-2.git}"
LTX_REV="${LTX_REV:-a95ab856bf29407b6b066ede0abe1846050db56c}"
if [ ! -d LTX-2 ]; then
    git clone --depth 1 "$LTX_REPO" LTX-2
fi
cd LTX-2
# Fetch the pinned commit if the shallow clone does not have it.
if ! git cat-file -e "${LTX_REV}^{commit}" 2>/dev/null; then
    git fetch --depth 1 origin "$LTX_REV"
fi
# -f discards the in-place XPU patches from a previous run; step [2/4] re-applies them.
git checkout --detach -f "$LTX_REV"
echo "  LTX-2 @ $(git rev-parse HEAD)"
cd ..

echo "--- [2/4] Applying XPU patches ---"
cd LTX-2
# vocoder — XPU fp32 fallback
sed -i 's/if device_type == "mps"/if device_type in ("mps", "xpu")/' \
  packages/ltx-core/src/ltx_core/model/audio_vae/vocoder.py
# devices — XPU sync + empty_cache
python3 -c "
fp = 'packages/ltx-core/src/ltx_core/devices.py'
with open(fp) as f: s = f.read()
for old, new in [
    ('torch.mps.synchronize()',
     'torch.mps.synchronize()\\n    elif resolved.type == \"xpu\" and hasattr(torch, \"xpu\") and torch.xpu.is_available():\\n        torch.xpu.synchronize(resolved)'),
    ('torch.mps.empty_cache()',
     'torch.mps.empty_cache()\\n    elif resolved.type == \"xpu\" and hasattr(torch, \"xpu\") and torch.xpu.is_available():\\n        torch.xpu.empty_cache()'),
]:
    s = s.replace(old, new)
with open(fp, 'w') as f: f.write(s)
"
# fp8 prequant fold — tolerate the ``_orig_mod`` prefix torch.compile inserts
# (transformer_blocks.N._orig_mod.) so LTX_COMPILE=1 can fold the *_scale keys.
python3 - <<'PY'
fp = 'packages/ltx-core/src/ltx_core/quantization/fp8_cast.py'
with open(fp) as f:
    s = f.read()
old_on = '''    def _on_param(param_key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
        scale = scales.get(param_key)'''
new_on = '''    def _lookup_scale(param_key: str) -> torch.Tensor | None:
        # torch.compile moves block params under transformer_blocks.N._orig_mod.
        # (modify_sd_ops_for_compilation); the scales dict is keyed without it.
        if param_key in scales:
            return scales[param_key]
        return scales.get(param_key.replace("._orig_mod.", "."))

    def _on_param(param_key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
        scale = _lookup_scale(param_key)'''
old_drop = '''        param_key = scale_key.removesuffix("_scale")
        if param_key not in scales:'''
new_drop = '''        param_key = scale_key.removesuffix("_scale")
        if _lookup_scale(param_key) is None:'''
if old_on in s and old_drop in s:
    s = s.replace(old_on, new_on).replace(old_drop, new_drop)
    with open(fp, 'w') as f:
        f.write(s)
    print('  patched fp8_cast.py (compile-aware prequant scales)')
elif '_lookup_scale' in s:
    print('  fp8_cast.py already compile-aware')
else:
    raise SystemExit('fp8_cast.py patch anchors not found')
PY
# R1-R15 kernel-optimization port: fp8 widen hook + R8/R9A/R10D SYCL kernel sources.
# Applied on the post-XPU-patch baseline; binaries are built in step [6/4].
git apply ../patches/r_kernels.patch
echo "  applied patches/r_kernels.patch"
echo "  done"
cd ..

echo "--- [3/4] uv sync ---"
uv sync

echo "--- [4/4] Fixing generation scripts ---"
# Model paths -> ./models/
python3 -c "
import re
for f in ['run_t2v_xpu_perf.py', 'encode_prompts.py']:
    with open(f) as fh: src = fh.read()
    src = re.sub(r'\"([^\"]+?)ltx-2\.3-22b-distilled-fp8\.safetensors\"',
        r'str(Path(__file__).resolve().parent / \"models\" / \"ltx-2.3-22b-distilled-fp8.safetensors\")', src)
    src = re.sub(r'\"([^\"]+?)ltx-2\.3-spatial-upscaler-x2-1\.1\.safetensors\"',
        r'str(Path(__file__).resolve().parent / \"models\" / \"ltx-2.3-spatial-upscaler-x2-1.1.safetensors\")', src)
    src = re.sub(r'\"([^\"]+?)gemma-3-12b-it\"',
        r'str(Path(__file__).resolve().parent / \"models\" / \"gemma-3-12b-it\")', src)
    if 'from pathlib import Path' not in src:
        src = src.replace('import time', 'import time\\nfrom pathlib import Path')
    with open(f, 'w') as fh: fh.write(src)
    print('  patched', f)
"
sed -i 's/DiffusionStage(/DiffusionStage.from_checkpoint(/g' run_t2v_xpu_perf.py
# Add Path import if missing
grep -q "from pathlib import Path" run_t2v_xpu_perf.py || \
  sed -i 's/^import time/import time\nfrom pathlib import Path/' run_t2v_xpu_perf.py

echo "--- [5/4] Installing LTX-2 packages ---"
uv pip install --no-deps -e LTX-2/packages/ltx-core
uv pip install --no-deps -e LTX-2/packages/ltx-pipelines

echo "--- [6/4] Building R8/R9A/R10D SYCL kernels ---"
if [ "${LTX_SKIP_KERNELS:-0}" = "1" ]; then
    echo "  skipped (LTX_SKIP_KERNELS=1)"
else
    make -C LTX-2/packages/ltx-kernels/csrc/sycl_r8 -j"$(nproc)" >/tmp/ltx_sycl_build.log 2>&1 \
        && echo "  built libr8_sycl.so" \
        || echo "  WARNING: SYCL build failed (see /tmp/ltx_sycl_build.log); R8/R9A fall back to eager"
fi

echo ""
echo "Done. Next:"
echo "  uv run download.py                       # fetch LTX-2.3 weights into models/"
echo "  uv run python run_t2v_xpu_perf.py        # single test"

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
sed -i 's|OUTPUT_PATH = os.environ.get("LTX_OUTPUT_PATH", "/home/lm/paul/ltx23-run/output_1024.mp4")|OUTPUT_PATH = os.environ.get("LTX_OUTPUT_PATH", str(Path(__file__).resolve().parent / "output" / "output_1024.mp4"))|' run_t2v_xpu_perf.py
# Add Path import if missing
grep -q "from pathlib import Path" run_t2v_xpu_perf.py || \
  sed -i 's/^import time/import time\nfrom pathlib import Path/' run_t2v_xpu_perf.py

echo "--- [5/4] Installing LTX-2 packages ---"
uv pip install --no-deps -e LTX-2/packages/ltx-core
uv pip install --no-deps -e LTX-2/packages/ltx-pipelines

echo ""
echo "Done. Next:"
echo "  ln -sf /home/lm/paul/ltx23-models/* /home/lm/videogen/models/    # symlink existing models"
echo "  uv run python run_t2v_xpu_perf.py        # single test"

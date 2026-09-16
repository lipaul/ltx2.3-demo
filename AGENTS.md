# AGENTS.md

LTX-2.3 text-to-video demo on Intel Arc Pro B60 XPUs (32 devices, one node).
This repo is a thin harness; the model code lives in the gitignored `LTX-2/`
clone (installed editable from `LTX-2/packages/{ltx-core,ltx-pipelines}`).

## Environment

- Python 3.12 only. Use the project venv: `.venv/bin/python` (torch
  `2.12.1+xpu`, 32 XPU devices). `uv` is at `~/.local/bin/uv`.
- Setup: `bash setup_env.sh` (clones `LTX-2`, applies XPU patches, `uv sync`,
  editable-installs the two LTX packages with `--no-deps`). Re-run it if
  `LTX-2/` is re-cloned; the patches are not committed upstream.
- `setup_env.sh` pins `LTX-2` to a fixed commit (`LTX_REV`, currently
  `a95ab85`) and force-checks it out (`--detach -f`) every run, so upstream
  changes are never pulled implicitly. Bump `LTX_REV` deliberately and
  re-verify the harness; the pin is what keeps the editable install reproducible.
- Upstream APIs drift between LTX-2 revisions and the harness is written against
  a specific one (`setup_env.sh` path-patches script bodies, not call signatures).
  Known examples: `PromptEncoder` takes `model_paths=ModelPaths.from_monolith(ckpt,
  gemma_root)`, not `checkpoint_path`/`gemma_root`; the decode tiling default is
  `TileSizeConfig.default()` (`TilingConfig` is only a `TileSizeConfig | TileCountConfig`
  alias and has no `.default()`).
- `setup_env.sh` patches `LTX-2/.../devices.py` and `.../audio_vae/vocoder.py`
  in place. `patches/xpu.patch` is a reference diff (larger than what the
  script currently applies). Check `git -C LTX-2 status` before assuming.
- Model weights live in gitignored `models/` (~30 GB); fetch with
  `uv run download.py`. Scripts set `HF_HUB_OFFLINE=1`.
- **Trap:** `run.sh`, `run_b.sh`, `run_multi.sh`, `start_ltx_server.sh` all
  exec a hardcoded `/home/lm/paul/ltx23-env/bin/python` that does not exist.
  Ignore their interpreter lines; use `.venv/bin/python` (or the README.txt
  commands).

## Commands

- README.txt is the accurate quickstart:
  `bash setup_env.sh` -> `uv run download.py` -> `uv run python ltx_server.py`.
- Single clip: `.venv/bin/python run_t2v_xpu_perf.py` (env-configurable, see
  below). There is no `run_t2v_xpu.py` despite docstrings referencing it.
- Multi-clip (N videos): `.venv/bin/python run_multi_xpu.py --prompts-file
  prompts.json --job-dir OUT` (8 workers) or `run_multi_16.py` (16 workers).
  These pre-encode all prompts once via `encode_prompts.py`, then spawn
  subprocesses that load embeddings.
- Server: `.venv/bin/python ltx_server.py`; FastAPI on `:8001`, bearer token
  required when `LTX_HOST` is non-loopback. API in `ltx_server_common.py`
  (`POST/GET /api/multi-jobs`, SSE `/api/events`).
- Benchmarks hit a running server: `multi_benchmark.py` (10x16 videos).

## Architecture

- Device layout per worker `i`: transformer on `xpu:2i`, VAE/decoders on
  `xpu:2i+1`; Gemma-3-12B text encoder on CPU (bf16 ~24 GB does not fit a
  24 GB B60). Max 16 workers on 32 XPUs. `LTX_MULTI_MODE` (8 or 16) sets the
  server/`ltx_server.py` worker count.
- Generation is two-stage (stage 1 low-res denoise, 2x spatial upsample,
  stage 2 refine). Target resolution must be divisible by 64; stage 1 is half.
- The server spawns subprocesses per job (via `run_t2v_xpu_perf.py`) instead
  of loading models in-process, to avoid OOM from model lifecycle buildup.
- Multi-worker spawns are staggered (`LTX_SPAWN_DELAY`, seconds) to avoid XPU
  driver races during concurrent model init.

## Env vars

- Single run: `LTX_PROMPT`, `LTX_WIDTH`/`LTX_HEIGHT`, `LTX_FRAMES`,
  `LTX_TDEV`/`LTX_CDEV`, `LTX_OUTPUT_PATH`, `LTX_EMBEDDINGS_PATH`.
  Defaults: 1024x1024, 121 frames @ 24 fps.
- `LTX_FRAMES` must be `8k+1`; 73 hangs the XPU driver (see
  `run_t2v_xpu_perf.py:61`).
- Prompts for encode: `LTX_PROMPTS_FILE` (JSON array) or stdin JSON.
- Server: `LTX_HOST`, `LTX_PORT`, `LTX_API_TOKEN`, `LTX_QUEUE_SIZE`,
  `LTX_OUTPUT_DIR`, `LTX_DB`.

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
- `setup_env.sh` patches `LTX-2/.../devices.py`, `.../audio_vae/vocoder.py`, and
  `.../quantization/fp8_cast.py` (compile-aware prequant scales) in place.
  `patches/xpu.patch` is a reference diff (larger than what the script currently
  applies). Check `git -C LTX-2 status` before assuming.
- Model weights live in gitignored `models/` (~30 GB); fetch with
  `uv run download.py`. Scripts set `HF_HUB_OFFLINE=1`.
- **Trap:** `run.sh`, `run_b.sh`, `run_multi.sh`, `start_ltx_server.sh` exec a
  hardcoded `/home/lm/paul/ltx23-env/bin/python` that does not exist. Ignore
  their interpreter lines; use `.venv/bin/python` (or the README.txt commands).
  `run_multi_xpu.py`/`run_multi_16.py` default to the same dead path but honor
  `LTX_PYTHON`; the server's own subprocesses use `<repo>/.venv/bin/python`.
- No test, lint, typecheck, or CI config exists in-repo. Verify changes with a
  real generation run; the fastest sanity check is a single clip
  (`run_t2v_xpu_perf.py`) and inspecting `git -C LTX-2 status` for the patches.

## Commands

- README.txt is the accurate quickstart:
  `bash setup_env.sh` -> `uv run download.py` -> `uv run python ltx_server.py`.
- Single clip: `.venv/bin/python run_t2v_xpu_perf.py` (env-configurable, see
  below). There is no `run_t2v_xpu.py` despite docstrings referencing it.
- Single clip with `torch.compile`: `bash run_t2v_compiled.sh` (wraps the same
  script with `LTX_COMPILE=1` and the minimal oneAPI/triton environment the
  dual-GPU host needs; caches JIT artifacts in `.torch_cache/`).
- Profiling: `torch.profiler` with CPU+XPU activities gives the per-kernel XPU
  breakdown, and `bench_xpu_bw.py` measures HBM/H2D bandwidth. `intel_gpu_top`
  cannot read the B70 (i915-only PMU; the B70 uses `xe`) and `xpu-smi` does not
  enumerate it on this host.
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
  `xpu:2i+1`. Max 16 workers on 32 XPUs. `LTX_MULTI_MODE` (8 or 16) sets the
  server/`ltx_server.py` worker count.
- On a host where torch exposes only one XPU (e.g. a single B70 box:
  `torch.xpu.device_count()==1` and the iGPU is not exposed), `LTX_CDEV` must
  equal `LTX_TDEV` (both `0`) — the default `LTX_CDEV=1` fails. That single-B70
  clip runs in ~58.6 s by default (overlap + non-memory-efficient decode) and
  ~55.8 s with `LTX_COMPILE=1` (warm compile cache); peak 18.16 GB. See README.txt.
- Text encoding is a shared pre-generation step (`encode_prompts.py`): Gemma-3-12B
  bf16 (~23 GB) does not fit a 24 GB B60, so it runs block-streamed on a spare
  XPU by default (`LTX_GEMMA_DEVICE=xpu:0`, `LTX_GEMMA_OFFLOAD=cpu`; ~2 blocks
  resident, weights pinned in RAM). Encoding precedes generation, so all XPUs
  are free. Set `LTX_GEMMA_DEVICE=cpu` to fall back to the old CPU path. The
  legacy per-prompt mode is `LTX_ENCODE_MODE=loop`; the default `batch` encodes
  the whole prompt list in one forward. `encode_prompts.py` must stay under
  `torch.no_grad()` — otherwise Gemma's 48-layer autograd graph balloons memory.
- Generation is two-stage (stage 1 low-res denoise, 2x spatial upsample,
  stage 2 refine). Target resolution must be divisible by 64; stage 1 is half.
- `run_t2v_xpu_perf.py` builds the transformer once and keeps it resident across
  both stages instead of letting `DiffusionStage.__call__` rebuild it for stage 2
  (~5 s saved per video). Safe because both stages run on the transformer device
  while the upsampler/decoders use the VAE device. Disable with
  `LTX_KEEP_TRANSFORMER=0`.
- The server spawns subprocesses per job (via `run_t2v_xpu_perf.py`) instead
  of loading models in-process, to avoid OOM from model lifecycle buildup. It
  processes one job at a time on a single background thread (queue hardcoded to
  4); `multi_mode` is videos per job, not concurrent jobs.
- Multi-worker spawns are staggered to avoid XPU driver races during model init.
  Only `run_multi_xpu.py`/`run_multi_16.py` read `LTX_SPAWN_DELAY` (default 1 s);
  the server uses a hardcoded 1 s stagger instead and ignores `LTX_SPAWN_DELAY`.
- Bearer auth (when `LTX_API_TOKEN` is set) guards the job endpoints only; the
  `/api/multi-jobs/{id}/videos/{i}` endpoint is unauthenticated.
- Server default `LTX_GEMMA_DEVICE` is `xpu:0` (vs `cpu` in the single/multi
  scripts), and `LTX_ENCODER_SERVICE=1` only engages when it is an `xpu:*` spec.

## Env vars

- Single run: `LTX_PROMPT`, `LTX_WIDTH`/`LTX_HEIGHT`, `LTX_FRAMES`,
  `LTX_TDEV`/`LTX_CDEV`, `LTX_OUTPUT_PATH`, `LTX_EMBEDDINGS_PATH`.
  Defaults: 1024x1024, 121 frames @ 24 fps.
- Single-clip speed knobs: `LTX_PREBUILD_TRANSFORMER` (default 1; build the
  transformer while the prompt encodes), `LTX_DECODER_MEM_EFFICIENT` (default 0;
  the plain conv decode is ~1.8x faster on XPU), and `LTX_COMPILE` (default 0;
  torch.compile the transformer — launch via `run_t2v_compiled.sh`, which also
  sets the clean env compile needs).
- `LTX_FRAMES` must be `8k+1`; 73 hangs the XPU driver (see
  `run_t2v_xpu_perf.py:63`).
- Prompts for encode: `LTX_PROMPTS_FILE` (JSON array) or stdin JSON.
- Text encoder: `LTX_GEMMA_DEVICE`, `LTX_GEMMA_OFFLOAD`, `LTX_GEMMA_FP8`,
  `LTX_GEMMA_RESIDENT`, `LTX_ENCODE_MODE`. The server's persistent encoder
  service is `LTX_ENCODER_SERVICE=1` (+ `LTX_ENCODER_FP8`, `LTX_ENCODER_SOCK`);
  `encode_service.py` keeps the Gemma pinned source warm across jobs (~16.6 s
  warm encode for 16 prompts vs a ~44.7 s per-job subprocess encode phase).
- Server: `LTX_HOST` (default `127.0.0.1`), `LTX_PORT` (8001), `LTX_API_TOKEN`
  (required when non-loopback), `LTX_MULTI_MODE` (default 8), `LTX_OUTPUT_DIR`,
  `LTX_DB`. `LTX_QUEUE_SIZE` is documented but not wired to the worker queue.
- Benchmark: `multi_benchmark.py` reads `LTX_BENCH_URL`/`LTX_BENCH_TOKEN`
  (defaults `http://127.0.0.1:8001`, token `111`) and `LTX_BENCH_OUTPUT`.

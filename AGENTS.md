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
- `setup_env.sh` patches `LTX-2/` in place (steps are numbered `[1/4]`..`[6/4]`
  because later edits were appended). It applies `patches/r_kernels.patch` (the
  R1-R15 kernel port) plus `sed` patches to `.../devices.py` (XPU sync/empty_cache),
  `.../audio_vae/vocoder.py` (XPU fp32 fallback) and `.../quantization/fp8_cast.py`
  (compile-aware prequant scales). `patches/xpu.patch` is a reference diff of the
  older XPU fixes, not exactly what the script applies. Verify with
  `git -C LTX-2 status` (r_kernels touches `transformer/{model,ops,rope,transformer}.py`
  and adds `r7_fused.py`, `r8_sycl.py`, `ltx-kernels/csrc/sycl_r8/`).
- Model weights live in gitignored `models/` (~30 GB); fetch with
  `uv run download.py`. Scripts set `HF_HUB_OFFLINE=1`.
- The shell launchers (`run.sh`, `run_b.sh`, `run_multi.sh`,
  `start_ltx_server.sh`) and `run_multi_xpu.py`/`run_multi_16.py` resolve the
  interpreter to `<repo>/.venv/bin/python`, overridable with `LTX_PYTHON`; the
  server's own subprocesses do the same.
- No test, lint, typecheck, or CI config exists in-repo. Verify changes with a
  real generation run; the fastest sanity check is a single clip
  (`run_t2v_xpu_perf.py`) and inspecting `git -C LTX-2 status` for the patches.

## Commands

- README.txt is the accurate quickstart:
  `bash setup_env.sh` -> `uv run download.py` -> `uv run python ltx_server.py`.
- Always run with the project venv: `.venv/bin/python <script>.py` (prepend
  `HF_HUB_OFFLINE=1` to skip hub lookups). Plain `python`/`python3` is 3.10 and
  cannot import these packages.
- Single clip: `.venv/bin/python run_t2v_xpu_perf.py` (env-configurable, see
  below). There is no `run_t2v_xpu.py` despite docstrings referencing it.
  `LTX_PROFILE` defaults to `b60dual`; use `LTX_PROFILE=b70` on a single-B70 box.
- Single clip with `torch.compile`: `bash run_t2v_compiled.sh` (wraps the same
  script with `LTX_COMPILE=1` and the minimal oneAPI/triton environment the
  dual-GPU host needs; caches JIT artifacts in `.torch_cache/`).
- Profiling: `torch.profiler` with CPU+XPU activities gives the per-kernel XPU
  breakdown, and `bench_xpu_bw.py` measures HBM/H2D bandwidth. `intel_gpu_top`
  cannot read the B70 (i915-only PMU; the B70 uses `xe`) and `xpu-smi` does not
  enumerate it on this host.
- LTX-2.5 (opt-in): `.venv/bin/python run_t2v_25_xpu.py` (single XPU). The 2.5
  bf16 transformer (42 GB) is fp8-cast at load (~21 GB) because the official
  comfy-int8-convrot / nvfp4 variants need CUDA (`ltx_kernels`) kernels; Gemma-4
  is streamed. 2.3 (`run_t2v_xpu_perf.py`) stays the default.
- LTX-2.5 server: `bash start_ltx_server_25.sh` (127.0.0.1:8002) ->
  `ltx_server_25.py`, same design as `ltx_server.py` but single-path via
  `run_t2v_25_xpu.py` (`ModelProfile(pre_encode=False, device_pairs=False,
  multi_mode=1, retries=1)`). `retries` re-runs a job whose worker died with a
  signal (intermittent XPU segfault during Gemma-4 load).
- Multi-clip (N videos): `.venv/bin/python run_multi_xpu.py --prompts-file
  prompts.json --job-dir OUT` (8 workers) or `run_multi_16.py` (16 workers).
  These pre-encode all prompts once via `encode_prompts.py`, then spawn
  subprocesses that load embeddings.
- Server: `.venv/bin/python ltx_server.py`; FastAPI on `:8001`, bearer token
  required when `LTX_HOST` is non-loopback. API in `ltx_server_common.py`
  (`POST/GET /api/multi-jobs`, SSE `/api/events`).
- Benchmarks hit a running server: `multi_benchmark.py` (10x16 videos).

## Architecture

- Device layout is selected by `LTX_PROFILE` (explicit `LTX_TDEV`/`LTX_CDEV`/
  `LTX_GEMMA_DEVICE` still override). `b60dual` (default on a 32x B60 node):
  transformer on `xpu:1`, VAE/decoders + streamed Gemma on `xpu:0`. `b70` (single
  B70, 30 GiB): everything shares `xpu:0` (the stock `LTX_CDEV=1` does not exist
  there). Do not put transformer + VAE + Gemma all on one 24 GB B60 — that OOMs
  while Gemma streams; that is what the profile split exists to avoid.
- Per-worker pairing `xpu:(2i, 2i+1)` is only for `b60dual` (max 16 workers on
  32 XPUs); on `b70` the multi-runner and the server collapse every worker to
  `xpu:(0,0)`. `LTX_MULTI_MODE` (8 or 16) sets the server/`ltx_server.py` worker
  count — use `1` on B70.
- Text encoding is a shared pre-generation step (`encode_prompts.py`): Gemma-3-12B
  bf16 (~23 GB) does not fit a 24 GB B60, so it runs block-streamed on an XPU
  (`LTX_GEMMA_DEVICE=xpu:0` by profile, `LTX_GEMMA_OFFLOAD=cpu`; ~2 blocks
  resident, weights pinned in RAM). Encoding precedes generation, so Gemma and
  the VAE can share an XPU without overlapping. Set `LTX_GEMMA_DEVICE=cpu` to
  fall back to the old CPU path. The legacy per-prompt mode is
  `LTX_ENCODE_MODE=loop`; the default `batch` encodes the whole prompt list in one
  forward. `encode_prompts.py` must stay under `torch.no_grad()` — otherwise
  Gemma's 48-layer autograd graph balloons memory.
- Generation is two-stage (stage 1 low-res denoise, 2x spatial upsample,
  stage 2 refine). Target resolution must be divisible by 64; stage 1 is half.
- `run_t2v_xpu_perf.py` builds the transformer once and keeps it resident across
  both stages instead of letting `DiffusionStage.__call__` rebuild it for stage 2
  (~5 s saved per video). Safe because both stages run on the transformer device
  while the upsampler/decoders use the VAE device. Disable with
  `LTX_KEEP_TRANSFORMER=0`.
- Single-clip runner is `run_t2v_xpu_perf.py`; `run.sh`/`run_b.sh` are thin wrappers
  around it (both 1024x1024/121; `run.sh` has a stale "73 frames" comment — the
  default is 121). `LTX_PREBUILD_TRANSFORMER=1` (default on 2.3) builds the fp8
  transformer in a background thread while Gemma encodes, hiding the ~4 s load;
  on 2.5 it defaults off (concurrent Gemma-4 streaming + build intermittently
  raises `UR_RESULT_ERROR_DEVICE_LOST`).
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

- Host layout: `LTX_PROFILE` (`b60dual` default / `b70`) picks the device tuple;
  `LTX_TDEV` / `LTX_CDEV` / `LTX_GEMMA_DEVICE` override it individually. Unknown
  profiles fail fast.
- Single run: `LTX_PROMPT`, `LTX_WIDTH`/`LTX_HEIGHT`, `LTX_FRAMES`,
  `LTX_TDEV`/`LTX_CDEV`, `LTX_OUTPUT_PATH`, `LTX_EMBEDDINGS_PATH`.
  Defaults: 1024x1024, 121 frames @ 24 fps.
- Single-clip speed knobs: `LTX_PREBUILD_TRANSFORMER` (default 1; build the
  transformer while the prompt encodes), `LTX_DECODER_MEM_EFFICIENT` (default 0;
  the plain conv decode is ~1.8x faster on XPU), and `LTX_COMPILE` (default 0;
  torch.compile the transformer — launch via `run_t2v_compiled.sh`, which also
  sets the clean env compile needs). Compile tuning: `LTX_COMPILE_MODE`,
  `LTX_INDUCTOR_CONFIG`/`LTX_DYNAMO_CONFIG` (JSON), `LTX_SEQ_DYNAMIC`,
  `LTX_FULLGRAPH`. Measured dead ends: compiling the VAE decoder is slower than
  eager oneDNN, and `seq_dim_dynamic=False` is net worse (see README.txt).
- LTX-2.3 kernel-opt port flags (from the delivered R1-R15 project; README.txt):
  `LTX_R2_NOTILE` (VAE decode without tiling, default 1), `LTX_R9A_FP8K` (SYCL
  fp8->bf16 widen, default 1), and the elementwise bundle `LTX_R3_FUSE`,
  `LTX_R5_FUSE`, `LTX_R6_FUSE`, `LTX_R8_SYCL`, `LTX_R9D_K3V`, `LTX_R10D_GATE`,
  `LTX_R10D_GATE2` (default 1 on 2.3). The env helpers live in the patched
  `LTX-2` files (`_env_on(name, default="1")`); some experiment sub-flags are
  default-off (`LTX_R7_FUSE`, `LTX_R10D_GELU`, `LTX_R10D_K1HOLD`, `LTX_R9A_V2`,
  `LTX_R9A_WG`). On 2.5 only `LTX_R5_FUSE` + `LTX_R9A_FP8K` + `LTX_R2_NOTILE`
  stay on (bitwise-identical; `run_t2v_25_xpu.py` `setdefault`s the other six to
  `0`); their ~1 ULP per-op change is amplified across the 11 diffusion steps and
  alters the sample. `LTX_ALL_ORIG=1` forces the pre-port path. The SYCL library
  is built by `setup_env.sh` via `make -C LTX-2/packages/ltx-kernels/csrc/sycl_r8`
  (needs a oneAPI compiler shipping `libsycl.so.8`; missing `.so` falls back to
  eager and `LTX_SKIP_KERNELS=1` skips the build). `torch.compile` cannot trace the
  libr8 SYCL ops, so `LTX_COMPILE=1` auto-disables the SYCL flags; with the R port
  the fast path is **eager** (`run_t2v_xpu_perf.py`), not `run_t2v_compiled.sh`.
  Per-flag A/B numbers are rolled up in README.txt (R2, R9A, R3/R5/R6/R8/R9D/R10D);
  a flag is only kept on when an on/off A/B clears the noise gate.
- Experimental quantized attention: `LTX_ATTN_PATTERN=factorized`
  (`LTX_ATTN_COMBINE=mean|sum`, default `full`) swaps the video self-attention
  for a spatio-temporal factorization (`ltx_factorized_attn.py`). It is ~26%
  faster on stage-2 but **fails fidelity badly** (PSNR 7-10 dB, LPIPS ~0.8 vs
  full); do not enable it for real generation.
- `LTX_FRAMES` must be `8k+1`; 73 hangs the XPU driver (see
  `run_t2v_xpu_perf.py:71`). `LTX_DECODER_MEM_EFFICIENT=1` restores the
  memory-efficient conv path (slower on XPU, but the plain decode's peak is
  closer to the limit on a 24 GB B60).
- Multi-clip launchers additionally read `LTX_PROMPTS_FILE` (JSON array),
  `LTX_MULTI_OUTPUT_DIR` (output default) and `LTX_SPAWN_DELAY` (default 1 s).
  With `--prompts-file`, the batch size must match the launcher (8 in
  `run_multi_xpu.py`, 16 in `run_multi_16.py`). `encode_prompts.py` also accepts
  prompts on stdin JSON.
- Text encoder: `LTX_GEMMA_DEVICE`, `LTX_GEMMA_OFFLOAD`, `LTX_GEMMA_FP8`,
  `LTX_GEMMA_RESIDENT`, `LTX_ENCODE_MODE`. The server's persistent encoder
  service is `LTX_ENCODER_SERVICE=1` (+ `LTX_ENCODER_FP8`, `LTX_ENCODER_SOCK`,
  default `/tmp/ltx_encoder.sock`); `encode_service.py` keeps the Gemma pinned
  source warm across jobs (~16.6 s warm encode for 16 prompts vs a ~44.7 s
  per-job subprocess encode phase). It only engages when the server's
  `LTX_GEMMA_DEVICE` is an `xpu:*` spec.
- Server: `LTX_HOST` (default `127.0.0.1`), `LTX_PORT` (8001; the 2.5 server uses
  8002), `LTX_API_TOKEN` (required when non-loopback), `LTX_MULTI_MODE` (default
  8), `LTX_OUTPUT_DIR`, `LTX_DB`. `LTX_QUEUE_SIZE` is documented and appears in
  server logs but the worker's `queue.Queue` is hardcoded to `maxsize=4`.
- Benchmark: `multi_benchmark.py` reads `LTX_BENCH_URL`/`LTX_BENCH_TOKEN`
  (defaults `http://127.0.0.1:8001`, token `111`) and `LTX_BENCH_OUTPUT`.

## Reference docs

- `README.txt` — full measured history: per-stage timings, compile dead ends,
  R-flag A/Bs, TE analysis, and the B60/B70 comparison.
- `B70使用说明.md` — Chinese quickstart for the single-B70 host (fresh clone →
  run → serve), including the `LTX_TDEV=LTX_CDEV=0` requirement and device thermal
  checks; assumes repo at `/home/acm/paul_arc/ltx2.3-demo`.

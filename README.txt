LTX-2.3 text-to-video on Intel Arc Pro B60 (XPU)
================================================

Quickstart
----------
bash setup_env.sh
uv run download.py
LTX_MULTI_MODE=16 LTX_HOST="0.0.0.0" LTX_API_TOKEN="lotusmind" uv run python ltx_server.py

Single clip:
  .venv/bin/python run_t2v_xpu_perf.py

Multi clip (N videos):
  .venv/bin/python run_multi_xpu.py --prompts-file prompts.json --job-dir OUT
  .venv/bin/python run_multi_16.py --prompts-file prompts.json --job-dir OUT


Web server (16-video service)
-----------------------------
Use the project venv interpreter (.venv/bin/python). Do NOT use run.sh /
start_ltx_server.sh: they hardcode a nonexistent interpreter.

A) Local test (loopback, no token required):
  cd /home/lm/work/ltx2.3-demo
  LTX_MULTI_MODE=16 LTX_HOST=127.0.0.1 \
    .venv/bin/python ltx_server.py
  # open http://127.0.0.1:8001/

B) LAN access (token required when LTX_HOST is not loopback):
  LTX_MULTI_MODE=16 LTX_HOST=0.0.0.0 LTX_API_TOKEN=<token> \
    .venv/bin/python ltx_server.py
  # open http://<host-ip>:8001/ and paste the same token into the API Token box

C) With the persistent encoder service (T3) and a larger spawn stagger:
  LTX_MULTI_MODE=16 LTX_HOST=127.0.0.1 \
  LTX_ENCODER_SERVICE=1 LTX_ENCODER_FP8=1 LTX_SPAWN_DELAY=4 \
    .venv/bin/python ltx_server.py

Run in the background with a log:
  ... .venv/bin/python -u ltx_server.py > /tmp/ltx_server.log 2>&1 &

Options: LTX_PORT (default 8001), LTX_OUTPUT_DIR, LTX_DB, LTX_ENCODER_SOCK,
LTX_ENCODER_FP8=0 for bf16.

In the UI: 16 prompt boxes (count = LTX_MULTI_MODE; fewer is allowed), press
"Generate". Output: $LTX_OUTPUT_DIR/<job_id>/video_i.mp4, per-worker logs
video_i.log. Progress: SSE /api/events.

Pre-flight / cleanup:
  ps -eo pid,cmd | grep -E "ltx_server|run_t2v|encode_service" | grep -v grep
  ss -ltn | grep 8001
  for d in 0 30 31; do xpu-smi stats -d $d | grep -i "GPU Memory Used"; done

Submit a job without the UI:
  curl -s -X POST http://127.0.0.1:8001/api/multi-jobs \
    -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
    -d '{"prompts": ["... 16 prompts ..."]}'

Known issue: 16-way concurrent model init is occasionally unstable (workers
stall in "Building transformer"; recent runs failed 7/16). Raise
LTX_SPAWN_DELAY, or cold-boot to recover the driver, or A/B with
LTX_ENCODER_SERVICE=0 to check whether the encoder service is involved.


Text-encoder performance
------------------------
Progress: the shared pre-encode step (Gemma-3-12B) used to be the single
largest cost of a multi-video job. encode_prompts.py called PromptEncoder
once per prompt and PromptEncoder rebuilds+disposes Gemma on every call, so
a 16-prompt job reloaded the full ~23 GB model 16 times. It also ran without
torch.no_grad(), so Gemma's 48-layer autograd graph was retained, ballooning
CPU memory to ~361 GB and OOM-ing when moved to a 24 GB XPU.

Fixed in f4e97b9 "Run Gemma text encoding on XPU via block streaming":
  - encode the whole prompt list in one batched forward (default);
    LTX_ENCODE_MODE=loop keeps the legacy per-prompt path
  - wrap the encode in torch.no_grad()
  - run Gemma block-streamed on a spare XPU (LTX_GEMMA_DEVICE=xpu:0,
    LTX_GEMMA_OFFLOAD=cpu): ~2 blocks resident (~4.2 GB VRAM), weights
    pinned in CPU RAM. Encoding runs before generation, so every XPU is free.
  - LTX_GEMMA_DEVICE=cpu falls back to the old CPU path

16-prompt encoding (clean environment, same prompts):

  config              encode     wall    s/prompt
  ---------------     ------     -----   --------
  CPU per-prompt      144.4 s    150.4 s   9.02     (old default)
  CPU batched         127.1 s    132.7 s   7.94
  XPU streaming        23.6 s     29.2 s   1.47     (new default)

  -> ~6x faster text encoding; CPU/XPU embeddings agree to <1% relative L2.

End-to-end (16 concurrent 1024x1024, 121-frame videos):
  job wall time 262 s -> 186 s (~29% faster), 16/16 succeeded.
  Encode phase 129 s -> 45 s; generation unchanged at ~130-145 s.

Env vars:
  LTX_GEMMA_DEVICE   text-encoder device: cpu (default in harness) or
                     xpu:0 (server default). Setting an xpu device without
                     LTX_GEMMA_OFFLOAD auto-forces cpu offload.
  LTX_GEMMA_OFFLOAD  none | cpu | disk (cpu = pinned RAM, ~2 blocks on XPU)
  LTX_ENCODE_MODE    batch (default) | loop


Generation performance
----------------------
Per-worker cost at 1024x1024 / 121 frames (16-worker run, before the change):

  stage-1 denoise (8 steps, incl. transformer build)     42.2 s
  spatial-upsample 2x                                     2.1 s
  stage-2 denoise (3 steps, incl. transformer build)     44.3 s
  video+audio decode (audio VAE+vocoder; video lazy)      8.2 s
  mux to mp4 (mostly LAZY video VAE decode)              18.3 s
  -------------------------------------------------------------
  total                                                 ~115 s

Note: VideoDecoder returns a lazy iterator, so the conv VAE video decode
(~16 s) happens while encode_video consumes it and is billed to "mux to mp4".
x264 itself is only ~1 s for a 121-frame 1024x1024 clip.

Shipped: transformer resident across both stages (9e9ee20)
----------------------------------------------------------
DiffusionStage.__call__ rebuilds and disposes the transformer on every call,
so stage 2 reloaded the fp8 weights (~5 s warm, ~13 s cold). Both stages run
on the transformer device and the upsampler/decoders run on the VAE device, so
the model stays resident from stage 1 through stage 2.

  stage-2 denoise    44.6 s -> 39.2 s per worker
  8-worker wall     165.6 s -> 158.0 s
  16-worker wall    182.2 s -> 171.7 s   (8/8 and 16/16 succeeded)

Confirmed rerun (16 workers, 16/16 succeeded): 178.4 s wall
  encode phase (16 prompts)    44.7 s  (29.9 s encode + subprocess startup)
  generation phase            133.6 s
  per worker: stage-1 42.8, upsample 2.1, stage-2 39.4, audio 8.2, mux 18.4

  => 16-video end-to-end wall is ~172-178 s (run-to-run variance).

  LTX_KEEP_TRANSFORMER=0 restores the per-stage rebuild (default 1).

Single clip (transformer on xpu:30, VAE/decoders on xpu:31):
  LTX_TDEV=30 LTX_CDEV=31 LTX_GEMMA_DEVICE=xpu:0 LTX_GEMMA_OFFLOAD=cpu \
    .venv/bin/python run_t2v_xpu_perf.py
  1024x1024 / 121 frames / default prompt: total 131.8 s
    prompt-encode (xpu:0)   19.1 s
    stage-1 denoise         39.3 s
    spatial-upsample 2x      2.0 s
    stage-2 denoise         39.0 s
    video+audio decode      14.2 s
    mux to mp4              18.3 s
  peak xpu:30 18.2 GB, xpu:31 0.8 GB

Single clip on a single B70 (Battlemage G31, 30.3 GiB):
  On this host torch exposes only one XPU (torch.xpu.device_count()==1,
  get_device_name(0)=="Intel(R) Graphics [0xe223]"; the Arrow Lake-S iGPU
  is not exposed), so transformer and VAE/decoders share xpu:0:
    HF_HUB_OFFLINE=1 LTX_TDEV=0 LTX_CDEV=0 \
      LTX_GEMMA_DEVICE=xpu:0 LTX_GEMMA_OFFLOAD=cpu \
      .venv/bin/python -u run_t2v_xpu_perf.py
  1024x1024 / 121 frames / default prompt: total 68.05 s
    prompt-encode (xpu:0)    9.54 s
    stage-1 denoise         17.90 s
    spatial-upsample 2x      0.67 s
    stage-2 denoise         22.09 s
    video+audio decode       5.38 s
    mux to mp4              12.46 s
  peak xpu:0 18.15 GB

  B60 (xpu:30/31) vs B70 (xpu:0), same 1024x1024 / 121 frames / default prompt:
    stage                      B60        B70      speedup
    prompt-encode             19.10 s     9.54 s   2.00x
    stage-1 denoise (8 steps) 39.30 s    17.90 s   2.20x
    spatial-upsample 2x        2.00 s     0.67 s   2.99x
    stage-2 denoise (3 steps) 39.00 s    22.09 s   1.77x
    video+audio decode        14.20 s     5.38 s   2.64x
    mux to mp4                18.30 s    12.46 s   1.47x
    total                    131.80 s    68.05 s   1.94x
    peak VRAM             18.2+0.8 GB  18.15 GB   -
                           (2 cards)    (1 card)
  => ~1.94x end-to-end. On B70 the remaining bottlenecks are stage-2 denoise
     (32.5%) and mux (18.3%, mostly the lazy VAE video decode).
  Caveats: the B60 entry is a 2-device run (transformer/VAE split, with
  cross-device moves) on a different host, while the B70 run keeps everything
  on one XPU. Different host CPU/RAM, driver/oneAPI versions, and run-to-run
  variance (one sample each) all bear on the comparison.

B70 optimizations (single XPU, 30 GiB)
--------------------------------------
Starting from the 68.05 s baseline above, same host / prompt / 1024x1024x121.
Peak is 18.16 GB on xpu:0 in every case.

  config                                   total     main change
  ------------------------------------     -------   --------------------------
  baseline                                 68.05 s
  + overlap encode / weight load           64.27 s   stage-1 17.90 -> 13.71 s
  + non-memory-efficient VAE decode        58.64 s   mux    12.42 ->  7.04 s
  + torch.compile, warm cache              55.82 s   stage-2 22.09 -> 18.30 s
    same, first run (cold compile cache)   69.83 s   one-time JIT cost

* Overlap (default on): run_t2v_xpu_perf.py builds the fp8 transformer in a
  background thread while Gemma encodes the prompt, hiding the ~4 s weight
  load. Disable with LTX_PREBUILD_TRANSFORMER=0.
* Decode (default on): VideoDecoder(memory_efficient=False) is ~1.8x faster
  than the memory-efficient path on XPU at the default tiling (7.1 s vs 12.5 s
  for a 1024x1024/121 latent); activation peak is only ~5.5 GB. Restore the
  old path with LTX_DECODER_MEM_EFFICIENT=1.
* torch.compile (opt-in): run via ./run_t2v_compiled.sh (sets LTX_COMPILE=1).
  Per-block fusion gives ~20% faster denoise steps, but the setup is fussy:
  - this dual-GPU host needs an Intel toolchain matching torch's bundled SYCL
    (libsycl.so.8). The login shell's oneAPI 2026.1 (libsycl.so.9) breaks
    triton ("Backends mismatch" / undefined urDeviceWaitExp), so the wrapper
    runs with `env -i` and an 2025.x icpx (autodetected; libsycl.so.8 matches).
  - triton must be pinned with TRITON_DEFAULT_BACKEND=intel, and the harness
    neutralises triton's NVIDIA backend (both drivers otherwise read "active").
  - setup_env.sh patches LTX-2's fp8_cast.py so the prequant *_scale fold
    tolerates the `_orig_mod.` prefix torch.compile inserts.
  - compile artifacts are cached in .torch_cache/ (gitignored), keyed by the
    fixed shapes, so only the first run pays the JIT.

B70 profiling & utilization
---------------------------
Kernel XPU-time shares for one clip (torch.profiler with CPU+XPU activities;
profiler overhead inflates wall time ~1.3x, so use the shares, not the totals):

  elementwise/norm (add/mul/gelu/rms_norm/layernorm)   ~32%
  GEMM (gemm_kernel / addmm)                             21%
  flash attention (cute::XeFMHAFwdKernel)                15%
  VAE conv decode (gen_conv)                             13%
  memcpy M2D (host -> device)                            10%
  dtype cast (_to_copy, the fp8 -> bf16 upcast)          ~10%

Compute utilization (hand-derived FLOPs vs the B70's ~184 TFLOPS bf16 XMX peak;
the isolated big GEMM reaches ~150-177 TFLOPS, so the model runs at ~half peak):

  stage-1 denoise   84 TFLOPS (46%)    -> 107 compiled (58%)
  stage-2 denoise   92 TFLOPS (50%)    -> 111 compiled (61%)
  theoretical floor: stage-2 3.7 s vs 7.4 s actual (compile closes ~20%)

Memory bandwidth (bench_xpu_bw.py, 1 GiB bf16 tensors):

  HBM reduce (read)      592.6 GB/s   (~97% of the 608 GB/s spec)
  D2D copy (R1+W1)      1149.7 GB/s
  add out  (R2+W1)      1720.9 GB/s
  host -> device (H2D)    28.2 GB/s

  - The denoise stages are compute-bound: the ~18.5 GB of fp8 weights are read
    once per step at <2% of HBM peak, so HBM/PCIe sit mostly idle there.
  - H2D is ~28 GB/s, i.e. NOT the x1 Gen1 the sysfs link attributes suggested.
    The transformer load (~4.4 GB/s) is disk/CPU-bound and Gemma streaming
    (~2.4 GB/s) is compute/host-bound -- neither is link-bound.
  - CPU is idle during generation (~9% usr / 88% idle on 24 threads).
  - The biggest untapped lever is the ~32% elementwise/norm share (fusion via
    torch.compile); next is the fp8->bf16 upcast (~10%).

  Tooling notes: `intel_gpu_top` cannot read the B70's PMU (i915-only; the B70
  uses the `xe` driver) and `xpu-smi` only enumerated the integrated GPU on this
  host, so use `bench_xpu_bw.py` and `torch.profiler` instead.

Operator-level acceleration (measured)
--------------------------------------
With `LTX_COMPILE=1` the model has no Dynamo graph breaks (`fullgraph=True`
compiles) and inductor already fuses the hot elementwise chains
(`triton_red_fused__fused_rms_norm__to_copy_add_mul_sl...`,
`triton_poi_fused_gelu_*`, `triton_poi_fused__to_copy_*`), so most of the eager
~32% elementwise share is captured by compile. Sweep results:

  - GEMM is near XMX peak in isolation (~150-177 TFLOPS) but denoise runs at
    ~84-111 TFLOPS end-to-end (46-61% of ~184). The gap is attention (~55% peak
    for the cute XeFMHA kernel) plus the residual elementwise mix, not raw GEMM.
  - fp8 native GEMM is unavailable (`torch._scaled_mm` unimplemented on XPU) and
    a bf16 cache of the whole transformer (37 GB) does not fit 30 GB, so the
    ~10% fp8 -> bf16 upcast cannot be removed cheaply.
  - Compiling the conv VAE decoder is a LOSS: warm mux 10.85 s vs 7.05 s
    (eager oneDNN convs win), with a ~130 s cold compile.
  - `seq_dim_dynamic=False` helps stage-1 (14.05 -> 11.95 s) but hurts stage-2
    (18.30 -> 20.46 s); net slightly worse, so the dynamic default stays.
  - Kernel-launch overhead is small (~16k triton + 19k gemm launches per run),
    so XPU graph capture is not a promising lever.

  => On this torch/XPU stack the operator graph is close to its practical
     limit; the remaining gap is attention-kernel efficiency and vendor
     (oneDNN) coverage, not something the harness can fuse away.

  Compile knobs (all default off/unset): LTX_COMPILE_MODE, LTX_INDUCTOR_CONFIG,
  LTX_DYNAMO_CONFIG (JSON), LTX_SEQ_DYNAMIC, LTX_FULLGRAPH (0|1).

LTX-2.5 (opt-in)
----------------
`run_t2v_25_xpu.py` runs LTX-2.5 distilled text-to-video on a single XPU; the
LTX-2.3 path (`run_t2v_xpu_perf.py`) stays the default.

Why it needs its own runner: the official 2.5 release ships only bf16 (42 GB),
comfy-int8-convrot (21.5 GB) and nvfp4 (18.7 GB) transformers. On a 30 GiB B70
the bf16 does not fit and the int8-convrot / nvfp4 formats need ComfyUI /
ltx_kernels CUDA kernels, so the only loadable form is our own fp8 cast of the
bf16 checkpoint (~21 GB resident). The Gemma-4 text encoder (26 GB) is streamed
with CPU offload, and the fp8 transformer is kept resident across both stages.

Measured (1024x1024, 121 frames, 8+3 distilled steps, seed 42):
  prompt-encode (Gemma-4, streamed)   ~14 s
  transformer fp8-cast build           ~6 s
  stage-1 denoise (8 steps)           ~14 s
  stage-2 denoise (3 steps)           ~22 s
  video+audio decode + mux             ~12 s
  generation 58.5 s + mux 7.1 s; peak xpu:0 reserved 25.3 GB
=> 2.5 is roughly 2.3 speed (same 8+3 schedule) with 2.5 quality; ~10 s slower
   overall from the larger Gemma-4 TE and the fp8-cast build.

Knobs: `LTX_KEEP_TRANSFORMER` (default 1), `LTX_DECODER_MEM_EFFICIENT` (default 0),
`LTX_PREBUILD_TRANSFORMER` (default 0 -- concurrent Gemma-4 streaming + transformer
build intermittently raises UR_RESULT_ERROR_DEVICE_LOST on XPU, unlike 2.3).
`LTX_25_MODELS` points at the split pack (default /home/acm/work/models/ltx-2.5).

LTX-2.5 web server (opt-in)
---------------------------
`start_ltx_server_25.sh` -> `ltx_server_25.py` reuses the 2.3 server design
(FastAPI + bearer auth + SSE + SQLite history + HTML UI) but runs single-path:
one prompt, one video per job, one XPU, via `run_t2v_25_xpu.py`. Defaults to
127.0.0.1:8002 (the 2.3 server stays on 8001); set `LTX_HOST=0.0.0.0` and
`LTX_API_TOKEN` for LAN access.

  ./start_ltx_server_25.sh
  curl -s -X POST http://127.0.0.1:8002/api/multi-jobs \
    -H 'Content-Type: application/json' -d '{"prompts":["..."]}'

The API is unchanged (`/api/multi-jobs`, `/api/events`, `.../videos/0`). Jobs
run one at a time (2.5 needs the whole 30 GiB). Each job reloads the fp8
transformer (~6 s) and re-streams Gemma-4 (~14 s) -- there is no gemma4
persistent encoder service (the 2.3 T3 service is gemma3-only). The video
endpoint is unauthenticated, as in 2.3.

Dead ends (measured, reverted)
------------------------------
- x264 / torch thread tuning: "mux to mp4" is dominated by the lazy video
  VAE decode, not the encoder, so there is <1 s of headroom.
- VAE decode tiling / memory_efficient: on the 24 GB B60 the non-memory-
  efficient conv decode peaked near the limit, but on the 30 GiB B70 it is
  ~1.8x faster and only ~5.5 GB (now the default). Growing/removing the
  spatial tiles still OOMs and can reset the XPU driver: keep the default
  768/64 tiling.
- torch.compile via DiffusionStage's CompilationConfig: the compiled
  "._orig_mod" key rewrite used to desync the prequant *_scale keys; setup_env.sh
  now patches fp8_cast.py to make the fold prefix-aware, so compile works.
  The remaining obstacle is environmental (triton dual-driver + SYCL version),
  handled by run_t2v_compiled.sh.
- torch.compile on the conv VAE decoder: warm mux 10.85 s vs 7.05 s eager (plus
  a ~130 s cold compile) -- eager oneDNN convs are faster.
- Compile shape/inductor variants: `fullgraph=True` compiles (no graph breaks
  to fix); `seq_dim_dynamic=False` helps stage-1 but hurts stage-2, net worse.
- Spatio-temporal factorized self-attention (spatial within each frame +
  temporal across frames; two dense flash SDPA passes, reshape-only, no sparse
  kernel): still correct by construction (F=1/S=1 reduce to full exactly) and
  **26% faster stage-2 (22.1 -> 16.2 s)**, but the fidelity gate fails badly --
  vs full attention on 3 prompts, PSNR 7-10 dB and LPIPS 0.75-0.83 (gate was
  PSNR>=24 dB / LPIPS<=0.10). The distilled model needs full 3D attention; a
  static factorization is not a usable approximation. Kept opt-in for
  reference: `LTX_ATTN_PATTERN=factorized` (+ `LTX_ATTN_COMBINE=mean|sum`),
  default is full.

Text-encoder (TE) analysis and Phase 4 A/B
------------------------------------------
TE is the shared pre-encode step: it runs serially on one XPU before the 16
generation workers start, and is ~25% of a 16-video job (44.7 s of 178.4 s).
Cost model per job (16 prompts, measured):
  build Gemma (read 23 GB bf16 -> pinned/stream -> device)   ~11-20 s
  Gemma forward ([16,1024], incl. H2D weight streaming)      ~10.6 s
  embeddings processor (49 layers -> aggregate -> connectors) ~4.8 s
  subprocess torch/ltx import                                 ~10-15 s
So the fixed per-job model load/import dominates; the marginal cost per prompt is
small (batch amortizes the weight streaming).

Phase 4 T1/T2 A/B (8 and 16 prompts; wall / encode):
  T0 bf16 streaming (current)      8p: 29.4 / 21.0   16p: 34.8 / 26.5
  T1 fp8 streaming                 8p: 27.2 / 19.2   16p: 30.7 / 23.3  (-12%)
  T2a fp8 resident, 1 XPU          8p: 30.1 / 22.3   16p: OOM (batch16)
  T2b fp8 resident, K XPUs (shard) 16p: K=2 47.7 s, K=4 32.1 s, K=8 48.5 s
Findings:
  - fp8 (LTX_GEMMA_FP8=1) only halves the H2D bytes; compute stays bf16, so a
    mere ~12% win. It also shifts the embeddings (video rel-L2 4.3%, audio 2.4%).
  - Sharding across K XPUs gives no speedup: every shard re-pays the ~20 s model
    load, so wall is flat and K=8 even contends/fails. Data-parallel replication
    is the wrong shape while the fixed load dominates.
  - Resident fp8 does not fit batch 16 (10 GB weights + 6 GB hidden states +
    upcast temporaries + processor > 24 GB); batch 8 fits.

=> The real lever is to avoid re-loading Gemma every job: a long-lived encoder
process that keeps the pinned CPU weight source warm, rebuilds only the small
GPU wrapper per job, and frees the XPU before generation.

T3 (shipped): persistent encoder service
----------------------------------------
encode_service.py runs a long-lived process that keeps the Gemma pinned CPU
weight source cached (PinnedWeightSource.cleanup neutralised) and serves encode
requests over a Unix socket. The server starts it lazily and falls back to the
encode_prompts.py subprocess on any error.

Measured (16 prompts, one warm process):
  bf16 streaming   cold 24.2 s -> warm 17.5 s
  fp8 streaming    cold 23.9 s -> warm 16.6 s
i.e. the per-job encode drops from the ~44.7 s encode phase (which pays a fresh
subprocess import + pinned rebuild every job) to a warm ~16.6 s. The process
holds no XPU memory between requests, so generation workers get a clean device.
Validated end-to-end via the server (2 jobs x 2 videos, 2/2 each, warm reuse).

A CPU<->XPU model-move variant (keep the built fp8 model on CPU, move it in/out
per job) was rejected: the move alone costs ~27 s/job (many small fp8 copies)
and intermittently raises level_zero OUT_OF_RESOURCES.

Env switches added (default off, so behavior is unchanged):
  LTX_GEMMA_FP8=1        fp8-cast the Gemma linears (streaming path)
  LTX_GEMMA_RESIDENT=1   build Gemma fully resident (implies fp8)
  LTX_ENCODER_SERVICE=1  server uses the persistent encode_service.py process
  LTX_ENCODER_FP8=1      fp8 weights for that service (default on)
  LTX_ENCODER_SOCK       socket path (default /tmp/ltx_encoder.sock)

See AGENTS.md for the full architecture and device layout.

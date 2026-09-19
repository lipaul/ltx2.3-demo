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

Dead ends (measured, reverted)
------------------------------
- x264 / torch thread tuning: "mux to mp4" is dominated by the lazy video
  VAE decode, not the encoder, so there is <1 s of headroom.
- VAE decode tiling / memory_efficient=False: the conv VAE decode peaks near
  the 24 GB limit with the default tiles; larger/untiled configs OOM.
- torch.compile via DiffusionStage's CompilationConfig: incompatible with the
  fp8-cast policy on the pinned LTX-2 revision (the compiled "._orig_mod"
  key rewrite desyncs the prequant *_scale keys). Not shipped.

Remaining known levers (not done)
---------------------------------
- torch.compile the eager-built transformer at the harness level (bypasses the
  sd_ops rewrite). High risk on XPU, unverified payoff.
- conv VAE decode memory/partitioning (dominates the post-denoise tail).
- audio VAE+vocoder (~8 s); the XPU patch upcasts it to fp32 for correctness.

See AGENTS.md for the full architecture and device layout.

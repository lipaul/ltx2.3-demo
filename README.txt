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

See AGENTS.md for the full architecture and device layout.

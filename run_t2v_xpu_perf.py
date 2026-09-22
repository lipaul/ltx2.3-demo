"""LTX-2.3 text-to-video on the first two Intel Arc Pro B60 XPUs.

Layout:
  xpu:0 -> fp8 distilled transformer (DiffusionStage, ~22 GB weights)
  xpu:1 -> video VAE / spatial upsampler / video decoder / audio VAE+vocoder
  cpu   -> Gemma-3-12B text encoder + embeddings processor

Gemma-3-12B in bf16 is ~24 GB, which does not fit a single 24 GB B60, so the
text encoder runs on CPU (one forward pass, no generation; 2 TB RAM + 128 cores).
Models are built and freed sequentially (gpu_model context), so each device only
ever holds one model at a time -> peak XPU VRAM ~= the fp8 transformer (~22 GB),
which fits a 24 GB B60. Tensors are moved across devices at stage boundaries.
For text-to-video (no input images) the image conditioner produces no
conditioning latents, so the only cross-device moves are the prompt context and
the video/audio latents between stages.
"""
import logging
import os
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import torch

from ltx_core.components.noisers import GaussianNoiser
from ltx_core.model.video_vae import TileSizeConfig, get_video_chunks_number
from ltx_core.quantization.fp8_cast import build_policy as fp8_cast_policy
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import assert_resolution, cleanup_memory
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ltx23")

# --- paths ---
DISTILLED_CKPT = str(Path(__file__).resolve().parent / "models" / "ltx-2.3-22b-distilled-fp8.safetensors")
UPSCALER_CKPT = str(Path(__file__).resolve().parent / "models" / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
GEMMA_ROOT = str(Path(__file__).resolve().parent / "models" / "gemma-3-12b-it")
OUTPUT_PATH = os.environ.get("LTX_OUTPUT_PATH", str(Path(__file__).resolve().parent / "output" / "output_1024.mp4"))

# --- generation params (two-stage -> resolution divisible by 64) ---
_DEFAULT_PROMPT = (
    "A cinematic shot of a red panda sitting on a mossy branch in a misty bamboo forest, "
    "gentle morning light, soft bokeh, the panda turns its head and chews a bamboo leaf, "
    "photorealistic, 4k, shallow depth of field."
)
PROMPT = os.environ.get("LTX_PROMPT", _DEFAULT_PROMPT)
EMBEDDINGS_PATH = os.environ.get("LTX_EMBEDDINGS_PATH", "")
SEED = 42
_TARGET_W = int(os.environ.get("LTX_WIDTH", "1024"))
_TARGET_H = int(os.environ.get("LTX_HEIGHT", "1024"))
STAGE1_H, STAGE1_W = _TARGET_H // 2, _TARGET_W // 2  # stage 2 -> target
NUM_FRAMES = int(os.environ.get("LTX_FRAMES", "121"))  # 8k + 1 (73 hangs XPU driver)
FRAME_RATE = 24.0

TDEV = torch.device("xpu", int(os.environ.get("LTX_TDEV", "0")))  # transformer
CDEV = torch.device("xpu", int(os.environ.get("LTX_CDEV", "1")))  # vae / decoders

# Video VAE decode: the memory-efficient path is much slower on XPU at the default
# tiling (measured 12.5 s vs 7.1 s for 1024x1024/121 frames) and its activation
# peak is small (~5.5 GB allocated), so default to the plain conv path.
# Set LTX_DECODER_MEM_EFFICIENT=1 to restore the old behavior.
_DECODER_MEM_EFFICIENT = os.environ.get("LTX_DECODER_MEM_EFFICIENT", "0") == "1"

# Experimental spatio-temporal factorized self-attention for the video stream
# (LTX_ATTN_PATTERN=factorized, LTX_ATTN_COMBINE=mean|sum). Default "full" keeps
# the stock dense attention. See ltx_factorized_attn.py.
_ATTN_PATTERN = os.environ.get("LTX_ATTN_PATTERN", "full").strip().lower()
_ATTN_COMBINE = os.environ.get("LTX_ATTN_COMBINE", "mean").strip().lower()

# Optional torch.compile of the transformer blocks (LTX_COMPILE=1). Compilation
# is only usable on this host with a clean oneAPI/triton environment and the
# NVIDIA triton backend neutralised; see AGENTS.md / README.txt.
# LTX_COMPILE_MODE / LTX_INDUCTOR_CONFIG / LTX_DYNAMO_CONFIG / LTX_SEQ_DYNAMIC /
# LTX_FULLGRAPH override CompilationConfig fields (JSON for the *_CONFIG ones,
# 0|1 for the rest).
_COMPILATION_CONFIG = None
if os.environ.get("LTX_COMPILE", "0") == "1":
    import json as _json

    from ltx_core.model.transformer.compiling import CompilationConfig

    _compile_kwargs: dict = {}
    if os.environ.get("LTX_COMPILE_MODE"):
        _compile_kwargs["mode"] = os.environ["LTX_COMPILE_MODE"]
    if os.environ.get("LTX_INDUCTOR_CONFIG"):
        _compile_kwargs["inductor_config"] = _json.loads(os.environ["LTX_INDUCTOR_CONFIG"])
    if os.environ.get("LTX_DYNAMO_CONFIG"):
        _compile_kwargs["dynamo_config"] = _json.loads(os.environ["LTX_DYNAMO_CONFIG"])
    if os.environ.get("LTX_SEQ_DYNAMIC"):
        _compile_kwargs["seq_dim_dynamic"] = os.environ["LTX_SEQ_DYNAMIC"] == "1"
    if os.environ.get("LTX_FULLGRAPH"):
        _compile_kwargs["fullgraph"] = os.environ["LTX_FULLGRAPH"] == "1"
    _COMPILATION_CONFIG = CompilationConfig(**_compile_kwargs)
    log.info("CompilationConfig: %s", _COMPILATION_CONFIG)


def _neutralise_nvidia_triton_backend() -> None:
    """Make triton's driver discovery pick the Intel backend on a dual-GPU host.

    ``triton.backends.nvidia.driver._cuda_driver_is_active`` probes libcuda
    directly (ignoring CUDA_VISIBLE_DEVICES); with both the Intel and NVIDIA
    drivers active, triton refuses to select one ("2 active drivers").
    """
    try:
        import triton.backends.nvidia.driver as _tn_driver

        _tn_driver._cuda_driver_is_active = lambda: False
        log.info("neutralised triton NVIDIA backend (XPU-only triton)")
    except Exception as e:  # noqa: BLE001
        log.warning("could not neutralise triton NVIDIA backend: %s", e)


_factorized_config_cache = None


def _factorized_configurator():
    global _factorized_config_cache
    if _factorized_config_cache is None:
        import ltx_factorized_attn

        _factorized_config_cache = ltx_factorized_attn.make_configurator(combine=_ATTN_COMBINE)
    return _factorized_config_cache


def _set_attn_grid(pixel_h: int, pixel_w: int) -> None:
    """Set the (F, S) video grid for factorized attention (32 = VAE spatial scale)."""
    if _ATTN_PATTERN != "factorized":
        return
    import ltx_factorized_attn

    frames = (NUM_FRAMES - 1) // 8 + 1
    spatial = (pixel_h // 32) * (pixel_w // 32)
    ltx_factorized_attn.set_video_grid(frames, spatial)
    log.info("factorized attention grid: F=%d S=%d", frames, spatial)


def _resolve_gemma_device(spec: str) -> torch.device:
    spec = (spec or "cpu").strip().lower()
    if spec in ("", "cpu"):
        return torch.device("cpu")
    if spec.startswith("xpu"):
        parts = spec.split(":")
        return torch.device("xpu", int(parts[1]) if len(parts) > 1 else 0)
    return torch.device(spec)


# Gemma text encoder: default CPU (bf16 ~23 GB does not fit a single 24 GB B60).
# Set LTX_GEMMA_DEVICE=xpu:<n> (+ LTX_GEMMA_OFFLOAD=cpu) to stream it on an XPU.
GDEV = _resolve_gemma_device(os.environ.get("LTX_GEMMA_DEVICE", "cpu"))
GEMMA_OFFLOAD = {
    "cpu": OffloadMode.CPU,
    "disk": OffloadMode.DISK,
}.get(os.environ.get("LTX_GEMMA_OFFLOAD", "none").strip().lower(), OffloadMode.NONE)
if GDEV.type == "xpu" and GEMMA_OFFLOAD == OffloadMode.NONE:
    log.warning("Gemma bf16 does not fit a 24GB XPU; forcing LTX_GEMMA_OFFLOAD=cpu")
    GEMMA_OFFLOAD = OffloadMode.CPU


def _mem(tag: str, dev: torch.device) -> None:
    if dev.type != "xpu":
        return
    try:
        used = torch.xpu.memory_allocated(dev) / 1024**3
        reserved = torch.xpu.memory_reserved(dev) / 1024**3
        log.info("[%s] xpu:%s allocated=%.2fGB reserved=%.2fGB", tag, dev.index, used, reserved)
    except Exception as e:  # noqa: BLE001
        log.warning("[%s] mem query failed: %s", tag, e)


# ---- performance timing ----
_stage_timings: dict[str, float] = {}
_peak_mem: dict[int, float] = {}


def _track_peak(dev: torch.device) -> None:
    if dev.type == "xpu":
        idx = dev.index
        cur = torch.xpu.memory_allocated(dev) / 1024**3
        if cur > _peak_mem.get(idx, 0.0):
            _peak_mem[idx] = cur


class _Timer:
    def __init__(self, name: str, dev: torch.device | None = None):
        self.name = name
        self.dev = dev

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        dt = time.perf_counter() - self.t0
        _stage_timings[self.name] = _stage_timings.get(self.name, 0.0) + dt
        if self.dev is not None:
            _track_peak(self.dev)
        try:
            if self.dev is not None and self.dev.type == "xpu":
                torch.xpu.synchronize(self.dev)
        except Exception:
            pass
        log.info("⏱  %-22s %6.2f s", self.name, dt)
        return False


# --- keep the transformer resident across both stages ---
# DiffusionStage.__call__ builds and disposes the transformer on every call, so
# stage 2 reloads the fp8 weights (~13 s cold / ~5 s warm). Both stages run on
# TDEV and the upsampler/decoders live on CDEV, so the model can stay resident
# from stage 1 through stage 2. Set LTX_KEEP_TRANSFORMER=0 to disable.
_KEEP_TRANSFORMER = os.environ.get("LTX_KEEP_TRANSFORMER", "1") == "1"
_transformer_cache: dict[int, object] = {}

if _KEEP_TRANSFORMER:

    def _reuse_transformer_ctx(self, **kwargs):
        model = _transformer_cache.get(id(self))
        if model is None:
            model = self._build_transformer(**kwargs)
            _transformer_cache[id(self)] = model

        @contextmanager
        def _cached():
            yield model

        return _cached()

    DiffusionStage._transformer_ctx = _reuse_transformer_ctx


def _prebuild_transformer_async(stage) -> threading.Thread | None:
    """Build the fp8 transformer while the prompt is being encoded.

    The transformer build (~4 s) is weight-load / H2D-copy bound while the Gemma
    encode (~10 s) is XPU-compute bound, so overlapping them hides the build.
    The result lands in ``_transformer_cache`` so ``stage(...)`` reuses it.
    Set ``LTX_PREBUILD_TRANSFORMER=0`` to keep the old sequential order.
    """
    if not _KEEP_TRANSFORMER:
        return None
    if os.environ.get("LTX_PREBUILD_TRANSFORMER", "1") != "1":
        return None

    def _work() -> None:
        try:
            t0 = time.perf_counter()
            _transformer_cache[id(stage)] = stage._build_transformer(video_tools=None)
            log.info("prebuilt transformer during prompt-encode in %.2f s", time.perf_counter() - t0)
        except Exception as e:  # noqa: BLE001
            log.warning("async transformer prebuild failed (will build on demand): %s", e)

    th = threading.Thread(target=_work, name="prebuild-transformer", daemon=True)
    th.start()
    return th


@torch.inference_mode()
def main() -> None:
    height, width = STAGE1_H * 2, STAGE1_W * 2
    assert_resolution(height=height, width=width, is_two_stage=True)
    dtype = torch.bfloat16
    torch.set_num_threads(os.cpu_count() or 8)  # CPU-side work (Gemma streaming, mux)

    log.info("devices: transformer=%s  vae/decoders=%s  text-encoder=%s", TDEV, CDEV, GDEV)
    log.info("target: %dx%d, %d frames @ %.0ffps (stage1 %dx%d)", width, height, NUM_FRAMES, FRAME_RATE, STAGE1_W, STAGE1_H)

    if _COMPILATION_CONFIG is not None:
        _neutralise_nvidia_triton_backend()

    log.info("building fp8-cast quantization policy from %s", DISTILLED_CKPT)
    quantization = fp8_cast_policy(DISTILLED_CKPT)

    log.info("constructing pipeline blocks")
    prompt_encoder = PromptEncoder(
        model_paths=ModelPaths.from_monolith(DISTILLED_CKPT, GEMMA_ROOT),
        dtype=dtype, device=GDEV, offload_mode=GEMMA_OFFLOAD,
    )
    stage = DiffusionStage.from_checkpoint(
        checkpoint_path=DISTILLED_CKPT, dtype=dtype, device=TDEV,
        loras=(), quantization=quantization,
        compilation_config=_COMPILATION_CONFIG,
        **({"model_configurator": _factorized_configurator()} if _ATTN_PATTERN == "factorized" else {}),
    )
    upsampler = VideoUpsampler(
        checkpoint_path=DISTILLED_CKPT, upsampler_path=UPSCALER_CKPT, dtype=dtype, device=CDEV,
    )
    video_decoder = VideoDecoder(
        checkpoint_path=DISTILLED_CKPT, dtype=dtype, device=CDEV,
        memory_efficient=_DECODER_MEM_EFFICIENT,
    )
    audio_decoder = AudioDecoder(checkpoint_path=DISTILLED_CKPT, dtype=dtype, device=CDEV)

    # noiser/generator lives on the transformer device (latents are created there)
    generator = torch.Generator(device=TDEV).manual_seed(SEED)
    noiser = GaussianNoiser(generator=generator)
    decode_generator = torch.Generator(device=CDEV).manual_seed(SEED)

    # --- prompt encoding (cpu) ---
    if EMBEDDINGS_PATH:
        log.info("loading pre-encoded embeddings from %s", EMBEDDINGS_PATH)
        data = torch.load(EMBEDDINGS_PATH, map_location="cpu", weights_only=True)
        video_context = data["video_encoding"].to(TDEV)
        audio_context = data["audio_encoding"].to(TDEV)
        del data
    else:
        prebuild_thread = _prebuild_transformer_async(stage)
        with _Timer(f"prompt-encode ({GDEV})", CDEV):
            log.info("encoding prompt on %s", GDEV)
            (ctx_p,) = prompt_encoder([PROMPT], enhance_first_prompt=False, enhance_prompt_image=None)
        if prebuild_thread is not None:
            prebuild_thread.join()
        _mem("after prompt-encode", CDEV)
        # move context to transformer device
        video_context = ctx_p.video_encoding.to(TDEV)
        audio_context = ctx_p.audio_encoding.to(TDEV)

    stage_1_sigmas = DISTILLED_SIGMAS.to(dtype=torch.float32, device=TDEV)
    stage_2_sigmas = STAGE_2_DISTILLED_SIGMAS.to(dtype=torch.float32, device=TDEV)
    s1_w, s1_h = STAGE1_W, STAGE1_H

    # --- Stage 1: low-res denoise on xpu:0 ---
    log.info("stage 1: %dx%d %d frames on %s", s1_w, s1_h, NUM_FRAMES, TDEV)
    _set_attn_grid(s1_h, s1_w)
    with _Timer("stage-1 denoise (8 steps)", TDEV):
        video_state, audio_state = stage(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_1_sigmas, noiser=noiser,
            width=s1_w, height=s1_h, frames=NUM_FRAMES, fps=FRAME_RATE,
            video=ModalitySpec(context=video_context, conditionings=[]),
            audio=ModalitySpec(context=audio_context),
        )
    _mem("after stage1", TDEV)

    # --- spatial upsample 2x on xpu:1 ---
    with _Timer("spatial-upsample 2x", CDEV):
        log.info("spatial upsample 2x on %s", CDEV)
        upscaled_video_latent = upsampler(video_state.latent[:1].to(CDEV))
        upscaled_video_latent = upscaled_video_latent.to(TDEV)

    # --- Stage 2: high-res refine on xpu:0 ---
    log.info("stage 2: %dx%d %d frames on %s", width, height, NUM_FRAMES, TDEV)
    _set_attn_grid(height, width)
    with _Timer("stage-2 denoise (3 steps)", TDEV):
        video_state, audio_state = stage(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage_2_sigmas, noiser=noiser,
            width=width, height=height, frames=NUM_FRAMES, fps=FRAME_RATE,
            video=ModalitySpec(
                context=video_context, conditionings=[],
                noise_scale=stage_2_sigmas[0].item(), initial_latent=upscaled_video_latent,
            ),
            audio=ModalitySpec(
                context=audio_context, noise_scale=stage_2_sigmas[0].item(),
                initial_latent=audio_state.latent.to(TDEV),
            ),
        )
    _mem("after stage2", TDEV)

    if _KEEP_TRANSFORMER:
        for _m in _transformer_cache.values():
            try:
                _m.dispose()
            except Exception:
                _m.to("meta")
        _transformer_cache.clear()
        cleanup_memory()

    # --- decode video + audio on xpu:1 ---
    with _Timer("video+audio decode", CDEV):
        log.info("decoding video + audio on %s", CDEV)
        tiling_config = TileSizeConfig.default()
        decoded_video = video_decoder(video_state.latent.to(CDEV), tiling_config, decode_generator)
        decoded_audio = audio_decoder(audio_state.latent.to(CDEV))
    _mem("after decode", CDEV)

    video_chunks_number = get_video_chunks_number(NUM_FRAMES, tiling_config)
    with _Timer("mux to mp4"):
        log.info("encoding to %s (chunks=%d)", OUTPUT_PATH, video_chunks_number)
        encode_video(
            video=decoded_video, fps=FRAME_RATE, audio=decoded_audio,
            output_path=OUTPUT_PATH, video_chunks_number=video_chunks_number,
        )
    log.info("DONE -> %s", OUTPUT_PATH)

    # ---- performance summary ----
    total = sum(_stage_timings.values())
    log.info("=" * 60)
    log.info("PERFORMANCE SUMMARY  (%dx%d, %d frames @ %.0ffps)", width, height, NUM_FRAMES, FRAME_RATE)
    log.info("-" * 60)
    for name, dt in _stage_timings.items():
        log.info("  %-24s %7.2f s  %5.1f%%", name, dt, 100.0 * dt / total if total else 0)
    log.info("  %-24s %7.2f s", "TOTAL", total)
    n_latent_frames = (NUM_FRAMES - 1) // 8 + 1
    log.info("  latent frames=%d  video tokens(stage2)~%d", n_latent_frames, n_latent_frames * (width // 32) * (height // 32))
    for idx in sorted(_peak_mem):
        log.info("  peak xpu:%d allocated = %.2f GB", idx, _peak_mem[idx])
    log.info("=" * 60)


if __name__ == "__main__":
    main()

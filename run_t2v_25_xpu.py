"""LTX-2.5 distilled text-to-video on a single Intel XPU (B70). Opt-in.

The official 2.5 release ships only bf16 (42 GB), comfy-int8-convrot (21.5 GB)
and nvfp4 (18.7 GB) transformers. On a 30 GiB B70 the only loadable format is
our own fp8 cast of the bf16 checkpoint (~21 GB resident); int8-convrot / nvfp4
need ComfyUI / ltx_kernels CUDA kernels. The Gemma-4 text encoder (26 GB) is
streamed with CPU offload while the fp8 transformer stays resident.

This is the LTX-2.5 counterpart to run_t2v_xpu_perf.py (LTX-2.3, the default).
See README.txt "LTX-2.5 (opt-in)" for the caveats and measured numbers.

Run:
    .venv/bin/python run_t2v_25_xpu.py
Env: LTX_PROMPT, LTX_WIDTH/HEIGHT, LTX_FRAMES, LTX_TDEV, LTX_OUTPUT_PATH,
     LTX_25_MODELS (split pack root; default <repo>/models/ltx-2.5).
"""

import logging
import os
import time
from contextlib import contextmanager
from pathlib import Path

import torch

# The R3/R5/R6/R8/R9D/R10D elementwise fusions target the LTX-2.3 block layout
# (AdaLN slice shapes, gated attention, RoPE split); the same block code runs
# for 2.5 but was not validated against it, so default them OFF here (export the
# flag to opt in). R9A (fp8->bf16 widen) is shared via fp8_cast and stays on; it
# is bitwise-identical to torch's upcast.
for _flag in (
    "LTX_R3_FUSE",
    "LTX_R5_FUSE",
    "LTX_R6_FUSE",
    "LTX_R8_SYCL",
    "LTX_R9D_K3V",
    "LTX_R10D_GATE",
    "LTX_R10D_GATE2",
):
    os.environ.setdefault(_flag, "0")


def _env_on(name: str, default: str = "0") -> bool:
    if os.environ.get("LTX_ALL_ORIG", "0") == "1":
        return False
    return os.environ.get(name, default) != "0"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ltx25")

MODELS = Path(os.environ.get("LTX_25_MODELS", Path(__file__).resolve().parent / "models" / "ltx-2.5"))
TRANSFORMER = str(MODELS / "diffusion_models" / "ltx-2.5-22b-distilled-transformer-bf16.safetensors")
TEXT_ENCODER = str(MODELS / "text_encoders" / "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors")
VIDEO_VAE = str(MODELS / "vae" / "ltx-2.5-video-vae-conv-bf16.safetensors")
AUDIO_VAE = str(MODELS / "vae" / "ltx-2.5-audio-vae-bf16.safetensors")
DURATION_HEAD = str(MODELS / "model_patches" / "ltx-2.5-duration-head-bf16.safetensors")
SPATIAL_UPSCALER = str(MODELS / "latent_upscale_models" / "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors")

_DEFAULT_PROMPT = (
    "A cinematic shot of a red panda sitting on a mossy branch in a misty bamboo forest, "
    "gentle morning light, soft bokeh, the panda turns its head and chews a bamboo leaf, "
    "photorealistic, 4k, shallow depth of field."
)
PROMPT = os.environ.get("LTX_PROMPT", _DEFAULT_PROMPT)
WIDTH = int(os.environ.get("LTX_WIDTH", "1024"))
HEIGHT = int(os.environ.get("LTX_HEIGHT", "1024"))
FRAMES = int(os.environ.get("LTX_FRAMES", "121"))  # 8k+1
FPS = 24.0
SEED = 42
DEVICE = torch.device("xpu", int(os.environ.get("LTX_TDEV", "0")))
OUTPUT = os.environ.get(
    "LTX_OUTPUT_PATH", str(Path(__file__).resolve().parent / "output" / "output_25_1024.mp4")
)

# Same knobs as the 2.3 harness (see README.txt).
_KEEP_TRANSFORMER = os.environ.get("LTX_KEEP_TRANSFORMER", "1") == "1"
_DECODER_MEM_EFFICIENT = os.environ.get("LTX_DECODER_MEM_EFFICIENT", "0") == "1"
_TRANSFORMER_CACHE: dict[int, object] = {}


def _install_speed_knobs() -> None:
    """Reuse the transformer across both stages and use the fast conv decode.

    DiffusionStage.__call__ rebuilds/disposes the transformer per stage (~5 s
    each, re-reading the fp8 weights); the pipeline calls one stage twice, so a
    per-instance cache removes the stage-2 rebuild. The conv VAE decode is ~1.8x
    faster than the memory-efficient path on XPU (see README.txt).
    """
    from ltx_pipelines.utils.blocks import DiffusionStage, VideoDecoder

    if _KEEP_TRANSFORMER:

        def _reuse_transformer_ctx(self, **kwargs):
            model = _TRANSFORMER_CACHE.get(id(self))
            if model is None:
                model = self._build_transformer(**kwargs)
                _TRANSFORMER_CACHE[id(self)] = model

            @contextmanager
            def _cached():
                yield model

            return _cached()

        DiffusionStage._transformer_ctx = _reuse_transformer_ctx

    if not _DECODER_MEM_EFFICIENT:
        _orig_init = VideoDecoder.__init__

        def _init(self, *args, **kwargs):
            kwargs["memory_efficient"] = False
            _orig_init(self, *args, **kwargs)

        VideoDecoder.__init__ = _init


def _prebuild_transformer_async(stage) -> "threading.Thread | None":
    """Build the fp8 transformer while the prompt is encoded (hides ~5 s).

    Off by default: on 2.5 the concurrent Gemma-4 streaming + transformer build
    intermittently raises UR_RESULT_ERROR_DEVICE_LOST on XPU (it was stable on
    2.3). Opt in with LTX_PREBUILD_TRANSFORMER=1.
    """
    if not _KEEP_TRANSFORMER or os.environ.get("LTX_PREBUILD_TRANSFORMER", "0") != "1":
        return None
    import threading

    def _work() -> None:
        try:
            t0 = time.perf_counter()
            _TRANSFORMER_CACHE[id(stage)] = stage._build_transformer(video_tools=None)
            log.info("prebuilt transformer during encode in %.1f s", time.perf_counter() - t0)
        except Exception as e:  # noqa: BLE001
            log.warning("async transformer prebuild failed: %s", e)

    th = threading.Thread(target=_work, name="prebuild-transformer", daemon=True)
    th.start()
    return th


def _require_25_pack() -> None:
    """Fail fast with actionable guidance when the 2.5 split pack is missing."""
    required = {
        "transformer": TRANSFORMER,
        "text_encoder": TEXT_ENCODER,
        "video_vae": VIDEO_VAE,
        "audio_vae": AUDIO_VAE,
        "duration_head": DURATION_HEAD,
        "spatial_upscaler": SPATIAL_UPSCALER,
    }
    missing = [f"  {name}: {path}" for name, path in required.items() if not Path(path).is_file()]
    if missing:
        default_dir = Path(__file__).resolve().parent / "models" / "ltx-2.5"
        raise SystemExit(
            "LTX-2.5 model pack not found.\n"
            f"Set LTX_25_MODELS to the split pack root, or place/symlink it at {default_dir}.\n"
            f"Current root: {MODELS}\nMissing:\n" + "\n".join(missing)
        )


def _mem(tag: str) -> None:
    try:
        used = torch.xpu.memory_allocated(DEVICE) / 1024**3
        reserved = torch.xpu.memory_reserved(DEVICE) / 1024**3
        log.info("[%s] xpu:%s allocated=%.2fGB reserved=%.2fGB", tag, DEVICE.index, used, reserved)
    except Exception as e:  # noqa: BLE001
        log.warning("[%s] mem query failed: %s", tag, e)


@torch.inference_mode()
def main() -> None:
    torch.set_num_threads(os.cpu_count() or 8)
    _require_25_pack()

    from ltx_core.model.video_vae import AUTO_TILING, get_video_chunks_number
    from ltx_core.quantization.fp8_cast import build_policy as fp8_cast_policy
    from ltx_pipelines.distilled import DistilledPipeline
    from ltx_pipelines.utils.blocks import PromptEncoder
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_pipelines.utils.model_paths import ModelPaths
    from ltx_pipelines.utils.types import OffloadMode

    log.info("LTX-2.5 t2v: %dx%d, %d frames @ %.0ffps on %s", WIDTH, HEIGHT, FRAMES, FPS, DEVICE)
    model_paths = ModelPaths.from_split(
        transformer_path=TRANSFORMER,
        text_encoder_path=TEXT_ENCODER,
        video_vae_path=VIDEO_VAE,
        audio_vae_path=AUDIO_VAE,
        duration_head_path=DURATION_HEAD,
    )

    log.info("quantizing transformer to fp8 at load (bf16 source is 42 GB)")
    quantization = fp8_cast_policy(TRANSFORMER)

    _install_speed_knobs()

    t0 = time.perf_counter()
    pipe = DistilledPipeline(
        model_paths,
        spatial_upsampler_path=SPATIAL_UPSCALER,
        loras=[],
        device=DEVICE,
        quantization=quantization,
        offload_mode=OffloadMode.NONE,
    )
    # The pipeline shares one offload mode for TE + transformer. Rebuild the
    # text encoder with CPU offload so Gemma-4 (26 GB) streams instead of
    # trying to be co-resident with the 21 GB fp8 transformer.
    pipe.prompt_encoder = PromptEncoder(
        model_paths,
        torch.bfloat16,
        DEVICE,
        offload_mode=OffloadMode.CPU,
    )
    log.info("pipeline constructed in %.1f s", time.perf_counter() - t0)
    _mem("after construct")

    log.info("generating...")
    t0 = time.perf_counter()
    prebuild = _prebuild_transformer_async(pipe.stage)
    result = pipe(
        prompt=PROMPT,
        seed=SEED,
        height=HEIGHT,
        width=WIDTH,
        frame_rate=FPS,
        images=[],
        num_frames=FRAMES,
        tiling_config=(None if _env_on("LTX_R2_NOTILE", "1") else AUTO_TILING),
    )
    if prebuild is not None:
        prebuild.join()
    log.info("generation finished in %.1f s", time.perf_counter() - t0)
    _mem("after generate")

    out = Path(OUTPUT)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    encode_video(
        video=result.video,
        fps=FPS,
        audio=result.audio,
        output_path=str(out),
        video_chunks_number=get_video_chunks_number(result.num_frames, result.tiling_config),
    )
    log.info("DONE -> %s (mux %.1f s)", out, time.perf_counter() - t0)


if __name__ == "__main__":
    main()

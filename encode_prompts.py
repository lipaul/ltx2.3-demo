"""Pre-encode prompts with one Gemma instance.

Saves each embedding as a .pt file; run this once before run_multi_xpu.py
to share a single text encoder across all generation workers.

Encoding mode (``LTX_ENCODE_MODE``):
  batch (default) -> one PromptEncoder call for the whole list, so Gemma is
                     built once and the batch runs as a single forward.
  loop            -> legacy per-prompt calls (rebuilds Gemma each prompt).

Device (``LTX_GEMMA_DEVICE``): ``cpu`` (default) or ``xpu:<n>``.
Offload (``LTX_GEMMA_OFFLOAD``): ``none`` (default), ``cpu`` or ``disk``.
"""

import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

import torch

import gemma_fp8
from ltx_core.quantization.fp8_cast import build_policy as fp8_cast_policy
from ltx_pipelines.utils.blocks import PromptEncoder
from ltx_pipelines.utils.model_paths import ModelPaths
from ltx_pipelines.utils.types import OffloadMode

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("encode_prompts")

DISTILLED_CKPT = str(Path(__file__).resolve().parent / "models" / "ltx-2.3-22b-distilled-fp8.safetensors")
GEMMA_ROOT = str(Path(__file__).resolve().parent / "models" / "gemma-3-12b-it")


def _resolve_device(spec: str) -> torch.device:
    spec = (spec or "cpu").strip().lower()
    if spec in ("", "cpu"):
        return torch.device("cpu")
    if spec.startswith("xpu"):
        parts = spec.split(":")
        return torch.device("xpu", int(parts[1]) if len(parts) > 1 else 0)
    return torch.device(spec)


def _resolve_offload(spec: str) -> OffloadMode:
    spec = (spec or "none").strip().lower()
    if spec == "cpu":
        return OffloadMode.CPU
    if spec == "disk":
        return OffloadMode.DISK
    return OffloadMode.NONE


def _install_gemma_fp8() -> None:
    """Make PromptEncoder's internally-built Gemma ops fp8-cast (streaming path)."""
    import ltx_pipelines.utils.blocks as blocks_mod

    if getattr(blocks_mod, "_ltx_gemma_fp8_installed", False):
        return
    original = blocks_mod.get_gemma_ops

    def patched(path):
        sd_ops, module_ops = original(path)
        return gemma_fp8.with_gemma_fp8(sd_ops, module_ops)

    blocks_mod.get_gemma_ops = patched
    blocks_mod._ltx_gemma_fp8_installed = True
    log.info("installed Gemma fp8-cast ops (streaming)")


def _build_resident_fp8_encoder(device: torch.device) -> PromptEncoder:
    """Build a PromptEncoder whose Gemma runs fully resident with fp8 weights."""
    from ltx_core.loader.registry import ModelRegistry
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
    from ltx_core.text_encoders.gemma import (
        GemmaTextEncoderConfigurator,
        get_gemma_ops,
        resolve_gemma_weight_paths,
    )

    sd_ops, module_ops = get_gemma_ops(GEMMA_ROOT)
    sd_ops, module_ops = gemma_fp8.with_gemma_fp8(sd_ops, module_ops)
    builder = Builder(
        model_path=resolve_gemma_weight_paths(GEMMA_ROOT),
        model_class_configurator=GemmaTextEncoderConfigurator.with_gemma_model_path(GEMMA_ROOT),
        model_sd_ops=sd_ops,
        module_ops=module_ops,
        registry=ModelRegistry(cache_models=True, cache_weights=False),
    )
    return PromptEncoder(
        model_paths=ModelPaths.from_monolith(DISTILLED_CKPT, GEMMA_ROOT),
        dtype=torch.bfloat16,
        device=device,
        text_encoder_builder=builder,
    )


@torch.no_grad()
def main() -> None:
    torch.set_num_threads(os.cpu_count() or 8)

    # Read prompts: from LTX_PROMPTS_FILE env, or stdin JSON, or default
    prompts_json = os.environ.get("LTX_PROMPTS_FILE", "")
    if prompts_json:
        with open(prompts_json) as f:
            prompts = json.load(f)
    elif not sys.stdin.isatty():
        prompts = json.load(sys.stdin)
    else:
        prompts = [
            "A cinematic shot of a red panda sitting on a mossy branch in a misty bamboo forest, "
            "gentle morning light, soft bokeh, the panda turns its head and chews a bamboo leaf, "
            "photorealistic, 4k, shallow depth of field.",
            "A majestic eagle soaring over a deep canyon at golden hour, warm sunlight "
            "illuminating its wings, dramatic clouds, slow camera pan, "
            "photorealistic, cinematic lighting, 8k.",
            "An underwater scene with a sea turtle swimming through a coral reef, "
            "sunbeams piercing through the water surface, colorful fish, "
            "photorealistic, volumetric lighting, 4k.",
            "A cyberpunk city at night with neon signs reflecting on wet streets, "
            "a lone figure walking under an umbrella, flying cars in the distance, "
            "cinematic, blade runner aesthetic, 8k, anamorphic lens.",
            "A serene mountain lake at sunrise with mist rising from the water, "
            "pine trees reflected in the calm surface, a wooden dock extending into the lake, "
            "photorealistic, warm golden light, hyper-realistic.",
            "A macro shot of a dragonfly perched on a dewy leaf, morning light, "
            "translucent wings with intricate vein patterns, shallow depth of field, "
            "photorealistic, 4k, ultradetailed.",
            "A medieval castle on a stormy cliff edge, lightning flashing behind it, "
            "waves crashing against the rocks, dramatic clouds, dark moody atmosphere, "
            "cinematic, epic scale, photorealistic.",
            "A futuristic greenhouse on Mars under a transparent dome, "
            "lush exotic plants, Earth visible in the twilight sky, "
            "soft artificial lighting, photorealistic, sci-fi aesthetic, 8k.",
        ]

    mode = os.environ.get("LTX_ENCODE_MODE", "batch").strip().lower()
    device = _resolve_device(os.environ.get("LTX_GEMMA_DEVICE", "cpu"))
    offload_mode = _resolve_offload(os.environ.get("LTX_GEMMA_OFFLOAD", "none"))
    use_fp8 = os.environ.get("LTX_GEMMA_FP8", "0") == "1"
    resident = os.environ.get("LTX_GEMMA_RESIDENT", "0") == "1"

    if resident:
        # Resident fp8 Gemma ignores offload (needs the whole model on one device).
        offload_mode = OffloadMode.NONE
        use_fp8 = True
    if device.type == "xpu" and offload_mode == OffloadMode.NONE and not resident:
        log.warning("Gemma bf16 does not fit a 24GB XPU; forcing LTX_GEMMA_OFFLOAD=cpu")
        offload_mode = OffloadMode.CPU
    if use_fp8 and not resident:
        _install_gemma_fp8()

    log.info("building fp8-cast quantization policy...")
    _ = fp8_cast_policy(DISTILLED_CKPT)

    t_build = time.perf_counter()
    log.info(
        "building PromptEncoder (device=%s, offload=%s, mode=%s, fp8=%s, resident=%s)...",
        device, offload_mode.value, mode, use_fp8, resident,
    )
    if resident:
        prompt_encoder = _build_resident_fp8_encoder(device)
    else:
        prompt_encoder = PromptEncoder(
            model_paths=ModelPaths.from_monolith(DISTILLED_CKPT, GEMMA_ROOT),
            dtype=torch.bfloat16,
            device=device,
            offload_mode=offload_mode,
        )
    log.info("PromptEncoder constructed in %.1f s", time.perf_counter() - t_build)

    out_dir = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp(prefix="ltx_embeddings_")
    os.makedirs(out_dir, exist_ok=True)

    t_encode = time.perf_counter()
    if mode == "loop":
        results: list = []
        for i, prompt in enumerate(prompts):
            t0 = time.perf_counter()
            log.info("encoding prompt %d/%d: %s", i + 1, len(prompts), prompt[:70])
            (ctx_p,) = prompt_encoder([prompt], enhance_first_prompt=False, enhance_prompt_image=None)
            results.append(ctx_p)
            log.info("  prompt %d done in %.1f s", i + 1, time.perf_counter() - t0)
    else:
        log.info("encoding %d prompts in one batch: %s", len(prompts), prompts[0][:70])
        results = prompt_encoder(prompts, enhance_first_prompt=False, enhance_prompt_image=None)
    encode_s = time.perf_counter() - t_encode
    log.info("Gemma encode of %d prompts done in %.1f s (%.2f s/prompt)", len(prompts), encode_s, encode_s / len(prompts))

    t_save = time.perf_counter()
    for i, ctx_p in enumerate(results):
        data = {
            "video_encoding": ctx_p.video_encoding.cpu(),
            "audio_encoding": ctx_p.audio_encoding.cpu(),
        }
        torch.save(data, os.path.join(out_dir, f"embeddings_{i}.pt"))
    log.info("saved %d embeddings in %.1f s", len(results), time.perf_counter() - t_save)

    del prompt_encoder
    log.info("all done — %d files in %s", len(prompts), out_dir)
    print(out_dir)  # print path so launcher can capture it


if __name__ == "__main__":
    main()

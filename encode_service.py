"""Long-lived Gemma text encoder for the server / multi-job launchers.

Keeps the block-streaming pinned CPU weight source (the Gemma weights) resident
across jobs, so a warm encode only rebuilds the small GPU wrapper and runs the
forward instead of re-reading/re-pinning the checkpoint every job. GPU state is
released after each encode so generation subprocesses get a clean device.

The pinned source is owned by the streaming builder and normally freed by
``PinnedWeightSource.cleanup()`` during teardown; this module keeps it alive
(bounded, deliberate resident cost). Optionally fp8-casts the Gemma Linears
(``fp8=True``) to halve the pinned/H2D bytes.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import time
from pathlib import Path

import torch

log = logging.getLogger("encode_service")

DISTILLED_CKPT = str(Path(__file__).resolve().parent / "models" / "ltx-2.3-22b-distilled-fp8.safetensors")
GEMMA_ROOT = str(Path(__file__).resolve().parent / "models" / "gemma-3-12b-it")

_PIN_CACHE_INSTALLED = False
_GEMMA_FP8_INSTALLED = False
_PINNED_SOURCES: dict[int, tuple] = {}


def _install_pinned_source_cache() -> None:
    """Reuse one pinned CPU source per streaming builder for the process lifetime."""
    global _PIN_CACHE_INSTALLED
    if _PIN_CACHE_INSTALLED:
        return
    from ltx_core.block_streaming import builder as bs_builder
    from ltx_core.block_streaming import source as bs_source

    original = bs_builder.StreamingModelBuilder._build_pinned_source

    def cached_build_pinned_source(self, *args, **kwargs):
        key = id(self)
        hit = _PINNED_SOURCES.get(key)
        if hit is None:
            hit = original(self, *args, **kwargs)
            _PINNED_SOURCES[key] = hit
            log.info("pinned Gemma weight source cached in CPU RAM")
        return hit

    bs_builder.StreamingModelBuilder._build_pinned_source = cached_build_pinned_source
    bs_source.PinnedWeightSource.cleanup = lambda self: None
    _PIN_CACHE_INSTALLED = True


def _install_gemma_fp8() -> None:
    """Make PromptEncoder's internally-built Gemma ops fp8-cast (streaming path)."""
    global _GEMMA_FP8_INSTALLED
    if _GEMMA_FP8_INSTALLED:
        return
    import gemma_fp8
    import ltx_pipelines.utils.blocks as blocks_mod

    original = blocks_mod.get_gemma_ops

    def patched(path):
        sd_ops, module_ops = original(path)
        return gemma_fp8.with_gemma_fp8(sd_ops, module_ops)

    blocks_mod.get_gemma_ops = patched
    _GEMMA_FP8_INSTALLED = True
    log.info("installed Gemma fp8-cast ops")


class PersistentPromptEncoder:
    """PromptEncoder whose Gemma weight source survives between calls."""

    def __init__(self, device: torch.device, offload_mode=None, fp8: bool = False) -> None:
        from ltx_pipelines.utils.blocks import PromptEncoder
        from ltx_pipelines.utils.model_paths import ModelPaths
        from ltx_pipelines.utils.types import OffloadMode

        _install_pinned_source_cache()
        if fp8:
            _install_gemma_fp8()
        if offload_mode is None:
            offload_mode = OffloadMode.CPU
        if device.type == "xpu" and offload_mode == OffloadMode.NONE:
            log.warning("Gemma bf16 does not fit a 24GB XPU; forcing cpu offload")
            offload_mode = OffloadMode.CPU

        self._device = device
        self._pe = PromptEncoder(
            model_paths=ModelPaths.from_monolith(DISTILLED_CKPT, GEMMA_ROOT),
            dtype=torch.bfloat16,
            device=device,
            offload_mode=offload_mode,
        )

    @torch.no_grad()
    def encode(self, prompts: list[str]) -> list:
        """Encode a batch; PromptEncoder's TRIM lifecycle frees the XPU after."""
        return self._pe(prompts, enhance_first_prompt=False, enhance_prompt_image=None)

    def close(self) -> None:
        self._pe = None


# ---------------------------------------------------------------------------
# Unix-socket service (one request per connection)
# ---------------------------------------------------------------------------


def _save_embeddings(results: list, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for i, ctx in enumerate(results):
        torch.save(
            {"video_encoding": ctx.video_encoding.cpu(), "audio_encoding": ctx.audio_encoding.cpu()},
            os.path.join(out_dir, f"embeddings_{i}.pt"),
        )


def serve(sock_path: str, device: str = "xpu:0", fp8: bool = True) -> None:
    """Serve encode requests until killed. Encoder is built lazily on first use."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if ":" in device:
        kind, idx = device.split(":", 1)
        dev = torch.device(kind, int(idx))
    else:
        dev = torch.device(device)
    Path(sock_path).unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(sock_path)
    server.listen(2)
    log.info("encoder service listening on %s (device=%s fp8=%s)", sock_path, device, fp8)
    encoder: PersistentPromptEncoder | None = None

    while True:
        conn, _ = server.accept()
        try:
            data = b""
            while not data.endswith(b"\n"):
                chunk = conn.recv(1 << 20)
                if not chunk:
                    break
                data += chunk
            req = json.loads(data.decode())
            if req.get("ping"):
                conn.sendall(b'{"ok": true, "pong": true}\n')
                continue
            if encoder is None:
                encoder = PersistentPromptEncoder(dev, fp8=fp8)
            t0 = time.perf_counter()
            results = encoder.encode(req["prompts"])
            _save_embeddings(results, req["out_dir"])
            del results
            log.info("encoded %d prompts in %.1f s", len(req["prompts"]), time.perf_counter() - t0)
            conn.sendall((json.dumps({"ok": True, "out_dir": req["out_dir"]}) + "\n").encode())
        except Exception as exc:  # noqa: BLE001
            log.exception("encode request failed")
            try:
                conn.sendall((json.dumps({"ok": False, "error": str(exc)}) + "\n").encode())
            except OSError:
                pass
        finally:
            conn.close()


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", required=True)
    ap.add_argument("--device", default="xpu:0")
    ap.add_argument("--fp8", action="store_true")
    ap.add_argument("--bf16", action="store_true", help="disable fp8 (kept for symmetry)")
    args = ap.parse_args()
    serve(args.sock, device=args.device, fp8=args.fp8 and not args.bf16)


if __name__ == "__main__":
    main()

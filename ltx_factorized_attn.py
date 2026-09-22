"""Spatio-temporal factorized self-attention for the LTX transformer (B70 experiment).

Replaces the full dense self-attention over ``F*S`` video tokens with two dense,
unmasked SDPA calls on reshaped views:

  spatial  : [B, F*S, H*Dh] -> [B*F, H, S, Dh] -> SDPA -> back
  temporal : [B, F*S, H, Dh] -> [B*S, H, F, Dh] -> SDPA -> back

Both branches are unmasked, so the XPU flash backend stays active (a window
implemented as an additive mask would fall back to the slow ``math`` backend).
The tokenizer is frame-major (``VideoLatentPatchifier.patchify``:
``b c (f) (h) (w) -> b (f h w) (c)``), so ``S = H*W`` is contiguous and frames
stride by ``S``.  A single 3D RoPE application (already done upstream) is
compatible: the temporal component is constant within a frame and the spatial
component constant across frames, so each branch sees only its own rotation.

Only the *unmasked self-attention* path is factorized.  Cross-attention
(``k``/``v`` length differs), audio self-attention (token count != F*S), and the
masked path all delegate to the stock :class:`PytorchAttention`.  Set the grid
per stage with :func:`set_video_grid` before calling the transformer.

Enabled from the harness with ``LTX_ATTN_PATTERN=factorized`` (default ``full``)
and ``LTX_ATTN_COMBINE=mean|sum``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ltx_core.model.transformer.attention import PytorchAttention

_VIDEO_GRID: tuple[int, int] | None = None


def set_video_grid(frames: int, spatial: int) -> None:
    """Set the active video token grid ``(F, S)``; ``S`` tokens are contiguous."""
    global _VIDEO_GRID
    _VIDEO_GRID = (int(frames), int(spatial))


def clear_video_grid() -> None:
    global _VIDEO_GRID
    _VIDEO_GRID = None


class FactorizedSelfAttention:
    """``AttentionCallable`` that factorizes video self-attention into F + S."""

    def __init__(self, combine: str = "mean", fallback: PytorchAttention | None = None) -> None:
        if combine not in ("mean", "sum", "spatial", "temporal"):
            raise ValueError(f"combine must be mean|sum|spatial|temporal, got {combine!r}")
        self._combine = combine
        self._fallback = fallback if fallback is not None else PytorchAttention()

    @property
    def label(self) -> str:
        return f"Factorized(F+S,{self._combine})"

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        grid = _VIDEO_GRID
        if mask is not None or grid is None or q.shape[1] != k.shape[1]:
            return self._fallback(q, k, v, heads, mask)
        frames, spatial = grid
        if q.shape[1] != frames * spatial:
            return self._fallback(q, k, v, heads)
        return self._factorized(q, k, v, heads, frames, spatial)

    def _factorized(self, q, k, v, heads, frames, spatial):
        b, n, hd = q.shape
        dh = hd // heads
        assert n == frames * spatial
        qr = q.view(b, frames, spatial, heads, dh)
        kr = k.view(b, frames, spatial, heads, dh)
        vr = v.view(b, frames, spatial, heads, dh)

        # Spatial: each frame is an independent batch element.
        qs = qr.permute(0, 1, 3, 2, 4).reshape(b * frames, heads, spatial, dh)
        ks = kr.permute(0, 1, 3, 2, 4).reshape(b * frames, heads, spatial, dh)
        vs = vr.permute(0, 1, 3, 2, 4).reshape(b * frames, heads, spatial, dh)
        os_ = F.scaled_dot_product_attention(qs, ks, vs)
        spatial_out = os_.view(b, frames, heads, spatial, dh).permute(0, 1, 3, 2, 4).reshape(b, n, hd)

        # Temporal: each spatial position is an independent batch element.
        qt = qr.permute(0, 2, 3, 1, 4).reshape(b * spatial, heads, frames, dh)
        kt = kr.permute(0, 2, 3, 1, 4).reshape(b * spatial, heads, frames, dh)
        vt = vr.permute(0, 2, 3, 1, 4).reshape(b * spatial, heads, frames, dh)
        ot_ = F.scaled_dot_product_attention(qt, kt, vt)
        temporal_out = ot_.view(b, spatial, heads, frames, dh).permute(0, 3, 1, 2, 4).reshape(b, n, hd)

        if self._combine == "spatial":
            return spatial_out
        if self._combine == "temporal":
            return temporal_out
        if self._combine == "mean":
            return (spatial_out + temporal_out) * 0.5
        return spatial_out + temporal_out


def build_ops(*, combine: str = "mean", fallback: PytorchAttention | None = None):
    """Return a ``TransformerOpsConfig`` with the factorized self-attention.

    Every other op (ada_zero, post_sa, preattention, gated attention) keeps the
    stock default, so this is a drop-in swap for the attention callable.
    """
    from ltx_core.model.transformer.transformer import TransformerOpsConfig

    return TransformerOpsConfig.from_functions(
        attention=FactorizedSelfAttention(combine=combine, fallback=fallback)
    )


def make_configurator(*, combine: str = "mean"):
    """Build a ``LTXModelConfigurator`` subclass that installs the factorized ops."""
    from ltx_core.model.transformer.model_configurator import LTXModelConfigurator
    from ltx_core.model.transformer.transformer import DEFAULT_TRANSFORMER_OPS, TransformerOpsConfig

    ops = build_ops(combine=combine)

    class _FactorizedConfigurator(LTXModelConfigurator):
        @classmethod
        def from_metadata(cls, metadata, ops: TransformerOpsConfig = DEFAULT_TRANSFORMER_OPS):
            return super().from_metadata(metadata, ops=_FactorizedConfigurator._factorized_ops)

    _FactorizedConfigurator._factorized_ops = ops
    return _FactorizedConfigurator

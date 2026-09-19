"""FP8 weight storage for the Gemma text encoder (upcast to bf16 at forward).

Mirrors ``ltx_core.quantization.fp8_cast`` for the LTX transformer, adapted to
the Gemma module tree. Gemma-3-12B bf16 is ~23 GB; storing the language-model
Linear weights in fp8 halves the pinned/streamed bytes (~20 GB -> ~10 GB) and
the H2D traffic. Compute stays bf16 (each Fp8CastLinear upcasts per forward),
so this is a transfer/memory optimization, not a matmul change.
"""

from __future__ import annotations

import torch
from torch import nn

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.sd_ops import SDOps, KeyValueOperationResult
from ltx_core.quantization.fp8_cast import _replace_fwd_with_upcast

# Language-model Linears that are fp8-cast. Names are the module path suffixes;
# the downcast sd_op matches the same suffix on the renamed checkpoint key.
_FP8_GEMMA_SUFFIXES: tuple[str, ...] = (
    ".self_attn.q_proj",
    ".self_attn.k_proj",
    ".self_attn.v_proj",
    ".self_attn.o_proj",
    ".mlp.gate_proj",
    ".mlp.up_proj",
    ".mlp.down_proj",
)

_LAYER_MARK = ".language_model.layers."


def _is_gemma_fp8_linear(module_name: str) -> bool:
    if _LAYER_MARK not in module_name:
        return False
    return any(module_name.endswith(suffix) for suffix in _FP8_GEMMA_SUFFIXES)


def amend_gemma_fp8(model: nn.Module) -> nn.Module:
    """Retype matching Gemma Linears to fp8 storage + upcast forward."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _is_gemma_fp8_linear(name):
            _replace_fwd_with_upcast(module, with_stochastic_rounding=False)
    return model


GEMMA_FP8_UPCAST = ModuleOps(
    name="gemma_fp8_upcast_during_linear_forward",
    matcher=lambda model: True,
    mutator=amend_gemma_fp8,
)


def _downcast(key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
    return [KeyValueOperationResult(key, value.to(dtype=torch.float8_e4m3fn))]


def gemma_downcast_sd_ops() -> SDOps:
    ops = SDOps("GEMMA_LINEAR_DOWNCAST")
    for suffix in _FP8_GEMMA_SUFFIXES:
        ops = ops.with_kv_operation(key_suffix=suffix + ".weight", operation=_downcast)
        ops = ops.with_kv_operation(key_suffix=suffix + ".bias", operation=_downcast)
    return ops


def with_gemma_fp8(
    sd_ops: SDOps, module_ops: tuple[ModuleOps, ...]
) -> tuple[SDOps, tuple[ModuleOps, ...]]:
    """Return ``(sd_ops, module_ops)`` with Gemma fp8 cast/downcast appended."""
    downcast = gemma_downcast_sd_ops()
    merged = SDOps(
        name=f"{sd_ops.name}+gemma_fp8",
        mapping=(*sd_ops.mapping, *downcast.mapping),
        allowed_keys=sd_ops.allowed_keys,
    )
    if any(op.name == GEMMA_FP8_UPCAST.name for op in module_ops):
        return merged, module_ops
    return merged, (*module_ops, GEMMA_FP8_UPCAST)

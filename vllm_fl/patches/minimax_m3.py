# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Route missing vLLM MiniMax-M3 operators through FlagOS dispatch."""

from __future__ import annotations

import logging

import torch

from vllm_fl.dispatch import CachedOp

logger = logging.getLogger(__name__)

_OP_NAME = "fused_minimax_m3_qknorm_rope_kv_insert"
_dispatch_op = CachedOp(_OP_NAME)


def _dispatch_fused_preprocess(
    qkv: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    index_q_norm_weight: torch.Tensor | None = None,
    index_k_norm_weight: torch.Tensor | None = None,
    num_index_heads: int = 0,
    slot_mapping: torch.Tensor | None = None,
    index_slot_mapping: torch.Tensor | None = None,
    kv_cache: torch.Tensor | None = None,
    index_cache: torch.Tensor | None = None,
    block_size: int = 0,
    q_out: torch.Tensor | None = None,
    index_q_out: torch.Tensor | None = None,
    kv_cache_dtype: str = "auto",
) -> None:
    """Preserve vLLM's wrapper contract and enter the Plugin dispatcher."""
    return _dispatch_op(
        qkv,
        q_norm_weight,
        k_norm_weight,
        cos_sin_cache,
        positions,
        num_heads,
        num_kv_heads,
        rotary_dim,
        eps,
        index_q_norm_weight,
        index_k_norm_weight,
        num_index_heads,
        slot_mapping,
        index_slot_mapping,
        kv_cache,
        index_cache,
        block_size,
        q_out,
        index_q_out,
        kv_cache_dtype,
    )


def patch_minimax_m3_fused_preprocess() -> bool:
    """Patch the Python call site only when vLLM's native op is absent.

    MiniMax-M3's NVIDIA and AMD modules both retain a reference to the
    :mod:`vllm._custom_ops` module. Replacing this one model-specific wrapper
    therefore reaches both callers without editing vLLM source or registering
    a synthetic operator in vLLM's private ``torch.ops._C`` namespace.
    """
    from vllm import _custom_ops as ops

    current = getattr(ops, _OP_NAME, None)
    if current is None:
        logger.debug("vLLM has no Python wrapper for %s", _OP_NAME)
        return False
    if getattr(current, "_vllm_fl_dispatch_patch", False):
        return False
    if hasattr(torch.ops._C, _OP_NAME):
        logger.debug("vLLM native %s is available", _OP_NAME)
        return False

    _dispatch_fused_preprocess._vllm_fl_dispatch_patch = True
    _dispatch_fused_preprocess._vllm_fl_original = current
    setattr(ops, _OP_NAME, _dispatch_fused_preprocess)
    logger.info("Patched vLLM %s to use FlagOS dispatch", _OP_NAME)
    return True


__all__ = ["patch_minimax_m3_fused_preprocess"]

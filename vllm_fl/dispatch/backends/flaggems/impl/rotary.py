# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems rotary embedding operator implementations.
"""

from __future__ import annotations

import torch


def rotary_embedding_flaggems(
    obj,
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: torch.Tensor,
    rotary_interleaved: bool = False,
    inplace: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding using FlagGems.

    Args:
        obj: The calling obj (for interface consistency)
        query: Query tensor
        key: Key tensor
        cos: Cosine cache
        sin: Sine cache
        position_ids: Position indices
        rotary_interleaved: Whether to use interleaved rotary
        inplace: Whether to modify tensors in-place

    Returns:
        Tuple of (embedded_query, embedded_key)
    """
    from flag_gems.modules.rotary_embedding import gems_rope_forward

    return gems_rope_forward(
        query,
        key,
        cos,
        sin,
        position_ids=position_ids,
        rotary_interleaved=rotary_interleaved,
        inplace=inplace,
    )


def apply_rotary_emb_flaggems(
    obj,
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Adapt vLLM's standalone/partial RoPE contract to FlagGems-vLLM."""
    if cos.shape != sin.shape:
        raise ValueError("cos and sin must have identical shapes")
    rotary_dim = cos.shape[-1] * 2
    if rotary_dim > x.shape[-1]:
        raise ValueError("rotary dimension cannot exceed the input head size")
    if x.ndim not in (3, 4):
        raise ValueError("x must be [seq, heads, dim] or [batch, seq, heads, dim]")

    try:
        from flaggems_vllm import apply_rotary_pos_emb
    except ImportError:
        from flag_gems.modules.rotary_embedding import (
            gems_rope_forward as apply_rotary_pos_emb,
        )

    origin_shape = x.shape
    origin_dtype = x.dtype
    if x.ndim == 3:
        x = x.unsqueeze(0)
    work = x.float() if obj.enable_fp32_compute else x
    x_rot = work[..., :rotary_dim].contiguous()
    x_pass = work[..., rotary_dim:]

    # The public FlagGems-vLLM kernel accepts Q and K together. Reusing the
    # same rotary slice for both preserves its cross-platform launch/runtime
    # handling; only the first result is consumed here.
    rotated, _ = apply_rotary_pos_emb(
        x_rot,
        x_rot,
        cos,
        sin,
        position_ids=None,
        rotary_interleaved=not obj.is_neox_style,
        inplace=False,
    )
    output = (
        torch.cat((rotated, x_pass), dim=-1)
        if rotary_dim < work.shape[-1]
        else rotated
    )
    if len(origin_shape) == 3:
        output = output.squeeze(0)
    if obj.enable_fp32_compute:
        output = output.to(origin_dtype)
    return output

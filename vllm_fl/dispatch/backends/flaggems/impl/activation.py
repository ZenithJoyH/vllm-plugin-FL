# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems activation operator implementations.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _swigluoai_uninterleave_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    output_width: tl.constexpr,
    clamp_limit: tl.constexpr,
    alpha: tl.constexpr,
    beta: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    row = offsets // output_width
    column = offsets % output_width
    input_offsets = row * (2 * output_width) + column

    gate = tl.load(input_ptr + input_offsets, mask=mask).to(tl.float32)
    up = tl.load(
        input_ptr + input_offsets + output_width,
        mask=mask,
    ).to(tl.float32)
    gate = tl.minimum(gate, clamp_limit)
    up = tl.maximum(tl.minimum(up, clamp_limit), -clamp_limit)
    value = gate / (1.0 + tl.exp(-alpha * gate)) * (up + beta)
    tl.store(output_ptr + offsets, value, mask=mask)


def swigluoai_uninterleave_flaggems(
    output: torch.Tensor,
    input: torch.Tensor,
    *,
    clamp_limit: float,
    alpha: float,
    beta: float,
) -> None:
    """Plugin-owned FlagGems fallback for the missing standalone operator."""
    if not input.is_contiguous() or not output.is_contiguous():
        raise ValueError("input and output must be contiguous")
    if input.device != output.device or input.device.type == "cpu":
        raise ValueError("input and output must be on the same accelerator")
    if input.dtype != output.dtype:
        raise ValueError("input and output must have the same dtype")
    if input.ndim != 2 or output.ndim != 2:
        raise ValueError("input and output must be 2D")
    if input.shape[0] != output.shape[0] or input.shape[1] != 2 * output.shape[1]:
        raise ValueError("input width must be twice the output width")
    if output.numel() == 0:
        return

    block_size = 256
    n_elements = output.numel()
    _swigluoai_uninterleave_kernel[(triton.cdiv(n_elements, block_size),)](
        input,
        output,
        n_elements,
        output.size(-1),
        clamp_limit,
        alpha,
        beta,
        BLOCK_SIZE=block_size,
    )


def silu_and_mul_flaggems(obj, x: torch.Tensor) -> torch.Tensor:
    """
    SiLU activation followed by element-wise multiplication using FlagGems.

    Args:
        obj: The calling obj (for interface consistency)
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    from flag_gems.modules.activation import gems_silu_and_mul

    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return gems_silu_and_mul(x1, x2)


def gelu_and_mul_flaggems(obj, x: torch.Tensor) -> torch.Tensor:
    """
    GELU activation followed by element-wise multiplication using FlagGems.

    Args:
        obj: The calling obj (for interface consistency)
        x: Input tensor of shape [..., 2*d]

    Returns:
        Output tensor of shape [..., d]
    """
    from flag_gems.fused import gelu_and_mul

    approximate = getattr(obj, "approximate", "none") if obj is not None else "none"
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return gelu_and_mul(x1, x2, approximate)

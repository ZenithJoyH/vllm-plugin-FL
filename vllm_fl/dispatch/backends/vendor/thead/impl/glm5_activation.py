# SPDX-License-Identifier: Apache-2.0
"""PPU graph-safe wrapper for FlagGems bounded SwiGLU."""

from __future__ import annotations

import math

import torch

from flag_gems.fused.silu_and_mul_with_clamp import (
    silu_and_mul_with_clamp_kernel,
)


_LIMIT_TENSOR_CACHE: dict[tuple[torch.device, torch.dtype, float], torch.Tensor] = {}


def _get_limit_tensor(x: torch.Tensor, limit: float) -> torch.Tensor:
    key = (x.device, x.dtype, float(limit))
    cached = _LIMIT_TENSOR_CACHE.get(key)
    if cached is not None:
        return cached
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "PPU bounded SwiGLU limit tensor must be initialized by eager warmup "
            "before graph capture"
        )
    cached = torch.tensor(float(limit), device=x.device, dtype=x.dtype)
    _LIMIT_TENSOR_CACHE[key] = cached
    return cached


def silu_and_mul_with_clamp(
    x: torch.Tensor,
    y: torch.Tensor,
    limit: float,
) -> torch.Tensor:
    """Reuse FlagGems' Triton kernel with a stable, prewarmed limit tensor."""

    _validate_inputs(x, y, limit)
    limit_tensor = _get_limit_tensor(x, float(limit))
    return silu_and_mul_with_clamp_kernel(x, y, limit_tensor)


def silu_and_mul_with_clamp_out(
    x: torch.Tensor,
    y: torch.Tensor,
    out: torch.Tensor,
    limit: float,
) -> torch.Tensor:
    """Graph-safe out variant used by the plugin-owned fused MoE path."""

    _validate_inputs(x, y, limit)
    if out.device != x.device or out.dtype != x.dtype or out.shape != x.shape:
        raise ValueError(
            "PPU bounded SwiGLU out must match x device, dtype, and shape; "
            f"got out={out.shape}/{out.dtype}/{out.device}, "
            f"x={x.shape}/{x.dtype}/{x.device}"
        )
    limit_tensor = _get_limit_tensor(x, float(limit))
    silu_and_mul_with_clamp_kernel(x, y, limit_tensor, out0=out)
    return out


def _validate_inputs(x: torch.Tensor, y: torch.Tensor, limit: float) -> None:
    if x.device.type != "cuda" or y.device != x.device:
        raise ValueError("PPU bounded SwiGLU requires x/y on the same CUDA-like device")
    if x.dtype != torch.bfloat16 or y.dtype != x.dtype:
        raise ValueError("PPU bounded SwiGLU currently requires BF16 x/y")
    if x.shape != y.shape:
        raise ValueError(
            f"PPU bounded SwiGLU requires equal x/y shapes, got {x.shape} and {y.shape}"
        )
    if not math.isfinite(float(limit)) or not float(limit) > 0.0:
        raise ValueError(
            f"PPU bounded SwiGLU requires a positive finite limit, got {limit}"
        )


__all__ = ["silu_and_mul_with_clamp", "silu_and_mul_with_clamp_out"]

# SPDX-License-Identifier: Apache-2.0
"""Graph-safe bounded SwiGLU binding to FlagGems, with stable scalar storage."""

import math

import torch

_LIMIT_TENSORS: dict[tuple[torch.device, torch.dtype, float], torch.Tensor] = {}


def supports(x, limit, alpha=1.0, beta=0.0):
    return (
        x.device.type == "cuda"
        and x.dtype == torch.bfloat16
        and x.shape[-1] % 2 == 0
        and alpha == 1.0
        and beta == 0.0
        and math.isfinite(float(limit))
        and limit > 0
    )


def silu_and_mul_with_clamp(x, limit, alpha=1.0, beta=0.0):
    from flag_gems.fused.silu_and_mul_with_clamp import silu_and_mul_with_clamp_kernel

    if not supports(x, limit, alpha, beta):
        raise ValueError("Unsupported bounded activation inputs")
    key = (x.device, x.dtype, float(limit))
    scalar = _LIMIT_TENSORS.get(key)
    if scalar is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Bounded SwiGLU needs eager warmup before graph capture")
        scalar = torch.tensor(float(limit), device=x.device, dtype=x.dtype)
        _LIMIT_TENSORS[key] = scalar
    gate, up = x.chunk(2, dim=-1)
    return silu_and_mul_with_clamp_kernel(gate, up, scalar)

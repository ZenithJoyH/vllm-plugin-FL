# Copyright (c) 2026 BAAI. All rights reserved.
"""Pure PyTorch references for manifold-constrained hyper-connections (mHC).

Accumulate in FP32, preserve the residual/output activation dtype, and support
arbitrary leading batch dimensions. No Triton or FlagGems calls live here.
"""

import torch
import torch.nn.functional as F


def sinkhorn_normalize(comb: torch.Tensor, eps: float, num_iters: int) -> torch.Tensor:
    """Row softmax then alternating column/row normalization in FP32."""
    if num_iters < 1:
        raise ValueError("mHC requires at least one Sinkhorn iteration")
    comb = torch.softmax(comb.float(), dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(num_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return comb


def mhc_pre_reference(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return post (..., H, 1), combination (..., H, H), input (..., C).

    The affine order is normalized projection * scale + bias, not
    (normalized projection + bias) * scale. n_splits is a kernel scheduling hint
    and does not affect this unpartitioned reference calculation.
    """
    if residual.ndim < 2 or min(residual.shape[-2:]) < 1:
        raise ValueError("residual must have shape (..., H, C) with H,C > 0")
    if residual.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("mHC reference activations require FP16, BF16 or FP32")
    h, c = residual.shape[-2:]
    size = h * h + 2 * h
    if fn.shape != (size, h * c) or hc_scale.shape != (3,) or hc_base.shape != (size,):
        raise ValueError("Invalid mHC projection, scale or bias shape")
    if any(t.dtype != torch.float32 for t in (fn, hc_scale, hc_base)):
        raise TypeError("mHC projection, scale and bias must be FP32")
    if any(t.device != residual.device for t in (fn, hc_scale, hc_base)):
        raise ValueError("All mHC inputs must be on the same device")
    if n_splits < 1:
        raise ValueError("n_splits must be positive")

    residual_fp32 = residual.float()
    flat = residual_fp32.flatten(start_dim=-2)
    mixes = F.linear(flat, fn) * torch.rsqrt(
        flat.square().mean(dim=-1, keepdim=True) + rms_eps
    )
    pre = torch.sigmoid(mixes[..., :h] * hc_scale[0] + hc_base[:h]) + hc_pre_eps
    post = torch.sigmoid(mixes[..., h : 2 * h] * hc_scale[1] + hc_base[h : 2 * h])
    post = (post * hc_post_mult_value).unsqueeze(-1)
    comb = (mixes[..., 2 * h :] * hc_scale[2] + hc_base[2 * h :]).unflatten(-1, (h, h))
    comb = sinkhorn_normalize(comb, hc_sinkhorn_eps, sinkhorn_repeat)
    layer_input = (pre.unsqueeze(-1) * residual_fp32).sum(dim=-2).to(residual.dtype)
    return post, comb, layer_input


def mhc_post_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor:
    """Combine streams using the backend ABI: comb[..., source, destination].

    XingChen's model wrapper transposes its mixing matrix before entering this
    backend API. Preserve this transpose here to match the FlagGems/CUDA ABI.
    torch.matmul supports unbatched and multi-batch tensors as well as strides.
    """
    output = torch.matmul(comb.float().transpose(-2, -1), residual.float())
    output = output + x.float().unsqueeze(-2) * post.float()
    return output.to(x.dtype)

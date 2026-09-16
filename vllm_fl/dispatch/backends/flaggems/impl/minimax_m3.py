# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Portable Triton kernels for MiniMax-M3 attention preprocessing.

The implementation preserves vLLM's public custom-op contract while keeping
missing FlagGems functionality in the Plugin's default FlagOS backend. It
supports the BF16/FP16 ``auto`` cache path used by MiniMax-M3; unsupported FP8
storage is rejected rather than silently changing cache semantics.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from flag_gems.runtime import torch_device_fn

_HEAD_DIM = 128


@triton.jit
def _gemma_norm_rope_kernel(
    x_ptr,
    weight_ptr,
    cos_sin_ptr,
    positions_ptr,
    out_ptr,
    x_stride_token: tl.constexpr,
    x_stride_dim: tl.constexpr,
    out_stride_token: tl.constexpr,
    out_stride_dim: tl.constexpr,
    x_head_offset: tl.constexpr,
    out_head_offset: tl.constexpr,
    num_heads: tl.constexpr,
    rotary_dim: tl.constexpr,
    cos_stride_pos: tl.constexpr,
    cos_stride_dim: tl.constexpr,
    eps: tl.constexpr,
):
    program = tl.program_id(0)
    token = program // num_heads
    head = program - token * num_heads
    dim = tl.arange(0, 128)
    x_offset = token * x_stride_token + (
        x_head_offset + head * 128 + dim
    ) * x_stride_dim

    x = tl.load(x_ptr + x_offset).to(tl.float32)
    weight = tl.load(weight_ptr + dim).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / 128.0 + eps)
    normed = x * inv_rms * (1.0 + weight)

    half_rotary = rotary_dim // 2
    peer_dim = tl.where(dim < half_rotary, dim + half_rotary, dim - half_rotary)
    peer_offset = token * x_stride_token + (
        x_head_offset + head * 128 + peer_dim
    ) * x_stride_dim
    peer_x = tl.load(x_ptr + peer_offset).to(tl.float32)
    peer_weight = tl.load(weight_ptr + peer_dim).to(tl.float32)
    peer_normed = peer_x * inv_rms * (1.0 + peer_weight)

    position = tl.load(positions_ptr + token)
    rotary_offset = dim % half_rotary
    cos = tl.load(
        cos_sin_ptr
        + position * cos_stride_pos
        + rotary_offset * cos_stride_dim
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_ptr
        + position * cos_stride_pos
        + (half_rotary + rotary_offset) * cos_stride_dim
    ).to(tl.float32)
    rotation_sign = tl.where(dim < half_rotary, -1.0, 1.0)
    roped = normed * cos + rotation_sign * peer_normed * sin
    result = tl.where(dim < rotary_dim, roped, normed)

    out_offset = token * out_stride_token + (
        out_head_offset + head * 128 + dim
    ) * out_stride_dim
    tl.store(out_ptr + out_offset, result)


@triton.jit
def _gemma_norm_kernel(
    x_ptr,
    weight_ptr,
    out_ptr,
    x_stride_token: tl.constexpr,
    x_stride_dim: tl.constexpr,
    out_stride_token: tl.constexpr,
    out_stride_dim: tl.constexpr,
    x_head_offset: tl.constexpr,
    out_head_offset: tl.constexpr,
    eps: tl.constexpr,
):
    head = tl.program_id(0)
    token = tl.program_id(1)
    dim = tl.arange(0, 128)
    x_offset = token * x_stride_token + (
        x_head_offset + head * 128 + dim
    ) * x_stride_dim
    x = tl.load(x_ptr + x_offset).to(tl.float32)
    weight = tl.load(weight_ptr + dim).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / 128.0 + eps)
    normed = x * inv_rms * (1.0 + weight)
    out_offset = token * out_stride_token + (
        out_head_offset + head * 128 + dim
    ) * out_stride_dim
    tl.store(out_ptr + out_offset, normed)


@triton.jit
def _rope_64_kernel(
    x_ptr,
    cos_sin_ptr,
    positions_ptr,
    x_stride_token: tl.constexpr,
    x_stride_dim: tl.constexpr,
    head_offset: tl.constexpr,
    cos_stride_pos: tl.constexpr,
    cos_stride_dim: tl.constexpr,
):
    head = tl.program_id(0)
    token = tl.program_id(1)
    dim = tl.arange(0, 32)
    base = token * x_stride_token + (head_offset + head * 128) * x_stride_dim
    first = tl.load(x_ptr + base + dim * x_stride_dim).to(tl.float32)
    second = tl.load(x_ptr + base + (32 + dim) * x_stride_dim).to(tl.float32)
    position = tl.load(positions_ptr + token)
    cos = tl.load(cos_sin_ptr + position * cos_stride_pos + dim * cos_stride_dim)
    sin = tl.load(
        cos_sin_ptr + position * cos_stride_pos + (32 + dim) * cos_stride_dim
    )
    tl.store(x_ptr + base + dim * x_stride_dim, first * cos - second * sin)
    tl.store(x_ptr + base + (32 + dim) * x_stride_dim, second * cos + first * sin)


@triton.jit
def _kv_cache_insert_kernel(
    qkv_ptr,
    slot_mapping_ptr,
    kv_cache_ptr,
    qkv_stride_token: tl.constexpr,
    qkv_stride_dim: tl.constexpr,
    kv_stride_block: tl.constexpr,
    kv_stride_kv: tl.constexpr,
    kv_stride_token: tl.constexpr,
    kv_stride_head: tl.constexpr,
    kv_stride_dim: tl.constexpr,
    k_offset: tl.constexpr,
    v_offset: tl.constexpr,
    num_kv_heads: tl.constexpr,
    block_size: tl.constexpr,
):
    head = tl.program_id(0)
    token = tl.program_id(1)
    dim = tl.arange(0, 128)
    slot = tl.load(slot_mapping_ptr + token)
    valid = slot >= 0
    block = slot // block_size
    block_offset = slot - block * block_size

    row = token * qkv_stride_token
    k = tl.load(
        qkv_ptr + row + (k_offset + head * 128 + dim) * qkv_stride_dim
    )
    v = tl.load(
        qkv_ptr + row + (v_offset + head * 128 + dim) * qkv_stride_dim
    )
    cache_base = (
        block * kv_stride_block
        + block_offset * kv_stride_token
        + head * kv_stride_head
        + dim * kv_stride_dim
    )
    tl.store(kv_cache_ptr + cache_base, k, mask=valid)
    tl.store(kv_cache_ptr + cache_base + kv_stride_kv, v, mask=valid)


@triton.jit
def _index_cache_insert_kernel(
    qkv_ptr,
    slot_mapping_ptr,
    index_cache_ptr,
    qkv_stride_token: tl.constexpr,
    qkv_stride_dim: tl.constexpr,
    cache_stride_block: tl.constexpr,
    cache_stride_token: tl.constexpr,
    cache_stride_dim: tl.constexpr,
    index_k_offset: tl.constexpr,
    block_size: tl.constexpr,
):
    token = tl.program_id(0)
    dim = tl.arange(0, 128)
    slot = tl.load(slot_mapping_ptr + token)
    valid = slot >= 0
    block = slot // block_size
    block_offset = slot - block * block_size
    value = tl.load(
        qkv_ptr
        + token * qkv_stride_token
        + (index_k_offset + dim) * qkv_stride_dim
    )
    cache_offset = (
        block * cache_stride_block
        + block_offset * cache_stride_token
        + dim * cache_stride_dim
    )
    tl.store(index_cache_ptr + cache_offset, value, mask=valid)


def _check_tensor(
    name: str,
    tensor: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tensor.device.type == "cpu" or tensor.device != device:
        raise ValueError(f"{name} must be on the same accelerator as qkv")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} dtype must match qkv ({dtype})")


def fused_minimax_m3_qknorm_rope_kv_insert_flaggems(
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
    """Apply MiniMax-M3 Gemma QK norm, partial NeoX RoPE, and cache writes."""
    if qkv.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("qkv must be float16 or bfloat16")
    if qkv.device.type == "cpu" or not qkv.is_contiguous():
        raise ValueError("qkv must be a contiguous accelerator tensor")
    if positions.dtype != torch.int64 or positions.device != qkv.device:
        raise ValueError("positions must be an int64 tensor on the qkv device")
    if positions.numel() != qkv.shape[0]:
        raise ValueError("positions length must match qkv token count")
    if rotary_dim != 64:
        raise ValueError("MiniMax-M3 preprocessing requires rotary_dim=64")
    if kv_cache_dtype != "auto":
        raise NotImplementedError(
            "MiniMax-M3 FlagGems preprocessing currently supports only "
            "kv_cache_dtype='auto'"
        )

    dtype = qkv.dtype
    for name, tensor in (
        ("q_norm_weight", q_norm_weight),
        ("k_norm_weight", k_norm_weight),
        ("cos_sin_cache", cos_sin_cache),
    ):
        _check_tensor(name, tensor, dtype, qkv.device)
    if q_norm_weight.numel() != _HEAD_DIM or k_norm_weight.numel() != _HEAD_DIM:
        raise ValueError("q/k norm weights must each have 128 elements")
    if not cos_sin_cache.is_contiguous() or cos_sin_cache.shape[1] != rotary_dim:
        raise ValueError("cos_sin_cache must be contiguous [max_pos, rotary_dim]")

    has_index = num_index_heads > 0
    expected_heads = num_heads + 2 * num_kv_heads
    if has_index:
        expected_heads += num_index_heads + 1
    if qkv.ndim != 2 or qkv.shape[1] != expected_heads * _HEAD_DIM:
        raise ValueError("qkv packed width does not match the configured head counts")
    if qkv.shape[0] == 0:
        return

    if has_index:
        if index_q_norm_weight is None or index_k_norm_weight is None:
            raise ValueError("index branch requires both index norm weights")
        _check_tensor("index_q_norm_weight", index_q_norm_weight, dtype, qkv.device)
        _check_tensor("index_k_norm_weight", index_k_norm_weight, dtype, qkv.device)
        if (
            index_q_norm_weight.numel() != _HEAD_DIM
            or index_k_norm_weight.numel() != _HEAD_DIM
        ):
            raise ValueError("index norm weights must each have 128 elements")

    q_size = num_heads * _HEAD_DIM
    kv_size = num_kv_heads * _HEAD_DIM
    index_q_size = num_index_heads * _HEAD_DIM
    launches = [
        (0, q_norm_weight, num_heads, q_out, 0),
        (q_size, k_norm_weight, num_kv_heads, None, q_size),
    ]
    if has_index:
        index_q_offset = q_size + 2 * kv_size
        index_k_offset = index_q_offset + index_q_size
        launches.extend(
            [
                (
                    index_q_offset,
                    index_q_norm_weight,
                    num_index_heads,
                    index_q_out,
                    0,
                ),
                (index_k_offset, index_k_norm_weight, 1, None, index_k_offset),
            ]
        )

    with torch_device_fn.device(qkv.device):
        for x_offset, weight, heads, external_out, in_place_offset in launches:
            out = qkv if external_out is None else external_out
            out_offset = in_place_offset if external_out is None else 0
            if external_out is not None:
                _check_tensor("output", external_out, dtype, qkv.device)
                if not external_out.is_contiguous():
                    raise ValueError("external outputs must be contiguous")
            _gemma_norm_kernel[(heads, qkv.shape[0])](
                qkv,
                weight,
                out,
                qkv.stride(0),
                qkv.stride(1),
                out.stride(0),
                out.stride(1),
                x_offset,
                out_offset,
                eps,
                num_warps=4,
            )
            _rope_64_kernel[(heads, qkv.shape[0])](
                out,
                cos_sin_cache,
                positions,
                out.stride(0),
                out.stride(1),
                out_offset,
                cos_sin_cache.stride(0),
                cos_sin_cache.stride(1),
                num_warps=1,
            )

        if kv_cache is None:
            return
        if not has_index:
            raise ValueError("KV insertion requires the sparse index branch")
        if slot_mapping is None or index_cache is None:
            raise ValueError("KV insertion requires slot_mapping and index_cache")
        if index_slot_mapping is None:
            index_slot_mapping = slot_mapping
        if block_size <= 0 or kv_cache.ndim != 5 or index_cache.ndim != 3:
            raise ValueError("invalid paged-cache shape or block_size")
        if kv_cache.shape[2] != block_size or index_cache.shape[1] != block_size:
            raise ValueError("block_size must match both cache tensors")
        _check_tensor("kv_cache", kv_cache, dtype, qkv.device)
        _check_tensor("index_cache", index_cache, dtype, qkv.device)
        for name, mapping in (
            ("slot_mapping", slot_mapping),
            ("index_slot_mapping", index_slot_mapping),
        ):
            if mapping.device != qkv.device or mapping.dtype != torch.int64:
                raise ValueError(f"{name} must be an int64 tensor on the qkv device")
            if mapping.numel() != qkv.shape[0]:
                raise ValueError(f"{name} length must match qkv token count")

        _kv_cache_insert_kernel[(num_kv_heads, qkv.shape[0])](
            qkv,
            slot_mapping,
            kv_cache,
            qkv.stride(0),
            qkv.stride(1),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            kv_cache.stride(3),
            kv_cache.stride(4),
            q_size,
            q_size + kv_size,
            num_kv_heads,
            block_size,
            num_warps=4,
        )
        _index_cache_insert_kernel[(qkv.shape[0],)](
            qkv,
            index_slot_mapping,
            index_cache,
            qkv.stride(0),
            qkv.stride(1),
            index_cache.stride(0),
            index_cache.stride(1),
            index_cache.stride(2),
            index_k_offset,
            block_size,
            num_warps=4,
        )

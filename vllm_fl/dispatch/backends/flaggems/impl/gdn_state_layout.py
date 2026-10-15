# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.

"""Paged V-major to compact K-major state conversion for FlagOS GDN.

Keep the external cache's strides and storage unchanged: hybrid attention
may interleave/pad pages, so its leading stride need not equal H * V * K.
Only active states are gathered; neither conversion copies the full cache.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _transfer_state(
    cache, compact, indices,
    S0: tl.constexpr, S1: tl.constexpr, S2: tl.constexpr, S3: tl.constexpr,
    H: tl.constexpr, V: tl.constexpr, K: tl.constexpr,
    INDEX_STRIDE: tl.constexpr, PACK: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(1)
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offset < H * K * V
    # Compact row 0 is a private zero/sentinel state. Original slot 0 is
    # vLLM's NULL_BLOCK_ID and must never be written, including padded rows.
    slot = tl.load(indices + (row - 1) * INDEX_STRIDE, mask=row > 0, other=0)
    valid = (row > 0) & (slot > 0)
    h = offset // (K * V)
    k = offset // V % K
    v = offset % V
    cache_offset = slot.to(tl.int64) * S0 + h * S1 + v * S2 + k * S3
    compact_offset = row.to(tl.int64) * H * K * V + offset
    if PACK:
        value = tl.load(cache + cache_offset, mask=mask & valid, other=0)
        tl.store(compact + compact_offset, value, mask=mask)
    else:
        value = tl.load(compact + compact_offset, mask=mask & valid, other=0)
        tl.store(cache + cache_offset, value, mask=mask & valid)


def pack_gdn_state(cache: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather V-major pages into private contiguous [N+1,H,K,V] state."""
    _, heads, values, keys = cache.shape
    compact = torch.empty(
        (indices.numel() + 1, heads, keys, values),
        dtype=cache.dtype, device=cache.device,
    )
    _transfer_state[(triton.cdiv(heads * keys * values, 512), indices.numel() + 1)](
        cache, compact, indices, *cache.stride(), heads, values, keys,
        indices.stride(0), True, 512,
    )
    return compact


def unpack_gdn_state(
    compact: torch.Tensor, cache: torch.Tensor, indices: torch.Tensor,
) -> None:
    """Scatter active unique pages, preserving null pages and padding bytes."""
    _, heads, values, keys = cache.shape
    _transfer_state[(triton.cdiv(heads * keys * values, 512), indices.numel() + 1)](
        cache, compact, indices, *cache.stride(), heads, values, keys,
        indices.stride(0), False, 512,
    )

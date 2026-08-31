# SPDX-License-Identifier: Apache-2.0
"""Persist each packed prefill request's unfinished kpool into its own page.

FlagGems 5.3.5 has no compatible raw K+gate tail-ring operation. Its indexer
cache writer quantizes a single K plane and cannot implement this contract.
This plugin-owned operator uses the existing indexer backend dispatch.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _persist_prefill_tail_kernel(
    K,
    G,
    SLOTS,
    CACHE,
    N: tl.constexpr,
    D: tl.constexpr,
    POOL: tl.constexpr,
    BLOCKS: tl.constexpr,
    K0: tl.constexpr,
    K1: tl.constexpr,
    G0: tl.constexpr,
    G1: tl.constexpr,
    S0: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    C2: tl.constexpr,
    C3: tl.constexpr,
    BD: tl.constexpr,
):
    row = tl.program_id(0)
    slot = tl.load(SLOTS + row * S0).to(tl.int64)
    page = slot // POOL
    offset = slot % POOL
    valid = (slot >= 0) & (page < BLOCKS)
    # A complete pool must have its last token in this packed request.
    # Distinct active requests own distinct tail pages, even when positions
    # restart or their prompt/chunk lengths differ.
    completion_row = row + POOL - 1 - offset
    completion_slot = tl.load(
        SLOTS + completion_row * S0,
        mask=valid & (completion_row >= 0) & (completion_row < N),
        other=-1,
    ).to(tl.int64)
    complete = (completion_slot >= 0) & (completion_slot // POOL == page)
    save = valid & ~complete
    dim = tl.arange(0, BD)
    k = tl.load(K + row * K0 + dim * K1, mask=save & (dim < D), other=0)
    gate = tl.load(G + row * G0 + dim * G1, mask=save & (dim < D), other=0)
    dst = CACHE + page * C0 + offset * C2 + dim * C3
    tl.store(dst, k, mask=save & (dim < D))
    tl.store(dst + C1, gate, mask=save & (dim < D))


def persist_prefill_tail(k, gate, slot_mapping, tail_cache, pool_size=4):
    """In-place, allocation-free, graph-safe raw BF16/FP16/FP32 tail write.

    Rows must be packed by request; each request has one unique physical tail
    page. Slots encode page*pool_size + position%pool_size, with -1 padding.
    Completed pools are left untouched; only tokens of each final incomplete
    pool are saved. A partial chunk may retain earlier offsets already cached.
    """
    if k.ndim != 2 or gate.shape != k.shape or slot_mapping.ndim != 1:
        raise ValueError("Expected K/gate [tokens, dim] and slots [tokens]")
    if k.shape[0] != slot_mapping.shape[0] or pool_size != 4 or k.shape[1] != 128:
        raise ValueError("GLM5 tail requires matching rows, pool=4 and dim=128")
    if tail_cache.ndim != 4 or tail_cache.shape[1:] != (2, pool_size, k.shape[1]):
        raise ValueError("Expected tail cache [pages, 2, pool, dim]")
    if (
        any(t.device != k.device for t in (gate, slot_mapping, tail_cache))
        or not k.is_cuda
    ):
        raise ValueError("Tail tensors must share one accelerator device")
    floats = (torch.bfloat16, torch.float16, torch.float32)
    if any(t.dtype not in floats for t in (k, gate, tail_cache)):
        raise TypeError("Raw tail K/gate/cache must be floating point")
    if slot_mapping.dtype not in (torch.int32, torch.int64):
        raise TypeError("Tail slots must be int32 or int64")
    if not k.shape[0]:
        return
    _persist_prefill_tail_kernel[(k.shape[0],)](
        k,
        gate,
        slot_mapping,
        tail_cache,
        k.shape[0],
        k.shape[1],
        pool_size,
        tail_cache.shape[0],
        *k.stride(),
        *gate.stride(),
        slot_mapping.stride(0),
        *tail_cache.stride(),
        triton.next_power_of_2(k.shape[1]),
        num_warps=4,
    )

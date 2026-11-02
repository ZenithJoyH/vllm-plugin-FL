# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Framework-owned sparse token/page mapping and stable slot compaction."""

import torch
import triton
import triton.language as tl


@triton.jit
def _convert_sparse_indices_kernel(
    Requests,
    Blocks,
    Indices,
    Physical,
    Lengths,
    request_stride: tl.constexpr,
    block_row_stride: tl.constexpr,
    block_col_stride: tl.constexpr,
    index_row_stride: tl.constexpr,
    index_col_stride: tl.constexpr,
    N_REQUESTS: tl.constexpr,
    N_BLOCKS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    TILE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    req = tl.load(Requests + row * request_stride).to(tl.int64)
    # Match PyTorch's negative row indexing for otherwise legal request IDs.
    req = tl.where(req < 0, req + N_REQUESTS, req)
    row_valid = (req >= 0) & (req < N_REQUESTS)
    lane = tl.arange(0, TILE)
    count = tl.full((), 0, tl.int32)
    for start in range(tl.cdiv(WIDTH, TILE)):
        col = start * TILE + lane
        token = tl.load(
            Indices + row * index_row_stride + col.to(tl.int64) * index_col_stride,
            col < WIDTH,
            other=-1,
        ).to(tl.int64)
        logical_block = token // BLOCK_SIZE
        valid = (col < WIDTH) & (token >= 0) & (logical_block < N_BLOCKS)
        valid = valid & row_valid
        block = tl.load(
            Blocks + req * block_row_stride + logical_block * block_col_stride,
            valid,
            other=-1,
        ).to(tl.int64)
        valid = valid & (block >= 0)
        dst = count + tl.cumsum(valid.to(tl.int32), 0) - 1
        physical = block * BLOCK_SIZE + token % BLOCK_SIZE
        tl.store(Physical + row * WIDTH + dst.to(tl.int64), physical, valid)
        count += tl.sum(valid.to(tl.int32), 0)
    # Tail and compact prefix are disjoint: no racing initialization/scatter.
    for start in range(tl.cdiv(WIDTH, TILE)):
        col = start * TILE + lane
        tl.store(
            Physical + row * WIDTH + col.to(tl.int64),
            -1,
            (col < WIDTH) & (col >= count),
        )
    tl.store(Lengths + row, count)


def convert_sparse_indices(request_ids, block_table, token_indices, block_size):
    """Map and stably compact sparse token indices to int32 physical slots.

    Inputs are integer request IDs [M], page table [R, B], and logical tokens
    [M, K]. Negative logical tokens, absent pages and out-of-range logical pages
    are removed, duplicates remain, and each output row has a -1 tail. Request
    IDs must be legal page-table rows (negative row indexing is supported).
    This inference-only API does not modify inputs or synchronize the host.
    """
    if request_ids.ndim != 1 or block_table.ndim != 2 or token_indices.ndim != 2:
        raise ValueError("expected request_ids[M], block_table[R,B], indices[M,K]")
    if request_ids.shape[0] != token_indices.shape[0]:
        raise ValueError("request_ids and indices must have the same row count")
    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size <= 0
    ):
        raise ValueError("block_size must be a positive integer")
    tensors = (request_ids, block_table, token_indices)
    if any(t.dtype not in (torch.int32, torch.int64) for t in tensors):
        raise NotImplementedError("sparse indices require int32 or int64 inputs")
    if any(t.device != token_indices.device for t in tensors):
        raise ValueError("all sparse index inputs must be on the same device")
    if token_indices.device.type == "cpu":
        raise NotImplementedError("sparse index conversion requires a Triton device")
    rows, width = token_indices.shape
    if rows and not block_table.shape[0]:
        raise ValueError("nonempty indices require at least one page-table row")
    output = torch.empty((rows, width), dtype=torch.int32, device=token_indices.device)
    lengths = torch.empty((rows,), dtype=torch.int32, device=token_indices.device)
    if rows:
        # A fixed-size scan bounds metadata workspace independently of width.
        with torch.get_device_module(token_indices.device).device(token_indices.device):
            _convert_sparse_indices_kernel[(rows,)](
                request_ids,
                block_table,
                token_indices,
                output,
                lengths,
                request_ids.stride(0),
                *block_table.stride(),
                *token_indices.stride(),
                block_table.shape[0],
                block_table.shape[1],
                width,
                block_size,
                1024,
                num_warps=4,
            )
    return output, lengths

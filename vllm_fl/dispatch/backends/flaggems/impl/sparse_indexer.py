# SPDX-License-Identifier: Apache-2.0
"""Sparse-indexer contracts bound to existing FlagGems implementations.

Only metadata checks live here. Provider ordering/fallback belongs to OpManager.
Query scales are folded into weights by the model before either MQA operation.
"""

import torch


def supports_native_e4m3(device):
    # A CUDA-compatible device API alone does not establish Triton fp8e4nv
    # support (e.g. pinned PPU and NVIDIA SM80 cannot compile this arithmetic).
    from triton.runtime import driver

    target = driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability(device) >= (
        8,
        9,
    )


def supports_prepare_query(
    q, group_size, *, output_dtype, column_major_scales=False, use_ue8m0=False
):
    return (
        q.device.type == "cuda"
        and q.ndim == 2
        and q.is_contiguous()
        and q.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and output_dtype == torch.float8_e4m3fn
        and supports_native_e4m3(q.device)
        and group_size > 0
        and q.shape[-1] % group_size == 0
    )


def prepare_query(
    q, group_size, *, output_dtype, column_major_scales=False, use_ue8m0=False
):
    from flag_gems.ops.per_token_group_quant_fp8 import per_token_group_quant_fp8

    return per_token_group_quant_fp8(
        q,
        group_size,
        dtype=output_dtype,
        column_major_scales=column_major_scales,
        scale_ue8m0=use_ue8m0,
    )


def supports_rotation(q):
    return (
        q.device.type == "cuda"
        and q.shape[-1] == 128
        and q.dtype in (torch.float32, torch.float16, torch.bfloat16)
    )


def rotate_indexer_query(q):
    from flag_gems.ops.hadamard_transform import hadamard_transform

    return hadamard_transform(q.float(), scale=128**-0.5).to(q.dtype)


def supports_mqa(q, kv, weights, starts, ends, clean_logits=True):
    values, scale = q
    keys, key_scale = kv
    return (
        values.device.type == "cuda"
        and scale is None
        and values.ndim == 3
        and keys.ndim == 2
        and values.dtype == keys.dtype
        and values.dtype in (torch.bfloat16, torch.float16, torch.float8_e4m3fn)
        and (values.dtype != torch.float8_e4m3fn or supports_native_e4m3(values.device))
        and values.shape[-1] == keys.shape[-1] == 128
        and key_scale is not None
        and key_scale.numel() == keys.shape[0]
        and key_scale.is_contiguous()
        and key_scale.dtype == torch.float32
        and weights.shape == values.shape[:2]
        and starts.numel() == ends.numel() == values.shape[0]
        and starts.is_contiguous()
        and ends.is_contiguous()
        and starts.dtype in (torch.int32, torch.int64)
        and ends.dtype in (torch.int32, torch.int64)
        and all(
            t.device == values.device for t in (keys, key_scale, weights, starts, ends)
        )
    )


def mqa_logits(*args, **kwargs):
    from flag_gems.fused.fp8_fp4_mqa_logits import fp8_fp4_mqa_logits

    return fp8_fp4_mqa_logits(*args, **kwargs)


def supports_paged_mqa(
    q,
    cache,
    weights,
    lengths,
    table,
    schedule_metadata,
    *,
    max_model_len,
    clean_logits=False,
):
    values, scale = q
    return (
        values.device.type == "cuda"
        and values.ndim == 4
        and scale is None
        and values.dtype == cache.dtype == torch.bfloat16
        and values.shape[-2] in (32, 64)
        and values.shape[-1] == 128
        and cache.ndim in (3, 4)
        and cache.shape[1] == 64
        and cache.shape[-1] == 128
        and (cache.ndim == 3 or cache.shape[2] == 1)
        and values.is_contiguous()
        and cache.is_contiguous()
        and weights.is_contiguous()
        and lengths.is_contiguous()
        and weights.shape == (values.shape[0] * values.shape[1], values.shape[2])
        and lengths.numel() == values.shape[0] * values.shape[1]
        and table.ndim == 2
        and table.shape[0] == values.shape[0]
        and table.shape[1] * 64 >= max_model_len
        and table.stride(1) == 1
        and table.dtype in (torch.int32, torch.int64)
        and lengths.dtype in (torch.int32, torch.int64)
        and all(t.device == values.device for t in (cache, weights, lengths, table))
    )


def paged_mqa_logits(
    q,
    cache,
    weights,
    lengths,
    table,
    schedule_metadata,
    *,
    max_model_len,
    clean_logits=False,
):
    from flag_gems.fused.bf16_paged_mqa_logits import bf16_paged_mqa_logits

    result = bf16_paged_mqa_logits(
        q[0],
        cache,
        weights,
        lengths,
        table,
        schedule_metadata,
        max_model_len,
        False,
    )
    if clean_logits:
        # The pinned public cleanup loops over context_lens.item(), which is
        # illegal in capture. Keep its MQA kernel and mask on device instead.
        columns = torch.arange(max_model_len, device=result.device)
        result.masked_fill_(columns[None, :] >= lengths.reshape(-1, 1), -torch.inf)
    return result


def supports_gather(cache, output, scales, table, cu_seqlen):
    return (
        cache.device.type == "cuda"
        and cache.dtype == torch.uint8
        and cache.ndim == 3
        and output.ndim == 2
        and output.dtype == torch.float8_e4m3fn
        and output.shape[1] == 128
        and scales is not None
        and scales.dtype == torch.float32
        and scales.shape == (output.shape[0], 1)
        and cache.shape[2] == 132
        and cache.is_contiguous()
        and scales.is_contiguous()
        and output.stride(-1) == 1
        and table.ndim == 2
        and table.stride(1) == 1
        and cu_seqlen.ndim == 1
        and cu_seqlen.is_contiguous()
        and cu_seqlen.numel() == table.shape[0] + 1
        and table.dtype in (torch.int32, torch.int64)
        and cu_seqlen.dtype in (torch.int32, torch.int64)
        and all(t.device == cache.device for t in (output, scales, table, cu_seqlen))
    )


def gather_cache(cache, output, scales, table, cu_seqlen):
    from flag_gems.fused.cp_gather_indexer_k_quant_cache import (
        cp_gather_indexer_k_quant_cache as impl,
    )

    # The pinned API measures the scale width in bytes, not float32 elements.
    return impl(cache, output, scales.view(torch.uint8), table, cu_seqlen)


def supports_cache_write(k, cache, slots, quant_block_size, scale_fmt):
    return (
        k.device.type == "cuda"
        and k.ndim == 2
        and k.is_contiguous()
        and k.dtype in (torch.float32, torch.bfloat16, torch.float16)
        and k.shape[-1] == 128
        and quant_block_size == 128
        and cache.dtype == torch.uint8
        and cache.ndim == 3
        and cache.shape[-1] == 132
        and cache.is_contiguous()
        and slots.ndim == 1
        and slots.is_contiguous()
        and slots.numel() <= k.shape[0]
        and slots.dtype in (torch.int32, torch.int64)
        and k.device == cache.device == slots.device
        and scale_fmt in (None, "ue8m0")
        and supports_native_e4m3(k.device)
    )


def indexer_k_quant_and_cache(*args, **kwargs):
    from flag_gems.fused.indexer_k_quant_and_cache import (
        indexer_k_quant_and_cache as impl,
    )

    return impl(*args, **kwargs)


def supports_topk(logits, *args, **kwargs):
    return logits.device.type == "cuda"


def topk_prefill(*args, **kwargs):
    from flag_gems.fused.top_k_per_row_prefill import top_k_per_row_prefill

    return top_k_per_row_prefill(*args, **kwargs)


def topk_decode(*args, max_seq_len=None, **kwargs):
    from flag_gems.fused.top_k_per_row_decode import top_k_per_row_decode

    return top_k_per_row_decode(*args, **kwargs)


def supports_pack(tensor, lengths, *args, **kwargs):
    # Pinned public wrappers use lengths.max()/sum().item() to allocate output.
    # These calls are outside capture in the current indexer. Fail closed if
    # reused inside capture; do not silently synchronize or run a CPU fallback.
    return (
        tensor.device.type == "cuda"
        and tensor.is_contiguous()
        and not torch.cuda.is_current_stream_capturing()
    )


def pack_seq(*args, **kwargs):
    from flag_gems.fused.pack_seq import pack_seq_triton

    return pack_seq_triton(*args, **kwargs)


def unpack_seq(*args, **kwargs):
    from flag_gems.fused.unpack_seq import unpack_seq_triton

    return unpack_seq_triton(*args, **kwargs)

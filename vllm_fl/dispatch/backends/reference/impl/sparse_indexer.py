# SPDX-License-Identifier: Apache-2.0
"""Indexer reference decompositions; not a generic graph backend."""

import torch


def _dequantize_grouped(
    values: torch.Tensor, scales: torch.Tensor | None
) -> torch.Tensor:
    output = values.float()
    if scales is None:
        return output
    scales = scales.float()
    if scales.ndim == output.ndim - 1:
        return output * scales.unsqueeze(-1)
    num_groups = scales.shape[-1]
    if num_groups == 1:
        return output * scales
    if output.shape[-1] % num_groups:
        raise ValueError("Quantized width must be divisible by the scale groups")
    group_size = output.shape[-1] // num_groups
    grouped = output.reshape(*output.shape[:-1], num_groups, group_size)
    return (grouped * scales.unsqueeze(-1)).reshape_as(output)


def _torch_mqa_logits(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool = True,
) -> torch.Tensor:
    del clean_logits
    q_values, q_scale = q
    k_values, k_scale = kv
    q_float = _dequantize_grouped(q_values, q_scale)
    k_float = _dequantize_grouped(k_values, k_scale)
    score = torch.einsum("mhd,nd->hmn", q_float, k_float)
    logits = (score.relu() * weights.float().transpose(0, 1).unsqueeze(-1)).sum(0)
    columns = torch.arange(k_values.shape[0], device=q_values.device).unsqueeze(0)
    valid = (columns >= cu_seqlen_ks.reshape(-1, 1)) & (
        columns < cu_seqlen_ke.reshape(-1, 1)
    )
    return logits.masked_fill(~valid, float("-inf"))


def _torch_topk(
    logits: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    top_k: int,
    relative_to_start: bool,
) -> torch.Tensor:
    columns = torch.arange(logits.shape[1], device=logits.device).unsqueeze(0)
    starts = starts.reshape(-1, 1).to(columns.dtype)
    ends = ends.reshape(-1, 1).to(columns.dtype)
    masked = logits.masked_fill((columns < starts) | (columns >= ends), float("-inf"))
    actual_k = min(top_k, logits.shape[1])
    values, indices = torch.topk(masked, k=actual_k, dim=-1)
    indices = indices.to(torch.int32)
    if relative_to_start:
        indices = indices - starts.to(torch.int32)
    indices = torch.where(values == float("-inf"), -1, indices)
    if actual_k == top_k:
        return indices
    out = torch.full(
        (logits.shape[0], top_k), -1, dtype=torch.int32, device=logits.device
    )
    out[:, :actual_k] = indices
    return out


def _torch_pack_seq(
    tensor: torch.Tensor, lengths: torch.Tensor, pad_value=-float("inf")
) -> torch.Tensor:
    lengths_cpu = lengths.detach().to("cpu", torch.int64).tolist()
    max_length = max(lengths_cpu, default=0)
    out = torch.full(
        (len(lengths_cpu), max_length, *tensor.shape[1:]),
        pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    cursor = 0
    for request, length in enumerate(lengths_cpu):
        out[request, :length].copy_(tensor[cursor : cursor + length])
        cursor += length
    return out


def _torch_unpack_seq(tensor: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    lengths_cpu = lengths.detach().to("cpu", torch.int64).tolist()
    pieces = [tensor[request, :length] for request, length in enumerate(lengths_cpu)]
    if not pieces:
        return tensor.new_empty((0, *tensor.shape[2:]))
    return torch.cat(pieces, dim=0)


def rotate_indexer_query(q):
    from vllm_fl.kernels.glm5_next.portable import hadamard128

    if q.shape[-1] != 128:
        raise ValueError("Indexer rotation requires head_dim=128")
    return hadamard128(q.float()).to(q.dtype)


def topk_prefill(
    logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
):
    indices.copy_(_torch_topk(logits, row_starts, row_ends, top_k, True))


def topk_decode(
    logits,
    next_n,
    seq_lens,
    indices,
    num_rows,
    stride0,
    stride1,
    top_k,
    *,
    max_seq_len=None,
):
    ends = (
        seq_lens.reshape(-1)
        if seq_lens.ndim == 2
        else seq_lens.repeat_interleave(next_n)
    )[:num_rows]
    indices.copy_(_torch_topk(logits, torch.zeros_like(ends), ends, top_k, False))


def supports_pack(tensor, lengths, *args, **kwargs):
    # Output shape depends on device values; never synchronize during capture.
    return tensor.device.type != "cuda" or not torch.cuda.is_current_stream_capturing()

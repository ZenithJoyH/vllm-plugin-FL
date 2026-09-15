# Copyright (c) 2026 BAAI. All rights reserved.

"""
FlagGems fused moe operator implementations.
"""

import os
from typing import Optional

import torch
from vllm.triton_utils import triton
from vllm.utils.math_utils import round_up


def moe_align_block_size_flaggems(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    expert_map: Optional[torch.Tensor] = None,
    pad_sorted_ids: bool = False,
    ignore_invalid_experts: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    from flag_gems import moe_align_block_size_triton

    max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)
    if pad_sorted_ids:
        max_num_tokens_padded = round_up(max_num_tokens_padded, block_size)
    if topk_ids.numel() < num_experts:
        max_num_tokens_padded = min(
            topk_ids.numel() * block_size, max_num_tokens_padded
        )
    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)
    # TODO(lms): ignore_invalid_experts not effective now
    # moe_align_block_size has optimize version to filtered out
    # all invalid experts directly when counting the number of experts
    moe_align_block_size_triton(
        topk_ids,
        num_experts,
        block_size,
        sorted_ids,
        expert_ids,
        num_tokens_post_pad,
    )
    if expert_map is not None:
        expert_ids = expert_map[expert_ids]

    return sorted_ids, expert_ids, num_tokens_post_pad


def topk_softmax_flaggems(
    topk_weights, topk_indices, token_expert_indices, gating_output, renormalize=False
):
    from flag_gems import topk_softmax

    try:
        topk_softmax(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            renormalize,
        )
    except:
        topk_softmax(topk_weights, topk_indices, token_expert_indices, gating_output)
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_indices


_DEFAULT_LARGE_BF16_MOE_CONFIG = {
    "BLOCK_SIZE_M": 128,
    "BLOCK_SIZE_N": 128,
    "BLOCK_SIZE_K": 64,
    "GROUP_SIZE_M": 1,
    "SPLIT_K": 1,
    "num_warps": 8,
    "num_stages": 3,
}


def _maybe_tune_large_bf16_moe_stage_config(
    A,
    B,
    C,
    A_scale,
    B_scale,
    top_k,
    config,
    use_fp8_w8a8,
    use_int8_w8a8,
    use_int8_w8a16,
    use_int4_w4a16,
    per_channel_quant,
    block_shape,
    B_bias,
):
    """Experimental PPU tuning for the trace-proven unquantized shape."""
    if os.environ.get("VLLM_FL_PPU_MOE_STAGE_CONFIG") != "1":
        return config
    if (
        A.dtype != torch.bfloat16
        or B.dtype != torch.bfloat16
        or C.dtype != torch.bfloat16
        or C.size(0) < 4096
        or A_scale is not None
        or B_scale is not None
        or B_bias is not None
        or use_fp8_w8a8
        or use_int8_w8a8
        or use_int8_w8a16
        or use_int4_w4a16
        or per_channel_quant
        or block_shape is not None
        or any(
            config.get(key) != value
            for key, value in _DEFAULT_LARGE_BF16_MOE_CONFIG.items()
        )
    ):
        return config

    tuned_config = dict(config)
    if B.size() == (288, 256, 4096) and C.size(-1) == 256 and top_k == 8:
        tuned_config["BLOCK_SIZE_K"] = 128
    elif B.size() == (288, 4096, 128) and C.size(-1) == 4096 and top_k == 1:
        tuned_config["BLOCK_SIZE_N"] = 256
        tuned_config["GROUP_SIZE_M"] = 16
    return tuned_config


def invoke_fused_moe_triton_kernel_flaggems(
    A,
    B,
    C,
    A_scale,
    B_scale,
    topk_weights,
    sorted_token_ids,
    expert_ids,
    num_tokens_post_padded,
    mul_routed_weight,
    top_k,
    config,
    compute_type,
    use_fp8_w8a8,
    use_int8_w8a8,
    use_int8_w8a16,
    use_int4_w4a16,
    per_channel_quant,
    block_shape=None,
    B_bias=None,
):
    from flag_gems import invoke_fused_moe_triton_kernel

    config = _maybe_tune_large_bf16_moe_stage_config(
        A,
        B,
        C,
        A_scale,
        B_scale,
        top_k,
        config,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        per_channel_quant,
        block_shape,
        B_bias,
    )

    invoke_fused_moe_triton_kernel(
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
        top_k,
        config,
        compute_type,
        use_fp8_w8a8,
        use_int8_w8a8,
        use_int8_w8a16,
        use_int4_w4a16,
        per_channel_quant,
        block_shape=block_shape,
        B_bias=B_bias,
    )


def grouped_topk_flaggems(
    scores,
    n_group,
    topk_group,
    topk,
    renormalize,
    routed_scaling_factor,
    bias,
    scoring_func=0,
):
    from flag_gems import grouped_topk

    return grouped_topk(
        scores,
        n_group,
        topk_group,
        topk,
        renormalize,
        routed_scaling_factor,
        bias,
        scoring_func,
    )


def moe_sum_flaggems(inp, out):
    from flag_gems import moe_sum

    moe_sum(inp, out)

# SPDX-License-Identifier: Apache-2.0
"""FlagGems regression tests for packed MiniMax SwiGLU-OAI activation."""

import os

import pytest
import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_fl.ops.fused_moe.activation import apply_moe_activation

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or "PPU_SDK" not in os.environ,
    reason="T-Head PPU is not available",
)


def _reference(input, limit, alpha, beta):
    gate, up = input.float().chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    up = up.clamp(min=-limit, max=limit)
    return (gate * torch.sigmoid(alpha * gate) * (up + beta)).to(input.dtype)


def _invoke(input, limit=7.0, alpha=1.702, beta=1.0):
    output = torch.empty(
        input.shape[0],
        input.shape[1] // 2,
        device=input.device,
        dtype=input.dtype,
    )
    return apply_moe_activation(
        MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        output,
        input,
        clamp_limit=limit,
        alpha=alpha,
        beta=beta,
    )


def test_minimax_packed_activation_matches_reference():
    torch.manual_seed(31)
    input = (
        torch.randn(7, 6144, device="cuda", dtype=torch.bfloat16) * 9
    ).contiguous()

    actual = _invoke(input)
    expected = _reference(input, 7.0, 1.702, 1.0)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_minimax_packed_activation_cuda_graph_changed_input():
    if not hasattr(torch.cuda, "CUDAGraph"):
        pytest.skip("CUDA-compatible graph capture is unavailable")
    torch.manual_seed(32)
    input = torch.randn(
        5, 6144, device="cuda", dtype=torch.bfloat16
    ).contiguous()
    output = torch.empty(5, 3072, device="cuda", dtype=torch.bfloat16)

    apply_moe_activation(
        MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        output,
        input,
        clamp_limit=7.0,
        alpha=1.702,
        beta=1.0,
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        apply_moe_activation(
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            output,
            input,
            clamp_limit=7.0,
            alpha=1.702,
            beta=1.0,
        )
    torch.cuda.synchronize()

    replay_input = (
        torch.randn_like(input) * 9
    ).contiguous()
    input.copy_(replay_input)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output,
        _reference(replay_input, 7.0, 1.702, 1.0),
        rtol=1e-2,
        atol=1e-2,
    )


def test_minimax_packed_activation_requires_clamp_limit():
    input = torch.zeros(2, 6144, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(2, 3072, device="cuda", dtype=torch.bfloat16)

    with pytest.raises(AssertionError, match="requires a clamp limit"):
        apply_moe_activation(
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            output,
            input,
            alpha=1.702,
            beta=1.0,
        )

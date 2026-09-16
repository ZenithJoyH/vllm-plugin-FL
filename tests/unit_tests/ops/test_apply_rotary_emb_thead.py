# SPDX-License-Identifier: Apache-2.0
"""FlagGems regression tests for partial ApplyRotaryEmb OOT semantics."""

import os

import pytest
import torch

from vllm_fl.ops.rotary_embedding import ApplyRotaryEmbFL

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or "PPU_SDK" not in os.environ,
    reason="T-Head PPU is not available",
)


@pytest.fixture
def default_vllm_config():
    """Provide the upstream CustomOp configuration context locally."""
    from vllm.config import VllmConfig, set_current_vllm_config

    config = VllmConfig()
    with set_current_vllm_config(config):
        yield config


def _reference(x, cos, sin):
    rotary_dim = cos.shape[-1] * 2
    x_float = x.float()
    x_rot = x_float[..., :rotary_dim]
    x_pass = x_float[..., rotary_dim:]
    x1, x2 = x_rot.chunk(2, dim=-1)
    cos = cos.float().unsqueeze(-2)
    sin = sin.float().unsqueeze(-2)
    rotated = torch.cat(
        (x1 * cos - x2 * sin, x2 * cos + x1 * sin),
        dim=-1,
    )
    return torch.cat((rotated, x_pass), dim=-1).to(x.dtype)


def test_partial_rotary_preserves_tail_and_matches_reference(
    default_vllm_config,
):
    torch.manual_seed(10)
    x = torch.randn(2, 17, 1, 80, device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(17, 39, device="cuda", dtype=torch.float32)
    sin = torch.randn(17, 39, device="cuda", dtype=torch.float32)
    op = ApplyRotaryEmbFL(enforce_enable=True, enable_fp32_compute=True)

    actual = op(x, cos, sin)
    expected = _reference(x, cos, sin)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(actual[..., 78:], x[..., 78:], rtol=0, atol=0)


def test_full_rotary_matches_reference(default_vllm_config):
    torch.manual_seed(11)
    x = torch.randn(2, 9, 3, 128, device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(9, 64, device="cuda", dtype=torch.float32)
    sin = torch.randn(9, 64, device="cuda", dtype=torch.float32)
    op = ApplyRotaryEmbFL(enforce_enable=True, enable_fp32_compute=True)

    torch.testing.assert_close(
        op(x, cos, sin),
        _reference(x, cos, sin),
        rtol=1e-5,
        atol=1e-5,
    )


def test_partial_rotary_cuda_graph_replay(default_vllm_config):
    if not hasattr(torch.cuda, "CUDAGraph"):
        pytest.skip("CUDA-compatible graph capture is unavailable")
    torch.manual_seed(12)
    x = torch.randn(2, 7, 1, 80, device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(7, 39, device="cuda", dtype=torch.float32)
    sin = torch.randn(7, 39, device="cuda", dtype=torch.float32)
    op = ApplyRotaryEmbFL(enforce_enable=True, enable_fp32_compute=True)

    op(x, cos, sin)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = op(x, cos, sin)
    torch.cuda.synchronize()

    replay_input = torch.randn_like(x)
    x.copy_(replay_input)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        output,
        _reference(replay_input, cos, sin),
        rtol=1e-5,
        atol=1e-5,
    )

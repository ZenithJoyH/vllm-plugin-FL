# SPDX-License-Identifier: Apache-2.0
"""M3 MoE activation uses the common semantic dispatch, including graph replay."""

import os
from unittest.mock import Mock

import pytest
import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation

from vllm_fl.dispatch import (
    SelectionPolicy,
    get_default_manager,
    reset_default_manager,
    reset_global_policy,
    set_global_policy,
)
from vllm_fl.ops.fused_moe.activation import apply_moe_activation


def test_m3_activation_forwards_semantic_parameters(monkeypatch):
    from vllm_fl.ops.fused_moe import activation

    impl = Mock()
    monkeypatch.setattr(activation, "_swigluoai_uninterleave", impl)
    x, out = torch.empty(7, 384), torch.empty(7, 192)
    assert (
        apply_moe_activation(
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            out,
            x,
            clamp_limit=3.0,
            alpha=0.75,
            beta=-0.25,
        )
        is out
    )
    impl.assert_called_once_with(out, x, 3.0, 0.75, -0.25)


@pytest.fixture
def activation_dispatch(monkeypatch):
    from vllm_fl.dispatch.backends.flaggems import register_ops

    reset_default_manager()
    reset_global_policy()
    manager = get_default_manager()
    manager._state.initialized = True
    manager._state.init_pid = os.getpid()
    monkeypatch.setattr(
        register_ops, "use_flaggems_op", lambda name: name == "swigluoai_uninterleave"
    )
    register_ops.register_builtins(manager.registry)
    set_global_policy(SelectionPolicy(strict=True))
    yield manager
    reset_default_manager()
    reset_global_policy()


@pytest.mark.gpu
@pytest.mark.parametrize("rows", [0, 1, 7, 32])
@pytest.mark.parametrize("parameters", [(7.0, 1.702, 1.0), (3.0, 0.75, -0.25)])
def test_m3_activation_matches_fp32_and_prior_kernel(
    activation_dispatch, rows, parameters
):
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA-like accelerator")
    from vllm_fl.ops.minimax_m3.ops import swiglu as prior_swiglu

    limit, alpha, beta = parameters
    x = (torch.randn(rows, 384, device="cuda") * 9).bfloat16()
    out = torch.empty(rows, 192, device="cuda", dtype=x.dtype)
    apply_moe_activation(
        MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        out,
        x,
        clamp_limit=limit,
        alpha=alpha,
        beta=beta,
    )
    old = prior_swiglu(x, limit, alpha, beta)
    gate, up = x.cpu().float().chunk(2, dim=-1)
    gate = gate.clamp(max=limit)
    reference = gate * torch.sigmoid(alpha * gate) * (up.clamp(-limit, limit) + beta)
    if rows:
        denominator = reference.norm().clamp_min(1e-12)
        old_error = (old.cpu().float() - reference).norm() / denominator
        new_error = (out.cpu().float() - reference).norm() / denominator
        assert new_error <= max(0.0035, float(old_error) * 1.1)
        torch.testing.assert_close(out, old, rtol=0.008, atol=0.001)
    assert activation_dispatch._called_ops["swigluoai_uninterleave"] == "default.flagos"
    assert "m3_swiglu" not in activation_dispatch._called_ops


@pytest.mark.gpu
def test_m3_activation_graph_reads_updated_input(activation_dispatch):
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA-like accelerator")
    x = torch.randn(8, 384, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(8, 192, device="cuda", dtype=x.dtype)

    def call():
        return apply_moe_activation(
            MoEActivation.SWIGLUOAI_UNINTERLEAVE,
            out,
            x,
            clamp_limit=7.0,
            alpha=1.702,
            beta=1.0,
        )

    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    x.mul_(3).add_(1)
    graph.replay()
    actual = out.clone()
    call()
    assert torch.equal(actual, out)

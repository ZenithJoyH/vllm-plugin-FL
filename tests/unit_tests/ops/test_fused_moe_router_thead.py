# SPDX-License-Identifier: Apache-2.0
"""FlagGems regression tests for sigmoid+bias MoE routing."""

import os
from types import SimpleNamespace

import pytest
import torch

from vllm_fl.ops.fused_moe import router as router_module

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or "PPU_SDK" not in os.environ,
    reason="T-Head PPU is not available",
)


def _router(bias, scale=1.25):
    return SimpleNamespace(
        e_score_correction_bias=SimpleNamespace(data=bias),
        top_k=4,
        renormalize=True,
        scoring_func="sigmoid",
        routed_scaling_factor=scale,
    )


def _reference(logits, bias, ids, scale):
    scores = torch.sigmoid(logits.float())
    expected_ids = (scores + bias.float()).topk(
        ids.shape[-1], dim=-1, sorted=False
    ).indices
    assert all(
        set(actual) == set(expected)
        for actual, expected in zip(
            ids.cpu().tolist(), expected_ids.cpu().tolist()
        )
    )
    weights = scores.gather(1, ids.long())
    return weights / weights.sum(dim=-1, keepdim=True) * scale


def _invoke(router, hidden_states, logits):
    return router_module.FusedTopKBiasRouterFL._compute_routing(
        router,
        hidden_states,
        logits,
        indices_type=None,
    )


def test_flaggems_sigmoid_bias_matches_reference():
    torch.manual_seed(21)
    hidden_states = torch.randn(
        17, 64, device="cuda", dtype=torch.bfloat16
    )
    logits = torch.randn(17, 128, device="cuda", dtype=torch.float32)
    bias = torch.randn(128, device="cuda", dtype=torch.float32) * 0.1
    route = _router(bias)

    weights, ids = _invoke(route, hidden_states, logits)

    torch.testing.assert_close(
        weights,
        _reference(logits, bias, ids, route.routed_scaling_factor),
        rtol=2e-5,
        atol=2e-5,
    )
    assert weights.dtype == torch.float32
    assert ids.dtype == torch.int32


def test_flaggems_sigmoid_bias_cuda_graph_changed_input():
    torch.manual_seed(22)
    hidden_states = torch.randn(
        9, 64, device="cuda", dtype=torch.bfloat16
    )
    logits = torch.randn(9, 128, device="cuda", dtype=torch.float32)
    bias = torch.randn(128, device="cuda", dtype=torch.float32) * 0.1
    route = _router(bias)

    _invoke(route, hidden_states, logits)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        weights, ids = _invoke(route, hidden_states, logits)
    torch.cuda.synchronize()

    replay_logits = torch.randn_like(logits)
    logits.copy_(replay_logits)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(
        weights,
        _reference(replay_logits, bias, ids, route.routed_scaling_factor),
        rtol=2e-5,
        atol=2e-5,
    )


def test_sigmoid_bias_routes_through_cached_grouped_topk(monkeypatch):
    expected_weights = torch.ones(2, 4, device="cuda", dtype=torch.float32)
    expected_ids = torch.zeros(2, 4, device="cuda", dtype=torch.int32)
    calls = []

    def fake_grouped_topk(**kwargs):
        calls.append(kwargs)
        return expected_weights, expected_ids

    monkeypatch.setattr(
        router_module,
        "_fl_grouped_topk",
        fake_grouped_topk,
    )
    hidden_states = torch.zeros(2, 64, device="cuda", dtype=torch.bfloat16)
    logits = torch.zeros(2, 128, device="cuda", dtype=torch.float32)
    bias = torch.zeros(128, device="cuda", dtype=torch.float32)
    route = _router(bias, scale=1.0)

    weights, ids = _invoke(route, hidden_states, logits)

    assert calls and calls[0]["scoring_func"] == "sigmoid"
    assert calls[0]["num_expert_group"] == 1
    assert calls[0]["topk_group"] == 1
    assert weights is expected_weights
    assert ids is expected_ids

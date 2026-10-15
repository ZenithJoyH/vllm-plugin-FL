"""Public FlagGems-vLLM adapter contracts, independent of caller platform."""

import sys
from types import ModuleType

import pytest
import torch
from vllm_fl.dispatch.backends.flaggems.impl import fused_moe as adapter


@pytest.fixture
def public(monkeypatch):
    m = ModuleType("flaggems_vllm")
    monkeypatch.setitem(sys.modules, "flaggems_vllm", m)
    monkeypatch.setitem(sys.modules, "flag_gems", None)
    return m


@pytest.mark.parametrize("renormalize", [False, True])
def test_topk_public_contract(public, renormalize):
    calls = []
    public.topk_softmax = lambda *a: calls.append(a)
    args = (
        torch.empty(2, 2),
        torch.empty(2, 2, dtype=torch.int32),
        torch.empty(2, 2, dtype=torch.int32),
        torch.randn(2, 8),
        renormalize,
    )
    result = adapter.topk_softmax_flaggems(*args)
    assert result[0] is args[0] and result[1] is args[1]
    assert len(calls) == 1 and all(a is b for a, b in zip(calls[0][:-1], args[:-1]))
    assert calls[0][-1] is renormalize


@pytest.mark.parametrize("renormalize", [False, True])
def test_topk_legacy_four_argument_compatibility(public, renormalize):
    calls = []

    def legacy(weights, indices, token_indices, logits):
        calls.append((weights, indices, token_indices, logits))
        weights.copy_(torch.tensor([[1.0, 3.0]]))
        indices.fill_(2)

    public.topk_softmax = legacy
    weights = torch.empty(1, 2)
    indices = torch.empty(1, 2, dtype=torch.int32)
    token_indices = torch.empty_like(indices)
    logits = torch.zeros(1, 4)
    result = adapter.topk_softmax_flaggems(
        weights, indices, token_indices, logits, renormalize
    )
    assert len(calls) == 1 and calls[0][0] is weights
    assert result[1] is indices and torch.all(indices == 2)
    expected = (
        torch.tensor([[0.25, 0.75]]) if renormalize else torch.tensor([[1.0, 3.0]])
    )
    assert torch.equal(result[0], expected)
    assert (result[0] is weights) == (not renormalize)


def test_topk_legacy_error_propagates(public):
    def legacy(weights, indices, token_indices, logits):
        raise RuntimeError("legacy kernel failure")

    public.topk_softmax = legacy
    with pytest.raises(RuntimeError, match="legacy kernel failure"):
        adapter.topk_softmax_flaggems(None, None, None, None, False)


def test_grouped_topk_public_arguments(public):
    calls = []
    sentinel = object()
    public.grouped_topk = lambda *a: calls.append(a) or sentinel
    args = (object(), 4, 2, 8, True, 2.5, object(), 1)
    assert adapter.grouped_topk_flaggems(*args) is sentinel and calls == [args]


def test_sqrtsoftplus_bias_hash_scale(public):
    calls = []
    public.topk_softplus_sqrt = lambda *a: calls.append(a)
    x, logits = torch.empty(2, 4), torch.empty(2, 8)
    bias, tokens, table = object(), object(), object()
    weights, ids = adapter.fused_topk_bias_flaggems(
        x, logits, "sqrtsoftplus", bias, 2, True, torch.int64, tokens, table, 1.75
    )
    assert ids.dtype == torch.int64 and weights.dtype == torch.float32
    assert calls[0][0] is weights and calls[0][1] is ids
    assert calls[0][3:] == (logits, True, 1.75, bias, tokens, table)


def test_sum_uses_public_output(public):
    calls = []
    public.moe_sum = lambda *a: calls.append(a)
    x, out = object(), object()
    adapter.moe_sum_flaggems(x, out)
    assert calls == [(x, out)]


@pytest.mark.parametrize(
    "flags",
    [
        (False, False, False, False),
        (True, False, False, False),
        (False, True, False, False),
        (False, False, True, False),
        (False, False, False, True),
    ],
)
def test_gemm_public_contract_including_quantized_modes(public, flags):
    calls = []
    public.invoke_fused_moe_triton_kernel = lambda *a, **kw: calls.append((a, kw))
    args = (*[object() for _ in range(9)], False, 8, object(), object(), *flags, False)
    adapter.invoke_fused_moe_triton_kernel_flaggems(*args)
    assert calls == [(args, {"block_shape": None, "B_bias": None})]


@pytest.mark.parametrize("tokens", [0, 1, 4])
@pytest.mark.parametrize("padding", [False, True])
@pytest.mark.parametrize("mapped", [False, True])
def test_alignment_uses_public_lowlevel_contract(public, tokens, padding, mapped):
    calls = []

    def align(ids, experts, block, sorted_ids, expert_ids, total):
        calls.append((ids, experts, block, sorted_ids, expert_ids, total))
        sorted_ids.copy_(torch.arange(sorted_ids.numel(), dtype=torch.int32))
        expert_ids.zero_()
        total.fill_(ids.numel())

    public.moe_align_block_size_triton = align
    ids = torch.zeros(tokens, 2, dtype=torch.int32)
    expert_map = torch.tensor([3, 2, 1, 0], dtype=torch.int32) if mapped else None
    sorted_ids, expert_ids, total = adapter.moe_align_block_size_flaggems(
        ids, 4, 4, expert_map, padding
    )
    n = ids.numel() + 4 * 3
    if padding:
        n = (n + 3) // 4 * 4
    if ids.numel() < 4:
        n = min(ids.numel() * 4, n)
    assert len(calls) == 1 and calls[0][0] is ids and calls[0][1:3] == (4, 4)
    assert sorted_ids.shape == (n,) and expert_ids.shape == ((n + 3) // 4,)
    assert total.item() == ids.numel()
    assert torch.all(expert_ids == (3 if mapped else 0))

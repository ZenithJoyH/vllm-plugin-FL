#!/usr/bin/env python3
"""Focused integration checks for the T-Head native sparse MLA backend."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import torch

from vllm_fl.dispatch import get_default_manager
from vllm.v1.attention.backends.mla.flashmla_sparse import FlashMLASparseImpl
from vllm_fl.dispatch.backends.vendor.thead.impl.mla import (
    TheadMLASparseBackend,
    TheadMLASparseImpl,
)


PLUGIN_ROOT = Path(__file__).resolve().parents[2]


def check_structure() -> None:
    source = (
        PLUGIN_ROOT / "vllm_fl/dispatch/backends/vendor/thead/impl/mla.py"
    ).read_text()
    assert "flag_gems" not in source
    assert "MLASparseFL" not in source
    assert "sparse_mla_h4" not in source
    assert "CachedOp" not in source
    for relative in ('flaggems/register_ops.py', 'reference/register_ops.py',
                     'vendor/thead/register_ops.py'):
        registrations = (PLUGIN_ROOT / 'vllm_fl/dispatch/backends' / relative).read_text()
        assert 'op_name="concat_mla_q"' not in registrations
        assert 'op_name="mla_kv_cache_update"' not in registrations
    assert not (
        PLUGIN_ROOT
        / "vllm_fl/dispatch/backends/vendor/thead/impl/sparse_mla_h4.py"
    ).exists()
    assert TheadMLASparseBackend.get_name() == "THEAD_FLASHMLA_SPARSE"
    assert TheadMLASparseBackend.supported_kv_cache_dtypes == ["auto", "bfloat16"]
    print("STRUCTURE_OK")


def check_dispatch_eager_and_graph() -> None:
    manager = get_default_manager()
    attention_impl = manager._resolve_impl("attention_backend")
    assert attention_impl.impl_id == "vendor.thead"
    assert attention_impl.fn(use_mla=True, use_sparse=True) == (
        "vllm_fl.dispatch.backends.vendor.thead.impl.mla."
        "TheadMLASparseBackend"
    )

    impl = object.__new__(TheadMLASparseImpl)
    cache_update = impl.do_kv_cache_update
    torch.manual_seed(20260903)
    ql_nope = torch.randn(8, 4, 512, device="cuda", dtype=torch.bfloat16)
    q_pe = torch.randn(8, 4, 64, device="cuda", dtype=torch.bfloat16)
    q_out = torch.empty(8, 4, 576, device="cuda", dtype=torch.bfloat16)
    kv_c = torch.randn(8, 512, device="cuda", dtype=torch.bfloat16)
    k_pe = torch.randn(8, 1, 64, device="cuda", dtype=torch.bfloat16)
    kv_cache = torch.zeros(2, 64, 576, device="cuda", dtype=torch.bfloat16)
    slots = torch.tensor([0, 3, 63, 64, 65, 80, 100, 127], device="cuda")
    scale = torch.ones(1, device="cuda", dtype=torch.float32)
    impl.q_concat_buffer = q_out

    def concat(ql_nope, q_pe, out):
        # Isolate the inherited attention body; exercise the actual backend
        # tuple preparation hook, not a separately dispatched helper.
        with patch.object(FlashMLASparseImpl, 'forward_mqa',
                          lambda self, q, *args: q):
            actual = impl.forward_mqa((ql_nope, q_pe), kv_cache, None, None)
        assert actual.data_ptr() == out.data_ptr()

    with patch.object(manager, '_resolve_impl', side_effect=AssertionError('nested dispatch')):
        concat(ql_nope, q_pe, q_out)
        cache_update(kv_c, k_pe, kv_cache, slots, "auto", scale)
    with patch.object(FlashMLASparseImpl, 'forward_mqa', lambda self, q, *args: q):
        assert impl.forward_mqa(q_out, kv_cache, None, None) is q_out
    cache_update(kv_c, k_pe, kv_cache[:0], slots, 'auto', scale)

    concat(ql_nope, q_pe, q_out)
    cache_update(kv_c, k_pe, kv_cache, slots, "auto", scale)
    torch.cuda.synchronize()
    torch.testing.assert_close(q_out, torch.cat((ql_nope, q_pe), dim=-1))
    torch.testing.assert_close(
        kv_cache.view(-1, 576)[slots],
        torch.cat((kv_c, k_pe.squeeze(1)), dim=-1),
    )
    print(
        "BACKEND_HELPERS_EAGER_OK",
        attention_impl.impl_id,
    )

    q_out.zero_()
    kv_cache.zero_()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        concat(ql_nope, q_pe, q_out)
        cache_update(kv_c, k_pe, kv_cache, slots, "auto", scale)
    ql_nope.add_(1)
    kv_c.add_(1)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(q_out, torch.cat((ql_nope, q_pe), dim=-1))
    torch.testing.assert_close(
        kv_cache.view(-1, 576)[slots],
        torch.cat((kv_c, k_pe.squeeze(1)), dim=-1),
    )
    print("BACKEND_HELPERS_GRAPH_CHANGED_INPUT_OK")


def native_reference(q, kv, indices, scale, sinks, length):
    keys = kv[indices[0, 0, :length].long(), 0].float()
    scores = torch.einsum("hd,kd->hk", q[0].float(), keys) * scale
    maximum = scores.max(dim=-1).values
    weights = torch.exp(scores - maximum[:, None])
    denominator = weights.sum(dim=-1) + torch.exp(sinks - maximum)
    return (
        torch.einsum("hk,kd->hd", weights, keys[:, :512])
        / denominator[:, None]
    ).to(torch.bfloat16).unsqueeze(0)


def check_native_attention_eager_and_graph() -> None:
    torch.manual_seed(20260904)
    q = (torch.randn(1, 4, 576, device="cuda") * 0.05).to(torch.bfloat16)
    kv = (torch.randn(2048, 576, device="cuda") * 0.05).to(torch.bfloat16)
    indices = torch.arange(2048, device="cuda", dtype=torch.int32).view(1, 2048)
    lengths = torch.tensor([1984], device="cuda", dtype=torch.int32)
    sinks = torch.randn(4, device="cuda", dtype=torch.float32)
    scale = 576**-0.5
    expected = native_reference(q, kv.view(-1, 1, 576), indices.view(1, 1, -1), scale, sinks, 1984)

    impl = object.__new__(TheadMLASparseImpl)
    impl.softmax_scale = scale
    impl.kv_lora_rank = 512
    impl.sinks = sinks
    eager = impl._bf16_flash_mla_kernel(q, kv, indices, lengths)
    torch.cuda.synchronize()
    torch.testing.assert_close(eager, expected, atol=2e-2, rtol=2e-2)
    print("NATIVE_EAGER_OK", tuple(eager.shape))

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = impl._bf16_flash_mla_kernel(q, kv, indices, lengths)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_out, expected, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(graph_out, eager, atol=0, rtol=0)
    print("NATIVE_GRAPH_OK")


def main() -> None:
    check_structure()
    check_dispatch_eager_and_graph()
    check_native_attention_eager_and_graph()


if __name__ == "__main__":
    main()

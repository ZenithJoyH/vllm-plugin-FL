# SPDX-License-Identifier: Apache-2.0
"""Actual dispatch + independent math checks, without weights or a service."""

import pytest
import torch

from vllm_fl.dispatch import (
    CachedOp,
    SelectionPolicy,
    get_default_manager,
    policy_context,
)


@pytest.fixture(autouse=True)
def strict_flaggems_first():
    with policy_context(SelectionPolicy.from_dict(prefer="flagos", strict=True)):
        yield


def selected(name, expected):
    assert get_default_manager()._called_ops[name] == expected


def kda_reference(q, k, v, g, beta, state, ids):
    q, k, v, g, beta = [t.cpu().float() for t in (q, k, v, g, beta)]
    state = state.cpu().clone()
    result = torch.zeros_like(v)
    for i, row in enumerate(ids.cpu().reshape(-1).tolist()):
        qi, ki = q[0, i], k[0, i]
        qi = qi / (qi.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        ki = ki / (ki.square().sum(-1, keepdim=True) + 1e-6).sqrt()
        h = state[row] * g[0, i].exp().unsqueeze(-1)
        delta = (v[0, i] - (h * ki.unsqueeze(-1)).sum(-2)) * beta[0, i, :, None]
        h += ki.unsqueeze(-1) * delta.unsqueeze(-2)
        state[row] = h
        result[0, i] = (h * (qi / 128**0.5).unsqueeze(-1)).sum(-2)
    return result.to(torch.bfloat16), state


@pytest.mark.parametrize("batch", [1, 2, 8, 32])
@pytest.mark.parametrize("strided_state", [False, True])
def test_recurrent_kda_dispatch_and_graph(batch, strided_state):
    torch.manual_seed(81)
    q, k, v = [
        torch.randn(1, batch, 4, 128, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    ]
    g = -torch.rand(q.shape, device="cuda", dtype=torch.float32)
    beta = torch.rand(q.shape[:3], device="cuda", dtype=torch.float32)
    storage = (
        torch.randn(batch * 2, 4, 128, 256 if strided_state else 128, device="cuda")
        * 0.02
    )
    state = storage[..., ::2] if strided_state else storage
    baseline = state.clone()
    ids = torch.arange(batch - 1, -1, -1, device="cuda", dtype=torch.int32)
    cu = torch.arange(batch + 1, device="cuda", dtype=torch.int32)
    name = "fused_recurrent_kda"
    op = CachedOp(name)

    def run():
        return op(
            q,
            k,
            v,
            g,
            beta=beta,
            initial_state=state,
            cu_seqlens=cu,
            ssm_state_indices=ids,
        )

    expected, expected_state = kda_reference(q, k, v, g, beta, baseline, ids)
    output, returned = run()
    assert returned.data_ptr() == state.data_ptr()
    selected(name, "vendor.thead" if strided_state else "default.flagos")
    torch.testing.assert_close(output.cpu(), expected, atol=0.004, rtol=0.02)
    torch.testing.assert_close(state.cpu(), expected_state, atol=2e-5, rtol=2e-4)
    state.copy_(baseline)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, _ = run()
    for _ in range(2):
        q.neg_()
        ids.copy_(ids.flip(0))
        state.copy_(baseline)
        expected, expected_state = kda_reference(q, k, v, g, beta, baseline, ids)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, atol=0.004, rtol=0.02)
        torch.testing.assert_close(state.cpu(), expected_state, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("page", [32, 64])
def test_paged_mqa_dispatch_and_graph(page):
    torch.manual_seed(48)
    q = torch.randn(2, 1, 32, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    cache = torch.randn(4, page, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    weights = torch.rand(2, 32, device="cuda")
    lengths = torch.tensor([[page + 7], [page - 3]], device="cuda", dtype=torch.int32)
    table = torch.tensor([[2, 0], [3, 1]], device="cuda", dtype=torch.int32)
    name = "sparse_indexer_paged_mqa_logits"
    op = CachedOp(name)

    def run():
        return op(
            (q, None),
            cache,
            weights,
            lengths,
            table,
            None,
            max_model_len=page * 2,
            clean_logits=True,
        )

    def reference():
        out = torch.full((2, page * 2), -torch.inf)
        for b in range(2):
            keys = cache.cpu()[table.cpu()[b].long()].reshape(-1, 128).float()
            n = int(lengths.cpu()[b, 0])
            out[b, :n] = (
                (q.cpu()[b, 0].float() @ keys[:n].T).relu() * weights.cpu()[b, :, None]
            ).sum(0)
        return out

    torch.testing.assert_close(run().cpu(), reference(), atol=0.015, rtol=0.02)
    selected(name, "default.flagos" if page == 64 else "vendor.thead")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    q.neg_()
    table.copy_(table.flip(0))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(result.cpu(), reference(), atol=0.015, rtol=0.02)


@pytest.mark.parametrize("quantized", [False, True])
def test_cache_gather_dispatch_and_graph(quantized):
    torch.manual_seed(52)
    page, width, blocks = 64, 128, 4
    table = torch.tensor([[2, 0], [3, 1]], device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, 67, 129], device="cuda", dtype=torch.int32)
    if quantized:
        cache = torch.zeros(blocks, page, width + 4, device="cuda", dtype=torch.uint8)
        flat = cache.view(blocks, -1)
        flat[:, : page * width].copy_(
            torch.randint(
                0, 127, (blocks, page * width), device="cuda", dtype=torch.uint8
            )
        )
        flat[:, page * width :].view(torch.float32).fill_(1.25)
        out = torch.empty(129, width, device="cuda", dtype=torch.float8_e4m3fn)
        scales = torch.zeros(129, 1, device="cuda")
    else:
        cache = torch.randn(blocks, page, width, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(129, width, device="cuda", dtype=torch.bfloat16)
        scales = torch.ones(129, 1, device="cuda")
    name = "sparse_indexer_gather_cache"
    op = CachedOp(name)

    def run():
        op(cache, out, scales, table, cu)

    def verify():
        values = (
            cache.cpu().view(blocks, -1)[:, : page * width].reshape(blocks, page, width)
            if quantized
            else cache.cpu()
        )
        expected = torch.cat(
            [
                values[table.cpu()[0].long()].reshape(-1, width)[:67],
                values[table.cpu()[1].long()].reshape(-1, width)[:62],
            ]
        )
        torch.testing.assert_close(
            out.cpu().view(torch.uint8) if quantized else out.cpu(),
            expected,
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            scales.cpu(), torch.full((129, 1), 1.25 if quantized else 1.0)
        )

    run()
    selected(name, "default.flagos" if quantized else "vendor.thead")
    verify()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    table.copy_(table.flip(0))
    graph.replay()
    torch.cuda.synchronize()
    verify()


def test_mqa_topk_and_pack_bindings():
    from vllm_fl.dispatch.backends.reference.impl.sparse_indexer import (
        _torch_mqa_logits,
        _torch_topk,
    )

    torch.manual_seed(24)
    q = torch.randn(4, 32, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    k = torch.randn(64, 128, device="cuda", dtype=torch.bfloat16) * 0.1
    scale = torch.ones(64, 1, device="cuda")
    weights = torch.rand(4, 32, device="cuda")
    starts = torch.tensor([0, 2, 5, 8], device="cuda", dtype=torch.int32)
    ends = torch.tensor([32, 48, 64, 60], device="cuda", dtype=torch.int32)
    op = CachedOp("sparse_indexer_mqa_logits")

    def run():
        return op((q, None), (k, scale), weights, starts, ends)

    def expected():
        return _torch_mqa_logits(
            (q.cpu(), None),
            (k.cpu(), scale.cpu()),
            weights.cpu(),
            starts.cpu(),
            ends.cpu(),
        )

    torch.testing.assert_close(run().cpu(), expected(), atol=0.02, rtol=0.02)
    selected("sparse_indexer_mqa_logits", "default.flagos")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    q.neg_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(result.cpu(), expected(), atol=0.02, rtol=0.02)
    for decode in (False, True):
        name = "sparse_indexer_topk_decode" if decode else "sparse_indexer_topk_prefill"
        indices = torch.empty(4, 8, device="cuda", dtype=torch.int32)
        logits = torch.randperm(4 * 64, device="cuda").reshape(4, 64).float()
        topk = CachedOp(name)

        def topk_run(topk=topk, logits=logits, decode=decode, indices=indices):
            topk(logits, 1 if decode else starts, ends, indices, 4, 64, 1, 8)

        topk_run()
        selected(name, "default.flagos")
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            topk_run()
        logits.neg_()
        capture.replay()
        torch.cuda.synchronize()
        ref = _torch_topk(
            logits.cpu(),
            torch.zeros_like(starts.cpu()) if decode else starts.cpu(),
            ends.cpu(),
            8,
            not decode,
        )
        torch.testing.assert_close(indices.cpu().sort(-1).values, ref.sort(-1).values)
    lengths = torch.tensor([2, 1, 3], device="cuda", dtype=torch.int32)
    data = torch.arange(24, device="cuda", dtype=torch.float32).reshape(6, 4)
    packed = CachedOp("sparse_indexer_pack_seq")(data, lengths)
    unpacked = CachedOp("sparse_indexer_unpack_seq")(packed, lengths)
    selected("sparse_indexer_pack_seq", "default.flagos")
    selected("sparse_indexer_unpack_seq", "default.flagos")
    torch.testing.assert_close(unpacked, data)


@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_query_storage_dispatch_and_graph(output_dtype):
    q = torch.linspace(-4, 4, 4 * 128).reshape(4, 128).to("cuda", torch.bfloat16)
    name = "sparse_indexer_prepare_query"
    op = CachedOp(name)

    from vllm_fl.dispatch.backends.flaggems.impl.sparse_indexer import (
        supports_native_e4m3,
        supports_prepare_query,
    )

    if output_dtype == torch.float8_e4m3fn and not supports_native_e4m3(q.device):
        assert not supports_prepare_query(q, 128, output_dtype=output_dtype)
        with pytest.raises(ValueError, match="PPU identity-scale"):
            op(q, 128, output_dtype=output_dtype)
        return  # Negative capability test; not an FP8 numerical pass.

    def run():
        return op(q, 128, output_dtype=output_dtype)

    def verify(values, scales):
        if output_dtype == torch.bfloat16:
            torch.testing.assert_close(values, q)
            torch.testing.assert_close(scales.cpu(), torch.ones(4, 1))
        else:
            ref_scale = (
                q.cpu().float().abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448
            )
            reference = (q.cpu().float() / ref_scale).to(torch.float8_e4m3fn).float()
            torch.testing.assert_close(scales.cpu(), ref_scale, rtol=1e-5, atol=1e-7)
            torch.testing.assert_close(values.cpu().float(), reference, rtol=0, atol=0)

    verify(*run())
    selected(
        name,
        "default.flagos" if output_dtype == torch.float8_e4m3fn else "vendor.thead",
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        values, scales = run()
    q.neg_()
    graph.replay()
    torch.cuda.synchronize()
    verify(values, scales)


def test_flaggems_cache_writer_graph():
    k = torch.linspace(-3, 3, 4 * 128).reshape(4, 128).to("cuda", torch.bfloat16)
    cache = torch.zeros(4, 64, 132, device="cuda", dtype=torch.uint8)
    slots = torch.tensor([2, 64, -1, 130], device="cuda", dtype=torch.int64)
    op = CachedOp("sparse_indexer_indexer_k_quant_and_cache")

    from vllm_fl.dispatch.backends.flaggems.impl.sparse_indexer import (
        supports_cache_write,
        supports_native_e4m3,
    )

    if not supports_native_e4m3(k.device):
        assert not supports_cache_write(k, cache, slots, 128, None)
        with pytest.raises(NotImplementedError, match="PPU BF16 indexer"):
            op(k, cache, slots, 128, None)
        return  # Actual arithmetic needs a compatible FP8 device; PPU uses BF16.

    def run():
        op(k, cache, slots, 128, None)

    def verify():
        flat = cache.cpu().view(4, -1)
        values = (
            flat[:, : 64 * 128]
            .contiguous()
            .view(torch.float8_e4m3fn)
            .reshape(256, 128)
            .float()
        )
        scales = flat[:, 64 * 128 :].contiguous().view(torch.float32).reshape(256, 1)
        ref_scales = k.cpu().float().abs().amax(-1, keepdim=True).clamp_min(1e-4) / 448
        ref_values = (k.cpu().float() / ref_scales).to(torch.float8_e4m3fn).float()
        for row, slot in enumerate(slots.cpu().tolist()):
            if slot >= 0:
                torch.testing.assert_close(
                    values[slot], ref_values[row], rtol=0, atol=0
                )
                torch.testing.assert_close(
                    scales[slot], ref_scales[row], rtol=1e-5, atol=1e-7
                )
        untouched = [i for i in range(256) if i not in slots.cpu().tolist()]
        assert (
            not values[untouched].count_nonzero()
            and not scales[untouched].count_nonzero()
        )

    run()
    selected("sparse_indexer_indexer_k_quant_and_cache", "default.flagos")
    verify()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    k.neg_()
    graph.replay()
    torch.cuda.synchronize()
    verify()


def test_kda_empty_padded_request_does_not_access_sentinel_state():
    q = torch.ones(1, 3, 4, 128, device="cuda", dtype=torch.bfloat16)
    g = torch.full(q.shape, -0.5, device="cuda")
    beta = torch.full(q.shape[:3], 0.5, device="cuda")
    state = torch.zeros(2, 4, 128, 128, device="cuda")
    baseline = state.clone()
    ids = torch.tensor([1, 0, -1], device="cuda", dtype=torch.int32)
    cu = torch.tensor([0, 1, 2, 2], device="cuda", dtype=torch.int32)
    op = CachedOp("fused_recurrent_kda")

    def run():
        return op(
            q,
            q,
            q,
            g,
            beta=beta,
            initial_state=state,
            cu_seqlens=cu,
            ssm_state_indices=ids,
        )

    expected, expected_state = kda_reference(
        q[:, :2], q[:, :2], q[:, :2], g[:, :2], beta[:, :2], baseline, ids[:2]
    )
    run()
    selected("fused_recurrent_kda", "default.flagos")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, _ = run()
    state.copy_(baseline)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out[:, :2].cpu(), expected, atol=0.004, rtol=0.02)
    torch.testing.assert_close(state.cpu(), expected_state, atol=2e-5, rtol=2e-4)
    assert not out[:, 2].count_nonzero()

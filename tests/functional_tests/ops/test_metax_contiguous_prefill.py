# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.

"""MetaX single-prefill layout, numerical, mixed-batch and decode replay tests."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def maca_attention():
    from vllm.platforms import current_platform

    if current_platform.vendor_name != "metax":
        pytest.skip("MetaX vendor FlashAttention integration")
    from vllm_fl.dispatch.backends.vendor.metax.impl.attention import flash_attn

    return flash_attn


@pytest.fixture(scope="module")
def device(maca_attention):
    if not torch.cuda.is_available():
        pytest.skip("MetaX GPU required")
    return torch.device("cuda:0")


def _impl(module):
    impl = module.FlashAttentionImpl(
        num_heads=12, head_size=256, scale=256 ** -0.5, num_kv_heads=2,
        alibi_slopes=None, sliding_window=None, kv_cache_dtype="auto",
    )
    impl.dcp_world_size = 1
    return impl


def _case(device, dtype, block_size, num_prefills=1, decode=False):
    torch.manual_seed(71)
    qlens = ([1] if decode else []) + [17, 9][:num_prefills]
    lengths = [block_size + n for n in qlens]
    rows = len(qlens)
    # Non-monotonic pages and a partial final page catch incorrect flattening.
    pages = torch.tensor(
        [[7, 2, 4], [5, 1, 8], [9, 0, 6]][:rows], device=device, dtype=torch.int32,
    )
    cache = torch.randn(
        2, 10, block_size, 2, 256, device=device, dtype=dtype,
    ) * 0.1
    # Q is a strided view, as in the model's mixed QKV projection.
    mixed = torch.randn(sum(qlens), 16, 256, device=device, dtype=dtype) * 0.1
    query = mixed[:, :12]
    starts = [0]
    for n in qlens:
        starts.append(starts[-1] + n)
    nd = int(decode)
    prefill_starts = [0]
    for n in qlens[nd:]:
        prefill_starts.append(prefill_starts[-1] + n)
    i32 = lambda values: torch.tensor(values, device=device, dtype=torch.int32)
    metadata = SimpleNamespace(
        num_actual_tokens=sum(qlens), query_start_loc=i32(starts),
        seq_lens=i32(lengths), max_query_len=max(qlens),
        max_seq_len=max(lengths), block_table=pages,
        scheduler_metadata=None, use_cascade=False, causal=True,
        num_decode_tokens=nd, num_decodes=nd,
        decode_block_table=pages[:nd], decode_seq_lens=i32(lengths[:nd]),
        num_prefills=num_prefills, prefill_block_table=pages[nd:],
        prefill_seq_lens=i32(lengths[nd:]),
        prefill_query_start_loc=i32(prefill_starts),
        prefill_max_seq_len=max(lengths[nd:], default=0),
    )
    return query, cache, metadata, qlens, lengths, pages


def _reference(query, cache, qlens, lengths, pages):
    result = []
    offset = 0
    for row, (qlen, length) in enumerate(zip(qlens, lengths)):
        key = cache[0].index_select(0, pages[row].long()).flatten(0, 1)[:length]
        value = cache[1].index_select(0, pages[row].long()).flatten(0, 1)[:length]
        q = query[offset:offset + qlen].float().transpose(0, 1)
        k = key.float().repeat_interleave(6, dim=1).transpose(0, 1)
        v = value.float().repeat_interleave(6, dim=1).transpose(0, 1)
        scores = (q @ k.transpose(-1, -2)) * 256 ** -0.5
        qi = torch.arange(qlen, device=query.device)[:, None] + length - qlen
        ki = torch.arange(length, device=query.device)[None, :]
        scores.masked_fill_(ki > qi, float("-inf"))
        result.append((scores.softmax(-1) @ v).transpose(0, 1))
        offset += qlen
    return torch.cat(result).to(query.dtype)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize(
    "setting,num_prefills,decode,contiguous",
    [(None, 1, False, False), ("0", 1, False, False),
     ("1", 1, False, True), (" ON ", 1, True, True),
     ("1", 2, True, False)],
)
def test_prefill_forward_layout_and_numerics(
    maca_attention, device, monkeypatch, dtype, block_size,
    setting, num_prefills, decode, contiguous,
):
    module = maca_attention
    if setting is None:
        monkeypatch.delenv("VLLM020_CONTIGUOUS_SINGLE_PREFILL", raising=False)
    else:
        monkeypatch.setenv("VLLM020_CONTIGUOUS_SINGLE_PREFILL", setting)
    query, cache, metadata, qlens, lengths, pages = _case(
        device, dtype, block_size, num_prefills, decode,
    )
    expected = _reference(query, cache, qlens, lengths, pages)
    calls = []
    real_flash = module.flash_attn_varlen_func

    def observed_flash(**kwargs):
        # Observe routing without replacing the real device computation.
        calls.append((kwargs["k"].dim(), "block_table" in kwargs))
        return real_flash(**kwargs)

    monkeypatch.setattr(module, "flash_attn_varlen_func", observed_flash)
    output = torch.full_like(query, float("nan"))
    original_cache = cache.clone()
    impl = _impl(module)
    layer = SimpleNamespace()
    with torch.inference_mode():
        returned = impl.forward(layer, query, None, None, cache, metadata, output)
    assert returned is output
    assert calls == ([(3, False)] if contiguous else [(4, True)])
    torch.testing.assert_close(output, expected, atol=0.002, rtol=0.02)
    torch.testing.assert_close(cache, original_cache, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_contiguous_non_power_of_two_pages(maca_attention, device, monkeypatch, dtype):
    # The dense-KV fast path does not inherit paged FA2's power-of-two limit.
    test_prefill_forward_layout_and_numerics(
        maca_attention, device, monkeypatch, dtype, 784, "1", 1, False, True,
    )


def test_decode_capture_changed_input_replay(maca_attention, device, monkeypatch):
    """The opt-in prefill branch must not change the FULL decode path."""
    monkeypatch.setenv("VLLM020_CONTIGUOUS_SINGLE_PREFILL", "1")
    query, cache, metadata, qlens, lengths, pages = _case(
        device, torch.bfloat16, 16, num_prefills=0, decode=True,
    )
    impl = _impl(maca_attention)
    layer = SimpleNamespace()
    output = torch.empty_like(query)

    def forward():
        return impl.forward(layer, query, None, None, cache, metadata, output)

    with torch.inference_mode():
        forward()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            forward()
            forward()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward()
        previous = None
        for _ in range(2):
            query.copy_(torch.randn_like(query) * 0.1)
            cache.copy_(torch.randn_like(cache) * 0.1)
            expected = _reference(query, cache, qlens, lengths, pages)
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(output, expected, atol=0.002, rtol=0.02)
            if previous is not None:
                assert not torch.equal(output, previous)
            previous = output.clone()

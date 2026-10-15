# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.

"""Numerical regression for hybrid paged state conversion and GDN replay."""
import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize('padding', [0, 8192])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_state_transfer_preserves_strides_padding_and_null(device, padding, dtype):
    from vllm_fl.dispatch.backends.flaggems.impl.gdn_state_layout import (
        pack_gdn_state, unpack_gdn_state,
    )
    heads, values, keys, pages = 4, 96, 64, 6
    width = heads * values * keys
    storage = torch.randn(pages, width + padding, device=device, dtype=dtype)
    cache = storage[:, :width].view(pages, heads, values, keys)
    original = storage.clone()
    indices = torch.tensor([3, 1, 0, 0], device=device, dtype=torch.int32)
    compact = pack_gdn_state(cache, indices)
    expected = torch.zeros_like(compact)
    expected[1] = cache[3].transpose(-1, -2)
    expected[2] = cache[1].transpose(-1, -2)
    torch.testing.assert_close(compact, expected, atol=0, rtol=0)
    replacement = torch.randn_like(compact)
    unpack_gdn_state(replacement, cache, indices)
    expected_storage = original.clone()
    expected_storage[3, :width] = replacement[1].transpose(-1, -2).reshape(-1)
    expected_storage[1, :width] = replacement[2].transpose(-1, -2).reshape(-1)
    torch.testing.assert_close(storage, expected_storage, atol=0, rtol=0)
    assert cache.stride(0) == width + padding
    assert cache.data_ptr() == storage.data_ptr()


@pytest.mark.parametrize('padding', [0, 8192])
def test_metax_gdn_state_eager_and_changed_input_replay(device, padding):
    import flag_gems
    from flaggems_vllm import vendor_name
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend
    if vendor_name != 'metax':
        pytest.skip('MetaX-specific FlagGems-vLLM recurrent ABI adapter')
    torch.manual_seed(41)
    rows, heads, groups, keys, values, pages = 4, 8, 3, 128, 128, 6
    hv = heads * groups
    # Real model Q/K/V are strided views of a mixed projection.
    mixed = torch.randn(rows, (2 * heads + hv) * keys, device=device, dtype=torch.bfloat16)
    q0, k0, v0 = mixed.split([heads * keys, heads * keys, hv * values], dim=-1)
    q = q0.view(1, rows, heads, keys)
    k = k0.view(1, rows, heads, keys)
    v = v0.view(1, rows, hv, values)
    a = torch.randn(rows, hv, device=device, dtype=torch.bfloat16)
    b = torch.randn_like(a)
    A_log = torch.randn(hv, device=device)
    dt_bias = torch.randn_like(A_log)
    width = hv * values * keys
    storage = torch.randn(pages, width + padding, device=device) * .01
    state = storage[:, :width].view(pages, hv, values, keys)
    initial = storage.clone()
    indices = torch.tensor([1, 3, 2, 0], device=device, dtype=torch.int32)
    cu = torch.tensor([0, 1, 2, 3, 3], device=device, dtype=torch.int32)
    backend = FlagGemsBackend()
    kwargs = dict(A_log=A_log, a=a, b=b, dt_bias=dt_bias, q=q, k=k, v=v,
                  initial_state=state, inplace_final_state=True, cu_seqlens=cu,
                  ssm_state_indices=indices, use_qk_l2norm_in_kernel=True)

    def reference():
        expected_storage = initial.clone()
        updated = expected_storage[:, :width].view(pages, hv, values, keys)
        result = torch.zeros_like(v)
        qq = (q[0].float() / (q[0].float().square().sum(-1, keepdim=True) + 1e-6).sqrt()).repeat_interleave(groups, dim=1)
        kk = (k[0].float() / (k[0].float().square().sum(-1, keepdim=True) + 1e-6).sqrt()).repeat_interleave(groups, dim=1)
        g = -A_log.exp() * F.softplus(a.float() + dt_bias)
        beta = b.float().sigmoid()
        for token, slot in enumerate(indices.cpu().tolist()):
            if slot == 0:
                continue
            h = updated[slot] * g[token].exp()[:, None, None]
            delta = (v[0, token].float() - (h * kk[token, :, None, :]).sum(-1)) * beta[token, :, None]
            h += delta[:, :, None] * kk[token, :, None, :]
            updated[slot] = h
            result[0, token] = (h * qq[token, :, None, :]).sum(-1) * keys ** -.5
        return result, expected_storage

    def check(out):
        ref, expected_storage = reference()
        torch.testing.assert_close(out, ref, atol=.002, rtol=.02)
        torch.testing.assert_close(storage, expected_storage, atol=.002, rtol=.02)
        torch.testing.assert_close(storage[0], initial[0], atol=0, rtol=0)
        torch.testing.assert_close(storage[:, width:], initial[:, width:], atol=0, rtol=0)

    with torch.inference_mode():
        with flag_gems.use_gems():
            out, returned_state = backend.gdn_decode(**kwargs)
        assert returned_state is state
        check(out)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), flag_gems.use_gems():
            backend.gdn_decode(**kwargs)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), flag_gems.use_gems():
            out, _ = backend.gdn_decode(**kwargs)
        for step in range(2):
            mixed.copy_(torch.randn_like(mixed))
            a.copy_(torch.randn_like(a)); b.copy_(torch.randn_like(b))
            indices.copy_(torch.tensor([3, 1, 2, 0] if step else [1, 3, 2, 0], device=device, dtype=torch.int32))
            storage.copy_(initial)
            graph.replay()
            torch.cuda.synchronize()
            check(out)

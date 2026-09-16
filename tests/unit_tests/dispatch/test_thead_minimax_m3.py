# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlagGems tests for MiniMax-M3 fused preprocessing on PPU.

The reference equations mirror the upstream vLLM kernel contract.  This test
intentionally covers the BF16 ``auto`` cache path accepted for PPU.
"""

import os

import pytest
import torch

HEAD_DIM = 128
ROTARY_DIM = 64

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or "PPU_SDK" not in os.environ,
    reason="T-Head PPU is not available",
)


def _make_cos_sin(max_pos=64):
    inv_freq = 1.0 / (
        5_000_000.0
        ** (torch.arange(0, ROTARY_DIM, 2, device="cuda").float() / ROTARY_DIM)
    )
    positions = torch.arange(max_pos, device="cuda").float()
    frequencies = torch.einsum("i,j->ij", positions, inv_freq)
    return torch.cat((frequencies.cos(), frequencies.sin()), dim=-1).bfloat16()


def _reference(x, weight, positions, cos_sin):
    x_float = x.float()
    x_float = x_float * torch.rsqrt(
        x_float.square().mean(dim=-1, keepdim=True) + 1e-6
    )
    x_float = x_float * (1.0 + weight.float())
    half = ROTARY_DIM // 2
    selected = cos_sin[positions].float()
    cos = selected[:, None, :half]
    sin = selected[:, None, half:]
    first = x_float[..., :half]
    second = x_float[..., half:ROTARY_DIM]
    out = x_float.clone()
    out[..., :half] = first * cos - second * sin
    out[..., half:ROTARY_DIM] = second * cos + first * sin
    return out.bfloat16()


def _patched_op():
    from vllm import _custom_ops as ops
    from vllm.plugins import load_general_plugins

    load_general_plugins()
    op = ops.fused_minimax_m3_qknorm_rope_kv_insert
    assert getattr(op, "_vllm_fl_dispatch_patch", False)
    return op


def test_patch_does_not_synthesize_private_vllm_abi():
    native_op_existed = hasattr(
        torch.ops._C, "fused_minimax_m3_qknorm_rope_kv_insert"
    )
    _patched_op()
    assert (
        hasattr(torch.ops._C, "fused_minimax_m3_qknorm_rope_kv_insert")
        is native_op_existed
    )


def test_dense_matches_reference():
    fused_preprocess = _patched_op()
    torch.manual_seed(0)
    tokens, num_heads, num_kv_heads = 7, 8, 2
    q_size = num_heads * HEAD_DIM
    kv_size = num_kv_heads * HEAD_DIM
    qkv = torch.randn(
        tokens,
        q_size + 2 * kv_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    original = qkv.clone()
    q_weight = torch.randn(HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 0.1
    k_weight = torch.randn(HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 0.1
    cos_sin = _make_cos_sin()
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64)

    fused_preprocess(
        qkv,
        q_weight,
        k_weight,
        cos_sin,
        positions,
        num_heads,
        num_kv_heads,
        ROTARY_DIM,
        1e-6,
        None,
        None,
        0,
        None,
        None,
        None,
        None,
        0,
        None,
        None,
        "auto",
    )

    q, k, value = qkv.split((q_size, kv_size, kv_size), dim=-1)
    q0, k0, value0 = original.split((q_size, kv_size, kv_size), dim=-1)
    q_ref = _reference(
        q0.view(tokens, num_heads, HEAD_DIM), q_weight, positions, cos_sin
    ).view(tokens, q_size)
    k_ref = _reference(
        k0.view(tokens, num_kv_heads, HEAD_DIM), k_weight, positions, cos_sin
    ).view(tokens, kv_size)
    torch.testing.assert_close(q, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k, k_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(value, value0, rtol=0, atol=0)


def test_sparse_outputs_and_cache_inserts():
    fused_preprocess = _patched_op()
    torch.manual_seed(1)
    tokens = 7
    num_heads, num_kv_heads, num_index_heads = 8, 2, 2
    q_size = num_heads * HEAD_DIM
    kv_size = num_kv_heads * HEAD_DIM
    index_q_size = num_index_heads * HEAD_DIM
    sizes = (q_size, kv_size, kv_size, index_q_size, HEAD_DIM)
    qkv = torch.randn(
        tokens,
        sum(sizes),
        device="cuda",
        dtype=torch.bfloat16,
    )
    original = qkv.clone()
    weights = [
        torch.randn(HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 0.1
        for _ in range(4)
    ]
    cos_sin = _make_cos_sin()
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64)
    block_size = 128
    slots = torch.tensor([2, 4, -1, 130, 131, 5, 6], device="cuda")
    index_slots = torch.tensor([9, 8, -1, 133, 134, 7, 3], device="cuda")
    kv_cache = torch.zeros(
        2,
        2,
        block_size,
        num_kv_heads,
        HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    index_cache = torch.zeros(
        2, block_size, HEAD_DIM, device="cuda", dtype=torch.bfloat16
    )
    q_out = torch.empty(tokens, q_size, device="cuda", dtype=torch.bfloat16)
    index_q_out = torch.empty(
        tokens, index_q_size, device="cuda", dtype=torch.bfloat16
    )

    fused_preprocess(
        qkv,
        weights[0],
        weights[1],
        cos_sin,
        positions,
        num_heads,
        num_kv_heads,
        ROTARY_DIM,
        1e-6,
        weights[2],
        weights[3],
        num_index_heads,
        slots,
        index_slots,
        kv_cache,
        index_cache,
        block_size,
        q_out,
        index_q_out,
        "auto",
    )

    _, k, value, _, index_k = qkv.split(sizes, dim=-1)
    q0, k0, value0, index_q0, index_k0 = original.split(sizes, dim=-1)
    expected = (
        _reference(q0.view(tokens, num_heads, HEAD_DIM), weights[0], positions, cos_sin),
        _reference(k0.view(tokens, num_kv_heads, HEAD_DIM), weights[1], positions, cos_sin),
        _reference(
            index_q0.view(tokens, num_index_heads, HEAD_DIM),
            weights[2],
            positions,
            cos_sin,
        ),
        _reference(index_k0.view(tokens, 1, HEAD_DIM), weights[3], positions, cos_sin),
    )
    torch.testing.assert_close(q_out, expected[0].view_as(q_out), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k, expected[1].view_as(k), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(index_q_out, expected[2].view_as(index_q_out), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(index_k, expected[3].view_as(index_k), rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(value, value0, rtol=0, atol=0)
    for token, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        block, offset = divmod(slot, block_size)
        torch.testing.assert_close(
            kv_cache[block, 0, offset],
            k[token].view(num_kv_heads, HEAD_DIM),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            kv_cache[block, 1, offset],
            value0[token].view(num_kv_heads, HEAD_DIM),
            rtol=0,
            atol=0,
        )
    for token, slot in enumerate(index_slots.tolist()):
        if slot >= 0:
            block, offset = divmod(slot, block_size)
            torch.testing.assert_close(
                index_cache[block, offset], index_k[token], rtol=0, atol=0
            )


def test_dense_cuda_graph_replay():
    fused_preprocess = _patched_op()
    if not hasattr(torch.cuda, "CUDAGraph"):
        pytest.skip("CUDA-compatible graph capture is unavailable")

    torch.manual_seed(2)
    tokens, num_heads, num_kv_heads = 4, 4, 2
    q_size = num_heads * HEAD_DIM
    kv_size = num_kv_heads * HEAD_DIM
    qkv = torch.randn(
        tokens,
        q_size + 2 * kv_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    q_weight = torch.randn(HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 0.1
    k_weight = torch.randn(HEAD_DIM, device="cuda", dtype=torch.bfloat16) * 0.1
    cos_sin = _make_cos_sin()
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64)

    def invoke():
        fused_preprocess(
            qkv,
            q_weight,
            k_weight,
            cos_sin,
            positions,
            num_heads,
            num_kv_heads,
            ROTARY_DIM,
            1e-6,
            None,
            None,
            0,
            None,
            None,
            None,
            None,
            0,
            None,
            None,
            "auto",
        )

    # Warm up Triton outside capture, then restore the static input buffer.
    initial = qkv.clone()
    invoke()
    torch.cuda.synchronize()
    qkv.copy_(initial)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke()
    torch.cuda.synchronize()

    replay_input = torch.randn_like(qkv)
    qkv.copy_(replay_input)
    graph.replay()
    torch.cuda.synchronize()

    q, k, value = qkv.split((q_size, kv_size, kv_size), dim=-1)
    q0, k0, value0 = replay_input.split((q_size, kv_size, kv_size), dim=-1)
    q_ref = _reference(
        q0.view(tokens, num_heads, HEAD_DIM), q_weight, positions, cos_sin
    ).view(tokens, q_size)
    k_ref = _reference(
        k0.view(tokens, num_kv_heads, HEAD_DIM), k_weight, positions, cos_sin
    ).view(tokens, kv_size)
    torch.testing.assert_close(q, q_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(k, k_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(value, value0, rtol=0, atol=0)

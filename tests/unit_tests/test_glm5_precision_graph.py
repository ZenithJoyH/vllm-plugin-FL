# SPDX-License-Identifier: Apache-2.0
"""Small independent math references; not a full-model accuracy evaluation."""

import pytest
import torch
from types import SimpleNamespace
from vllm_fl.dispatch import CachedOp, policy_context, SelectionPolicy


def hadamard_matrix(n):
    h = torch.ones(1, 1)
    while h.shape[0] < n:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    return h / n**0.5


def test_all_capture_shapes_keep_live_block_tables():
    from vllm_fl.kernels.glm5_next.graph_metadata import update_compressed_block_table

    owner = SimpleNamespace()
    captures = []
    for batch in (1, 2, 4, 8, 16, 32):
        table = torch.arange(batch * 16, dtype=torch.int32).reshape(batch, 16).cuda()
        buffer = update_compressed_block_table(owner, table, 4)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = update_compressed_block_table(owner, table, 4)
        assert buffer.data_ptr() == captured.data_ptr()
        captures.append((table, captured, graph))
    assert len(owner._glm5_indexer_block_tables) == 6
    for table, captured, graph in captures:
        table.add_(64)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured.cpu(), table.cpu()[:, ::4] // 4)


@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16, 32])
def test_query_rotation_graph(batch):
    from vllm_fl.kernels.glm5_next.indexer_backend import INDEXER_BACKEND

    data = torch.randn(batch, 32, 128, generator=torch.Generator().manual_seed(33))
    q = data.to("cuda", torch.bfloat16)
    expected = (q.cpu().float() @ hadamard_matrix(128)).to(q.dtype)
    eager = INDEXER_BACKEND.rotate_indexer_query(q)
    torch.testing.assert_close(eager.cpu(), expected, rtol=0.01, atol=0.016)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = INDEXER_BACKEND.rotate_indexer_query(q)
    q.neg_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual.cpu(), -expected, rtol=0.01, atol=0.016)


@pytest.mark.parametrize("batch", [1, 32])
def test_mhc_norm_eager_graph(batch):
    # Zero projection has analytically known, uniform mixtures. Non-unit norm
    # weights catch both missing normalization and double-normalization.
    cpu = torch.randn(batch, 4, 4096, generator=torch.Generator().manual_seed(91)).to(
        torch.bfloat16
    )
    residual = cpu.cuda()
    fn = torch.zeros(24, 16384, device="cuda", dtype=torch.float32)
    scale = torch.ones(3, device="cuda", dtype=torch.float32)
    base = torch.zeros(24, device="cuda", dtype=torch.float32)
    norm = torch.full((4096,), 2.0, device="cuda", dtype=torch.float32)
    args = (fn, scale, base, 1e-5, 1e-6, 1e-6, 2.0, 20)
    # Compare to upstream's independent torch equations plus one explicit norm.
    from vllm_fl.dispatch.backends.reference.impl.glm5_mhc import mhc_pre

    expected = mhc_pre(
        cpu,
        fn.cpu(),
        scale.cpu(),
        base.cpu(),
        *args[3:],
        norm_weight=norm.cpu(),
        norm_eps=1e-5,
    )
    op = CachedOp("glm5_mhc_pre")
    eager = op(residual, *args, norm_weight=norm, norm_eps=1e-5)
    for a, b in zip(eager, expected):
        torch.testing.assert_close(a.cpu(), b, atol=0.032, rtol=0.02)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = op(residual, *args, norm_weight=norm, norm_eps=1e-5)
    residual.neg_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual[2].cpu(), -expected[2], atol=0.032, rtol=0.02)


def test_model_local_bounded_activation_graph():
    from vllm_fl.ops.glm5_next import SiluAndMulWithClamp

    with policy_context(
        SelectionPolicy.from_dict(
            per_op_order={"silu_and_mul_with_clamp": ["vendor:thead"]}
        )
    ):
        x = torch.linspace(-20, 20, 8192).reshape(1, -1).to("cuda", torch.bfloat16)
        op = SiluAndMulWithClamp(10.0)
        op(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = op(x)
        x.neg_()
        graph.replay()
        torch.cuda.synchronize()
        gate, up = x.cpu().chunk(2, -1)
        expected = torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
        torch.testing.assert_close(actual.cpu(), expected, atol=0.125, rtol=0.02)

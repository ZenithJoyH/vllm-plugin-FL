# SPDX-License-Identifier: Apache-2.0
"""CPU architecture/registration regressions; no model weights or service."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def load_file(relative):
    spec = importlib.util.spec_from_file_location("glm5_test_module", ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_has_no_vendor_or_experiment_imports():
    paths = [ROOT / "vllm_fl/models/glm5_next.py", ROOT / "vllm_fl/ops/glm5_next.py"]
    for path in paths:
        source = path.read_text()
        assert "glm53_" not in source
        assert "provider" not in source
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom):
                assert not any(
                    token in (node.module or "")
                    for token in ("vendor.", "flag_gems", "_ppu")
                )


def test_no_global_mhc_replacement():
    source = (ROOT / "vllm_fl/patches/glm5_next_v024.py").read_text()
    assert "forward_oot =" not in source
    assert "concat_mla_q =" not in source
    assert "VLLM_FL_GLM5_" not in source


def test_shared_indexer_has_no_vendor_kernel_or_skip_write_switches():
    source = (
        ROOT / "vllm_fl/kernels/glm5_next/sparse_attn_indexer_kpool.py"
    ).read_text()
    assert "VLLM_KPOOL_SKIP_" not in source
    assert "torch.ops._C." not in source
    assert "rocm_aiter" not in source
    assert "INDEXER_BACKEND.is_nvidia" not in source


def test_reference_topk_contract_accepts_vendor_neutral_metadata():
    from vllm_fl.dispatch.backends.reference.impl.sparse_indexer import topk_decode

    logits = torch.tensor([[1.0, 9.0, 3.0, 7.0], [8.0, 2.0, 6.0, 5.0]])
    output = torch.empty(2, 2, dtype=torch.int32)
    topk_decode(logits, 1, torch.tensor([3, 4]), output, 2, 4, 1, 2, max_seq_len=100)
    torch.testing.assert_close(
        output, torch.tensor([[1, 2], [0, 2]], dtype=torch.int32)
    )


@pytest.mark.parametrize(
    "top_k,max_seq_len,expected", [(512, 1024, "persistent"), (2, 1024, "row")]
)
def test_cuda_topk_backend_preserves_selection(
    monkeypatch, top_k, max_seq_len, expected
):
    import vllm.v1.worker.workspace as workspace

    from vllm_fl.dispatch.backends.vendor.cuda.impl.sparse_indexer import IndexerOps

    calls = []
    monkeypatch.setattr(
        torch.ops,
        "_C",
        SimpleNamespace(
            persistent_topk=lambda *args: calls.append("persistent"),
            top_k_per_row_decode=lambda *args: calls.append("row"),
        ),
    )
    monkeypatch.setattr(
        workspace,
        "current_workspace_manager",
        lambda: SimpleNamespace(
            get_simultaneous=lambda *args: (torch.empty(1, dtype=torch.uint8),)
        ),
    )
    IndexerOps().topk_decode(
        torch.empty(1, 1024),
        1,
        torch.tensor([1024]),
        torch.empty(1, top_k, dtype=torch.int32),
        1,
        1024,
        1,
        top_k,
        max_seq_len=max_seq_len,
    )
    assert calls == [expected]


@pytest.mark.parametrize("vendor", ["cuda", "thead"])
def test_indexer_contract_registered_lazily(vendor):
    from vllm_fl.dispatch.backends.model_ops import (
        INDEXER_OPS,
        REFERENCE_INDEXER,
        register_vendor_glm5,
    )
    from vllm_fl.dispatch.registry import OpRegistry

    registry = OpRegistry()
    register_vendor_glm5(registry, SimpleNamespace(is_available=lambda: True), vendor)
    entries = registry.snapshot().impls_by_op
    for name in INDEXER_OPS:
        if vendor == "thead" and name in REFERENCE_INDEXER:
            assert "sparse_indexer_" + name not in entries
            continue
        (impl,) = entries["sparse_indexer_" + name]
        assert impl.vendor == vendor
        assert impl.is_available()


@pytest.mark.parametrize(
    "vendor,major,expected",
    [("thead", 8, 32), ("thead", None, 32), ("cuda", 9, 64), ("cuda", 10, 32)],
)
def test_compressed_pages_follow_vendor_contract(vendor, major, expected):
    from vllm_fl.dispatch.backends.model_ops import compressed_page_size

    platform = SimpleNamespace(
        get_device_capability=lambda: SimpleNamespace(major=major)
    )
    assert compressed_page_size(vendor, platform) == expected


def test_unknown_vendor_does_not_inherit_ppu_layout():
    from vllm_fl.dispatch.backends.model_ops import compressed_page_size

    with pytest.raises(NotImplementedError):
        compressed_page_size("unsupported", SimpleNamespace())


def test_blacklist_covers_fused_mhc_dependencies(monkeypatch):
    import vllm_fl.utils
    from vllm_fl.dispatch.backends.model_ops import register_mhc
    from vllm_fl.dispatch.registry import OpRegistry
    from vllm_fl.dispatch.types import BackendImplKind

    monkeypatch.setattr(
        vllm_fl.utils, "use_flaggems_op", lambda name: name != "mhc_post"
    )
    registry = OpRegistry()
    register_mhc(
        registry, SimpleNamespace(is_available=lambda: True), BackendImplKind.DEFAULT
    )
    entries = registry.snapshot().impls_by_op
    assert "mhc_pre_with_norm" in entries
    assert "mhc_post" not in entries
    assert "mhc_fused_post_pre_with_norm" not in entries


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bounded_activation_matches_definition(dtype):
    ref = load_file("vllm_fl/dispatch/backends/reference/impl/mhc.py")
    x = torch.linspace(-40, 40, 128).reshape(4, 32).to(dtype)
    gate, up = x.chunk(2, -1)
    gate = gate.clamp(max=10)
    expected = torch.nn.functional.silu(gate) * up.clamp(-10, 10)
    torch.testing.assert_close(
        ref.silu_and_mul_with_clamp(x, 10), expected, atol=0.125, rtol=0.01
    )


def test_config_roundtrip():
    config = load_file("vllm_fl/configs/glm5_next.py")
    text = config.Glm5NextTextConfig()
    assert text.layer_types.count("linear_attention") == 34
    assert text.layer_types.count("deepseek_sparse_attention") == 11
    restored = config.Glm5NextTextConfig(**text.to_dict())
    assert restored.layer_types == text.layer_types


def test_config_validation_is_local():
    import transformers.configuration_utils as upstream

    allowed = upstream.ALLOWED_LAYER_TYPES
    config = load_file("vllm_fl/configs/glm5_next.py")
    config.Glm5NextTextConfig()
    assert allowed == upstream.ALLOWED_LAYER_TYPES
    with pytest.raises(ValueError):
        config.Glm5NextTextConfig(layer_types=["unknown"] * 45)


def test_v024_registration_and_model_import():
    from vllm.model_executor.layers.mhc import MHCPreOp

    import vllm_fl

    original = MHCPreOp.forward_oot
    vllm_fl.register_model()
    from vllm_fl.models.glm5_next import Glm5NextForCausalLM
    from vllm_fl.models.glm5_next_multimodal import Glm5NextForConditionalGeneration

    assert Glm5NextForCausalLM and Glm5NextForConditionalGeneration
    assert MHCPreOp.forward_oot is original


def test_moe_clamp_and_unclamped_paths(monkeypatch):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    from vllm_fl.ops.fused_moe import activation

    ref = load_file("vllm_fl/dispatch/backends/reference/impl/mhc.py")
    monkeypatch.setattr(
        activation, "_silu_and_mul_with_clamp", ref.silu_and_mul_with_clamp
    )
    monkeypatch.setattr(
        activation,
        "_silu_and_mul",
        lambda _, x: torch.nn.functional.silu(x.chunk(2, -1)[0]) * x.chunk(2, -1)[1],
    )
    x = torch.linspace(-40, 40, 128).reshape(4, 32)
    out = torch.empty(4, 16)
    activation.apply_moe_activation(MoEActivation.SILU, out, x)
    gate, up = x.chunk(2, -1)
    torch.testing.assert_close(out, torch.nn.functional.silu(gate) * up)
    activation.apply_moe_activation(MoEActivation.SILU, out, x, clamp_limit=10.0)
    torch.testing.assert_close(out, ref.silu_and_mul_with_clamp(x, 10.0))


def test_vendor_does_not_hide_flaggems_provider_selection():
    source = (
        ROOT / "vllm_fl/dispatch/backends/vendor/thead/impl/sparse_indexer.py"
    ).read_text()
    assert "flag_gems" not in source
    assert "_call_flag" not in source
    assert "_flag_ops" not in source


def test_portable_exports_resolve():
    from vllm_fl.kernels.glm5_next import portable

    assert all(hasattr(portable, name) for name in portable.__all__)


@pytest.mark.parametrize(
    "dependency,op",
    [
        ("hadamard_transform", "rotate_indexer_query"),
        ("cp_gather_indexer_k_quant_cache", "gather_cache"),
        ("per_token_group_quant_fp8", "prepare_query"),
    ],
)
def test_flaggems_dependency_blacklist(dependency, op, monkeypatch):
    import vllm_fl.utils
    from vllm_fl.dispatch.backends.model_ops import register_flaggems_model_ops
    from vllm_fl.dispatch.registry import OpRegistry

    monkeypatch.setattr(
        vllm_fl.utils, "use_flaggems_op", lambda name: name != dependency
    )
    registry = OpRegistry()
    register_flaggems_model_ops(registry, SimpleNamespace(is_available=lambda: True))
    assert "sparse_indexer_" + op not in registry.snapshot().impls_by_op


def test_pack_and_unpack_reject_capture_before_reading_lengths(monkeypatch):
    from vllm_fl.dispatch.backends.flaggems.impl.sparse_indexer import supports_pack
    from vllm_fl.dispatch.backends.reference.impl.sparse_indexer import (
        supports_pack as ref_guard,
    )

    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    tensor = SimpleNamespace(
        device=SimpleNamespace(type="cuda"), is_contiguous=lambda: True
    )
    assert not supports_pack(tensor, None)
    assert not ref_guard(tensor, None)


def test_runner_adapter_leaves_other_models_unchanged(monkeypatch):
    from vllm_fl.patches.glm5_next_runner_v024 import (
        ModelRunnerFL,
        install_glm5_runner_adapter,
    )

    expected = object()
    monkeypatch.setattr(
        ModelRunnerFL, "_reshape_kv_cache_tensors", lambda self, *a, **kw: expected
    )
    install_glm5_runner_adapter()
    once = ModelRunnerFL._reshape_kv_cache_tensors
    install_glm5_runner_adapter()
    assert ModelRunnerFL._reshape_kv_cache_tensors is once
    instance = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="other"))
    )
    assert once(instance, {}, []) is expected

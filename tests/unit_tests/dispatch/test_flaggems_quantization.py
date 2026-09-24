# Copyright (c) 2026 BAAI. All rights reserved.

import sys
from types import ModuleType
from unittest.mock import Mock

import pytest
import torch

from vllm_fl.dispatch.backends.flaggems.impl.quantization import (
    dynamic_per_token_quant_int8_flaggems_triton,
    dynamic_per_token_quant_int8_flaggems_vllm,
)


def _install_flaggems_vllm_quant_module(monkeypatch, quant_fn):
    package = ModuleType("flaggems_vllm")
    package.__path__ = []
    ops_package = ModuleType("flaggems_vllm.ops")
    ops_package.__path__ = []
    quant_module = ModuleType("flaggems_vllm.ops.scaled_int8_quant")
    quant_module.scaled_int8_quant = quant_fn

    monkeypatch.setitem(sys.modules, "flaggems_vllm", package)
    monkeypatch.setitem(sys.modules, "flaggems_vllm.ops", ops_package)
    monkeypatch.setitem(
        sys.modules,
        "flaggems_vllm.ops.scaled_int8_quant",
        quant_module,
    )


def test_dynamic_per_token_quant_prefers_flaggems_vllm(monkeypatch):
    x = torch.ones((2, 4), dtype=torch.bfloat16)
    expected_q = torch.ones_like(x, dtype=torch.int8)
    expected_scale = torch.full((2, 1), 0.25, dtype=torch.float32)
    calls = []

    def quant_fn(value, *, scale, azp, symmetric):
        calls.append((value, scale, azp, symmetric))
        return expected_q, expected_scale, None

    _install_flaggems_vllm_quant_module(monkeypatch, quant_fn)

    actual_q, actual_scale = dynamic_per_token_quant_int8_flaggems_vllm(x)

    assert calls == [(x, None, None, True)]
    assert actual_q is expected_q
    assert actual_scale is expected_scale


def test_dynamic_per_token_quant_propagates_failure_for_dispatch_fallback(
    monkeypatch,
):
    def unavailable(_value, *, scale, azp, symmetric):
        raise RuntimeError("FlagGems-vLLM kernel is unavailable")

    _install_flaggems_vllm_quant_module(monkeypatch, unavailable)

    with pytest.raises(RuntimeError, match="kernel is unavailable"):
        dynamic_per_token_quant_int8_flaggems_vllm(torch.ones((1, 4)))


def test_local_triton_quant_validates_input_contract():
    with pytest.raises(ValueError, match="2D"):
        dynamic_per_token_quant_int8_flaggems_triton(torch.ones((1, 2, 4)))
    with pytest.raises(TypeError, match="floating point"):
        dynamic_per_token_quant_int8_flaggems_triton(
            torch.ones((1, 4), dtype=torch.int8)
        )
    with pytest.raises(ValueError, match="hidden_size"):
        dynamic_per_token_quant_int8_flaggems_triton(torch.ones((1, 0)))


def test_flaggems_quantization_registers_ordered_fallbacks(monkeypatch):
    from vllm_fl.dispatch.backends.flaggems import register_ops
    from vllm_fl.dispatch.types import BackendPriority

    registered = []

    class Registry:
        def register_many(self, impls):
            registered.extend(impls)

    monkeypatch.setattr(
        register_ops,
        "use_flaggems_op",
        lambda op_name: op_name == "dynamic_per_token_quant_int8",
    )

    register_ops.register_builtins(Registry())

    assert [impl.impl_id for impl in registered] == [
        "default.flagos",
        "default.flagos_triton",
    ]
    assert [impl.priority for impl in registered] == [
        BackendPriority.DEFAULT + 10,
        BackendPriority.DEFAULT,
    ]


def test_minimax_and_cache_ops_register_on_default_flagos_backend(monkeypatch):
    from vllm_fl.dispatch.backends.flaggems import register_ops
    from vllm_fl.dispatch.types import BackendImplKind

    expected = {
        "apply_rotary_emb",
        "swigluoai_uninterleave",
        "fused_minimax_m3_qknorm_rope_kv_insert",
        "reshape_and_cache_flash",
    }
    registered = []

    class Registry:
        def register_many(self, impls):
            registered.extend(impls)

    monkeypatch.setattr(
        register_ops,
        "use_flaggems_op",
        lambda op_name: op_name in expected,
    )

    register_ops.register_builtins(Registry())

    assert {impl.op_name for impl in registered} == expected
    assert all(impl.kind == BackendImplKind.DEFAULT for impl in registered)
    assert all(impl.vendor is None for impl in registered)
    assert all(impl.fn._is_available.__name__ == "is_available" for impl in registered)


def test_minimax_backend_forwards_to_library(monkeypatch):
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend
    from vllm_fl.dispatch.backends.flaggems.impl.activation import (
        swigluoai_uninterleave_flaggems,
    )

    calls = []
    package = ModuleType("flaggems_vllm")
    package.fused_minimax_m3_qknorm_rope_kv_insert = (
        lambda *a, **kw: calls.append(("preprocess", a, kw))
    )
    package.swigluoai_uninterleave = (
        lambda *a, **kw: calls.append(("activation", a, kw))
    )
    package.reshape_and_cache_flash = (
        lambda *a, **kw: calls.append(("cache", a, kw))
    )
    monkeypatch.setitem(sys.modules, "flaggems_vllm", package)

    FlagGemsBackend().fused_minimax_m3_qknorm_rope_kv_insert("qkv", eps=1e-6)
    FlagGemsBackend().reshape_and_cache_flash("key", "value", layout="flash")
    swigluoai_uninterleave_flaggems(
        "output", "input", clamp_limit=7.0, alpha=1.702, beta=1.0
    )
    assert calls == [
        ("preprocess", ("qkv",), {"eps": 1e-6}),
        ("cache", ("key", "value"), {"layout": "flash"}),
        (
            "activation",
            ("input", 7.0, 1.702, 1.0),
            {"out": "output"},
        ),
    ]


def test_cache_write_registered_for_common_and_metax_paths(monkeypatch):
    from vllm_fl.dispatch.backends.flaggems import register_ops as gems_ops
    from vllm_fl.dispatch.backends.vendor.metax import register_ops as metax_ops
    from vllm_fl.dispatch.types import BackendImplKind

    registered = []
    registry = Mock()
    registry.register_many.side_effect = registered.extend
    monkeypatch.setattr(
        gems_ops, "use_flaggems_op", lambda name: name == "reshape_and_cache_flash"
    )
    gems_ops.register_builtins(registry)
    metax_ops.register_builtins(registry)

    cache_impls = [
        impl for impl in registered if impl.op_name == "reshape_and_cache_flash"
    ]
    assert len(cache_impls) == 2
    assert {impl.kind for impl in cache_impls} == {
        BackendImplKind.DEFAULT,
        BackendImplKind.VENDOR,
    }
    assert {impl.vendor for impl in cache_impls} == {None, "metax"}


def test_metax_cache_probe_and_forwarding_stay_in_backend(monkeypatch):
    import vllm

    from vllm_fl.dispatch.backends.vendor.metax.metax import MacaBackend

    calls = []
    native_ops = ModuleType("vllm._custom_ops")
    native_ops.reshape_and_cache_flash = lambda *args: calls.append(args)
    monkeypatch.setattr(vllm, "_custom_ops", native_ops, raising=False)
    monkeypatch.setitem(sys.modules, "vllm._custom_ops", native_ops)
    monkeypatch.setattr(MacaBackend, "is_available", lambda self: True)

    probe = Mock(return_value=True)
    monkeypatch.setattr(torch._C, "_dispatch_has_kernel_for_dispatch_key", probe)
    backend = MacaBackend()
    assert backend.is_reshape_and_cache_flash_available()
    args = tuple(object() for _ in range(8))
    backend.reshape_and_cache_flash(*args)
    assert calls == [args]
    probe.assert_called_once_with("_C_cache_ops::reshape_and_cache_flash", "CUDA")

    probe.return_value = False
    assert not backend.is_reshape_and_cache_flash_available()
    probe.side_effect = RuntimeError("kernel not registered")
    assert not backend.is_reshape_and_cache_flash_available()


@pytest.mark.gpu
@pytest.mark.flaggems
def test_local_triton_quant_matches_reference_on_cuda(device):
    if device.type != "cuda":
        pytest.skip("local Triton quantization contract is validated on CUDA")

    from vllm_fl.quantization.w8a8.reference import (
        dynamic_per_token_quant_int8 as reference_quant,
    )

    x = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, -1.0, 0.5 / 127.0, 1.5 / 127.0],
            [7.75, -8.0, 0.03125, -0.5],
        ],
        device=device,
        dtype=torch.float32,
    )

    actual_q, actual_scale = dynamic_per_token_quant_int8_flaggems_triton(x)
    expected_q, expected_scale = reference_quant(x)

    assert torch.equal(actual_q, expected_q)
    torch.testing.assert_close(actual_scale, expected_scale, rtol=0, atol=1e-7)

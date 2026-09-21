# SPDX-License-Identifier: Apache-2.0
"""Cache implementation selection must be independent of vendor attention."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_fl.dispatch import (
    SelectionPolicy,
    get_default_manager,
    reset_default_manager,
    reset_global_policy,
    set_global_policy,
)
from vllm_fl.ops.kv_cache import reshape_and_cache_flash

OP = "reshape_and_cache_flash"


@pytest.fixture(autouse=True)
def fresh_dispatch():
    reset_default_manager()
    reset_global_policy()
    manager = get_default_manager()
    manager._state.initialized = True
    manager._state.init_pid = os.getpid()
    yield manager
    reset_default_manager()
    reset_global_policy()


def register_cache(monkeypatch, manager, *, enable_flaggems=True):
    from vllm_fl.dispatch.backends.flaggems import register_ops as fg
    from vllm_fl.dispatch.backends.vendor.metax import register_ops as metax

    monkeypatch.setattr(
        fg, "use_flaggems_op", lambda name: enable_flaggems and name == OP
    )
    # Cache registration should not depend on model-specific operator loading.
    monkeypatch.setitem(
        sys.modules,
        "vllm_fl.ops.minimax_m3.ops",
        SimpleNamespace(register=lambda registry: None),
    )
    fg.register_builtins(manager.registry)
    metax.register_builtins(manager.registry)


@pytest.mark.parametrize(
    "prefer,native,enabled,expected",
    [
        ("flagos", True, True, "default.flaggems"),
        ("vendor", True, True, "vendor.metax"),
        ("vendor", False, True, "default.flaggems"),
        ("flagos", True, False, "vendor.metax"),
    ],
)
def test_cache_policy_and_usage_identity(
    monkeypatch, fresh_dispatch, prefer, native, enabled, expected
):
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend
    from vllm_fl.dispatch.backends.vendor.metax.metax import MacaBackend

    calls = []
    monkeypatch.setattr(
        FlagGemsBackend, "is_reshape_and_cache_flash_available", lambda self: True
    )
    monkeypatch.setattr(
        MacaBackend, "is_reshape_and_cache_flash_available", lambda self: native
    )
    for cls, name in [
        (FlagGemsBackend, "default.flaggems"),
        (MacaBackend, "vendor.metax"),
    ]:
        monkeypatch.setattr(
            cls, OP, lambda self, *args, name=name: calls.append((name, args))
        )
    register_cache(monkeypatch, fresh_dispatch, enable_flaggems=enabled)
    set_global_policy(SelectionPolicy(prefer=prefer))
    args = tuple(object() for _ in range(8))
    reshape_and_cache_flash(*args)
    assert calls == [(expected, args)]
    assert fresh_dispatch._called_ops[OP] == expected


def test_disabled_flaggems_and_missing_native_do_not_silently_use_gems(
    monkeypatch, fresh_dispatch
):
    from vllm_fl.dispatch.backends.vendor.metax.metax import MacaBackend

    monkeypatch.setattr(
        MacaBackend, "is_reshape_and_cache_flash_available", lambda self: False
    )
    register_cache(monkeypatch, fresh_dispatch, enable_flaggems=False)
    set_global_policy(SelectionPolicy(prefer="vendor"))
    with pytest.raises(RuntimeError, match="No available implementation"):
        reshape_and_cache_flash(*(object() for _ in range(8)))


def test_cached_cache_write_refreshes_on_policy_change(monkeypatch, fresh_dispatch):
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend
    from vllm_fl.dispatch.backends.vendor.metax.metax import MacaBackend

    calls = []
    for cls, name in [(FlagGemsBackend, "gems"), (MacaBackend, "native")]:
        monkeypatch.setattr(
            cls, "is_reshape_and_cache_flash_available", lambda self: True
        )
        monkeypatch.setattr(cls, OP, lambda self, *args, name=name: calls.append(name))
    register_cache(monkeypatch, fresh_dispatch)
    args = tuple(object() for _ in range(8))
    for prefer in ["flagos", "vendor", "flagos"]:
        set_global_policy(SelectionPolicy(prefer=prefer))
        reshape_and_cache_flash(*args)
    assert calls == ["gems", "native", "gems"]


@pytest.mark.parametrize("strict", [False, True])
def test_native_failure_respects_strict_mode(monkeypatch, fresh_dispatch, strict):
    from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend
    from vllm_fl.dispatch.backends.vendor.metax.metax import MacaBackend

    calls = []

    def fail(self, *args):
        calls.append("native")
        raise RuntimeError("native cache unavailable for this input")

    for cls in [FlagGemsBackend, MacaBackend]:
        monkeypatch.setattr(
            cls, "is_reshape_and_cache_flash_available", lambda self: True
        )
    monkeypatch.setattr(MacaBackend, OP, fail)
    monkeypatch.setattr(FlagGemsBackend, OP, lambda self, *args: calls.append("gems"))
    register_cache(monkeypatch, fresh_dispatch)
    set_global_policy(SelectionPolicy(prefer="vendor", strict=strict))
    args = tuple(object() for _ in range(8))
    if strict:
        with pytest.raises(RuntimeError, match="native cache unavailable"):
            reshape_and_cache_flash(*args)
        assert calls == ["native"]
    else:
        reshape_and_cache_flash(*args)
        assert calls == ["native", "gems"]
        assert fresh_dispatch._called_ops[OP] == "default.flaggems"


@pytest.mark.parametrize("registered", [True, False, None])
def test_native_availability_requires_a_kernel(monkeypatch, registered):
    from vllm import _custom_ops  # noqa: F401 -- initialize before mocking the probe

    from vllm_fl.dispatch.backends.vendor.metax.impl.kv_cache import (
        native_cache_available,
    )

    probe = Mock(return_value=registered)
    if registered is None:
        probe.side_effect = RuntimeError("operator does not exist")
    monkeypatch.setattr(torch._C, "_dispatch_has_kernel_for_dispatch_key", probe)
    assert native_cache_available() is (registered is True)
    probe.assert_called_once_with("_C_cache_ops::reshape_and_cache_flash", "CUDA")


def test_vendor_attention_uses_the_shared_cache_dispatch():
    from vllm.platforms import current_platform

    if not current_platform.is_out_of_tree():
        pytest.skip("the MetaX attention helper is an out-of-tree backend")
    from vllm_fl.dispatch.backends.vendor.metax.impl.attention.utils import fa_utils

    assert fa_utils.reshape_and_cache_flash is reshape_and_cache_flash


@pytest.mark.gpu
@pytest.mark.parametrize("cache_dtype", ["auto", "bfloat16"])
def test_flaggems_cache_writes_and_graph_replay(
    monkeypatch, fresh_dispatch, cache_dtype
):
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA-like accelerator")
    register_cache(monkeypatch, fresh_dispatch)
    set_global_policy(SelectionPolicy(strict=True, per_op_order=((OP, ("flagos",)),)))
    # Padded/strided K/V and interleaved K/V cache views match the FA call site.
    x = torch.randn(9, 3, 128, device="cuda", dtype=torch.bfloat16)
    key, value = x[:, 0:1], x[:, 1:2]
    cache = torch.full((3, 2, 128, 1, 128), -77, device="cuda", dtype=torch.bfloat16)
    kc, vc = cache[:, 0], cache[:, 1]
    slots = torch.tensor([129, 0, -1, 255, 128, 7, 256], device="cuda")
    scale = torch.ones((), device="cuda")

    def call():
        reshape_and_cache_flash(key, value, kc, vc, slots, cache_dtype, scale, scale)

    def check():
        expected = torch.full(cache.shape, -77, dtype=torch.bfloat16)
        k, v = key.cpu(), value.cpu()
        for token, slot in enumerate(slots.cpu().tolist()):
            if slot >= 0:
                block, offset = divmod(slot, 128)
                expected[block, 0, offset] = k[token]
                expected[block, 1, offset] = v[token]
        assert torch.equal(cache.cpu(), expected)

    call()
    check()
    assert fresh_dispatch._called_ops[OP] == "default.flaggems"
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    x.add_(0.5)
    slots.copy_(torch.tensor([3, -1, 130, 131, 257, 8, 9], device="cuda"))
    cache.fill_(-77)
    graph.replay()
    torch.cuda.synchronize()
    check()

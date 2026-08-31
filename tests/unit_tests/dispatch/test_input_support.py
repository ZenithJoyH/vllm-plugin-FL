# SPDX-License-Identifier: Apache-2.0
"""Pre-launch input support is distinct from retrying a failed kernel."""

import os

import pytest

import vllm_fl.dispatch as dispatch
from vllm_fl.dispatch import (
    BackendImplKind,
    OpImpl,
    OpManager,
    SelectionPolicy,
    policy_context,
)


@pytest.fixture
def manager(monkeypatch):
    mgr = OpManager()
    mgr._state.initialized = True
    mgr._state.init_pid = os.getpid()
    monkeypatch.setattr(dispatch, "get_default_manager", lambda: mgr)
    return mgr


@pytest.mark.parametrize("mode", ["call", "cached", "resolve"])
@pytest.mark.parametrize("strict", [False, True])
def test_shape_support_does_not_poison_cache(manager, mode, strict):
    manager.registry.register_impl(
        OpImpl(
            "op",
            "default.flagos",
            BackendImplKind.DEFAULT,
            lambda page: "gems",
            supports=lambda page: page == 64,
        )
    )
    manager.registry.register_impl(
        OpImpl(
            "op",
            "vendor.thead",
            BackendImplKind.VENDOR,
            lambda page: "vendor",
            vendor="thead",
        )
    )
    with policy_context(SelectionPolicy.from_dict(prefer="flagos", strict=strict)):
        call = (
            (lambda x: manager.call("op", x))
            if mode == "call"
            else (
                dispatch.CachedOp("op") if mode == "cached" else manager.resolve("op")
            )
        )
        assert [call(p) for p in (64, 32, 64)] == ["gems", "vendor", "gems"]
        assert manager.get_failed_impls() == {}
        assert manager._called_ops["op"] == "default.flagos"


@pytest.mark.parametrize("mode", ["call", "cached"])
def test_stateful_error_never_retries(manager, mode):
    calls = []

    def broken(x):
        calls.append("mutated")
        raise RuntimeError("kernel failed after cache write")

    manager.registry.register_impl(
        OpImpl(
            "state",
            "default.flagos",
            BackendImplKind.DEFAULT,
            broken,
            allow_runtime_fallback=False,
        )
    )
    manager.registry.register_impl(
        OpImpl(
            "state",
            "reference.torch",
            BackendImplKind.REFERENCE,
            lambda x: calls.append("retry"),
        )
    )
    with policy_context(SelectionPolicy.from_dict(strict=False)):
        call = (
            dispatch.CachedOp("state")
            if mode == "cached"
            else lambda x: manager.call("state", x)
        )
        with pytest.raises(RuntimeError, match="after cache write"):
            call(1)
    assert calls == ["mutated"]


def test_input_support_never_overrides_user_backend_allowlist(manager):
    manager.registry.register_impl(
        OpImpl(
            "op",
            "default.flagos",
            BackendImplKind.DEFAULT,
            lambda x: "gems",
            supports=lambda x: False,
        )
    )
    manager.registry.register_impl(
        OpImpl(
            "op",
            "vendor.thead",
            BackendImplKind.VENDOR,
            lambda x: "vendor",
            vendor="thead",
        )
    )
    with (
        policy_context(SelectionPolicy.from_dict(per_op_order={"op": ["flagos"]})),
        pytest.raises(RuntimeError, match="No implementation supports"),
    ):
        manager.call("op", 32)


def test_support_check_exception_is_not_silenced(manager):
    def guard(x):
        raise ValueError("invalid metadata")

    manager.registry.register_impl(
        OpImpl(
            "op",
            "default.flagos",
            BackendImplKind.DEFAULT,
            lambda x: None,
            supports=guard,
        )
    )
    with pytest.raises(ValueError, match="invalid metadata"):
        manager.call("op", 1)

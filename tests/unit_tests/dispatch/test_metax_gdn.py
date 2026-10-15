# Copyright (c) 2026 BAAI. All rights reserved.

"""MetaX GDN vendor/FlagOS backend contracts."""

import pytest
import torch
import torch.nn.functional as F

from vllm_fl.dispatch.backends.flaggems.flaggems import FlagGemsBackend
from vllm_fl.dispatch.backends.flaggems import register_ops as flagos_register
from vllm_fl.dispatch.backends.vendor.metax.metax import MacaBackend
from vllm_fl.dispatch.registry import OpRegistry


def test_gdn_registration_obeys_flagos_blacklist(monkeypatch):
    monkeypatch.setattr(flagos_register, "use_flaggems_op", lambda _: False)
    monkeypatch.setattr(FlagGemsBackend, "is_available", lambda self: False)
    registry = OpRegistry()
    flagos_register.register_builtins(registry)

    for name in ("gdn_prefill", "gdn_decode", "gdn_packed_decode"):
        assert registry.get_implementation(name, "default.flagos") is None
    for name in (
        "chunk_gated_delta_rule_fwd",
        "fused_recurrent_gated_delta_rule_fwd",
    ):
        assert registry.get_implementations(name) == []
    assert registry.get_implementation("attention_backend", "default.flagos") is None


@pytest.mark.parametrize("candidate_exists", [False, True])
def test_gdn_hook_preserves_original_when_no_candidate(monkeypatch, candidate_exists):
    from vllm.model_executor.layers.mamba import gdn_linear_attn as gdn_mod
    from vllm_fl.patches import gdn_linear_attn as hook

    seen = []
    native_result = object()
    dispatch_result = object()

    def native_prefill(*args, **kwargs):
        seen.append(("native_prefill", args, kwargs))
        return native_result

    def native_decode(**kwargs):
        seen.append(("native_decode", kwargs))
        return native_result

    def native_packed(**kwargs):
        seen.append(("native_packed", kwargs))
        return native_result

    class Registry:
        def get_implementations(self, name):
            return [type("Impl", (), {"is_available": lambda self: True})()] if candidate_exists else []

    class Manager:
        registry = Registry()

        def ensure_initialized(self):
            pass

        def get_selected_impl_id(self, name):
            return "vendor.metax"

    monkeypatch.setattr(gdn_mod, "fla_chunk_gated_delta_rule", native_prefill)
    monkeypatch.setattr(
        gdn_mod, "fused_sigmoid_gating_delta_rule_update", native_decode,
    )
    monkeypatch.setattr(
        gdn_mod,
        "fused_recurrent_gated_delta_rule_packed_decode", native_packed,
    )
    monkeypatch.setattr(gdn_mod, "_flagos_gdn_installed", False, raising=False)
    monkeypatch.setattr(hook, "get_default_manager", lambda: Manager())
    monkeypatch.setattr(
        hook, "call_op",
        lambda name, **kwargs: seen.append(("dispatch", name, kwargs))
        or dispatch_result,
    )
    hook.install_gdn_dispatch()

    prefill = gdn_mod.fla_chunk_gated_delta_rule(1, 2, 3, 4, 5)
    decode = gdn_mod.fused_sigmoid_gating_delta_rule_update(
        1, 2, 3, 4, 5, 6, 7,
    )
    packed = gdn_mod.fused_recurrent_gated_delta_rule_packed_decode(
        1, 2, 3, 4, 5, 6, 7, 8, 9,
    )
    assert packed is (dispatch_result if candidate_exists else native_result)
    assert (prefill, decode) == (
        (dispatch_result, dispatch_result) if candidate_exists
        else (native_result, native_result)
    )
    assert [entry[0] for entry in seen] == (
        ["dispatch", "dispatch", "dispatch"] if candidate_exists
        else ["native_prefill", "native_decode", "native_packed"]
    )


def test_worker_installs_gdn_patch_before_model_construction(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from vllm_fl.patches import gdn_linear_attn as hook
    from vllm_fl.worker import worker as worker_mod

    calls = []
    monkeypatch.setattr(worker_mod, "set_current_vllm_config", lambda _: nullcontext())
    monkeypatch.setattr(hook, "install_gdn_dispatch", lambda: calls.append("patch"))
    fake = SimpleNamespace(
        vllm_config=object(),
        model_runner=SimpleNamespace(
            load_model=lambda **kwargs: calls.append(("model", kwargs)),
        ),
    )
    worker_mod.WorkerFL.load_model(fake, load_dummy_weights=True)
    assert calls == ["patch", ("model", {"load_dummy_weights": True})]


def test_vendor_gdn_calls_original_fla_exports(monkeypatch):
    from vllm.model_executor.layers.fla import ops as fla_ops

    calls = []
    sentinel = object()

    def prefill(*args, **kwargs):
        calls.append(("prefill", args, kwargs))
        return sentinel

    def decode(*args, **kwargs):
        calls.append(("decode", args, kwargs))
        return sentinel

    monkeypatch.setattr(fla_ops, "chunk_gated_delta_rule", prefill)
    monkeypatch.setattr(fla_ops, "fused_sigmoid_gating_delta_rule_update", decode)
    backend = MacaBackend()
    assert backend.gdn_prefill(1, g=2) is sentinel
    assert backend.gdn_decode(A_log=3, q=4) is sentinel
    assert calls == [
        ("prefill", (1,), {"g": 2}),
        ("decode", (), {"A_log": 3, "q": 4}),
    ]


def test_flagos_gdn_prefill_adapts_seven_value_result(monkeypatch):
    backend = FlagGemsBackend()
    q = torch.ones((1, 2, 1, 4), dtype=torch.bfloat16)
    k = q.clone()
    v = q.clone()
    g = torch.zeros((1, 2, 1))
    beta = torch.ones_like(g)
    state = torch.zeros((1, 1, 4, 4))
    output = torch.ones_like(q, dtype=torch.float32)
    final_state = state.clone()
    seen = {}

    def low_level(**kwargs):
        seen.update(kwargs)
        return None, output, None, final_state, None, None, None

    monkeypatch.setattr(backend, "chunk_gated_delta_rule_fwd", low_level)
    result, updated = backend.gdn_prefill(
        q, k, v, g, beta, initial_state=state, output_final_state=True,
    )
    assert result.dtype == q.dtype
    assert torch.equal(result, output.to(q.dtype))
    assert updated is final_state
    assert seen["initial_state"] is state
    assert seen["scale"] == 0.5
    assert seen["output_final_state"] is True


def test_flagos_gdn_decode_preserves_gate_and_state_contract(monkeypatch):
    backend = FlagGemsBackend()
    A_log = torch.tensor([0.0])
    a = torch.tensor([0.5])
    b = torch.tensor([0.0])
    dt_bias = torch.tensor([0.1])
    q = torch.ones((1, 1, 1, 4), dtype=torch.bfloat16)
    state = torch.zeros((1, 1, 4, 4))
    indices = torch.tensor([1], dtype=torch.int32)
    cu_seqlens = torch.tensor([0, 1], dtype=torch.int32)
    output = q.clone()
    seen = {}

    def low_level(**kwargs):
        seen.update(kwargs)
        return output, state

    monkeypatch.setattr(backend, "fused_recurrent_gated_delta_rule_fwd", low_level)
    result, updated = backend.gdn_decode(
        A_log, a, b, dt_bias, q, q, q,
        initial_state=state, ssm_state_indices=indices,
        cu_seqlens=cu_seqlens,
    )
    assert result is output
    assert updated is state
    torch.testing.assert_close(
        seen["g"], -A_log.exp() * F.softplus(a + dt_bias),
    )
    torch.testing.assert_close(seen["beta"], b.sigmoid())
    assert seen["initial_state"] is state
    assert seen["ssm_state_indices"] is indices
    assert seen["cu_seqlens"] is cu_seqlens
    assert seen["scale"] == 0.5


@pytest.mark.parametrize("unsupported", ["is_kda"])
def test_flagos_gdn_decode_rejects_unsupported_modes(unsupported):
    kwargs = {unsupported: True if unsupported == "is_kda" else torch.tensor([1])}
    with pytest.raises(NotImplementedError):
        FlagGemsBackend().gdn_decode(
            None, None, None, None, None, None, None, **kwargs,
        )

"""Model-scoped dispatch registration, selection and upstream isolation."""

import importlib

from vllm_fl.dispatch import BackendImplKind, OpImpl, OpRegistry
from vllm_fl.dispatch.backends.triton.register_ops import register_builtins
from vllm_fl.dispatch.manager import OpManager
from vllm_fl.dispatch.policy import SelectionPolicy, policy_context


def test_portable_registration_is_lazy_and_vendor_neutral(monkeypatch):
    registration = importlib.import_module(
        "vllm_fl.dispatch.backends.triton.register_ops"
    )
    monkeypatch.setattr(
        registration,
        "import_module",
        lambda name: (_ for _ in ()).throw(
            AssertionError("Registration must not import a kernel: " + name)
        ),
    )
    registry = OpRegistry()
    register_builtins(registry)
    assert len(registry.list_operators()) == 10
    for name in registry.list_operators():
        (impl,) = registry.get_implementations(name)
        assert name.startswith("qwen38_")
        assert impl.kind == BackendImplKind.DEFAULT
        assert impl.vendor is None


def test_qwen_gdn_subclass_does_not_modify_upstream():
    upstream = importlib.import_module(
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
    )
    recurrent = importlib.import_module(
        "vllm.model_executor.layers.fla.ops.fused_recurrent"
    )
    method = upstream.QwenGatedDeltaNetAttention._forward_core_decode_non_spec
    kernel = recurrent.fused_recurrent_gated_delta_rule_packed_decode_kernel
    from vllm_fl.models.qwen3_8_flash_next.gpu.gdn import Qwen38GatedDeltaNetAttention

    assert upstream.QwenGatedDeltaNetAttention._forward_core_decode_non_spec is method
    assert recurrent.fused_recurrent_gated_delta_rule_packed_decode_kernel is kernel
    assert Qwen38GatedDeltaNetAttention._forward_core_decode_non_spec is not method
    assert (
        Qwen38GatedDeltaNetAttention.__init__
        is upstream.QwenGatedDeltaNetAttention.__init__
    )


def test_vendor_override_uses_existing_policy_and_denylists(monkeypatch):
    manager = OpManager()
    manager.ensure_initialized()
    name = "qwen38_ple_state_gather"
    vendor_fn = lambda *args: "test vendor"
    manager.registry.register_impl(
        OpImpl(
            op_name=name,
            impl_id="vendor.test",
            kind=BackendImplKind.VENDOR,
            vendor="test",
            fn=vendor_fn,
        )
    )
    with policy_context(SelectionPolicy(prefer="vendor")):
        assert manager.resolve(name) is vendor_fn
    with policy_context(
        SelectionPolicy(prefer="vendor", deny_vendors=frozenset({"test"}))
    ):
        assert manager.resolve(name) is not vendor_fn
    with policy_context(
        SelectionPolicy.from_dict(per_op_order={name: ["impl:vendor.test"]})
    ):
        assert manager.resolve(name) is vendor_fn
    # A portable implementation must not mask a usable vendor implementation
    # when the platform has no Triton backend, even with default preference.
    import vllm.triton_utils

    monkeypatch.setattr(vllm.triton_utils, "HAS_TRITON", False)
    manager.bump_policy_epoch()
    with policy_context(SelectionPolicy(prefer="flagos")):
        assert manager.resolve(name) is vendor_fn


def test_model_registration_preserves_other_models_and_upstream_kernel():
    from vllm.model_executor.models import registry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    from vllm_fl.patches.qwen3_8_flash_next import apply_qwen3_8_flash_next_patches

    original = MODELS_CONFIG_MAP.get("Qwen3_5ForConditionalGeneration")
    apply_qwen3_8_flash_next_patches()
    apply_qwen3_8_flash_next_patches()
    assert MODELS_CONFIG_MAP.get("Qwen3_5ForConditionalGeneration") is original
    for arch in ("Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"):
        assert arch in registry.ModelRegistry.get_supported_archs()
    assert "Qwen4ExpMTP" not in MODELS_CONFIG_MAP

from types import SimpleNamespace

from vllm_fl.patches import qwen3_5_v024 as compat


def test_text_config_convertor_remaps_conditional_architecture():
    hf_config = SimpleNamespace(
        model_type="qwen3_5_moe_text",
        architectures=["Qwen3_5MoeForConditionalGeneration"],
    )
    convertor = compat.Qwen3_5TextModelArchConfigConvertor(hf_config, hf_config)

    assert convertor.get_architectures() == ["Qwen3_5MoeForCausalLM"]
    assert hf_config.architectures == ["Qwen3_5MoeForCausalLM"]


def test_text_config_convertor_supplies_default_architecture():
    hf_config = SimpleNamespace(
        model_type="qwen3_5_text",
        architectures=None,
    )
    convertor = compat.Qwen3_5TextModelArchConfigConvertor(hf_config, hf_config)

    assert convertor.get_architectures() == ["Qwen3_5ForCausalLM"]
    assert hf_config.architectures == ["Qwen3_5ForCausalLM"]


def test_causal_config_removes_multimodal_rope_keys(monkeypatch):
    parent_calls = []
    monkeypatch.setattr(
        compat.Qwen3_5ForConditionalGenerationConfig,
        "verify_and_update_config",
        lambda config: parent_calls.append(config),
    )
    rope_parameters = {
        "rope_type": "default",
        "mrope_section": [16, 24, 24],
        "mrope_interleaved": True,
    }
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(rope_parameters=rope_parameters)
        )
    )

    compat.Qwen3_5ForCausalLMConfig.verify_and_update_config(vllm_config)

    assert parent_calls == [vllm_config]
    assert rope_parameters == {"rope_type": "default"}


def test_apply_registers_only_plugin_owned_lazy_models(monkeypatch):
    from vllm.model_executor.models import config as model_config
    from vllm.model_executor.models import registry as model_registry
    from vllm.transformers_utils import config as transformers_config
    from vllm.transformers_utils import model_arch_config_convertor

    registered = {}
    fake_registry = SimpleNamespace(
        register_model=lambda architecture, model: registered.__setitem__(
            architecture, model
        )
    )
    monkeypatch.setattr(transformers_config, "_CONFIG_REGISTRY", {})
    monkeypatch.setattr(model_config, "MODELS_CONFIG_MAP", {})
    monkeypatch.setattr(
        model_arch_config_convertor, "MODEL_ARCH_CONFIG_CONVERTORS", {}
    )
    monkeypatch.setattr(model_registry, "_TEXT_GENERATION_MODELS", {})
    monkeypatch.setattr(model_registry, "_VLLM_MODELS", {})
    monkeypatch.setattr(model_registry, "ModelRegistry", fake_registry)

    assert compat.apply_qwen3_5_v024_patches() is True

    assert registered == {
        "Qwen3_5ForCausalLM": (
            "vllm_fl.models.qwen3_5:Qwen3_5ForCausalLM"
        ),
        "Qwen3_5MoeForCausalLM": (
            "vllm_fl.models.qwen3_5:Qwen3_5MoeForCausalLM"
        ),
    }
    assert transformers_config._CONFIG_REGISTRY == {
        "qwen3_5_text": "Qwen3_5TextConfig",
        "qwen3_5_moe_text": "Qwen3_5MoeTextConfig",
    }
    assert model_config.MODELS_CONFIG_MAP == {
        "Qwen3_5ForCausalLM": compat.Qwen3_5ForCausalLMConfig,
        "Qwen3_5MoeForCausalLM": compat.Qwen3_5ForCausalLMConfig,
    }


def test_lazy_model_shim_marks_upstream_classes_hybrid():
    from vllm_fl.models.qwen3_5 import (
        Qwen3_5ForCausalLM,
        Qwen3_5MoeForCausalLM,
    )

    for model_cls in (Qwen3_5ForCausalLM, Qwen3_5MoeForCausalLM):
        assert model_cls.is_hybrid is True
        assert hasattr(model_cls, "get_mamba_state_dtype_from_config")
        assert hasattr(model_cls, "get_mamba_state_shape_from_config")
        assert hasattr(model_cls, "get_mamba_state_copy_func")


def test_lazy_model_shim_remaps_vl_checkpoint_weights(monkeypatch):
    from vllm_fl.models import qwen3_5

    calls = {}

    class FakeLoader:
        def __init__(self, model, **kwargs):
            calls["init"] = (model, kwargs)

        def load_weights(self, weights, **kwargs):
            calls["load"] = (weights, kwargs)
            return {"loaded"}

    class FakeMapper:
        def __init__(self, **kwargs):
            calls["mapper"] = kwargs

    monkeypatch.setattr(qwen3_5, "AutoWeightsLoader", FakeLoader)
    monkeypatch.setattr(qwen3_5, "WeightsMapper", FakeMapper)
    model = object()
    weights = [("model.language_model.proj.weight", object())]

    assert qwen3_5._load_weights(model, weights) == {"loaded"}
    assert calls["init"] == (
        model,
        {
            "skip_prefixes": ["mtp."],
            "ignore_unexpected_prefixes": ["model.visual."],
        },
    )
    assert calls["mapper"] == {
        "orig_to_new_prefix": {"model.language_model.": "model."}
    }
    assert calls["load"][0] is weights
    assert isinstance(calls["load"][1]["mapper"], FakeMapper)

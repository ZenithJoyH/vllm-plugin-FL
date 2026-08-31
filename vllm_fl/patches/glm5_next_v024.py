# SPDX-License-Identifier: Apache-2.0
"""Register the plugin-owned GLM5-Next implementation on vLLM 0.24."""

import logging


from vllm.model_executor.models.config import (
    HybridAttentionMambaModelConfig,
)
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
)

from vllm_fl.patches._version import is_vllm_024


logger = logging.getLogger(__name__)

_CAUSAL_ARCH = "Glm5NextForCausalLM"
_CONDITIONAL_ARCH = "Glm5NextForConditionalGeneration"


class Glm5NextModelArchConfigConvertor(ModelArchConfigConvertorBase):
    """Preserve VLM checkpoints and default bare text configs to CausalLM."""

    def get_architectures(self) -> list[str]:
        architectures = super().get_architectures()
        if not architectures:
            architectures = [_CAUSAL_ARCH]
        self.hf_config.architectures = architectures.copy()
        return architectures


class Glm5NextForCausalLMConfig(HybridAttentionMambaModelConfig):
    """Compose v0.24 hybrid-cache and DSA validation."""

    @classmethod
    def verify_and_update_config(cls, vllm_config) -> None:
        from vllm_fl.patches.glm5_next_runner_v024 import install_glm5_runner_adapter

        install_glm5_runner_adapter()
        HybridAttentionMambaModelConfig.verify_and_update_config(vllm_config)

        from vllm.platforms import current_platform

        if getattr(current_platform, "vendor_name", None) == "thead":
            from vllm.v1.attention.backends.mla.prefill.registry import (
                MLAPrefillBackendEnum,
                register_mla_prefill_backend,
            )

            register_mla_prefill_backend(
                MLAPrefillBackendEnum.FLASH_ATTN,
                "vllm_fl.dispatch.backends.flaggems.impl.mla_prefill.FlagGemsMLAPrefillBackend",
            )

        text_config = vllm_config.model_config.hf_text_config
        cache_config = vllm_config.cache_config
        if cache_config.cache_dtype == "bfloat16":
            cache_config.cache_dtype = "auto"
        if getattr(text_config, "index_kpool_compress", False):
            kpool = int(getattr(text_config, "index_kpool", 1))
            required = kpool * 32
            if cache_config.block_size % required:
                logger.info(
                    "GLM5-Next kpool changes KV block_size from %d to %d "
                    "for a 32-entry DeepGEMM compressed page",
                    cache_config.block_size,
                    required,
                )
                cache_config.block_size = required


def apply_glm5_next_v024_patches() -> bool:
    """Install idempotent config, convertor, and lazy-model registrations."""
    if not is_vllm_024():
        return False

    from vllm.model_executor.models import config as model_config
    from vllm.model_executor.models import registry as model_registry
    from vllm.transformers_utils import config as transformers_config
    from vllm.transformers_utils import model_arch_config_convertor

    from vllm_fl.configs.glm5_next import (
        Glm5NextConfig,
        Glm5NextTextConfig,
        Glm5NextVisionConfig,
    )
    from vllm_fl.patches.glm5_next_kpool_v024 import (
        install_glm5_next_kpool_v024,
    )

    install_glm5_next_kpool_v024()

    config_registry = transformers_config._CONFIG_REGISTRY
    config_registry["glm5_next"] = Glm5NextConfig
    config_registry["glm5_next_text"] = Glm5NextTextConfig
    config_registry["glm5_next_vision"] = Glm5NextVisionConfig

    convertors = model_arch_config_convertor.MODEL_ARCH_CONFIG_CONVERTORS
    convertors["glm5_next"] = Glm5NextModelArchConfigConvertor
    convertors["glm5_next_text"] = Glm5NextModelArchConfigConvertor

    for architecture in (_CAUSAL_ARCH, _CONDITIONAL_ARCH):
        model_config.MODELS_CONFIG_MAP[architecture] = Glm5NextForCausalLMConfig

    model_registry._TEXT_GENERATION_MODELS.setdefault(
        _CAUSAL_ARCH, ("glm5_next", _CAUSAL_ARCH)
    )
    model_registry._VLLM_MODELS.setdefault(_CAUSAL_ARCH, ("glm5_next", _CAUSAL_ARCH))
    model_registry.ModelRegistry.register_model(
        _CAUSAL_ARCH,
        f"vllm_fl.models.glm5_next:{_CAUSAL_ARCH}",
    )

    # Keep the checkpoint's conditional architecture so vLLM constructs the
    # vision tower and enables --mm-encoder-tp-mode data instead of silently
    # reducing the model to its text-only runtime.
    model_registry._VLLM_MODELS.setdefault(
        _CONDITIONAL_ARCH, ("glm5_next", _CONDITIONAL_ARCH)
    )
    model_registry.ModelRegistry.register_model(
        _CONDITIONAL_ARCH,
        f"vllm_fl.models.glm5_next_multimodal:{_CONDITIONAL_ARCH}",
    )

    logger.info(
        "Installed vLLM 0.24 GLM5-Next text/VLM runtime with bounded KDA "
        "gate, kpool, ViT data parallelism, and platform-dispatched operators",
    )
    return True


__all__ = [
    "Glm5NextForCausalLMConfig",
    "Glm5NextModelArchConfigConvertor",
    "apply_glm5_next_v024_patches",
]

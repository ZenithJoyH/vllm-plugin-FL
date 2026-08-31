# SPDX-License-Identifier: Apache-2.0
"""GLM5-Next multimodal wrapper for pristine vLLM 0.24.

The language runtime remains plugin-owned while the vision tower and the
batch-level ViT data-parallel transport reuse vLLM's GLM-OCR/GLM-4V support.
"""

from typing import ClassVar, Literal, Mapping

from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models.glm4_1v import (
    Glm4vDummyInputsBuilder,
    Glm4vForConditionalGeneration,
    Glm4vMultiModalProcessor,
    Glm4vProcessingInfo,
)
from vllm.model_executor.models.glm_ocr import (
    GlmOcrPatchMerger,
    GlmOcrVisionTransformer,
)
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid
from vllm.model_executor.models.utils import (
    init_vllm_registered_model,
    maybe_prefix,
)
from vllm.multimodal import MULTIMODAL_REGISTRY


class Glm5NextVisionPatchMerger(GlmOcrPatchMerger):
    pass


class Glm5NextVisionTransformer(GlmOcrVisionTransformer):
    """GLM-OCR tower with GLM5-Next's wider projection bottleneck."""

    def __init__(
        self,
        text_config,
        vision_config,
        norm_eps: float = 1e-5,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(
            text_config,
            vision_config,
            norm_eps=norm_eps,
            quant_config=quant_config,
            prefix=prefix,
        )
        self.merger = Glm5NextVisionPatchMerger(
            d_model=vision_config.out_hidden_size,
            context_dim=vision_config.projection_intermediate_size,
            quant_config=quant_config,
            bias=False,
            prefix=f"{prefix}.merger",
        )


class Glm5NextProcessingInfo(Glm4vProcessingInfo):
    """Build the checkpoint's custom image/video processor locally."""

    def get_hf_processor(self, **kwargs: object):
        del kwargs
        processor = getattr(self, "_glm5_hf_processor", None)
        if processor is None:
            from vllm_fl.transformers_utils.processors.glm5_next import (
                Glm5NextProcessor,
            )

            processor = Glm5NextProcessor.from_pretrained(self.ctx.model_config.model)
            self._glm5_hf_processor = processor
        return processor

    def get_mm_max_tokens_per_item(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
    ) -> Mapping[str, int] | None:
        """Per-item multimodal budget for the MultiModalBudget startup check.

        Mirrors ``Glm4vProcessingInfo.get_mm_max_tokens_per_item`` but reads the
        video pixel budget from ``_smart_resize_max``: the GLM-5-Next video
        processor normalizes ``size`` into a height/width form for the generic
        resize path and keeps the real pixel budget (``longest_edge``) in that
        attribute, so the stock ``size["longest_edge"]`` lookup would return
        None and crash with ``None // int`` during engine init.
        """
        result: dict[str, int] = {}

        if mm_counts.get("image", 0) > 0:
            result["image"] = self.get_max_image_tokens()

        if mm_counts.get("video", 0) > 0:
            video_processor = self.get_video_processor()
            max_pixels = video_processor._smart_resize_max

            vision_config = self.get_hf_config().vision_config
            temporal_patch_size = vision_config.temporal_patch_size
            patch_size = vision_config.patch_size
            merge_size = vision_config.spatial_merge_size

            max_vision_tokens = max_pixels // (
                temporal_patch_size * patch_size**2 * merge_size**2
            )

            # GLMGA supports up to 640 frames (max_frames).
            max_grid_t = 640 // temporal_patch_size

            tokenizer = self.get_tokenizer()
            max_ts_tokens = max(
                len(tokenizer.encode(f"{t:.1f} seconds", add_special_tokens=False))
                for t in range(min(max_grid_t, 300))
            )

            result["video"] = max_vision_tokens + max_grid_t * (2 + max_ts_tokens) + 2

        return result


@MULTIMODAL_REGISTRY.register_processor(
    Glm4vMultiModalProcessor,
    info=Glm5NextProcessingInfo,
    dummy_inputs=Glm4vDummyInputsBuilder,
)
class Glm5NextForConditionalGeneration(
    Glm4vForConditionalGeneration, HasInnerState, IsHybrid
):
    """GLM5-Next VLM: ViT-DP plus TP language layers and EP experts."""

    has_inner_state: ClassVar[Literal[True]] = True
    is_hybrid: ClassVar[Literal[True]] = True
    supports_encoder_tp_data = True

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        from vllm_fl.models.glm5_next import Glm5NextForCausalLM

        return Glm5NextForCausalLM.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        from vllm_fl.models.glm5_next import Glm5NextForCausalLM

        return Glm5NextForCausalLM.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        from vllm_fl.models.glm5_next import Glm5NextForCausalLM

        return Glm5NextForCausalLM.get_mamba_state_copy_func()

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # Bypass Glm4vForConditionalGeneration.__init__: its language-model
        # architecture selection does not know GLM5-Next.
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        multimodal_config = vllm_config.model_config.multimodal_config
        assert multimodal_config is not None

        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = multimodal_config
        self.use_data_parallel = multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = (
            multimodal_config.is_multimodal_pruning_enabled()
        )

        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Glm5NextVisionTransformer(
                config.text_config,
                config.vision_config,
                norm_eps=config.vision_config.rms_norm_eps,
                # The vision checkpoint is BF16 even when the language model
                # comes from the separately published FP8 directory.
                quant_config=None,
                prefix=maybe_prefix(prefix, "visual"),
            )

        with self._mark_language_model(vllm_config):
            self.language_model = init_vllm_registered_model(
                vllm_config=vllm_config,
                hf_config=config.text_config,
                prefix=maybe_prefix(prefix, "language_model"),
                architectures=["Glm5NextForCausalLM"],
            )


__all__ = [
    "Glm5NextForConditionalGeneration",
    "Glm5NextProcessingInfo",
    "Glm5NextVisionTransformer",
]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only XingChen4: DeepSeek MLA/MoE blocks with plugin-dispatched mHC.

The checkpoint stores four residual streams and merged mHC biases. Split biases
are legacy duplicates; MTP weights are not used by the non-speculative model.
No model-specific backend selection or process-global kernel patching lives here.
"""

from collections.abc import Iterable, Iterator

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV2DecoderLayer,
    DeepseekV2ForCausalLM,
)
from vllm.model_executor.models.utils import (
    make_empty_intermediate_tensors_factory,
    make_layers,
)
from vllm.sequence import IntermediateTensors

from vllm_fl.configs.xingchen4 import XingChen4Config
from vllm_fl.dispatch import CachedOp

_pre = CachedOp("mhc_pre")
_post = CachedOp("mhc_post")


@torch.library.custom_op("xingchen4_plugin::mhc_pre", mutates_args=())
def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    eps: float,
    pre_eps: float,
    post_mult: float,
    iterations: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _pre(residual, fn, scale, base, eps, pre_eps, eps, post_mult, iterations, 1)


@mhc_pre.register_fake
def _pre_fake(residual, fn, scale, base, eps, pre_eps, post_mult, iterations):
    outer, n, c = residual.shape[:-2], residual.shape[-2], residual.shape[-1]
    return (
        torch.empty((*outer, n, 1), dtype=torch.float32, device=residual.device),
        torch.empty((*outer, n, n), dtype=torch.float32, device=residual.device),
        torch.empty((*outer, c), dtype=residual.dtype, device=residual.device),
    )


@torch.library.custom_op("xingchen4_plugin::mhc_post", mutates_args=())
def mhc_post(
    x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor
) -> torch.Tensor:
    # Opaque boundary prevents compiler removal of required contiguous layout.
    return _post(x, residual, post, comb.transpose(-1, -2).contiguous())


@mhc_post.register_fake
def _post_fake(x, residual, post, comb):
    return torch.empty_like(residual)


class XingChen4MHC(nn.Module):
    def __init__(self, config: XingChen4Config) -> None:
        super().__init__()
        self.n, self.hidden_size = config.num_residual_streams, config.hidden_size
        self.eps, self.pre_eps = config.mhc_norm_eps, config.mhc_pre_eps
        self.post_mult = config.mhc_post_mult_value
        self.iterations = config.mhc_sinkhorn_iterations
        size = self.n * self.n + 2 * self.n
        self.hc_fn = nn.Parameter(
            torch.empty(size, self.n * self.hidden_size, dtype=torch.float32)
        )
        self.hc_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_base = nn.Parameter(torch.empty(size, dtype=torch.float32))

    def forward(self, hidden_states):
        residual = hidden_states.view(-1, self.n, self.hidden_size)
        post, comb, layer_input = mhc_pre(
            residual,
            self.hc_fn,
            self.hc_scale,
            self.hc_base,
            self.eps,
            self.pre_eps,
            self.post_mult,
            self.iterations,
        )
        return layer_input, comb, post

    def combine(
        self,
        residual: torch.Tensor,
        output: torch.Tensor,
        comb: torch.Tensor,
        post: torch.Tensor,
    ) -> torch.Tensor:
        """Mix the residual streams and broadcast the sublayer output."""
        residual = residual.view(-1, self.n, self.hidden_size)
        return mhc_post(output, residual, post, comb).flatten(-2)


class XingChen4DecoderLayer(DeepseekV2DecoderLayer):
    """Reuse MLA, MLP/MoE and norms; replace only the residual connections."""

    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        super().__init__(vllm_config, prefix)
        config = vllm_config.model_config.hf_config
        self.attn_hc = XingChen4MHC(config)
        self.ffn_hc = XingChen4MHC(config)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, None]:
        layer_input, comb, post = self.attn_hc(hidden_states)
        attention = self.self_attn(
            positions=positions,
            hidden_states=self.input_layernorm(layer_input),
            llama_4_scaling=llama_4_scaling,
        )
        hidden_states = self.attn_hc.combine(hidden_states, attention, comb, post)
        layer_input, comb, post = self.ffn_hc(hidden_states)
        output = self.mlp(self.post_attention_layernorm(layer_input))
        return self.ffn_hc.combine(hidden_states, output, comb, post), None


@support_torch_compile
class XingChen4Model(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise NotImplementedError("XingChen4 mHC currently requires PP=1")
        if vllm_config.model_config.dtype != torch.bfloat16:
            raise ValueError("XingChen4 mHC currently supports BF16 only")
        if config.num_residual_streams != 4:
            raise ValueError("XingChen4 currently requires four mHC streams")
        if getattr(config, "index_topk", None) is not None:
            raise NotImplementedError("DSA is not supported by XingChen4")
        if getattr(config, "llama_4_scaling", None) is not None:
            raise NotImplementedError("XingChen4 uses YaRN, not Llama-4 scaling")
        if vllm_config.speculative_config is not None:
            raise NotImplementedError("XingChen4 speculative decoding is not supported")
        self.vocab_size = config.vocab_size
        self.num_residual_streams = config.num_residual_streams
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: XingChen4DecoderLayer(vllm_config, prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"],
            config.hidden_size * config.num_residual_streams,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if intermediate_tensors is not None:
            raise ValueError("XingChen4 does not accept pipeline intermediate tensors")
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided")
            inputs_embeds = self.embed_input_ids(input_ids)
        tokens, hidden_size = inputs_embeds.shape
        streams = self.num_residual_streams
        hidden_states = (
            inputs_embeds.unsqueeze(1)
            .expand(tokens, streams, hidden_size)
            .reshape(tokens, streams * hidden_size)
        )
        for layer in self.layers:
            hidden_states, _ = layer(positions, hidden_states)
        return self.norm(hidden_states.view(tokens, streams, hidden_size).mean(dim=1))


class XingChen4ForCausalLM(DeepseekV2ForCausalLM):
    model_cls = XingChen4Model
    # The base constructor augments this mapping; do not share its dictionary.
    packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Stream backbone weights and load every mHC parameter exactly once.

        Only scalar alpha fragments are buffered, never the full checkpoint.
        Unknown, duplicate or incomplete mHC groups are errors rather than
        silently leaving uninitialized parameters in a runnable model.
        """
        params = dict(self.named_parameters())
        expected = {
            name
            for name in params
            if name.rpartition(".")[0].rsplit(".", 1)[-1] in ("attn_hc", "ffn_hc")
        }
        loaded_hc: set[str] = set()
        pending: dict[str, dict[str, torch.Tensor]] = {}
        mtp_prefix = f"model.layers.{self.config.num_hidden_layers}."
        alpha_names = ("alpha_pre", "alpha_post", "alpha_res")

        def load(name: str, tensor: torch.Tensor) -> None:
            if name not in expected:
                raise ValueError(f"Unexpected XingChen4 mHC parameter: {name}")
            if name in loaded_hc:
                raise ValueError(f"Duplicate XingChen4 mHC parameter: {name}")
            param = params[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, tensor)
            loaded_hc.add(name)

        def backbone_weights() -> Iterator[tuple[str, torch.Tensor]]:
            for name, tensor in weights:
                if name.startswith(mtp_prefix):
                    continue
                prefix, _, field = name.rpartition(".")
                if prefix.rsplit(".", 1)[-1] not in ("attn_hc", "ffn_hc"):
                    yield name, tensor
                    continue
                if field in ("bias_pre", "bias_post", "bias_res"):
                    continue
                if field in alpha_names:
                    group = pending.setdefault(prefix, {})
                    if field in group or f"{prefix}.hc_scale" in loaded_hc:
                        raise ValueError(f"Duplicate XingChen4 alpha: {name}")
                    if tensor.numel() != 1:
                        raise ValueError(f"XingChen4 alpha must be scalar: {name}")
                    group[field] = tensor.reshape(()).to(dtype=torch.float32).clone()
                    if len(group) == 3:
                        load(
                            f"{prefix}.hc_scale",
                            torch.stack([group[key] for key in alpha_names]),
                        )
                        del pending[prefix]
                    continue
                target = {"mapping_weight": "hc_fn", "bias": "hc_base"}.get(
                    field, field
                )
                load(f"{prefix}.{target}", tensor)
            if pending:
                raise ValueError(
                    f"Incomplete XingChen4 alpha groups: {sorted(pending)}"
                )

        loaded = super().load_weights(backbone_weights())
        if missing := expected - loaded_hc:
            raise ValueError(f"Unloaded XingChen4 mHC parameters: {sorted(missing)}")
        return loaded | loaded_hc

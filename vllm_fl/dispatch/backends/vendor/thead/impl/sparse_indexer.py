# SPDX-License-Identifier: Apache-2.0
"""PPU storage bindings and fallback kernels; no FlagGems provider selection."""

import torch

from vllm_fl.kernels.glm5_next import portable


class IndexerOps:
    def validate_environment(self):
        return None

    def prepare_query(self, q, group_size, *, output_dtype, **kwargs):
        """BF16 storage uses identity scales; this is not FP8 quantization."""
        if output_dtype != torch.bfloat16:
            raise ValueError("PPU identity-scale storage requires BF16 output")
        return q.to(output_dtype), torch.ones(
            q.shape[:-1] + (1,), dtype=torch.float32, device=q.device
        )

    def indexer_k_quant_and_cache(self, *args, **kwargs):
        raise NotImplementedError("PPU BF16 indexer requires kpool cache insertion")

    def gather_cache(self, *args, **kwargs):
        from .gather_cache import cp_gather_indexer_k_quant_cache_ppu

        return cp_gather_indexer_k_quant_cache_ppu(*args, **kwargs)

    def paged_mqa_logits(self, q, *args, **kwargs):
        from .paged_mqa import paged_mqa_bf16_logits

        if q[1] is not None:
            raise ValueError("Query scales must already be folded into weights")
        return paged_mqa_bf16_logits(q[0], *args, **kwargs)

    def persist_prefill_tail(self, *args, **kwargs):
        from vllm_fl.kernels.glm5_next.prefill_tail import persist_prefill_tail

        return persist_prefill_tail(*args, **kwargs)

    def kpool_compress_and_write_cache(self, *args, **kwargs):
        from vllm_fl.kernels.glm5_next.kpool_compress import (
            kpool_compress_and_write_cache,
        )

        return kpool_compress_and_write_cache(*args, **kwargs)

    def kpool_decode_update_and_maybe_write_cache_batched(self, *args, **kwargs):
        from vllm_fl.kernels.glm5_next.kpool_compress import (
            kpool_decode_update_and_maybe_write_cache_batched,
        )

        return kpool_decode_update_and_maybe_write_cache_batched(*args, **kwargs)

    def expand_pools_to_tokens(self, *args, **kwargs):
        return portable.expand_pools_to_tokens(*args, **kwargs)

    def append_tail_to_topk(self, *args, **kwargs):
        return portable.append_tail_to_topk(*args, **kwargs)

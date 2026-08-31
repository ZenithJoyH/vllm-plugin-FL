# SPDX-License-Identifier: Apache-2.0
"""Model-facing indexer contract. Backend selection belongs to OpManager."""

from vllm_fl.dispatch import CachedOp


class Glm5NextIndexerBackend:
    @property
    def compressed_page_size(self):
        return CachedOp("glm5_indexer_capabilities")()["compressed_page_size"]

    @property
    def cache_dtype(self):
        return CachedOp("glm5_indexer_capabilities")()["cache_dtype"]

    @property
    def query_dtype(self):
        return CachedOp("glm5_indexer_capabilities")()["query_dtype"]

    validate_environment = staticmethod(CachedOp("glm5_indexer_validate_environment"))
    rotate_indexer_query = staticmethod(CachedOp("glm5_indexer_rotate_indexer_query"))
    per_token_group_quant_fp8 = staticmethod(
        CachedOp("glm5_indexer_per_token_group_quant_fp8")
    )
    indexer_k_quant_and_cache = staticmethod(
        CachedOp("glm5_indexer_indexer_k_quant_and_cache")
    )
    cp_gather_indexer_k_quant_cache = staticmethod(
        CachedOp("glm5_indexer_cp_gather_indexer_k_quant_cache")
    )
    mqa_logits = staticmethod(CachedOp("glm5_indexer_mqa_logits"))
    paged_mqa_logits = staticmethod(CachedOp("glm5_indexer_paged_mqa_logits"))
    topk_prefill = staticmethod(CachedOp("glm5_indexer_topk_prefill"))
    topk_decode = staticmethod(CachedOp("glm5_indexer_topk_decode"))
    pack_seq = staticmethod(CachedOp("glm5_indexer_pack_seq"))
    unpack_seq = staticmethod(CachedOp("glm5_indexer_unpack_seq"))
    persist_prefill_tail = staticmethod(CachedOp("glm5_indexer_persist_prefill_tail"))
    kpool_compress_and_write_cache = staticmethod(
        CachedOp("glm5_indexer_kpool_compress_and_write_cache")
    )
    kpool_decode_update_and_maybe_write_cache_batched = staticmethod(
        CachedOp("glm5_indexer_kpool_decode_update_and_maybe_write_cache_batched")
    )
    expand_pools_to_tokens = staticmethod(
        CachedOp("glm5_indexer_expand_pools_to_tokens")
    )
    append_tail_to_topk = staticmethod(CachedOp("glm5_indexer_append_tail_to_topk"))


INDEXER_BACKEND = Glm5NextIndexerBackend()

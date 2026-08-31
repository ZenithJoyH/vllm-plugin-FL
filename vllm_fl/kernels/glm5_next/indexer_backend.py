# SPDX-License-Identifier: Apache-2.0
"""Model-facing indexer contract. Backend selection belongs to OpManager."""

from vllm_fl.dispatch import CachedOp


class Glm5NextIndexerBackend:
    @property
    def compressed_page_size(self):
        return CachedOp("sparse_indexer_capabilities")()["compressed_page_size"]

    @property
    def cache_dtype(self):
        return CachedOp("sparse_indexer_capabilities")()["cache_dtype"]

    @property
    def query_dtype(self):
        return CachedOp("sparse_indexer_capabilities")()["query_dtype"]

    validate_environment = staticmethod(CachedOp("sparse_indexer_validate_environment"))
    rotate_indexer_query = staticmethod(CachedOp("sparse_indexer_rotate_indexer_query"))
    prepare_query = staticmethod(
        CachedOp("sparse_indexer_prepare_query")
    )
    indexer_k_quant_and_cache = staticmethod(
        CachedOp("sparse_indexer_indexer_k_quant_and_cache")
    )
    gather_cache = staticmethod(
        CachedOp("sparse_indexer_gather_cache")
    )
    mqa_logits = staticmethod(CachedOp("sparse_indexer_mqa_logits"))
    paged_mqa_logits = staticmethod(CachedOp("sparse_indexer_paged_mqa_logits"))
    topk_prefill = staticmethod(CachedOp("sparse_indexer_topk_prefill"))
    topk_decode = staticmethod(CachedOp("sparse_indexer_topk_decode"))
    pack_seq = staticmethod(CachedOp("sparse_indexer_pack_seq"))
    unpack_seq = staticmethod(CachedOp("sparse_indexer_unpack_seq"))
    persist_prefill_tail = staticmethod(CachedOp("sparse_indexer_persist_prefill_tail"))
    kpool_compress_and_write_cache = staticmethod(
        CachedOp("sparse_indexer_kpool_compress_and_write_cache")
    )
    kpool_decode_update_and_maybe_write_cache_batched = staticmethod(
        CachedOp("sparse_indexer_kpool_decode_update_and_maybe_write_cache_batched")
    )
    expand_pools_to_tokens = staticmethod(
        CachedOp("sparse_indexer_expand_pools_to_tokens")
    )
    append_tail_to_topk = staticmethod(CachedOp("sparse_indexer_append_tail_to_topk"))


INDEXER_BACKEND = Glm5NextIndexerBackend()

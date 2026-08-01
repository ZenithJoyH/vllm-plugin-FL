# Copyright (c) 2026 BAAI. All rights reserved.
# Adapted from vllm/v1/attention/backends/mla/flashmla_sparse.py

"""
FlagGems-based FlashMLA Sparse Attention Backend.

This module provides a FlagGems-accelerated implementation of the FlashMLA Sparse
attention backend. It reuses the metadata/builder from vLLM's native
FlashMLASparseBackend and overrides only the kernel dispatch methods to call
FlagGems Triton kernels instead of the native CUDA FlashMLA kernels.
"""

from typing import ClassVar

import torch

from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionCGSupport,
)
from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseBackend,
    FlashMLASparseImpl,
    FlashMLASparseMetadata,
    FlashMLASparseMetadataBuilder,
)

logger = init_logger(__name__)


class MLASparseFLMetadataBuilder(FlashMLASparseMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_BATCH
    )


class MLASparseFLBackend(FlashMLASparseBackend):
    """FlagGems-based FlashMLA Sparse attention backend."""

    @staticmethod
    def get_name() -> str:
        return "FLAGGEMS_FLASHMLA_SPARSE"

    @staticmethod
    def get_impl_cls() -> type["MLASparseFLImpl"]:
        return MLASparseFLImpl

    @staticmethod
    def get_builder_cls() -> type["MLASparseFLMetadataBuilder"]:
        return MLASparseFLMetadataBuilder


class MLASparseFLImpl(FlashMLASparseImpl):
    """FlagGems implementation of FlashMLA Sparse attention.

    Overrides the kernel dispatch methods to use FlagGems Triton kernels
    while reusing all control flow (padding, metadata handling, prefill/decode
    routing) from the parent class.
    """

    def _fp8_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        kernel_metadata: FlashMLASparseMetadata.FP8KernelMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """FP8 decode kernel using FlagGems flash_mla_with_kvcache."""
        from flag_gems.fused.flash_mla_with_kvcache import (
            FlashMLASchedMeta,
            flash_mla_with_kvcache,
        )

        # q shape: (batch, seq_len, num_heads, head_dim)
        actual_num_heads = q.size(2)
        padded_num_heads = self.fp8_decode_padded_heads

        # Pad query if needed (kernel only supports h_q = 64 or 128)
        if actual_num_heads < padded_num_heads:
            logger.warning_once(
                f"Padding num_heads from {actual_num_heads} to "
                f"{padded_num_heads} for FP8 sparse decode kernel"
            )
            q_padded = q.new_zeros(
                (q.size(0), q.size(1), padded_num_heads, q.size(3))
            )
            q_padded[:, :, :actual_num_heads, :] = q
            q = q_padded

        # Adapt vLLM's FlashMLASchedMeta to FlagGems' FlashMLASchedMeta
        original_meta = kernel_metadata.scheduler_metadata
        if isinstance(original_meta, FlashMLASchedMeta):
            flaggems_meta = original_meta
        else:
            flaggems_meta = FlashMLASchedMeta(
                have_initialized=original_meta.have_initialized,
                config=original_meta.config,
                tile_scheduler_metadata=original_meta.tile_scheduler_metadata,
                num_splits=original_meta.num_splits,
            )

        out, lse = flash_mla_with_kvcache(
            q=q,
            k_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(-2),
            block_table=kernel_metadata.dummy_block_table,
            cache_seqlens=kernel_metadata.cache_lens,
            head_dim_v=512,
            tile_scheduler_metadata=flaggems_meta,
            is_fp8_kvcache=True,
            indices=topk_indices,
            softmax_scale=self.softmax_scale,
        )

        # Sync metadata back if we created a new object
        if flaggems_meta is not original_meta:
            original_meta.have_initialized = flaggems_meta.have_initialized
            original_meta.config = flaggems_meta.config
            original_meta.tile_scheduler_metadata = (
                flaggems_meta.tile_scheduler_metadata
            )
            original_meta.num_splits = flaggems_meta.num_splits

        # Slice output back to actual head count if we padded
        if actual_num_heads < padded_num_heads:
            out = out[:, :, :actual_num_heads, :]

        return out, lse

    def _bf16_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """BF16 prefill/decode kernel using FlagGems flash_mla_sparse_fwd."""
        import flag_gems

        num_tokens = q.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )

        # Kernel requires num_local_head to be a multiple of
        # 64 on hopper and 128 on blackwell
        if self.num_heads % self.prefill_padding != 0:
            assert self.prefill_padding % self.num_heads == 0
            logger.warning_once(
                f"Padding num_heads from {self.num_heads} to "
                f"{self.prefill_padding} for BF16 sparse prefill kernel"
            )
            q_padded = q.new_empty(
                (q.shape[0], self.prefill_padding, q.shape[2])
            )
            q_padded[:, : self.num_heads, :] = q
            q = q_padded

        topk_indices = topk_indices.view(num_tokens, 1, -1)
        output, _, _ = flag_gems.flash_mla_sparse_fwd(
            q=q,
            kv=kv_c_and_k_pe_cache,
            indices=topk_indices,
            sm_scale=self.softmax_scale,
        )

        output = output[:, : self.num_heads, :]
        return output

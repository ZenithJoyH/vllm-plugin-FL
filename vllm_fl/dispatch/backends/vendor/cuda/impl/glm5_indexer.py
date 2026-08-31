# SPDX-License-Identifier: Apache-2.0
"""GLM indexer implementations for this vendor only."""

import torch


class IndexerOps:
    def validate_environment(self):
        from vllm.utils.deep_gemm import has_deep_gemm

        if not has_deep_gemm():
            raise RuntimeError("GLM CUDA sparse indexer requires DeepGEMM")

    def rotate_indexer_query(self, q: torch.Tensor) -> torch.Tensor:
        """Use the reference CUDA transform in kpool's normalized H128 basis."""
        if q.shape[-1] != 128:
            raise ValueError("GLM5-Next indexer query rotation requires head_dim=128")
        if q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError(f"Unsupported indexer query dtype: {q.dtype}")
        from fast_hadamard_transform import hadamard_transform

        return hadamard_transform(q, scale=128 ** (-0.5))

    def per_token_group_quant_fp8(self, *args, **kwargs):
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            per_token_group_quant_fp8,
        )

        return per_token_group_quant_fp8(*args, **kwargs)

    def indexer_k_quant_and_cache(self, *args, **kwargs) -> None:
        from vllm import _custom_ops as ops

        return ops.indexer_k_quant_and_cache(*args, **kwargs)

    def cp_gather_indexer_k_quant_cache(self, *args, **kwargs) -> None:
        from vllm import _custom_ops as ops

        return ops.cp_gather_indexer_k_quant_cache(*args, **kwargs)

    def mqa_logits(self, *args, **kwargs) -> torch.Tensor:
        from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

        return fp8_fp4_mqa_logits(*args, **kwargs)

    def paged_mqa_logits(self, *args, **kwargs) -> torch.Tensor:
        from vllm.utils.deep_gemm import fp8_fp4_paged_mqa_logits

        return fp8_fp4_paged_mqa_logits(*args, **kwargs)

    def topk_prefill(
        self,
        logits: torch.Tensor,
        row_starts: torch.Tensor,
        row_ends: torch.Tensor,
        indices: torch.Tensor,
        num_rows: int,
        stride0: int,
        stride1: int,
        top_k: int,
    ) -> None:
        torch.ops._C.top_k_per_row_prefill(
            logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
        )
        return

    def topk_decode(
        self,
        logits: torch.Tensor,
        next_n: int,
        seq_lens: torch.Tensor,
        indices: torch.Tensor,
        num_rows: int,
        stride0: int,
        stride1: int,
        top_k: int,
        *,
        max_seq_len: int | None = None,
    ) -> None:
        if top_k in (512, 1024, 2048) and max_seq_len is not None:
            from vllm.v1.worker.workspace import current_workspace_manager

            (workspace,) = current_workspace_manager().get_simultaneous(
                ((1024 * 1024,), torch.uint8),
            )
            torch.ops._C.persistent_topk(
                logits, seq_lens, indices, workspace, top_k, max_seq_len
            )
            return
        torch.ops._C.top_k_per_row_decode(
            logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
        )
        return

    def pack_seq(self, tensor, lengths, pad_value=-float("inf")):
        from vllm.v1.attention.ops.common import pack_seq_triton

        return pack_seq_triton(tensor, lengths, pad_value=pad_value)

    def unpack_seq(self, tensor, lengths):
        from vllm.v1.attention.ops.common import unpack_seq_triton

        return unpack_seq_triton(tensor, lengths)

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
        from vllm_fl.kernels.glm5_next.kpool_compress import expand_pools_to_tokens

        return expand_pools_to_tokens(*args, **kwargs)

    def append_tail_to_topk(self, *args, **kwargs):
        from vllm_fl.kernels.glm5_next.kpool_compress import append_tail_to_topk

        return append_tail_to_topk(*args, **kwargs)

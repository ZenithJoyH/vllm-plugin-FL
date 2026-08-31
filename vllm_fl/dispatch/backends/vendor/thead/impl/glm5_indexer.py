# SPDX-License-Identifier: Apache-2.0
"""GLM indexer implementations for this vendor only."""

import torch
from typing import Callable
from vllm.logger import init_logger
from vllm_fl.kernels.glm5_next import portable

logger = init_logger(__name__)

import importlib
from vllm_fl.utils import use_flaggems_op
from vllm_fl.dispatch.backends.reference.impl.glm5_indexer import (
    _torch_mqa_logits,
    _torch_topk,
    _torch_pack_seq,
    _torch_unpack_seq,
)


def _load_flaggems_op(module, name):
    # Resolve optional dependencies before invocation, never recover from a
    # kernel RuntimeError after it may already have mutated a cache.
    if not use_flaggems_op(name):
        return None
    try:
        return getattr(importlib.import_module("flag_gems.fused." + module), name)
    except (ImportError, AttributeError):
        return None


def _load_flaggems_ops_op(module, name):
    if not use_flaggems_op(name):
        return None
    try:
        return getattr(importlib.import_module("flag_gems.ops." + module), name)
    except (ImportError, AttributeError):
        return None


class IndexerOps:
    def validate_environment(self):
        # No CUDA DeepGEMM dependency; required kernels load on selection.
        return None

    def __init__(self):
        self._flag_ops = {}
        self._provider_logged = set()

    def _flag(self, module: str, name: str) -> Callable | None:
        if name not in self._flag_ops:
            self._flag_ops[name] = _load_flaggems_op(module, name)
        fn = self._flag_ops[name]
        if name not in self._provider_logged:
            logger.info(
                "GLM5-Next op %s provider: %s",
                name,
                "FlagGems" if fn is not None else "PyTorch correctness fallback",
            )
            self._provider_logged.add(name)
        return fn

    def _call_flag(self, name, fn, fallback, *args, **kwargs):
        return (fn if fn is not None else fallback)(*args, **kwargs)

    def rotate_indexer_query(self, q: torch.Tensor) -> torch.Tensor:
        """Match the normalized Hadamard-128 basis written by kpool.

        Use FP32 butterflies, then round to the original dtype, matching the
        supplied reference and avoiding low-precision transform accumulation.
        Provider selection remains in the indexer dispatch layer and respects
        the FlagOS blacklist. Both paths have shape-static graph-safe tensor
        operations; provider discovery is completed by normal eager warm-up.
        """
        if q.shape[-1] != 128:
            raise ValueError("GLM5-Next indexer query rotation requires head_dim=128")
        if q.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError(f"Unsupported indexer query dtype: {q.dtype}")
        name = "hadamard_transform"
        if name not in self._flag_ops:
            self._flag_ops[name] = _load_flaggems_ops_op(name, name)
        fn = self._flag_ops[name]
        if name not in self._provider_logged:
            logger.info(
                "GLM5-Next query rotation provider: %s",
                "FlagGems FP32 Hadamard" if fn is not None else "PyTorch Hadamard",
            )
            self._provider_logged.add(name)

        def fallback(value, scale):
            del scale
            return portable.hadamard128(value)

        rotated = self._call_flag(name, fn, fallback, q.float(), scale=128 ** (-0.5))
        return rotated.to(q.dtype)

    def per_token_group_quant_fp8(self, *args, **kwargs):
        q, group_size = (args[0], args[1])
        del group_size
        q_bf16 = q.to(torch.bfloat16)
        scale = torch.ones(q.shape[:-1] + (1,), dtype=torch.float32, device=q.device)
        return (q_bf16, scale)

    def indexer_k_quant_and_cache(self, *args, **kwargs) -> None:
        raise NotImplementedError("PPU BF16 indexer requires kpool cache insertion")

    def cp_gather_indexer_k_quant_cache(self, *args, **kwargs) -> None:
        # The pinned FlagGems gather only supports quantized cache layouts.
        from .glm5_gather_cache import cp_gather_indexer_k_quant_cache_ppu

        return cp_gather_indexer_k_quant_cache_ppu(*args, **kwargs)

    def mqa_logits(self, *args, **kwargs) -> torch.Tensor:
        fn = self._flag("fp8_fp4_mqa_logits", "fp8_fp4_mqa_logits")
        return self._call_flag(
            "fp8_fp4_mqa_logits", fn, _torch_mqa_logits, *args, **kwargs
        )

    def paged_mqa_logits(self, *args, **kwargs) -> torch.Tensor:
        fn = self._flag("bf16_paged_mqa_logits", "bf16_paged_mqa_logits")
        if fn is not None:
            q_values, q_scale = args[0]
            del q_scale
            kv_cache = args[1]
            if q_values.is_floating_point() and kv_cache.is_floating_point():
                block_size = kv_cache.shape[1]
                if block_size == 64:
                    return fn(
                        q_values,
                        kv_cache,
                        args[2],
                        args[3],
                        args[4],
                        args[5],
                        max_context_len=kwargs["max_model_len"],
                        clean_logits=kwargs.get("clean_logits", False),
                    )
                from .glm5_paged_mqa import paged_mqa_bf16_logits

                logger.info_once(
                    f"FlagGems bf16_paged_mqa_logits requires block_size=64; got {block_size}; using plugin BF16 Triton paged-MQA"
                )
                return paged_mqa_bf16_logits(
                    q_values,
                    kv_cache,
                    args[2],
                    args[3],
                    args[4],
                    args[5],
                    max_model_len=kwargs["max_model_len"],
                    clean_logits=kwargs.get("clean_logits", False),
                )
        from .glm5_paged_mqa import paged_mqa_bf16_logits

        return paged_mqa_bf16_logits(args[0][0], *args[1:], **kwargs)

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
        fn = self._flag("top_k_per_row_prefill", "top_k_per_row_prefill")
        if fn is not None:
            fn(logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k)
            return
        indices.copy_(_torch_topk(logits, row_starts, row_ends, top_k, True))

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
        del max_seq_len
        fn = self._flag("top_k_per_row_decode", "top_k_per_row_decode")
        if fn is not None:
            fn(logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k)
            return
        if seq_lens.ndim == 2:
            ends = seq_lens.reshape(-1)[:num_rows]
        else:
            ends = seq_lens.repeat_interleave(next_n)[:num_rows]
        starts = torch.zeros_like(ends)
        indices.copy_(_torch_topk(logits, starts, ends, top_k, False))

    def pack_seq(self, tensor, lengths, pad_value=-float("inf")):
        fn = self._flag("pack_seq", "pack_seq_triton")
        return self._call_flag(
            "pack_seq_triton", fn, _torch_pack_seq, tensor, lengths, pad_value=pad_value
        )

    def unpack_seq(self, tensor, lengths):
        fn = self._flag("unpack_seq", "unpack_seq_triton")
        return self._call_flag(
            "unpack_seq_triton", fn, _torch_unpack_seq, tensor, lengths
        )

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

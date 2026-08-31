# SPDX-License-Identifier: Apache-2.0
"""FlagGems-backed MLA prefill backend for OOT (non-NVIDIA) platforms.

vLLM 0.24 instantiates an MLA prefill backend inside every
``MLAAttention`` layer via ``get_mla_prefill_backend(vllm_config)``.  On the
T-Head PPU the vLLM-native ``flash_attn_varlen_func`` is a stub (no CUDA C
extensions), so the stock ``FlashAttnPrefillBackend`` fails its import-time
assertion and model construction dies before any forward ever runs.

This backend mirrors ``FlashAttnPrefillBackend``'s control flow but drives the
FlagGems ``flash_attn_varlen_func`` (keyword-style, requires equal k/v head
dims).  It is registered as an override of ``MLAPrefillBackendEnum.FLASH_ATTN``
on non-NVIDIA platforms by the GLM5-Next v0.24 patches.
"""

from __future__ import annotations

import torch

from vllm.v1.attention.backends.mla.prefill.base import (
    MLAPrefillBackend,
)

__all__ = ["FlagGemsMLAPrefillBackend"]


class FlagGemsMLAPrefillBackend(MLAPrefillBackend):
    """MLA prefill backend implemented with FlagGems Triton kernels."""

    supported_dtypes: list[torch.dtype] = [torch.float16, torch.bfloat16]

    @staticmethod
    def get_name() -> str:
        return "FLAGGEMS_MLA_PREFILL"

    @classmethod
    def is_available(cls) -> bool:
        try:
            from flag_gems import flash_attn_varlen_func  # noqa: F401

            return True
        except ImportError:
            return False

    def __init__(
        self,
        num_heads: int,
        scale: float,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        vllm_config,
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            scale=scale,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            vllm_config=vllm_config,
        )
        from flag_gems import flash_attn_varlen_func

        self._flag_gems_varlen = flash_attn_varlen_func

    def _flash_attn_varlen_diff_headdims(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool = False,
        softmax_scale: float | None = None,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # FlagGems' flash_attn_varlen_func requires k.size() == v.size().
        # Pad v with zeros to match q/k's head dimension when they differ.
        v_head_dim = v.shape[-1]
        if v.shape[-1] != q.shape[-1]:
            v = torch.nn.functional.pad(v, [0, q.shape[-1] - v.shape[-1]], value=0)

        # FlagGems does not support the vLLM FA-specific parameters.
        kwargs.pop("fa_version", None)
        kwargs.pop("output_scale", None)

        if return_softmax_lse:
            kwargs["return_softmax_lse"] = True

        attn_out = self._flag_gems_varlen(
            q=q,
            k=k,
            v=v,
            softmax_scale=softmax_scale,
            out=out,
            **kwargs,
        )

        lse = None
        if isinstance(attn_out, tuple):
            attn_out, lse = attn_out[0], attn_out[1]

        # Slice output back to the original v_head_dim (padded dims are zeros).
        if attn_out.shape[-1] != v_head_dim:
            attn_out = attn_out[..., :v_head_dim]

        if return_softmax_lse:
            return (
                attn_out.clone(),
                lse.clone() if isinstance(lse, torch.Tensor) else lse,
            )
        return attn_out.clone()

    def run_prefill_new_tokens(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_softmax_lse: bool,
        out: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=self._prefill_metadata.query_start_loc,
            cu_seqlens_k=self._prefill_metadata.query_start_loc,
            max_seqlen_q=self._prefill_metadata.max_query_len,
            max_seqlen_k=self._prefill_metadata.max_query_len,
            softmax_scale=self.scale,
            causal=True,
            return_softmax_lse=return_softmax_lse,
            out=out,
            output_scale=output_scale,
        )

    def run_prefill_context_chunk(
        self,
        chunk_idx: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self._prefill_metadata.chunked_context is not None
        return self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=self._prefill_metadata.query_start_loc,
            cu_seqlens_k=(
                self._prefill_metadata.chunked_context.cu_seq_lens[chunk_idx]
            ),
            max_seqlen_q=self._prefill_metadata.max_query_len,
            max_seqlen_k=(
                self._prefill_metadata.chunked_context.max_seq_lens[chunk_idx]
            ),
            softmax_scale=self.scale,
            causal=False,  # Context is unmasked
            return_softmax_lse=True,
        )

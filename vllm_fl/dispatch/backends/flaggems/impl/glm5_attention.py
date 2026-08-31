# SPDX-License-Identifier: Apache-2.0
"""GLM sparse MLA: NoPE query assembly and FlagGems cache insertion."""

import torch
from vllm_fl.dispatch.backends.flaggems.impl.mla_sparse import (
    MLASparseFLBackend,
    MLASparseFLImpl,
)


class Glm5SparseImpl(MLASparseFLImpl):
    def do_kv_cache_update(
        self, kv_c_normed, k_pe, kv_cache, slot_mapping, kv_cache_dtype, k_scale
    ):
        from flag_gems import concat_and_cache_mla

        if kv_cache.numel():
            concat_and_cache_mla(
                kv_c_normed,
                k_pe.squeeze(1),
                kv_cache,
                slot_mapping.flatten(),
                kv_cache_dtype=kv_cache_dtype,
                scale=k_scale,
            )

    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        if isinstance(q, tuple):
            nope, rope = q
            q = self.q_concat_buffer[: nope.shape[0]]
            torch.cat((nope, rope), dim=-1, out=q)
        return super().forward_mqa(q, kv_c_and_k_pe_cache, attn_metadata, layer)


class Glm5SparseBackend(MLASparseFLBackend):
    @staticmethod
    def get_impl_cls():
        return Glm5SparseImpl

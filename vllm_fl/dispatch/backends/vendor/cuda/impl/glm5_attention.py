# SPDX-License-Identifier: Apache-2.0
"""GLM NoPE support without replacing CUDA's global concat_mla_q."""

from vllm.v1.attention.backends.mla.flashmla_sparse import (
    FlashMLASparseBackend,
    FlashMLASparseImpl,
)


class Glm5SparseImpl(FlashMLASparseImpl):
    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        if isinstance(q, tuple) and q[1].shape[-1] == 0:
            nope, rope = q
            q = self.q_concat_buffer[: nope.shape[0]]
            q.copy_(nope)
        return super().forward_mqa(q, kv_c_and_k_pe_cache, attn_metadata, layer)


class Glm5SparseBackend(FlashMLASparseBackend):
    @staticmethod
    def get_impl_cls():
        return Glm5SparseImpl

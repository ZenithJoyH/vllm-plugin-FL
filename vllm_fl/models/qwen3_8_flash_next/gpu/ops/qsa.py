# SPDX-License-Identifier: Apache-2.0
"""Model bindings resolved once before graph capture.

Cache-writing operators deliberately use resolve_op, not call_op: a failed
state update must not be retried with a different backend. Restart the worker
after changing dispatch policy.
"""

from vllm_fl.dispatch import resolve_op

qsa_mqa_paged = resolve_op("qwen38_qsa_mqa_paged")
expand_qsa_block_indices = resolve_op("qwen38_expand_qsa_block_indices")
qsa_select_paged_tokens = resolve_op("qwen38_qsa_select_paged_tokens")
qsa_sparse_paged_attention = resolve_op("qwen38_qsa_sparse_paged_attention")
qsa_store_cache_rows = resolve_op("qwen38_qsa_store_cache_rows")
qsa_compress_groups_with_ratio = resolve_op("qwen38_qsa_compress_groups_with_ratio")

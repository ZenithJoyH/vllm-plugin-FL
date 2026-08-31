# SPDX-License-Identifier: Apache-2.0
"""PLE state dispatch bindings; resolve before graph capture, never retry writes."""

from vllm_fl.dispatch import resolve_op

ple_state_gather = resolve_op("qwen38_ple_state_gather")
ple_state_scatter_ = resolve_op("qwen38_ple_state_scatter_")

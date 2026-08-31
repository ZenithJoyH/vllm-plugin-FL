# SPDX-License-Identifier: Apache-2.0
"""Keep GLM compatibility adapters scoped to the supported vLLM 0.24 API."""

from __future__ import annotations

import vllm
from packaging.version import Version

_VLLM_024 = Version(vllm.__version__).release[:2] == (0, 24)


def is_vllm_024() -> bool:
    return _VLLM_024


__all__ = ["is_vllm_024"]

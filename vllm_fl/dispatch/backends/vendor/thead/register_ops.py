# SPDX-License-Identifier: Apache-2.0
"""GLM-specific PPU registrations; do not replace generic attention backends."""

from vllm_fl.dispatch.backends.glm5_registration import register_vendor_glm5


class Glm5TheadBackend:
    @staticmethod
    def is_available():
        from vllm.platforms import current_platform

        return getattr(current_platform, "vendor_name", None) == "thead"


def register_builtins(registry):
    register_vendor_glm5(registry, Glm5TheadBackend(), "thead")

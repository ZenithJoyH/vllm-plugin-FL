# SPDX-License-Identifier: Apache-2.0
"""Checkpoint configuration and scoped MLA conversion for XingChen4."""

from transformers import DeepseekV3Config


def _resolve_alias(legacy, canonical, default, name):
    if legacy is not None and canonical is not None and legacy != canonical:
        raise ValueError(f"Conflicting XingChen4 config alias: {name}")
    return (
        canonical
        if canonical is not None
        else legacy
        if legacy is not None
        else default
    )


class XingChen4Config(DeepseekV3Config):
    model_type = "xingchen4"

    def __init__(
        self,
        hc_mult=None,
        hc_sinkhorn_iters=None,
        hc_eps=None,
        num_residual_streams=None,
        mhc_sinkhorn_iterations=None,
        mhc_norm_eps=None,
        **kwargs,
    ):
        streams = _resolve_alias(
            hc_mult, num_residual_streams, 4, "num_residual_streams"
        )
        iterations = _resolve_alias(
            hc_sinkhorn_iters, mhc_sinkhorn_iterations, 20, "mhc_sinkhorn_iterations"
        )
        eps = _resolve_alias(hc_eps, mhc_norm_eps, 1e-6, "mhc_norm_eps")
        if not isinstance(streams, int) or isinstance(streams, bool) or streams <= 0:
            raise ValueError("num_residual_streams must be a positive integer")
        if (
            not isinstance(iterations, int)
            or isinstance(iterations, bool)
            or iterations <= 0
        ):
            raise ValueError("mhc_sinkhorn_iterations must be a positive integer")
        if not eps > 0:
            raise ValueError("mhc_norm_eps must be positive")
        # Preserve legacy clamp fields as metadata. Like this checkpoint's HF
        # implementation and the previous adapter, mHC does not apply clamping.
        # Transformers normalizes RoPE dictionaries in-place. Keep the caller's
        # checkpoint mapping intact and normalize a private copy instead.
        rope_scaling = dict(kwargs.get("rope_scaling") or {})
        if "rope_scaling" in kwargs:
            kwargs["rope_scaling"] = dict(rope_scaling)
        if isinstance(kwargs.get("rope_parameters"), dict):
            kwargs["rope_parameters"] = dict(kwargs["rope_parameters"])
        kwargs.pop("model_type", None)
        super().__init__(**kwargs)
        self.num_residual_streams = self.hc_mult = streams
        self.mhc_sinkhorn_iterations = self.hc_sinkhorn_iters = iterations
        self.mhc_norm_eps = self.hc_eps = eps
        self.mhc_pre_eps = kwargs.get("mhc_pre_eps", eps)
        self.mhc_post_mult_value = kwargs.get("mhc_post_mult_value", 2.0)
        # vLLM's DeepSeek attention consumes rope_parameters. Preserve every
        # checkpoint YaRN value and remove only the redundant legacy type key.
        if rope_scaling:
            self.rope_parameters = dict(rope_scaling)
            self.rope_parameters.pop("type", None)
            self.rope_parameters["rope_type"] = "deepseek_yarn"
            self.rope_parameters["rope_theta"] = kwargs.get("rope_theta", 10000.0)
            self.rope_parameters["apply_yarn_scaling"] = True
        # The upstream attention implementation uses hasattr to detect DSA.
        if getattr(self, "index_topk", None) is None and hasattr(self, "index_topk"):
            del self.index_topk

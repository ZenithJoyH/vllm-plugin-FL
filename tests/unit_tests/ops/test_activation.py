# Copyright (c) 2025 BAAI. All rights reserved.

"""
Tests for activation ops.
"""

from unittest.mock import patch

import pytest
import torch


class TestSiluAndMulFL:
    """Test SiluAndMulFL class behavior."""

    @pytest.fixture
    def mock_cached_op(self):
        with patch("vllm_fl.ops.activation._silu_and_mul") as mock:
            yield mock

    @pytest.fixture
    def mock_parent_init(self):
        with patch("vllm_fl.ops.activation.SiluAndMul.__init__", return_value=None):
            yield

    def test_forward_oot_dispatches_correctly(self, mock_parent_init, mock_cached_op):
        """Test forward_oot calls dispatch system with correct op name and input."""
        from vllm_fl.ops.activation import SiluAndMulFL

        mock_cached_op.return_value = torch.randn(2, 4)
        layer = SiluAndMulFL()
        x = torch.randn(2, 8)

        result = layer.forward_oot(x)

        mock_cached_op.assert_called_once_with(layer, x)
        assert result.shape == (2, 4)


def test_silu_and_mul_with_clamp_reference_capability():
    from vllm_fl.dispatch.backends.reference.impl.activation import (
        silu_and_mul_with_clamp,
    )

    x = torch.tensor([[3.0, -2.0, 4.0, -5.0]])
    actual = silu_and_mul_with_clamp(x, limit=2.5, alpha=0.75, beta=0.25)
    gate, up = x.chunk(2, dim=-1)
    gate = gate.clamp(max=2.5)
    up = up.clamp(min=-2.5, max=2.5)
    expected = gate * torch.sigmoid(0.75 * gate) * (up + 0.25)
    torch.testing.assert_close(actual, expected)

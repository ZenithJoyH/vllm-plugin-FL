# Copyright (c) 2026 BAAI. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_fl.dispatch.backends.vendor.thead.impl import causal_conv1d


def make_inputs(tokens: int, requests: int = 1):
    x = torch.empty((tokens, 512), dtype=torch.bfloat16).T
    weight = torch.empty((512, 4), dtype=torch.bfloat16)
    conv_states = torch.empty((requests + 1, 512, 3), dtype=torch.bfloat16)
    query_start_loc = torch.arange(
        0, tokens + 1, tokens // requests, dtype=torch.int32
    )
    return x, weight, conv_states, query_start_loc


@pytest.mark.parametrize(
    ("tokens", "requests", "expected"),
    ((32, 1, False), (128, 1, True), (1024, 32, False), (4096, 32, True)),
)
def test_use_native_has_a_bounded_shape_crossover(tokens, requests, expected):
    x, weight, conv_states, query_start_loc = make_inputs(tokens, requests)
    assert (
        causal_conv1d._use_native(
            x,
            weight,
            conv_states,
            query_start_loc,
            metadata=SimpleNamespace(),
            block_idx_first_scheduled_token=None,
            block_idx_last_scheduled_token=None,
            initial_state_idx=None,
            num_computed_tokens=None,
        )
        is expected
    )


def test_use_native_rejects_unsupported_contracts():
    x, weight, conv_states, query_start_loc = make_inputs(128)
    common = {
        "metadata": SimpleNamespace(),
        "block_idx_first_scheduled_token": None,
        "block_idx_last_scheduled_token": None,
        "initial_state_idx": None,
        "num_computed_tokens": None,
    }
    assert not causal_conv1d._use_native(
        x.float(), weight, conv_states, query_start_loc, **common
    )
    assert not causal_conv1d._use_native(
        x, weight[:, :3], conv_states, query_start_loc, **common
    )
    assert not causal_conv1d._use_native(
        x, weight, conv_states, query_start_loc, **{**common, "metadata": None}
    )
    assert not causal_conv1d._use_native(
        x,
        weight,
        conv_states,
        query_start_loc,
        **{**common, "block_idx_last_scheduled_token": torch.empty(1)},
    )


@pytest.mark.parametrize(("tokens", "expected"), ((32, "flaggems"), (128, "native")))
def test_adapter_routes_without_changing_default_slot_ids(monkeypatch, tokens, expected):
    x, weight, conv_states, query_start_loc = make_inputs(tokens)
    native = Mock(return_value=torch.empty_like(x))
    flaggems = Mock(return_value=torch.empty_like(x))
    monkeypatch.setattr(causal_conv1d, "_native_causal_conv1d_fn", native)
    monkeypatch.setattr(causal_conv1d, "_flaggems_causal_conv1d_fn", flaggems)

    causal_conv1d.causal_conv1d_fn(
        x,
        weight,
        None,
        conv_states,
        query_start_loc,
        metadata=SimpleNamespace(),
    )

    selected = native if expected == "native" else flaggems
    rejected = flaggems if expected == "native" else native
    selected.assert_called_once()
    rejected.assert_not_called()
    assert selected.call_args.kwargs["pad_slot_id"] == -1
    assert selected.call_args.kwargs["null_block_id"] == -1

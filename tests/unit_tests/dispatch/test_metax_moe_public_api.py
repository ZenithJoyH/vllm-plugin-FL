# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU-only contracts for the real TritonExpertsFL.apply method.

Load the method from its AST to avoid initializing platform SDKs. These tests
prove routing and buffer/argument contracts, not hardware numerical accuracy.
"""

import ast
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch


class Activation(Enum):
    SILU = "silu"
    GELU = "gelu"


class StagedFallback(Exception):
    pass


@pytest.fixture
def caller():
    source = (
        Path(__file__).resolve().parents[3] / "vllm_fl/ops/fused_moe/fused_moe_utils.py"
    )
    tree = ast.parse(source.read_text())
    klass = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "TritonExpertsFL"
    )
    method = next(
        n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == "apply"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            method,
        ],
        type_ignores=[],
    )
    platform = SimpleNamespace(name="metax", device_name="metax")
    namespace = {
        "torch": torch,
        "MoEActivation": Activation,
        "get_platform_name": lambda: platform.name,
        "current_platform": platform,
    }
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    obj = SimpleNamespace(
        _lora_context=None,
        quant_config=SimpleNamespace(
            use_int8_w8a8=False,
            use_fp8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
        ),
        w1_bias=object(),
        w2_bias=object(),
        per_act_token_quant=False,
        w1_scale=None,
        w2_scale=None,
        block_shape=None,
    )

    def staged(*args):
        raise StagedFallback()

    obj.moe_problem_size = staged
    x = torch.ones(2, 8, dtype=torch.bfloat16)
    w1 = torch.ones(4, 16, 8, dtype=x.dtype)
    w2 = torch.ones(4, 8, 8, dtype=x.dtype)
    weights = torch.ones(2, 2)
    ids = torch.zeros(2, 2, dtype=torch.int32)
    ws13, ws2 = torch.empty(64, dtype=x.dtype), torch.empty(64, dtype=x.dtype)
    args = [
        torch.empty_like(x),
        x,
        w1,
        w2,
        weights,
        ids,
        Activation.SILU,
        4,
        None,
        None,
        None,
        ws13,
        ws2,
        None,
        True,
    ]
    return obj, namespace["apply"], args, platform


def library(monkeypatch, calls):
    gems = ModuleType("flaggems_vllm")

    def fused(*args, **kwargs):
        calls.append((args, kwargs))
        if "out" in kwargs:
            kwargs["out"].fill_(3)
            return kwargs["out"]
        return torch.full_like(args[0], 3)

    gems.fused_experts_impl = fused
    monkeypatch.setitem(sys.modules, "flag_gems", None)
    monkeypatch.setitem(sys.modules, "flaggems_vllm", gems)
    # Any access to the old private import fails.
    monkeypatch.setitem(sys.modules, "flaggems_vllm.runtime.backend._metax", None)
    return gems


@pytest.mark.parametrize("platform_name", ["metax", "nvidia"])
def test_one_public_library_and_no_workspace_probe(caller, monkeypatch, platform_name):
    obj, apply, args, platform = caller
    platform.name = platform.device_name = platform_name
    calls = []
    gems = library(monkeypatch, calls)

    def fail_probe(*a, **kw):
        raise AssertionError("Plugin must not probe caller workspace")

    gems.supports_fused_experts_workspace = fail_probe
    apply(obj, *args)
    assert len(calls) == 1
    actual, kw = calls[0]
    assert all(a is b for a, b in zip(actual, args[1:6]))
    assert not {"out", "workspace13", "workspace2"} & kw.keys()
    assert kw["inplace"] is False and kw["activation"] == "silu"
    assert kw["apply_router_weight_on_input"] is True
    assert kw["global_num_experts"] == 4 and kw["expert_map"] is None
    assert kw["w1_bias"] is obj.w1_bias and kw["w2_bias"] is obj.w2_bias
    assert torch.all(args[0] == 3)


@pytest.mark.parametrize("capability", ["absent", "noncallable", "raising"])
def test_workspace_capability_does_not_change_public_call(
    caller, monkeypatch, capability
):
    obj, apply, args, _ = caller
    calls = []
    gems = library(monkeypatch, calls)
    if capability == "noncallable":
        gems.supports_fused_experts_workspace = None
    elif capability == "raising":

        def fail(*a, **kw):
            raise RuntimeError("must not inspect this API")

        gems.supports_fused_experts_workspace = fail
    apply(obj, *args)
    assert len(calls) == 1 and torch.all(args[0] == 3)


@pytest.mark.parametrize("platform_name", ["metax", "nvidia"])
def test_public_execution_failure_does_not_retry(caller, monkeypatch, platform_name):
    obj, apply, args, platform = caller
    platform.name = platform.device_name = platform_name
    calls = []
    gems = library(monkeypatch, calls)

    def fail(*a, **kw):
        calls.append((a, kw))
        raise RuntimeError("public kernel failure")

    gems.fused_experts_impl = fail
    with pytest.raises(RuntimeError, match="public kernel failure"):
        apply(obj, *args)
    assert len(calls) == 1


@pytest.mark.parametrize("platform_name", ["metax", "nvidia"])
def test_missing_library_operator_is_not_hidden(caller, monkeypatch, platform_name):
    obj, apply, args, platform = caller
    platform.name = platform.device_name = platform_name
    monkeypatch.setitem(sys.modules, "flaggems_vllm", ModuleType("flaggems_vllm"))
    monkeypatch.setitem(sys.modules, "flag_gems", None)
    with pytest.raises(ImportError, match="fused_experts_impl"):
        apply(obj, *args)


def test_lora_keeps_framework_staged_path(caller, monkeypatch):
    obj, apply, args, _ = caller
    obj._lora_context = object()
    monkeypatch.setitem(sys.modules, "flaggems_vllm", None)
    with pytest.raises(StagedFallback):
        apply(obj, *args)


def test_w8a8_keeps_dedicated_experts(caller):
    obj, apply, args, _ = caller
    obj.quant_config.use_int8_w8a8 = True
    with pytest.raises(RuntimeError, match="W8A8 MoE must use TritonW8A8Experts"):
        apply(obj, *args)


@pytest.mark.parametrize("platform_name", ["metax", "nvidia"])
@pytest.mark.parametrize(
    "boundary",
    ["expert_map", "activation", "float32", "weight_dtype", "fp8", "w8a16", "w4a16"],
)
def test_library_receives_semantics_without_platform_guards(
    caller, monkeypatch, platform_name, boundary
):
    obj, apply, args, platform = caller
    platform.name = platform.device_name = platform_name
    if boundary == "expert_map":
        args[8] = torch.arange(4)
    elif boundary == "activation":
        args[6] = Activation.GELU
    elif boundary == "float32":
        args[1:4] = [x.float() for x in args[1:4]]
    elif boundary == "weight_dtype":
        args[2] = args[2].float()
    else:
        setattr(
            obj.quant_config,
            {
                "fp8": "use_fp8_w8a8",
                "w8a16": "use_int8_w8a16",
                "w4a16": "use_int4_w4a16",
            }[boundary],
            True,
        )
    calls = []
    library(monkeypatch, calls)
    apply(obj, *args)
    assert len(calls) == 1
    actual, kw = calls[0]
    assert all(a is b for a, b in zip(actual, args[1:6]))
    assert kw["expert_map"] is args[8] and kw["activation"] == args[6].value
    assert kw["use_fp8_w8a8"] == obj.quant_config.use_fp8_w8a8
    assert kw["use_int8_w8a16"] == obj.quant_config.use_int8_w8a16
    assert kw["use_int4_w4a16"] == obj.quant_config.use_int4_w4a16
    assert not {"out", "workspace13", "workspace2"} & kw.keys()


@pytest.mark.parametrize("platform_name", ["metax", "nvidia"])
@pytest.mark.parametrize("tokens", [0, 1, 2])
def test_empty_single_and_regular_outputs(caller, monkeypatch, platform_name, tokens):
    obj, apply, args, platform = caller
    platform.name = platform.device_name = platform_name
    args[0] = torch.empty(tokens, 8, dtype=torch.bfloat16)
    args[1] = torch.ones_like(args[0])
    args[4] = torch.ones(tokens, 2)
    args[5] = torch.zeros(tokens, 2, dtype=torch.int32)
    calls = []
    library(monkeypatch, calls)
    apply(obj, *args)
    assert len(calls) == 1 and args[0].shape == (tokens, 8)
    assert torch.all(args[0] == 3)


@pytest.mark.parametrize("platform_name", ["metax", "nvidia"])
def test_strided_output_copy_and_workspace_canary(caller, monkeypatch, platform_name):
    obj, apply, args, platform = caller
    platform.name = platform.device_name = platform_name
    backing = torch.full((2, 16), 7.0, dtype=torch.bfloat16)
    args[0] = backing[:, ::2]
    args[11].fill_(11)
    args[12].fill_(12)
    calls = []
    library(monkeypatch, calls)
    apply(obj, *args)
    assert len(calls) == 1
    assert torch.all(backing[:, ::2] == 3) and torch.all(backing[:, 1::2] == 7)
    assert torch.all(args[11] == 11) and torch.all(args[12] == 12)


@pytest.mark.parametrize("platform_name", ["ascend", "rocm", "cpu", "hygon", "musa"])
def test_other_platforms_keep_staged_path(caller, monkeypatch, platform_name):
    obj, apply, args, platform = caller
    platform.name = platform.device_name = platform_name
    monkeypatch.setitem(sys.modules, "flaggems_vllm", None)
    with pytest.raises(StagedFallback):
        apply(obj, *args)


def test_production_imports_and_explicit_platform_scope():
    root = Path(__file__).resolve().parents[3]
    for rel in [
        "vllm_fl/ops/fused_moe/fused_moe_utils.py",
        "vllm_fl/dispatch/backends/flaggems/impl/fused_moe.py",
    ]:
        source = (root / rel).read_text()
        assert "runtime.backend._metax" not in source
        assert "supports_fused_experts_workspace" not in source
        assert "from flag_gems import" not in source
    utils = (root / "vllm_fl/ops/fused_moe/fused_moe_utils.py").read_text()
    assert 'elif current_platform.device_name in ("nvidia", "metax"):' in utils
    helper = (root / "vllm_fl/dispatch/backends/flaggems/impl/fused_moe.py").read_text()
    assert "metax" not in helper

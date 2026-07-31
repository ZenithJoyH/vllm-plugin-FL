#!/usr/bin/env python3
"""Non-invasive layer output dumper for any model in vLLM.

Captures intermediate tensor outputs from each layer during inference
using PyTorch forward hooks. No vLLM source code modification required.

Automatically detects model structure (embedding, decoder layers, norm,
lm_head) regardless of model architecture (Llama, Qwen, DeepSeek, etc.).

Usage:
    python tools/dump_layer_outputs.py \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --backend cuda \
        --prompt "Hello, world!" \
        --max-tokens 5 \
        --output-dir ./layer_dumps

    # Fine-grained dump (includes attn/mlp/norm sub-layers):
    python tools/dump_layer_outputs.py \
        --model meta-llama/Llama-3.2-1B-Instruct \
        --backend flaggems \
        --dump-mode fine \
        --layers 0,1,2 \
        --output-dir ./layer_dumps
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn


# ── Model structure detection ──

# Common names for embedding modules
_EMBED_NAMES = {"embed_tokens", "embed", "wte", "word_embeddings",
                "token_embedding", "embedding"}

# Common names for final normalization
_NORM_NAMES = {"norm", "ln_f", "final_layernorm", "final_layer_norm",
               "model_norm"}

# Common names for output head
_HEAD_NAMES = {"lm_head", "output", "head"}

# Keywords in class names that indicate a decoder/encoder layer
_LAYER_CLASS_KEYWORDS = {"decoderlayer", "encoderlayer", "transformerblock",
                         "block", "layer"}

# Common names for the module list containing decoder layers
_LAYERS_ATTR_NAMES = {"layers", "blocks", "h", "layer"}

# Sub-module names for fine-grained hooks
_ATTN_NAMES = {"self_attn", "attention", "attn", "self_attention"}
_MLP_NAMES = {"mlp", "feed_forward", "ffn", "dense"}
_NORM_SUB_NAMES = {"input_layernorm", "post_attention_layernorm",
                   "pre_norm", "post_norm", "ln_1", "ln_2",
                   "input_norm", "post_norm"}


def _find_inner_model(model: nn.Module) -> nn.Module:
    """Navigate to the inner model (e.g., LlamaForCausalLM.model -> LlamaModel).

    Many CausalLM wrappers have a .model attribute pointing to the actual
    transformer body. We try common attribute names.
    """
    for attr in ("model", "transformer", "gpt", "backbone"):
        inner = getattr(model, attr, None)
        if inner is not None and isinstance(inner, nn.Module):
            return inner
    return model


def _find_decoder_layers(model: nn.Module) -> Optional[Tuple[str, nn.ModuleList]]:
    """Find the ModuleList containing decoder/encoder layers.

    Returns (attribute_name, module_list) or None.
    """
    for name in _LAYERS_ATTR_NAMES:
        layers = getattr(model, name, None)
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            return name, layers

    # Fallback: search named children for any ModuleList with "layer"-like elements
    for name, child in model.named_children():
        if isinstance(child, nn.ModuleList) and len(child) > 0:
            cls_name = type(child[0]).__name__.lower()
            if any(kw in cls_name for kw in _LAYER_CLASS_KEYWORDS):
                return name, child

    return None


def _find_named_module(model: nn.Module, candidates: Set[str]) -> Optional[Tuple[str, nn.Module]]:
    """Find a direct child module whose name matches one of the candidates."""
    for name, child in model.named_children():
        if name in candidates and not isinstance(child, nn.Identity):
            return name, child
    return None


def _find_sub_modules(layer: nn.Module, candidates: Set[str]) -> List[Tuple[str, nn.Module]]:
    """Find direct children of a layer whose names match candidates."""
    found = []
    for name, child in layer.named_children():
        if name in candidates and not isinstance(child, nn.Identity):
            found.append((name, child))
    return found


class HookManager:
    """Manages forward hooks on model layers and captures tensor outputs."""

    def __init__(self, dump_mode: str = "coarse",
                 layers: Optional[List[int]] = None):
        self.dump_mode = dump_mode
        self.target_layers = layers
        self.captured: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)
        self.handles: List[torch.utils.hooks.RemovableHook] = []
        self._call_counts: Dict[str, int] = defaultdict(int)
        self._model_info: Dict[str, str] = {}  # metadata about detected structure

    def _get_step_name(self, module_name: str) -> str:
        """Determine step name (prefill vs decode) based on call count."""
        self._call_counts[module_name] += 1
        count = self._call_counts[module_name]
        if count == 1:
            return "prefill"
        return f"decode_step_{count - 1:03d}"

    def _make_hook(self, layer_name: str, tensor_names: List[str]):
        """Create a forward hook closure for the given layer."""

        def hook_fn(module, input, output):
            step = self._get_step_name(layer_name)
            key_prefix = f"{step}/{layer_name}"

            if isinstance(output, tuple):
                for i, name in enumerate(tensor_names):
                    if i < len(output) and output[i] is not None:
                        if isinstance(output[i], torch.Tensor):
                            self.captured[key_prefix][name] = (
                                output[i].detach().clone().cpu()
                            )
                # If tuple has more elements than named, capture extras
                if len(output) > len(tensor_names):
                    for i in range(len(tensor_names), len(output)):
                        if output[i] is not None and isinstance(output[i], torch.Tensor):
                            self.captured[key_prefix][f"output_{i}"] = (
                                output[i].detach().clone().cpu()
                            )
            elif isinstance(output, torch.Tensor):
                name = tensor_names[0] if tensor_names else "output"
                self.captured[key_prefix][name] = (
                    output.detach().clone().cpu()
                )

        return hook_fn

    def register_hooks(self, model: nn.Module):
        """Register forward hooks on target layers of the model.

        Automatically detects model structure regardless of architecture.
        """
        outer_model = model
        inner_model = _find_inner_model(model)

        # ── Embedding ──
        embed_result = _find_named_module(inner_model, _EMBED_NAMES)
        if embed_result is None:
            embed_result = _find_named_module(outer_model, _EMBED_NAMES)
        if embed_result is not None:
            embed_name, embed_mod = embed_result
            h = embed_mod.register_forward_hook(
                self._make_hook("embed", ["output"])
            )
            self.handles.append(h)
            self._model_info["embed"] = f"{embed_name} ({type(embed_mod).__name__})"

        # ── Decoder Layers ──
        layers_result = _find_decoder_layers(inner_model)
        if layers_result is None:
            layers_result = _find_decoder_layers(outer_model)

        if layers_result is not None:
            layers_attr, layer_list = layers_result
            self._model_info["layers"] = (
                f"{layers_attr} ({len(layer_list)} x {type(layer_list[0]).__name__})"
            )

            for idx, layer in enumerate(layer_list):
                if layer is None:
                    continue
                if self.target_layers is not None and idx not in self.target_layers:
                    continue

                layer_name = f"layer_{idx:03d}"

                # Main layer hook
                h = layer.register_forward_hook(
                    self._make_hook(layer_name, ["hidden_states", "residual"])
                )
                self.handles.append(h)

                # Fine-grained sub-module hooks
                if self.dump_mode == "fine":
                    # Attention
                    for sub_name, sub_mod in _find_sub_modules(layer, _ATTN_NAMES):
                        h = sub_mod.register_forward_hook(
                            self._make_hook(
                                f"{layer_name}/{sub_name}", ["attn_output"]
                            )
                        )
                        self.handles.append(h)

                        # Rotary embedding if present
                        rotary = getattr(sub_mod, "rotary_emb", None)
                        if rotary is not None and isinstance(rotary, nn.Module):
                            h = rotary.register_forward_hook(
                                self._make_hook(
                                    f"{layer_name}/{sub_name}/rotary_emb",
                                    ["cos", "sin"],
                                )
                            )
                            self.handles.append(h)

                    # MLP
                    for sub_name, sub_mod in _find_sub_modules(layer, _MLP_NAMES):
                        h = sub_mod.register_forward_hook(
                            self._make_hook(
                                f"{layer_name}/{sub_name}", ["mlp_output"]
                            )
                        )
                        self.handles.append(h)

                    # Layer norms
                    for sub_name, sub_mod in _find_sub_modules(layer, _NORM_SUB_NAMES):
                        h = sub_mod.register_forward_hook(
                            self._make_hook(
                                f"{layer_name}/{sub_name}", ["norm_output"]
                            )
                        )
                        self.handles.append(h)
        else:
            print("WARNING: Could not detect decoder layers in model structure.")

        # ── Final Norm ──
        norm_result = _find_named_module(inner_model, _NORM_NAMES)
        if norm_result is not None:
            norm_name, norm_mod = norm_result
            h = norm_mod.register_forward_hook(
                self._make_hook("final_norm", ["output"])
            )
            self.handles.append(h)
            self._model_info["norm"] = f"{norm_name} ({type(norm_mod).__name__})"

        # ── LM Head ──
        head_result = _find_named_module(outer_model, _HEAD_NAMES)
        if head_result is not None:
            head_name, head_mod = head_result
            h = head_mod.register_forward_hook(
                self._make_hook("lm_head", ["logits"])
            )
            self.handles.append(h)
            self._model_info["lm_head"] = f"{head_name} ({type(head_mod).__name__})"

        print(f"Registered {len(self.handles)} hooks (mode={self.dump_mode})")
        if self._model_info:
            print(f"Detected structure: {json.dumps(self._model_info, indent=2)}")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def save(self, output_dir: Path, backend: str) -> dict:
        """Save captured tensors to disk and return manifest entries."""
        steps_manifest = {}

        for key_prefix, tensors in sorted(self.captured.items()):
            parts = key_prefix.split("/", 1)
            step_name = parts[0]
            layer_path = parts[1] if len(parts) > 1 else ""

            dir_path = output_dir / backend / step_name / layer_path
            dir_path.mkdir(parents=True, exist_ok=True)

            if step_name not in steps_manifest:
                steps_manifest[step_name] = {}

            layer_manifest = {}
            for tensor_name, tensor in tensors.items():
                file_name = f"{tensor_name}.pt"
                file_path = dir_path / file_name
                torch.save(tensor, file_path)

                rel_path = str(
                    Path(backend) / step_name / layer_path / file_name
                )
                layer_manifest[tensor_name] = rel_path

            # Nest into steps_manifest
            if layer_path:
                if "layers" not in steps_manifest[step_name]:
                    steps_manifest[step_name]["layers"] = {}
                steps_manifest[step_name]["layers"][layer_path] = layer_manifest
            else:
                steps_manifest[step_name].update(layer_manifest)

        return steps_manifest


def register_hooks_on_model(llm, hook_manager: HookManager):
    """Register hooks via LLM.apply_model() — works for all executor types.

    Because apply_model serializes the closure to a worker process,
    we attach the hook_manager to the model itself so we can retrieve
    captured data later with a second apply_model call.
    """
    dump_mode = hook_manager.dump_mode
    target_layers = hook_manager.target_layers

    def _register(model: torch.nn.Module):
        mgr = HookManager(dump_mode=dump_mode, layers=target_layers)
        mgr.register_hooks(model)
        model._dump_hook_manager = mgr  # noqa: SLF001

    llm.apply_model(_register)


def retrieve_captured_data(llm) -> Dict[str, Dict[str, torch.Tensor]]:
    """Retrieve captured tensor data from the worker process."""

    def _retrieve(model: torch.nn.Module):
        mgr = getattr(model, "_dump_hook_manager", None)
        if mgr is None:
            return {}
        return dict(mgr.captured)

    results = llm.apply_model(_retrieve)
    if results:
        return results[0]
    return {}


def retrieve_model_info(llm) -> Dict[str, str]:
    """Retrieve detected model structure info from the worker process."""

    def _retrieve(model: torch.nn.Module):
        mgr = getattr(model, "_dump_hook_manager", None)
        if mgr is None:
            return {}
        return dict(mgr._model_info)

    results = llm.apply_model(_retrieve)
    if results:
        return results[0]
    return {}


def cleanup_hooks_on_model(llm):
    """Remove hooks from the worker model."""

    def _cleanup(model: torch.nn.Module):
        mgr = getattr(model, "_dump_hook_manager", None)
        if mgr is not None:
            mgr.remove_hooks()
            del model._dump_hook_manager

    llm.apply_model(_cleanup)


def parse_layers_arg(layers_str: str) -> List[int]:
    """Parse layer specification string into list of indices.

    Supports:
        - Single indices: "0,1,2"
        - Ranges: "0-5"
        - Mixed: "0,2-4,7"
    """
    indices = []
    for part in layers_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            indices.extend(range(int(start.strip()), int(end.strip()) + 1))
        else:
            indices.append(int(part))
    return sorted(set(indices))


def main():
    parser = argparse.ArgumentParser(
        description="Dump intermediate layer outputs from any model in vLLM"
    )
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument(
        "--backend", required=True,
        help="Backend identifier for directory naming (e.g., cuda, flaggems, metax)"
    )
    parser.add_argument("--prompt", default="Hello, world!", help="Input prompt")
    parser.add_argument(
        "--max-tokens", type=int, default=5, help="Max tokens to generate"
    )
    parser.add_argument(
        "--output-dir", default="./layer_dumps", help="Output root directory"
    )
    parser.add_argument(
        "--dump-mode",
        choices=["coarse", "fine"],
        default="coarse",
        help="coarse: decoder layer level; fine: include attn/mlp/norm sublayers",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default=None,
        help="Layer indices to dump: '0,1,2' or '0-5' or '0,2-4,7' (default: all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max-model-len", type=int, default=None, help="Max model context length"
    )
    parser.add_argument(
        "--tp-size", type=int, default=1, help="Tensor parallel size"
    )
    parser.add_argument(
        "--dtype", default="auto", help="Model dtype (auto, float16, bfloat16)"
    )
    parser.add_argument(
        "--trust-remote-code", action="store_true",
        help="Trust remote code from HuggingFace"
    )

    args = parser.parse_args()

    target_layers = None
    if args.layers:
        target_layers = parse_layers_arg(args.layers)

    # Allow pickle serialization for apply_model to pass closures
    os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    # Lazy import vLLM to allow --help without GPU
    from vllm import LLM, SamplingParams

    print(f"Initializing vLLM with model={args.model}, enforce_eager=True")
    print(f"Backend: {args.backend}, Dump mode: {args.dump_mode}")
    if target_layers:
        print(f"Target layers: {target_layers}")

    llm_kwargs = dict(
        model=args.model,
        enforce_eager=True,
        enable_chunked_prefill=False,
        seed=args.seed,
        tensor_parallel_size=args.tp_size,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
    )
    if args.max_model_len:
        llm_kwargs["max_model_len"] = args.max_model_len

    llm = LLM(**llm_kwargs)

    # Register hooks via public apply_model API
    hook_manager = HookManager(dump_mode=args.dump_mode, layers=target_layers)
    register_hooks_on_model(llm, hook_manager)

    # Run inference
    sampling_params = SamplingParams(
        max_tokens=args.max_tokens, temperature=0.0
    )
    print(f"\nRunning inference: prompt={args.prompt!r}, max_tokens={args.max_tokens}")
    outputs = llm.generate([args.prompt], sampling_params)

    generated_text = outputs[0].outputs[0].text if outputs else ""
    print(f"Generated: {generated_text!r}")

    # Retrieve captured data from worker process
    captured_data = retrieve_captured_data(llm)
    model_info = retrieve_model_info(llm)
    hook_manager.captured = captured_data
    print(f"\nRetrieved {len(captured_data)} layer/step combinations from worker")

    # Save tensors
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    steps_manifest = hook_manager.save(output_dir, args.backend)

    # Write manifest
    manifest = {
        "model": args.model,
        "backend": args.backend,
        "prompt": args.prompt,
        "generated_text": generated_text,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "dtype": args.dtype,
        "tp_size": args.tp_size,
        "dump_mode": args.dump_mode,
        "layers_dumped": target_layers or "all",
        "model_structure": model_info,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "steps": steps_manifest,
    }

    manifest_path = output_dir / f"{args.backend}_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nSaved manifest to {manifest_path}")
    print(
        f"Dumped {len(hook_manager.captured)} layer/step combinations "
        f"to {output_dir / args.backend}/"
    )

    # Print summary of what was captured
    steps = set()
    layers = set()
    for key in hook_manager.captured:
        parts = key.split("/", 1)
        steps.add(parts[0])
        if len(parts) > 1:
            layers.add(parts[1])
    print(f"\nSteps captured: {sorted(steps)}")
    print(f"Layers captured: {sorted(layers)[:20]}"
          f"{'...' if len(layers) > 20 else ''}")

    # Cleanup hooks in worker
    cleanup_hooks_on_model(llm)


if __name__ == "__main__":
    main()

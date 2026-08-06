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


# Common attributes that wrap the actual transformer body. Ordered so that
# classic single-level wrappers (ForCausalLM.model) resolve first, while
# multimodal wrappers (Qwen3-VL style: ForConditionalGeneration.language_model
# -> ForCausalLM.model) still get unwrapped.
_INNER_MODEL_ATTRS = ("model", "transformer", "gpt", "backbone",
                      "language_model", "text_model", "decoder", "llm")


def _find_inner_model(model: nn.Module) -> nn.Module:
    """Navigate to the innermost transformer body.

    Classic case: LlamaForCausalLM.model -> LlamaModel.
    Multimodal case (Qwen3.5 MoE): Qwen3_5MoeForConditionalGeneration
    -> language_model (Qwen3_5MoeForCausalLM) -> model (Qwen3_5MoeModel).
    We drill down iteratively until no more known wrapper attributes are found.
    """
    current = model
    for _ in range(8):
        inner = None
        for attr in _INNER_MODEL_ATTRS:
            candidate = getattr(current, attr, None)
            if candidate is not None and isinstance(candidate, nn.Module):
                inner = candidate
                break
        if inner is None or inner is current:
            break
        current = inner
    return current


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


def _find_named_module_deep(
    model: nn.Module,
    candidates: Set[str],
    skip_prefixes: Tuple[str, ...] = ("visual", "vision", "tower"),
) -> Optional[Tuple[str, nn.Module]]:
    """Find a module anywhere in the tree whose leaf name matches the candidates.

    Used for modules that sit on a wrapper level (e.g. lm_head on
    ForCausalLM, which is *above* the drilled-down inner model). Sub-trees
    like the vision tower are skipped so their heads/embeddings don't shadow
    the language model's.
    """
    for name, child in model.named_modules():
        if isinstance(child, nn.Identity):
            continue
        if any(name == p or name.startswith(p + ".") for p in skip_prefixes):
            continue
        if name.split(".")[-1] in candidates:
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
                # Skip PPMissingLayer placeholders in pipeline parallel models
                if type(layer).__name__ in ("PPMissingLayer", "StageMissingLayer"):
                    continue
                if isinstance(layer, nn.Identity):
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
        if head_result is None:
            # lm_head often sits on a wrapper level above the drilled-down
            # inner model (e.g. Qwen3.5: language_model.lm_head).
            head_result = _find_named_module_deep(outer_model, _HEAD_NAMES)
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

    def save(self, output_dir: Path, backend: str, tp_size: int, pp_size: int) -> dict:
        """Save captured tensors to disk and return manifest entries.

        Args:
            output_dir: Root output directory
            backend: Backend identifier (cuda, flaggems, etc.)
            tp_size: Tensor parallel size
            pp_size: Pipeline parallel size

        Note: self.captured should be in the new format:
            {layer_name: [{tp_rank, pp_rank, tensors}, ...]}
        """
        steps_manifest = {}

        for layer_name, rank_data_list in sorted(self.captured.items()):
            parts = layer_name.split("/", 1)
            step_name = parts[0]
            layer_path = parts[1] if len(parts) > 1 else ""

            if step_name not in steps_manifest:
                steps_manifest[step_name] = {}

            # Save each rank's data to a separate file
            for rank_data in rank_data_list:
                tp_rank = rank_data["tp_rank"]
                pp_rank = rank_data["pp_rank"]
                tensors = rank_data["tensors"]

                # File naming: layer_path.pp{pp_rank}.tp{tp_rank}.pt
                if layer_path:
                    base_name = layer_path.replace("/", "_")
                else:
                    base_name = "root"

                file_name = f"{base_name}.pp{pp_rank}.tp{tp_rank}.pt"
                dir_path = output_dir / backend / step_name
                dir_path.mkdir(parents=True, exist_ok=True)

                file_path = dir_path / file_name

                # Save tensor data with rank info
                torch.save({
                    "tp_rank": tp_rank,
                    "pp_rank": pp_rank,
                    "tensors": tensors,
                }, file_path)

                # Build relative path for manifest
                rel_path = str(Path(backend) / step_name / file_name)

                # Add to manifest
                if "layers" not in steps_manifest[step_name]:
                    steps_manifest[step_name]["layers"] = {}
                if layer_path not in steps_manifest[step_name]["layers"]:
                    steps_manifest[step_name]["layers"][layer_path] = []

                steps_manifest[step_name]["layers"][layer_path].append({
                    "tp_rank": tp_rank,
                    "pp_rank": pp_rank,
                    "file": rel_path
                })

        return steps_manifest


class _RegisterHooks:
    """Picklable callable for registering hooks on worker models."""

    def __init__(self, dump_mode: str, target_layers: Optional[List[int]]):
        self.dump_mode = dump_mode
        self.target_layers = target_layers

    def __call__(self, model: torch.nn.Module):
        from vllm.distributed.parallel_state import (
            get_pp_group,
            get_tensor_model_parallel_rank,
        )

        tp_rank = get_tensor_model_parallel_rank()
        pp_group = get_pp_group()
        pp_rank = pp_group.rank_in_group if pp_group is not None else 0

        mgr = HookManager(dump_mode=self.dump_mode, layers=self.target_layers)
        mgr.register_hooks(model)
        mgr.tp_rank = tp_rank
        mgr.pp_rank = pp_rank
        model._dump_hook_manager = mgr  # noqa: SLF001


def register_hooks_on_model(llm, hook_manager: HookManager):
    """Register hooks via LLM.apply_model() — works for all executor types.

    Because apply_model serializes the callable to a worker process,
    we use a module-level callable class (picklable) instead of a closure.
    We attach the hook_manager to the model itself so we can retrieve
    captured data later with a second apply_model call.
    """
    llm.apply_model(_RegisterHooks(hook_manager.dump_mode, hook_manager.target_layers))


class _RetrieveCaptured:
    """Picklable callable to retrieve captured tensor data from worker models.

    Must be a class (not a plain function): vLLM serializes apply_model()
    callables twice — cloudpickle on the main->EngineCore hop, then plain
    pickle on the EngineCore->workers hop. A plain function is copied *by
    value* in the first hop, so the second hop can't match it against
    __main__ by name and fails with:
        Can't pickle <function ...>: it's not the same object as __main__.xxx
    Instances of a __main__ class serialize by class reference, which
    resolves in every process (same pattern as _RegisterHooks).
    """

    def __call__(self, model: torch.nn.Module):
        mgr = getattr(model, "_dump_hook_manager", None)
        if mgr is None:
            return {"tp_rank": -1, "pp_rank": -1, "captured": {}}

        return {
            "tp_rank": mgr.tp_rank,
            "pp_rank": mgr.pp_rank,
            "captured": dict(mgr.captured)
        }


def retrieve_captured_data(llm) -> Dict[str, List[Dict]]:
    """Retrieve captured tensor data from all workers.

    Returns:
        Dict mapping layer names to lists of rank data:
        {
            "step/layer_name": [
                {"tp_rank": 0, "pp_rank": 0, "tensors": {"input": ..., "output": ...}},
                {"tp_rank": 1, "pp_rank": 0, "tensors": {"input": ..., "output": ...}},
                ...
            ]
        }
    """
    results = llm.apply_model(_RetrieveCaptured())

    # Reorganize: {layer_name: [{tp_rank, pp_rank, tensors}, ...]}
    merged = {}
    for r in results:
        if r["tp_rank"] == -1:  # Skip invalid workers
            continue

        for layer_name, tensors in r["captured"].items():
            if layer_name not in merged:
                merged[layer_name] = []

            merged[layer_name].append({
                "tp_rank": r["tp_rank"],
                "pp_rank": r["pp_rank"],
                "tensors": tensors
            })

    # Sort by (pp_rank, tp_rank) for consistent ordering
    for layer_name in merged:
        merged[layer_name].sort(key=lambda x: (x["pp_rank"], x["tp_rank"]))

    return merged


class _RetrieveInfo:
    """Picklable callable to retrieve model structure info from workers.

    See _RetrieveCaptured for why this is a class rather than a function.
    """

    def __call__(self, model: torch.nn.Module):
        mgr = getattr(model, "_dump_hook_manager", None)
        if mgr is None:
            return {"tp_rank": -1, "pp_rank": -1, "info": {}}

        return {
            "tp_rank": mgr.tp_rank,
            "pp_rank": mgr.pp_rank,
            "info": dict(mgr._model_info)
        }


def retrieve_model_info(llm) -> Dict[str, List[Dict]]:
    """Retrieve detected model structure info from all workers.

    Returns:
        Dict mapping info keys to lists of rank data:
        {
            "info_key": [
                {"tp_rank": 0, "pp_rank": 0, "value": "..."},
                {"tp_rank": 1, "pp_rank": 0, "value": "..."},
                ...
            ]
        }
    """
    results = llm.apply_model(_RetrieveInfo())

    # Reorganize: {info_key: [{tp_rank, pp_rank, value}, ...]}
    merged = {}
    for r in results:
        if r["tp_rank"] == -1:
            continue

        for key, value in r["info"].items():
            if key not in merged:
                merged[key] = []
            merged[key].append({
                "tp_rank": r["tp_rank"],
                "pp_rank": r["pp_rank"],
                "value": value
            })

    return merged


class _Cleanup:
    """Picklable callable to remove hooks from worker models.

    See _RetrieveCaptured for why this is a class rather than a function.
    """

    def __call__(self, model: torch.nn.Module):
        mgr = getattr(model, "_dump_hook_manager", None)
        if mgr is not None:
            mgr.remove_hooks()
            del model._dump_hook_manager


def cleanup_hooks_on_model(llm):
    """Remove hooks from the worker model."""
    llm.apply_model(_Cleanup())


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
        "--pp-size", type=int, default=1, help="Pipeline parallel size"
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.9,
        help="GPU memory utilization ratio (0.0 ~ 1.0, default: 0.9)"
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
        pipeline_parallel_size=args.pp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
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

    # Count total tensors across all ranks
    total_rank_data = sum(len(rank_list) for rank_list in captured_data.values())
    print(f"\nRetrieved {len(captured_data)} layer/step combinations from {total_rank_data} rank instances")

    # Save tensors
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    steps_manifest = hook_manager.save(output_dir, args.backend, args.tp_size, args.pp_size)

    # Flatten model_info for manifest (take first rank's values)
    model_info_flat = {}
    for key, rank_list in model_info.items():
        if rank_list:
            model_info_flat[key] = rank_list[0]["value"]

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
        "pp_size": args.pp_size,
        "dump_mode": args.dump_mode,
        "layers_dumped": target_layers or "all",
        "model_structure": model_info_flat,
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

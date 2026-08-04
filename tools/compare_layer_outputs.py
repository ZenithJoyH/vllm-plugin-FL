#!/usr/bin/env python3
"""Compare layer outputs between two backend dumps.

Loads manifest files from two dump runs and computes per-tensor diff metrics.
Useful for identifying which layer introduces precision divergence between
backends (e.g., CUDA vs FlagGems, CUDA vs NPU).

Supports TP/PP parallel configurations - compares tensors rank by rank.

Usage:
    python tools/compare_layer_outputs.py \
        --baseline ./layer_dumps/cuda_manifest.json \
        --target ./layer_dumps/flaggems_manifest.json \
        --output ./comparison_report.json

    # Compare only prefill step:
    python tools/compare_layer_outputs.py \
        --baseline ./layer_dumps/cuda_manifest.json \
        --target ./layer_dumps/flaggems_manifest.json \
        --steps prefill

    # Stricter tolerance:
    python tools/compare_layer_outputs.py \
        --baseline ./layer_dumps/cuda_manifest.json \
        --target ./layer_dumps/flaggems_manifest.json \
        --atol 1e-5 --rtol 1e-5
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


def compute_metrics(
    baseline: torch.Tensor, target: torch.Tensor, atol: float, rtol: float
) -> dict:
    """Compute comparison metrics between two tensors."""
    if baseline.shape != target.shape:
        return {
            "shape_match": False,
            "baseline_shape": list(baseline.shape),
            "target_shape": list(target.shape),
            "pass": False,
        }

    # Cast to float32 for stable numeric comparison
    b = baseline.float()
    t = target.float()

    diff = (b - t).abs()
    max_abs_diff = diff.max().item()
    mean_abs_diff = diff.mean().item()

    # Relative difference (avoid division by zero)
    denom = torch.clamp(b.abs(), min=1e-12)
    max_rel_diff = (diff / denom).max().item()
    mean_rel_diff = (diff / denom).mean().item()

    # Cosine similarity (flatten to 1D)
    b_flat = b.flatten()
    t_flat = t.flatten()
    if b_flat.numel() > 0:
        cos_sim = torch.nn.functional.cosine_similarity(
            b_flat.unsqueeze(0), t_flat.unsqueeze(0)
        ).item()
    else:
        cos_sim = 1.0

    allclose = torch.allclose(b, t, atol=atol, rtol=rtol)

    return {
        "shape_match": True,
        "shape": list(baseline.shape),
        "dtype": str(baseline.dtype),
        "numel": baseline.numel(),
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
        "max_rel_diff": max_rel_diff,
        "mean_rel_diff": mean_rel_diff,
        "cosine_sim": cos_sim,
        "allclose": allclose,
        "pass": allclose,
    }


def load_layer_data_from_manifest(
    manifest: dict, manifest_path: Path, steps_filter: str
) -> Dict[str, List[Dict]]:
    """Load layer data from new format manifest.

    Returns:
        {
            "step_name/layer_name": [
                {"tp_rank": 0, "pp_rank": 0, "tensors": {...}},
                {"tp_rank": 1, "pp_rank": 0, "tensors": {...}},
                ...
            ]
        }
    """
    layer_data = {}
    steps = manifest.get("steps", {})

    for step_name in sorted(steps.keys()):
        if steps_filter == "prefill" and step_name != "prefill":
            continue
        if steps_filter == "decode" and step_name == "prefill":
            continue

        step_data = steps[step_name]
        layers = step_data.get("layers", {})

        for layer_name, rank_list in layers.items():
            if not isinstance(rank_list, list):
                continue  # Old format, skip

            full_key = f"{step_name}/{layer_name}"
            layer_data[full_key] = []

            for rank_entry in rank_list:
                file_path = manifest_path.parent / rank_entry["file"]
                if not file_path.exists():
                    print(f"WARNING: File not found: {file_path}")
                    continue

                data = torch.load(file_path, map_location="cpu", weights_only=True)
                layer_data[full_key].append({
                    "tp_rank": data["tp_rank"],
                    "pp_rank": data["pp_rank"],
                    "tensors": data["tensors"]
                })

    # Sort by (pp_rank, tp_rank)
    for key in layer_data:
        layer_data[key].sort(key=lambda x: (x["pp_rank"], x["tp_rank"]))

    return layer_data


def validate_parallel_config(baseline_manifest: dict, target_manifest: dict):
    """Validate that parallel configurations match between baseline and target."""
    b_tp = baseline_manifest.get("tp_size", 1)
    t_tp = target_manifest.get("tp_size", 1)
    b_pp = baseline_manifest.get("pp_size", 1)
    t_pp = target_manifest.get("pp_size", 1)

    if b_tp != t_tp:
        print(f"ERROR: tp_size mismatch! baseline={b_tp}, target={t_tp}")
        sys.exit(1)
    if b_pp != t_pp:
        print(f"ERROR: pp_size mismatch! baseline={b_pp}, target={t_pp}")
        sys.exit(1)

    return b_tp, b_pp


def compare_rank_tensors(
    baseline_tensors: dict,
    target_tensors: dict,
    atol: float,
    rtol: float
) -> Dict[str, dict]:
    """Compare tensors from a single rank between baseline and target.

    Returns:
        {tensor_name: metrics_dict}
    """
    results = {}

    all_tensor_names = set(baseline_tensors.keys()) | set(target_tensors.keys())

    for tensor_name in all_tensor_names:
        if tensor_name not in baseline_tensors:
            results[tensor_name] = {
                "error": "missing_in_baseline",
                "pass": False
            }
            continue
        if tensor_name not in target_tensors:
            results[tensor_name] = {
                "error": "missing_in_target",
                "pass": False
            }
            continue

        metrics = compute_metrics(
            baseline_tensors[tensor_name],
            target_tensors[tensor_name],
            atol,
            rtol
        )
        results[tensor_name] = metrics

    return results


def resolve_tensor_path(manifest_path: Path, rel_path: str) -> Path:
    """Resolve a relative tensor path against the manifest's directory."""
    return manifest_path.parent / rel_path


def collect_tensor_pairs(
    baseline_manifest: dict,
    target_manifest: dict,
    baseline_root: Path,
    target_root: Path,
    steps_filter: str,
) -> Tuple[Dict[str, List[Dict]], Dict[str, List[Dict]]]:
    """Collect layer data from both manifests using new format.

    Returns:
        (baseline_layer_data, target_layer_data)
        where each is {layer_key: [{tp_rank, pp_rank, tensors}, ...]}
    """
    baseline_data = load_layer_data_from_manifest(
        baseline_manifest, baseline_root, steps_filter
    )
    target_data = load_layer_data_from_manifest(
        target_manifest, target_root, steps_filter
    )

    return baseline_data, target_data


def print_table(results: List[dict], show_all: bool = False):
    """Print a compact terminal table of results."""
    header = (
        f"{'Step/Layer':<40} {'Rank':<12} "
        f"{'Tensor':<20} {'MaxAbsDiff':>11} {'MeanAbsDiff':>12} "
        f"{'CosSim':>8} {'Status':>6}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for r in results:
        if not show_all and r["metrics"].get("pass", False):
            continue

        rank_str = f"pp{r['pp_rank']}.tp{r['tp_rank']}"

        if not r["metrics"].get("shape_match", True):
            status = "\033[33mSHAPE!\033[0m"
            print(
                f"{r['layer']:<40} {rank_str:<12} "
                f"{r['tensor']:<20} {'N/A':>11} {'N/A':>12} "
                f"{'N/A':>8} {status:>6}"
            )
        else:
            m = r["metrics"]
            status = "\033[32mPASS\033[0m" if m["pass"] else "\033[31mFAIL\033[0m"
            print(
                f"{r['layer']:<40} {rank_str:<12} "
                f"{r['tensor']:<20} {m['max_abs_diff']:>11.6f} {m['mean_abs_diff']:>12.8f} "
                f"{m['cosine_sim']:>8.6f} {status}"
            )

    print(sep)

    if not show_all:
        pass_count = sum(1 for r in results if r["metrics"].get("pass", False))
        if pass_count > 0:
            print(f"  ({pass_count} PASS results hidden, use --show-all to display)")


def find_first_divergence(results: List[dict]) -> Optional[dict]:
    """Find the first layer/rank where divergence exceeds tolerance."""
    for r in results:
        if not r["metrics"].get("pass", False):
            return r
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Compare layer outputs between two backend dumps"
    )
    parser.add_argument(
        "--baseline", required=True, help="Path to baseline manifest.json"
    )
    parser.add_argument(
        "--target", required=True, help="Path to target manifest.json"
    )
    parser.add_argument("--atol", type=float, default=1e-3,
                        help="Absolute tolerance (default: 1e-3)")
    parser.add_argument("--rtol", type=float, default=1e-3,
                        help="Relative tolerance (default: 1e-3)")
    parser.add_argument("--output", default=None, help="Output JSON report path")
    parser.add_argument(
        "--steps",
        choices=["all", "prefill", "decode"],
        default="all",
        help="Which steps to compare",
    )
    parser.add_argument(
        "--show-all", action="store_true",
        help="Show all results including PASSing layers"
    )

    args = parser.parse_args()

    baseline_path = Path(args.baseline)
    target_path = Path(args.target)

    if not baseline_path.exists():
        print(f"ERROR: Baseline manifest not found: {baseline_path}")
        sys.exit(1)
    if not target_path.exists():
        print(f"ERROR: Target manifest not found: {target_path}")
        sys.exit(1)

    baseline_manifest = load_manifest(baseline_path)
    target_manifest = load_manifest(target_path)

    # Validate parallel configurations
    tp_size, pp_size = validate_parallel_config(baseline_manifest, target_manifest)

    print(
        f"Comparing: {baseline_manifest['backend']} (baseline) "
        f"vs {target_manifest['backend']} (target)"
    )
    print(f"Model: {baseline_manifest['model']}")
    print(f"Parallelism: tp_size={tp_size}, pp_size={pp_size}")
    print(f"Tolerance: atol={args.atol}, rtol={args.rtol}")
    print(f"Baseline prompt: {baseline_manifest.get('prompt', 'N/A')!r}")
    print(f"Target prompt: {target_manifest.get('prompt', 'N/A')!r}")
    print()

    # Warn if prompts differ
    if baseline_manifest.get("prompt") != target_manifest.get("prompt"):
        print("WARNING: Prompts differ between baseline and target!")
        print()

    baseline_data, target_data = collect_tensor_pairs(
        baseline_manifest,
        target_manifest,
        baseline_path,
        target_path,
        args.steps,
    )

    if not baseline_data and not target_data:
        print("No layer data found. Check manifests.")
        sys.exit(1)

    # Find common layers
    common_layers = set(baseline_data.keys()) & set(target_data.keys())
    if not common_layers:
        print("No matching layers found between baseline and target.")
        print(f"  Baseline layers: {list(baseline_data.keys())[:5]}...")
        print(f"  Target layers: {list(target_data.keys())[:5]}...")
        sys.exit(1)

    print(f"Found {len(common_layers)} matching layers to compare.\n")

    results = []
    pass_count = 0
    fail_count = 0

    for layer_key in sorted(common_layers):
        baseline_ranks = baseline_data[layer_key]
        target_ranks = target_data[layer_key]

        # Check rank count match
        if len(baseline_ranks) != len(target_ranks):
            print(f"\nWARNING: [{layer_key}] rank count mismatch!")
            print(f"  baseline: {len(baseline_ranks)} ranks")
            print(f"  target: {len(target_ranks)} ranks")
            continue

        # Compare rank by rank
        for b_rank, t_rank in zip(baseline_ranks, target_ranks):
            if b_rank["tp_rank"] != t_rank["tp_rank"] or b_rank["pp_rank"] != t_rank["pp_rank"]:
                print(f"\nERROR: [{layer_key}] rank mismatch!")
                print(f"  baseline: pp={b_rank['pp_rank']}, tp={b_rank['tp_rank']}")
                print(f"  target: pp={t_rank['pp_rank']}, tp={t_rank['tp_rank']}")
                continue

            # Compare all tensors in this rank
            tensor_results = compare_rank_tensors(
                b_rank["tensors"],
                t_rank["tensors"],
                args.atol,
                args.rtol
            )

            for tensor_name, metrics in tensor_results.items():
                results.append({
                    "layer": layer_key,
                    "tp_rank": b_rank["tp_rank"],
                    "pp_rank": b_rank["pp_rank"],
                    "tensor": tensor_name,
                    "metrics": metrics,
                })

                if metrics.get("pass", False):
                    pass_count += 1
                else:
                    fail_count += 1

    print_table(results, show_all=args.show_all)

    # Summary
    print(f"\nSummary: {pass_count} passed, {fail_count} failed, "
          f"{len(results)} total")

    # First divergence hint
    first_fail = find_first_divergence(results)
    if first_fail:
        print(f"\n{'='*60}")
        print(f"FIRST DIVERGENCE: [{first_fail['layer']}] "
              f"pp{first_fail['pp_rank']}.tp{first_fail['tp_rank']} "
              f"{first_fail['tensor']}")
        m = first_fail["metrics"]
        if m.get("shape_match", True):
            print(f"  Max abs diff: {m['max_abs_diff']:.8f}")
            print(f"  Mean abs diff: {m['mean_abs_diff']:.8f}")
            print(f"  Cosine similarity: {m['cosine_sim']:.8f}")
        else:
            print(f"  Shape mismatch: {m.get('baseline_shape')} vs {m.get('target_shape')}")
        print(f"{'='*60}")

    # Save JSON report
    if args.output:
        report = {
            "baseline": str(baseline_path),
            "target": str(target_path),
            "baseline_backend": baseline_manifest["backend"],
            "target_backend": target_manifest["backend"],
            "model": baseline_manifest["model"],
            "tp_size": tp_size,
            "pp_size": pp_size,
            "atol": args.atol,
            "rtol": args.rtol,
            "summary": {
                "total": len(results),
                "passed": pass_count,
                "failed": fail_count,
            },
            "first_divergence": (
                f"{first_fail['layer']}/pp{first_fail['pp_rank']}.tp{first_fail['tp_rank']}/{first_fail['tensor']}"
                if first_fail else None
            ),
            "results": results,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {output_path}")

    # Exit with non-zero if any failures
    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

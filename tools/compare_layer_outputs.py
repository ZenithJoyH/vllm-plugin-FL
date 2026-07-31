#!/usr/bin/env python3
"""Compare layer outputs between two backend dumps.

Loads manifest files from two dump runs and computes per-tensor diff metrics.
Useful for identifying which layer introduces precision divergence between
backends (e.g., CUDA vs FlagGems, CUDA vs NPU).

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


def load_manifest(path: Path) -> dict:
    """Load a dump manifest JSON file."""
    with open(path) as f:
        return json.load(f)


def resolve_tensor_path(manifest_path: Path, rel_path: str) -> Path:
    """Resolve a relative tensor path against the manifest's directory."""
    return manifest_path.parent / rel_path


def collect_tensor_pairs(
    baseline_manifest: dict,
    target_manifest: dict,
    baseline_root: Path,
    target_root: Path,
    steps_filter: str,
) -> List[Tuple[str, str, Path, Path]]:
    """Collect matching (step, tensor_name, baseline_path, target_path) pairs."""
    pairs = []

    baseline_steps = baseline_manifest.get("steps", {})
    target_steps = target_manifest.get("steps", {})

    for step_name in sorted(baseline_steps.keys()):
        if steps_filter == "prefill" and step_name != "prefill":
            continue
        if steps_filter == "decode" and step_name == "prefill":
            continue

        if step_name not in target_steps:
            continue

        b_step = baseline_steps[step_name]
        t_step = target_steps[step_name]

        # Direct tensors at step level (e.g., embed output)
        for key in b_step:
            if key == "layers":
                continue
            if key in t_step and isinstance(b_step[key], str):
                pairs.append((
                    step_name,
                    key,
                    resolve_tensor_path(baseline_root, b_step[key]),
                    resolve_tensor_path(target_root, t_step[key]),
                ))

        # Layer-level tensors
        b_layers = b_step.get("layers", {})
        t_layers = t_step.get("layers", {})

        for layer_name in sorted(b_layers.keys()):
            if layer_name not in t_layers:
                continue

            for tensor_name in sorted(b_layers[layer_name].keys()):
                if tensor_name not in t_layers[layer_name]:
                    continue

                pairs.append((
                    step_name,
                    f"{layer_name}/{tensor_name}",
                    resolve_tensor_path(
                        baseline_root, b_layers[layer_name][tensor_name]
                    ),
                    resolve_tensor_path(
                        target_root, t_layers[layer_name][tensor_name]
                    ),
                ))

    return pairs


def print_table(results: List[dict], show_all: bool = False):
    """Print a compact terminal table of results."""
    header = (
        f"{'Step':<18} {'Layer/Tensor':<35} "
        f"{'MaxAbsDiff':>11} {'MeanAbsDiff':>12} "
        f"{'CosSim':>8} {'Status':>6}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for r in results:
        if not show_all and r["metrics"].get("pass", False):
            continue

        if not r["metrics"].get("shape_match", True):
            status = "\033[33mSHAPE!\033[0m"
            print(
                f"{r['step']:<18} {r['tensor']:<35} "
                f"{'N/A':>11} {'N/A':>12} "
                f"{'N/A':>8} {status:>6}"
            )
        else:
            m = r["metrics"]
            status = "\033[32mPASS\033[0m" if m["pass"] else "\033[31mFAIL\033[0m"
            print(
                f"{r['step']:<18} {r['tensor']:<35} "
                f"{m['max_abs_diff']:>11.6f} {m['mean_abs_diff']:>12.8f} "
                f"{m['cosine_sim']:>8.6f} {status}"
            )

    print(sep)

    if not show_all:
        pass_count = sum(1 for r in results if r["metrics"].get("pass", False))
        if pass_count > 0:
            print(f"  ({pass_count} PASS results hidden, use --show-all to display)")


def find_first_divergence(results: List[dict]) -> Optional[dict]:
    """Find the first layer where divergence exceeds tolerance."""
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

    print(
        f"Comparing: {baseline_manifest['backend']} (baseline) "
        f"vs {target_manifest['backend']} (target)"
    )
    print(f"Model: {baseline_manifest['model']}")
    print(f"Tolerance: atol={args.atol}, rtol={args.rtol}")
    print(f"Baseline prompt: {baseline_manifest.get('prompt', 'N/A')!r}")
    print(f"Target prompt: {target_manifest.get('prompt', 'N/A')!r}")
    print()

    # Warn if prompts differ
    if baseline_manifest.get("prompt") != target_manifest.get("prompt"):
        print("WARNING: Prompts differ between baseline and target!")
        print()

    pairs = collect_tensor_pairs(
        baseline_manifest,
        target_manifest,
        baseline_path,
        target_path,
        args.steps,
    )

    if not pairs:
        print("No matching tensor pairs found. Check manifests.")
        print(f"  Baseline steps: {list(baseline_manifest.get('steps', {}).keys())}")
        print(f"  Target steps: {list(target_manifest.get('steps', {}).keys())}")
        sys.exit(1)

    print(f"Found {len(pairs)} matching tensor pairs to compare.\n")

    results = []
    pass_count = 0
    fail_count = 0
    missing_count = 0

    for step_name, tensor_name, b_path, t_path in pairs:
        if not b_path.exists():
            print(f"WARNING: baseline tensor not found: {b_path}")
            missing_count += 1
            continue
        if not t_path.exists():
            print(f"WARNING: target tensor not found: {t_path}")
            missing_count += 1
            continue

        b_tensor = torch.load(b_path, map_location="cpu", weights_only=True)
        t_tensor = torch.load(t_path, map_location="cpu", weights_only=True)

        metrics = compute_metrics(b_tensor, t_tensor, args.atol, args.rtol)

        results.append({
            "step": step_name,
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
          f"{len(results)} total"
          f"{f', {missing_count} missing' if missing_count else ''}")

    # First divergence hint
    first_fail = find_first_divergence(results)
    if first_fail:
        print(f"\n{'='*60}")
        print(f"FIRST DIVERGENCE: [{first_fail['step']}] {first_fail['tensor']}")
        m = first_fail["metrics"]
        if m.get("shape_match", True):
            print(f"  Max abs diff: {m['max_abs_diff']:.8f}")
            print(f"  Mean abs diff: {m['mean_abs_diff']:.8f}")
            print(f"  Cosine similarity: {m['cosine_sim']:.8f}")
        else:
            print(f"  Shape mismatch: {m.get('baseline_shape')} vs {m.get('target_shape')}")
        print(f"{'='*60}")
        print("\nTip: Re-run dump with --dump-mode fine --layers "
              f"{first_fail['tensor'].split('/')[0].replace('layer_', '')}"
              " to get sub-layer detail.")

    # Save JSON report
    if args.output:
        report = {
            "baseline": str(baseline_path),
            "target": str(target_path),
            "baseline_backend": baseline_manifest["backend"],
            "target_backend": target_manifest["backend"],
            "model": baseline_manifest["model"],
            "atol": args.atol,
            "rtol": args.rtol,
            "summary": {
                "total": len(results),
                "passed": pass_count,
                "failed": fail_count,
                "missing": missing_count,
            },
            "first_divergence": first_fail["tensor"] if first_fail else None,
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

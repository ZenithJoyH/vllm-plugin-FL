#!/bin/bash
# Compare layer outputs between two backend dumps.
#
# Usage:
#   ./tools/compare_tensors.sh <baseline_manifest> <target_manifest> [extra_args...]
#
# Examples:
#   # Basic comparison:
#   ./tools/compare_tensors.sh \
#       ./layer_dumps/cuda_manifest.json \
#       ./layer_dumps/flaggems_manifest.json
#
#   # With custom tolerance and JSON report:
#   ./tools/compare_tensors.sh \
#       ./layer_dumps/cuda_manifest.json \
#       ./layer_dumps/flaggems_manifest.json \
#       --atol 1e-4 --rtol 1e-4 \
#       --output ./comparison_report.json
#
#   # Compare only prefill step:
#   ./tools/compare_tensors.sh \
#       ./layer_dumps/cuda_manifest.json \
#       ./layer_dumps/flaggems_manifest.json \
#       --steps prefill

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BASELINE="${1:?Usage: $0 <baseline_manifest.json> <target_manifest.json> [extra_args...]}"
TARGET="${2:?Usage: $0 <baseline_manifest.json> <target_manifest.json> [extra_args...]}"
shift 2

python "${SCRIPT_DIR}/compare_layer_outputs.py" \
    --baseline "${BASELINE}" \
    --target "${TARGET}" \
    --output ./comparison_report.json \
    "$@"

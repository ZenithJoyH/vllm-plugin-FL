#!/bin/bash
# Dump layer outputs for precision debugging.
#
# Usage:
#   ./tools/dump_tensors.sh <model_path> <backend> [extra_args...]
#
# Examples:
#   # Basic dump with CUDA backend:
#   ./tools/dump_tensors.sh meta-llama/Llama-3.2-1B-Instruct cuda
#
#   # Dump with FlagGems backend, fine-grained mode:
#   ./tools/dump_tensors.sh meta-llama/Llama-3.2-1B-Instruct flaggems --dump-mode fine
#
#   # Dump specific layers with custom prompt:
#   ./tools/dump_tensors.sh /path/to/model metax --layers 0-5 --prompt "Test input"

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL="${1:?Usage: $0 <model_path> <backend> [extra_args...]}"
BACKEND="${2:?Usage: $0 <model_path> <backend> [extra_args...]}"
shift 2

python "${SCRIPT_DIR}/dump_layer_outputs.py" \
    --model "${MODEL}" \
    --backend "${BACKEND}" \
    --output-dir ./layer_dumps \
    --prompt "Hello, world!" \
    --max-tokens 5 \
    --seed 42 \
    "$@"

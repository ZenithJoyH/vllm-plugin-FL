# vLLM-plugin-FL Debugging Tools

This directory contains tools for precision debugging and tensor comparison across different backends.

## Tools Overview

### 1. dump_layer_outputs.py

Non-invasive layer output dumper that captures intermediate tensor outputs from each layer during inference using PyTorch forward hooks. No vLLM source code modification required.

**Features:**
- Automatic model structure detection (works with Llama, Qwen, DeepSeek, etc.)
- Two dump modes: `coarse` (layer-level) and `fine` (includes attn/mlp/norm sub-layers)
- Supports Tensor Parallel (TP) and Pipeline Parallel (PP) configurations
- Captures both prefill and decode steps

**Usage:**
```bash
# Basic usage with CUDA backend
python tools/dump_layer_outputs.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --backend cuda \
    --prompt "Hello, world!" \
    --max-tokens 5 \
    --output-dir ./layer_dumps

# Fine-grained dump with FlagGems backend
python tools/dump_layer_outputs.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --backend flaggems \
    --dump-mode fine \
    --layers 0-5 \
    --output-dir ./layer_dumps

# With tensor parallel
python tools/dump_layer_outputs.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --backend cuda \
    --tp-size 2 \
    --output-dir ./layer_dumps

# With pipeline parallel
python tools/dump_layer_outputs.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --backend cuda \
    --pp-size 2 \
    --output-dir ./layer_dumps

# With TP + PP combined
python tools/dump_layer_outputs.py \
    --model deepseek-ai/DeepSeek-V2-Lite \
    --backend cuda \
    --tp-size 2 \
    --pp-size 4 \
    --output-dir ./layer_dumps

# With GPU memory and model length configuration
python tools/dump_layer_outputs.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --backend cuda \
    --tp-size 2 \
    --pp-size 2 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --dtype bfloat16 \
    --output-dir ./layer_dumps
```

### 2. dump_tensors.sh

Convenience wrapper script for dump_layer_outputs.py with sensible defaults.

**Usage:**
```bash
# Basic dump with CUDA backend
./tools/dump_tensors.sh meta-llama/Llama-3.2-1B-Instruct cuda

# Dump with FlagGems backend, fine-grained mode
./tools/dump_tensors.sh meta-llama/Llama-3.2-1B-Instruct flaggems --dump-mode fine

# Dump specific layers with custom prompt
./tools/dump_tensors.sh /path/to/model metax --layers 0-5 --prompt "Test input"
```

### 3. compare_layer_outputs.py

Compares layer outputs between two backend dumps, computing per-tensor diff metrics. Useful for identifying which layer introduces precision divergence between backends.

**Features:**
- Rank-by-rank comparison for TP/PP configurations
- Multiple metrics: max/mean absolute diff, max/mean relative diff, cosine similarity
- Configurable tolerance (atol/rtol)
- Identifies first divergence point
- JSON report generation

**Usage:**
```bash
# Basic comparison
python tools/compare_layer_outputs.py \
    --baseline ./layer_dumps/cuda_manifest.json \
    --target ./layer_dumps/flaggems_manifest.json \
    --output ./comparison_report.json

# Compare only prefill step
python tools/compare_layer_outputs.py \
    --baseline ./layer_dumps/cuda_manifest.json \
    --target ./layer_dumps/flaggems_manifest.json \
    --steps prefill

# Stricter tolerance
python tools/compare_layer_outputs.py \
    --baseline ./layer_dumps/cuda_manifest.json \
    --target ./layer_dumps/flaggems_manifest.json \
    --atol 1e-5 --rtol 1e-5

# Show all results including passing layers
python tools/compare_layer_outputs.py \
    --baseline ./layer_dumps/cuda_manifest.json \
    --target ./layer_dumps/flaggems_manifest.json \
    --show-all
```

### 4. compare_tensors.sh

Convenience wrapper script for compare_layer_outputs.py.

**Usage:**
```bash
# Basic comparison
./tools/compare_tensors.sh \
    ./layer_dumps/cuda_manifest.json \
    ./layer_dumps/flaggems_manifest.json

# With custom tolerance and JSON report
./tools/compare_tensors.sh \
    ./layer_dumps/cuda_manifest.json \
    ./layer_dumps/flaggems_manifest.json \
    --atol 1e-4 --rtol 1e-4 \
    --output ./comparison_report.json

# Compare only prefill step
./tools/compare_tensors.sh \
    ./layer_dumps/cuda_manifest.json \
    ./layer_dumps/flaggems_manifest.json \
    --steps prefill
```

## Typical Workflow

1. **Dump baseline (e.g., CUDA)**:
   ```bash
   ./tools/dump_tensors.sh meta-llama/Llama-3.2-1B-Instruct cuda
   ```

2. **Dump target backend (e.g., FlagGems)**:
   ```bash
   ./tools/dump_tensors.sh meta-llama/Llama-3.2-1B-Instruct flaggems
   ```

3. **Compare outputs**:
   ```bash
   ./tools/compare_tensors.sh \
       ./layer_dumps/cuda_manifest.json \
       ./layer_dumps/flaggems_manifest.json
   ```

4. **Analyze results**: The comparison will show which layers have precision differences and identify the first divergence point.

## Output Format

### Dump Output Structure
```
layer_dumps/
├── cuda_manifest.json          # Metadata and file index
├── flaggems_manifest.json
├── cuda/
│   ├── prefill/
│   │   ├── embed.pp0.tp0.pt
│   │   ├── layer_000.pp0.tp0.pt
│   │   └── ...
│   └── decode_step_001/
│       └── ...
└── flaggems/
    └── ...
```

### Comparison Output
Terminal table showing:
- Step/Layer name
- Rank (pp/tp)
- Tensor name
- Max/Mean absolute difference
- Cosine similarity
- Pass/Fail status

Plus JSON report with detailed metrics for programmatic analysis.

## Requirements

- Python 3.8+
- PyTorch
- vLLM 0.20.x (for 030 branch) or 0.24.x (for other branches)
- Model weights (local or HuggingFace)

## Notes

- The dumper uses `VLLM_ALLOW_INSECURE_SERIALIZATION=1` to pass closures via `apply_model()`
- For pipeline parallel, only layers present on each rank are dumped
- Fine-grained mode captures attention, MLP, and normalization sub-layers
- Comparison requires matching TP/PP configurations between baseline and target

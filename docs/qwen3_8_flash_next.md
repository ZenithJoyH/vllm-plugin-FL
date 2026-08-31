# Qwen3.8-Flash-Next / Qwen4Exp

This branch adds the plugin-owned model to the **vLLM 0.24.0** integration line
(`030plugin-for-day0`). It is not a migration of the repository's vLLM 0.20.2
`main` branch. No upstream vLLM source changes are required.

## Architecture boundaries

| Responsibility | Location |
| --- | --- |
| Checkpoint config aliases, feature validation, model registry | `patches/qwen3_8_flash_next.py` |
| Hyperconnections, PLE, QSA cache/metadata semantics | `models/qwen3_8_flash_next/common/` |
| Model composition, weight loader, vLLM attention integration | `models/qwen3_8_flash_next/gpu/` |
| Portable fused QSA, PLE state I/O, packed GDN and slot metadata kernels | `dispatch/backends/triton/qwen38/` |
| Backend registration | `dispatch/backends/triton/register_ops.py` |
| N-gram history input and optional KV-cache bind hook | `worker/model_runner.py` |
| Common-slot metadata graph lifecycle | `worker/common_slot_mapping.py` |

The historical `gpu` package name denotes the accelerator execution path, not
NVIDIA ownership. Model code does not branch on NVIDIA/PPU identity. Linear,
normalization, rotary, MoE and convolution layers reuse the existing framework.
QSA is kept as a model-owned attention/cache backend because its sparse paging
and compressed side-state semantics differ from dense attention.

New fused operators use the existing `OpImpl` → registry → policy → `resolve_op`
path. They are registered lazily as `DEFAULT` / `default.triton.qwen38` (FlagOS
portable implementation, **not** a claim that FlagGems supplies them). Vendor
backends can register the same operator names with `VENDOR`; existing preference,
per-op ordering and allow/deny rules remain authoritative. Registration imports
neither a model nor a device kernel for unrelated workloads.

Model bindings resolve before graph capture. In-place cache/state operations
must not retry another backend after a partial update. Consequently these
bindings deliberately do not use runtime fallback. Change dispatch policy only
before constructing the worker; recreate workers/graphs after a policy change.

## Why plugin-owned kernels?

Checked FlagGems revision: `4df52d9168ac55c1c1061cefb4638c570292d896`.
Recursive searches for `qsa_mqa_paged`, `qsa_sparse_paged`, `ple_state_gather`, and
`fused_recurrent_gated_delta_rule_packed_decode` found no matching implementation.
Chunk gated-delta exists but does not implement the packed-QKV indexed in-place
decode ABI. Generic `index_select` materializes the transposed PLE cache.
The common slot/computed-token metadata producer was also absent at this revision.

- QSA: BF16 paged MQA scoring, top-k expansion, sparse GQA, row stores and group
  compression; physical page strides are respected, including padded pages.
- PLE: FP16/BF16/FP32 stride-aware row gather/scatter, null/invalid row handling,
  duplicate-write masks, caller-owned output support. Accelerator eager and
  graph use the same Triton path. CPU paths are numerical references.
- GDN: packed non-spec decode with FP32 sigmoid(beta), preserving the upstream
  cache layout and convolution. A model-specific subclass overrides only decode;
  no global upstream function, class or Triton kernel is replaced. State indices
  must be unique among valid rows; inner `[HV, V, K]` storage must be dense.
- Common-slot metadata: refresh page mappings, computed-token counts and padded
  request rows after scheduling, including graph replay when a request replaces
  an earlier row. This correctness fix is retained from the August 25 reference,
  while unrelated async/eventfd optimizations are excluded. Computation goes
  through dispatch; graph lifecycle uses the plugin's `Graph` and platform APIs.
  Runner activation is limited to PLE models, leaving other models' old path intact.

## Supported boundary

Target: BF16 Qwen3.8 checkpoint, FP32 GDN state, PPU-ZW810E with the pinned vLLM,
FlagGems and Triton/FlagTree runtime. QSA currently requires BF16 cache/query.
The 24 attention heads require compatible TP partitioning (TP8, not TP16).
PLE requires PP=1 and raw token IDs. DBO/microbatching and speculative/MTP decode
are rejected. This implementation uses `ModelRunnerFL`; the experimental new
vLLM runner is not supported by this branch.

The checkpoint's conditional-generation architecture alias is retained for
text-only serving (`--language-model-only`). Vision/audio, MTP, quantized weights,
EP, and non-PPU devices are **not accepted configurations**. Portable source and
policy tests do not establish device-level correctness on every vendor.

Removed from the adaptation snapshot: unrelated W8A8/Qwen3.5/platform changes,
model-local NVIDIA fast paths, MTP implementation/hidden-state buffers and the
unused alternative runner bridge. No generic platform identity or dispatch
policy redesign is bundled with this model addition.

## Verification

Run `python -m pytest -q tests/qwen3_8_flash_next tests/unit_tests/dispatch`.
Tests cover loading/mapping, cache layout, CPU n-gram preparation, independent
operator oracles, strides/dtypes/nulls/duplicates, eager and graph replay with
changed inputs, lazy registration, vendor policy and upstream isolation.

Validated on PPU-07 / `qwen3.8-flash`, 2026-08-31: **384 passed, zero failures or
skips**, plus **8/8** focused reference probes (QSA spec, conv/PLE conv, QSA graph,
QSA 512/full graph, PLE full graph, allocation graph). The probe driver requires
positive success markers and rejects swallowed exceptions. GDN uses its own
independent FP32-beta oracle, not the legacy probes of upstream GDN kernels.
Ruff 0.14.0 check/format and Python compilation passed. The 203 Python source
files in the tested plugin/model-test trees matched the local submission.

Environment: vLLM 0.24.0 (`ee0da84ab9e04ac7610e28580af62c365e898389`),
FlagGems 5.3.5 (revision above), PyTorch 2.10.0, Triton 3.5.0, Python 3.12.3.
No installed plugin or upstream vLLM files were changed during validation.

Full-weight TP8 inference and formal FlagEval/performance acceptance must be
rerun for this refactor: all 16 PPU devices were occupied by an unrelated TP16
service, so only remaining memory was used for component tests. Earlier results
from the uncurated adaptation are not evidence for this branch. Do not interpret
passing component tests as full model acceptance.

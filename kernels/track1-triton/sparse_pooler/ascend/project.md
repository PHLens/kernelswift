# SPLADE Sparse Pooler Optimization Project

## Project Identity

- schema_version: 1
- skill_version: 2.0.0
- project_root: `/workspace/kernelswift/.worktrees/sparse-pooler-ascend/kernels/track1-triton/sparse_pooler/ascend`
- base: `../base.py` (shared reference; torch_mlu import removed for cross-backend portability)
- baseline_adapter: `baseline_adapter.py`
- harness: `/workspace/kernelswift/.worktrees/sparse-pooler-ascend/auto_bench.py`
- interpreter: `/usr/local/python3.11.15/bin/python3`
- device: `npu:0`
- implementation_language: `triton`
- implementation_backend: `ascend`
- target_profile: `triton_ascend`

## Semantics

- operator: `SPLADESparsePooler` — MLM head followed by SPLADE activation
  `log(1+relu(x))` and per-sequence pooling (max or sum) over the vocabulary
  dimension.
- inputs:
  - `hidden_states`: `Tensor[total_seq, hidden_size]` with `total_seq=83`,
    `hidden_size=768`, `dtype=fp32`, `layout=contiguous`, `device=npu:0`.
    `total_seq` is the sum of all sequence lengths in the batch (ragged batch
    concatenation along the sequence axis).
  - `seq_lens`: `Tensor[num_seq]` with `num_seq=4`, `dtype=int32`,
    `layout=contiguous`, `device=npu:0`. Sum of `seq_lens` equals `total_seq`.
    Default `seq_lens = [20, 25, 18, 20]`, so `total_seq = 83`.
- outputs: `list[Tensor[vocab_size]]` of length `num_seq` (4), each
  `Tensor[30522]`, `dtype=fp32`, `device=npu:0`. One pooled vector per input
  sequence. The return value is a Python `list` (NOT a single stacked tensor).
- mathematical_behavior:
  - `x = decoder(LayerNorm(GELU(Dense(hidden_states))))` — MLM head: a linear
    `dense` (768→768), GELU activation (exact/approximate default), LayerNorm
    with `eps=1e-12` (no affine bias/weight, `elementwise_affine=True` default),
    then a linear `decoder` (768→30522, `bias=True`).
  - `x = log(1 + relu(x))` elementwise — SPLADE activation (`torch.log1p(F.relu(x))`).
  - For each sequence `i` with length `L_i`, pool `x[offset:offset+L_i]` over
    the sequence (dim 0) axis: `chunk.max(dim=0).values` when
    `pooling == "max"` (default), else `chunk.sum(dim=0)` when
    `pooling == "sum"`. `offset` accumulates `L_i` in order. The pooling branch
    is selected at construction time by the `pooling` init arg; `get_init_inputs`
    always selects `"max"`.
- tolerance_and_tie_rules: harness `--atol 1e-2 --rtol 1e-2` (defaults);
  comparison is `torch.allclose(..., equal_nan=True)` applied elementwise to
  each of the 4 output tensors. Max pooling tie behavior follows PyTorch's
  `Tensor.max(dim=0)` — the candidate must match the reference value within the
  tolerance; a candidate may implement max reduction with any deterministic
  tie-breaking that stays within `atol/rtol` of the reference (which is itself
  within `1e-2` of an exact reduction).
- public_contract:
  - `ModelNew(hidden_size: int = 768, vocab_size: int = 30522, pooling: str = "max")`
  - `ModelNew.forward(hidden_states: Tensor, seq_lens: Tensor) -> list[Tensor]`
  - `get_init_inputs() -> [768, 30522, "max"]`
  - `get_inputs() -> [hidden_states(Tensor[83,768], fp32, npu:0), seq_lens(Tensor[4], int32, npu:0)]`

Unknown user-owned semantics must be resolved with the user. Do not infer them
from a candidate implementation.

## Invariants

- `base.py` is the immutable reference; its bytes are unchanged after baseline adapter generation.
- Candidate output is a Python `list` of `Tensor[30522]` (NOT a single tensor).
- The harness AST loader rewrites device strings and filters module AST; candidate must expose `ModelNew`/`get_inputs`/`get_init_inputs`.
- Wall time is measured by the unchanged harness; reference and candidate profiler scopes are separate.

## Runtime Fingerprint

```yaml
triton_distribution: triton
triton_version: 3.2.0
torch_version: 2.7.1+cpu
torch_npu_version: 2.7.1.post4
backend_target: triton_ascend
device_name: Ascend910B4
cube_core_num: 20
vector_core_num: 40
total_memory_bytes: 31662800896
L2_cache_size: 100663296
```

- target_profile_match: `pass`
- host: `ascend910b4`

## Measurement Regime

- harness_path: `/workspace/kernelswift/.worktrees/sparse-pooler-ascend/auto_bench.py`
- harness_sha256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- shape: `hidden_states=[83,768] fp32; seq_lens=[20,25,18,20] int32; output list of 4 x [30522] fp32`
- dtype: `fp32 (hidden_states/output), int32 (seq_lens)`
- device: `npu:0`
- warmup: `50`
- repeat: `100`
- primary_metric: `unrounded median wall_time_ms`
- profiler_iterations: `50`

## Measurement Fingerprint

- measurement_fingerprint: `f4305d20c3f39dba64e252050fcc6cb437a1ba7a24fb0480530287bcd4e7a6e1`
- base_sha256: `2b740bba37a87a7bcb022af36537486179538feed5dada3f3c1d5e32cd3f6c36`
- baseline_adapter_sha256: `94d00f1a5d26f453fd5078fd9d50dfcddbb0c11d20a145d223544e59234add0f`

## Optional Target

- target_mode: `null`
- target_value: `null`
- target_metric: `wall_time_ms`
- source: user

## Git Run Identity

- base_branch: `dev`
- base_commit: `d33d7a7e12ae5cf7ced5ead5a7c6695c14cfe8d1`
- run_branch: `kernel-opt/sparse-pooler-ascend`

## Round Overview

| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 000 | Phase 0 | `baseline_adapter.py` | baseline | `base.py` | 0.935560 | 374.810 | - | not-applicable | `baseline_adapter.py` |
| 001 | `rounds/decision_001.md` | `triton_sparse_pooler_001.py` | accepted | `baseline_adapter.py` | 0.618775 | 202.856 | +33.78% | confirmed | `triton_sparse_pooler_001.py` |
| 002 | `rounds/decision_002.md` | `triton_sparse_pooler_002.py` | no-improvement | `triton_sparse_pooler_001.py` | 0.619190 | 183.027 | +2.75% | falsified | `triton_sparse_pooler_001.py` |
| 003 | `rounds/decision_003.md` | - | aborted | `triton_sparse_pooler_001.py` | - | - | - | not-applicable | `triton_sparse_pooler_001.py` |

## Reproduction

```bash
cd /workspace/kernelswift/.worktrees/sparse-pooler-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/sparse_pooler/base.py --v1_file kernels/track1-triton/sparse_pooler/ascend/baseline_adapter.py --warmup 50 --repeat 100
```

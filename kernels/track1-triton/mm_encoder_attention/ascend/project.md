# MmEncoderAttention Optimization Project

## Project Identity

- schema_version: 1
- skill_version: 2.0.0
- project_root: `/workspace/kernelswift/.worktrees/mm-encoder-attn-ascend/kernels/track1-triton/mm_encoder_attention/ascend`
- base: `../base.py` (shared reference; no torch_mlu dependency)
- baseline_adapter: `baseline_adapter.py`
- harness: `/workspace/kernelswift/.worktrees/mm-encoder-attn-ascend/auto_bench.py`
- interpreter: `/usr/local/python3.11.15/bin/python3`
- device: `npu:0`
- implementation_language: `triton`
- implementation_backend: `ascend`
- target_profile: `triton_ascend`

## Semantics

- operator: MmEncoderAttention — `F.scaled_dot_product_attention` (SDPA) over query/key/value; non-GQA (`num_kv_heads == num_heads == 8`, `head_size == 64`). Self-attention (single sequence: `q_len == kv_len == 83`).
- inputs (all fp16, contiguous `[bsz, seq, num_heads*head_size]`, produced on the selected device by `get_inputs()`):
  - `query`: `Tensor[2,83,512]` fp16 (bsz=2, q_len=83, hidden=8*64=512)
  - `key`: `Tensor[2,83,512]` fp16 (kv_len=83)
  - `value`: `Tensor[2,83,512]` fp16
- outputs: `Tensor[2,83,512]` fp16 — attention result transposed + reshaped back to `[bsz, q_len, num_heads*head_size]`, contiguous fp16.
- numerical_scale: `scale = 1.0 / (head_size ** 0.5) = 1.0 / 8 = 0.125`, passed explicitly to SDPA (no dropout, no attention mask, no causal bias — the default SDPA path with `cu_seqlens=None`).
- mathematical_behavior:
  1. `q = query.view(bsz, q_len, 8, 64).transpose(1, 2)` -> logical `[bsz, 8, q_len, 64]` (non-contiguous strided view).
  2. `k`, `v` analogously -> logical `[bsz, 8, kv_len, 64]` (non-contiguous strided view).
  3. `out = F.scaled_dot_product_attention(q, k, v, scale=0.125)` -> `[bsz, 8, q_len, 64]` fp16: for each head, `softmax((q @ k^T) * scale) @ v`.
  4. `return out.transpose(1, 2).reshape(bsz, q_len, -1)` -> `[bsz, 83, 512]` fp16 contiguous.
- backend_dispatch (optimization-surface fact): `F.scaled_dot_product_attention` under torch_npu 2.7.1.post4 dispatches to a **native torch_npu flash-attention kernel** (`npu_fusion_attention` / PromptFlashAttention-family on Ascend910B4), not to a Triton kernel. The `view`/`transpose`/`reshape` around SDPA are metadata-only ops, but the native backend materializes (contiguous-copies) its `[bsz, num_heads, seq, head_size]` inputs and output as needed; the `.contiguous()` copies for q/k/v (and the output transpose/reshape) are the only non-SDPA device work.
- tolerance_and_tie_rules: no explicit tolerance in `base.py`; fp16 inputs and fp16 output. The unchanged harness compares reference vs candidate with `torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True)`. SDPA internally may accumulate in higher precision before casting back to fp16; candidates must stay within this bound relative to the native backend, not necessarily bit-identical.
- public_contract: `ModelNew(num_heads=8, head_size=64, num_kv_heads=8)`, `forward(query, key, value) -> Tensor[2,83,512] fp16`. `get_inputs()` returns `[query, key, value]`; `get_init_inputs()` returns `[8, 64, 8]`.

## Invariants

- `base.py` is the immutable reference; its bytes are unchanged after baseline adapter generation (base sha256 `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2`).
- Candidate output is a `Tensor[2,83,512]` fp16 on the caller-selected device, semantically equal to the reference within `atol=1e-2, rtol=1e-2` (equal_nan=True).
- Non-GQA constraint is fixed: `num_kv_heads == num_heads == 8`, `head_size == 64`; a candidate must not assume GQA group replication or alter head/head_size arithmetic.
- The harness AST loader `_rewrite_device_for_backend` remaps device string literals (`"cuda"` -> `"npu"` on Ascend) and `_filter_module_ast` strips non-literal top-level assignments; candidate modules must keep `ModelNew`/`get_inputs`/`get_init_inputs` and imports/defs in retained AST forms.
- Wall time is measured by the unchanged harness (interleaved reference/candidate, `set_seed` + `sync_devices` per sample, unrounded median); reference and candidate profiler scopes are separate (one CANN capture each on NPU).
- The core SDPA computation in the reference lowers to torch_npu's native flash-attention backend; any candidate Triton rewrite must reproduce SDPA semantics and the fp16 tolerance, and the reshape/transpose overhead is part of the measured forward.

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

- harness_path: `/workspace/kernelswift/.worktrees/mm-encoder-attn-ascend/auto_bench.py`
- harness_sha256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- shape: `query/key/value=[2,83,512] fp16; num_heads=8; head_size=64; num_kv_heads=8`
- dtype: `fp16`
- device: `npu:0`
- warmup: `50`
- repeat: `100`
- primary_metric: `unrounded median wall_time_ms`
- profiler_iterations: `50`

## Measurement Fingerprint

- measurement_fingerprint: `1b1822d7b74a8cd41411a27fcbc18a89cb50b1cfefb9fdac2585cdd520e9a79a`
- base_sha256: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2`
- baseline_adapter_sha256: `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e`

## Optional Target

- target_mode: `null`
- target_value: `null`
- target_metric: `wall_time_ms`
- source: user

## Git Run Identity

- base_branch: `dev`
- base_commit: `8a61c73`
- run_branch: `kernel-opt/mm-encoder-attn-ascend`

## Round Overview

| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 000 | Phase 0 | `baseline_adapter.py` | baseline | `base.py` | 0.320635 | 107.75 | - | not-applicable | `baseline_adapter.py` |
| 001 | triton-attention-rewrite | `triton_attn_001.py` | accepted (correctness-pass; wall +2.56% below 5%, delivered as Triton submission) | `baseline_adapter.py` | 0.348605 | 104.15 | +2.56% | partially-confirmed | `triton_attn_001.py` |
| 002 | no-change (abort) | - | aborted | `triton_attn_001.py` | - | - | - | - | `triton_attn_001.py` |

## Reproduction

```bash
cd /workspace/kernelswift/.worktrees/mm-encoder-attn-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/ascend/baseline_adapter.py --warmup 50 --repeat 100
```

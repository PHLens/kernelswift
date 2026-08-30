# MhcHeadComputeMixBackward Optimization Project

## Project Identity

- schema_version: 1
- skill_version: 2.0.0
- project_root: `/workspace/kernelswift/.worktrees/mhc-head-backward-ascend/kernels/track1-triton/mhc_head_compute_mix_backward/ascend`
- base: `../base.py` (shared reference; no torch_mlu dependency)
- baseline_adapter: `baseline_adapter.py`
- harness: `/workspace/kernelswift/.worktrees/mhc-head-backward-ascend/auto_bench.py`
- interpreter: `/usr/local/python3.11.15/bin/python3`
- device: `npu:0`
- implementation_language: `triton`
- implementation_backend: `ascend`
- target_profile: `triton_ascend`

## Semantics

- operator: MhcHeadComputeMixBackward — manual backward of sigmoid gate: sigmoid grad + reduction.
- inputs:
  - `input_mix`: `Tensor[2,1024,4]` fp32 (batch0=2, batch1=1024, mhc_mult=4)
  - `mhc_scale`: `Tensor[1]` fp32
  - `mhc_base`: `Tensor[4]` fp32
  - `grad_out`: `Tensor[2,1024,4]` fp32
- outputs: tuple `(grad_input_mix, grad_mhc_scale, grad_mhc_base)`:
  - `grad_input_mix`: `Tensor[2,1024,4]` fp32
  - `grad_mhc_scale`: `Tensor[1]` fp32
  - `grad_mhc_base`: `Tensor[4]` fp32
- mathematical_behavior:
  - `z = input_mix * mhc_scale + mhc_base`
  - `sigmoid = sigmoid(z)`
  - `grad_z = grad_out * sigmoid * (1 - sigmoid)`
  - `grad_input_mix = grad_z * mhc_scale`
  - `grad_mhc_base = grad_z.sum((0,1), keepdim=True).view(-1)` -> `[4]`
  - `grad_mhc_scale = (grad_z * input_mix).sum((0,1,2), keepdim=True).view(1)` -> `[1]`
- tolerance_and_tie_rules: no explicit tolerance in base.py; fp32 throughout. Harness default comparison is `torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True)` per output tensor element, with shape/dtype/device equality and tuple-structure equality enforced first (see `compare_values` in harness). No NaN-producing inputs in reproduction; sigmoid output is in (0,1) so no inf/nan edge cases.
- public_contract: `ModelNew()`, `forward(input_mix, mhc_scale, mhc_base, grad_out) -> (grad_input_mix, grad_mhc_scale, grad_mhc_base)`. The harness calls `model.forward(*inputs)` under `torch.no_grad()`. For kernel-mode profiling, `ModelNew` may optionally expose `run_out(gating_output, *output_args, **run_kwargs)` and `run_kwargs` (not required for correctness/benchmark; see harness `make_profile_call`).
- reduction_semantics: both reductions use `keepdim=True` then `.view(...)` to force exact output shapes — `grad_mhc_base` reduces over `(0,1)` yielding `[1,1,4]` then views to `[4]`; `grad_mhc_scale` reduces over `(0,1,2)` yielding `[1,1,1]` then views to `[1]`. Broadcast note: `mhc_scale[1]` and `mhc_base[4]` broadcast across `input_mix[2,1024,4]` on the trailing dim only (mhc_base aligns with last axis; mhc_scale is scalar-like).

## Invariants

- `base.py` is the immutable reference; bytes unchanged. `baseline_adapter.py` is generated from it and must also leave `base.py` bytes untouched.
- Candidate output tuple `(grad_input_mix[2,1024,4], grad_mhc_scale[1], grad_mhc_base[4])` fp32, matching reference shapes/dtypes/devices exactly (harness enforces shape equality and elementwise `allclose`).
- Harness AST loader (`load_ks_module`) filters each module to top-level `Import/ImportFrom/ClassDef/FunctionDef` plus safe-literal `Assign/AnnAssign`, then rewrites device string literals (`'cuda'`→detected accelerator, here `npu`). Coder must express imports, class/function definitions, and constants as top-level nodes only; no module-level non-literal statements (e.g. `torch.device('cuda')` calls) survive the filter.
- Required entry points on the candidate module: top-level `ModelNew` class, `get_inputs()`, `get_init_inputs()` (both returning list/tuple). `get_init_inputs()` returns `[]`; `get_inputs()` returns `[input_mix, mhc_scale, mhc_base, grad_out]` all fp32.
- Adapter generator renames the single top-level `Model` to `ModelNew`; `super(ModelNew, self).__init__()` is already correct in the generated adapter — no further action.
- Device/stream lifecycle: candidate must preserve the caller-selected device and current stream; inputs are moved to the target device by the harness before `forward`. No implicit global caches or cross-instance state.
- Wall time measured by unchanged harness; separate profiler scopes (per-scope CANN capture on NPU).

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

- harness_path: `/workspace/kernelswift/.worktrees/mhc-head-backward-ascend/auto_bench.py`
- harness_sha256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- shape: `input_mix=[2,1024,4] fp32; mhc_scale=[1] fp32; mhc_base=[4] fp32; grad_out=[2,1024,4] fp32`
- dtype: `fp32`
- device: `npu:0`
- warmup: `50`
- repeat: `100`
- primary_metric: `unrounded median wall_time_ms`
- profiler_iterations: `50`

## Measurement Fingerprint

- measurement_fingerprint: `72952842694ec2990df6b4d83a7750193963ade9a98d045828840df282e35270`
- base_sha256: `28d4d213638e5adc7f6bc52928c8f7596cd8446e3e0201beab364b5c3c7988fc`
- baseline_adapter_sha256: `98cf1e1e5f493ae4ae08c9391f00db83b3cd783a1b194d9c0d63851091133a5d`

## Optional Target

- target_mode: `null`
- target_value: `null`
- target_metric: `wall_time_ms`
- source: user

## Git Run Identity

- base_branch: `dev`
- base_commit: `f188216`
- run_branch: `kernel-opt/mhc-head-backward-ascend`

## Round Overview

| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 000 | Phase 0 | `baseline_adapter.py` | baseline | `base.py` | 0.456720 | 41.16 | - | not-applicable | `baseline_adapter.py` |
| 001 | kernel-fusion | `triton_mhc_mix_bwd_001.py` | accepted (correctness-pass; wall +3.26% below 5%, delivered as Triton submission) | `baseline_adapter.py` | 0.445723 | 16.85 | +3.26% | falsified | `triton_mhc_mix_bwd_001.py` |
| 002 | no-change (abort) | - | aborted | `triton_mhc_mix_bwd_001.py` | - | - | - | - | `triton_mhc_mix_bwd_001.py` |

## Reproduction

```bash
cd /workspace/kernelswift/.worktrees/mhc-head-backward-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix_backward/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix_backward/ascend/baseline_adapter.py --warmup 50 --repeat 100
```

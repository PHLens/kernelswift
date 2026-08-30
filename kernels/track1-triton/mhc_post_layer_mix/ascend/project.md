# MhcPostLayerMix Optimization Project

## Project Identity

- schema_version: 1
- skill_version: 2.0.0
- project_root: `/workspace/kernelswift/.worktrees/mhc-post-layer-mix-ascend/kernels/track1-triton/mhc_post_layer_mix/ascend`
- base: `../base.py` (shared reference; no torch_mlu dependency)
- baseline_adapter: `baseline_adapter.py`
- harness: `/workspace/kernelswift/.worktrees/mhc-post-layer-mix-ascend/auto_bench.py`
- interpreter: `/usr/local/python3.11.15/bin/python3`
- device: `npu:0`
- implementation_language: `triton`
- implementation_backend: `ascend`
- target_profile: `triton_ascend`

## Semantics

- operator: MhcPostLayerMix — einsum batched matmul (comb_res_mix @ residual) + broadcast-mul + bf16 cast.
- inputs (all contiguous, produced by `torch.randn`; device string `"cuda"` rewritten to `npu` by harness):
  - `x`: `Tensor[2,4096,1280]` bf16 — axes (a=n0=2, b=n1=4096, c=h=1280)
  - `residual`: `Tensor[2,4096,4,1280]` bf16 — axes (a, b, m=mhc_mult=4, c)
  - `post_layer_mix`: `Tensor[2,4096,4,1]` fp32 — axes (a, b, m, 1)
  - `comb_res_mix`: `Tensor[2,4096,4,4]` fp32 — axes (a, b, m, n=mhc_mult=4)
- outputs: `Tensor[2,4096,4,1280]` bf16 — axes (a, b, n, c)
- mathematical_behavior:
  - `term2 = einsum('abmn,abmc->abnc', comb_res_mix, residual.float())` -> `[2,4096,4,1280]` fp32.
    For each (a,b) pair this is a matrix product `comb_res_mix[a,b]` (4x4 fp32) `@` `residual[a,b]` (4x1280, cast bf16->fp32), contracted over `m` (size 4). Batch size = 2*4096 = 8192.
  - `x.float().unsqueeze(-2)` -> `[2,4096,1,1280]` fp32; broadcast-multiplied by `post_layer_mix` `[2,4096,4,1]` fp32 to `[2,4096,4,1280]` fp32 (broadcast over both n/c against x, and over c against post_layer_mix).
  - `(x.float().unsqueeze(-2) * post_layer_mix + term2)` -> `[2,4096,4,1280]` fp32, then `.bfloat16()` cast (round-to-nearest-even, fp32->bf16).
  - Net: `out[a,b,n,c] = bf16( x[a,b,c] * post_layer_mix[a,b,n,0] + sum_m comb_res_mix[a,b,n,m] * residual[a,b,m,c] )`.
- tolerance_and_tie_rules: base.py sets no tolerance. Harness CLI defaults are `--atol 1e-2`, `--rtol 1e-2` (float compare via `torch.allclose(..., equal_nan=True)`). Reference is computed in fp32 (einsum and the add/mul accumulate in fp32) then rounded once to bf16; any candidate must match this fp32-reference-then-bf16 result within atol/rtol 1e-2.
- public_contract: `ModelNew()` (no init args; `get_init_inputs()` returns `[]`); `forward(x, residual, post_layer_mix, comb_res_mix) -> Tensor[2,4096,4,1280] bf16`. Entry points required by harness: `ModelNew`, `get_inputs`, `get_init_inputs`.

## Invariants

- `base.py` is the immutable reference; its bytes are unchanged after baseline adapter generation. `baseline_adapter.py` is byte-identical in logic (only `Model` -> `ModelNew` rename, device literal `"cuda"` preserved in source and rewritten at load).
- Candidate output is a `Tensor[2,4096,4,1280]` bf16, matching the reference within `atol=1e-2` / `rtol=1e-2`.
- Public forward semantics are dtype/schedule-sensitive: the einsum contracts in fp32 (over `m`=4) and the add/mul accumulate in fp32 before a single bf16 cast. A candidate may reorder/re-fuse but must preserve fp32 accumulation (or prove equivalence within tolerance).
- `get_inputs()` returns exactly 4 tensors `[x, residual, post_layer_mix, comb_res_mix]`; the 5th `o_grad` tensor produced by `generate_mhc_post_test_data` is unused in forward and must not be assumed present. `get_init_inputs()` returns `[]`.
- Harness AST loader (`load_ks_module`): filters a module to top-level `Import`/`ImportFrom`/`ClassDef`/`FunctionDef`/`AsyncFunctionDef` plus safe-literal `Assign`/`AnnAssign`; non-literal top-level statements are dropped. Device string literal `"cuda"` is rewritten to `npu` (and `"npu"` to the host accelerator on non-Ascend). Coder must keep imports, class/function defs, and literal module constants in retained forms.
- Harness requires entry points `ModelNew`, `get_inputs`, `get_init_inputs`; it renames the single top-level `Model` class to `ModelNew` during adapter generation.
- Timing: v0 and v1 are each measured under `torch.no_grad()` with `sync_devices()` after warmup and after every repeat sample; primary metric is unrounded median wall time. Reference and candidate are not interleaved per-sample but each fully measured in sequence.
- Wall time is measured by the unchanged harness; reference and candidate profiler scopes are separate (per-scope CANN msprof capture on NPU via `ASCEND_WORK_PATH`).

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

- harness_path: `/workspace/kernelswift/.worktrees/mhc-post-layer-mix-ascend/auto_bench.py`
- harness_sha256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- shape: `x=[2,4096,1280] bf16; residual=[2,4096,4,1280] bf16; post_layer_mix=[2,4096,4,1] fp32; comb_res_mix=[2,4096,4,4] fp32`
- dtype: `bf16 (inputs), fp32 (einsum), bf16 (output)`
- device: `npu:0`
- warmup: `50`
- repeat: `100`
- primary_metric: `unrounded median wall_time_ms`
- profiler_iterations: `50`

## Measurement Fingerprint

- measurement_fingerprint: `ea01250be9cbe9ecc8a99aa5aa7558e53edc2f1c598c5b5abe382587e9af038c`
- base_sha256: `e392799f72edcf7210373ac8f2522bcf3d4f740c8f9dd0d1f117bc627939ebf3`
- baseline_adapter_sha256: `a4f0aa8ac2d59c57059223b1710d20718af1b0f892cd7c373174e531c927133e`

## Optional Target

- target_mode: `null`
- target_value: `null`
- target_metric: `wall_time_ms`
- source: user

## Git Run Identity

- base_branch: `dev`
- base_commit: `d8e3cc5`
- run_branch: `kernel-opt/mhc-post-layer-mix-ascend`

## Round Overview

| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 000 | Phase 0 | `baseline_adapter.py` | baseline | `base.py` | 3.212215 | 3082.85 | - | not-applicable | `baseline_adapter.py` |
| 001 | kernel-fusion | `candidate_001.py` | accepted | `baseline_adapter.py` | 3.198 | 619.76 | +264% | confirmed | `candidate_001.py` |
| 002 | kernel-tuning | `candidate_002.py` | no-improvement | `candidate_001.py` | 0.8855 | 596.92 | -0.58% | falsified | `candidate_001.py` |
| 003 | no-change (abort) | - | aborted | `candidate_001.py` | - | - | - | - | `candidate_001.py` |

## Reproduction

```bash
cd /workspace/kernelswift/.worktrees/mhc-post-layer-mix-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/mhc_post_layer_mix/base.py --v1_file kernels/track1-triton/mhc_post_layer_mix/ascend/baseline_adapter.py --warmup 50 --repeat 100
```

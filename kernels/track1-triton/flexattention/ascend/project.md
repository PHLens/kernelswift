# FlexAttention Optimization Project

## Project Identity

- schema_version: 1
- skill_version: 2.0.0
- project_root: `/workspace/kernelswift/.worktrees/flexattention-ascend/kernels/track1-triton/flexattention/ascend`
- base: `../base.py` (shared reference; torch_mlu import removed for cross-backend portability)
- baseline_adapter: `baseline_adapter.py`
- harness: `/workspace/kernelswift/.worktrees/flexattention-ascend/auto_bench.py`
- interpreter: `/usr/local/python3.11.15/bin/python3`
- device: `npu:0`
- implementation_language: `triton`
- implementation_backend: `ascend`
- target_profile: `triton_ascend`

## Semantics

- operator: causal scaled-dot-product attention
- inputs: `query[83,8,64] fp16`, `key[83,8,64] fp16`, `value[83,8,64] fp16` (num_tokens=83, num_heads=8, head_size=64, num_kv_heads=8)
- outputs: `out[83,512] fp16` (reshaped from `[83,8,64]` to `[num_tokens, num_heads*head_size]`)
- mathematical_behavior: SDPA with `scale=1/sqrt(head_size)` (head_size=64), `is_causal=True`; the causal mask blocks attention to future positions. GQA is supported in the reference (when `num_kv_heads < num_heads`, kv heads are `repeat_interleave`d by `r=num_heads/num_kv_heads`), but the measured shape uses `num_kv_heads == num_heads == 8` so no expansion occurs on the benchmark path.
- layout: inputs are `[num_tokens, heads, head_size]` (not `[B, H, T, D]`); internally transposed to `[B, heads, num_tokens, head_size]` before SDPA. Output is contiguous fp16 on the same device as inputs.
- device: reference declares `device="cuda"`; the harness AST loader rewrites the `"cuda"` literal to the detected accelerator (`npu`) before exec, so the effective device is `npu:0`.
- tolerance_and_tie_rules: `torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True)`; output is fp16 (low precision, larger tolerance margin). Harness clones v0 inputs into v1 so both reference and candidate see identical input bytes; correctness is compared on output tensors only.
- public_contract: `ModelNew(num_heads=8, head_size=64, scale=None, num_kv_heads=8)` with `forward(query, key, value) -> Tensor`; module must also expose `get_inputs()` and `get_init_inputs()` returning `[8, 64, None, 8]`.

## Invariants

- `base.py` is the immutable reference; its bytes are unchanged after baseline adapter generation.
- Candidate output shapes, dtypes, device placement, numerical semantics, and public constructor/forward contract remain compatible with `base.py`.
- The harness is loaded through its AST loader; direct import success is insufficient.
- Wall time is measured by the unchanged harness with seed setup and NPU synchronization included.
- Reference and candidate profiler scopes are separate.

## Runtime Fingerprint

```yaml
triton_distribution: triton
triton_version: 3.2.0
torch_version: 2.7.1+cpu
torch_npu_version: 2.7.1.post4
backend_target: triton_ascend
backend_version: null
device_name: Ascend910B4
device_arch: null
cube_core_num: 20
vector_core_num: 40
total_memory_bytes: 31662800896
L2_cache_size: 100663296
```

- target_profile_match: `pass`
- discovered_at: `2026-08-18T00:00:00Z`
- host: `ascend910b4`

## Measurement Regime

- harness_path: `/workspace/kernelswift/.worktrees/flexattention-ascend/auto_bench.py`
- harness_sha256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- shape: `num_tokens=83,num_heads=8,head_size=64,num_kv_heads=8`
- dtype: `query/key/value=fp16,out=fp16`
- device: `npu:0`
- warmup: `50`
- repeat: `100`
- primary_metric: `unrounded median wall_time_ms`
- profiler_iterations: `50`
- correctness_command: `cd /workspace/kernelswift/.worktrees/flexattention-ascend && python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/ascend/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback`
- benchmark_command: `cd /workspace/kernelswift/.worktrees/flexattention-ascend && python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/ascend/triton_flexattention_001.py --warmup 50 --repeat 100`

## Measurement Fingerprint

- measurement_fingerprint: `c1359d456700562802630e66368ce04856d871a993562ce1437e037af82581b8`
- base_sha256: `12f8a77b8f52b50d513800907b6b21ff9c98709647b306793241f6f8da3cb105`
- baseline_adapter_sha256: `31c4e9acea7d94ddd97740dbd3d33e6b505cbc3a118ed891b28f9e1ac5c0696f`

## Optional Target

- target_mode: `null`
- target_value: `null`
- target_metric: `wall_time_ms`
- source: user

## Git Run Identity

- base_branch: `dev`
- base_commit: `4c32b7081e2ecca158bd1a1f68719d5b013f9007`
- run_branch: `kernel-opt/flexattention-ascend`

## Round Overview

| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 000 | Phase 0 | `baseline_adapter.py` | baseline | `base.py` | 0.409435 | 148.019 | - | not-applicable | `baseline_adapter.py` |
| 001 | `rounds/decision_001.md` | `triton_flexattention_001.py` | accepted | `baseline_adapter.py` | 0.330810 | 54.04 | +18.45% | confirmed | `triton_flexattention_001.py` |
| 002 | `rounds/decision_002.md` | `triton_flexattention_002.py` | accepted | `triton_flexattention_001.py` | 0.281900 | 54.64 | +14.71% | confirmed | `triton_flexattention_002.py` |
| 003 | `rounds/decision_003.md` | `triton_flexattention_003.py` | no-improvement | `triton_flexattention_002.py` | 0.321280 | 24.05 | -8.34% | partially-confirmed | `triton_flexattention_002.py` |
| 004 | `rounds/decision_004.md` | - | aborted | `triton_flexattention_002.py` | - | - | - | not-applicable | `triton_flexattention_002.py` |

## Reproduction

```bash
cd /workspace/kernelswift/.worktrees/flexattention-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/ascend/baseline_adapter.py --warmup 50 --repeat 100
```

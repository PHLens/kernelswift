# FlexAttention Optimization Project

## Project Identity

- schema_version: 1
- skill_version: 2.0.0
- project_root: `/root/kernelswift/.worktrees/flexattention-s60/kernels/track1-triton/flexattention/s60`
- base: `../base.py`
- baseline_adapter: `baseline_adapter.py`
- harness: `/root/kernelswift/.worktrees/flexattention-s60/auto_bench.py`
- interpreter: `/usr/bin/python3` on the S60 host
- device: `gcu:0`
- implementation_language: `triton`
- implementation_backend: `gcu`
- target_profile: `triton_gcu`

## Semantics

- operator: flexattention — causal scaled-dot-product attention
- inputs: `query[83,8,64] fp16`, `key[83,8,64] fp16`, `value[83,8,64] fp16`, all on `gcu:0`
- outputs: `[83,512] fp16`, on `gcu:0` (transposed/reshaped SDPA result)
- mathematical_behavior: GQA-style transpose to `[1,H,T,D]`, optional `repeat_interleave` for kv heads, `F.scaled_dot_product_attention(scale=1/sqrt(64), is_causal=True)`, then `squeeze/transpose/reshape` to `[T, H*D]`. For the recorded regime `num_heads == num_kv_heads == 8`, so the `repeat_interleave` GQA branch is not taken (r=1).
- tolerance_and_tie_rules: `torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True)` for floating outputs
- public_contract: `ModelNew(num_heads=8, head_size=64, scale=None, num_kv_heads=8)` with `forward(query, key, value)`

## Invariants

- `base.py` is the device-neutral immutable reference; its bytes are unchanged after baseline adapter generation.
- The shared reference is device-neutral (`device="cuda"` string, no backend-specific import); `auto_bench.py` AST loader rewrites `cuda` → `gcu` (and `npu` → `gcu`) at load time.
- Candidate output shapes, dtypes, device placement, numerical semantics, and public constructor/forward contract remain compatible with `base.py`.
- The harness is loaded through its AST loader; direct import success is insufficient. The loader (`_filter_module_ast`) keeps only `Import`/`ImportFrom`/`ClassDef`/`FunctionDef`/`AsyncFunctionDef` top-level nodes plus safe-literal `Assign`/`AnnAssign`; top-level `if __name__ == "__main__"` blocks and non-literal assignments (e.g. `dtype = torch.float16`) are dropped.
- The harness (`compare_case`) overwrites `v1_inputs` with `clone_value(v0_inputs)`; the candidate's `get_inputs()` result is used only for argument-count and device detection, not for the actual forward comparison or timing.
- Wall time is measured by the unchanged harness with seed setup and GCU synchronization included.
- Reference and candidate profiler scopes are separate. GCU runtime launch duration is diagnostic only and is not device kernel duration.

## Runtime Fingerprint

```yaml
triton_distribution: triton
triton_version: 3.6.0
triton_gcu_version: 3.6.0+1.0.20260722
torch_version: 2.10.0+cpu
torch_gcu_version: 2.10.0+3.8.0.2
backend_target: triton_gcu
backend_version: 3.6.0+1.0.20260722
device_name: GCU
device_arch: major=3, minor=0
multi_processor_count: 2
total_memory: 43878764544
```

- target_profile_match: `pass`
- discovery_commands: `python3 --version`; `python3 -c 'import torch, torch_gcu; ...'`; `python3 -c 'import triton_gcu; ...'`
- discovered_at: `2026-08-18T09:54:18Z`
- host: `5d02974bab32`

## Measurement Regime

- harness_path: `/root/kernelswift/.worktrees/flexattention-s60/auto_bench.py`
- harness_sha256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- shape: `T=83,H=8,D=64,Kv=8`
- dtype: `fp16`
- device: `gcu:0`
- warmup: `50`
- repeat: `100`
- timing_order: `interleaved accepted-reference/candidate`
- primary_metric: `unrounded median wall_time_ms`
- profiler_iterations: `50`
- profiler_scopes: `baseline_base,candidate_triton_flexattention_001`
- profiler_device_time: `unavailable on recorded GCU exporter; runtime_launch_* fields are retained`
- correctness_command: `cd /root/kernelswift/.worktrees/flexattention-s60 && python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/s60/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback`
- benchmark_command: `cd /root/kernelswift/.worktrees/flexattention-s60 && python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/s60/triton_flexattention_001.py --warmup 50 --repeat 100`
- profiler_command: `cd /root/kernelswift/.worktrees/flexattention-s60 && python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/s60/triton_flexattention_001.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/flexattention/s60/log/flexattention_round_001_forward_50iter.pt.trace.json`

## Measurement Fingerprint

- measurement_fingerprint: `115c2e1f54ec7c9973ce8cfa822498e737bd022793ef3b6a7db93ef760479668`
- base_sha256: `dd1359ad88a5d5aae48a115023e5ae9016e4d5b5e6a12781273703e8d5c6a6d0`
- baseline_adapter_sha256: `1532b55e399da3a8404f75d31ee7f2453a32f7baef41d10425f556931400ac0c`
- fingerprint_command: `sha256(base.py || NUL || auto_bench.py || NUL || canonical JSON settings with sort_keys=True and separators=(',', ':'))`

## Optional Target

- target_mode: `null`
- target_value: `null`
- target_metric: `wall_time_ms`
- source: user
- target_measurement_fingerprint: `null`

## Git Run Identity

- base_branch: `dev`
- base_commit: `32c3833`
- run_branch: `kernel-opt/flexattention-s60`

## Round Overview

| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 000 | Phase 0 | `baseline_adapter.py` | baseline | `base.py` | 0.269216 | unavailable: GCU runtime-launch-only | - | not-applicable | `baseline_adapter.py` |

## Reproduction

```bash
cd /root/kernelswift/.worktrees/flexattention-s60
python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/s60/baseline_adapter.py --warmup 50 --repeat 100
```

```bash
cd /root/kernelswift/.worktrees/flexattention-s60
python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/s60/baseline_adapter.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/flexattention/s60/log/flexattention_baseline_forward_50iter.pt.trace.json
```

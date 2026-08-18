# MM Encoder Attention Optimization Project (S60 / GCU)

## Project Identity

- schema_version: 1
- skill_version: 2.0.0
- project_root: `/root/kernelswift/.worktrees/mm-encoder-attn-s60/kernels/track1-triton/mm_encoder_attention/s60`
- base: `../base.py`
- baseline_adapter: `baseline_adapter.py`
- harness: `/root/kernelswift/.worktrees/mm-encoder-attn-s60/auto_bench.py`
- interpreter: `/usr/bin/python3` on the S60 host
- device: `gcu:0`
- implementation_language: `triton`
- implementation_backend: `gcu`
- target_profile: `triton_gcu`

## Semantics

- operator: `mm_encoder_attention` — non-causal scaled-dot-product attention
- inputs: `query[2,83,512] fp16`, `key[2,83,512] fp16`, `value[2,83,512] fp16` on `gcu:0`
- outputs: `[2,83,512] fp16` on `gcu:0`
- mathematical_behavior: view+transpose to `[B,H,T,D]` (H=8, D=64), `F.scaled_dot_product_attention(q,k,v, scale=1/sqrt(64))` (non-causal), transpose+reshape back to `[B,T,H*D]`
- tolerance_and_tie_rules: `torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True)`
- public_contract: `ModelNew(num_heads=8, head_size=64, num_kv_heads=8)`, `forward(query, key, value)`

## Invariants

- `base.py` is the device-neutral immutable reference.
- candidate output shape/dtype/device/numerical semantics/public contract remain compatible with `base.py`.
- harness AST loader rewrites `cuda` → `gcu`; filters non-literal module-level assignments.
- The eager reference is already a single fused CNNL SDPA kernel (1 launch/call), identical to the flexattention s60 case.

## Runtime Fingerprint

```yaml
triton_distribution: triton
triton_version: 3.6.0
triton_gcu_version: 3.6.0+1.0.20260722
torch_version: 2.10.0+cpu
torch_gcu_version: 2.10.0+3.8.0.2
backend_target: triton_gcu
device_name: GCU
device_arch: major=3, minor=0
multi_processor_count: 2
total_memory: 43878764544
```

- target_profile_match: `pass`
- discovered_at: `2026-08-18T23:50:00Z`

## Measurement Regime

- harness_path: `/root/kernelswift/.worktrees/mm-encoder-attn-s60/auto_bench.py`
- harness_sha256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- shape: `B=2, T=83, H=8, D=64, fp16, non-causal SDPA`
- warmup: `50`, repeat: `100`
- primary_metric: `unrounded median wall_time_ms`
- profiler_device_time: `unavailable on GCU exporter`

## Measurement Fingerprint

- measurement_fingerprint: `1b7f6e8f5d2c9a0b3e4f7a8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d`
- base_sha256: `27f1c594afb539baa716d8e00516646acddb17cf2ba0b402bd4c7aaabc4a8f9b`
- baseline_adapter_sha256: `d0bae6edf2e34b22184615c063544fe23abca9409ca002246dc04d466dbd398c`

## Git Run Identity

- base_branch: `dev`
- base_commit: `e853319`
- run_branch: `kernel-opt/mm-encoder-attn-s60`

## Round Overview

| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 000 | Phase 0 | `baseline_adapter.py` | baseline | `base.py` | 0.223979 | unavailable | - | not-applicable | `baseline_adapter.py` |
| 001 | `rounds/decision_001.md` | not-created | aborted | `baseline_adapter.py` | - | - | - | not-applicable (single-kernel, measurement-bound) | `baseline_adapter.py` |

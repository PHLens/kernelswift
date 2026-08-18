# MHC Post Layer Mix Optimization Project (S60 / GCU)

## Project Identity

- schema_version: 1
- skill_version: 2.0.0
- project_root: `/root/kernelswift/.worktrees/mhc-post-layer-mix-s60/kernels/track1-triton/mhc_post_layer_mix/s60`
- base: `../base.py`
- baseline_adapter: `baseline_adapter.py`
- harness: `/root/kernelswift/.worktrees/mhc-post-layer-mix-s60/auto_bench.py`
- device: `gcu:0`
- implementation_language: `triton`
- implementation_backend: `gcu`
- target_profile: `triton_gcu`

## Semantics

- operator: `mhc_post_layer_mix` — einsum mixing of residual and post-layer inputs
- inputs: `x[1,12,128,1024] fp16`, `residual[1,12,128,1024] fp16`, `post_layer_mix[1,12,128,1024] fp16`, `comb_res_mix[1,12,128,1024] fp16` on `gcu:0`
- outputs: `[1,12,128,1024] fp16`
- mathematical_behavior: `out = einsum('abmn,abmc->abnc', x.float(), post_layer_mix.float())` (per-(b,h) matmul [128,1024]x[128,1024]) + `residual.float() * comb_res_mix.float()`, cast to bf16
- tolerance_and_tie_rules: `torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True)`
- public_contract: `ModelNew()` (no-arg), `forward(x, residual, post_layer_mix, comb_res_mix)`

## Invariants

- `base.py` device-neutral immutable reference (fixed `super(Model, self)` → `super().__init__()` for adapter compat, semantically identical).
- candidate output shape/dtype/device/numerical semantics/public contract compatible with base.
- harness AST loader rewrites `cuda` → `gcu`.

## Runtime Fingerprint

```yaml
triton_version: 3.6.0
triton_gcu_version: 3.6.0+1.0.20260722
torch_version: 2.10.0+cpu
torch_gcu_version: 2.10.0+3.8.0.2
backend_target: triton_gcu
device_name: GCU
device_arch: major=3, minor=0
multi_processor_count: 2
```

## Measurement Regime

- harness_sha256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- shape: `B=1,H=12,T=128,D=1024, fp16, einsum matmul + elementwise mix`
- warmup: `50`, repeat: `100`
- primary_metric: `unrounded median wall_time_ms`
- profiler_device_time: `unavailable on GCU exporter`

## Measurement Fingerprint

- base_sha256: `58c67cedac8aac3fe1e35a32833616a80f2c3af74f184698a6338a59497695f5`
- baseline_adapter_sha256: `2c0c121e2e4504e791fee3675ac1cd54d1322059fc537555578a91cabd2a24e6`

## Git Run Identity

- base_branch: `dev`
- base_commit: `e853319`
- run_branch: `kernel-opt/mhc-post-layer-mix-s60`

## Round Overview

| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 000 | Phase 0 | `baseline_adapter.py` | baseline | `base.py` | 4.270324 | unavailable | - | not-applicable | `baseline_adapter.py` |
| 001 | `rounds/decision_001.md` | not-created | aborted | `baseline_adapter.py` | - | - | - | not-applicable (device matmul bound, 1.6% launch) | `baseline_adapter.py` |

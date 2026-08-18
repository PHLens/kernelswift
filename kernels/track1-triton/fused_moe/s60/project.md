# Fused MoE Optimization Project (S60 / GCU)

## Project Identity

- schema_version: 1
- skill_version: 2.0.0
- project_root: `/root/kernelswift/.worktrees/fused-moe-s60/kernels/track1-triton/fused_moe/s60`
- base: `../base.py`
- baseline_adapter: `baseline_adapter.py`
- harness: `/root/kernelswift/.worktrees/fused-moe-s60/auto_bench.py`
- interpreter: `/usr/bin/python3` on the S60 host
- device: `gcu:0`
- implementation_language: `triton`
- implementation_backend: `gcu`
- target_profile: `triton_gcu`

## Semantics

- operator: fused_moe (grouped top-2 mixture-of-experts FFN)
- inputs:
    - hidden_states: `[83, 128]`, float16, on `gcu:0`
    - router_logits: `[83, 8]`, float32, on `gcu:0`
- outputs:
    - out: `[83, 128]`, float16, on `gcu:0`
- mathematical_behavior:
    routing: `scores = softmax(router_logits.float(), dim=-1)`; `topk_weights/topk_ids = topk(scores, 2, dim=-1)`; if renormalize `topk_weights /= topk_weights.sum(-1, keepdim=True)`; `topk_weights = topk_weights.to(float16)`
    dispatch: `x_rep = hidden_states` repeated top_k times -> `[T*2, H]`; per-expert mask on `flat_ids == e`
    per-expert FFN: `gate_up = x_e @ w1[e].T -> [n_e, 2I]`; `gate/up = chunk(gate_up, 2)`; `act = silu(gate)*up`; `out_e = act @ w2[e].T`
    reduction: `expert_out *= flat_w`; `reshape(T, 2, H).sum(dim=1)`
- tolerance_and_tie_rules: `torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True)`; output float16
- public_contract: `ModelNew(nn.Module)` with `__init__(num_experts, top_k, hidden_size, intermediate_size, renormalize=True)` and `forward(hidden_states, router_logits)`; must define `get_init_inputs() -> [8, 2, 128, 64]` and `get_inputs() -> [2 tensors]`

## Invariants

- semantic invariant: state_dict keys must be exactly `{w1, w2}` with shapes `w1 [E,2I,H]=[8,128,128]`, `w2 [E,H,I]=[8,128,64]`, both `nn.Parameter` float32, init `normal_(std=0.02)`. `compare_case` calls `load_state_dict(model.state_dict())` before `.to(device)`; missing/unexpected keys silently corrupt weights (the exception is swallowed).
- semantic invariant: candidate `get_inputs` must return exactly 2 arguments (used only for arg-count check + device detection); actual forward inputs are v0's inputs via `v1_inputs = clone_value(v0_inputs)`, dtype fp16/fp32.
- environment invariant: device literal `'cuda'`/`'npu'` in source is rewritten to `'gcu'` by `_rewrite_device_for_backend` (target == gcu branch).
- environment invariant: `topk_ids` is int64 in torch but GCU casts to int32 (UserWarning); candidate kernel must use int32 indices, never `tl.int64`.
- lifecycle invariant: `load_ks_module` filters the AST to `Import`/`ImportFrom`, `ClassDef`, `FunctionDef`/`AsyncFunctionDef`, and safe-literal `Assign`/`AnnAssign` only; no module-level non-literal state or `__main__` side effects.
- measurement invariant: comparison is v0 vs v1 on same device, interleaved warmup/repeat, primary metric median wall_time_ms; profile uses per-forward calls (147 runtime launches observed in eager baseline).

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
- discovered_at: `2026-08-18T10:34:35Z`
- host: `5d02974bab32`

## Measurement Regime

- harness_path: `/root/kernelswift/.worktrees/fused-moe-s60/auto_bench.py`
- harness_sha256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- shape: `T=83,H=128,E=8,top_k=2,I=64`
- dtype: `fp16 hidden / fp32 router`
- device: `gcu:0`
- warmup: `50`
- repeat: `100`
- timing_order: `interleaved accepted-reference/candidate`
- primary_metric: `unrounded median wall_time_ms`
- profiler_iterations: `50`
- profiler_scopes: `baseline_base,candidate_triton_fused_moe_001`
- profiler_device_time: `unavailable on recorded GCU exporter; runtime_launch_* fields are retained`
- correctness_command: `cd /root/kernelswift/.worktrees/fused-moe-s60 && python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/s60/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback`
- benchmark_command: `cd /root/kernelswift/.worktrees/fused-moe-s60 && python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/s60/triton_fused_moe_001.py --warmup 50 --repeat 100`
- profiler_command: `cd /root/kernelswift/.worktrees/fused-moe-s60 && python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/s60/triton_fused_moe_001.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/fused_moe/s60/log/fused_moe_round_001_forward_50iter.pt.trace.json`

## Measurement Fingerprint

- measurement_fingerprint: `d8f8f6bf8965ab279eb59215a7cc0c6f24f7dd0ad5ea7d8436162336955af6c3`
- base_sha256: `21e758535af58c21684d6e657ae44f91a0248d246eb424b6923a3c180bc32a3d`
- baseline_adapter_sha256: `b939d91f0f85e299a1102bfceb00da0e38c484a81c8d23ec78777fce68a3ee6f`
- fingerprint_command: `sha256(base.py || NUL || auto_bench.py || NUL || canonical JSON settings with sort_keys=True and separators=(',', ':'))`

## Optional Target

- target_mode: `null`
- target_value: `null`
- target_metric: `wall_time_ms`
- source: user
- target_measurement_fingerprint: `null`

## Git Run Identity

- base_branch: `dev`
- base_commit: `e853319`
- run_branch: `kernel-opt/fused-moe-s60`

## Round Overview

| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 000 | Phase 0 | `baseline_adapter.py` | baseline | `base.py` | 5.112406 | unavailable: GCU runtime-launch-only | - | not-applicable | `baseline_adapter.py` |
| 001 | `rounds/decision_001.md` | `triton_fused_moe_001.py` | accepted | `baseline_adapter.py` | 0.498811 | unavailable: GCU runtime-launch-only | +90.52% (10.55x) | confirmed (147→8 launches) | `triton_fused_moe_001.py` |
| 002 | `rounds/decision_002.md` | `triton_fused_moe_002.py` | accepted | `triton_fused_moe_001.py` | 0.390289 | unavailable: GCU runtime-launch-only | +26.57% (1.36x) | confirmed (8→3 launches) | `triton_fused_moe_002.py` |

## Reproduction

```bash
cd /root/kernelswift/.worktrees/fused-moe-s60
python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/s60/baseline_adapter.py --warmup 50 --repeat 100
```

```bash
cd /root/kernelswift/.worktrees/fused-moe-s60
python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/s60/baseline_adapter.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/fused_moe/s60/log/fused_moe_baseline_forward_50iter.pt.trace.json
```

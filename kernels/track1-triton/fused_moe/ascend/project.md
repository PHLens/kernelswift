# Fused MoE Optimization Project

## Project Identity

- schema_version: 1
- skill_version: 2.0.0
- project_root: `/workspace/kernelswift/.worktrees/fused-moe-ascend/kernels/track1-triton/fused_moe/ascend`
- base: `../base.py` (shared reference; torch_mlu import removed for cross-backend portability)
- baseline_adapter: `baseline_adapter.py`
- harness: `/workspace/kernelswift/.worktrees/fused-moe-ascend/auto_bench.py`
- interpreter: `/usr/local/python3.11.15/bin/python3`
- device: `npu:0`
- implementation_language: `triton`
- implementation_backend: `ascend`
- target_profile: `triton_ascend`

## Semantics

- operator: fused mixture-of-experts (softmax router + top-k gating + per-expert feedforward + weighted reduce)
- inputs:
  - `hidden_states`: shape `[83, 128]`, dtype `float16`, contiguous, device `npu:0`
  - `router_logits`: shape `[83, 8]`, dtype `float32`, device `npu:0`
  - parameters `w1`: shape `[8, 128, 128]` = `[E, 2*intermediate, hidden]`, `float16` after `.to(dtype)`; `w2`: shape `[8, 128, 64]` = `[E, hidden, intermediate]`, `float16` after `.to(dtype)`
- outputs: `out`: shape `[83, 128]`, dtype `float16`, device `npu:0`
- mathematical_behavior:
  1. routing: `scores = softmax(router_logits.float(), dim=-1)` -> `[83, 8]` (fp32); `topk_weights, topk_ids = topk(scores, top_k=2, dim=-1)` -> `[83, 2]` each
  2. renormalize: `topk_weights /= topk_weights.sum(-1, keepdim=True)` (since `renormalize=True`), then cast to fp16 -> `[83, 2]`
  3. flatten/dispatch: `flat_ids = topk_ids.view(-1)` `[166]`, `flat_w = topk_weights.view(-1)` `[166]`, `x_rep = hidden_states.unsqueeze(1).expand(-1,2,-1).reshape(-1,128)` `[166, 128]`, where `166 = 83 * 2`
  4. per-expert loop over `e in [0..8)`: `mask = (flat_ids == e)` selects `n_e` rows; for each expert:
     - gate/up GEMM: `gate_up = x_e[n_e,128] @ w1[e].T[128,128]` -> `[n_e, 128]` (contraction dim 128 = hidden)
     - `gate, up = gate_up.chunk(2, dim=-1)` -> each `[n_e, 64]`
     - `act = silu(gate) * up` -> `[n_e, 64]`
     - down GEMM: `expert_out[mask] = act[n_e,64] @ w2[e].T[64,128]` -> `[n_e, 128]` (contraction dim 64 = intermediate)
  5. weighted reduce: `expert_out = expert_out[166,128] * flat_w.unsqueeze(-1)[166,1]`, then `out = expert_out.view(83,2,128).sum(dim=1)` -> `[83, 128]` (sum of the top-2 weighted expert outputs)
- public_contract: `ModelNew(num_experts=8, top_k=2, hidden_size=128, intermediate_size=64, renormalize=True)` with `forward(hidden_states, router_logits)`; also exposes module-level `get_init_inputs() -> [8, 2, 128, 64]` and `get_inputs() -> [hidden_states, router_logits]`
- tolerance_and_tie_rules: harness `compare_values` uses `torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True)`; output fp16 with fp32 routing then fp16 cast of weights; top-k ties resolved by `torch.topk` default (largest values, ties broken by index order)

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
device_name: Ascend910B4
cube_core_num: 20
vector_core_num: 40
total_memory_bytes: 31662800896
L2_cache_size: 100663296
```

- target_profile_match: `pass`
- host: `ascend910b4`

## Measurement Regime

- harness_path: `/workspace/kernelswift/.worktrees/fused-moe-ascend/auto_bench.py`
- harness_sha256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- shape: `T=83,H=128,E=8,top_k=2,intermediate=64`
- dtype: `hidden_states=fp16,router_logits=fp32,out=fp16`
- device: `npu:0`
- warmup: `50`
- repeat: `100`
- primary_metric: `unrounded median wall_time_ms`
- profiler_iterations: `50`

## Measurement Fingerprint

- measurement_fingerprint: `47e60b0db91c4c67e55f92cf79f5dddf591925620ec4db38704dfb42f0f185dd`
- base_sha256: `a0269ac15833098c11c63045318e829e404b09ca49ae8dce22b244e0d2894d2b`
- baseline_adapter_sha256: `a7fc0001db3ee9e636241954d2c071b62acee518b23f4c59c19efee886203a02`

## Optional Target

- target_mode: `null`
- target_value: `null`
- target_metric: `wall_time_ms`
- source: user

## Git Run Identity

- base_branch: `dev`
- base_commit: `e321768a7981f7ce278f96a3a88dd0b41e5ef704`
- run_branch: `kernel-opt/fused-moe-ascend`

## Round Overview

| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 000 | Phase 0 | `baseline_adapter.py` | baseline | `base.py` | 7.158795 | 743.948 | - | not-applicable | `baseline_adapter.py` |
| 001 | `rounds/decision_001.md` | `triton_fused_moe_001.py` | accepted | `baseline_adapter.py` | 0.569590 | 97.366 | +92.71% | confirmed | `triton_fused_moe_001.py` |
| 002 | `rounds/decision_002.md` | `triton_fused_moe_002.py` | accepted | `triton_fused_moe_001.py` | 0.368980 | 26.678 | +35.94% | confirmed | `triton_fused_moe_002.py` |
| 003 | `rounds/decision_003.md` | `triton_fused_moe_003.py` | accepted | `triton_fused_moe_002.py` | 0.373490 | 26.622 | +6.70% | confirmed | `triton_fused_moe_003.py` |
| 004 | `rounds/decision_004.md` | - | aborted | `triton_fused_moe_003.py` | - | - | - | not-applicable | `triton_fused_moe_003.py` |

## Reproduction

```bash
cd /workspace/kernelswift/.worktrees/fused-moe-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/ascend/baseline_adapter.py --warmup 50 --repeat 100
```

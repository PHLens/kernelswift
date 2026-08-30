# Grouped TopK Optimization Project

## Project Identity

- schema_version: 1
- skill_version: 2.0.0
- project_root: `/workspace/kernelswift/.worktrees/groupedtopk-ascend/kernels/track1-triton/groupedtopk/ascend`
- base: `../base.py` (shared device-neutral reference at the operator level)
- baseline_adapter: `baseline_adapter.py`
- harness: `/workspace/kernelswift/.worktrees/groupedtopk-ascend/auto_bench.py`
- interpreter: `/usr/local/python3.11.15/bin/python3`
- device: `npu:0`
- implementation_language: `triton`
- implementation_backend: `ascend`
- target_profile: `triton_ascend`

## Semantics

- operator: grouped top-k router selection
- inputs: `hidden_states[83,7168] fp16` and `gating_output[83,256] fp32`, both on `npu:0`; hidden states are used only for the batch-size assertion
- outputs: `topk_weights[83,8] fp32` and `topk_ids[83,8] int32`, on `npu:0`
- mathematical_behavior: apply softmax over experts, select the top 4 expert groups by per-group maximum, mask other experts, select top 8 experts, renormalize selected weights, then apply routed scaling
- tolerance_and_tie_rules: `torch.allclose(atol=1e-2, rtol=1e-2)` for floating outputs and exact equality for integer outputs; preserve PyTorch top-k ordering and tie behavior for the recorded regime
- public_contract: `ModelNew(topk, renormalize, num_expert_group, topk_group, scoring_func="softmax", routed_scaling_factor=1.0)` with `forward(hidden_states, gating_output)`

## Invariants

- `base.py` is the immutable device-neutral reference; its bytes are unchanged after baseline adapter generation.
- Candidate output shapes, dtypes, device placement, numerical semantics, and public constructor/forward contract remain compatible with `base.py`.
- The harness is loaded through its AST loader; direct import success is insufficient.
- Wall time is measured by the unchanged harness with seed setup and NPU synchronization included.
- Reference and candidate profiler scopes are separate. NPU runtime launch duration is diagnostic only and is not device kernel duration.

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
total_memory_MB: 30196
L2_cache_size: 100663296
```

- target_profile_match: `pass`
- discovery_commands: `python3 --version`; `python3 -c 'import torch, torch_npu, triton; ...'`; `python3 -c 'import triton.backends; ...'`
- discovered_at: `2026-08-18T00:00:00Z`
- host: `ascend910b4`

## Measurement Regime

- harness_path: `/workspace/kernelswift/.worktrees/groupedtopk-ascend/auto_bench.py`
- harness_sha256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- shape: `T=83,E=256,hidden=7168,topk=8,num_expert_group=8,topk_group=4`
- dtype: `hidden_states=fp16,gating_output=fp32,weights=fp32,ids=int32`
- device: `npu:0`
- warmup: `50`
- repeat: `100`
- timing_order: `ordered reference/candidate pairs; each pair uses the unchanged harness`
- primary_metric: `unrounded median wall_time_ms`
- profiler_iterations: `50`
- profiler_scopes: `baseline_base,candidate_triton_grouped_topk_001`
- correctness_command: `cd /workspace/kernelswift/.worktrees/groupedtopk-ascend && python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/ascend/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback`
- benchmark_command: `cd /workspace/kernelswift/.worktrees/groupedtopk-ascend && python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/ascend/triton_grouped_topk_001.py --warmup 50 --repeat 100`
- profiler_command: `cd /workspace/kernelswift/.worktrees/groupedtopk-ascend && python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/ascend/triton_grouped_topk_001.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output log/groupedtopk_round_001_forward_50iter.pt.trace.json`

## Measurement Fingerprint

- measurement_fingerprint: `d2dc2d5a61930039371da06149b3156c4911a136c6c5df859f50d68ea0e3b871`
- base_sha256: `12f3324896d6b72bd4eac556839db53fb7045b2965b7f8caf2734551daad0f58`
- baseline_adapter_sha256: `3eda2738d12ed93f4718bf67eca276e1bbc09eb4e3f8fac6b724b5c9e4981134`
- fingerprint_command: `sha256(base.py || NUL || auto_bench.py || NUL || canonical JSON settings with sort_keys=True and separators=(',', ':'))`

## Optional Target

- target_mode: `null`
- target_value: `null`
- target_metric: `wall_time_ms`
- source: user
- target_measurement_fingerprint: `null`

## Git Run Identity

- base_branch: `dev`
- base_commit: `99814d5`
- run_branch: `kernel-opt/groupedtopk-ascend`

## Round Overview

| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 000 | Phase 0 | `baseline_adapter.py` | baseline | `base.py` | 0.760135 | 172.835 | - | not-applicable | `baseline_adapter.py` |
| 001 | `rounds/decision_001.md` | `triton_grouped_topk_001.py` | accepted | `baseline_adapter.py` | 0.321620 | 34.634 | +54.88% | confirmed | `triton_grouped_topk_001.py` |
| 002 | `rounds/decision_002.md` | `triton_grouped_topk_002.py` | accepted | `triton_grouped_topk_001.py` | 0.267220 | 35.134 | +18.21% | confirmed | `triton_grouped_topk_002.py` |
| 003 | `rounds/decision_003.md` | - | abort | `triton_grouped_topk_002.py` | - | - | - | not-applicable | `triton_grouped_topk_002.py` |

## Reproduction

```bash
cd /workspace/kernelswift/.worktrees/groupedtopk-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/ascend/baseline_adapter.py --warmup 50 --repeat 100
```

```bash
cd /workspace/kernelswift/.worktrees/groupedtopk-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/ascend/triton_grouped_topk_001.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output log/groupedtopk_round_001_forward_50iter.pt.trace.json
```

The remote working directory is the worktree root; the shared `base.py` lives at
`kernels/track1-triton/groupedtopk/base.py`.

# Grouped TopK Optimization Project

## Project Identity

- schema_version: 1
- skill_version: 2.0.0
- project_root: `/home/cambricon/kernelswift/s60/groupedtopk`
- base: `base.py` (historical: recorded at campaign root during the run;
  after the 2026-08 layout restructure the shared reference lives at
  `../base.py`; hashes below correspond to the pre-restructure S60-adapted copy)
- baseline_adapter: `baseline_adapter.py`
- harness: `/home/cambricon/kernelswift/auto_bench.py`
- interpreter: `/usr/bin/python3` on the S60 host
- device: `gcu:0`
- implementation_language: `triton`
- implementation_backend: `gcu`
- target_profile: `triton_gcu`

## Semantics

- operator: grouped top-k router selection
- inputs: `hidden_states[83,7168] fp16` and `gating_output[83,256] fp32`, both on `gcu:0`; hidden states are used only for the batch-size assertion
- outputs: `topk_weights[83,8] fp32` and `topk_ids[83,8] int32`, on `gcu:0`
- mathematical_behavior: apply softmax over experts, select the top 4 expert groups by per-group maximum, mask other experts, select top 8 experts, renormalize selected weights, then apply routed scaling
- tolerance_and_tie_rules: `torch.allclose(atol=1e-2, rtol=1e-2)` for floating outputs and exact equality for integer outputs; preserve PyTorch top-k ordering and tie behavior for the recorded regime
- public_contract: `ModelNew(topk, renormalize, num_expert_group, topk_group, scoring_func="softmax", routed_scaling_factor=1.0)` with `forward(hidden_states, gating_output)`

## Invariants

- `base.py` is the GCU-adapted immutable reference; its bytes are unchanged after baseline adapter generation.
- The MLU source reference remains in `/home/cambricon/kernelswift/mlu/groupedtopk/base.py` and is not used as a cross-backend comparison.
- Candidate output shapes, dtypes, device placement, numerical semantics, and public constructor/forward contract remain compatible with `base.py`.
- The harness is loaded through its AST loader; direct import success is insufficient.
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
sip_count: 24
total_memory: 41846MB
```

- target_profile_match: `pass`
- discovery_commands: `python3 --version`; `python3 -c 'import torch, torch_gcu; ...'`; `python3 -c 'import triton_gcu; ...'`
- discovered_at: `2026-08-17T08:52:03Z`
- host: `5d02974bab32`

## Measurement Regime

- harness_path: `/home/cambricon/kernelswift/auto_bench.py`
- harness_sha256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- shape: `T=83,E=256,hidden=7168,topk=8,num_expert_group=8,topk_group=4`
- dtype: `hidden_states=fp16,gating_output=fp32,weights=fp32,ids=int32`
- device: `gcu:0`
- warmup: `50`
- repeat: `100`
- timing_order: `ordered reference/candidate pairs; each pair uses the unchanged harness`
- primary_metric: `unrounded median wall_time_ms`
- profiler_iterations: `50`
- profiler_scopes: `baseline_base,candidate_triton_grouped_topk_001; baseline_reference_triton_grouped_topk_001,candidate_triton_grouped_topk_002`
- profiler_device_time: `unavailable on recorded GCU exporter; runtime_launch_* fields are retained`
- correctness_command: `cd /root/kernelswift-s60 && python3 auto_bench.py --v0_file base.py --v1_file baseline_adapter.py --warmup 5 --repeat 10 --full-traceback`
- benchmark_command: `cd /root/kernelswift-s60 && python3 auto_bench.py --v0_file base.py --v1_file triton_grouped_topk_001.py --warmup 50 --repeat 100`
- profiler_command: `cd /root/kernelswift-s60 && python3 auto_bench.py --v0_file base.py --v1_file triton_grouped_topk_001.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output log/groupedtopk_round_001_forward_50iter.pt.trace.json`

## Measurement Fingerprint

- measurement_fingerprint: `3942e25aebbe7690a55cf27768a3bc3fd552cc8106f6bd2dd7416cea2d274bf3`
- base_sha256: `a5b37db46753a7458802c87bd7996ca9fd073795c914178d3e1298ccfb6aea0f`
- baseline_adapter_sha256: `6713aa567c945e98628f5b3c58d2bf5d71c3df85af8ad19438c00a447890fdd1`
- fingerprint_command: `sha256(base.py || NUL || auto_bench.py || NUL || canonical JSON settings with sort_keys=True and separators=(',', ':'))`

## Optional Target

- target_mode: `null`
- target_value: `null`
- target_metric: `wall_time_ms`
- source: user
- target_measurement_fingerprint: `null`

## Git Run Identity

- base_branch: `dev`
- base_commit: `6a970c9`
- run_branch: `kernel-opt/groupedtopk-s60-continue`

## Round Overview

| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 000 | Phase 0 | `baseline_adapter.py` | baseline | `base.py` | 0.459285 | unavailable: GCU runtime-launch-only | - | not-applicable | `baseline_adapter.py` |
| 001 | `rounds/decision_001.md` | `triton_grouped_topk_001.py` | accepted | `baseline_adapter.py` | 0.273881 | unavailable: GCU runtime-launch-only | 39.0869% | confirmed | `triton_grouped_topk_001.py` |
| 002 | `rounds/decision_002.md` | `triton_grouped_topk_002.py` | accepted | `reference_triton_grouped_topk_001.py` | 0.274740 | unavailable: GCU runtime-launch-only | 9.0214% | confirmed | `triton_grouped_topk_002.py` |
| 003 | `rounds/decision_003.md` | `triton_grouped_topk_003.py` | accepted | `reference_triton_grouped_topk_002.py` | 0.273673 | unavailable: GCU runtime-launch-only | 6.4647% | confirmed | `triton_grouped_topk_003.py` |
| 004 | `rounds/decision_004.md` | `triton_grouped_topk_004.py` | no-improvement | `reference_triton_grouped_topk_003.py` | 0.271659 | unavailable: GCU runtime-launch-only | 2.05898% | not-confirmed | `triton_grouped_topk_003.py` |

## Reproduction

```bash
cd /root/kernelswift-s60
python3 auto_bench.py --v0_file base.py --v1_file baseline_adapter.py --warmup 50 --repeat 100
python3 auto_bench.py --v0_file base.py --v1_file triton_grouped_topk_001.py --warmup 50 --repeat 100
```

```bash
cd /root/kernelswift-s60
python3 auto_bench.py --v0_file base.py --v1_file triton_grouped_topk_001.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output log/groupedtopk_round_001_forward_50iter.pt.trace.json
```

```bash
cd /root/kernelswift-s60
python3 auto_bench.py --v0_file reference_triton_grouped_topk_001.py --v1_file triton_grouped_topk_002.py --warmup 50 --repeat 100
python3 auto_bench.py --v0_file reference_triton_grouped_topk_001.py --v1_file triton_grouped_topk_002.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output log/groupedtopk_round_002_forward_50iter.pt.trace.json
```

```bash
cd /root/kernelswift-s60
python3 auto_bench.py --v0_file reference_triton_grouped_topk_002.py --v1_file triton_grouped_topk_003.py --warmup 50 --repeat 100
python3 auto_bench.py --v0_file reference_triton_grouped_topk_002.py --v1_file triton_grouped_topk_003.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output log/groupedtopk_round_003_forward_50iter.pt.trace.json
```

```bash
cd /root/kernelswift-s60
python3 auto_bench.py --v0_file reference_triton_grouped_topk_003.py --v1_file triton_grouped_topk_004.py --warmup 50 --repeat 100
python3 auto_bench.py --v0_file reference_triton_grouped_topk_003.py --v1_file triton_grouped_topk_004.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output log/groupedtopk_round_004_forward_50iter.pt.trace.json
```

The remote working directory is `/root/kernelswift-s60`; the local project and
harness are the source-controlled copies in this repository.

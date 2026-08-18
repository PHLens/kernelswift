# Grouped TopK Optimization Project

## Project Identity

- schema_version: 1
- skill_version: 2.0.0
- project_root: `/root/kernelswift-bi150/bi150/groupedtopk`
- base: `base.py`
- baseline_adapter: `baseline_adapter.py`
- harness: `/root/kernelswift-bi150/auto_bench.py`
- interpreter: `/usr/local/bin/python3`
- device: `cuda:0 (Iluvatar BI-V150)`
- implementation_language: `triton`
- implementation_backend: `cuda`
- target_profile: `triton_cuda`

## Semantics

- operator: grouped top-k router selection
- inputs: `hidden_states[83,7168] fp16` and `gating_output[83,256] fp32`, both on `cuda:0`; hidden states are used only for the batch-size assertion
- outputs: `topk_weights[83,8] fp32` and `topk_ids[83,8] int32`, on `cuda:0`
- mathematical_behavior: apply softmax over experts, select the top 4 expert groups by per-group maximum, mask other experts, select top 8 experts, renormalize selected weights, then apply routed scaling
- tolerance_and_tie_rules: `torch.allclose(atol=1e-2, rtol=1e-2)` for floating outputs and exact equality for integer outputs; preserve PyTorch top-k ordering and tie behavior for the recorded regime
- public_contract: `ModelNew(topk, renormalize, num_expert_group, topk_group, scoring_func="softmax", routed_scaling_factor=1.0)` with `forward(hidden_states, gating_output)`

## Invariants

- `base.py` is the BI150-adapted immutable reference; its bytes are unchanged after baseline adapter generation.
- The harness is loaded through its AST loader; direct import success is insufficient.
- The harness detects and moves inputs/models to the selected `cuda:0` device; reference and candidate use the same input values.
- The CoreX bootstrap is required before importing `torch`, `triton`, or using `ixsmi`: `export COREX_VERSION=4.4.0; . /usr/local/corex/enable`.
- Candidate output shapes, dtypes, device placement, mathematical semantics, and public constructor/forward contract remain compatible with `base.py`.
- Wall time is measured by the unchanged harness with seed setup and CUDA synchronization included.
- Reference and candidate profiler scopes are separate. The BI150 trace exposes `cat=kernel` device durations and CUDA runtime events; device time is normalized per forward call.

## Runtime Fingerprint

```yaml
triton_distribution: triton 3.1.0 (/usr/local/corex-4.4.0/lib64/python3/dist-packages/triton)
triton_version: 3.1.0
torch_version: 2.7.1
torch_cuda_version: 10.2
backend_target: Iluvatar CoreX 4.4.0 CUDA-compatible runtime
backend_version: CoreX 4.4.0; nvcc V10.2.89; torch 2.7.1
device_arch: Iluvatar BI-V150 capability major=7, minor=1
multi_processor_count: 16
total_memory: 17179869184
```

- target_profile_match: `pass`
- discovery_commands: `export COREX_VERSION=4.4.0; . /usr/local/corex/enable`; `python3 --version`; `python3 -c 'import torch,triton; print(torch.__version__, triton.__version__)'`; `python3 -c 'import torch; print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0)); print(torch.cuda.get_device_properties(0).multi_processor_count); print(torch.cuda.get_device_properties(0).total_memory)'`; `nvcc --version`; `ixsmi -L`
- discovered_at: `2026-08-18T05:02:14Z`
- host: `saas-de-pjsys2-pjsys2-bi150-44-33c5-6dbcd9c4b8-2rwgs`

These values are observed in Phase 0. They are not assumed from the profile.

## Measurement Regime

- harness_path: `/root/kernelswift-bi150/auto_bench.py`
- harness_sha256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- shape: `hidden_states=[83,7168] fp16; gating_output=[83,256] fp32; outputs=[83,8] fp32+int32`
- dtype: `fp16 (hidden_states), fp32 (gating_output, topk_weights), int32 (topk_ids)`
- device: `cuda:0 (Iluvatar BI-V150)`
- warmup: `50`
- repeat: `100`
- timing_order: `ordered reference/candidate pairs; each pair uses the unchanged harness`
- primary_metric: `unrounded median wall_time_ms`
- profiler_iterations: `50`
- profiler_scopes: `baseline_base,candidate_baseline_adapter`
- correctness_command: `cd /root/kernelswift-bi150 && export COREX_VERSION=4.4.0 && . /usr/local/corex/enable && python3 auto_bench.py --v0_file bi150/groupedtopk/base.py --v1_file bi150/groupedtopk/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback`
- benchmark_command: `cd /root/kernelswift-bi150 && export COREX_VERSION=4.4.0 && . /usr/local/corex/enable && python3 auto_bench.py --v0_file bi150/groupedtopk/base.py --v1_file bi150/groupedtopk/baseline_adapter.py --warmup 50 --repeat 100`
- profiler_command: `cd /root/kernelswift-bi150 && export COREX_VERSION=4.4.0 && . /usr/local/corex/enable && python3 auto_bench.py --v0_file bi150/groupedtopk/base.py --v1_file bi150/groupedtopk/baseline_adapter.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output bi150/groupedtopk/log/groupedtopk_baseline_forward_50iter.pt.trace.json`

Benchmark wall time controls adoption. Profiler data is attributable diagnostic evidence and is normalized per forward call.

## Measurement Fingerprint

- measurement_fingerprint: `57bf01d317ee03ca2b09730e648f0f93d2bf4f226639ca3af2b1ff57b2865575`
- base_sha256: `d57ace7d9196e2e44bdcfd17d1738482e7fd1bbb2d86fc6c9449c43938953eb5`
- baseline_adapter_sha256: `689d458c7abe07323508fc054bfef609dc4bd1cd9c94e3bb706d6f2d2cd00016`
- fingerprint_command: `python3 -c "import hashlib,json; base=open('bi150/groupedtopk/base.py','rb').read(); harness=open('auto_bench.py','rb').read(); settings={'shape':'hidden_states=[83,7168] fp16; gating_output=[83,256] fp32; outputs=[83,8] fp32+int32','dtype':'fp16 (hidden_states), fp32 (gating_output, topk_weights), int32 (topk_ids)','device':'cuda:0 (Iluvatar BI-V150)','warmup':50,'repeat':100,'profile_mode':'forward','profile_warmup':20,'profile_iterations':50}; payload=json.dumps(settings,sort_keys=True,separators=(',',':')).encode(); print(hashlib.sha256(base+b'\\x00'+harness+b'\\x00'+payload).hexdigest())"`

A fingerprint change requires a new comparable baseline before optimization can continue.

## Optional Target

- target_mode: `null`
- target_value: `null`
- target_metric: `wall_time_ms`
- source: `null`
- target_measurement_fingerprint: `null`

## Git Run Identity

- base_branch: `dev`
- base_commit: `6a970c921dfb0c031b885190122ce1335d8d4cd7`
- run_branch: `kernel-opt/bi150-prepare-20260818`

These fields mirror `team-state.md` and identify the dedicated optimization branch. The run branch is never `main`, `master`, or `dev`.

## Round Overview

| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 000 | Phase 0 | `baseline_adapter.py` | baseline | `base.py` | 0.474995 | 179.0703515625 | - | not-applicable | `baseline_adapter.py` |
| 001 | `rounds/decision_001.md` | not-created | aborted | `baseline_adapter.py` | - | - | - | capability-miss | `baseline_adapter.py` |
| 002 | `rounds/decision_002.md` | `triton_grouped_topk_002.py` | candidate-failed | `baseline_adapter.py` | - | - | - | topk-tie-ordering-active-set-mismatch | `baseline_adapter.py` |
| 003 | `rounds/decision_003.md` | not-created | design-rejected | `baseline_adapter.py` | - | - | - | post-selection-mask-requires-second-stage | `baseline_adapter.py` |

## Reproduction

```bash
cd /root/kernelswift-bi150
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
python3 auto_bench.py --v0_file bi150/groupedtopk/base.py --v1_file bi150/groupedtopk/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback
python3 auto_bench.py --v0_file bi150/groupedtopk/base.py --v1_file bi150/groupedtopk/baseline_adapter.py --warmup 50 --repeat 100
```

```bash
cd /root/kernelswift-bi150
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
python3 auto_bench.py --v0_file bi150/groupedtopk/base.py --v1_file bi150/groupedtopk/baseline_adapter.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output bi150/groupedtopk/log/groupedtopk_baseline_forward_50iter.pt.trace.json
python3 skills/kernel-opt-loop/scripts/summarize_trace.py bi150/groupedtopk/log/groupedtopk_baseline_forward_50iter.pt.trace.json --iterations 50 --scope baseline_base --wall-ms 0.474612
python3 skills/kernel-opt-loop/scripts/summarize_trace.py bi150/groupedtopk/log/groupedtopk_baseline_forward_50iter.pt.trace.json --iterations 50 --scope candidate_baseline_adapter --wall-ms 0.474995
```

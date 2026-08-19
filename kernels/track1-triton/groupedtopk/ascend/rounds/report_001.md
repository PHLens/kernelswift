# Report 001

Result: accepted

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md`
- Candidate: `triton_grouped_topk_001.py`
- Accepted reference: `baseline_adapter.py`
- Accepted reference report: `rounds/report_000.md`
- Decision SHA256: `e57e3fb560f7d8b39ec1b1a90be80a144a59564415813d3a783758c1351ea344`
- Candidate SHA256: `b7b47d1fec7eaed59eba784dd3300393df12bdc94cab164b9e9d238afb39357a`
- Accepted reference SHA256: `3eda2738d12ed93f4718bf67eca276e1bbc09eb4e3f8fac6b724b5c9e4981134`
- Base SHA256: `12f3324896d6b72bd4eac556839db53fb7045b2965b7f8caf2734551daad0f58`
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `d2dc2d5a61930039371da06149b3156c4911a136c6c5df859f50d68ea0e3b871`
- verification_tier: `authoritative`
- screening_pairs: `not-run; formal ordered pairs completed after correctness`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | pass | Candidate matched reference outputs with `atol=1e-2, rtol=1e-2`. | pass | auto_bench `PASS accuracy` |
| output shape | `[83,8]` weights and ids | Candidate returned `[83,8]` for both outputs. | pass | auto_bench compare_values |
| output dtype | weights fp32, ids int32 | Candidate returned fp32 weights and int32 ids. | pass | auto_bench compare_values |
| target/device | preserve NPU execution | Candidate ran on `npu:0` with direct Triton-Ascend launch. | pass | compile smoke and formal benchmark |
| launcher conformance | direct launch is the selected Ascend path | Direct Triton launch compiled and ran. | pass | candidate smoke |

## Screening Evidence

Screening was not run as a separate two-pair stage; the formal ordered
reference/candidate benchmark after correctness was used for the adoption
decision.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `reference, candidate` ordered pair; unchanged harness command
- reference_raw_samples_ms: `not collected individually (harness reports median)`
- candidate_raw_samples_ms: `not collected individually (harness reports median)`
- reference_median_ms: `0.712855`
- candidate_median_ms: `0.321620`
- improvement_pct: `54.88475414304643`

```text
improvement_pct = (0.712855 - 0.321620) / 0.712855 * 100
                = 54.88475414304643
```

The unrounded median clears the 5% adoption threshold.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| kernel_count_per_call | decrease from 19.0 toward 1.0 | `19.0` reference to `1.0` candidate | pass | `summarize_cann_trace.py` per-scope CANN sqlite |
| device_us_per_call | decrease from 172.835 | `172.835` reference to `34.634` candidate | pass | same per-scope summary |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-001`
- intervention: `fuse grouped softmax routing and masked top-k selection into one direct Triton-Ascend kernel`
- expected_causal_chain: `separate NPU kernels for softmax, group selection, masking, and top-k disappear -> kernel_count_per_call decreases -> device_us_per_call decreases -> benchmark wall time decreases`
- primary_metric: `wall_time`
- Hypothesis verdict: `confirmed`

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `available` (CANN msprof per-scope captures via torch_npu.profiler)
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`

Reference and candidate are captured in separate CANN msprof runs (one
`ASCEND_WORK_PATH` per scope), so each `ai_core_op_summary.db` belongs to
exactly one scope.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference (`baseline_base`) | 8641.74 | 172.835 | 950 | 19.0 | 0.712855 | 0.242 |
| candidate | 1731.72 | 34.634 | 50 | 1.0 | 0.321620 | 0.108 |

```text
device_ratio = device_us_per_call / (wall_ms * 1000)
```

### Accepted Reference Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| aclnnTopk_TopkV2AiCore_TopKV2 | 150 | 3.0 | 2749.90 | 54.998 |
| aclnnInplaceScatterValue_ScatterElementsNoTranspose_ScatterElementsV2 | 50 | 1.0 | 1237.02 | 24.740 |
| aclnnTopk_GatherElements_GatherElements | 50 | 1.0 | 994.74 | 19.895 |
| aclnnSoftmax_SoftmaxAiCore_SoftmaxV2 | 50 | 1.0 | 665.64 | 13.313 |
| aclnnDiv_RealDivAiCore_RealDiv | 50 | 1.0 | 441.52 | 8.830 |

### Candidate Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| _grouped_topk_kernel | 50 | 1.0 | 1731.72 | 34.634 |

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable | `b7b47d1fec7eaed59eba784dd3300393df12bdc94cab164b9e9d238afb39357a` | correctness, smoke, timing, and profile completed |

No Verifier-to-Coder repair was required.

## evidence_for_next_round

- Direct kernel fusion reduces NPU kernels from 19.0 to 1.0 per forward call on
  the exact `T=83,E=256` regime.
- Device time drops from 172.835 us/call to 34.634 us/call (5.0x reduction).
- Wall time improves from 0.712855 ms to 0.321620 ms at the unrounded median
  (+54.88%), clearing the 5% threshold.
- Device ratio drops from 0.242 to 0.108, meaning the candidate is now
  host-bound (host dispatch dominates the remaining ~89% of wall time); the
  single kernel's device time is only 34.6 us.
- The candidate still allocates two output tensors per forward call;
  allocator lifecycle was intentionally outside this decision.

## Stop Recommendation

- recommendation: `continue`
- evidence: Round 001 accepted; remaining wall time is host-bound, suggesting a
  further host-side or multi-token-parallelization opportunity.

## Exact Reproduction Commands

```bash
cd /workspace/kernelswift/.worktrees/groupedtopk-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/ascend/triton_grouped_topk_001.py --warmup 50 --repeat 100
```

```bash
cd /workspace/kernelswift/.worktrees/groupedtopk-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/ascend/triton_grouped_topk_001.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/groupedtopk/ascend/log/groupedtopk_round_001_forward_50iter.pt.trace.json
python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py kernels/track1-triton/groupedtopk/ascend/log/profiling_data/candidate_triton_grouped_topk_001/profiling_data --iterations 50 --scope candidate_triton_grouped_topk_001 --wall-ms 0.321620
```

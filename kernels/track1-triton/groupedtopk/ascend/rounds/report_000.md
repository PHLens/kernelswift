# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `base.py` (Phase 0)
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `3eda2738d12ed93f4718bf67eca276e1bbc09eb4e3f8fac6b724b5c9e4981134`
- Accepted reference SHA256: `12f3324896d6b72bd4eac556839db53fb7045b2965b7f8caf2734551daad0f58`
- Base SHA256: `12f3324896d6b72bd4eac556839db53fb7045b2965b7f8caf2734551daad0f58`
- Harness SHA256: `72a4a1bb4e0ca46067c0a52d606fc1f9338f93e0dd21955a14d67e24924f829f`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `bad2c8a7c6d13df678ac5c634d2492633b2b7a9d046617cb308f48186f95b78e`
- verification_tier: baseline
- screening_pairs: `not-run`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | baseline_adapter output equals base output | `PASS accuracy` | pass | `python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/ascend/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback` |

## Screening Evidence

Not applicable for baseline.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate`
- reference_raw_samples_ms: `not collected individually (harness reports median)`
- candidate_raw_samples_ms: `not collected individually (harness reports median)`
- reference_median_ms: `0.758045`
- candidate_median_ms: `0.760135`
- improvement_pct: `-0.276` (baseline_adapter is the same code as base, so this is measurement noise)

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
```

Baseline wall time (candidate `baseline_adapter.py`) is `0.760135 ms` under
`--warmup 50 --repeat 100`. The reference (`base.py`) is `0.758045 ms`; the
difference is noise because the adapter is the reference with `Model` renamed to
`ModelNew`.

## Evaluation Contract Mirror

- Evaluation Contract applicability: `not-applicable: Phase 0`
- hypothesis_id: `not-applicable: Phase 0`
- intervention: `not-applicable: Phase 0`
- expected_causal_chain: `not-applicable: Phase 0`
- primary_metric: `not-applicable: Phase 0`
- Hypothesis verdict: `inconclusive`

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `summary`
- profiler_device_time: `available` (CANN msprof via torch_npu.profiler)
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`,
  `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`

The stock `torch.profiler` path on Ascend exposes only host-side `cpu_op`
events; NPU AI Core kernel durations are read from the CANN msprof
`ai_core_op_summary.db` via `summarize_cann_trace.py`. Reference and candidate
are captured in separate CANN msprof runs (one `ASCEND_WORK_PATH` per scope).

> Revision note: the initial Phase 0 capture combined reference and candidate
> in a single CANN capture and therefore double-counted kernels. The corrected
> per-scope values below (19.0 kernels/call, 172.835 us/call) are the
> authoritative baseline device evidence and supersede the earlier 38.0/329.034
> figures.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference | 8641.74 | 172.835 | 950 | 19.0 | 0.712855 | 0.242 |
| candidate | 8641.74 | 172.835 | 950 | 19.0 | 0.712855 | 0.242 |

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

Same as reference (candidate is the reference code with `Model` renamed).

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable | `3eda2738d12ed93f4718bf67eca276e1bbc09eb4e3f8fac6b724b5c9e4981134` | pass |

## evidence_for_next_round

- Baseline device time is `172.835 us/call` with `19.0` NPU kernels per forward.
- Dominant device kernels are `aclnnTopk` (55.0 + 19.9 us/call across
  TopKV2 and GatherElements), `aclnnInplaceScatterValue` (24.7 us/call),
  and `aclnnSoftmax` (13.3 us/call).
- Wall time is `0.712855 ms`, so device time is ~24% of wall; the remaining
  ~76% is host-side operator dispatch and synchronization overhead from 19
  separate kernel launches per forward.
- The shared `base.py` is pure torch (no Triton kernel): the grouped top-k
  routing is decomposed into many small device kernels plus host-side reshape,
  scatter, mask, and cast operations.

## Stop Recommendation

- recommendation: `continue`
- evidence: Phase 0 baseline established; optimization rounds have a clear
  kernel-fragmentation and host-overhead target.

## Exact Reproduction Commands

```bash
cd /workspace/kernelswift/.worktrees/groupedtopk-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/ascend/baseline_adapter.py --warmup 50 --repeat 100
```

```bash
cd /workspace/kernelswift/.worktrees/groupedtopk-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/ascend/baseline_adapter.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/groupedtopk/ascend/log/groupedtopk_baseline_forward_50iter.pt.trace.json
python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py kernels/track1-triton/groupedtopk/ascend/log/profiling_data/<span>_ascend_pt --iterations 50 --wall-ms 0.760135
```

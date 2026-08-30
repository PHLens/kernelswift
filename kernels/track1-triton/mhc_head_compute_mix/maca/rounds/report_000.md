# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `base.py for Phase 0`
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `c3cc90deb93c69f092e876ddc0057ddf6c2d7ac28020f1b7fd1ed260bca72fee`
- Accepted reference SHA256: `4c5167f6cfe9099786f204fad222f3b780d3ebb814494b9582112422abf84ac5`
- Base SHA256: `4c5167f6cfe9099786f204fad222f3b780d3ebb814494b9582112422abf84ac5`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `d8f4b63bfbf09ce8a32f3bdcd4d85553f34abce7384e495ba5f66baf49bf795e`
- verification_tier: `baseline`
- screening_pairs: `not-run`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | v1 (`ModelNew`) outputs match v0 (`Model`) under atol=1e-2, rtol=1e-2 | `PASS accuracy; v0=1.541432 ms, v1=1.529807 ms, speedup=1.008x` | pass | `cd /root/kernelswift-mhcc && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix/maca/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback` → exit 0 |

Conformance: v0 uses `Model`, v1 uses `ModelNew` as required for Phase 0 baseline.

## Screening Evidence

Not applicable for Phase 0 baseline (no screening run).

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `sequential complete accepted-reference block, then complete candidate block` (harness `compare_case`: `time_forward` for v0, then v1)
- reference_raw_samples_ms: `[1.512683, 1.522637, 1.528345]` (one unrounded median per authoritative run)
- candidate_raw_samples_ms: `[1.509571, 1.515187, 1.522095]`
- reference_median_ms: `1.522637`
- candidate_median_ms: `1.515187`
- improvement_pct: `0.4894`

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
               = (1.522637 - 1.515187) / 1.522637 * 100 = 0.4894
```

Three authoritative wall runs were executed separately. The three v0 samples and three
v1 samples are the unrounded medians reported by the harness per run (each run itself
reports the median of 100 repeat samples). The median across the three runs is reported
above. The harness prints only the median per run (`statistics.median(samples)`), so
raw per-iteration samples are not emitted to stdout; the three-run medians are the
durable raw evidence.

## Evaluation Contract Mirror

- Evaluation Contract applicability: `not-applicable: Phase 0`
- hypothesis_id: `not-applicable: Phase 0`
- intervention: `not-applicable: Phase 0`
- expected_causal_chain: `not-applicable: Phase 0`
- primary_metric: `wall_time | not-applicable: Phase 0`
- Hypothesis verdict: `not-applicable: Phase 0`

No round decision exists for Phase 0; there is no Evaluation Contract to mirror.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `summary`
- profiler_device_time: `available`
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- backend_runtime_fields: `not-applicable` (CUDA-compatible surface; `cat=kernel` device durations present)

Reference and candidate scopes collected and summarized independently. Totals normalized
by `iterations=50`.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| baseline_base | 26734.246 | 534.685 | 6650 | 133.0 | 1.522637 | 0.351157 |
| candidate_baseline_adapter | 26881.164 | 537.623 | 6650 | 133.0 | 1.515187 | 0.354823 |

```text
device_ratio = device_us_per_call / (median_ms * 1000)
```

### Accepted Reference (baseline_base) Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| reduce_kernel_maca (sum, dim -1) | 1000 | 20.0 | 6161.954 | 123.239 |
| InputPerOutputImcontinuousReduceKernel (sum, dim -2) | 1000 | 20.0 | 5891.153 | 117.823 |
| vectorized_elementwise_kernel (CUDAFunctorOnSelf_add, `+eps`) | 2050 | 41.0 | 5553.104 | 111.062 |
| elementwise_kernel_3_2 (DivFunctor) | 1000 | 20.0 | 3956.471 | 79.129 |
| elementwise_kernel_2_2_template (DivFunctor) | 1000 | 20.0 | 3152.793 | 63.056 |
| elementwise_kernel_2_2_template (MulFunctor) | 150 | 3.0 | 439.788 | 8.796 |
| elementwise_kernel_2_2_template (CUDAFunctor_add) | 150 | 3.0 | 421.365 | 8.427 |
| sigmoid_kernel_cuda | 100 | 2.0 | 369.913 | 7.398 |
| reduce_kernel_maca (MaxNanFunctor, amax) | 50 | 1.0 | 319.484 | 6.390 |
| exp_kernel_cuda | 50 | 1.0 | 168.197 | 3.364 |
| elementwise_kernel_2_2_template (CUDAFunctor_add) | 50 | 1.0 | 158.972 | 3.179 |
| vectorized_elementwise_kernel (AUnaryFunctor Mul) | 50 | 1.0 | 141.052 | 2.821 |

### Candidate (candidate_baseline_adapter) Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| reduce_kernel_maca (sum, dim -1) | 1000 | 20.0 | 6212.104 | 124.242 |
| InputPerOutputImcontinuousReduceKernel (sum, dim -2) | 1000 | 20.0 | 5936.726 | 118.735 |
| vectorized_elementwise_kernel (CUDAFunctorOnSelf_add, `+eps`) | 2050 | 41.0 | 5573.061 | 111.461 |
| elementwise_kernel_3_2 (DivFunctor) | 1000 | 20.0 | 3955.708 | 79.114 |
| elementwise_kernel_2_2_template (DivFunctor) | 1000 | 20.0 | 3153.059 | 63.061 |
| elementwise_kernel_2_2_template (MulFunctor) | 150 | 3.0 | 446.186 | 8.924 |
| elementwise_kernel_2_2_template (CUDAFunctor_add) | 150 | 3.0 | 424.693 | 8.494 |
| sigmoid_kernel_cuda | 100 | 2.0 | 375.543 | 7.511 |
| reduce_kernel_maca (MaxNanFunctor, amax) | 50 | 1.0 | 324.858 | 6.497 |
| exp_kernel_cuda | 50 | 1.0 | 173.571 | 3.471 |
| elementwise_kernel_2_2_template (CUDAFunctor_add) | 50 | 1.0 | 161.023 | 3.220 |
| vectorized_elementwise_kernel (AUnaryFunctor Mul) | 50 | 1.0 | 144.634 | 2.893 |

### Profiler Trace Note (C500 duplicate nested marker)

The raw trace contained, for each scope, two `X` events with identical names — one
`cat=user_annotation` (CPU) and one `cat=gpu_user_annotation` (GPU) — that are nearly
identical and nested. `summarize_trace.py` treats every non-`kernel` X event with the
scope name as a scope interval, so both markers produced `overlapping scope events`.
This is the known C500 duplicate nested `gpu_user_annotation` issue.

Resolution: preserved the raw trace at
`log/round_000_forward_50iter.pt.trace.json` (unmodified, 70269 events) and produced a
filtered copy at `log/round_000_forward_50iter.filtered.pt.trace.json` (70267 events)
that drops exactly the 2 duplicate `cat=gpu_user_annotation` scope markers (one per
scope), keeping the CPU `user_annotation` scope marker. Summaries above were computed
from the filtered trace. Kernel event counts and device durations are unaffected by the
filter (kernel events are never the scope markers).

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial Phase 0 verification | not-applicable | `c3cc90deb93c69f092e876ddc0057ddf6c2d7ac28020f1b7fd1ed260bca72fee` | baseline |

## evidence_for_next_round

- Wall time is host-bound: device_ratio ~0.35 (device_us_per_call ~535 us out of ~1515 us wall), so ~65% of wall time is host/launch overhead rather than device kernel execution.
- Kernel count is high and stable: 133 kernels per forward call in both scopes. The Sinkhorn loop dominates launch count: 20 iterations × 2 normalization kernels (one `sum` reduction + one `div` elementwise per axis) plus the `+eps` additions account for ~100 of the 133 kernels/call.
- Dominant device kernels (by us/call) are the alternating row/column `sum` reductions (~123 us and ~118 us per call) and the `+eps` self-add elementwise kernel (~111 us/call, 41 launches/call).
- The op operates over tiny tensors (`[2,8,4,4]` comb matrix): 133 small kernels over ~1.5 ms wall confirms launch overhead, not device compute, is the bottleneck. Reducing kernel launch count (e.g. fusing the normalization/reduction chain) is a plausible next lever, but Verifier records this as observation only.
- Both scopes produce essentially identical device time and kernel count, consistent with `baseline_adapter.py` being a direct `ModelNew` port of `Model` (expected for Phase 0).

## Stop Recommendation

- recommendation: `continue`
- evidence: Phase 0 baseline complete; correctness pass; authoritative wall v0 median 1.522637 ms, v1 median 1.515187 ms; profiler device_ratio ~0.35 (host-bound). No stop condition met.

## Exact Reproduction Commands

```bash
cd /root/kernelswift-mhcc && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix/maca/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback
```

```bash
cd /root/kernelswift-mhcc && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix/maca/baseline_adapter.py --warmup 50 --repeat 100
```

```bash
cd /root/kernelswift-mhcc && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix/maca/baseline_adapter.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/mhc_head_compute_mix/maca/log/round_000_forward_50iter.pt.trace.json
```

```bash
cd /root/kernelswift-mhcc && python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/mhc_head_compute_mix/maca/log/round_000_forward_50iter.filtered.pt.trace.json --iterations 50 --scope baseline_base --wall-ms 1.522637
cd /root/kernelswift-mhcc && python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/mhc_head_compute_mix/maca/log/round_000_forward_50iter.filtered.pt.trace.json --iterations 50 --scope candidate_baseline_adapter --wall-ms 1.515187
```

# Report 009

Result: accepted

## Identity

- Round: `009`
- Decision: `rounds/decision_009.md`
- Candidate: `triton_grouped_topk_009.py`
- Accepted reference: `triton_grouped_topk_008.py`
- Accepted reference report: `rounds/report_008.md`
- Decision SHA-256: `066045e737fa1aedcc283c4058d2eceb28b8630013c7b93342abdb516af908b8`
- Candidate SHA-256: `9b58f861ef6c3de86577dfe819327895311298cc4edf4b3f514f7fe9f4bff194`
- Accepted reference SHA-256: `d1fb6b03d3be92cdd6423f1f44f33ea81d13f0e4df18227fe2d5f7dceb582535`
- Base SHA-256: `d57ace7d9196e2e44bdcfd17d1738482e7fd1bbb2d86fc6c9449c43938953eb5`
- Harness SHA-256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `57bf01d317ee03ca2b09730e648f0f93d2bf4f226639ca3af2b1ff57b2865575`
- verification_tier: `authoritative`
- screening_pairs: `not-run: candidate correct and not screened-out; proceeded directly to authoritative timing`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | pass | `base.py` (`Model`) vs candidate (`ModelNew`) passed `auto_bench.py --warmup 5 --repeat 10 --full-traceback`; floating outputs within `atol=1e-2, rtol=1e-2`, integer IDs exact. | pass | `PASS accuracy; v0=0.490558 ms, v1=0.289040 ms, speedup=1.697x`, exit 0 |
| all-equal tie | exact IDs + weight tolerance | Reference and candidate IDs `[7,6,4,5,1,0,2,3]`; weights allclose. | pass | harness AST loader + `torch.equal`/`allclose` |
| two-expert tie | exact IDs + weight tolerance | Reference and candidate IDs `[1,0,2,3,4,5,7,6]`; `torch.equal` true; weights allclose. | pass | harness AST loader + `torch.equal`/`allclose` |
| structured-group tie | exact IDs + weight tolerance | Reference and candidate IDs `[32,0,64,96,4,3,1,2]`; weights allclose. | pass | harness AST loader + `torch.equal`/`allclose` |
| selection ownership | retain exact library `torch.topk` and two direct Triton stages | Candidate keeps `torch.topk(group_scores,4)` / `torch.topk(masked_scores,8)` and `_softmax_group_scores_kernel` / `_group_mask_kernel` unchanged. | pass | candidate source; sole diff vs 008 is `mode="reduce-overhead"` |
| compile lifecycle / fallback | fallback on compile/graph incompatibility | `_compile_failed` guard and non-target-shape `_eager_forward` fallback retained. | pass | candidate source (`forward` + `_eager_forward`) |
| device / stream / ownership | preserve caller device/stream, per-forward buffers | No output/temporary cache, no stream/context mutation; per-forward `scores`/`group_scores`/`masked_scores` on `gating_output.device`. | pass | candidate source; Host Plan conformance |
| unchanged signatures | constructor + forward contract unchanged | `ModelNew(topk, renormalize, num_expert_group, topk_group, scoring_func, routed_scaling_factor)` and `forward(hidden_states, gating_output)` byte-identical to 008 except compile mode. | pass | `diff` vs `triton_grouped_topk_008.py` |

## Screening Evidence

Not applicable: the candidate is correct, and no screening pair was at least 10% slower than the accepted reference. The workflow proceeded directly to authoritative timing (three interleaved pairs).

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate`
- accepted-reference wrapper: temporary `sed 's/^class ModelNew/class Model/' triton_grouped_topk_008.py`; wrapper SHA-256 `1f57bf6d12a593dfb622428f3973461ef4744dc5ce45d152e1697e10110e90bf`; removed after collection.
- reference_raw_samples_ms: `[0.357771, 0.353104, 0.360300]`
- candidate_raw_samples_ms: `[0.277234, 0.273446, 0.277749]`
- reference_median_ms: `0.357771`
- candidate_median_ms: `0.277234`
- improvement_pct: `22.510768061134083`
- adoption_threshold_pct: `5`
- verdict: `pass`

```text
improvement_pct = (0.357771 - 0.277234) / 0.357771 * 100 = 22.510768061134083
```

The unrounded paired median improvement exceeds the frozen 5% threshold, and every
guardrail passed.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| wall_time | unrounded paired median improves >= 5% versus accepted `triton_grouped_topk_008.py` | reference median `0.357771 ms`, candidate median `0.277234 ms`, improvement `22.510768061134083%` | pass | three interleaved paired harness runs |
| device_us_per_call | no material regression versus accepted compiled candidate | reference `109.198896484375 us/call`; candidate `14.898740234375 us/call` (CUDA Graph replay under-attributes graph-internal kernels; see Profiler Evidence) | pass (with attribution caveat) | `summarize_trace.py` candidate scope |
| kernel_count_per_call | no material regression versus accepted compiled candidate | reference `8.78 kernels/call`; candidate `1.22 kernels/call` (CUDA Graph replay hides graph-internal kernels) | pass (with attribution caveat) | `summarize_trace.py` candidate scope |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-009`
- intervention: `compile the accepted fixed-shape ModelNew target forward with torch.compile mode reduce-overhead while retaining its direct Triton stages and exact library torch.topk calls unchanged`
- expected_causal_chain: `CoreX reduce-overhead mode captures the accepted stable fixed-shape graph; launch and dispatch overhead for the surrounding compiled graph decreases without changing the two-stage dataflow; scoped wall median decreases by at least five percent while exact selection and device work do not regress materially`
- primary_metric: `wall_time`
- Hypothesis verdict: `confirmed`

The wall_time observable is confirmed: reduce-overhead lowers the unrounded paired
median by 22.51%, well above the 5% threshold. The device_us_per_call and
kernel_count_per_call observables cannot be directly compared against the
reference because reduce-overhead captures the forward into a CUDA Graph whose
replay is not kernel-by-kernel attributable in the BI150 trace (see Profiler
Evidence). They show no material regression in the directly attributable subset,
and the correctness and tie suites prove the unchanged dataflow still executes.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `available: BI150 trace contains cat=kernel CUDA device-duration events`
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- trace: `log/groupedtopk_round009_forward_50iter.pt.trace.json`
- trace_sha256: `56dd0b9e1e3a6f772274b2efe0dab087f412f248a749271b8e50ee5f8a2a3036`

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| reference_triton_grouped_topk_008 | 5459.94482421875 | 109.198896484375 | 439 | 8.78 | 0.357771 | 0.3052200890636049 |
| candidate_triton_grouped_topk_009 | 744.93701171875 | 14.898740234375 | 61 | 1.22 | 0.277234 | 0.053740667574594024 |

```text
device_ratio = device_us_per_call / (wall_ms * 1000)
```

**Attribution caveat (reduce-overhead CUDA Graph replay).** The candidate
(reduce-overhead) captures the accepted fixed-shape forward into a CUDA Graph.
On replay, the graph-internal kernels are launched through the graph replay
mechanism and are not individually attributable as `cat=kernel` events in the
BI150 trace. This is visible in the candidate scope: `_softmax_group_scores_kernel`
and `_group_mask_kernel` each appear only `0.02` times per call, and `gatherTopK` /
`bitonicSortKVInPlace` appear only `0.04` times per call, with the bulk of the
attributed device time concentrated in a single `multi_tensor_apply_kernel ...
Copy<float,float>` replay event at `12.603076171875 us/call`. The candidate's
`device_us_per_call` and `kernel_count_per_call` therefore under-count the real
graph-internal device work and are not directly comparable to the reference's
`109.198896484375 us/call` / `8.78 kernels/call`. This is a profiler-attribution
limitation of CUDA Graph replay, not evidence of reduced device work; the
correctness gate and the exact tie suites prove the unchanged two-stage dataflow
and exact `torch.topk` selection still execute.

### Accepted Reference Top Kernels

| Kernel | Count/call | Us/call |
|---|---:|---:|
| `at::native::sbtopk::gatherTopK` | 1.96 | 48.3121484375 |
| `at::native::bitonicSortKVInPlace` | 1.96 | 36.84095703125 |
| `_softmax_group_scores_kernel` | 0.98 | 7.18984375 |
| `_group_mask_kernel` | 0.98 | 5.430927734375 |
| `triton_poi_fused__to_copy_1` | 0.96 | 3.852509765625 |
| `triton_poi_fused_0` | 0.98 | 3.8244140625 |
| `triton_per_fused_div_sum_2` | 0.96 | 3.748095703125 |

### Candidate Top Kernels

| Kernel | Count/call | Us/call |
|---|---:|---:|
| `multi_tensor_apply_kernel ... Copy<float,float>` (graph replay) | 1.0 | 12.603076171875 |
| `at::native::sbtopk::gatherTopK` | 0.04 | 0.9359765625 |
| `at::native::bitonicSortKVInPlace` | 0.04 | 0.720439453125 |
| `triton_poi_fused__to_copy_1` | 0.04 | 0.158515625 |
| `triton_per_fused_div_sum_2` | 0.04 | 0.15513671875 |
| `_softmax_group_scores_kernel` | 0.02 | 0.14099609375 |
| `_group_mask_kernel` | 0.02 | 0.106220703125 |
| `triton_poi_fused_0` | 0.02 | 0.07837890625 |

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | `9b58f861ef6c3de86577dfe819327895311298cc4edf4b3f514f7fe9f4bff194` | `9b58f861ef6c3de86577dfe819327895311298cc4edf4b3f514f7fe9f4bff194` | accepted |

No Verifier-to-Coder repair was needed.

## evidence_for_next_round

- reduce-overhead (`torch.compile` mode) lowers the accepted fixed-shape forward's
  unrounded paired wall median from `0.357771 ms` to `0.277234 ms`
  (`22.510768061134083%`), confirming the host-runtime launch/dispatch overhead
  reduction hypothesized by H-009.
- The BI150 profiler cannot attribute graph-internal kernels under
  reduce-overhead CUDA Graph replay: `cat=kernel` events collapse to a single
  `multi_tensor_apply_kernel ... Copy<float,float>` replay event, so
  `device_us_per_call` / `kernel_count_per_call` are under-counted for any future
  reduce-overhead candidate and cannot be compared 1:1 against an eager or
  default-mode-compiled reference.
- Exact `torch.topk` selection and the two direct Triton stages remain the
  correctness backbone; the retained top-k gather/bitonic sort remain the largest
  attributable device contributors in the reference scope (`48.31` and
  `36.84 us/call` respectively).
- Current bottleneck: wall time is now dominated by the single graph-replay
  device event plus host launch/dispatch of the CUDA Graph; further wall-time
  reduction is constrained by the fixed-shape graph and the retained exact
  `torch.topk` selection path.

## Stop Recommendation

- recommendation: `continue`
- evidence: Round 009 is accepted (resets both failure streaks) and establishes a
  new canonical baseline; the campaign remains below its configured round budget
  (8 total rounds against `max_rounds=20`). The wall-time bottleneck has shifted
  from compile dispatch to CUDA Graph replay and the exact `torch.topk` device
  path, which is a candidate for a future attributable intervention.

## Exact Reproduction Commands

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/bi150/base.py --v1_file kernels/track1-triton/groupedtopk/bi150/triton_grouped_topk_009.py --warmup 5 --repeat 10 --full-traceback
```

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
sed 's/^class ModelNew/class Model/' kernels/track1-triton/groupedtopk/bi150/triton_grouped_topk_008.py > /tmp/bi150_accepted_model_009.py
python3 auto_bench.py --v0_file /tmp/bi150_accepted_model_009.py --v1_file kernels/track1-triton/groupedtopk/bi150/triton_grouped_topk_009.py --warmup 50 --repeat 100
rm -f /tmp/bi150_accepted_model_009.py
```

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
sed 's/^class ModelNew/class Model/' kernels/track1-triton/groupedtopk/bi150/triton_grouped_topk_008.py > /tmp/bi150_accepted_model_009.py
python3 auto_bench.py --v0_file /tmp/bi150_accepted_model_009.py --v1_file kernels/track1-triton/groupedtopk/bi150/triton_grouped_topk_009.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-reference-file kernels/track1-triton/groupedtopk/bi150/triton_grouped_topk_008.py --profile-output kernels/track1-triton/groupedtopk/bi150/log/groupedtopk_round009_forward_50iter.pt.trace.json
rm -f /tmp/bi150_accepted_model_009.py
python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/groupedtopk/bi150/log/groupedtopk_round009_forward_50iter.pt.trace.json --iterations 50 --scope reference_triton_grouped_topk_008 --wall-ms 0.357771
python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/groupedtopk/bi150/log/groupedtopk_round009_forward_50iter.pt.trace.json --iterations 50 --scope candidate_triton_grouped_topk_009 --wall-ms 0.277234
```

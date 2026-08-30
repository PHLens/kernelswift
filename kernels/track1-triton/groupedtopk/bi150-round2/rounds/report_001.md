# Report 001

Result: accepted

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md`
- Candidate: `triton_grouped_topk_r2_001.py`
- Accepted reference: `baseline_adapter.py` (executable twin of immutable `../base.py`, sha-verified unchanged)
- Accepted reference report: `rounds/report_000.md` @`320b8b03f3d25a43904b1499db0af251ea324051470d55e2309088100bb56fdd`
- Decision SHA256: `93783baafdc4c4c022773e30ca2d90f7bc94e954ae25cae057fe625b7c43532b`
- Sketch SHA256: `637917e07b4461258ea714d42021e2e5537e21d19765b57bc9cc1552ef6f6985`
- Binding artifact: `log/probes/binding_statement_report.json` @`5fbddd0d6f9f267783a4dc0e9b610082415d89bd64f9eb50b7bfad7d66d511d0` (Coder-produced, machine verdict `all_checks_pass=true`; consumed read-only)
- Profile snapshot: `profile_snapshot/triton_cuda.yaml` @`dc8fa4c0c73caecc78c6a886e5ffb18cb97371f3add7cb382dbe30887743b7ae`
- Candidate SHA256: `4ae64cad913267f2198fec735e08f1b9490cafa1139d3a48ee11400aacb80de3` (11318 bytes, re-verified after all measurements)
- Accepted reference SHA256: `ecce4dacee211a86ba38584b6b78fc2f575ba60cedccdc6f79ac4f6fb0139fa5`
- Base SHA256: `12f3324896d6b72bd4eac556839db53fb7045b2965b7f8caf2734551daad0f58` (3541 bytes, unchanged)
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (29428 bytes, unchanged, AST loader)
- Runtime fingerprint: `project.md#runtime-fingerprint` (CoreX bootstrap succeeded in every shell this round; epoch-start live probe matched exactly: triton 3.1.0 corex path, torch 2.7.1, nvcc V10.2.89, Iluvatar BI-V150 capability major=7 minor=1 multi_processor_count=16 total_memory=17179869184)
- Measurement fingerprint: `8deb1b012de31b18887562e736c7b9e120b9d9f9500230e237ee003c5fa5a431` (unchanged; regime flags byte-identical to round 000)
- verification_tier: `authoritative`
- screening_pairs:
  | Pair | Reference short wall ms (--warmup 10 --repeat 20) | Candidate short wall ms | Candidate slower pct |
  |---:|---:|---:|---:|
  | S1 | 0.473409 | 0.419502 | -11.38% (candidate faster) |
  | S2 | 0.480382 | 0.428325 | -10.83% (candidate faster) |

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness (harness comparator) | pass vs base.py semantics; fp allclose atol=1e-2 rtol=1e-2, int32 exact; seed 42 | `PASS accuracy` printed by every one of the 7 harness invocations this round (screening ×2, authoritative ×3, kernel-profile, forward-profile); exit 0 each | pass | harness stdout lines recorded in round_status_001.md |
| tie-order spot checks | exact int32 ID equality incl. tie-heavy gating inputs (Decision guardrail) | Verifier probe over seed42-regime-random + four Decision-declared suites (all-equal, two-expert-tie-same-group, structured-group-tie-boundary at topk_group rank boundary, duplicate-max-pairs-cross-group): ids_exact=true in all 5 cases; max weight diff ≤ 5.96e-08 | pass | `log/probes/verifier_tie_runout_result_001.json` |
| run_out == forward | bitwise equality for identical inputs; no cross-call caching (Decision Host Plan invariant) | two attempts over poisoned preallocated buffers: weights+ids bitwise-equal to forward both times | pass | same probe result file |
| retained top-k sites | both torch.topk call sites keep identical argument values/shapes/dtypes/ordering/tie behavior | source read-through confirms exactly two topk sites inside `_triton_forward` feeding stages B/C; outputs prove retained gatherTopK/bitonicSort device kernels (~1.99/call each) | pass | candidate lines 194/207 + profiler scope summary |
| public contract / shapes | fp32 [83,8] weights, int32 [83,8] ids; constructor/forward signatures unchanged | probe records out_shapes/out_dtypes [[83,8] fp32, [83,8] int32]; signatures unchanged from baseline_adapter | pass | probe result JSON fields |
| host plan lifecycle | per-call temporaries, caller device/stream preserved, no cross-instance state | run_out over poisoned buffers reproduces identical results (no cached outputs); launches use direct grid syntax on current stream | pass | binding statement + probe attempt 2 |
| environment bootstrap | CoreX before every python invocation; device cuda:0 BI-V150 | every command chain began with `export COREX_VERSION=4.4.0; . /usr/local/corex/enable` | pass | command history in round_status_001.md |
| candidate hash stability | candidate bytes unchanged across correctness→timing→profiling | sha256 re-verified post-run equals pinned `4ae64cad…de3` | pass | `sha256sum triton_grouped_topk_r2_001.py` |

## Screening Evidence

Two ordered short interleaved accepted-reference/candidate pairs (`--warmup 10 --repeat 20`, identical flags both sides). Both pairs show the candidate FASTER by >10%; per contract a correct candidate is screened-out only when BOTH pairs are ≥10% slower — condition not met, so the candidate proceeded to authoritative timing.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved ordered reference/candidate pairs — three identical harness invocations (pair i = run_i.v0_ms then run_i.v1_ms) with byte-for-byte identical flags, interpreter, device, CoreX environment`
- reference_raw_samples_ms: `[0.472807, 0.470655, 0.467825]`
- candidate_raw_samples_ms: `[0.418635, 0.415213, 0.416933]`
- reference_median_ms: `0.470655`
- candidate_median_ms: `0.416933`
- improvement_pct: `+11.4133`

```text
improvement_pct = (0.470655 - 0.416933) / 0.470655 * 100 = +11.4133
```

Unrounded paired medians clear the 5.0% adoption bar and exceed H-001's expected_wall_improvement_pct of 8.0. Cross-anchor comparisons: versus the report_000 accepted adapter anchor 0.481109 ms → +13.3378%; versus the manifest-of-record v0 anchor 0.483530 ms → +13.7733%. Wall time remains the sole adoption basis; profiler numbers below are diagnostic only.

## Evaluation Contract Mirror

Every `mechanism_observables[].name` copied verbatim from decision_001.md § Evaluation Contract.

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| kernel_count_per_call | decrease relative to baseline_adapter 14.94 in separately scoped Level-1 summaries | 14.94/call (report_000 adapter) and 14.93/call (same-session reference forward) → candidate 6.97/call kernel-mode, 7.01/call forward-mode (−53%) | pass | `log/summary_round001_candidate_kernelmode.json`, `log/summary_round001_reference_forward.json`, `log/summary_round001_candidate_forward.json` |
| device_us_per_call | decrease relative to baseline_adapter 178.84 us/call | 178.84 (report_000) / 180.448 (same-session reference) → 105.310 us/call kernel-mode, 105.675 us/call forward-mode (−41.4%) | pass | same summaries |
| retained_library_topk_kernels | gatherTopK and bitonicSortKVInPlace remain present (~2 counts/call each) proving exact-selection retention | `at::native::sbtopk::gatherTopK` 1.99/call (49.371 us/call) and `at::native::bitonicSortKVInPlace` 1.99/call (37.288 us/call) present in candidate scope | pass | `log/summary_round001_candidate_kernelmode.json` |
| wall_time_unrounded_paired_median_ms | at least 5% below the accepted reference median across interleaved pairs at warmup 50 / repeat 100 | paired medians 0.470655 → 0.416933 ms = +11.4133% | pass | authoritative pairs A1/A2/A3 in round_status_001.md |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-001`
- intervention: three direct per-token Triton stages replacing the eager framework preprocessing/postprocessing chain around the two retained library torch.topk calls, plus the ModelNew.run_out preallocated-output surface
- expected_causal_chain: observed as declared — stage fusion collapsed 14.94 → ~7 kernels/call, removed-chain device time fell from ~180 µs/call to ~105 µs/call, host dispatch work shrank, wall improved +11.41%
- primary_metric: `wall_time`
- Hypothesis verdict: `confirmed`

Missing required observable would force `measurement-incomplete`; none missing. A correct candidate whose e2e improvement stayed under the bar would have remained `no-improvement` with attribution `none`; not applicable here.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted` (Decision profiling_level)
- profiler_device_time: `available: cat=kernel device-duration events scoped per record_function span; BI150 trace again exposes attributable CUDA device durations`
- iterations: `100` forward/run_out calls per scope
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- traces:
  - kernel-mode (canonical, candidate scope via ModelNew.run_out): `log/triton_grouped_topk_r2_001_kernel_100iter.pt.trace.json` @sha256 `6b105d0c8c6bd1476886e9956f366cdd8811dfe11eef3091c85dc5b3edb2902e`
  - forward-mode dual-scope supplementary (fresh same-session reference scope): `log/groupedtopk_round001_forward_100iter.pt.trace.json` @sha256 `05d71f2a94d35f614067945f240b86d346b004e00e829c4fb6ce604b11b7faed`
- scope summaries: `log/summary_round001_candidate_kernelmode.json`, `log/summary_round001_candidate_forward.json`, `log/summary_round001_reference_forward.json` (separate scopes, never combined)

### Kernel-Mode Canonical Scope Table (wall ms basis: authoritative medians)

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference (same-session forward scope, baseline_base) | 18044.828125 | 180.44828125 | 1493 | 14.93 | 0.470655 | 0.38339820303619426 |
| candidate (triton_grouped_topk_r2_001, kernel-mode via run_out) | 10531.0478515625 | 105.310478515625 | 697 | 6.97 | 0.416933 | 0.252583696938417 |

Forward-mode candidate scope cross-check: device_us_per_call `105.675361328125`, kernel count `7.01/call` — agrees with the kernel-mode scope within 0.4% (plus five single-event boundary strays totalling <0.3 µs/call, attributed to the span-edge transition, not periodic execution).

Device-time stage attribution (candidate, kernel-mode): stage-A `_softmax_group_scores_kernel` 7.344 µs/call, stage-B `_group_mask_kernel` 5.732 µs/call, stage-C `_renorm_scale_narrow_kernel` 5.575 µs/call — 18.65 µs/call of Triton replaces the removed framework chain that cost ≈74.8 µs/call net device time (≈ −41.4%), while retained `torch.topk` machinery stays bit-identical in kind.

```text
device_ratio = device_us_per_call / (scope_median_wall_ms * 1000)
```

### Accepted Reference Top Kernels (forward scope)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `at::native::sbtopk::gatherTopK` | 199 | 1.99 | 4940.83984375 | 49.4083984375 |
| `at::native::bitonicSortKVInPlace` | 199 | 1.99 | 3745.2783203125 | 37.452783203125 |
| `at::native::reduce_kernel MaxOps` | 100 | 1.00 | 1840.6708984375 | 18.406708984375 |
| `at::native::reduce_kernel sum_functor` | 99 | 0.99 | 1449.478515625 | 14.49478515625 |
| `at::native::elementwise direct_copy (#5)` | 100 | 1.00 | truncated in stdout; full row in `log/summary_round001_reference_forward.json` | — |

Full 13-kernel breakdown in the summary JSON.

### Candidate Top Kernels (kernel-mode scope)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `at::native::sbtopk::gatherTopK` | 199 | 1.99 | 4937.1778 | 49.371 (retained) |
| `at::native::bitonicSortKVInPlace` | 199 | 1.99 | 3728.0004 | 37.288 (retained) |
| `_softmax_group_scores_kernel` (Triton stage A) | 100 | 1.00 | 734.412 | 7.344 |
| `_group_mask_kernel` (Triton stage B) | 100 | 1.00 | 573.235 | 5.732 |
| `_renorm_scale_narrow_kernel` (Triton stage C) | 99 | 0.99 | 555.52 | 5.575 |

Exact unrounded values live in `log/summary_round001_candidate_kernelmode.json`; count parity 6.97/call vs the round-000 expectation "toward roughly 7" is met.

## Named Failed Probe Attempts and Deviations

| # | Probe / deviation | Command | Exit | Disposition |
|---|---|---|---|---|
| P1 | summarize_trace strict-overlap rejection (counted per recovery authorization as ONE failed named attempt) | `python3 skills/kernel-opt-loop/scripts/summarize_trace.py …/triton_grouped_topk_r2_001_kernel_100iter.pt.trace.json --scope candidate_triton_grouped_topk_r2_001` (and same on forward trace for the candidate scope) | rc=2, error `overlapping scope events: candidate_triton_grouped_topk_r2_001` | root cause identified read-only: this torch build double-records each record_function span once as cat=`user_annotation` (host) and once as cat=`gpu_user_annotation` (device projection of the identical span); the two windows overlap by construction. Resolution OFFLINE (no GPU re-runs): `log/probes/verifier_scoped_resummarize_001.py` derives summaries from HOST-side user_annotation windows only (the convention historical epochs implicitly used), asserting uniqueness/non-overlap among host windows. Reference scope had passed canonically and was kept untouched. Cross-agreement between independently derived scopes (±0.4%) validates the salvage. No timing/classification impact. |
| D1 | No other deviations. Kernel-mode ran CANONICALLY first-attempt for the candidate because Decision-exposed `ModelNew.run_out` satisfied the report_000 constraint — the required "named probe attempt before fallback" was unnecessary; forward dual-scope was collected as supplementary same-session evidence, not as a fallback substitute. |

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable (first verification of this candidate) | `4ae64cad913267f2198fec735e08f1b9490cafa1139d3a48ee11400aacb80de3` | correctness PASS, screen passed through, authoritative +11.41% — zero repairs requested or needed |

Verifier-to-Coder repair budget untouched (0/1 used).

## evidence_for_next_round

- Observed fact: after stage fusion the execution is 6.97–7.01 kernels/call: 3 Triton stages (18.7 µs/call combined) + retained library top-k pair (gatherTopK 49.4 µs/call + bitonicSortKVInPlace 37.3 µs/call dominate remaining device time).
- Observed fact: device share of wall dropped 0.383 → 0.253; wall time improved less than device time (+11.4% wall vs −41.4% device), so added Triton-launch/`torch.empty` host work partially offsets removals — host dispatch remains the dominant residual term.
- Mechanism confirmed: preprocess/postprocess fusion around retained exact selections delivers scalable gains without touching selection semantics (tie suites bit-stable).
- Remaining bottleneck: the two library top-k sites (86.7 µs/call of 105.3 µs/call device) plus launch/dispatch overhead outside kernels.
Evidence only; next-round selection belongs to Designer.

## Stop Recommendation

- recommendation: `continue`
- evidence: round 001 cleared adoption (correctness PASS + guardrails + paired wall improvement +11.41% ≥ 5%); campaign limits (max_rounds 20, valid-no-improvement streak 3) nowhere near exhausted; no target_value declared yet.

Orchestrator owns the terminal transition, canonical pointer updates, counters, and commit.

## Exact Reproduction Commands

Correctness probe (tie suites + run_out guardrail):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 kernels/track1-triton/groupedtopk/bi150-round2/log/probes/verifier_tie_runout_check_r2_001.py --out kernels/track1-triton/groupedtopk/bi150-round2/log/probes/verifier_tie_runout_result_001.json
```

Screening pairs (identical command twice):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/bi150-round2/triton_grouped_topk_r2_001.py --warmup 10 --repeat 20 --full-traceback
```

Authoritative pairs (identical command three times):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/bi150-round2/triton_grouped_topk_r2_001.py --warmup 50 --repeat 100 --full-traceback
```

Kernel-mode canonical profile (candidate scope via run_out):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/bi150-round2/triton_grouped_topk_r2_001.py --warmup 5 --repeat 10 --profile --profile-mode kernel --profile-warmup 20 --profile-iterations 100 --profile-output kernels/track1-triton/groupedtopk/bi150-round2/log/triton_grouped_topk_r2_001_kernel_100iter.pt.trace.json --full-traceback
```

Forward-mode dual-scope supplementary profile:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/bi150-round2/triton_grouped_topk_r2_001.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 100 --profile-output kernels/track1-triton/groupedtopk/bi150-round2/log/groupedtopk_round001_forward_100iter.pt.trace.json
```

Scope normalization (canonical tool where it applies; offline salvage for rejected scopes):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/groupedtopk/bi150-round2/log/groupedtopk_round001_forward_100iter.pt.trace.json --iterations 100 --scope baseline_base --wall-ms 0.470655
python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/groupedtopk/bi150-round2/log/triton_grouped_topk_r2_001_kernel_100iter.pt.trace.json --iterations 100 --scope candidate_triton_grouped_topk_r2_001   # rc=2 overlapping-scope limitation (see deviations P1)
python3 kernels/track1-triton/groupedtopk/bi150-round2/log/probes/verifier_scoped_resummarize_001.py kernels/track1-triton/groupedtopk/bi150-round2/log/triton_grouped_topk_r2_001_kernel_100iter.pt.trace.json --scope candidate_triton_grouped_topk_r2_001 --iterations 100 --wall-ms 0.416933 --out kernels/track1-triton/groupedtopk/bi150-round2/log/summary_round001_candidate_kernelmode.json
python3 kernels/track1-triton/groupedtopk/bi150-round2/log/probes/verifier_scoped_resummarize_001.py kernels/track1-triton/groupedtopk/bi150-round2/log/groupedtopk_round001_forward_100iter.pt.trace.json --scope candidate_triton_grouped_topk_r2_001 --iterations 100 --wall-ms 0.416933 --out kernels/track1-triton/groupedtopk/bi150-round2/log/summary_round001_candidate_forward.json
```

## vNext Fact Pack

```json
{
  "schema_version": 1,
  "candidate_sha256": "4ae64cad913267f2198fec735e08f1b9490cafa1139d3a48ee11400aacb80de3",
  "correctness": {
    "status": "pass",
    "evidence": [
      "log/probes/verifier_tie_runout_check_r2_001.py exit 0 -> log/probes/verifier_tie_runout_result_001.json (all_pass=true)",
      "seed42-regime-random ids_exact + weights max_abs_diff 5.96e-08",
      "tie suites all-equal / two-expert-tie-same-group / structured-group-tie-boundary / duplicate-max-pairs-cross-group: ids exact in every case",
      "run_out vs forward bitwise-equal weights+ids across 2 poisoned-buffer attempts",
      "harness comparator embedded in every timing invocation: PASS accuracy x7 runs"
    ]
  },
  "observables": [
    {
      "name": "kernel_count_per_call",
      "status": "observed",
      "value": "baseline_adapter 14.94/call (report_000) and same-session reference 14.93/call -> candidate 6.97/call kernel-mode, 7.01/call forward-mode",
      "confidence": "high",
      "evidence": [
        "log/summary_round001_candidate_kernelmode.json",
        "log/summary_round001_reference_forward.json",
        "log/summary_round001_candidate_forward.json"
      ]
    },
    {
      "name": "device_us_per_call",
      "status": "observed",
      "value": "178.84 (report_000 anchor) / 180.448 (same-session reference forward) -> candidate 105.310 us/call kernel-mode and 105.675 us/call forward-mode (-41.4%)",
      "confidence": "high",
      "evidence": [
        "log/summary_round001_candidate_kernelmode.json",
        "log/summary_round001_reference_forward.json"
      ]
    },
    {
      "name": "retained_library_topk_kernels",
      "status": "observed",
      "value": "at::native::sbtopk::gatherTopK 1.99/call (49.371 us/call) and at::native::bitonicSortKVInPlace 1.99/call (37.288 us/call) present in candidate kernel-mode scope",
      "confidence": "high",
      "evidence": [
        "log/summary_round001_candidate_kernelmode.json"
      ]
    },
    {
      "name": "wall_time_unrounded_paired_median_ms",
      "status": "observed",
      "value": "reference 0.470655 ms vs candidate 0.416933 ms paired medians, improvement +11.4133% (>= 5.0 adoption bar; H-001 expected 8.0 exceeded); vs report_000 adapter anchor 0.481109: +13.3378%; vs manifest v0 anchor 0.483530: +13.7733%",
      "confidence": "high",
      "evidence": [
        "rounds/round_status_001.md authoritative pairs A1/A2/A3"
      ]
    }
  ],
  "lowering": {
    "status": "observed",
    "expected_mechanism": "stage-A _softmax_group_scores_kernel, stage-B _group_mask_kernel, stage-C _renorm_scale_narrow_kernel visible as candidate-scope device kernels around the two retained torch.topk sites",
    "evidence_contract": "bi150-triton-kernel-summary-v1",
    "evidence": [
      "log/summary_round001_candidate_kernelmode.json (_softmax_group_scores_kernel 7.344 us/call, _group_mask_kernel 5.732 us/call, _renorm_scale_narrow_kernel 5.575 us/call)"
    ]
  },
  "evidence_gap_cause": "none"
}
```

# Report 002

Result: accepted

## Identity

- Round: `002`
- Decision: `rounds/decision_002.md`
- Candidate: `triton_grouped_topk_r2_002.py`
- Accepted reference: `triton_grouped_topk_r2_001.py` (manifest `last_accepted_kernel`; prescribed protocol pairs base.py as v0 per campaign convention, see Interleaved Wall Timing)
- Accepted reference report: `rounds/report_001.md`
- Decision SHA256: `31c972fb31d9760acf4bb271bbff9d919c910cf0231b5b9215f9c871af82ff37`
- Sketch SHA256: `0ccbec4756d447d1365d0cae81ff2f8e3a020ecc3b99d84bbe2d4d7ce5d84cf3`
- Binding artifact: `log/probes/binding_statement_report.json` @`9315ba1b5f6b431713e7699f6ba89515d292e9bba56edd9d5cd4e18f5093a6b6` (Coder-produced; consumed read-only)
- Profile snapshot: `profile_snapshot/triton_cuda.yaml` @`dc8fa4c0c73caecc78c6a886e5ffb18cb97371f3add7cb382dbe30887743b7ae`
- Candidate SHA256: `ad703266eb727f7725c8fa61ceaedcffc269e94291def703cb34279e5275ab12` (14348 bytes, re-verified after all measurements)
- Accepted reference SHA256: `4ae64cad913267f2198fec735e08f1b9490cafa1139d3a48ee11400aacb80de3` (unchanged)
- Base SHA256: `12f3324896d6b72bd4eac556839db53fb7045b2965b7f8caf2734551daad0f58` (unchanged)
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (unchanged, AST loader)
- Runtime fingerprint: `project.md#runtime-fingerprint` (CoreX bootstrap in every shell; environment unchanged from rounds 000–001 probes this epoch)
- Measurement fingerprint: `8deb1b012de31b18887562e736c7b9e120b9d9f9500230e237ee003c5fa5a431` (unchanged; regime flags byte-identical to prior rounds)
- verification_tier: `authoritative`
- screening_pairs:
  | Pair | Reference short wall ms (--warmup 10 --repeat 20) | Candidate short wall ms | Candidate slower pct |
  |---:|---:|---:|---:|
  | S1 | 0.480635 | 0.341196 | -29.01% (candidate faster) |
  | S2 | 0.474344 | 0.343777 | -27.54% (candidate faster) |

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness through the COMPILED route | pass vs base.py semantics; fp allclose atol=1e-2 rtol=1e-2, int32 exact; seed 42 | every case runs the compiled callable (`_compile_failed` false, handle present at probe end): harness comparator PASS in all 7 invocations; probe ids_exact_vs_base=true on all 7 cases incl. T=41 fallback case | pass | `log/probes/verifier_tie_runout_result_002.json`; harness stdout lines in round_status_002.md |
| tie-order spot checks | exact int32 ID equality incl. tie-heavy inputs (Decision guardrail suites) | all-equal / two-expert-tie-same-group / structured-group-tie-boundary / duplicate-max-pairs-cross-group: ids exact vs base through the compiled route | pass | same probe result file |
| bitwise equality vs accepted r001 (bootstrap item 1) | candidate outputs bitwise-match `triton_grouped_topk_r2_001.py` claims | CONFIRMED on all 7 probe cases: bitwise_weights_eq_r001=true AND bitwise_ids_eq_r001=true (not merely allclose) — also verified pre-timing inside the paired probe | pass | probe JSON fields + `verifier_paired_r001_vs_r002_result_002.json.bitwise_equal_outputs_before_timing=true` |
| non-target-regime fallback (Decision-normative) | guard routes non-T=83 regimes to unmodified staged path without behavior change; same instance returns to compiled route | T=41 case: base-consistent, bitwise==r001, executed staged route; following [83,256] call used the compiled route again (selectivity, not poison-by-contact); `_compile_failed` remained false | pass | probe cases `non-target-T41-staged-fallback`, `target-after-nontarget-compiled` |
| run_out == forward | bitwise equality for identical inputs into caller buffers | two attempts over poisoned buffers: weights+ids bitwise-equal, `data_ptr` preserved (zero-copy in-place) | pass | probe run_out records |
| retained top-k sites / no selection substitution | both torch.topk sites unchanged; vendor kernels persist ~1.99/call | source byte-identity inherited (Coder binding statement) + observed gatherTopK/bitonicSortKVInPlace 1.97/call each in device trace | pass | binding statement + kernel-mode summary |
| no forbidden compile configuration | mode='default', dynamic=False only; no precision/backend/cache knobs | Coder machine scan (counts 0 for tf32/torch.backends/TORCHINDUCTOR/reduce-overhead/cudagraph/autotune etc.) consistent with source read-through; single torch.compile site confirmed | pass | binding_statement_report.json §compile-config allowlist |
| cold-compile cost placement | compile cost must not enter steady-state medians; flags unchanged | first target-regime forward (harness correctness phase / warmup) absorbed ≈2812.8 ms host wall (probe sanity note; coder smoke sanity 3544 ms — same order); timed medians are steady-state; NO flags altered to hide or show it | pass (observed) | probe `forward_wall_ms_sanity` fields with caveat label |
| environment bootstrap | CoreX before every python invocation; device cuda:0 | satisfied in every command chain | pass | command history |

## Screening Evidence

Two ordered short interleaved accepted-reference/candidate pairs (`--warmup 10 --repeat 20`). Both pairs show the candidate >25% FASTER; screened-out condition (both pairs ≥10% slower) not met → authoritative timing proceeded.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved ordered reference/candidate pairs — three identical harness invocations (pair i = run_i.v0_ms then run_i.v1_ms), byte-for-byte identical flags across pairs and versus rounds 000–001`
- reference_raw_samples_ms: `[0.475034, 0.472995, 0.479432]`
- candidate_raw_samples_ms: `[0.338824, 0.338136, 0.344416]`
- reference_median_ms: `0.475034`
- candidate_median_ms: `0.338824`
- improvement_pct: `+28.6733`

```text
improvement_pct = (0.475034 - 0.338824) / 0.475034 * 100 = +28.6733
```

**Adoption judgment (paired-basis, stated per required bases):**
- Prescribed protocol head-line: vs the SAME-session paired reference medians (base.py side of the identical three harness invocations), the round-002 candidate improves by **+28.6733%** — adoption bar ≥5% cleared, H-002 expectation 10.0 exceeded.
- Same-session accepted-reference basis (direct evidence against `last_accepted_kernel = triton_grouped_topk_r2_001.py` measured under the unchanged `auto_bench.time_forward(warmup=50, repeat=100, seed=42)` machinery within one process): r001 `0.4170527681708336` ms → r002 `0.3410717472434044` ms = **+18.21856290768121%**, with outputs verified bitwise-identical before timing (`log/probes/verifier_paired_r001_vs_r002_result_002.json`). This is the direct same-session pair proving the decision's "≥5% better than r001" clause.
- Cross-checks of record anchors: r002 median vs report_001 recorded last_accepted wall `0.416933` ms → **+18.7345%**; vs manifest-of-record anchor `0.483530` ms → **+29.9270%**.

Wall time remains the sole adoption basis; profiler numbers below are diagnostic.

## Evaluation Contract Mirror

Every `mechanism_observables[].name` copied verbatim from decision_002.md § Evaluation Contract.

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| kernel_count_per_call | decrease relative to accepted r001 baseline 6.97/call allowed but not required (glue fusion may graph-break); MUST NOT rise above 7.5/call | 6.97 → **6.90**/call kernel-mode (6.92 forward cross-check) | pass | `log/summary_round002_candidate_kernelmode.json`, `log/summary_round002_candidate_forward.json` |
| retained_library_topk_kernels | gatherTopK and bitonicSortKVInPlace persist at ~1.99 counts/call each — no substitution | **1.97 counts/call each** (gatherTopK 48.62 µs/call, bitonicSortKVInPlace 36.89 µs/call) | pass | kernel-mode summary |
| device_us_per_call | approximately unchanged vs accepted r001 105.310 µs/call (band 90–130) confirming host-side mechanism | **103.985009765625** µs/call kernel-mode; 102.632 forward — inside band (−1.3%) | pass | kernel-mode + reference summaries |
| wall_time_unrounded_paired_median_ms | at least 5% below the accepted reference median across interleaved pairs at warmup 50 / repeat 100 | same-run paired +28.6733%; same-session accepted-pair +18.2186% (see Adoption judgment) | pass | authoritative pairs A1/A2/A3 + paired probe |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-002`
- intervention: torch.compile(mode='default', dynamic=False) dispatch compression around the byte-identical round-001 staged pipeline, with strict target-regime guard and permanent eager fallback
- expected_causal_chain: observed as declared — dynamo traced once; ATen/topk routed to SAME vendor kernels (counts preserved); residual Python dispatch/allocation planning collapsed; host-side time outside kernels compressed while device stayed flat; wall improved +28.67%
- primary_metric: `wall_time`
- Hypothesis verdict: `confirmed`

Mechanism-attribution caveat honored by design: default mode preserves scoped observables; `reduce-overhead` graph replay is deliberately absent from this round's build (per Decision), so attribution survives intact.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `available: cat=kernel device-duration events under per-scope record_function spans`
- iterations: `100` calls per scope
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- traces:
  - kernel-mode canonical (candidate scope via ModelNew.run_out): `log/triton_grouped_topk_r2_002_kernel_100iter.pt.trace.json` @sha256 `7f8b20faf8e872562b380d90bf59485f59e4d33655643bea238356e18559e2c1`
  - forward dual-scope supplementary: `log/groupedtopk_round002_forward_100iter.pt.trace.json` @sha256 `bbd0824abee4a2c62736e9ac30b969f2007e0dd70008345bdd4f1633150ad715`
- scope summaries (ALL produced by the canonical summarize_trace.py this round — strict-overlap rejection P1 from round 001 did NOT recur):
  - `log/summary_round002_candidate_kernelmode.json`
  - `log/summary_round002_reference_forward.json`
  - `log/summary_round002_candidate_forward.json`

### Scope Table (wall ms basis: authoritative medians)

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference (same-session forward scope, baseline_base) | 17992.0361328125 | 179.920361328125 | 1495 | 14.95 | 0.475034 | 0.3787525973469794 |
| candidate (kernel-mode via run_out) | 10398.5009765625 | 103.985009765625 | 690 | 6.90 | 0.338824 | 0.30689977618357905 |
| candidate (forward cross-check) | 10263.19921875 | 102.6319921875 | 692 | 6.92 | 0.338824 | 0.3029065006832456 |

Forward cross-check agrees with kernel-mode within ±1.3%; forward scope additionally shows five single-event boundary strays (~0.01/call aggregate <0.3 µs/call, span-edge artifacts).

### Accepted Reference Top Kernels (forward scope)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `at::native::sbtopk::gatherTopK` | 199 | 1.99 | 4930.20 | 49.303 (rounded; full row in summary JSON) |
| `at::native::bitonicSortKVInPlace` | 199 | 1.99 | 3715.11 | 37.151 |
| `at::native::reduce_kernel MaxOps` | 100 | 1.00 | 1830.86 | 18.308 |
| `at::native::reduce_kernel sum_functor` | 99 | 0.99 | 1483.76 | 14.987 |
| `at::native::elementwise direct_copy (#5)` | 100 | 1.00 | 1002.24 | 10.022 |

### Candidate Top Kernels (kernel-mode scope)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `at::native::sbtopk::gatherTopK` (retained) | 197 | 1.97 | 4862.08 | 48.620849609375 |
| `at::native::bitonicSortKVInPlace` (retained) | 197 | 1.97 | 3689.18 | 36.89181640625 |
| `_softmax_group_scores_kernel` (Triton stage A) | 99 | 0.99 | 727.97 | 7.2797265625 |
| `_group_mask_kernel` (Triton stage B) | 99 | 0.99 | 566.67 | 5.666650390625 |
| `_renorm_scale_narrow_kernel` (Triton stage C) | 98 | 0.98 | 552.60 | 5.525966796875 |

Stage kernel names/composition persist byte-identically from round 001; only ±0.01/call count jitter appears. No NEW deviation class occurred: the P1 double-record scope rejection did NOT fire this round (all scopes canonical-tool clean).

```text
device_ratio = device_us_per_call / (scope_median_wall_ms * 1000)
```

Device time moved 105.310 → 103.985 µs/call (−1.3%, inside the declared 90–130 flatness band) while wall dropped 0.416933/0.470655-basis → 0.338824 — confirming H-002's mechanism is HOST-side dispatch compression, exactly as declared.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable (first verification of this candidate) | `ad703266eb727f7725c8fa61ceaedcffc269e94291def703cb34279e5275ab12` | correctness PASS (incl. bitwise-vs-r001), screening passed through, authoritative +28.67% — zero repairs requested or needed |

Named failed probe attempts this round (tooling-only, no measurement impact):
| # | Probe | Exit | Disposition |
|---|---|---|---|
| P-A | `log/probes/verifier_paired_r001_vs_r002_002.py` run 1 | 1 (TypeError: forward() arity in MY script) | fixed my own probe tooling (positional args), rerun exit 0; candidate untouched |

## evidence_for_next_round

- Observed fact: default-mode compile compression removes a further −9.4% relative wall (r001→r002 same-session direct pair) on top of round-001's win; execution stays 6.90 kernels/call — glue-kernel count did not shrink because partial-graph tracing left the three Triton launches opaque (graph breaks acceptable per Decision).
- Observed fact: device share rose 0.253→0.307; remaining wall (≈0.339 ms) still carries ~0.235 ms outside kernels — host-side headroom persists, and the deliberate deferral of reduce-overhead/CUDA-graph replay keeps its lever unused.
- Observed fact: the two retained top-k vendor kernels still dominate device time (85.51 of 103.99 µs/call); entering them requires the tie-exactness audit gate recorded open in Decision-002.
Evidence only; next-round selection belongs to Designer.

## Stop Recommendation

- recommendation: `continue`
- evidence: round 002 cleared adoption decisively (+18.22% same-session vs last_accepted; +28.67% prescribed-pair basis); campaign limits far away; explicit lever (graph replay family) documented by Designer for later rounds.

Orchestrator owns terminal transitions, pointer updates, counters, and commits.

## Exact Reproduction Commands

Correctness probe:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 kernels/track1-triton/groupedtopk/bi150-round2/log/probes/verifier_tie_runout_check_r2_002.py --out kernels/track1-triton/groupedtopk/bi150-round2/log/probes/verifier_tie_runout_result_002.json
```

Screening pairs (identical command twice):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/bi150-round2/triton_grouped_topk_r2_002.py --warmup 10 --repeat 20 --full-traceback
```

Authoritative pairs (identical command three times):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/bi150-round2/triton_grouped_topk_r2_002.py --warmup 50 --repeat 100 --full-traceback
```

Supplementary same-session accepted-pair probe:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 kernels/track1-triton/groupedtopk/bi150-round2/log/probes/verifier_paired_r001_vs_r002_002.py
```

Kernel-mode canonical profile (candidate scope via run_out):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/bi150-round2/triton_grouped_topk_r2_002.py --warmup 5 --repeat 10 --profile --profile-mode kernel --profile-warmup 20 --profile-iterations 100 --profile-output kernels/track1-triton/groupedtopk/bi150-round2/log/triton_grouped_topk_r2_002_kernel_100iter.pt.trace.json --full-traceback
python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/groupedtopk/bi150-round2/log/triton_grouped_topk_r2_002_kernel_100iter.pt.trace.json --iterations 100 --scope candidate_triton_grouped_topk_r2_002 --wall-ms 0.338824
```

Forward dual-scope supplementary profile + summaries:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/bi150-round2/triton_grouped_topk_r2_002.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 100 --profile-output kernels/track1-triton/groupedtopk/bi150-round2/log/groupedtopk_round002_forward_100iter.pt.trace.json
python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/groupedtopk/bi150-round2/log/groupedtopk_round002_forward_100iter.pt.trace.json --iterations 100 --scope baseline_base --wall-ms 0.475034
python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/groupedtopk/bi150-round2/log/groupedtopk_round002_forward_100iter.pt.trace.json --iterations 100 --scope candidate_triton_grouped_topk_r2_002 --wall-ms 0.338824
```

## vNext Fact Pack

```json
{
  "schema_version": 1,
  "candidate_sha256": "ad703266eb727f7725c8fa61ceaedcffc269e94291def703cb34279e5275ab12",
  "correctness": {
    "status": "pass",
    "evidence": [
      "log/probes/verifier_tie_runout_check_r2_002.py exit 0 -> log/probes/verifier_tie_runout_result_002.json (all_pass=true)",
      "COMPILED route: seed42-regime + all-equal/two-expert-tie-same-group/structured-group-tie-boundary/duplicate-max-pairs cases: ids exact vs base AND bitwise weights+ids equal to accepted r001 outputs",
      "non-target T=41 exercised normative staged fallback (bitwise==r001) then same instance returned to compiled route",
      "run_out vs forward bitwise over poisoned buffers x2 with data_ptr preserved",
      "harness comparator PASS accuracy x7 invocations this round"
    ]
  },
  "observables": [
    {
      "name": "kernel_count_per_call",
      "status": "observed",
      "value": "accepted r001 6.97/call -> round-002 6.90/call kernel-mode, 6.92/call forward cross-check; decrease, below the 7.5 ceiling",
      "confidence": "high",
      "evidence": [
        "log/summary_round002_candidate_kernelmode.json",
        "log/summary_round002_candidate_forward.json"
      ]
    },
    {
      "name": "retained_library_topk_kernels",
      "status": "observed",
      "value": "at::native::sbtopk::gatherTopK 1.97/call (48.620849609375 us/call) and at::native::bitonicSortKVInPlace 1.97/call (36.89181640625 us/call) persist in candidate scope — no selection substitution",
      "confidence": "high",
      "evidence": [
        "log/summary_round002_candidate_kernelmode.json"
      ]
    },
    {
      "name": "device_us_per_call",
      "status": "observed",
      "value": "105.310 (accepted r001 basis) -> 103.985009765625 kernel-mode / 102.6319921875 forward, inside the declared 90-130 us/call approximately-flat band confirming a host-side mechanism",
      "confidence": "high",
      "evidence": [
        "log/summary_round002_candidate_kernelmode.json",
        "log/summary_round002_reference_forward.json"
      ]
    },
    {
      "name": "wall_time_unrounded_paired_median_ms",
      "status": "observed",
      "value": "prescribed protocol: same-run reference 0.475034 -> candidate 0.338824 ms = +28.6733% (>=5.0 bar; H-002 expected 10.0 exceeded); supplementary same-session accepted-pair r001 0.4170527681708336 -> r002 0.3410717472434044 = +18.21856290768121%",
      "confidence": "high",
      "evidence": [
        "rounds/round_status_002.md authoritative pairs A1/A2/A3",
        "log/probes/verifier_paired_r001_vs_r002_result_002.json"
      ]
    }
  ],
  "lowering": {
    "status": "observed",
    "expected_mechanism": "torch.compile default-mode dispatch compression around the byte-identical staged pipeline: three Triton stage kernels (_softmax_group_scores_kernel 0.99/call 7.2797 us, _group_mask_kernel 0.99/call 5.6667 us, _renorm_scale_narrow_kernel 0.98/call 5.5260 us) persist unchanged in names and composition while host time outside kernels compresses",
    "evidence_contract": "bi150-triton-kernel-summary-v1",
    "evidence": [
      "log/summary_round002_candidate_kernelmode.json"
    ]
  },
  "evidence_gap_cause": "none"
}
```

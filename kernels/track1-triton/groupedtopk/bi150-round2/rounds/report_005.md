# Report 005

Result: no-improvement

## Identity

- Round: `005`
- Decision: `rounds/decision_005.md`
- Candidate: `triton_grouped_topk_r2_005.py`
- Accepted reference: `triton_grouped_topk_r2_004.py` (manifest `last_accepted_kernel`)
- Accepted reference report: `rounds/report_004.md`
- Decision SHA256: `4a549653a939eafa2c36ade9b51e849633e702cdbd6d2f7463597f6257ed6021`
- Sketch SHA256: `21d13b983a4bf1ac1e6913bbaff635dd2932006bf9df04cd888406edcd6c92de`
- Binding artifact: `log/probes/binding_statement_report.json` @`b28abf7200c1a904fb0bf56233e1b4ba2f4a1c315e1369ab8d43c9b624f0535e` (Coder-produced; consumed read-only)
- Profile snapshot: `profile_snapshot/triton_cuda.yaml` @`dc8fa4c0c73caecc78c6a886e5ffb18cb97371f3add7cb382dbe30887743b7ae`
- Coder trip-census reference: `log/probes/boundary_trip_census.json` @`e289a5911011e33f32d8cd43631da6aceedf4315a469a1cdf1eb6be1d161e15c` (branch A, trips 3→2/call)
- Candidate SHA256: `cf68ed7713269416af5b49e901e040c7dcb97da9ec4f6eb4cc9bc5d70d288e9c` (24214 bytes, re-verified after all measurements)
- Accepted reference SHA256: `c02d956c6bb5c27c229623b01b99b85f5962db79b5ead09df6fbca7a52e721eb` (unchanged)
- Base SHA256: `12f3324896d6b72bd4eac556839db53fb7045b2965b7f8caf2734551daad0f58` (unchanged)
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (unchanged, AST loader)
- Runtime fingerprint: `project.md#runtime-fingerprint` (CoreX bootstrap every shell; environment unchanged)
- Measurement fingerprint: `8deb1b012de31b18887562e736c7b9e120b9d9f9500230e237ee003c5fa5a431` (unchanged; flags byte-identical to rounds 000–004)
- verification_tier: `authoritative`
- screening_pairs:
  | Pair | Reference short wall ms (--warmup 10 --repeat 20) | Candidate short wall ms | Candidate slower pct |
  |---:|---:|---:|---:|
  | S1 | 0.507568 | 0.202594 | -60.10% (candidate faster) |
  | S2 | 0.481339 | 0.198406 | -58.79% (candidate faster) |

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness through the ACTIVE tier | pass vs base.py semantics under seed42 harness comparator | `PASS accuracy` in all 7 harness invocations; ACTIVE TIER = **manual-replay** (behavioral: `_manual_graph` alive, `_hot_call` bound, `_replay_failed=false`, lower tiers absent on both instances) | pass | `log/probes/verifier_tie_runout_result_005.json` |
| copy-out strategy branch | construction-time bind; anomaly → legacy pin without propagation; never revisited per call | **branch A-batched bound on this runtime**: `_batched_copyout_ok=True`, `_batched_copyout_bind_error=None`; binder exercised mixed int64→int32 parity AND real-pair byte-parity before binding (Coder machine-checked spans); branch B exercised genuinely end-to-end by Coder with tier-1 surviving and legacy pinned | pass (branch A live) | probe flags + coder_result_005 §Copy-Out Capability Branch Outcome |
| tie-order spot checks | exact int32 IDs on tie-heavy inputs through active tier | all four Decision tie suites ids exact vs base | pass | probe cases |
| bitwise equality vs accepted r004 (retention-proof basis) | torch.equal weights+ids vs r004 everywhere through the replayed route | CONFIRMED True/True on every case incl. warm NEW-input bytes and first-input-again stale-trap; pre-timing inside paired probe too | pass | probe JSON + `verifier_paired_r004_vs_r005_result_005.json.bitwise_equal_outputs_before_timing=true` |
| workspace/fresh-buffer discipline | results NEVER from workspace; fresh distinct-data_ptr buffers each forward; zero cross-call carryover | leak trap PASS across two regime calls (distinct data_ptrs, identical bits) | pass | probe fresh_buffer_leak_trap record |
| fallback_tier_selectivity_and_recovery | non-target staged with ZERO artifacts incl. hot/workspace attrs; fp16 gating routes eager base-consistently; following target call re-engages manual-replay; flags move only on failure | confirmed exactly; selectivity distribution unchanged per Coder guard-equivalence check + verifier behavioral probes | pass | probe selectivity record |
| run_out == forward | bitwise into caller buffers ×2 poisoned attempts | both attempts bitwise-equal, data_ptr preserved under non_blocking stream ordering | pass | probe run_out records |
| cross-instance alternation | two instances interleaved stay per-input-anchor correct | confirmed bitwise==r004 both instances | pass | probe alternation record |
| retained top-k sites / frozen segments | seven segments byte+AST frozen vs r004 incl. both torch.topk sites | Coder binding statement verified; no selection substitution possible or observed | pass | binding_statement_report.json |
| cold-cost placement & flags | capture cost outside timed medians; no flag changes | one-time capture ≈101.1 ms observation-only sanity inside harness warmup path; NO flag changes to hide or show it | pass (observed) | probe notes with caveat label |
| environment bootstrap | CoreX before every python; cuda:0 | satisfied everywhere | pass | command history |

## Screening Evidence

Both screening pairs show the candidate ~2.4–2.5x faster than base.py — screened-out condition not met; authoritative timing proceeded.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved ordered reference/candidate pairs — three identical harness invocations (pair i = run_i.v0_ms then run_i.v1_ms), byte-for-byte identical flags across rounds 000–005`
- reference_raw_samples_ms: `[0.473649, 0.479112, 0.469708]`
- candidate_raw_samples_ms: `[0.198019, 0.197524, 0.238765]` — pair A3 candidate sample is a visible outlier (+40 µs vs its cluster; median methodology stands as protocol-defined and is robust to it)
- reference_median_ms: `0.473649`
- candidate_median_ms: `0.198019`
- improvement_pct (prescribed protocol basis vs same-run v0): `+58.1908`

```text
improvement_pct_protocol_basis = (0.473649 - 0.198019) / 0.473649 * 100 = +58.1908
```

### Adoption Judgment (decision_005 clause: ≥5% better than triton_grouped_topk_r2_004.py) — ALL FOUR REQUIRED BASES

1. **Prescribed paired v0-basis** (recorded for completeness): +58.1908% — this measures the accepted rounds-001..004 stack against base.py and MUST NOT be credited to round 005.
2. **Direct same-session pair vs accepted r004** (`auto_bench.time_forward(warmup=50, repeat=100, seed=42)`, ABAB interleaved order control): pass-A r004 `0.19645411521196365` → r005 `0.19653234630823135` = **−0.0398%**; pass-B r004 `0.20089372992515564` → r005 `0.1961914822459221` = **+2.3407%**. The coalescing effect lands in the noise band (~0–4 µs), decisively below the ≈9.85 µs absolute adoption bar.
3. **Cross-anchor vs report_004 wall basis** `0.196909` ms → candidate median `0.198019` = **−0.5640%** (marginally SLOWER).
4. **Manifest anchor cumulative context**: `0.483530` ms → candidate median = +59.0526% total campaign gain banked by rounds 001–004.

Adoption judgment text: paired-basis improvement of the round-005 candidate against the SAME-session accepted reference is statistically indistinguishable from zero on the prescribed basis and between −0.04% and +2.34% on the direct machinery basis — below the 5%/≈9.85 µs bar in every reading. H-005 expected_wall_improvement_pct 6.0 not approached. Classification from verifier output authority: `no-improvement`.

Root-cause sizing (honest mechanistic accounting): the host census proves the dispatcher trip count DID collapse to 2/call exactly as designed, but the runtime still issues ~3 DtoD memcpy submissions and ~7 cudaMemcpyAsync-class runtime calls per call — i.e., the eliminated trips were the CHEAP python-dispatch portions while the expensive runtime submission count stayed constant. The decision's central estimate (~10–17 µs reachable) was therefore optimistic on THIS build: `torch._foreach_copy_` coalesces dispatch, not submission.

## Evaluation Contract Mirror

Every `mechanism_observables[].name` copied verbatim from decision_005.md § Evaluation Contract.

| Observable | Expectation | Observation | Verdict |
|---|---|---|---|
| wall_time_unrounded_paired_median_ms | at least 5% below the accepted reference median across interleaved pairs at warmup 50 / repeat 100 (both bases) | direct basis −0.04%/+2.34%; cross-anchor −0.56% slower | **fail** |
| bitwise_output_equivalence_to_accepted_r004 | bitwise weights+ids on seed42-regime, warm NEW-bytes, stale-trap, all four tie suites through the replayed route, plus run_out==forward poisoned ×2 | True/True everywhere (see Correctness table) | **pass** |
| fallback_tier_selectivity_and_recovery | non-target staged with ZERO artifacts; fp16→eager routing preserved; following target call uses replayed tier again; flags monotone-on-failure only | confirmed | **pass** |
| boundary_host_trip_count_per_call | TWO-BRANCH PASS: (branch A) batched capability binds → census shows ≤2 boundary tensor-op trips/call; OR (branch B) legacy binds with recorded error → documentation-only; failure requires branch-A flag true WITH trips remaining 3, or wall regression alongside neither branch binding | branch A bound; verifier census: ONE `aten::_foreach_copy_` trip/call + single copy-in trip = 2/call (r004 was 3). Failure clause not triggered (trips=2 ≠ 3; wall flat rather than regressed beyond noise) | **pass (branch A)** |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-005`
- intervention: boundary-dispatch coalescing of the manual-replay tier (batched copy-out, non_blocking boundary copies, hot-callable rebinding, guard micro-trim)
- expected_causal_chain: dispatcher-level link OBSERVED (trips 3→2 at python level) but the end-to-end wall link FAILED — removed python dispatch did not translate into wall time because device-submission counts were already minimal and unchanged
- primary_metric: `wall_time`
- Hypothesis verdict: `not confirmed`

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `not resolvable for candidate scope (ZERO cat=kernel events — standing branch-B/unattributable phenomenon under manual replay; attribution scoping contract applies verbatim)`
- iterations: `100` calls per scope
- traces:
  - kernel-mode canonical (candidate scope via ModelNew.run_out/hot path): `log/triton_grouped_topk_r2_005_kernel_100iter.pt.trace.json` @sha256 `ff4ee2dead7e902d2b3f7699ada52160a8732bc382c1b2f0da2fcbb7bcf8c56c`
  - forward dual-scope supplementary: `log/groupedtopk_round005_forward_100iter.pt.trace.json` @sha256 `c3236bd62531893b258c0cbe0af692c6aae3baaa14768e8f89ce87d4ce06abfc`
- scope summaries:
  - `log/summary_round005_reference_forward.json` (canonical PASS): device 181.790185546875 µs/call, 14.96 kernels/call, ratio 0.38380781031285827
  - `log/summary_round005_candidate_forward.json` (canonical PASS: only 4 stray span-edge kernel events = 0.04/call — replay internals invisible here too)
  - kernel-mode candidate scope: canonical tool rc=2 "scope has no kernel events" (standing phenomenon; counted as named attempt P-D)

### Scope Table (wall ms basis: authoritative medians)

| Scope | Attributed kernels/call | Device us/call (cat=kernel) | Wall ms | Device ratio |
|---|---:|---:|---:|---:|
| accepted_reference (same-session forward scope, baseline_base) | 14.96 | 181.790185546875 | 0.473649 | 0.38380781031285827 |
| candidate (kernel-mode via hot path) | 0.00 (unattributable) | not resolvable this round | 0.198019 | n/a per scoping contract |
| candidate (forward cross-check) | 0.04 strays | not resolvable this round | 0.198019 | n/a per scoping contract |

### Host-Side Trip Census (diagnostic evidence; read-only derived from traces; cross-checks Coder's `boundary_trip_census.json`)

| Scope window (100 calls) | window µs | aten::copy_ | aten::_foreach_copy_ | gpu_memcpy DtoD | cuda_runtime-class | cat=kernel |
|---|---:|---:|---:|---:|---:|---:|
| candidate kernel-mode scope | 19179.1 | 300 (≈3/call: copy-in + foreach children) | **100 (=1/call batched out)** | 298 (≈3/call) | 700 | 0 |
| candidate forward scope | 25096.7 | 300 + empty×200 | 100 | 298 | 700 | 4 strays |
| baseline_base reference scope | 86610.4 (host span inflated under profiling) | 400 | — | 100 | 1800 | 1496 |

Census corroboration: python-dispatcher trips collapsed 3→2/call exactly as designed and claimed by the Coder census; BUT underlying gpu_memcpy events remain ≈3/call and runtime submissions ≈7/call — `torch._foreach_copy_` coalesces DISPATCH on this build, NOT submission. This is the quantitative root cause of the noise-band wall outcome.

## Named Failed Probe Attempts and Deviations

| # | Probe / deviation | Command | Exit | Disposition |
|---|---|---|---|---|
| P-D | summarize_trace.py empty-attribution rejection on the kernel-mode candidate scope (standing branch-B phenomenon under manual replay, counted per convention as ONE named attempt) | `summarize_trace.py …/triton_grouped_topk_r2_005_kernel_100iter.pt.trace.json --scope candidate_triton_grouped_topk_r2_005` | rc=2, error `scope has no kernel events: candidate_triton_grouped_topk_r2_005` | Expected per attribution scoping contract; superseded by host-census diagnostics `log/diagnostic_scope_census_round005.json`. No salvage applicable (nothing attributable exists); no measurement/classification impact. |
| D1 | No other deviations. Retired-tier token audits clean; P1/P-C patterns unchanged in kind (P-C's forward-scope variant recurred identically for r005's forward scope within the same family and is covered by P-D's disposition). Zero repairs requested. |

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable (first verification of this candidate) | `cf68ed7713269416af5b49e901e040c7dcb97da9ec4f6eb4cc9bc5d70d288e9c` | correctness PASS throughout; adoption criterion FAILED on wall (noise-band delta); zero repairs requested or applicable |

## evidence_for_next_round

- Observed fact: boundary dispatcher coalescing delivers no measurable wall gain on this build — `torch._foreach_copy_` collapses python dispatch but runtime still submits ≈3 memcpys + graph launch per call; remaining host residue (~93 µs/call) is dominated by those runtime submissions, not python dispatch.
- Observed fact: no-improvement streak is now 1/3 (round 005), canonical remains r004 @0.196909 ms paired basis.
- Observed fact: remaining controllable levers are all gate-limited (CHECK-TIE vendor-entry audit for ~87 µs hidden device work/call) or infrastructure-scale (persistent result buffers to eliminate fresh-buffer allocations would violate Decision ownership lines; multi-graph batching exceeds operator scope).
Evidence only; next-round selection belongs to Designer.

## Stop Recommendation

- recommendation: `continue`
- evidence: streak 1/3 with campaign limits (max_rounds 20, valid-no-improvement streak 3) nowhere near exhausted; CHECK-TIE audit lever documented but currently judged unprofitable by Design economics — Orchestrator owns whether continued rounds remain justified.

Orchestrator owns terminal transitions, counters, and commits (last_accepted stays r004).

## Exact Reproduction Commands

Correctness/strategy-branch probe:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 kernels/track1-triton/groupedtopk/bi150-round2/log/probes/verifier_tie_runout_check_r2_005.py --out kernels/track1-triton/groupedtopk/bi150-round2/log/probes/verifier_tie_runout_result_005.json
```

Screening pairs (identical command twice):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/bi150-round2/triton_grouped_topk_r2_005.py --warmup 10 --repeat 20 --full-traceback
```

Authoritative pairs (identical command three times):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/bi150-round2/triton_grouped_topk_r2_005.py --warmup 50 --repeat 100 --full-traceback
```

Decisive same-session interleaved accepted-pair probe:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 kernels/track1-triton/groupedtopk/bi150-round2/log/probes/verifier_paired_r004_vs_r005_005.py
```

Kernel-mode profile + summaries:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/bi150-round2/triton_grouped_topk_r2_005.py --warmup 5 --repeat 10 --profile --profile-mode kernel --profile-warmup 20 --profile-iterations 100 --profile-output kernels/track1-triton/groupedtopk/bi150-round2/log/triton_grouped_topk_r2_005_kernel_100iter.pt.trace.json --full-traceback
python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/groupedtopk/bi150-round2/log/triton_grouped_topk_r2_005_kernel_100iter.pt.trace.json --iterations 100 --scope candidate_triton_grouped_topk_r2_005 --wall-ms 0.198019   # rc=2 empty-kernel-scope = standing branch-B phenomenon (P-D)
python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/groupedtopk/bi150-round2/log/groupedtopk_round005_forward_100iter.pt.trace.json --iterations 100 --scope baseline_base --wall-ms 0.473649
python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/groupedtopk/bi150-round2/log/groupedtopk_round005_forward_100iter.pt.trace.json --iterations 100 --scope candidate_triton_grouped_topk_r2_005 --wall-ms 0.198019
```

Forward dual-scope supplementary profile:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/bi150-round2/triton_grouped_topk_r2_005.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 100 --profile-output kernels/track1-triton/groupedtopk/bi150-round2/log/groupedtopk_round005_forward_100iter.pt.trace.json
```

## vNext Fact Pack

```json
{
  "schema_version": 1,
  "candidate_sha256": "cf68ed7713269416af5b49e901e040c7dcb97da9ec4f6eb4cc9bc5d70d288e9c",
  "correctness": {
    "status": "pass",
    "evidence": [
      "log/probes/verifier_tie_runout_check_r2_005.py exit 0 -> log/probes/verifier_tie_runout_result_005.json (all_pass=true)",
      "ACTIVE TIER behaviorally determined = manual-replay on both instances (_hot_call present, graph alive); COPY-OUT STRATEGY BRANCH = A-batched (_batched_copyout_ok=True, bind-error artifact None)",
      "seed42-regime + warm NEW-input bytes (seed 31415) + first-input-again stale-trap + all four tie suites: ids exact vs base AND bitwise==r004 True/True through the coalesced hot path",
      "fresh-buffer leak trap: consecutive forwards returned distinct-data_ptr buffers carrying identical bits",
      "selectivity: T=41-first instance created ZERO artifacts and used staged tier bitwise==r004; fp16-gating call routed framework-eager base-consistently; subsequent target call captured and served via manual-replay",
      "run_out vs forward bitwise x2 over poisoned buffers with data_ptr preserved; cross-instance alternation bitwise-correct"
    ]
  },
  "observables": [
    {
      "name": "wall_time_unrounded_paired_median_ms",
      "status": "observed",
      "value": "prescribed protocol: same-run reference 0.473649 -> candidate 0.198019 ms = +58.1908% (legacy-crediting only); DECISIVE same-session interleaved accepted-pair probe: pass-A -0.039821561479276%, pass-B +2.340664231275613%; cross-anchor vs report_004 accepted wall 0.196909 ms: -0.5640% (slower); adoption bar ~=9.85 us absolute NOT cleared",
      "confidence": "high",
      "evidence": [
        "rounds/round_status_005.md authoritative pairs A1/A2/A3",
        "log/probes/verifier_paired_r004_vs_r005_result_005.json"
      ]
    },
    {
      "name": "bitwise_output_equivalence_to_accepted_r004",
      "status": "observed",
      "value": "true on seed42-regime, warm new-input bytes, stale-trap re-inputs, all four tie suites, run_out-vs-forward poisoned x2, cross-instance alternation",
      "confidence": "high",
      "evidence": [
        "log/probes/verifier_tie_runout_result_005.json"
      ]
    },
    {
      "name": "fallback_tier_selectivity_and_recovery",
      "status": "observed",
      "value": "T=41-first instance zero artifacts -> staged bitwise==r004; fp16-gating eager-route case base-consistent; following target call captured and served via manual-replay tier; flags never moved during guard traffic",
      "confidence": "high",
      "evidence": [
        "log/probes/verifier_tie_runout_result_005.json (selectivity record)"
      ]
    },
    {
      "name": "boundary_host_trip_count_per_call",
      "status": "observed",
      "value": "BRANCH A adjudication PASS: verifier host census shows exactly ONE aten::_foreach_copy_ dispatcher trip per call alongside the single copy-in -> python-dispatcher-level boundary trips = 2/call replacing r004 3/call; independent observation: underlying gpu_memcpy DtoD events remain ~3/call and submissions ~7/call, explaining the noise-band wall outcome",
      "confidence": "high",
      "evidence": [
        "log/diagnostic_scope_census_round005.json"
      ]
    }
  ],
  "lowering": {
    "status": "observed",
    "expected_mechanism": "boundary dispatcher-trip coalescing engaged exactly as designed (branch A): _foreach_copy_ binds construction-time and replaces two output copy-out dispatches with one; non_blocking flags present on boundary copies; NO synchronization added; effect honestly sized at noise level because runtime memcpy submission count is unchanged",
    "evidence_contract": "bi150-triton-kernel-summary-v1",
    "evidence": [
      "log/diagnostic_scope_census_round005.json"
    ]
  },
  "evidence_gap_cause": "none"
}
```

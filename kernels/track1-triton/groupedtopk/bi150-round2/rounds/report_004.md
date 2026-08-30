# Report 004

Result: accepted

## Identity

- Round: `004`
- Decision: `rounds/decision_004.md`
- Candidate: `triton_grouped_topk_r2_004.py`
- Accepted reference: `triton_grouped_topk_r2_002.py` (manifest `last_accepted_kernel`; r003 retired evidence — never benchmarked against)
- Accepted reference report: `rounds/report_002.md`
- Decision SHA256: `e5465d7dfdbc35cdba8251b9d43a5d43eb05c64d63c57d89eb299723b0be3be1`
- Sketch SHA256: `ccf277f422ce254d09dc1402c997a6c311a1f63457423f23afd60a71b4d9ae59`
- Binding artifact: `log/probes/binding_statement_report.json` @`1e6b44a5d6db200d91a7686dea39069046e7e184c38de83eb54444a693ddf9bc` (Coder-produced; consumed read-only; DANGER rule 'reduce-overhead' ×0 machine-verified by Coder audit)
- Profile snapshot: `profile_snapshot/triton_cuda.yaml` @`dc8fa4c0c73caecc78c6a886e5ffb18cb97371f3add7cb382dbe30887743b7ae`
- Candidate SHA256: `c02d956c6bb5c27c229623b01b99b85f5962db79b5ead09df6fbca7a52e721eb` (19783 bytes, re-verified after all measurements)
- Accepted reference SHA256: `ad703266eb727f7725c8fa61ceaedcffc269e94291def703cb34279e5275ab12` (unchanged)
- Base SHA256: `12f3324896d6b72bd4eac556839db53fb7045b2965b7f8caf2734551daad0f58` (unchanged)
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (unchanged, AST loader)
- Runtime fingerprint: `project.md#runtime-fingerprint` (CoreX bootstrap every shell; environment unchanged)
- Measurement fingerprint: `8deb1b012de31b18887562e736c7b9e120b9d9f9500230e237ee003c5fa5a431` (unchanged; flags byte-identical to rounds 000–003)
- verification_tier: `authoritative`
- screening_pairs:
  | Pair | Reference short wall ms (--warmup 10 --repeat 20) | Candidate short wall ms | Candidate slower pct |
  |---:|---:|---:|---:|
  | S1 | 0.483520 | 0.204180 | -57.76% (candidate faster) |
  | S2 | 0.484725 | 0.198360 | -59.08% (candidate faster) |

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness through the ACTIVE tier | pass vs base.py semantics under seed42 harness comparator | `PASS accuracy` in all 7 harness invocations; ACTIVE TIER = **manual-replay** (behavioral determination: `_manual_graph` is a real `torch.cuda.CUDAGraph`, workspace present, `_replay_failed=false`, lower-tier handles ABSENT on both instances — tier-1 served all target-regime calls, no down-tier anywhere) | pass | `log/probes/verifier_tie_runout_result_004.json` |
| tie-order spot checks | exact int32 IDs on tie-heavy inputs through the replayed route | all-equal / two-expert-tie-same-group / structured-group-tie-boundary / duplicate-max-pairs-cross-group ids exact vs base | pass | probe cases |
| bitwise equality vs accepted r002 (retention-proof basis) | torch.equal weights+ids vs r002 on seed42 + tie suites through the replayed route; one seed + cheap coverage | CONFIRMED True/True on every case incl. warm NEW-input bytes (seed 31415), the first-input-again stale-trap (proves per-call recomputation from copy-in bits), and pre-timing inside the paired probe | pass | probe JSON + `verifier_paired_r002_vs_r004_result_004.json.bitwise_equal_outputs_before_timing=true` |
| fallback_tier_selectivity_and_recovery | non-target regime uses framework-eager staged tier with ZERO artifacts (workspace included); following target call captures and serves tier-1 again; flags move only on failure | separate instance: T=41 first call left NO graph handle, NO `_ws_gating` attribute, no compilers; staged outputs bitwise==r002; same instance then warmed/captured and served anchors via manual-replay tier | pass | probe selectivity record |
| run_out == forward | bitwise equality into caller buffers via copy-out boundary ×2 poisoned attempts | both attempts bitwise-equal, data_ptr preserved | pass | probe run_out records |
| workspace discipline (Decision ownership supersession) | placeholders fully overwritten each replay; results NEVER returned from workspace (copy-out into invocation-owned buffers); zero cross-call carryover | stale-trap proves fresh computation per input; leak-trap (Coder smoke) shows later forward returns distinct-data_ptr buffers carrying identical bits; verifier probe confirms externally-owned result paths through both entry surfaces | pass | probe + Coder smoke evidence (consumed read-only) |
| both retained top-k sites unchanged | byte-frozen capture of the unmodified eager pipeline incl. both torch.topk sites | six+1 inherited segments BYTE+AST identical per Coder binding statement (six named + forward additionally); captured region contains exactly `_triton_forward` over fixed shapes | pass | binding statement §frozen-segments & workspace-contract spans |
| compile-config discipline | EXACTLY one torch.compile site {mode:'default', dynamic:'False'} as lazy down-tier; 'reduce-overhead' ×0 anywhere | Coder binding statement DANGER rule verified (counts recorded in report artifact) and consistent with source read-through | pass | binding_statement_report.json |
| cold warmup+capture cost placement | outside timed medians; flags unchanged | one-time capture completed inside harness correctness phase/warmup; observation-only sanity: first target-regime forward ≈144.7 ms host wall (cache-warm from Coder smoke session's 139 ms — consistent order); NO flag changes to hide or show it | pass (observed) | probe notes with caveat label |
| environment bootstrap | CoreX before every python; cuda:0 | satisfied everywhere | pass | command history |

## Screening Evidence

Both screening pairs show the candidate >2.36x faster than base.py — screened-out condition not met; authoritative timing proceeded.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved ordered reference/candidate pairs — three identical harness invocations (pair i = run_i.v0_ms then run_i.v1_ms), byte-for-byte identical flags across rounds 000–004`
- reference_raw_samples_ms: `[0.477596, 0.474386, 0.467383]`
- candidate_raw_samples_ms: `[0.197615, 0.196909, 0.195931]`
- reference_median_ms: `0.474386`
- candidate_median_ms: `0.196909`
- improvement_pct (prescribed protocol basis vs same-run v0): `+58.4951`

```text
improvement_pct_protocol_basis = (0.474386 - 0.196909) / 0.474386 * 100 = +58.4951
```

### Adoption Judgment (decision_004 clause: ≥5% better than triton_grouped_topk_r2_002.py) — ALL FOUR REQUIRED BASES

1. **Prescribed paired v0-basis** (headline): vs the SAME-session base.py reference medians inside identical harness invocations → **+58.4951%**.
2. **Direct same-session pair vs accepted r002** (`auto_bench.time_forward(warmup=50, repeat=100, seed=42)`, one process, outputs bitwise-verified equal before timing): r002 `0.3463206812739372` ms → r004 `0.19897893071174622` ms = **+42.54488932633068%** — clears decision_004's ≥5% clause by ~8.5x.
3. **Cross-anchor vs report_002 wall basis** `0.338824` ms → candidate median −41.8851% relative.
4. **Manifest anchor cumulative context**: `0.483530` ms (report_000) → candidate median = **+59.2784%** total campaign gain.

Adoption judgment text: paired-basis improvement of the round-004 candidate against the reference measured in the SAME session pairs is decisively positive on both the prescribed protocol basis (+58.50%) and the accepted-reference machinery path (+42.54%); H-004 expected_wall_improvement_pct 15.0 exceeded ~2.8–3.9x depending on basis. Wall time remains the sole adoption basis.

The measured wall realizes the predicted host-residual compression WITHOUT workspace copies eating it: candidate wall 0.1969 ms carries ~104 µs device work + ~93 µs residual (vs r002's ~235 µs residual at 0.3388 ms) — the ~0.24 ms dispatch residue collapsed to ~0.12 ms while adding only ~O(10 µs)/call of boundary-copy overhead (traces show 3 DtoD memcpys/call).

## Evaluation Contract Mirror

Every `mechanism_observables[].name` copied verbatim from decision_004.md § Evaluation Contract.

| Observable | Expectation | Observation | Verdict |
|---|---|---|---|
| wall_time_unrounded_paired_median_ms | at least 5% below the accepted reference median across interleaved pairs at warmup 50 / repeat 100 | +58.4951% prescribed basis; +42.5449% direct accepted-pair basis | **pass** |
| bitwise_output_equivalence_to_accepted_r002 | bitwise weights+ids on seed42-regime and all four manufactured tie suites through the replayed route, plus run_out==forward over poisoned buffers | True/True everywhere incl. warm new-bytes, stale-trap re-inputs, alternation, run_out ×2 | **pass** |
| fallback_tier_selectivity_and_recovery | non-target regime executes framework-eager staged tier with base-consistent bitwise==r002 outputs; following target call uses replayed tier again; tier flags move downward only on failure | confirmed: zero artifacts for T=41-first instance (workspace attribute absent even); recovery engaged tier-1 capture | **pass** |
| kernel_count_per_call | TWO-BRANCH PASS: (branch A) attributed launches collapse far below 6.90 toward ≤2/call evidencing single-submission replay; OR (branch B) intra-replay launches explicitly unattributable per attribution scoping contract AND candidate scope instead shows single-submission/host-side graph evidence — record branch taken; failure requires attributed count ≈6.90 WITH flat wall | **BRANCH B taken with positive single-submission evidence**: attributed cat=kernel count in candidate kernel-mode scope = ZERO (canonical summarizer: "scope has no kernel events"); intra-replay launches explicitly unattributable on this build; host census `log/diagnostic_scope_census_round004.json` shows the designed structure exactly — ~3 aten::copy_ boundary ops/call (copy-in ×1 + copy-out ×2 → gpu_memcpy 298 events/100 calls), cuda_runtime submissions only, NO other GPU work visible per call while 100 correct results were produced; failure clause moot because wall is decisively NON-flat (+42% vs accepted). Firing evidence AT MEASUREMENT SCALE also carried by the 4-fact behavioral proof + timing collapse | **pass (branch B)** |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-004`
- intervention: one-time manual torch.cuda.CUDAGraph workspace capture of the accepted staged pipeline; per-call guard + static copy-in + ONE replay submission + two small copy-outs
- expected_causal_chain: observed END-TO-END for the first time in this campaign — per-call Python op-dispatch, allocation planning, and ~6.9 per-launch submissions collapsed into a single graph submission; host residue shrank roughly half while device work stayed essentially identical (~104 µs band inferred by construction + flat behavior; per-kernel device time itself unattributable this round per contract)
- primary_metric: `wall_time`
- Hypothesis verdict: `confirmed`

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `partially available: accepted-reference scope fully attributed (cat=kernel normal); CANDIDATE scope contains ZERO cat=kernel events — intra-replay launches unattributable on this build (declared branch B), so candidate device totals are NOT resolvable from traces this round; host-side census substitutes diagnostically`
- iterations: `100` calls per scope
- traces:
  - kernel-mode canonical (candidate scope via ModelNew.run_out): `log/triton_grouped_topk_r2_004_kernel_100iter.pt.trace.json` @sha256 `54d22cb150435626706879f0ba46c918c9f3e48d95cfa69e6100be373fb393d3`
  - forward dual-scope supplementary: `log/groupedtopk_round004_forward_100iter.pt.trace.json` @sha256 `6bc6fb1018c4d1cb95682a4e3cf32799ca8b6387bf0366b089802ddf81714e46`
- scope summaries:
  - `log/summary_round004_reference_forward.json` (canonical tool, PASS)
  - `log/summary_round004_candidate_forward.json` (canonical tool: 3 stray span-edge kernel events = 0.03/call — replay internals invisible here too)
  - kernel-mode candidate scope: canonical tool rc=2 "scope has no kernel events" (the branch-B phenomenon itself); superseded by census `log/diagnostic_scope_census_round004.json`

### Scope Table (wall ms basis: authoritative medians)

| Scope | Attributed kernels total | Attributed kernels/call | Device us/call (cat=kernel) | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|
| accepted_reference (same-session forward scope, baseline_base) | 1497 | 14.97 | 179.871337890625 | 0.474386 | 0.37916662357368264 |
| candidate (kernel-mode via run_out) | 0 | 0.00 (unattributable — branch B) | not resolvable this round | 0.196909 | n/a per scoping contract |
| candidate (forward cross-check) | 3 strays | 0.03 (unattributable — branch B) | not resolvable this round | 0.196909 | n/a per scoping contract |

### Host-Side Census (diagnostic evidence, read-only derived from traces)

| Scope window | window count | aten::copy_ (cpu_op) | cudaMemcpyAsync-class runtime | gpu_memcpy DtoD | cat=kernel |
|---|---:|---:|---:|---:|---:|
| candidate kernel-mode scope (100 replays) | 1 (18761 µs) | 300 (≈3/call: copy-in + 2 copy-outs) | 700 | 298 | **0** |
| candidate forward scope (100 iterations) | 1 (25651 µs) | 500 (≈5/call: 3 copies + 2 fresh buffer allocs) | 700 | 298 | 3 stray span-edge events |
| baseline_base reference scope (100 iterations) | 1 (82781 µs — profiling overhead inflates host spans) | 4300 | 1800 | 100 | 1497 |

This census IS the firing evidence at measurement scale: 100 consecutive correct results were produced by ~3 boundary memcpys/call and graph-resident work invisible to per-kernel attribution — the single-submission replay signature (round-002 structure would have shown ~700 cat=kernel launches per scope like its own references do).

### Accepted Reference Top Kernels (forward scope)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `at::native::sbtopk::gatherTopK` | 200 | 2.00 | 4973.53 | 49.735 |
| `at::native::bitonicSortKVInPlace` | 200 | 2.00 | 3774.52 | 37.745 |
| `at::native::reduce_kernel MaxOps` | 100 | 1.00 | 1855.60 | 18.556 |
| `at::native::reduce_kernel sum_functor` | 99 | 0.99 | 1360.95 | 13.747 |

(Candidate per-kernel table intentionally absent this round: unattributable per branch B; retention proven by output identity.)

## Named Failed Probe Attempts and Deviations

| # | Probe / deviation | Command | Exit | Disposition |
|---|---|---|---|---|
| P-C | summarize_trace.py correctly rejected an EMPTY-attribution scope (this IS the branch-B phenomenon surfacing in tooling, counted per established convention as one named attempt) | `summarize_trace.py …/triton_grouped_topk_r2_004_kernel_100iter.pt.trace.json --scope candidate_triton_grouped_topk_r2_004` | rc=2, error `scope has no kernel events: candidate_triton_grouped_topk_r2_004` | Root cause is the intended mechanism (graph internals unattributable under manual replay — unlike rounds 001–003 where launches appeared individually). Read-only category census `log/diagnostic_scope_census_round004.json` documents the full contained-event structure (memcpys + copies only). No salvage applicable or needed: attribution collapses to branch B BY DESIGN of the Decision. No measurement/classification impact. |
| D1 | No other deviations. P1 double-record salvage unused this round. Zero repairs requested. |

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable (first verification of this candidate) | `c02d956c6bb5c27c229623b01b99b85f5962db79b5ead09df6fbca7a52e721eb` | correctness PASS throughout; adoption DECISIVELY met (+58.50%/+42.54%) — zero repairs requested or needed |

## evidence_for_next_round

- Observed fact: manual workspace replay is the first mechanism to break below 0.2 ms wall (0.196909 median): residual is now ~93 µs/call of host time outside GPU-visible work — remaining levers are sub-round-scale (batching the two copy-outs into one fused copy, guard micro-costs, allocator chatter for the two fresh buffers).
- Observed fact: non-target regimes still pay full staged/eager cost (0.30–0.40 ms sanity level, now dominated by the pipeline that tier-1 encapsulates).
- Observed fact: campaign cumulative gain vs manifest anchor reaches +59.28%; gate-limited future headroom lives behind CHECK-TIE audit (vendor top-k sites, ~87 µs of hidden device work/call) or beyond-decision-scope infrastructure changes.
Evidence only; next-round selection belongs to Designer.

## Stop Recommendation

- recommendation: `continue`
- evidence: round 004 cleared adoption by wide margin on all bases; streak counters reset semantics owned by Orchestrator; manifest limits not near exhaustion.

Orchestrator owns terminal transitions, last_accepted pointer updates, counters/streak accounting, and commits.

## Exact Reproduction Commands

Correctness/active-tier probe:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 kernels/track1-triton/groupedtopk/bi150-round2/log/probes/verifier_tie_runout_check_r2_004.py --out kernels/track1-triton/groupedtopk/bi150-round2/log/probes/verifier_tie_runout_result_004.json
```

Screening pairs (identical command twice):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/bi150-round2/triton_grouped_topk_r2_004.py --warmup 10 --repeat 20 --full-traceback
```

Authoritative pairs (identical command three times):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/bi150-round2/triton_grouped_topk_r2_004.py --warmup 50 --repeat 100 --full-traceback
```

Decisive same-session accepted-pair probe:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 kernels/track1-triton/groupedtopk/bi150-round2/log/probes/verifier_paired_r002_vs_r004_004.py
```

Kernel-mode profile (candidate scope) + reference summary:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/bi150-round2/triton_grouped_topk_r2_004.py --warmup 5 --repeat 10 --profile --profile-mode kernel --profile-warmup 20 --profile-iterations 100 --profile-output kernels/track1-triton/groupedtopk/bi150-round2/log/triton_grouped_topk_r2_004_kernel_100iter.pt.trace.json --full-traceback
python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/groupedtopk/bi150-round2/log/triton_grouped_topk_r2_004_kernel_100iter.pt.trace.json --iterations 100 --scope candidate_triton_grouped_topk_r2_004 --wall-ms 0.196909   # rc=2 empty-kernel-scope = branch B (see deviations P-C)
python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/groupedtopk/bi150-round2/log/groupedtopk_round004_forward_100iter.pt.trace.json --iterations 100 --scope baseline_base --wall-ms 0.474386
python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/groupedtopk/bi150-round2/log/groupedtopk_round004_forward_100iter.pt.trace.json --iterations 100 --scope candidate_triton_grouped_topk_r2_004 --wall-ms 0.196909
```

Forward dual-scope supplementary profile:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/bi150-round2/triton_grouped_topk_r2_004.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 100 --profile-output kernels/track1-triton/groupedtopk/bi150-round2/log/groupedtopk_round004_forward_100iter.pt.trace.json
```

## vNext Fact Pack

```json
{
  "schema_version": 1,
  "candidate_sha256": "c02d956c6bb5c27c229623b01b99b85f5962db79b5ead09df6fbca7a52e721eb",
  "correctness": {
    "status": "pass",
    "evidence": [
      "log/probes/verifier_tie_runout_check_r2_004.py exit 0 -> log/probes/verifier_tie_runout_result_004.json (all_pass=true)",
      "ACTIVE TIER behaviorally determined = manual-replay (real torch.cuda.CUDAGraph handle + workspace present, lower-tier handles absent both instances)",
      "seed42-regime + warm NEW-input bytes (seed 31415) + first-input-again stale-trap + all four tie suites: ids exact vs base AND bitwise==r002 True/True through the replayed route",
      "selectivity: separate instance first call T=41 created ZERO artifacts (no graph handle, NO workspace attribute, no compilers); same instance then captured and served tier-1 anchors",
      "run_out vs forward bitwise x2 over poisoned buffers with data_ptr preserved; cross-instance alternation bitwise-correct"
    ]
  },
  "observables": [
    {
      "name": "wall_time_unrounded_paired_median_ms",
      "status": "observed",
      "value": "prescribed protocol: same-run reference 0.474386 -> candidate 0.196909 ms = +58.4951%; supplementary same-session accepted-pair r002 0.3463206812739372 -> r004 0.19897893071174622 = +42.54488932633068%",
      "confidence": "high",
      "evidence": [
        "rounds/round_status_004.md authoritative pairs A1/A2/A3",
        "log/probes/verifier_paired_r002_vs_r004_result_004.json"
      ]
    },
    {
      "name": "bitwise_output_equivalence_to_accepted_r002",
      "status": "observed",
      "value": "true on seed42-regime, warm new-input bytes, all four tie suites, run_out-vs-forward poisoned x2, cross-instance alternation, and stale-trap input-dependence",
      "confidence": "high",
      "evidence": [
        "log/probes/verifier_tie_runout_result_004.json"
      ]
    },
    {
      "name": "fallback_tier_selectivity_and_recovery",
      "status": "observed",
      "value": "T=41-first instance created zero artifacts (no workspace attribute even) and used the framework-eager staged tier bitwise==r002; following target call warmed/captured and served tier-1; tier flags never moved during guard-routed traffic",
      "confidence": "high",
      "evidence": [
        "log/probes/verifier_tie_runout_result_004.json (selectivity record)"
      ]
    },
    {
      "name": "kernel_count_per_call",
      "status": "observed",
      "value": "BRANCH B adjudication: attributed cat=kernel events are ZERO inside the candidate kernel-mode scope (canonical summarizer reports scope has no kernel events); intra-replay launches explicitly unattributable on this build; single-submission firing evidenced instead by host census log/diagnostic_scope_census_round004.json: exactly ~3 aten::copy_ DtoD boundary memcpys/call (1 copy-in + 2 copy-out) and no other GPU work visible per call while 100 correct results were produced",
      "confidence": "high",
      "evidence": [
        "log/diagnostic_scope_census_round004.json"
      ]
    }
  ],
  "lowering": {
    "status": "observed",
    "expected_mechanism": "manual torch.cuda.CUDAGraph workspace capture/replay executed the byte-frozen staged pipeline as ONE submission per call behind static addresses: traces show only boundary aten::copy_ ops and DtoD memcpys inside the candidate scope (zero individually-attributed kernels — attribution superseded by the decision attribution-scoping contract whose retention proof is the bitwise-vs-r002 sweep)",
    "evidence_contract": "bi150-triton-kernel-summary-v1",
    "evidence": [
      "log/diagnostic_scope_census_round004.json",
      "log/probes/verifier_tie_runout_result_004.json"
    ]
  },
  "evidence_gap_cause": "none"
}
```

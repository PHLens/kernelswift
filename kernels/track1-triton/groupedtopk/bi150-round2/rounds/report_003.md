# Report 003

Result: no-improvement

## Identity

- Round: `003`
- Decision: `rounds/decision_003.md`
- Candidate: `triton_grouped_topk_r2_003.py`
- Accepted reference: `triton_grouped_topk_r2_002.py` (manifest `last_accepted_kernel`; prescribed protocol pairs base.py as v0 per campaign convention)
- Accepted reference report: `rounds/report_002.md`
- Decision SHA256: `e214c29aa66d78654ffb65fba33b4870379bcf059902c8f7cc6409ebffc3a403`
- Sketch SHA256: `4a909a11cbd8df0ad0385cf6379dc77eb189bffd60ec2ab1b341dbdaa127a782`
- Binding artifact: `log/probes/binding_statement_report.json` @`b32eb677d43b7d2ad51cb4ec140aae4661495a1ce027098c2ff77301adafe1c7` (Coder-produced; consumed read-only)
- Profile snapshot: `profile_snapshot/triton_cuda.yaml` @`dc8fa4c0c73caecc78c6a886e5ffb18cb97371f3add7cb382dbe30887743b7ae`
- Candidate SHA256: `62f8883a2c6d1bdf65d84b29beb71d95500b40b8d6acaf484eb09fccdcf97d38` (16904 bytes, re-verified after all measurements)
- Accepted reference SHA256: `ad703266eb727f7725c8fa61ceaedcffc269e94291def703cb34279e5275ab12` (unchanged)
- Base SHA256: `12f3324896d6b72bd4eac556839db53fb7045b2965b7f8caf2734551daad0f58` (unchanged)
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (unchanged, AST loader)
- Runtime fingerprint: `project.md#runtime-fingerprint` (CoreX bootstrap every shell; environment unchanged)
- Measurement fingerprint: `8deb1b012de31b18887562e736c7b9e120b9d9f9500230e237ee003c5fa5a431` (unchanged; flags byte-identical to rounds 000–002)
- verification_tier: `authoritative`
- screening_pairs:
  | Pair | Reference short wall ms (--warmup 10 --repeat 20) | Candidate short wall ms | Candidate slower pct |
  |---:|---:|---:|---:|
  | S1 | 0.487853 | 0.376550 | -22.81% (candidate faster) |
  | S2 | 0.484471 | 0.373808 | -22.87% (candidate faster) |

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness through the ACTIVE tier | pass vs base.py semantics under seed42 harness comparator | `PASS accuracy` in all 7 harness invocations; probe ids exact vs base on every case; ACTIVE TIER = `replayed` (tier-1) served all target-regime calls — behavioral determination: main-instance flags `{replay_failed:false, replayed_handle_present:true, compile_failed:false, compiled_handle_present:false}` after traffic; no silent down-tier anywhere | pass | `log/probes/verifier_tie_runout_result_003.json` |
| tie-order spot checks | exact int32 IDs on tie-heavy inputs through active tier | all-equal / two-expert-tie-same-group / structured-group-tie-boundary / duplicate-max-pairs-cross-group ids exact vs base | pass | probe cases |
| bitwise equality vs accepted r002 (retention-proof basis per Decision §attribution-scoping contract) | weights+ids torch.equal to r002 outputs on seed42 + all four tie suites THROUGH THE REPLAYED ROUTE | CONFIRMED True/True on every case incl. warm NEW-input bytes (seed 31415); also pre-timing inside the paired probe | pass | probe JSON + `verifier_paired_r002_vs_r003_result_003.json.bitwise_equal_outputs_before_timing=true` |
| fallback_tier_selectivity_and_recovery | non-target regime uses eager staged tier with ZERO compiler artifacts; following target call re-engages tier-1; monotonic-only flag transitions | separate instance: T=41 first call left ALL handles None / flags false (no artifacts constructed), staged outputs bitwise==r002; same instance then entered replayed tier on [83,256] and matched anchors; `_replay_failed/_compile_failed` never moved during guard-routed traffic | pass | probe selectivity record |
| run_out == forward | bitwise equality into caller buffers ×2 poisoned attempts | both attempts bitwise-equal weights+ids, data_ptr preserved (zero-copy plumbing) | pass | probe run_out records |
| cross-instance alternation | two instances interleaved remain per-input-anchor correct | main and selectivity instances alternate on distinct inputs; all four pairwise comparisons bitwise==r002 anchors | pass | probe alternation record |
| both retained top-k sites unchanged | identical arguments/shapes/ordering/tie behavior | seven inherited segments machine-proven byte-frozen (Coder binding statement #1); device trace shows same two vendor kernels, no new selection kernel | pass | binding statement + profiler scopes |
| compile-config supersession discipline | mode token changed ONLY as authorized ('reduce-overhead' ×1 kwarg + 'default' ×1 legacy-tier kwarg; dynamic=False ×2; no other knobs) | binding statement mode-token audit passes; forbidden-token scan all-zero | pass | binding_statement_report.json |
| framework hazard surveillance (Decision pitfalls i–iv) | no pool-output leakage, no recapture pressure, live input bytes honored | cold capture completed (sanity 313.2 ms observation; coder smoke 562 ms — cache-warm variance, non-performance); warm repeat bit-identical; fresh externally-owned buffers returned every call; framework emitted `skipping cudagraphs due to mutated inputs (2 instances)` in EVERY harness invocation (documented diagnostically below) | pass (with documented sub-behavior) | probe notes + stderr captures |
| environment bootstrap | CoreX before every python; cuda:0 | satisfied everywhere | pass | command history |

## Screening Evidence

Both screening pairs show the candidate ~23% faster than base.py — screened-out condition not met; authoritative timing proceeded. (As established in round 001, screening speedup vs base.py is a legacy-basis artifact once prior accepted rounds already banked large gains.)

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved ordered reference/candidate pairs — three identical harness invocations (pair i = run_i.v0_ms then run_i.v1_ms), byte-for-byte identical flags across rounds 000–003`
- reference_raw_samples_ms: `[0.481532, 0.478203, 0.452523]`
- candidate_raw_samples_ms: `[0.374760, 0.374314, 0.353708]`
- reference_median_ms: `0.478203`
- candidate_median_ms: `0.374314`
- improvement_pct (prescribed protocol basis vs same-run v0): `+21.7262`

```text
improvement_pct_protocol_basis = (0.478203 - 0.374314) / 0.478203 * 100 = +21.7262
```

### Adoption Judgment (decision_003 clause: ≥5% better than triton_grouped_topk_r2_002.py)

- DECISIVE same-session direct pair (`auto_bench.time_forward(warmup=50, repeat=100, seed=42)`, one process, outputs bitwise-verified equal before timing): accepted r002 `0.3451220691204071` ms → candidate r003 `0.373033806681633` ms = **−8.087497166542533%** — the candidate is SLOWER than last_accepted, not merely under-bar.
- Cross-anchor: candidate median `0.374314` vs report_002 recorded r002 wall basis `0.338824` ms → **+10.4740% slower**.
- Cumulative-lens (for counters): vs manifest-of-record anchor `0.483530` ms → +22.5817% total campaign gain remains banked, but round 003 contributed NEGATIVELY relative to its own accepted predecessor.
- Verdict: H-003's ≥5% clause FAILED with negative sign; expected_wall_improvement_pct 15.0 not approached. Classification from verifier output authority: `no-improvement`. Wall time remains the sole adoption basis; the protocol-basis +21.73% figure is recorded for completeness and MUST NOT be credited to round 003 — it measures the already-accepted round-001/002 stack against base.py.

Root cause consistent with the disclosed elision risk: the framework logged `skipping cudagraphs due to mutated inputs (2 instances)` on EVERY invocation, meaning CUDA-graph replay never fired for buffer-carrying invocations under this CoreX build; reduce-overhead machinery then costs pure overhead (graph-tree wrappers/pool bookkeeping) versus plain default-mode compilation.

## Evaluation Contract Mirror

Every `mechanism_observables[].name` copied verbatim from decision_003.md § Evaluation Contract.

| Observable | Expectation (verbatim intent) | Observation | Verdict |
|---|---|---|---|
| wall_time_unrounded_paired_median_ms | at least 5% below the accepted reference median across interleaved pairs at warmup 50 / repeat 100 | same-session accepted-pair −8.0875%; cross-session −10.47% (slower); prescribed-basis +21.73% is legacy-crediting only | **fail** |
| bitwise_output_equivalence_to_accepted_r002 | weights and ids bitwise-equal to accepted r002 on seed42-regime and all four tie suites through the replayed route, plus run_out==forward bitwise over poisoned buffers | True/True on every case; run_out ×2 bitwise with data_ptr preserved; warm new-input bytes case included | **pass** |
| fallback_tier_selectivity_and_recovery | non-target regime executes eager staged tier with base-consistent bitwise==r002 outputs; following target call uses replayed tier again; flags monotonic-on-failure only | confirmed exactly; zero artifacts for T=41-first instance; recovery engaged tier-1; no flag movement during guard traffic | **pass** |
| kernel_count_per_call | TWO-BRANCH PASS: (branch A) attributed launches decrease below 6.90/call OR stay ≤6.90; OR (branch B) intra-replay launches explicitly unattributable — failure requires attributed count EXCEEDING 6.90/call | attributed 6.94/call kernel-mode (7.00 forward); NOTE: composition/name-set byte-identical to r002 (gatherTopK 1.98, bitonicSortKVInPlace 1.98, stages 1.00/0.99/0.99; NO new kernels); the literal exceedance equals exactly 3 span-edge events across the 700-launch window (round-to-round jitter band observed all campaigns: vendor pair totals ranged 197–200). Strict letter: exceeds 6.90 → counted **fail-by-letter, mechanism-neutral** (no substitution or added work evident; see Diagnostics — graphs never captured, so launch plan is the compiled-default one measured identically) | fail (letter; annotated) |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-003`
- intervention: escalate accepted compiled pipeline to mode='reduce-overhead' via authorized mode-only supersession, three-tier permanent fallback chain
- expected_causal_chain: NOT observed end-to-end — the first causal link (inductor wraps region for CUDA-graph capture; subsequent calls submit ONE graph launch) did not occur on this build/harness pattern: the framework skipped graph capture for mutated-input graphs every time, leaving dispatch largely as in round 002 plus wrapper overhead
- primary_metric: `wall_time`
- Hypothesis verdict: `not confirmed` (decisive wall criterion failed with negative sign; two secondary guardrail-style observables passed but cannot rescue adoption per wall-time-only rule)

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `available (cat=kernel events attributed normally — graphs were NOT captured, so nothing became unattributable)`
- iterations: `100` calls per scope
- traces:
  - kernel-mode canonical (candidate scope via ModelNew.run_out): `log/triton_grouped_topk_r2_003_kernel_100iter.pt.trace.json` @sha256 `b5c57d6c06b7f44d9466a3910cc542e4752933dd9d2170317aa0ed8f768e5613`
  - forward dual-scope supplementary: `log/groupedtopk_round003_forward_100iter.pt.trace.json` @sha256 `4a3eb36ae5ab2b31f6c45eb05880f3724e48739d4014df2dedd50caa1aad1235`
- scope summaries — ALL canonical summarize_trace.py first-attempt (P1 double-record pattern did not recur):
  - `log/summary_round003_candidate_kernelmode.json`
  - `log/summary_round003_reference_forward.json`
  - `log/summary_round003_candidate_forward.json`

### Scope Table (wall ms basis: authoritative medians)

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference (same-session forward scope, baseline_base) | 18133.0908203125 | 181.330908203125 | 1490 | 14.90 | 0.478203 | 0.3791923266962462 |
| candidate (kernel-mode via run_out) | 10500.5458984375 | 105.005458984375 | 694 | 6.94 | 0.374314 | 0.2805277360301111 |
| candidate (forward cross-check) | 10525.236328125 | 105.25236328125 | 700 | 7.00 | 0.374314 | 0.2811873541498582 |

Device time stayed flat (105.01 vs accepted-r002 103.99 µs/call, +1.0%) while WALL REGRESSED −8.09% vs accepted — direct confirmation the round's mechanism (dispatch elision) failed to engage: host-side cost INCREASED without any device work change.

### Diagnostics: which sub-behavior occurred (mandated documentation)

- Framework message `skipping cudagraphs due to mutated inputs (2 instances)` appeared in EVERY harness invocation this round including both profiling runs (stderr capture counts ≥1 each).
- Resolution: mutation-skip DEMOTED cudagraph capture to default-equivalent compiled execution for the buffer-carrying invocations (stage-C writes into caller-visible out buffers are input-mutations inside the traced region). Replay therefore NEVER fired despite the nominal tier remaining `replayed` (tier-1 handle healthy, no exception path taken).
- Consequently the decision's branch-B (intra-replay unattributability) is MOOT — there was no replay to blur attribution; traces attribute launches individually and match the compiled-default structure of round 002 exactly.
- Net effect of the round: added graph-pool/wrapper checks + skip handling ⇒ small but real host-side regression; bitwise values unaffected (proven independently).

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable (first verification of this candidate) | `62f8883a2c6d1bdf65d84b29beb71d95500b40b8d6acaf484eb09fccdcf97d38` | correctness PASS throughout; adoption criterion FAILED on wall; zero repairs requested or applicable (nothing structurally broken) |

Named failed probe attempts this round: none (all probes ran clean on first attempt; previous-round P1/P-A patterns did not recur).

## evidence_for_next_round

- Observed fact: on this CoreX build + harness pattern, buffer-carrying invocations (any caller-visible buffer written inside the traced region) are systematically excluded from CUDA-graph capture by the framework's mutated-input check; reduce-overhead escalation therefore yields overhead without replay benefit here.
- Observed fact: rejected-lever lineage stands — without replay firing, round-002's compiled-default route remains the best-known configuration for this operator family at [83,256]/[83,7168].
- Observed fact: candidate-vs-base landscape unchanged: +21.7% protocol-basis gain belongs to the accepted stack; further host compression requires either capture-compatible buffer strategy (framework-specific, currently blocker-gated) or selection-site work behind the CHECK-TIE audit gate.
Evidence only; next-round selection belongs to Designer.

## Stop Recommendation

- recommendation: `continue`
- evidence: one no-improvement round recorded; contract's valid-no-improvement streak budget (3) barely touched; other levers (CHECK-TIE audit, capture-compatible buffership redesign) remain open and documented.

Orchestrator owns terminal transitions, last_accepted pointer (stays at r002), counters/streak accounting, and commits.

## Exact Reproduction Commands

Correctness/active-tier probe:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 kernels/track1-triton/groupedtopk/bi150-round2/log/probes/verifier_tie_runout_check_r2_003.py --out kernels/track1-triton/groupedtopk/bi150-round2/log/probes/verifier_tie_runout_result_003.json
```

Screening pairs (identical command twice):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/bi150-round2/triton_grouped_topk_r2_003.py --warmup 10 --repeat 20 --full-traceback
```

Authoritative pairs (identical command three times):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/bi150-round2/triton_grouped_topk_r2_003.py --warmup 50 --repeat 100 --full-traceback
```

Decisive same-session accepted-pair probe:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 kernels/track1-triton/groupedtopk/bi150-round2/log/probes/verifier_paired_r002_vs_r003_003.py
```

Kernel-mode profile + summaries:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/bi150-round2/triton_grouped_topk_r2_003.py --warmup 5 --repeat 10 --profile --profile-mode kernel --profile-warmup 20 --profile-iterations 100 --profile-output kernels/track1-triton/groupedtopk/bi150-round2/log/triton_grouped_topk_r2_003_kernel_100iter.pt.trace.json --full-traceback
python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/groupedtopk/bi150-round2/log/triton_grouped_topk_r2_003_kernel_100iter.pt.trace.json --iterations 100 --scope candidate_triton_grouped_topk_r2_003 --wall-ms 0.374314
```

Forward dual-scope supplementary + summaries:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/bi150-round2/triton_grouped_topk_r2_003.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 100 --profile-output kernels/track1-triton/groupedtopk/bi150-round2/log/groupedtopk_round003_forward_100iter.pt.trace.json
python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/groupedtopk/bi150-round2/log/groupedtopk_round003_forward_100iter.pt.trace.json --iterations 100 --scope baseline_base --wall-ms 0.478203
python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/groupedtopk/bi150-round2/log/groupedtopk_round003_forward_100iter.pt.trace.json --iterations 100 --scope candidate_triton_grouped_topk_r2_003 --wall-ms 0.374314
```

## vNext Fact Pack

```json
{
  "schema_version": 1,
  "candidate_sha256": "62f8883a2c6d1bdf65d84b29beb71d95500b40b8d6acaf484eb09fccdcf97d38",
  "correctness": {
    "status": "pass",
    "evidence": [
      "log/probes/verifier_tie_runout_check_r2_003.py exit 0 -> log/probes/verifier_tie_runout_result_003.json (all_pass=true)",
      "ACTIVE TIER determined behaviorally = 'replayed' (tier-1 handle present, _replay_failed=false) serving all target-regime calls on both instances",
      "seed42-regime + warm NEW-input bytes (seed 31415) + all four tie suites: ids exact vs base AND bitwise weights+ids equal to accepted r002 through the active tier",
      "selectivity: separate instance first call T=41 created zero compiler artifacts (handles None/flags false), staged outputs bitwise==r002; same instance then engaged tier-1 on [83,256]",
      "run_out vs forward bitwise x2 over poisoned buffers with data_ptr preserved; cross-instance alternation bitwise-correct"
    ]
  },
  "observables": [
    {
      "name": "wall_time_unrounded_paired_median_ms",
      "status": "observed",
      "value": "DECISIVE same-session accepted-pair: r002 0.3451220691204071 -> r003 0.373033806681633 ms = -8.087497166542533% (REGRESSION vs last_accepted); cross-anchor vs report_002 wall 0.338824: +10.4740% slower; prescribed protocol basis vs base.py 0.478203 -> 0.374314 = +21.7262% (legacy-crediting only, must not be attributed to round 003)",
      "confidence": "high",
      "evidence": [
        "rounds/round_status_003.md authoritative pairs A1/A2/A3",
        "log/probes/verifier_paired_r002_vs_r003_result_003.json"
      ]
    },
    {
      "name": "bitwise_output_equivalence_to_accepted_r002",
      "status": "observed",
      "value": "true on seed42-regime, warm new-input bytes (seed 31415), all four tie suites, run_out-vs-forward poisoned x2, and cross-instance alternation",
      "confidence": "high",
      "evidence": [
        "log/probes/verifier_tie_runout_result_003.json",
        "log/probes/verifier_paired_r002_vs_r003_result_003.json"
      ]
    },
    {
      "name": "fallback_tier_selectivity_and_recovery",
      "status": "observed",
      "value": "T=41-first instance created zero compiler artifacts and used staged tier (bitwise==r002); subsequent target call engaged tier-1; tier flags never moved during guard-routed traffic",
      "confidence": "high",
      "evidence": [
        "log/probes/verifier_tie_runout_result_003.json (selectivity record)"
      ]
    },
    {
      "name": "kernel_count_per_call",
      "status": "observed",
      "value": "attributed 6.94/call kernel-mode (7.00 forward cross-check) vs accepted-r002 basis 6.90; literal exceedance = exactly 3 span-edge events across the 700-launch window (vendor pair 198 vs 197 each, stage-A 100 vs 99) with composition/name-set byte-identical to r002 and no new kernels; two-branch rule letter tripped but mechanism-neutral",
      "confidence": "high",
      "evidence": [
        "log/summary_round003_candidate_kernelmode.json",
        "log/summary_round003_candidate_forward.json"
      ]
    }
  ],
  "lowering": {
    "status": "observed",
    "expected_mechanism": "CUDA-graph capture-replay dispatch elision did NOT fire: framework emitted 'skipping cudagraphs due to mutated inputs (2 instances)' in every invocation; mutation-skip demoted capture to default-equivalent compiled execution, leaving per-launch dispatch structure identical to accepted r002 plus wrapper overhead (device flat 105.01 vs 103.99 us/call while wall regressed)",
    "evidence_contract": "bi150-triton-kernel-summary-v1",
    "evidence": [
      "log/summary_round003_candidate_kernelmode.json",
      "log/summary_round003_candidate_forward.json",
      "stderr captures reproduced in rounds/report_003.md Diagnostics section"
    ]
  },
  "evidence_gap_cause": "none"
}
```

# Decision 006

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"006","reference_implementation":"triton_grouped_topk_r2_004.py","reference_report":"rounds/report_004.md","language":"triton","backend":"cuda","target_profile":"triton_cuda","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-change"}
```

## Optimization Intent

```json
{
  "bottleneck_class": "none",
  "intervention": "STOP: no remaining legal mechanism family plausibly clears the 5% adoption bar versus the accepted manual-replay canonical at its re-priced basis (>=9.85 us absolute against a 0.196909 ms wall). The five-round evidence ledger closes every family: (1) replay-overhead escalation CLOSED by structural root cause (report_003: inductor 'skipping cudagraphs due to mutated inputs' every invocation); (2) boundary python-side coalescing NULL-EFFECT by build behavior (report_005: _foreach_copy_ merges dispatcher trips only — gpu_memcpy DtoD events remain ~3/call and cudaMemcpyAsync-class submissions ~7/call, so the bar was never python-trip approachable); (3) Inductor-default-beneath trims EXHAUSTED by round 002's demonstrated share; (4) tie-gated vendor-topk entry WRITTEN OFF on certifiability grounds twice with unchanged evidence (implementation-emergent sbtopk/bitonic tie permutations uncertifiable on Triton primitives, plus cross-implementation score-bit ambiguity between a tl.exp softmax path and torch.softmax bits that survives any permutation replica, plus unprofitable audit-slot economics at miss-streak 1/3); (5) captured-pipeline internals are frozen byte-exact by design and their device time is barrier-locked around two library selections; therefore no falsifiable five-percent hypothesis exists within role scope",
  "allowed_changes": [],
  "invariants": [
    "canonical deliverable remains last_accepted_kernel triton_grouped_topk_r2_004.py @c02d956c6bb5c27c229623b01b99b85f5962db79b5ead09df6fbca7a52e721eb with report_004 @c79cc018f9c61ec34f084fc589b06b61d9b8e9ba634710d2ba365e3d1c34fe35",
    "campaign committed state preserved: wall trajectory 0.483530 ms anchor -> 0.196909 ms final (+59.28%, 2.41x), correctness green through the replayed tier incl. all manufactured tie suites",
    "workspace/ownership discipline, three-tier fallback chain, stream/device semantics of the accepted architecture remain binding for any future work in this campaign directory",
    "measurement fingerprint 8deb1b01... and all attribution scoping contracts remain of record for any reopening"
  ],
  "expected_wall_improvement_pct": 0.0
}
```

## Unified Sketch

N/A: aborted

## Host Plan

```json
{"applicability":"not-applicable","reason":"aborted"}
```

## Evaluation Contract

```json
{"applicability":"not-applicable","reason":"aborted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md` in full; consistent with the template abort clause — all remaining paths repeat measured failures or lack a falsifiable five-percent hypothesis. Enumerated disposition of each conceivable family, so no successor re-litigates blind:
- **Vendor-top-k replacement (the only large lever, ~87 us/call device inside two `torch.topk` sites)** — written off on certifiability grounds in rounds 001/002 and unchanged since: observed tie orders from the vendor path (all-equal `[7,6,4,5,1,0,2,3]`; structured-group `[32,0,64,96,4,3,1,2]`) are implementation-emergent properties of sbtopk/gatherTopK+bitonicSortKVInPlace internals, not reproducible from any composition of Triton primitives proven-or-constrained on the frozen profile (`tl.argmax` unique-maximum-(8,)-only; repeated selection Constrained); independently, any custom selector computes scores through its own exp/div lowering, so manufactured ties in the reference can dissolve into distinct candidate scores (and vice versa) — exact-ID equality would then fail regardless of permutation replication. An audit round to establish the rule burns a performance-miss slot with zero attributable gain; a failed implementation round ends the campaign at worse counters. Anti-pattern catalog confirms: sort networks, gather compaction, cumsum compaction, winner-trees all measured regressions on sibling fingerprints (conditional evidence) and none yields a legal formulation here anyway.
- **Graph/replay families** — CLOSED: inductor-level replay structurally refused on this build (r003); manual workspace replay SUCCEEDED and is the committed canonical itself; nothing remains above it within the same class (updatable-node pointer patching has no public torch API; caller-buffer-targeting specializations violate soundness/ownership discipline).
- **Boundary/hot-path micro-items** — exhausted/null: python-dispatch coalescing proved inert against GPU submission counts (r005 root cause); remaining items (packed-layout single copy-out with dtype views, stage multi-token grids, allocator chatter, non_blocking variants) model to <=5-8 us combined central case versus a >=9.85 us bar, individually below measurement noise — bundling heterogeneous mechanisms into one round violates one-attributable-change discipline; `num_warps`/`num_stages` tuning stays normative-forbidden (profile Unknown status).
- **Inductor-default-beneath trims** — exhausted: round 002 took the demonstrated share (partial-graph host compression, -18.22% same-session).
- No anti-pattern entry is being violated by stopping; conversely, continuing would manufacture a knowingly-unfalsifiable round contrary to the intent-honesty rule.

## Rationale and Evidence

Deliverable check: an accepted Triton candidate is already committed as the campaign deliverable — `triton_grouped_topk_r2_004.py` @`c02d956c…` (manual-cuda-graph-workspace-replay), verified green on all four anchor bases of report_004 (+58.4951% prescribed pair; +42.5449% direct accepted-pair; +41.8851% vs report_002 basis; +59.2784% cumulative vs manifest anchor = 2.41x). The minimal-implementation requirement is exceeded by every measure, so this abort commits NO regression risk to the campaign record.

Quantitative residual decomposition grounding the stop (all numbers Verifier-backed): wall 196.909 us/call = ~104 us pipeline device band BY CONSTRUCTION (vendor pair ~87 us LOCKED behind the certifiability gate + three barrier-separated Triton stages 18.5 us) + ~93 us host floor whose reachable remainder was probed in r005 and found inert (python trip removal does not move GPU submission counts on this build: census-stable ~3 gpu_memcpy DtoD + ~7 cudaMemcpyAsync-class runtime submissions/call).

Family ledger of record across five rounds, each terminal state evidenced: preprocess/postprocess fusion ACCEPTED (+11.41%, report_001); compile-default dispatch compression ACCEPTED (+28.67% protocol / +18.22% direct, report_002); inductor replay NO-IMPROVEMENT with structural root cause (report_003); manual workspace replay ACCEPTED (+42.54% direct / +58.50% protocol, report_004); boundary coalescing NO-IMPROVEMENT with null-effect root cause (report_005 @`ada9d94a68e6ff3284ff1e3440df9fb047601285e2bf5fb8df42829a4a2cd122`, verdict @`cd0b3016e4213cc287c723ad084b18ef00c1e7246ed6d5f8af2ec3d149d903a7`). Streak arithmetic also favors stopping now cleanly rather than marching to the automatic limit with below-bar designs.

Explicit reopening conditions (out of current role/campaign scope, of record for future campaigns): (a) maintainer-authorized tie-rule disposition (archived_result_ref-grade on-device derivation replacing the uncertain emulation target); (b) harness/workspace infrastructure change restoring output ownership patterns compatible with inductor-level graph capture on this build (removing the mutated-inputs refusal source); (c) a revised frozen profile promoting argmax-family selection primitives beyond the current unique-maximum constraint. Until one of these lands, any further round here repeats measured failures without a falsifiable five-percent hypothesis.

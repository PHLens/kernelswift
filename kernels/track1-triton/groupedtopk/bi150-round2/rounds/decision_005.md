# Decision 005

## Metadata

```json
{
  "schema_version": 2,
  "decision": "proceed",
  "decision_kind": "optimization",
  "round": "005",
  "reference_implementation": "triton_grouped_topk_r2_004.py",
  "reference_report": "rounds/report_004.md",
  "language": "triton",
  "backend": "cuda",
  "target_profile": "triton_cuda",
  "runtime_fingerprint_ref": "project.md#runtime-fingerprint",
  "change_scope": "host",
  "change_family": "boundary-dispatch-coalescing",
  "sketch_ref": "rounds/sketch_005.json",
  "sketch_sha256": "21d13b983a4bf1ac1e6913bbaff635dd2932006bf9df04cd888406edcd6c92de",
  "implementation_profile_snapshot_ref": "profile_snapshot/triton_cuda.yaml",
  "implementation_profile_snapshot_sha256": "dc8fa4c0c73caecc78c6a886e5ffb18cb97371f3add7cb382dbe30887743b7ae",
  "project_capability_claim_ref": "profile_snapshot/capability_claim.json",
  "project_capability_claim_sha256": "2e6ee49ddd887a00e9a8a8ef6dfc746984ecaacd2256ee0b8666a3099a5b7f67"
}
```

## Optimization Intent

```json
{
  "bottleneck_class": "host-bound",
  "intervention": "coalesce the replay-boundary host operations of the accepted manual-replay architecture — replace the two separate per-call output copy-out dispatches with ONE construction-time-bound batched copy dispatch (torch._foreach_copy_ when the capability binds, legacy two-copy path otherwise) executed outside the replay boundary exactly as before; mark all three boundary copies non_blocking (stream-ordered same-device); hoist guard/routing residue onto build-time bound callables so the target-regime hot path performs guard-check + input copy-in + ONE replay submission + batched output copy-out with minimal Python attribute traffic — everything else byte-for-byte identical to triton_grouped_topk_r2_004.py including the graph handle, workspace semantics, both torch.topk sites, and the three Triton kernels",
  "allowed_changes": [
    "construction-time binding of the output copy-out strategy: try torch._foreach_copy_([caller_w, caller_ids], [ws_out_weights, ws_out_ids]) once at graph-build; on any TypeError/capability error bind the legacy two-copy path permanently for the instance; BOTH bindings produce byte-identical results and identical caller data_ptr preservation",
    "non_blocking=True added to the three boundary aten copies (same-device D2D, stream-order-safe)",
    "hot-path rebinding: after successful capture bind self._hot_call etc. so per-call routing is a single attribute load into a bound callable holding pre-resolved method handles (replay submission, copy ops); no semantic change — same callables invoked",
    "guard predicate micro-trims: precomputed immutable comparison tuple evaluated as one boolean expression",
    "no change to the three @triton.jit kernels, to either torch.topk call site's arguments or semantics, to the captured region content, or to the output contract"
  ],
  "invariants": [
    "correctness:pass under the unchanged harness comparator THROUGH THE ACTIVE REPLAYED TIER",
    "outputs bitwise-equal to accepted triton_grouped_topk_r2_004.py weights AND ids on seed42-regime, warm NEW-input bytes, first-input-again stale-trap, ALL four tie suites, cross-instance alternation; run_out==forward bitwise over poisoned buffers x2 with caller data_ptr preserved",
    "exact int32 ID equality including all manufactured tie cases",
    "floating outputs within allclose(atol=1e-2, rtol=1e-2)",
    "public ModelNew constructor and forward(hidden_states, gating_output) signatures unchanged",
    "both torch.topk call sites keep identical argument values, shapes, dtypes, ordering, and tie behavior — they remain the only selectors (no algorithm substitution)",
    "workspace ownership discipline inherited verbatim from decision_004: full-overwrite-per-call transient state only; results never returned from workspace; fresh distinct-data_ptr result buffers every forward call",
    "caller device preserved; boundary copies remain stream-ordered against the replay submission on the caller's current stream context; no synchronization beyond r004 behavior",
    "manual-replay -> compiled-default -> framework-eager three-tier chain unchanged with monotone downward tier flags"
  ],
  "expected_wall_improvement_pct": 6.0
}
```

## Unified Sketch

```json
{
  "artifact": "rounds/sketch_005.json",
  "sha256": "21d13b983a4bf1ac1e6913bbaff635dd2932006bf9df04cd888406edcd6c92de",
  "rendering": "normative contract is rounds/sketch_005.json; token-level dataflow is IDENTICAL to accepted rounds 001–004; round 005 changes ONLY the host-side boundary/hot-path handling of the manual-replay tier: the two output copy-out trips become ONE batched dispatch (capability-bound at construction), boundary copies become non_blocking, and guard/routing residue is hoisted onto build-time bound callables. The causal node cn.boundary-dispatch-coalescing attaches to the three boundary-crossing arrays (gating load, out_weights store, out_ids store)."
}
```

## Host Plan

```json
{
  "applicability": "required",
  "affected_scope": [
    "ModelNew.forward / ModelNew.run_out hot path on the manual-replay tier (bound-callable invocation)",
    "output copy-out construction-time strategy bind (batched vs legacy)",
    "boundary copy flags (non_blocking)",
    "guard predicate evaluation order and precomputed constants",
    "graph/workspace/tier machinery UNCHANGED from decision_004 except where listed above"
  ],
  "state_owner": "inherits decision_004 ownership entirely (graph handle, static workspace set, guard constants, tier-binding flags) EXTENDED by: one build-time-captured boolean _batched_copyout_ok, optionally prebound callable references for replay/copy operations, and one recorded construction-time exception artifact when the batched path fails to bind; no new tensor state of any kind",
  "lifetime": "identical to decision_004 (instance-lifetime graph/workspace/flags); the copy-out strategy binding is fixed at construction time and never revisited",
  "allocation_reuse": "identical to decision_004: zero model-code allocations on the replayed tier apart from the fresh result buffers each forward call; run_out performs zero allocations",
  "cache_key": ["shape", "dtype", "device"],
  "invalidation": "identical to decision_004: manual graph built only for the exact target regime; non-target regimes route directly to framework-eager without consulting artifacts; recapture never triggered by model code",
  "concurrency": "identical to decision_004 (sequential consumption; immediate copy-out satisfies output-lifetime rules); non_blocking copies are stream-ordered behind the replay on the SAME stream context, so consumers on that stream observe completed values exactly as with blocking copies",
  "device_stream_behavior": "preserve caller-selected device; replay submission and boundary copies all enqueue on the caller's current stream context in program order (copy-in -> replay -> copy-outs), preserving data dependencies by stream ordering alone; no events/syncs added or removed beyond r004 behavior; no device-context mutation by model code",
  "unchanged_behavior": [
    "forward(hidden_states, gating_output) public signature and observable batch-size assertion",
    "returned tuple (topk_weights fp32[83,8], topk_ids int32[83,8])",
    "run_out(gating_output, topk_weights, topk_ids) writes bitwise-identical results to forward for identical inputs into provided buffers before returning (harness auto_bench.make_profile_call lines 520-536 contract)",
    "the three @triton.jit stage kernels' source and launch semantics inside the capture",
    "both torch.topk call sites and their argument values",
    "scoring_func and routed_scaling_factor semantics",
    "three-tier fallback chain, workspace discipline, cold-cost placement, and compile-config discipline exactly as decision_004 (reduce-overhead remains retired)"
  ]
}
```

## Evaluation Contract

```json
{
  "hypothesis_id": "H-005",
  "intervention": "coalesce the replay-boundary host operations of the accepted manual-replay architecture — replace the two separate per-call output copy-out dispatches with ONE construction-time-bound batched copy dispatch (torch._foreach_copy_ when the capability binds, legacy two-copy path otherwise) executed outside the replay boundary exactly as before; mark all three boundary copies non_blocking (stream-ordered same-device); hoist guard/routing residue onto build-time bound callables so the target-regime hot path performs guard-check + input copy-in + ONE replay submission + batched output copy-out with minimal Python attribute traffic — everything else byte-for-byte identical to triton_grouped_topk_r2_004.py including the graph handle, workspace semantics, both torch.topk sites, and the three Triton kernels",
  "expected_causal_chain": [
    "each removed dispatcher trip (two copy-outs merged into one batched trip) plus non_blocking flags plus hoisted attribute traffic removes single-digit-microsecond chunks of pure CPU time per target-regime call while GPU-visible work stays byte-identical (~104 us pipeline band by construction)",
    "report_004 sized these levers explicitly: batching the two copy-outs into one fused copy, guard micro-costs, allocator chatter — jointly estimated ~10-17 us/call reachable against an adoption bar of ~9.85 us absolute (5% of ~0.197 ms paired basis), a regime shift caused by round-004 itself shrinking wall 0.338824 -> 0.196909 ms",
    "attribution scoping contract carried forward unchanged: candidate scope cat=kernel events expected ZERO/unattributable (branch-B phenomenon is the mechanism signature, NOT an anomaly); retention proof remains bitwise weight/id equality against the accepted kernel through the active tier",
    "unrounded interleaved paired median wall time improves by at least 5% versus triton_grouped_topk_r2_004.py under fingerprint 8deb1b01... on both reported bases (prescribed protocol pair and same-session direct accepted-pair probe)"
  ],
  "primary_metric": { "name": "wall_time", "expected_improvement_pct": 6.0 },
  "causal_graph": {
    "nodes": [
      "cn.boundary-dispatch-coalescing",
      "cn.host-dispatch-time",
      "cn.wall-time"
    ],
    "edges": [
      ["cn.boundary-dispatch-coalescing", "cn.host-dispatch-time"],
      ["cn.boundary-dispatch-coalescing", "cn.wall-time"],
      ["cn.host-dispatch-time", "cn.wall-time"]
    ]
  },
  "mechanism_observables": [
    { "name": "wall_time_unrounded_paired_median_ms", "expectation": "at least 5% below the accepted reference median across interleaved pairs at warmup 50 / repeat 100" },
    { "name": "bitwise_output_equivalence_to_accepted_r004", "expectation": "weights and ids bitwise-equal to accepted triton_grouped_topk_r2_004.py outputs on seed42-regime, warm NEW-input bytes, stale-trap, and all four manufactured tie suites through the replayed route, plus run_out==forward bitwise over poisoned buffers x2" },
    { "name": "fallback_tier_selectivity_and_recovery", "expectation": "non-target regime executes the framework-eager staged tier with base-consistent outputs and ZERO replay artifacts; following target-regime call uses the replayed tier again on the same instance; tier flags move downward only on failure" },
    { "name": "boundary_host_trip_count_per_call", "expectation": "TWO-BRANCH PASS: (branch A) batched capability binds at construction -> host census shows <=2 boundary tensor-op trips per replay-tier call (one input copy + ONE batched output copy replacing two); OR (branch B) legacy path binds via recorded construction-time capability error -> trip count stays 3 and this observable downgrades to documentation-only, verdict riding solely on observables 1-3; failure requires branch-A flag true WITH trip count remaining 3, or any regression in wall alongside neither branch binding" }
  ],
  "guardrails": [
    "correctness:pass through the replayed tier",
    "bitwise output equality vs accepted r004 incl. all tie suites",
    "outputs remain fp32 weights and int32 ids with shapes [83,8]",
    "current device preserved; boundary copies stream-ordered on the caller's current stream context; no model-code cross-instance RESULT state",
    "run_out result equality with forward for identical inputs; caller data_ptr preserved under non_blocking copies by stream-ordering guarantee within the calling iteration",
    "cold costs stay outside timed medians (r002/r004 precedent extended)"
  ],
  "profiling_level": "targeted"
}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`: no cataloged failure matches. No gathers/sorts/prefix-scans/winner-trees; kernels untouched; only host-side dispatch topology changes. MLU entries stay conditional.
- Profile boundaries respected: `tl.argmax` Constrained tie-order unused; `num_warps`/`num_stages` unset; hints empty; vendor top-k sites stay locked behind CHECK-TIE this round too.
- `torch._foreach_copy_` hazards declared for Coder: capability bind happens ONCE at graph construction inside the existing try-tree; mixed dtype/src-dst casting behavior must be exercised there (int64 ws source -> int32 caller dest must round-trip EXACTLY like plain copy_ cast); ANY anomaly binds the legacy two-copy path — never a partial mixture per call. Byte-equality between the two strategies is mandatory and covered by the bitwise sweep.
- non_blocking discipline: same-device D2D copies with later reads on the SAME stream are dependency-ordered regardless of blocking flag; the harness consumes on the calling thread within the same iteration — correctness unaffected; do NOT add cudaEventSynchronize/stream sync calls anywhere (that would re-add the cost being removed).
- Hot-path rebinding traps: bound callables must reference the CURRENT tier objects; if a tier transitions downward AFTER binding (capture failure mid-flight), the binding must be invalidated in the same failure handler — stale bound references to a dead tier are forbidden.
- Guard micro-trim trap: precomputed constants must be derived from the SAME constructor values the existing guard uses; adding/removing any check changes observable selectivity and would fail the selectivity probe.
- DANGER list carried verbatim: reduce-overhead strings ×0; no precision/backend/env/cache knobs; machine-scan discipline from rounds 002/004 applies.
- Tie gate restated: entering vendor top-k remains blocked (audit slot unprofitable; cross-implementation exp-bit ambiguity unresolved); four reports changed none of that math.

## Rationale and Evidence

Canonical anchors: last_accepted = triton_grouped_topk_r2_004.py @`c02d956c…`, paired medians 0.196909 ms (protocol pair 0.474386 → 0.196909 = +58.4951%; direct accepted-pair r002 0.3463206812739372 → r004 0.19897893071174622 = +42.54488932633068%; cumulative vs manifest anchor +59.28%). report_004 evidence_for_next_round names exactly this round's lever set as the remaining sub-round-scale items ("batching the two copy-outs into one fused copy, guard micro-costs, allocator chatter") and quantifies the residual structure (~93 µs host share of 196.9 µs wall; vendor pair ~87 µs still tie-gate-locked; host census: exactly ~3 aten::copy_ trips/call).

Economic-regime argument: round 004 shrank wall by ~44%, which re-prices previously sub-threshold levers — 5% now means ≈9.85 µs absolute, within reach of the enumerated dispatch/coalescing items whose joint central estimate (~10–17 µs) clears it narrowly. Expected declared 6.0% keeps the hypothesis falsifiable just above the bar without overclaiming.

Alternative families rejected for round 005: (i) tie-gated vendor-top-k entry — audit economics unchanged (a zero-wall-gain audit burns miss slot #1/3) AND the substantive blocker stands regardless of scale: observed sbtopk/bitonic tie permutations are implementation-emergent ([7,6,4,5,1,0,2,3] class) and cannot be certified on Triton primitives, while cross-implementation score-bit ambiguity (Triton exp path vs torch.softmax bits) independently threatens exact-ID equality even under perfect permutation replication — no report changed this math; (ii) abort — premature while a family plausibly clears the bar and counters are reset (worst case here: streak 1/3 with campaign alive and canonical r004 intact); (iii) sketch-level kernel changes — dataflow-illegal across library barriers as established since round 002.

One-attributable-change compliance: change_scope `host`; one causal lever (boundary dispatch coalescing incl. its constituent trims, all inseparable single-purpose hot-path work on the replayed tier); every piece separately legible in the host census yet mechanistically unified (dispatcher-trip elimination around one replay submission).

Artifacts consulted: `rounds/report_004.md` @`c79cc018…`, `rounds/verdict_004.json` @`13340553…`, `triton_grouped_topk_r2_004.py` @`c02d956c…`, `rounds/report_003.md`, `rounds/report_002.md`, `rounds/report_001.md`, `rounds/report_000.md`, `project.md`, `profile_snapshot/triton_cuda.yaml` @`dc8fa4c0…`, `profile_snapshot/capability_claim.json` @`2e6ee49d…`, `references/invariants.md`, `references/anti-patterns.md`, `references/bottleneck-judgment.md`, `auto_bench.py`, lineage reports under `../bi150/` (labeled noncanonical).

# Decision 003

## Metadata

```json
{
  "schema_version": 2,
  "decision": "proceed",
  "decision_kind": "optimization",
  "round": "003",
  "reference_implementation": "triton_grouped_topk_r2_002.py",
  "reference_report": "rounds/report_002.md",
  "language": "triton",
  "backend": "cuda",
  "target_profile": "triton_cuda",
  "runtime_fingerprint_ref": "project.md#runtime-fingerprint",
  "change_scope": "host",
  "change_family": "compile-graph-replay-reduce-overhead",
  "sketch_ref": "rounds/sketch_003.json",
  "sketch_sha256": "4a909a11cbd8df0ad0385cf6379dc77eb189bffd60ec2ab1b341dbdaa127a782",
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
  "intervention": "escalate the accepted compiled pipeline from torch.compile(mode='default') to mode='reduce-overhead' (CUDA Graph capture and replay) by changing ONLY the compilation mode while preserving dynamic=False, every @triton.jit kernel source byte-for-byte, both torch.topk call sites unchanged, unchanged strict target-regime gating, and a permanent per-instance three-tier fallback chain (replayed route -> compiled-default route -> framework-eager staged path); replay outputs must be bitwise-identical to accepted r002 outputs",
  "allowed_changes": [
    "SUPERSESSION CLAUSE: this Decision supersedes decision_002.md § Optimization Intent invariant 'compilation configuration restricted to mode=default,dynamic=False' SOLELY in its mode component via the Round-003 immutable decision process; dynamic=False and every no-backend/no-env/no-precision/no-cache-knob restriction of decision_002 remain fully in force",
    "torch.compile configuration becomes exactly mode='reduce-overhead', dynamic=False on the identical fixed-shape staged pipeline",
    "three-tier fallback construction at first use: replayed callable, then compiled-default callable, then unmodified framework-eager staged path; any exception during capture/construction or invocation binds permanently to the next tier",
    "no change to the three @triton.jit kernels, to either torch.topk call site's arguments or semantics, or to the output contract"
  ],
  "invariants": [
    "correctness:pass under the unchanged harness comparator THROUGH THE REPLAYED ROUTE",
    "outputs bitwise-equal to accepted triton_grouped_topk_r2_002.py weights AND ids on the seed42 regime and on ALL tie suites (all-equal/two-expert-tie-same-group/structured-group-tie-boundary/duplicate-max-pairs-cross-group)",
    "exact int32 ID equality including all manufactured tie cases",
    "floating outputs within allclose(atol=1e-2, rtol=1e-2)",
    "public ModelNew constructor and forward(hidden_states, gating_output) signatures unchanged",
    "both torch.topk call sites keep identical argument values, shapes, dtypes, ordering, and tie behavior — they remain the only selectors (no algorithm substitution)",
    "run_out results bitwise-equal to forward outputs for identical inputs, caller buffers zero-copy preserved",
    "caller device and current stream semantics preserved by framework stream-safe replay; no device-context mutation",
    "model code performs no cross-instance or cross-call tensor caching; any CUDA-graph memory pools holding placeholder tensors are framework-owned runtime caches without model logical state",
    "compiled execution must be bypassable: all three tiers reachable, eager tier byte-equivalent always"
  ],
  "expected_wall_improvement_pct": 15.0
}
```

## Unified Sketch

```json
{
  "artifact": "rounds/sketch_003.json",
  "sha256": "4a909a11cbd8df0ad0385cf6379dc77eb189bffd60ec2ab1b341dbdaa127a782",
  "rendering": "normative contract is rounds/sketch_003.json; token-level dataflow is IDENTICAL to accepted rounds 001–002 (stage-A softmax+group-max -> library topk#1 over group_scores_out -> stage-B arithmetic lane-membership masking -> library topk#2 over masked_scores -> stage-C renorm/scale/narrow); round 003 changes ONLY the execution strategy: the compiled fixed-shape region is captured once as a CUDA graph and replayed, so per-call Python dispatch, allocation planning, and per-launch CPU submission collapse into one graph-launch while values remain bit-identical"
}
```

## Host Plan

```json
{
  "applicability": "required",
  "affected_scope": [
    "ModelNew.forward routing across the three-tier chain",
    "ModelNew.run_out routing across the three-tier chain with copy-out performed OUTSIDE any captured region every call",
    "one-time construction of the reduce-overhead callable and fallback callables keyed on the target regime",
    "target-regime guard predicate evaluation per call",
    "permanent tier-binding state flags on the instance"
  ],
  "state_owner": "the ModelNew instance owns the three callable handles (replayed, compiled-default, eager staged), immutable guard constants, and immutable tier-binding flags; framework-owned CUDA-graph memory pools may hold static placeholder tensors between replays (runtime caches WITHOUT model logical state — analogous to build caches); process-global dynamo/inductor bytecode/binary caches remain as in round 002; every user-visible tensor, including run_out-provided output buffers, remains owned by the calling forward/run_out invocation",
  "lifetime": "callables persist for the instance lifetime after successful construction; tensors live exactly one call as in prior rounds except framework-managed replay placeholders; tier flags transition monotonically downward (replay->default->eager) at most once each and never recover upward within a failed regime",
  "allocation_reuse": "no model-code tensor reuse across forwards: the eager tier allocates fresh temporaries exactly as round 001; under replay, static placeholder buffers are owned and reused exclusively BY THE FRAMEWORK's graph pool machinery, whose user-visible effect is bounded to returning fresh-value results each call (verified bitwise each probe); run_out copy-out into caller buffers happens after the replay boundary every call",
  "cache_key": ["shape", "dtype", "device"],
  "invalidation": "the replayed callable is built only for the exact target regime (contiguous fp32 [83,256] gating on current device, constructor config topk=8/renormalize=True/G=8/KG=4); ANY other regime routes directly to the unmodified eager staged tier without consulting replay artifacts; graph recapture is never triggered by model code",
  "concurrency": "one model instance is not shared across concurrent forwards; sequential per-call consumption of outputs precedes any subsequent replay (harness iteration pattern satisfies CUDA-graph output-lifetime rules); no module-level mutable model state beyond framework-owned caches is introduced",
  "device_stream_behavior": "preserve caller-selected device; replays execute through torch's stream-safe cudagraph machinery on the current stream context; retained torch.topk inside the captured region keeps its recorded semantics; first-capture cost absorbed outside timed medians (report_002 cold-compile precedent, extended to capture cost); no device-context creation/removal by model code; no synchronization beyond base.py behavior plus the framework's own replay management",
  "unchanged_behavior": [
    "forward(hidden_states, gating_output) public signature and observable batch-size assertion",
    "returned tuple (topk_weights fp32[83,8], topk_ids int32[83,8])",
    "run_out(gating_output, topk_weights, topk_ids) writes bitwise-identical results to forward for identical inputs into provided buffers before returning (harness auto_bench.make_profile_call lines 520-536)",
    "the three @triton.jit stage kernels' source and launch semantics",
    "both torch.topk call sites and their argument values",
    "scoring_func and routed_scaling_factor semantics",
    "framework-eager staged behavior for non-target regimes inherited verbatim from round 002"
  ]
}
```

## Evaluation Contract

```json
{
  "hypothesis_id": "H-003",
  "intervention": "escalate the accepted compiled pipeline from torch.compile(mode='default') to mode='reduce-overhead' (CUDA Graph capture and replay) by changing ONLY the compilation mode while preserving dynamic=False, every @triton.jit kernel source byte-for-byte, both torch.topk call sites unchanged, unchanged strict target-regime gating, and a permanent per-instance three-tier fallback chain (replayed route -> compiled-default route -> framework-eager staged path); replay outputs must be bitwise-identical to accepted r002 outputs",
  "expected_causal_chain": [
    "inductor wraps the compiled fixed-shape region for CUDA graph capture during warmup; subsequent target-regime calls submit ONE graph launch instead of ~6.9 individual kernel launches plus Python dispatch",
    "per-launch CPU submission cost and residual dispatch/alloc planning elapse once-per-replay instead of once-per-kernel, shrinking host-side time outside kernel execution against the report_002 residual (~0.235 ms/call) while device work stays essentially identical (~104 us/call incl. retained vendor pair)",
    "ATTRIBUTION SCOPING CONTRACT (explicit): under CUDA graph replay, per-kernel cat=kernel events for intra-graph launches may become unattributable in traces (epoch-1 report_009 precedent; BI150 trace behavior). Consequently THIS round's retention proof transfers from trace-kernel identity to OUTPUT identity: bitwise weight/id equality against accepted r002 on seed42 + all tie suites through the replayed route, recorded per-probe. Device/kernel-count trace fields become diagnostic-only and their absence or coarsening must NOT be interpreted as mechanism failure while the wall observable passes.",
    "unrounded interleaved paired median wall time improves by at least 5% versus triton_grouped_topk_r2_002.py under fingerprint 8deb1b01..."
  ],
  "primary_metric": { "name": "wall_time", "expected_improvement_pct": 15.0 },
  "causal_graph": {
    "nodes": [
      "cn.graph-replay-dispatch-elision",
      "cn.host-dispatch-time",
      "cn.wall-time"
    ],
    "edges": [
      ["cn.graph-replay-dispatch-elision", "cn.host-dispatch-time"],
      ["cn.graph-replay-dispatch-elision", "cn.wall-time"],
      ["cn.host-dispatch-time", "cn.wall-time"]
    ]
  },
  "mechanism_observables": [
    { "name": "wall_time_unrounded_paired_median_ms", "expectation": "at least 5% below the accepted reference median across interleaved pairs at warmup 50 / repeat 100" },
    { "name": "bitwise_output_equivalence_to_accepted_r002", "expectation": "weights and ids bitwise-equal to accepted triton_grouped_topk_r2_002.py outputs on seed42-regime and all four manufactured tie suites through the replayed route, plus run_out==forward bitwise over poisoned buffers" },
    { "name": "fallback_tier_selectivity_and_recovery", "expectation": "non-target regime executes the eager staged tier with base-consistent bitwise==r002 outputs, and a following target-regime call uses the replayed tier again on the same instance, tier flags behaving monotonically only on failure" },
    { "name": "kernel_count_per_call", "expectation": "TWO-BRANCH PASS: (branch A) attributed launches decrease below 6.90/call OR stay <=6.90; OR (branch B) intra-replay launches are explicitly unattributable in the candidate scope per the attribution scoping contract — record branch taken; neither branch constitutes failure, and branch B corroborates successful graph capture; failure requires attributed count EXCEEDING 6.90/call" }
  ],
  "guardrails": [
    "correctness:pass through the replayed route",
    "bitwise output equality vs accepted r002 incl. all tie suites",
    "outputs remain fp32 weights and int32 ids with shapes [83,8]",
    "current device/stream semantics preserved; no model-code cross-instance tensor state",
    "run_out result equality with forward for identical inputs",
    "cold capture/compile cost stays outside timed medians (report_002 precedent extended)"
  ],
  "profiling_level": "targeted"
}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`: no cataloged failure matches. No `tl.gather`, no sort network, no prefix scan, no winner-tree added; Triton kernel sources are untouched bytes from round 001. Entries 011–016 stay conditional MLU-fingerprint evidence.
- Profile boundaries respected: `tl.argmax` Constrained tie-order capability remains unused; `num_warps`/`num_stages` unset; sketch hints empty. The two vendor top-k sites stay LOCKED behind the CHECK-TIE audit gate this round as well (entering them without an on-device derivation would be normative-unproven = capability-miss).
- Supersession discipline: decision_002.md's mode restriction is amended ONLY via this Round-003 decision (Orchestrator-authorized path); `dynamic=False` and every precision/backend/env/cache-knob prohibition carry over verbatim; a machine scan showing counts>0 for tf32/backends knobs/TORCHINDUCTOR env/reduced_precision etc. should FAIL the Coder binding statement exactly as in round 002.
- CUDA-graph hazards declared for Coder: (i) static input buffering means gating_output contents are read into framework placeholders at replay time — always pass LIVE tensor data (harness pattern is safe); (ii) graph-pool output placeholders must not be held across replays by MODEL code — forward returns pool-backed tensors whose consumer reads them within the same iteration, and run_out copies out immediately, so no stale-read window exists in the harness flow; (iii) no `.backward()`/in-place mutation of graph outputs anywhere; (iv) capture happens once per instance on the fixed shape — any unexpected recapture pressure indicates a regime violation and must fall down-tier instead.
- Numerics: replay re-executes the IDENTICAL captured kernels — values are deterministic per input; the bitwise-vs-r002 requirement makes any deviation immediately detectable rather than tolerance-swallowed.
- Attribution risk handled by contract: if overlap/double-record salvage patterns recur (report_001 P1) they apply to scopes we can still form; intra-replay unattributability is a DECLARED branch, not an anomaly.

## Rationale and Evidence

Canonical Verifier facts (rounds/report_002.md @`bd0932b9cae83a55e0d63f3b149f77937c143100e62e62daf28e850f97ca36ce`, verdict @`db173df820459e683595f2a5fba7c1e13e1cf2ddfb7f5acbf9e88f2c9e8de5f7`): paired medians reference 0.475034 → candidate 0.338824 ms (+28.6733% protocol basis) and same-session direct pair versus last_accepted r001 `0.4170527681708336 → 0.3410717472434044` = **+18.21856290768121%**; device_us flat 103.985 µs/call inside the declared band (mechanism confirmed host-side); kernels/call 6.90 ≤ cap; vendor pair persists 1.97/call each (48.62+36.89 µs/call = 85.51 µs of device); stage kernels byte-named unchanged; compiled-route bitwise==r001 on every case including all four tie suites; non-target T=41 exercised staged fallback selectively with recovery; cold compile ≈2.81 s absorbed outside timed medians.

Family selection among evidence-bounded options: (a) chosen — reduce-overhead targets the still-dominant host residual (~0.235 ms/call; device_ratio rose 0.253→0.307 because host shrank faster than device), carries the strongest available lineage validation (NONCANONICAL prior −22.51% on the exact same escalation step), and is explicitly named as the unused lever in report_002 evidence_for_next_round; (b) rejected for round 003 — the CHECK-TIE on-device audit gates entry into the vendor top-k sites, has no delivery-independent wall gain of its own, and reverse-engineering observed sbtopk/bitonic tie permutations cannot be certified against Triton primitives without that audit; (c) rejected — guard/shape hygiene beneath Inductor default has no demonstrated ≥5% headroom and much of it is subsumed by (a).

Expected ≥15% declaration rationale: lineage prior −22.51% on the identical family step; even halved efficacy clears the bar; falsifiable well above 5%.

One-attributable-change compliance: change_scope `host`; single causal lever cn.graph-replay-dispatch-elision; kernels and selection sites untouched; the mode-only supersession is the entire delta versus accepted r002.

Artifacts consulted: `rounds/report_002.md`, `rounds/verdict_002.json`, `triton_grouped_topk_r2_002.py` @`ad703266…`, `rounds/report_001.md`, `rounds/report_000.md`, `project.md`, `profile_snapshot/triton_cuda.yaml` @`dc8fa4c0…`, `profile_snapshot/capability_claim.json` @`2e6ee49d…`, `references/invariants.md`, `references/anti-patterns.md`, `references/bottleneck-judgment.md`, `auto_bench.py`, lineage reports under `../bi150/` (labeled noncanonical).

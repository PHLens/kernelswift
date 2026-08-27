# Decision 004

## Metadata

```json
{
  "schema_version": 2,
  "decision": "proceed",
  "decision_kind": "optimization",
  "round": "004",
  "reference_implementation": "triton_grouped_topk_r2_002.py",
  "reference_report": "rounds/report_002.md",
  "language": "triton",
  "backend": "cuda",
  "target_profile": "triton_cuda",
  "runtime_fingerprint_ref": "project.md#runtime-fingerprint",
  "change_scope": "host",
  "change_family": "manual-cuda-graph-workspace-replay",
  "sketch_ref": "rounds/sketch_004.json",
  "sketch_sha256": "ccf277f422ce254d09dc1402c997a6c311a1f63457423f23afd60a71b4d9ae59",
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
  "intervention": "capture the accepted staged pipeline ONCE as a manual torch.cuda.CUDAGraph over instance-owned static workspace buffers — static input placeholder filled by copy-in from live gating data at the capture boundary, fully-overwritten static internal temporaries and output placeholders, copy-out to caller-visible results outside the replay boundary — and replay it for every target-regime call, with strict regime gating unchanged and a permanent three-tier fallback chain (manual replay -> compiled-default -> framework-eager staged); the reduce-overhead compile tier is retired (report_003 root cause: inductor cudagraphs refuses mutated inputs structurally on this build) while manual capture is not subject to that heuristic; all three Triton kernels and both torch.topk call sites remain byte-for-byte unchanged",
  "allowed_changes": [
    "SUPERSESSION CLAUSE: this Decision supersedes the decision_002/decision_003 Host-Plan ownership sentences ('no cross-instance tensor caching', 'temporaries live exactly one forward call') SOLELY by introducing instance-owned static WORKSPACE buffers with full-overwrite-per-call semantics and copy-in/copy-out boundaries defined below; user-visible results remain copied into invocation-owned buffers every call, so no result reuse or cross-call data carryover exists",
    "one-time construction of the manual graph: warmup on a side stream per torch.cuda.graph recommended pattern, capture of stage-A launch + torch.topk(group_scores_out,k=KG)[1] + stage-B launch + torch.topk(masked_scores,k=K) + stage-C launch writing ONLY workspace placeholders, keyed to the exact target regime",
    "per-call host work on target regime reduced to: guard predicate, static input copy-in, ONE graph replay submission, two small copy-outs (fp32 weights [83,8], int32 ids [83,8])",
    "three-tier fallback construction at first use with lazy lower tiers: any exception during warmup/capture/first replay binds permanently down-tier",
    "no change to the three @triton.jit kernels, to either torch.topk call site's arguments or semantics, or to the output contract"
  ],
  "invariants": [
    "correctness:pass under the unchanged harness comparator THROUGH THE REPLAYED ROUTE",
    "outputs bitwise-equal to accepted triton_grouped_topk_r2_002.py weights AND ids on the seed42 regime and ALL tie suites (all-equal/two-expert-tie-same-group/structured-group-tie-boundary/duplicate-max-pairs-cross-group)",
    "exact int32 ID equality including all manufactured tie cases",
    "floating outputs within allclose(atol=1e-2, rtol=1e-2)",
    "public ModelNew constructor and forward(hidden_states, gating_output) signatures unchanged",
    "both torch.topk call sites keep identical argument values, shapes, dtypes, ordering, and tie behavior — they remain the only selectors (no algorithm substitution)",
    "run_out results bitwise-equal to forward outputs for identical inputs, caller buffers zero-copy preserved via copy-out every call",
    "caller device preserved; replays execute stream-safely per torch API semantics (captured once on the capture side stream; replay submitted against the caller's current stream context); no device-context mutation by model code",
    "workspace discipline: every placeholder is FULLY OVERWRITTEN during each replay before any consumer reads it; workspace never stores results across calls; nothing computed inside one invocation is returned or reused by another invocation",
    "compiled execution remains bypassable: eager tier reachable and byte-equivalent always"
  ],
  "expected_wall_improvement_pct": 15.0
}
```

## Unified Sketch

```json
{
  "artifact": "rounds/sketch_004.json",
  "sha256": "ccf277f422ce254d09dc1402c997a6c311a1f63457423f23afd60a71b4d9ae59",
  "rendering": "normative contract is rounds/sketch_004.json; token-level dataflow is IDENTICAL to accepted rounds 001–002 (stage-A softmax+group-max -> library topk#1 over group_scores_out -> stage-B arithmetic lane-membership masking -> library topk#2 over masked_scores -> stage-C renorm/scale/narrow); round 004 changes ONLY the execution strategy: the pipeline's kernel sequence is captured once as a manual CUDA graph over static workspace buffers and replayed per call, with live gating data copied in at the boundary and results copied out after it, so per-call Python dispatch, allocation planning, and per-launch CPU submission collapse into one graph-launch submission while values stay bit-identical"
}
```

## Host Plan

```json
{
  "applicability": "required",
  "affected_scope": [
    "ModelNew.forward routing across the three-tier chain",
    "ModelNew.run_out routing across the three-tier chain (copy-out OUTSIDE the replay boundary every call)",
    "one-time construction of the manual CUDA graph (warmup + capture) keyed to the target regime",
    "static workspace buffer set: gating_in[83,256] fp32, scores_out[83,256], group_scores_out[83,8], masked_scores[83,256], intermediate topk outputs produced inside capture (sel_groups/topk_vals/topk_ids), out_weights/out_ids[83,8] placeholders",
    "target-regime guard predicate evaluation per call",
    "permanent tier-binding state flags on the instance (replay_failed, compile_default_failed)"
  ],
  "state_owner": "the ModelNew instance owns the graph handle, the static workspace tensor set, immutable guard constants, and immutable tier-binding flags; framework-owned graph-private memory pools may back allocations performed DURING capture (torch.cuda.graph supported pattern); process-global dynamo/inductor build caches exist only for the fallback compiled-default tier; every user-visible result tensor originates as an invocation-owned buffer filled by per-call copy-out — workspace contents are transient computation state fully rewritten each call and are NEVER returned directly or read across calls",
  "lifetime": "graph handle and workspace persist for the instance lifetime after successful capture; tier flags transition monotonically downward (manual-replay -> compiled-default -> eager) at most once each; workspace becomes garbage with module destruction",
  "allocation_reuse": "MODEL CODE performs zero per-call allocations on the replayed tier (copy-in writes existing workspace, replay reuses captured addresses, copy-out fills caller buffers); lower tiers allocate exactly as accepted round 001/002; no allocation occurs after capture inside model code",
  "cache_key": ["shape", "dtype", "device"],
  "invalidation": "the manual graph is built only for the exact target regime (contiguous fp32 [83,256] gating on current device, constructor config topk=8/renormalize=True/G=8/KG=4); ANY other regime routes directly to the framework-eager tier without consulting artifacts; recapture is never triggered by model code; if capture fails once the tier is abandoned for the instance lifetime",
  "concurrency": "one model instance is not shared across concurrent forwards; sequential per-call consumption plus immediate copy-out satisfies graph output-lifetime rules; no module-level mutable model state beyond framework-owned caches/pools",
  "device_stream_behavior": "preserve caller-selected device; warmup+capture follow the torch.cuda.graph recommended side-stream pattern ONCE at first use; afterwards each target-regime call runs entirely on the CALLER'S current stream context via stream-safe replay; retained torch.topk keeps recorded semantics inside the captured region; no synchronization beyond base.py behavior plus one graph-launch submission; no device-context creation/removal by model code",
  "unchanged_behavior": [
    "forward(hidden_states, gating_output) public signature and observable batch-size assertion",
    "returned tuple (topk_weights fp32[83,8], topk_ids int32[83,8])",
    "run_out(gating_output, topk_weights, topk_ids) writes bitwise-identical results to forward for identical inputs into provided buffers before returning (harness auto_bench.make_profile_call lines 520-536 contract)",
    "the three @triton.jit stage kernels' source and launch semantics",
    "both torch.topk call sites and their argument values",
    "scoring_func and routed_scaling_factor semantics",
    "framework-eager staged behavior for non-target regimes inherited verbatim from accepted round 002"
  ]
}
```

## Evaluation Contract

```json
{
  "hypothesis_id": "H-004",
  "intervention": "capture the accepted staged pipeline ONCE as a manual torch.cuda.CUDAGraph over instance-owned static workspace buffers — static input placeholder filled by copy-in from live gating data at the capture boundary, fully-overwritten static internal temporaries and output placeholders, copy-out to caller-visible results outside the replay boundary — and replay it for every target-regime call, with strict regime gating unchanged and a permanent three-tier fallback chain (manual replay -> compiled-default -> framework-eager staged); the reduce-overhead compile tier is retired (report_003 root cause: inductor cudagraphs refuses mutated inputs structurally on this build) while manual capture is not subject to that heuristic; all three Triton kernels and both torch.topk call sites remain byte-for-byte unchanged",
  "expected_causal_chain": [
    "during first-use warmup/capture the six-to-seven-kernel pipeline is recorded against static addresses; afterwards each target-regime call performs only: guard check, workspace copy-in, ONE cudaGraphLaunch-class submission, two small copy-outs",
    "all per-call Python op-dispatch, temporary-allocation planning, and ~6.9 per-launch CPU submissions elapse into a single graph submission, attacking the report_002 residual host share (~0.235 ms/call of 0.3388 ms wall) while device work stays essentially identical (~104 us/call incl. the retained vendor pair at 85.51 us/call)",
    "ATTRIBUTION SCOPING CONTRACT (explicit, carried forward): under replay, intra-graph cat=kernel events may be unattributable in traces; retention proof transfers from trace-kernel identity to bitwise weight/id equality against accepted r002 through the replayed route on seed42 + all tie suites; trace fields are diagnostic-only and their coarsening must NOT be interpreted as mechanism failure while the wall observable passes",
    "unrounded interleaved paired median wall time improves by at least 5% versus triton_grouped_topk_r2_002.py under fingerprint 8deb1b01..."
  ],
  "primary_metric": { "name": "wall_time", "expected_improvement_pct": 15.0 },
  "causal_graph": {
    "nodes": [
      "cn.workspace-manual-replay",
      "cn.host-dispatch-time",
      "cn.wall-time"
    ],
    "edges": [
      ["cn.workspace-manual-replay", "cn.host-dispatch-time"],
      ["cn.workspace-manual-replay", "cn.wall-time"],
      ["cn.host-dispatch-time", "cn.wall-time"]
    ]
  },
  "mechanism_observables": [
    { "name": "wall_time_unrounded_paired_median_ms", "expectation": "at least 5% below the accepted reference median across interleaved pairs at warmup 50 / repeat 100" },
    { "name": "bitwise_output_equivalence_to_accepted_r002", "expectation": "weights and ids bitwise-equal to accepted triton_grouped_topk_r2_002.py outputs on seed42-regime and all four manufactured tie suites through the replayed route, plus run_out==forward bitwise over poisoned buffers" },
    { "name": "fallback_tier_selectivity_and_recovery", "expectation": "non-target regime executes the framework-eager staged tier with base-consistent bitwise==r002 outputs, and a following target-regime call uses the replayed tier again on the same instance; tier flags move downward only on failure" },
    { "name": "kernel_count_per_call", "expectation": "TWO-BRANCH PASS: (branch A) attributed launches collapse far below 6.90/call toward <=2/call evidencing single-submission replay; OR (branch B) intra-replay launches are explicitly unattributable per the attribution scoping contract and the candidate scope instead shows single-submission/host-side graph evidence — record branch taken; failure requires attributed count ≈6.90/call WITH flat wall (capture demonstrably absent)" }
  ],
  "guardrails": [
    "correctness:pass through the replayed route",
    "bitwise output equality vs accepted r002 incl. all tie suites",
    "outputs remain fp32 weights and int32 ids with shapes [83,8]",
    "current device preserved and stream-safe replay semantics; no model-code cross-instance RESULT state (workspace is full-overwrite transient computation state only)",
    "run_out result equality with forward for identical inputs",
    "cold warmup+capture cost stays outside timed medians (r002 cold-compile precedent extended)"
  ],
  "profiling_level": "targeted"
}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`: no cataloged failure matches; no gathers/sorts/prefix-scans/winner-trees added; Triton sources untouched. MLU entries stay conditional.
- Profile boundaries respected: `tl.argmax` Constrained capability unused; `num_warps`/`num_stages` unset; hints empty; vendor top-k sites stay behind the CHECK-TIE gate this round too.
- Root-cause separation from round 003: report_003's refusal message belongs to INDUCTOR cudagraph-trees heuristics ("mutated inputs") inside mode='reduce-overhead'. MANUAL `torch.cuda.CUDAGraph` capture does not consult that heuristic; its obligations (static addresses, fixed shapes, stable launch order, stream rules) are satisfied BY CONSTRUCTION here via the workspace design. The reduce-overhead compile tier is RETIRED from the chain rather than reordered below manual replay.
- Capture-environment hazards declared for Coder: (i) perform warmup iterations then capture per `torch.cuda.graph` context-manager recommended pattern (side capture stream), ONCE at first target-regime use; (ii) NO host-side branches, prints, .item(), cpu reads, or variable control flow inside the captured region — the captured region is exactly the accepted `_triton_forward` body over fixed shapes; (iii) allocator ops during capture are legal (graph-private pool); nothing may allocate on LATER replays from model code; (iv) never call torch.compile'd functions inside the capture — capture the EAGER staged path (framework-eager), keeping compiled-default strictly as a DOWN-tier fallback built lazily only if capture fails; (v) inputs must be read from the workspace placeholder (copy-in) — the original gating tensor is referenced only before the boundary; (vi) capture failure of ANY component (including torch.topk internals hitting unsupported paths) binds the compiled-default tier lazily, then eager.
- Numerics: replay re-executes identical kernels on identical bit-patterns (copy-in provides the same input bits as r002 received; kernels byte-identical), so the bitwise-vs-r002 requirement is structural, not aspirational; any deviation indicates a capture defect and fails correctness immediately.
- Output-lifetime rule: harness consumes/copies within the iteration; run_out copies out before returning; forward returns COPY-OUT results freshly written each call (never raw workspace references) — eliminating stale-read windows even under adversarial consumers.
- DANGER note for Coder binding statement: counts>0 for tf32/backends knobs/TORCHINDUCTOR env/reduced_precision/etc. still FAIL; additionally `mode='reduce-overhead'` strings anywhere in candidate source should FAIL this round (tier retired).

## Rationale and Evidence

Canonical anchors: last_accepted = triton_grouped_topk_r2_002.py @`ad703266…` with paired-median basis 0.338824 ms (report_002 @`bd0932b9…`); r003 (NO-IMPROVEMENT, −8.0875% same-session regression, canonical pointers unchanged) proved inductor-level replay structurally refused ("skipping cudagraphs due to mutated inputs" EVERY invocation) while leaving all bitwise guardrails green — the family is closed but NOT the mechanism class.

Why THIS family next: residual structure is host-dominated (~0.235 ms/call outside kernels; device_ratio 0.307). The single remaining legal mechanism family with large expected wall gain is manual-workspace replay: it removes ~6.9 per-launch submissions + dispatch residue that even Inductor default could not (its generated code was itself captured-opaque to further compression, kernels/call stuck at 6.90). Expected envelope: submission cost falls from ~28 µs/launch-class × 6.9 + dispatch/python residue toward one submission + O(10 µs) copy overheads; even assigning heavy slop, clearing ≥5% (≈17 µs) is the floor case while plausible attainment is several-fold larger. Declared expectation 15.0% stays falsifiable above the bar without inheriting unearned lineage numbers (epoch-1 r009's −22.51% came from THEIR build permitting inductor-level replay — prior information, noncanonical).

Streak calculus honesty: a graceful degradation outcome measures ≈r002 (±ε) → second no-improvement (streak 2/3) with campaign alive; abort today forfeits the largest available legal family while counters have room; audit-only rounds cannot exist profitably at streak 1. This Decision therefore proceeds on the highest-EV legal mechanism.

One-attributable-change compliance: change_scope `host`; single causal lever cn.workspace-manual-replay; kernels/selection untouched; ownership supersession is scoped precisely (transient full-overwrite workspace ≠ result caching).

Artifacts consulted: `rounds/report_003.md` @`e00efc94…`, `rounds/verdict_003.json` @`9336749c…`, `rounds/report_002.md` @`bd0932b9…`, `rounds/report_001.md`, `rounds/report_000.md`, `triton_grouped_topk_r2_002.py` @`ad703266…`, `project.md`, `profile_snapshot/triton_cuda.yaml` @`dc8fa4c0…`, `profile_snapshot/capability_claim.json` @`2e6ee49d…`, `references/invariants.md`, `references/anti-patterns.md`, `references/bottleneck-judgment.md`, `auto_bench.py`, lineage reports under `../bi150/` (labeled noncanonical).

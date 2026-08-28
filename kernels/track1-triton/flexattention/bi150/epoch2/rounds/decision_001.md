# Decision 001

## Metadata

```json
{
  "schema_version": 2,
  "decision": "proceed",
  "decision_kind": "optimization",
  "round": "001",
  "reference_implementation": "baseline_adapter.py",
  "reference_report": "rounds/report_000.md",
  "language": "triton",
  "backend": "cuda",
  "target_profile": "triton_cuda",
  "runtime_fingerprint_ref": "project.md#runtime-fingerprint",
  "change_scope": "host",
  "change_family": "manual-cuda-graph-workspace-replay",
  "sketch_ref": "rounds/sketch_001.json",
  "sketch_sha256": "199275b85e831238c2f0c9c694d3c4c03550c6681bd7a8e87f3474642b3c1fce",
  "implementation_profile_snapshot_ref": "profile_snapshot/triton_cuda.yaml",
  "implementation_profile_snapshot_sha256": "dc8fa4c0c73caecc78c6a886e5ffb18cb97371f3add7cb382dbe30887743b7ae",
  "project_capability_claim_ref": "profile_snapshot/capability_claim.json",
  "project_capability_claim_sha256": "07aa5d489acb9c21717032087812d264dd5170fe79e7ea2326edb04cab657c1d"
}
```

## Optimization Intent

```json
{
  "bottleneck_class": "host-bound",
  "intervention": "capture the byte-frozen base attention pipeline ONCE as a manual torch.cuda.CUDAGraph over instance-owned static workspace buffers — three small fp16 copy-ins of live query/key/value at the capture boundary, the retained F.scaled_dot_product_attention(q,k,v,scale=0.125,is_causal=True) vendor call plus its view/relayout epilogue recorded verbatim inside the graph over workspace placeholders, one [83,512] fp16 copy-out to caller-visible results outside the replay boundary — and replay it for every target-regime call, with strict regime gating and a permanent two-tier fallback chain (manual replay -> framework-eager base-equivalent); NO torch.compile tier exists anywhere in this candidate (no compile machinery to fail, no inductor heuristics consulted); a new ModelNew.run_out(query,key,value,out) preallocated-output surface is added as required by project.md public_contract and the report_000 kernel-mode limitation; zero tl.dot calls are introduced this round so capability probes P1-P4 are NOT triggered; all device kernels, their argument values, and output bits remain identical to the accepted reference by construction",
  "allowed_changes": [
    "one-time construction of the manual graph: warmup on a side capture stream per torch.cuda.graph recommended pattern, then capture of the EXACT base-equivalent pipeline body over fixed shapes — view prelude on workspace placeholders, retained SDPA call site with unchanged argument values, GQA broadcast omitted-by-construction (num_kv_heads == num_heads == 8 makes the base repeat_interleave branch unreachable), squeeze/transpose/reshape epilogue writing ONLY workspace placeholders",
    "per-call host work on the target regime reduced to: regime-guard predicate evaluation, three small workspace copy-ins (query/key/value [83,8,64] fp16), ONE cudaGraphLaunch-class replay submission, one small copy-out ([83,512] fp16 into an invocation-owned fresh buffer for forward / into the caller-provided buffer for run_out)",
    "SUPERSESSION CLAUSE: this Decision introduces instance-owned static WORKSPACE buffers with full-overwrite-per-call semantics and copy-in/copy-out boundaries defined in the Host Plan; user-visible results are copied out EVERY call into invocation-owned or caller-owned buffers, so no result reuse, no cross-call data carryover, and no returned-workspace-reference exists",
    "two-tier permanent fallback chain built lazily at first use: any exception during warmup/capture/first replay binds the instance permanently down-tier to the framework-eager base-equivalent path; lower tier reachable and correct at all times",
    "addition of ModelNew.run_out(query,key,value,out): writes results bitwise-equal to forward for identical inputs into the caller-provided preallocated buffer before returning; unlocks canonical profile_mode=kernel profiling from round 001 onward (report_000 named limitation)",
    "strictly NO change to: mathematical semantics (scale=0.125, is_causal=True, causal-masked softmax, per-head PV), view-chain semantics, output dtype/shape/layout [83,512] fp16, forward signature, constructor signature; strictly NO tl.dot, NO torch.compile, NO algorithm-substitution fallback anywhere"
  ],
  "invariants": [
    "correctness:pass under the unchanged harness comparator (allclose atol=1e-2 rtol=1e-2 equal_nan, seed 42) THROUGH THE REPLAYED ROUTE",
    "outputs remain single fp16 tensors [83,512]; floating values within tolerance against base semantics in every harness invocation",
    "outputs of the replayed tier are BITWISE-EQUAL to outputs of the framework-eager tier for identical input bits through both entry surfaces (forward and run_out) — the retention proof that capture re-executes identical kernels on identical bits at static addresses; any deviation indicates a capture defect and fails immediately",
    "public ModelNew(num_heads, head_size, scale=None, num_kv_heads=8) constructor and forward(query, key, value) signatures unchanged; run_out(query,key,value,out) added without disturbing forward",
    "the retained vendor SDPA call site keeps identical argument values, shapes, dtypes, ordering, and causal semantics inside the captured region — it remains the only attention computation (retained library call, not an algorithm substitution; capability-claim primary matrix.dot stays unconsumed and its before-fallback probe ladder P1-P4 untriggered)",
    "run_out results bitwise-equal to forward outputs for identical inputs; caller-provided out buffers are written via copy-out every call and never aliased to workspace",
    "caller-selected device preserved; warmup+capture execute ONCE on a dedicated side stream per torch API pattern; afterwards every target-regime call replays stream-safely on the caller's current stream context; model code performs no device-context creation/removal and no synchronization beyond base.py behavior plus one graph-launch submission",
    "workspace discipline: q_in/k_in/v_in are FULLY OVERWRITTEN by copy-in each call before any consumer reads them; every intermediate placeholder is fully rewritten during each replay before consumption; workspace never stores results across calls; nothing computed inside one invocation is returned or reused by another invocation",
    "framework-eager tier remains byte-equivalent to baseline_adapter.py behavior for any regime and always reachable"
  ],
  "expected_wall_improvement_pct": 15.0
}
```

## Unified Sketch

```json
{
  "artifact": "rounds/sketch_001.json",
  "sha256": "199275b85e831238c2f0c9c694d3c4c03550c6681bd7a8e87f3474642b3c1fce",
  "rendering": "normative contract is rounds/sketch_001.json; token-level dataflow is IDENTICAL to accepted rounds/baseline_adapter.py (three copy-ins -> retained fused vendor SDPA(scale=0.125,is_causal=True) on head-major views -> squeeze/transpose/reshape relayout to [83,512]); round 001 changes ONLY the execution strategy: the pipeline's kernel sequence is captured once as a manual CUDA graph over static workspace buffers and replayed per call, with live query/key/value bits copied in at the boundary and results copied out after it, so per-call Python dispatch, view-op routing, allocation planning, and per-launch CPU submission collapse into ONE graph submission while output bits stay identical"
}
```

## Host Plan

```json
{
  "applicability": "required",
  "affected_scope": [
    "ModelNew.forward routing across the two-tier chain (manual-replay tier vs framework-eager tier)",
    "ModelNew.run_out routing across the two-tier chain (copy-out OUTSIDE the replay boundary every call, into the caller-provided buffer)",
    "one-time construction of the manual CUDA graph (warmup + capture) keyed to the exact target regime",
    "static workspace buffer set: q_in[83,8,64]/k_in[83,8,64]/v_in[83,8,64] fp16 placeholders filled by per-call copy-in; internal SDPA-output and relayout temporaries allocated during capture from the framework-owned graph-private memory pool; attn_flat_ws[83,512] fp16 result placeholder",
    "per-call forward copy-out target: one invocation-owned fresh [83,512] fp16 buffer allocated OUTSIDE the graph each call",
    "target-regime guard predicate constants (contiguous fp16 [83,8,64] x3 on the captured device; constructor config num_heads=8/head_size=64/scale=None->0.125/num_kv_heads=8)",
    "permanent tier-binding state flags on the instance (replay_failed)"
  ],
  "state_owner": "the ModelNew instance owns the graph handle, the static workspace tensor set, immutable guard constants, and immutable tier-binding flags; framework-owned graph-private memory pools back allocations performed DURING capture (torch.cuda.graph supported pattern); there are NO module-level mutable caches of any other kind; every user-visible result originates as an invocation-owned buffer (forward) or the caller's own buffer (run_out) filled by per-call copy-out — workspace contents are transient computation state fully rewritten each call and NEVER returned directly or read across calls",
  "lifetime": "graph handle and workspace persist for the instance lifetime after successful capture; the replay_failed flag transitions downward at most once (manual-replay -> eager) and then stays; workspace becomes garbage with module destruction; per-call forward output buffers live exactly one forward call under normal ownership rules",
  "allocation_reuse": "MODEL CODE performs zero per-call allocations INSIDE the replayed region (copy-ins write existing workspace placeholders, replay reuses captured addresses); the ONLY per-call allocations allowed outside the boundary are the forward-path invocation-owned result buffer (fresh torch.empty each call) which run_out replaces with the caller-provided buffer; the eager tier allocates exactly as the accepted reference does; no allocation occurs on later replays from model code",
  "cache_key": ["shape", "dtype", "device"],
  "invalidation": "the manual graph is built only for the exact target regime (three contiguous fp16 [83,8,64] tensors on the current device with constructor config above); ANY other shape/dtype/device/config routes DIRECTLY to the framework-eager tier WITHOUT consulting or creating artifacts; recapture is never triggered by model code; if capture fails once the tier is abandoned for the instance lifetime",
  "concurrency": "one model instance is not shared across concurrent forwards; sequential per-call consumption plus immediate copy-out satisfies graph output-lifetime rules; no module-level mutable model state beyond the declared instance fields and framework-owned pools",
  "device_stream_behavior": "caller-selected device preserved throughout; warmup iterations + graph capture follow the torch.cuda.graph recommended side-stream pattern ONCE at first target-regime use; afterwards each target-regime call runs entirely on the CALLER'S current stream context via stream-safe replay; the retained vendor call keeps recorded semantics inside the captured region; no synchronization beyond base.py behavior plus one graph-launch submission; no device-context creation/removal by model code",
  "unchanged_behavior": [
    "forward(query, key, value) public signature and return structure: single fp16 [83,512] tensor",
    "numerical semantics: scale=0.125 exact power-of-two, causal mask (query index >= key index), per-head softmax(QK^T*scale)*V with num_kv_heads == num_heads == 8",
    "GQA broadcast branch absent-by-construction — behaviorally identical because base.py's repeat_interleave path is unreachable when num_kv_heads == num_heads",
    "framework-eager tier behavior inherited verbatim from baseline_adapter.py for non-target regimes and down-tiered instances",
    "run_out(query,key,value,out) returns None after filling the provided buffer (harness auto_bench.make_profile_call kernel-mode contract lines 489-536)",
    "zero torch.compile usage; zero triton kernel definitions; zero tl.dot calls"
  ]
}
```

## Evaluation Contract

```json
{
  "hypothesis_id": "H-001",
  "intervention": "capture the byte-frozen base attention pipeline ONCE as a manual torch.cuda.CUDAGraph over instance-owned static workspace buffers — three small fp16 copy-ins of live query/key/value at the capture boundary, the retained F.scaled_dot_product_attention(q,k,v,scale=0.125,is_causal=True) vendor call plus its view/relayout epilogue recorded verbatim inside the graph over workspace placeholders, one [83,512] fp16 copy-out to caller-visible results outside the replay boundary — and replay it for every target-regime call, with strict regime gating and a permanent two-tier fallback chain (manual replay -> framework-eager base-equivalent); NO torch.compile tier exists anywhere in this candidate; a new ModelNew.run_out(query,key,value,out) preallocated-output surface is added as required; zero tl.dot calls introduced",
  "expected_causal_chain": [
    "during first-use warmup/capture the single-fused-kernel pipeline (view prelude -> FlashAttnFwdF16Ixmma<128,128,16,64,64,CausalM_t=2> -> relayout epilogue) is recorded against static addresses; afterwards each target-regime call performs only: guard check, three small copy-ins, ONE cudaGraphLaunch-class submission, one small copy-out",
    "all per-call Python op-dispatch (~6 aten op routings incl. unsqueeze/transpose views), temporary-allocation planning, and per-launch CPU submissions elapse into a single graph submission, attacking the report_000 measured host share (~136 us/call of 151.107 us wall at device_ratio 0.0897) while device work stays essentially identical (13.56 us/call Ixmma band +/- small boundary-copy device time)",
    "ATTRIBUTION SCOPING CONTRACT (explicit): under manual replay, intra-graph cat=kernel events may be unattributable in traces (noncanon sibling precedent: groupedtopk r004 branch B — attributed kernel count collapsed to ZERO while outputs stayed correct); TWO-BRANCH adjudication defined below; trace coarsening must NOT be interpreted as mechanism failure while the wall observable passes",
    "unrounded interleaved paired median wall time improves by at least 5% versus baseline_adapter.py under fingerprint 6dc07009... (5% = 7.556 us absolute against 0.151107 ms reference median); expected mechanism magnitude ~15% comes from collapsing dispatch/residue within the measured ~90% host window, remaining conservative against unknown fixed-floor components (seed/sync stay)"
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
    { "name": "wall_time_unrounded_paired_median_ms", "expectation": "at least 5% below the accepted reference median (0.151107 ms) across interleaved pairs at warmup 50 / repeat 100" },
    { "name": "bitwise_tier_retention_equivalence", "expectation": "replayed-tier outputs bitwise-equal to framework-eager-tier outputs for identical input bits on the seed42 regime AND on at least two additional manufactured input suites exercising causal-boundary rows and extreme-magnitude rows, through BOTH forward and run_out entry surfaces (incl. poisoned-buffer run_out attempts x2); plus allclose correctness PASS vs base semantics everywhere" },
    { "name": "fallback_tier_selectivity_and_recovery", "expectation": "a separate non-target first call (e.g., T=41 contiguous fp16) executes the framework-eager tier with base-consistent correct outputs and creates ZERO replay artifacts (no graph handle, no workspace attributes); a following target-regime call on the same instance captures and serves the replayed tier again; tier flags move downward only on failure" },
    { "name": "kernel_count_per_call", "expectation": "TWO-BRANCH PASS: (branch A) attributed cat=kernel launches collapse far below 0.88/call toward <=~0.1/call evidencing single-submission replay with only boundary memcpys visible; OR (branch B) intra-replay launches explicitly unattributable per the attribution scoping contract AND host-side census evidence shows <=~6 cudaMemcpy-class calls/call with NO other GPU submissions while outputs stay correct — record branch taken; failure requires attributed count ≈0.88/call WITH flat wall (capture demonstrably absent)" },
    { "name": "host_dispatch_compression_signature", "expectation": "candidate forward-mode scope census (diagnostic) shows python-op/aten::view-op counts per call dropping versus the reference scope toward the designed minimum (3 aten::copy_-class ops for boundary copies), consistent with ONE replay submission; absence of BOTH compression signatures AND >=5% wall gain fails the hypothesis" }
  ],
  "guardrails": [
    "correctness:pass through the replayed route under the unchanged comparator",
    "outputs remain single fp16 [83,512] tensors within allclose(atol=1e-2, rtol=1e-2) of base semantics",
    "bitwise tier-retention equivalence between replayed and eager tiers on identical inputs through both entry surfaces",
    "current device preserved and stream-safe replay semantics (side-stream one-time capture; caller-stream replay); no model-code cross-instance RESULT state (workspace is full-overwrite transient computation state only)",
    "run_out result equality (bitwise) with forward for identical inputs; caller buffers never aliased to workspace",
    "cold warmup+capture cost stays outside timed medians",
    "no torch.compile / TORCHINDUCTOR usage anywhere in candidate source; zero 'reduce-overhead' strings"
  ],
  "profiling_level": "targeted"
}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`: no cataloged failure matches. The four MLU selection-network/gather/cumsum entries are conditional on expert-selection workloads with tie-ordering surfaces; this operator is dense, selection-free compute (no topk/argmax/sort/index reductions exist), and no gathers, sorts, prefix scans, or winner trees are added here.
- Profile boundaries respected: capability claim primary `matrix.dot` is UNCONSUMED this round (zero tl.dot calls anywhere), so its constrained tile envelope `(32,32)@(32,32)` CANNOT gate the recommended family BY CONSTRUCTION; the before-fallback probe ladder P1–P4 is preflighted in Designer context as NEXT-round enablers for possible device-path families ((ii)/(iii) sub-branches) and triggers only if a future decision introduces dot-based kernels; the reduction.sum fallback stays BLOCKED (maintainer waiver NOT granted).
- Attribution semantics of retaining the vendor kernel (explicit ruling requested by Orchestrator): the optimization family is HOST-DISPATCH COMPRESSION (framework-level). The artifact is governed end-to-end by the triton_cuda implementation profile (CoreX/triton toolchain runtime fingerprint, AST-loader binding, frozen snapshot hashes), but this round contributes ZERO new Triton kernels by design; preserving the vendor-optimal 13.56 µs/call device floor while compressing the ~91% host share IS the measured optimum (report_000 evidence_for_next_round bounds pure device-side gains at ~15 µs ≈ 10% of wall even in theory). The Decision therefore encodes sub-branch (i) RETAINED-Ixmma-CAPTURED-VERBATIM, NOT (ii) Triton-flash-attn-gated-by-ladder and NOT (iii) staged composition; both alternatives remain documented backlog items with named preconditions.
- Root-cause separation from sibling r003: groupedtopk's inductor cudagraph-trees refusal ("mutated inputs") concerned mode='reduce-overhead' Inductor machinery. This candidate contains NO compiler machinery at all — manual `torch.cuda.CUDAGraph` capture consults no such heuristic; its obligations (static addresses, fixed shapes, stable launch order, stream rules) are satisfied BY CONSTRUCTION via the workspace design; if ANY capture component still fails on this build, the permanent eager down-tier converts failure into a measured ≈baseline outcome rather than incorrectness.
- Capture-environment hazards declared for Coder: (i) perform warmup iterations then capture per the torch.cuda.graph context-manager recommended pattern (dedicated side capture stream), ONCE at first target-regime use; (ii) NO host-side branches, prints, .item(), cpu reads, dtype/device checks, or variable control flow inside the captured region — the region is exactly the base-equipeline body over fixed shapes; (iii) allocator ops during capture are legal (graph-private pool); nothing may allocate on LATER replays from model code except the declared per-call forward result buffer allocated OUTSIDE the boundary; (iv) inputs must be read ONLY from the workspace placeholders (q_in/k_in/v_in) inside the region — original query/key/value tensors are referenced solely by the pre-boundary copy-ins; (v) capture failure of ANY component binds the framework-eager tier permanently (lazy construction); no silent partial tiers.
- Numerics: replay re-executes identical device kernels on identical bit patterns (copy-in provides the same input bits the eager tier receives; kernels unchanged), so bitwise tier-retention is structural, not aspirational; softmax max-subtraction needs VALUES only (no index-of-max exists), so unlike groupedtopk no tie-ordering certification applies; fp16-out tolerance 1e-2 covers accumulation-order deltas structurally absent anyway.
- Output-lifetime rule: forward returns a freshly-written invocation-owned buffer each call (never raw workspace references); run_out copies out into the caller buffer BEFORE returning; harness consumes/copies within the iteration — stale-read windows eliminated even under adversarial consumers.
- DANGER notes for Coder binding statement: counts>0 for tf32 knobs/reduced_precision/TORCHINDUCTOR env/torch.compile strings/'reduce-overhead' strings ALL FAIL this round (tiers exclude compiles entirely); additionally any tl.dot / @triton.jit introduction FAILS (design contains none).

## Rationale and Evidence

Canonical anchors: last_accepted = `baseline_adapter.py` @`b8ec3458…` with paired-median basis 0.151107 ms (`rounds/report_000.md` @`a90df70d…`; fingerprint 6dc07009… verified live, formula cross-validated by reproducing the sibling-campaign control fingerprint). Round-000 Verifier-backed facts driving THIS decision:

1. DEVICE PATH IS MINIMAL ALREADY: the entire base computation is ONE fused vendor kernel — `ixattnbkd::FlashAttnFwdF16Ixmma<128u,128u,16u,64u,64u,(CausalM_t)2,(AlibiMode_t)0,__half>` — 0.88 launches/call = **13.56 µs/call** device in the reference scope (candidate scope 14.97 µs/call, identical single-kernel structure, delta noise-level); no bmm/softmax/view-cost kernels exist.
2. THE WALL IS HOST: device_ratio 0.0897/0.0991 ⇒ **≈91%/90% of wall is host floor** around the kernel call (~136 µs/call); report_000's campaign-shaping implication states a pure device-side ceiling of ~15 µs/call (≈10%) ABSENT a host-path intervention — below the bar alone and unreachable given vendor-kernel optimality claims in matrix §四 ("厂商库压制力: 强（Ixmma/TCU）").
3. THE MECHANISM CLASS HAS A PROVEN SIBLING ANALOG (labeled NONCANONICAL prior, same box/harness class): groupedtopk-bi150-round2 report_004 landed **+42.54% direct** via exactly this manual-workspace architecture; its r003 established WHY the inductor route fails (mutation-skip heuristic) and r005 why foreach coalescing does not substitute. Those precedents justify EXPECTATION MAGNITUDE only — every number this round must be re-measured under OUR fingerprint.
4. FAMILY RANKING RESOLVED EXPLICITLY (Orchestrator request): under ~91% host share NO device-side family outranks the graph-wrapper family. H-D (Triton flash-attn replacing SDPA, gated by probe ladder P1–P4) loses ordering on three measured grounds: (a) it does not touch the dominant cost class (it ADDS ~Triton-launch host work unless wrapped by H-A anyway); (b) winning requires beating a vendor-tuned 13.56 µs Ixmma kernel from an unproven Triton backend whose dot envelope beyond (32,32)@(32,32) is UNKNOWN-until-probed — and even PERFECT device parity yields ≈0% wall alone; (c) P1–P4 are UNRUN on this runtime, so binding the recommended family to them would gate Phase-legal progress behind unpromised probes. H-A consumes NONE of the risky envelope (zero tl.dot by construction) while claiming the full measured host window. Sub-branches (ii)/(iii) survive only as CONDITIONED backlog items: precondition = P1–P4 passing AND post-r001 re-measurement showing device share relevant after host compression.
5. ADOPTION-BAR ARITHMETIC AND TWO-SIDED FAILURE INTERPRETATIONS: bar = ≥5% paired on 0.151107 ms = ≥7.556 µs absolute — SMALLER than mechanisms' noise lessons from sibling r003/r005, so this Design (a) commits to the single largest-headroom legal lever (collapse ~137 µs host window into guard + ≤6 small memcpys + 1 submission), and (b) declares explicit two-sided readings in advance: wall <5% WITH compression signatures present ⇒ host floor harder than sibling analog (harness-fixed component larger than estimated) ⇒ honest no-improvement #1, counters alive (valid_no_improvement_limit 3), reopen path via Level-2 host decomposition next round; wall ≥5% WITHOUT compression signatures ⇒ mechanism attribution incomplete ⇒ treat as measurement-design weakness, do NOT bank the win silently; permanent capture failure on build ⇒ tier falls to eager ⇒ ≈baseline ± ε ⇒ no-improvement with capture-defect root cause recorded (never presented as success); any correctness/bitwise-retention deviation ⇒ design/candidate-failed channel, never slack reinterpretation.
6. STREAK CALCULUS HONESTY: total_rounds 0, both streaks 0, max_rounds 20 — aborting today forfeits the highest-EV legal family while counters have full room; proceed.

One-attributable-change compliance: change_scope `host`; single causal lever cn.workspace-manual-replay; device kernels/arguments/semantics untouched; ownership supersession scoped precisely (transient full-overwrite workspace ≠ result caching); the added run_out surface changes NO existing observable behavior (report_000 already required it as the kernel-mode unlock; it enters the public contract field, making it semantic-completion rather than a second intervention).

Artifacts consulted: `rounds/report_000.md` @`a90df70d…`, `rounds/sketch_001.json` @`199275b8…`, `baseline_adapter.py` @`b8ec3458…`, `../base.py` @`dd1359ad…`, `project.md`, `profile_snapshot/triton_cuda.yaml` @`dc8fa4c0…`, `profile_snapshot/capability_claim.json` @`07aa5d48…`, `team-state.md`, `references/invariants.md`, `references/anti-patterns.md`, `references/bottleneck-judgment.md`, `references/decision-template.md`, `auto_bench.py` @`71fb3ad0…` (AST loader + make_profile_call contract read), `kernels/track1-triton/summary_all_backends.md` §四 @`f899c82a…`, sibling campaign `groupedtopk/bi150-round2/final_summary.md` @`7278f1f8…` + `rounds/report_004.md` @`c79cc018…` + `rounds/decision_004.md` + `rounds/sketch_004.json` (labeled noncanonical priors), archived epoch-1 `bi150/{rounds/report_000.md,final_summary.md}` (stale fingerprints, noncanonical).

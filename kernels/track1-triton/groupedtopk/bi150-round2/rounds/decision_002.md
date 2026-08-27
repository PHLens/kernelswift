# Decision 002

## Metadata

```json
{
  "schema_version": 2,
  "decision": "proceed",
  "decision_kind": "optimization",
  "round": "002",
  "reference_implementation": "triton_grouped_topk_r2_001.py",
  "reference_report": "rounds/report_001.md",
  "language": "triton",
  "backend": "cuda",
  "target_profile": "triton_cuda",
  "runtime_fingerprint_ref": "project.md#runtime-fingerprint",
  "change_scope": "host",
  "change_family": "compile-graph-default",
  "sketch_ref": "rounds/sketch_002.json",
  "sketch_sha256": "0ccbec4756d447d1365d0cae81ff2f8e3a020ecc3b99d84bbe2d4d7ce5d84cf3",
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
  "intervention": "wrap the accepted round-001 staged Triton pipeline behind torch.compile(mode='default', dynamic=False) as one shared compiled callable used by both forward and run_out on the target regime (contiguous fp32 [T,256] gating, T=83 fixed construction config), gated by strict shape/config checks, with a permanent per-instance fallback to the unmodified round-001 staged execution (and its framework-eager fallback) on any dynamo/inductor failure or non-target regime; all three Triton stages and both torch.topk call sites remain byte-for-byte unchanged",
  "allowed_changes": [
    "ModelNew.forward and ModelNew.run_out routing through a shared compiled callable constructed once at first use for the target regime",
    "torch.compile configuration limited to mode='default', dynamic=False on the fixed-shape pipeline",
    "target-regime guard predicate and permanent eager-fallback binding on any compilation/tracing exception or non-target regime",
    "no change to the three @triton.jit kernels, to either torch.topk call site's arguments or semantics, or to the output contract"
  ],
  "invariants": [
    "correctness:pass under the unchanged harness comparator",
    "exact int32 ID equality including all-equal/two-expert-tie/structured-group/duplicate-max tie cases",
    "floating outputs within allclose(atol=1e-2, rtol=1e-2)",
    "public ModelNew constructor and forward(hidden_states, gating_output) signatures unchanged",
    "both torch.topk call sites keep identical argument values, shapes, dtypes, ordering, and tie behavior — they remain the only selectors (no algorithm substitution)",
    "run_out results bitwise-equal to forward outputs for identical inputs",
    "caller device and current stream preserved; no device-context mutation",
    "per-forward buffer ownership; no cross-instance or cross-call tensor caching by model code",
    "compiled execution must be bypassable: eager path stays reachable and byte-equivalent"
  ],
  "expected_wall_improvement_pct": 10.0
}
```

## Unified Sketch

```json
{
  "artifact": "rounds/sketch_002.json",
  "sha256": "0ccbec4756d447d1365d0cae81ff2f8e3a020ecc3b99d84bbe2d4d7ce5d84cf3",
  "rendering": "normative contract is rounds/sketch_002.json; the token-level dataflow is IDENTICAL to the accepted round-001 pipeline (stage-A softmax+group-max -> library topk#1 over group_scores_out -> stage-B arithmetic lane-membership masking -> library topk#2 over masked_scores -> stage-C renorm/scale/narrow); round 002 changes the EXECUTION strategy of that same dataflow: dynamo/inductor compile the fixed-shape region as one callable so residual Python dispatch, temporary-allocation planning, and launch-wrapper overhead collapse into compiled execution, while every value produced is mathematically and orderingly identical"
}
```

## Host Plan

```json
{
  "applicability": "required",
  "affected_scope": [
    "ModelNew.forward routing and compiled-callable invocation",
    "ModelNew.run_out routing and compiled-callable invocation",
    "one-time construction of the torch.compile wrapper keyed on the target regime",
    "target-regime guard predicate evaluation per call",
    "permanent eager-fallback binding state on the instance"
  ],
  "state_owner": "the ModelNew instance owns the compiled callable handle, the guard-predicate constants from __init__, and a small immutable fallback flag; process-global dynamo/inductor bytecode/binary caches may exist but contain NO tensor data (build caches analogous to runtime kernel-binary caches); every tensor, including run_out-provided output buffers, remains owned by the calling forward/run_out invocation",
  "lifetime": "the compiled callable persists for the instance lifetime after successful construction; all tensors live exactly one call as in round 001; fallback flag transitions at most once from compiled to eager and never back",
  "allocation_reuse": "no reuse across forwards: temporaries are allocated fresh inside each executed implementation exactly as round 001 does; compilation artifacts are code objects, not tensors",
  "cache_key": ["shape", "dtype", "device"],
  "invalidation": "the compiled callable is built only for the exact target regime (contiguous fp32 [83,256] gating on current device with constructor config topk=8/renormalize=True/G=8/KG=4); any cache-key component change routes to the unmodified eager staged path instead of reusing the artifact; if allocation reuse of tensors were ever introduced it would additionally require full cache-key revalidation before reuse",
  "concurrency": "one model instance is not shared across concurrent forwards; no module-level mutable model state beyond framework-owned process-global build caches is introduced; no model-code global scratch exists",
  "device_stream_behavior": "preserve the caller-selected device and current stream; dynamo guards run host-side only; compiled kernels launch on the current stream like their eager counterparts; retained torch.topk calls keep ordinary current-stream semantics; no device-context creation/removal; no synchronization beyond base.py behavior",
  "unchanged_behavior": [
    "forward(hidden_states, gating_output) public signature and observable batch-size assertion",
    "returned tuple (topk_weights fp32[83,8], topk_ids int32[83,8])",
    "run_out(gating_output, topk_weights, topk_ids) writes byte-identical results to forward for identical inputs into provided buffers before returning (harness contract auto_bench.make_profile_call lines 520-536)",
    "the three @triton.jit stage kernels' source and launch semantics",
    "both torch.topk call sites and their argument values",
    "scoring_func and routed_scaling_factor semantics",
    "fallback to framework-eager for non-target regimes exactly as accepted in round 001"
  ]
}
```

## Evaluation Contract

```json
{
  "hypothesis_id": "H-002",
  "intervention": "wrap the accepted round-001 staged Triton pipeline behind torch.compile(mode='default', dynamic=False) as one shared compiled callable used by both forward and run_out on the target regime (contiguous fp32 [T,256] gating, T=83 fixed construction config), gated by strict shape/config checks, with a permanent per-instance fallback to the unmodified round-001 staged execution (and its framework-eager fallback) on any dynamo/inductor failure or non-target regime; all three Triton stages and both torch.topk call sites remain byte-for-byte unchanged",
  "expected_causal_chain": [
    "dynamo traces the fixed-shape pipeline once; inductor partitions ATen ops where profitable while routing torch.topk to the SAME ATen vendor kernels, preserving gatherTopK/bitonicSortKVInPlace kind, counts, and ordering semantics",
    "residual Python op-dispatch, temporary-allocation planning, and Triton launch-wrapper overhead between the ~7 device kernels collapse into one compiled entry path",
    "host-side time outside kernel execution shrinks against the report_001 residual (~75% of wall, ~0.31 ms/call) while candidate-scope device time stays approximately flat (105.310 us/call basis) apart from glue-kernel removals",
    "unrounded interleaved paired median wall time improves by at least 5% versus triton_grouped_topk_r2_001.py under fingerprint 8deb1b01..."
  ],
  "primary_metric": { "name": "wall_time", "expected_improvement_pct": 10.0 },
  "causal_graph": {
    "nodes": [
      "cn.compile-dispatch-compression",
      "cn.host-dispatch-time",
      "cn.kernel-count-per-call",
      "cn.device-us-per-call",
      "cn.wall-time"
    ],
    "edges": [
      ["cn.compile-dispatch-compression", "cn.host-dispatch-time"],
      ["cn.compile-dispatch-compression", "cn.kernel-count-per-call"],
      ["cn.host-dispatch-time", "cn.wall-time"],
      ["cn.kernel-count-per-call", "cn.device-us-per-call"],
      ["cn.device-us-per-call", "cn.wall-time"]
    ]
  },
  "mechanism_observables": [
    { "name": "kernel_count_per_call", "expectation": "decrease relative to accepted r001 baseline 6.97/call (glue-kernel fusion allowed; not required to drop if tracing graph-breaks), with no increase above 7.5/call" },
    { "name": "retained_library_topk_kernels", "expectation": "gatherTopK and bitonicSortKVInPlace persist at ~1.99 counts/call each in the candidate scope, proving no selection substitution" },
    { "name": "device_us_per_call", "expectation": "approximately unchanged versus accepted r001 105.310 us/call (tolerance band 90-130 us/call), confirming the mechanism is host-side rather than device-side" },
    { "name": "wall_time_unrounded_paired_median_ms", "expectation": "at least 5% below the accepted reference median across interleaved pairs at warmup 50 / repeat 100" }
  ],
  "guardrails": [
    "correctness:pass",
    "exact ID equality including all-equal/two-expert-tie/structured-group-tie/duplicate-max tie suites",
    "outputs remain fp32 weights and int32 ids with shapes [83,8]",
    "current device/stream preserved; no cross-instance tensor state",
    "run_out result equality with forward for identical inputs",
    "compiled-to-eager fallback exercised by the non-target regime check without altering visible behavior"
  ],
  "profiling_level": "targeted"
}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`: no cataloged failure matches. No generic `tl.gather`, no sort network, no cumsum/prefix scan, no winner-tree is added — kernels are byte-for-byte those already accepted in round 001. Entries 011–016 remain conditional evidence from MLU590-H8 fingerprints; cautions only.
- Profile boundaries respected: `tl.argmax` Constrained tie-order capability is untouched (no new selection logic anywhere); `num_warps`/`num_stages` stay unset; sketch hints remain empty.
- `torch.compile` is Constrained on the frozen snapshot (legality/lifecycle unproven): therefore the fallback chain is NORMATIVE — guard predicate routes non-target regimes to the unmodified staged path, and ANY exception during construction or first invocation permanently binds the instance to that path. A partial-graph outcome (graph breaks around direct `@triton.jit` launches) is acceptable; the expectation remains falsifiable wall improvement, not perfect tracing.
- Attribution risk: default mode preserves record_function spans, but this build double-records scopes (report_001 deviation P1 precedent). If overlap rejection recurs, Verifier's offline host-window salvage convention applies; wall time remains the sole adoption basis either way.
- Numerics hazards declared: Inductor functionalization/cloning must preserve bits (comparator verifies regardless); Inductor partitioning of `torch.topk` keeps the SAME ATen vendor kernels (observed kind/count guardrail makes substitution detectable); custom-Triton numerics are untouched because kernels are opaque to Inductor.
- The two largest device contributors (86.7 µs/call in `torch.topk` sites) are NOT addressed this round: entering them requires CHECK-TIE-style on-device derivation of the vendor tie order, which cannot use tl.argmax ordering; that gate stays open for a later round.
- DANGER note for Coder: no `reduced_precision_reduction`, precision flags, backend option flips, or cache-size environment overrides are permitted; compilation configuration is restricted to `mode='default', dynamic=False`.

## Rationale and Evidence

Canonical Verifier facts (rounds/report_001.md @`f9fbb9bf38f8d63ff9eeeed39bbd2e823ed6a34784f5121901a86e279c7a4fcc`, verdict pinned `ff1e49c6…`): paired medians 0.470655 → 0.416933 ms (+11.41%); candidate scope 6.97 kernels/call, 105.310 µs/call device; retained vendor pair dominates device (`gatherTopK` 49.371 + `bitonicSortKVInPlace` 37.288 µs/call ≈ 86.7 µs/call of 105.3); device_ratio fell 0.383 → 0.253, leaving ~75% of wall (~0.31 ms/call) OUTSIDE kernel execution — added Triton-launch/torch.empty host work partially offset the round-001 device win (report fact).

Family selection logic among evidence-bounded options: (i) touching the retained `torch.topk` sites requires the tie-exactness audit gate (tl.argmax excluded by profile Constrained status) — deferred; (ii) legal stage-trio kernel merging does not exist because stages B and C depend on library selections (dataflow barriers in the accepted sketch); (iii) remaining launch-glue trimming alone cannot clear 5%; (iv) compile-graph dispatch compression targets precisely the dominant host share and carries the strongest lineage validation available: −19.99% then −22.51% further in epoch 1 (NONCANONICAL priors on measurement fingerprint `57bf01...` — risk information, not comparable evidence) applied on an almost identical staged pipeline. Round 002 takes the lower-risk default mode first, deliberately deferring `mode='reduce-overhead'` graph replay to a later round to keep mechanism observability (attribution survives default mode; it collapses under CUDA Graph replay, as epoch-1 r009 recorded).

Expected ≥10% declaration rationale: prior family gain on near-identical starting composition was ~−20%, and even a substantially degraded fraction clears the 5% bar; declaring 10.0 keeps the hypothesis falsifiable well above adoption threshold.

One-attributable-change compliance: change_scope is `host`; no kernel source changes; the single mechanism (compiled dispatch) connects causally to wall via cn.compile-dispatch-compression → cn.host-dispatch-time → cn.wall-time with the r001-candidate as controlled reference.

Artifacts consulted: `rounds/report_001.md`, `rounds/verdict_001.json`, `triton_grouped_topk_r2_001.py` @`4ae64cad…`, `rounds/report_000.md`, `project.md`, `profile_snapshot/triton_cuda.yaml` @`dc8fa4c0…`, `profile_snapshot/capability_claim.json` @`2e6ee49d…`, `references/invariants.md`, `references/anti-patterns.md`, `references/bottleneck-judgment.md`, `auto_bench.py`, and lineage reports 000/004/008/009 under `../bi150/` (labeled noncanonical).

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
  "change_scope": "mixed",
  "change_family": "preprocess-fusion-triton-stages",
  "sketch_ref": "rounds/sketch_001.json",
  "sketch_sha256": "637917e07b4461258ea714d42021e2e5537e21d19765b57bc9cc1552ef6f6985",
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
  "intervention": "replace the eager framework preprocessing/postprocessing chain (softmax, per-group max reduce, group-mask build, masked_fill, renormalize divide, int64-to-int32 id cast) with three direct per-token Triton stages while retaining both exact library torch.topk selection calls unchanged, and expose the ModelNew.run_out preallocated-output surface required for kernel-mode profiling",
  "allowed_changes": [
    "stage-A fused softmax + group-max Triton kernel writing scores_out and group_scores_out",
    "stage-B group-membership masking Triton kernel producing masked_scores from scores_out and library-selected group ids",
    "stage-C renormalize/routed-scaling/id-narrowing Triton kernel producing final fp32 weights and int32 ids",
    "ModelNew.forward orchestration, buffer allocation, and launch code around those stages",
    "ModelNew.run_out(gating_output, topk_weights, topk_ids) preallocated-output execution path"
  ],
  "invariants": [
    "correctness:pass under the unchanged harness comparator",
    "exact int32 ID equality including all-equal/two-expert-tie/structured-group-tie cases",
    "floating outputs within allclose(atol=1e-2, rtol=1e-2)",
    "public ModelNew constructor and forward(hidden_states, gating_output) signatures unchanged",
    "both torch.topk call sites keep identical argument values, shapes, dtypes, ordering, and tie behavior",
    "caller device and current stream preserved; no device-context mutation",
    "per-forward buffer ownership; no cross-instance or cross-call caching of outputs or temporaries",
    "scoring_func and routed_scaling_factor parameter semantics preserved"
  ],
  "expected_wall_improvement_pct": 8.0
}
```

## Unified Sketch

```json
{
  "artifact": "rounds/sketch_001.json",
  "sha256": "637917e07b4461258ea714d42021e2e5537e21d19765b57bc9cc1552ef6f6985",
  "rendering": "normative contract is rounds/sketch_001.json; structure: parallel token over T rows; stage-A loads gating row [E=256], computes softmax then per-group max over [G=8] groups of 32 contiguous experts, stores scores_out[T,E] and group_scores_out[T,G]; library torch.topk(group_scores_out,k=KG=4) yields sel_groups[T,KG]; stage-B reloads a scores row and applies lane membership (floor(expert/32) in sel_tile ? score : -inf) storing masked_scores[T,E]; library torch.topk(masked_scores,k=K=8) yields topk_vals/topk_ids; stage-C divides selected values by their row sum, applies routed_scaling_factor, narrows int64 ids to int32, stores out_weights[T,K] fp32 and out_ids[T,K] i32. hidden_states[T,H] is declared but never loaded: it participates only in the batch-size assertion."
}
```

## Host Plan

```json
{
  "applicability": "required",
  "affected_scope": [
    "ModelNew.forward orchestration",
    "ModelNew.run_out entry point",
    "stage-A/stage-B/stage-C Triton kernel wrappers",
    "per-call temporary tensors scores_out, group_scores_out, masked_scores"
  ],
  "state_owner": "the ModelNew instance owns compiled Triton kernel handles and immutable Python-side configuration constants only; every tensor, including the output buffers supplied by the harness to run_out, remains owned by the calling forward/run_out invocation",
  "lifetime": "temporaries live for exactly one forward call; run_out writes only caller-provided buffers and allocates nothing persistent; compiled kernel binaries persist with the instance through the runtime's own compile cache keyed on constexpr/dtype configuration",
  "allocation_reuse": "no reuse across forwards in this round: allocate fresh torch.empty temporaries on gating_output.device at every call; within one call each temporary is written once and read downstream",
  "cache_key": ["shape", "dtype", "device"],
  "invalidation": "not applicable this round because no cross-forward tensor cache exists; if allocation reuse is ever introduced it must re-validate every cache-key component (shape, dtype, device) before reuse",
  "concurrency": "one model instance is not shared across concurrent forwards; no module-level mutable state is introduced; kernels carry no global scratch",
  "device_stream_behavior": "preserve the caller-selected device and current stream; stages launched through direct grid syntax on the current stream; retained torch.topk calls keep ordinary current-stream semantics; no device-context creation/removal and no synchronization beyond what base.py already performs",
  "unchanged_behavior": [
    "forward(hidden_states, gating_output) public signature and the observable batch-size assertion",
    "returned tuple (topk_weights fp32[83,8], topk_ids int32[83,8])",
    "run_out(gating_output, topk_weights, topk_ids) computes results byte-identical to forward outputs for identical inputs into the provided buffers; harness auto_bench.make_profile_call invokes it as run_out(gating_output, *reference_outputs, **model.run_kwargs) with the return value ignored (contract read from auto_bench.py lines 516-536); model.run_kwargs defaults to {} via getattr when absent",
    "both torch.topk call sites and their argument values",
    "scoring_func and routed_scaling_factor semantics"
  ]
}
```

## Evaluation Contract

```json
{
  "hypothesis_id": "H-001",
  "intervention": "replace the eager framework preprocessing/postprocessing chain (softmax, per-group max reduce, group-mask build, masked_fill, renormalize divide, int64-to-int32 id cast) with three direct per-token Triton stages while retaining both exact library torch.topk selection calls unchanged, and expose the ModelNew.run_out preallocated-output surface required for kernel-mode profiling",
  "expected_causal_chain": [
    "stage-A replaces the framework softmax warp kernel plus the MaxOps per-group max reduce with one direct Triton kernel",
    "stage-B replaces zeros/scatter/expand/bitwise_not/masked_fill mask construction with arithmetic lane-membership selection over floor(expert/32)",
    "stage-C folds the renormalize divide, the routed-scaling multiply, and the int64-to-int32 id cast into one small kernel",
    "kernel launches per call drop from 14.94 toward roughly 7 while retained gatherTopK/bitonicSortKVInPlace selection stays intact",
    "framework-op host dispatch work shrinks together with the device time of the removed small kernels",
    "unrounded interleaved paired median wall time improves by at least 5% versus baseline_adapter under measurement fingerprint 8deb1b01..."
  ],
  "primary_metric": { "name": "wall_time", "expected_improvement_pct": 5.0 },
  "causal_graph": {
    "nodes": [
      "cn.triton-stage-fusion",
      "cn.kernel-count-per-call",
      "cn.host-dispatch-time",
      "cn.device-us-per-call",
      "cn.wall-time"
    ],
    "edges": [
      ["cn.triton-stage-fusion", "cn.kernel-count-per-call"],
      ["cn.triton-stage-fusion", "cn.host-dispatch-time"],
      ["cn.kernel-count-per-call", "cn.device-us-per-call"],
      ["cn.host-dispatch-time", "cn.wall-time"],
      ["cn.device-us-per-call", "cn.wall-time"]
    ]
  },
  "mechanism_observables": [
    { "name": "kernel_count_per_call", "expectation": "decrease relative to baseline_adapter 14.94 in separately scoped Level-1 summaries" },
    { "name": "device_us_per_call", "expectation": "decrease relative to baseline_adapter 178.84 us/call" },
    { "name": "retained_library_topk_kernels", "expectation": "gatherTopK and bitonicSortKVInPlace remain present (~2 counts/call each) in the candidate scope, proving exact-selection retention" },
    { "name": "wall_time_unrounded_paired_median_ms", "expectation": "at least 5% below the accepted reference median across interleaved pairs at warmup 50 / repeat 100" }
  ],
  "guardrails": [
    "correctness:pass",
    "exact ID equality on the harness comparator including all-equal/two-expert-tie/structured-group tie suites",
    "outputs remain fp32 weights and int32 ids with shapes [83,8]",
    "current device/stream preserved; no cross-instance state",
    "run_out result equality with forward for identical inputs"
  ],
  "profiling_level": "targeted"
}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`. No cataloged failure matches this design's formulation: there is NO generic `tl.gather` compaction, NO sort network, NO cumsum/prefix scan, and NO hierarchical parallel winner-tree. Masking is pure lane-membership arithmetic (`floor(expert/32) in sel_tile`), a narrow fixed-lane select. Entries 011/012/013/016 are CONDITIONAL evidence recorded on MLU590-H8 / Triton-3.2.0 fingerprints whose preconditions do not match this runtime; they are treated as cautions, never as rules.
- Near-tie ordering risk (primary correctness hazard): candidate softmax feeds the RETAINED library `torch.topk`, but comparator IDs must match against reference-computed values bit-for-bit in comparison outcome. Mitigations: replicate the standard fp32 `exp(x - rowmax) / sum(exp(x - rowmax))` path end-to-end in fp32, avoid fused-multiply reassociation in reductions, and treat all three historical tie suites as mandatory gates: all-equal case `[7,6,4,5,1,0,2,3]`; two-expert-tie `[1,0,2,3,4,5,6,7]`; structured-group `[32,0,64,96,4,3,1,2]` (suite expectations are NONCANONICAL epoch-1 records used as test-design guidance, not as canonical evidence).
- `tl.argmax` is NOT used anywhere in this design, so the Constrained repeated-argmax/tie capability is never normatively exercised. `num_warps`/`num_stages` remain Unknown on this profile revision and stay UNSET: no launch hint enters sketch hints, launches use proven direct syntax defaults.
- `run_out` binding is hard: report_000 records that `--profile-mode kernel` structurally requires callable `ModelNew.run_out` (harness raises `KsCompareError` otherwise). The exact contract was read from `auto_bench.py::make_profile_call`: called as `run_out(gating_output, *output_args, **run_kwargs)` where `output_args` are the reference forward's tensors and `run_kwargs = dict(getattr(model, 'run_kwargs', {}))`; the return value is ignored, so run_out must complete its writes into the provided buffers before returning.
- `-inf` masked lanes reach the retained expert `torch.topk` exactly as in base semantics because stage-B produces the same logical value pattern (`score` where member else `-inf`); no reordering of lanes occurs.
- Profile binding (historical note, resolved): during initial round-001 authoring no machine-readable `triton_cuda` implementation profile existed and the project declared only the Markdown rendering; Orchestrator subsequently promoted the canonical `skills/kernel-opt-loop/profiles/triton_cuda/profile.yaml` v1 (`profile_status=partial`) and materialized the frozen campaign snapshot `profile_snapshot/triton_cuda.yaml` (`load_profile` verified), re-pinning `profile_snapshot/capability_claim.json` to it. This Decision binds directly to that machine-readable snapshot; the Markdown rendering remains in place purely as provenance. The claim's `primary_contract: reduction.argmax-grouped-selection` with `probe_policy: optional` governs fallback planning; NO algorithm substitution is declared (no `uses_algorithm_substitution`, no fallback_provenance).

## Rationale and Evidence

Canonical Verifier facts (rounds/report_000.md, fingerprint-matched): eager baseline wall medians 0.483530 ms (reference side) / 0.481109 ms (adapter side); ~14.94 kernels/call; device 180.11 vs 178.84 us/call with device_ratio ≈ 0.372 — host-dominated, ≥60% of each call outside kernel execution; contributor table shows removable framework chain beyond the retained top-k pair (`gatherTopK` 49.26 us/call + `bitonicSortKVInPlace` 37.21 us/call stay by design; removable pieces include `reduce_kernel MaxOps` 18.13, softmax warp kernel, scatter/fill/not/masked_fill chain, `direct_copy` 10.02, div, and misc copies).

Expected-gain justification: removing roughly half the launches plus ~60–90 us/call of device time from an execution profile whose dominant term is launch/dispatch overhead plausibly clears the 5% bar while staying conservative relative to the host-dominance headroom (62%+ of wall). The same change family delivered −7.46% once in lineage — NONCANONICAL prior on a different measurement fingerprint (`57bf01...` vs canonical `8deb1b01...`) — so it informs risk, not evidence.

One-attributable-change compliance: the mixed change's kernel piece (three Triton stages) and host piece (orchestration + run_out surface) are inseparable — the wrapper exists solely to host the stages — and separately observable through scoped kernel-count/device breakdown plus the forward-vs-run_out equality guardrail. Compile/graph-capture families are deliberately deferred to later rounds (their profiler-attribution caveat under CUDA Graph replay would corrupt round-001 mechanism observables).

Adoption remains controlled exclusively by unrounded paired median wall time versus `last_accepted_kernel` under fingerprint `8deb1b01...`; profiler numbers are diagnostic. Backlog source: `state/designer_context.md` item SEL-FUSE-01 (rank 1). Artifacts consulted: `rounds/report_000.md`, `project.md`, `profile_snapshot/triton_cuda.md`, `profile_snapshot/capability_claim.json`, `../base.py` @`12f33248...`, `references/invariants.md`, `references/anti-patterns.md`, `references/bottleneck-judgment.md`.

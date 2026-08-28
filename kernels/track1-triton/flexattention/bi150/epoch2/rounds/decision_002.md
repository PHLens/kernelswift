# Decision 002

## Metadata

```json
{
  "schema_version": 2,
  "decision": "proceed",
  "decision_kind": "optimization",
  "round": "002",
  "reference_implementation": "baseline_adapter.py",
  "reference_report": "rounds/report_000.md",
  "language": "triton",
  "backend": "cuda",
  "target_profile": "triton_cuda",
  "runtime_fingerprint_ref": "project.md#runtime-fingerprint",
  "change_scope": "mixed",
  "change_family": "triton-attention-dispatch-collapse",
  "sketch_ref": "rounds/sketch_002.json",
  "sketch_sha256": "fb5bec0b957a04ffa19d20edb2f0fdb92de156c0aea6429b1c796a86b89bd87c",
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
  "intervention": "replace the entire ~34-aten-op base path (SDPA dispatch stack + 12 as_strided + 8 transpose + 7 empty + 3 unsqueeze + output views) with ONE direct-launched Triton blocked-attention kernel plus a minimal two-op forward (one torch.empty + one kernel launch): grid = (H x ceil(T/BM)) = (8x3) = 24 programs, num_warps=1, BM=BN=32, head-dim D=64 processed as two 32-chunks; fp16 inputs loaded and WIDENED to fp32 BEFORE tl.dot so every dot call stays strictly inside the proven (32,32)@(32,32) fp32/fp32->fp32 envelope; online (running-max) softmax over the 3 key-tiles in fp32 with -inf causal masking (exp(-inf)=0 exact); PV accumulation as the same proven-shape dots; results STORED DIRECTLY into the final [83,512] fp16 token-major layout so no output view/copy ops exist at all; run_out(query,key,value,out) writes the caller buffer the same way; the kernel is the only device work and the launch is the only submission",
  "allowed_changes": [
    "new Triton attention kernel replacing the retained-vendor-SDPA device path (the one sanctioned device change; all tl.dot usage confined to proven shapes/dtypes: (32,32)@(32,32) fp32 operands, fp32 accumulator, num_warps=1, num_stages unset)",
    "forward host path collapse: torch.empty([83,512] fp16) + single kernel[(24,)] launch; no view ops, no intermediate tensors, no reshape/copy; constexpr-frozen shapes/strides (contiguous [83,8,64] inputs)",
    "run_out(query,key,value,out) direct kernel write into the caller-provided buffer (same 4-arg signature as r001 per project.md public_contract; report_001 D2 harness-arity limitation is a harness-side fact for 3-input ops and stays documented, not worked around)",
    "stateless module: NO workspace, NO graph, NO cache, NO cross-call state of any kind (r001's instance-owned artifacts are deliberately absent); every call is independent",
    "strictly NO: torch.compile / reduce-overhead / inductor machinery, tl.dot outside the proven envelope, fp16-operand dots, num_warps>1, algorithm-substitution fallback (reduction.sum stays BLOCKED, waiver NOT granted)"
  ],
  "invariants": [
    "correctness:pass under the unchanged harness comparator (allclose atol=1e-2 rtol=1e-2 equal_nan, seed 42)",
    "outputs remain single fp16 tensors [83,512] with causal MHA semantics (scale=0.125, per-head online softmax, masked upper triangle contributes exactly zero)",
    "public ModelNew constructor and forward(query, key, value) signature unchanged; run_out(query,key,value,out) fills the caller buffer before returning None",
    "run_out results BITWISE-EQUAL to forward outputs for identical inputs (same kernel, same bits, deterministic launch configuration)",
    "capability legality: tl.dot instances are EXACTLY (32,32)@(32,32) with fp32 operands/accumulator (proven envelope; fp16->fp32 widening casts are the only precision bridge); fp16 tl.load/tl.store paths; num_warps=1; no num_stages; binding ledger must audit every tl.dot call against this constraint",
    "stateless execution: no instance attributes written at forward time; caller-selected device preserved; kernel launched on the caller's current stream; no device-context creation/removal; no synchronization added beyond base.py behavior",
    "zero torch.compile / TORCHINDUCTOR / 'reduce-overhead' strings anywhere in candidate source",
    "AST-loader compliance: module-level constants are safe literals; @triton.jit kernel and helpers survive the loader unchanged (imports/classdefs/functiondefs retained)"
  ],
  "expected_wall_improvement_pct": 8.0
}
```

## Unified Sketch

```json
{
  "artifact": "rounds/sketch_002.json",
  "sha256": "fb5bec0b957a04ffa19d20edb2f0fdb92de156c0aea6429b1c796a86b89bd87c",
  "rendering": "normative contract is rounds/sketch_002.json; per program (head, mtile): load q as two fp32-widened 32-chunks, loop the 3 key-tiles loading k/v chunks, accumulate QK^T via two proven-shape (32,32)@(32,32) fp32 dots per tile scaled by 0.125, apply causal -inf mask, online running-max softmax update producing rescaled accumulator chunks and p~, PV via the same proven-shape dots, then normalize by the total softmax denominator and store directly into the final [83,512] fp16 token-major layout; ONE kernel launch replaces the entire SDPA dispatch stack, all view routings, all allocations, and the output relayout"
}
```

## Host Plan

```json
{
  "applicability": "required",
  "affected_scope": [
    "ModelNew.forward host path: torch.empty output allocation + single direct kernel launch (two python-visible ops + one cudaLaunchKernel-class submission)",
    "ModelNew.run_out host path: single direct kernel launch writing the caller-provided buffer (no allocation)",
    "kernel launch configuration: grid (24,), num_warps=1, constexpr shapes T=83/H=8/D=64/strides frozen at compile time",
    "per-call output allocation on the forward path (invocation-owned fresh buffer)"
  ],
  "state_owner": "NOBODY — the module is deliberately stateless: no instance attributes are written or read at forward time, there is no workspace, no graph handle, no cache, and no cross-call tensor; the only persistent objects are the @triton.jit kernel object and its compile cache owned by the Triton framework (one-time JIT compile at first call, outside timed medians per harness warmup)",
  "lifetime": "no model-owned lifetime beyond the module object itself; per-call forward output buffers live exactly one call under normal ownership; run_out writes memory owned by the caller",
  "allocation_reuse": "NONE by design: forward allocates one fresh [83,512] fp16 output per call via torch.empty; run_out performs zero allocations; no buffer reuse cache exists to key or invalidate (r001's workspace machinery is intentionally absent)",
  "cache_key": ["not-applicable: candidate owns no cross-call caches; every forward allocates its output fresh and nothing else persists"],
  "invalidation": "not-applicable to model state (stateless); Triton JIT specialization keys off constexpr shape/dtype constants fixed at module level — any hypothetical shape change would recompile naturally outside the target regime, and the target regime is the only benchmarked one",
  "concurrency": "stateless design is concurrency-safe by construction: no shared mutable state exists; one model instance per harness model slot as the harness constructs it; no cross-instance hazards are possible",
  "device_stream_behavior": "caller-selected device preserved (kernel launches on the current device context); kernel launches on the CALLER'S CURRENT STREAM; no side streams, no capture, no synchronization added by model code; harness's own seed/synchronize behavior untouched",
  "unchanged_behavior": [
    "forward(query, key, value) public signature and return structure: single fp16 [83,512] tensor",
    "numerical semantics: causal MHA, scale=0.125 exact power-of-two, fp32 accumulation, masked upper-triangle keys contribute exactly zero (exp(-inf)=0), per-head independent softmax denominators",
    "GQA branch absent-by-construction (num_kv_heads == num_heads == 8 makes base.py's repeat_interleave unreachable)",
    "run_out(query,key,value,out) contract identical to r001: fills caller buffer, returns None, never aliases model state (there is none)",
    "zero torch.compile usage; the ONE device kernel is the candidate's own Triton kernel (intentional, sanctioned change this round)"
  ]
}
```

## Evaluation Contract

```json
{
  "hypothesis_id": "H-002",
  "intervention": "replace the entire ~34-aten-op base path with ONE direct-launched Triton blocked-attention kernel plus a minimal two-op forward (torch.empty + kernel launch): 24-program grid, num_warps=1, proven-envelope (32,32)@(32,32) fp32 dots on widened operands, online fp32 softmax with exact -inf causal masking, direct final-layout [83,512] fp16 stores; run_out writes caller buffers identically; module stateless",
  "expected_causal_chain": [
    "host side: ~34 aten cpu_ops/call (12 as_strided + 8 transpose + 7 empty + 3 unsqueeze + sdpa stack) collapse to <=3 (empty + launch-visible ops), removing the per-op aten dispatch cost measured transitively by r001 at ~0.6-1.0 us/op — a ~20-32 us/call python-side prize against a 7.556 us absolute bar",
    "submission side: base pays 1 cudaLaunchKernel; candidate pays 1 kernel launch — submission count UNCHANGED, so r001's regression source (5 GPU submissions + per-call cudaDeviceSynchronize/cudaDriverGetVersion added by the replay route) is structurally ABSENT here",
    "device side (TWO-SIDED, honestly carried): candidate kernel device time T_triton replaces the 13.25 us/call Ixmma floor; wall wins iff python_savings - launch_overhead_delta - (T_triton - 13.25) > 7.556 us; the epoch-1 naive analog (grid=(8,), wall 0.2377 ms vs 0.1500 base) suggests scalar-path device can reach ~85 us — THIS design differs structurally (24-program tensor-core-dot tiled kernel vs 8-program naive), and the proven-envelope dot consumption is exactly the mechanism expected to keep T_triton in the 15-40 us band; if instead T_triton lands >= ~60 us the regression dominates and the round fails honestly",
    "unrounded interleaved paired median wall time improves by at least 5% versus baseline_adapter.py under fingerprint 6dc07009... (same-session paired basis authoritative; r001 established session drift +2.6% making cross-session anchors context-only)"
  ],
  "primary_metric": { "name": "wall_time", "expected_improvement_pct": 8.0 },
  "causal_graph": {
    "nodes": [
      "cn.dispatch-collapse",
      "cn.aten-dispatch-time",
      "cn.device-time-delta",
      "cn.wall-time"
    ],
    "edges": [
      ["cn.dispatch-collapse", "cn.aten-dispatch-time"],
      ["cn.dispatch-collapse", "cn.device-time-delta"],
      ["cn.aten-dispatch-time", "cn.wall-time"],
      ["cn.device-time-delta", "cn.wall-time"],
      ["cn.dispatch-collapse", "cn.wall-time"]
    ]
  },
  "mechanism_observables": [
    { "name": "wall_time_unrounded_paired_median_ms", "expectation": "at least 5% below the same-session accepted reference median across interleaved pairs at warmup 50 / repeat 100" },
    { "name": "aten_cpu_ops_per_call", "expectation": "collapse from ~34/call to <=3/call in the candidate forward scope census" },
    { "name": "launch_and_submission_count_per_call", "expectation": "exactly 1.00 kernel launch (cudaLaunchKernel-class) per call, ZERO memcpys, ZERO graph submissions, ZERO model-code synchronizations — the r001 regression structure (5 submissions + per-call sync) must be absent" },
    { "name": "device_us_per_call", "expectation": "TWO-SIDED with pre-declared readings: (a) T_triton <= ~40 us with wall >= +5% => mechanism confirmed on both edges; (b) T_triton >= ~60 us (epoch-1 naive territory) with wall flat/negative => device regression ate the dispatch prize => honest no-improvement with named root cause; (c) T_triton in between => read the wall observable alone, attribution per census" },
    { "name": "run_out_bitwise_equals_forward", "expectation": "bitwise equality over poisoned caller buffers x2 with data_ptr preserved; forward outputs bitwise-stable across repeated identical-input calls (deterministic kernel)" },
    { "name": "proven_envelope_binding_audit", "expectation": "every tl.dot call site compiles to (32,32)@(32,32) fp32 with widened operands; num_warps=1; count of torch.compile/TORCHINDUCTOR/'reduce-overhead' strings = 0" }
  ],
  "guardrails": [
    "correctness:pass under the unchanged comparator",
    "outputs remain single fp16 [83,512] tensors",
    "stateless module: no instance attributes, no caches, no workspace; caller device and current stream preserved; no added synchronization",
    "run_out bitwise==forward for identical inputs; caller buffers never aliased",
    "cold first-call JIT compile stays outside timed medians (harness warmup 50 absorbs it)",
    "AST-loader-safe module composition (safe-literal module constants; retained defs)"
  ],
  "profiling_level": "targeted"
}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`: no cataloged failure matches. The MLU winner-tree/sort-network/gather/cumsum entries concern expert-SELECTION workloads with tie surfaces; this operator is selection-free dense compute and none of those primitives appear. The S60 scalar-FMA degradation lesson (tl.sum manual dot losing an order of magnitude to tensor cores) is specifically ADDRESSED BY DESIGN: all QK^T and PV products use tl.dot at the proven fp32 envelope — the reduction.sum / tl.sum fallback is FORBIDDEN here anyway (waiver NOT granted), so the design cannot regress into that pattern silently.
- r001 root-cause separation (why THIS is not round-001 again): r001's replay route LOST because it ADDED 5 GPU submissions + 4 memcpys + per-call cudaDeviceSynchronize/cudaDriverGetVersion to remove only dispatch-count — net +2.6 us. THIS design removes dispatch count while keeping the submission structure IDENTICAL to base (1 launch) and adding ZERO copies/syncs: the regression source is structurally absent, and the r001 arithmetic that quantified per-aten-op cost (~0.6-1.0 us from 34→6 at −28 ops) is the very prize this design collects in full (−32 ops) rather than partially.
- Device-regression risk is the round's real danger and is carried EXPLICITLY: the epoch-1 archive naive (grid=(8,), scalar-era) measured ~0.61x wall — if that was device-dominated (~85 us kernel), a tiled tensor-core-dot rewrite must land 2-5x faster to clear the bar. The two-sided device observable (c) names the inconclusive middle band in advance so the verdict cannot be gamed in either direction.
- Triton-launch-overhead risk named: this build's direct Triton launch path has NO recorded fast-launcher evidence (matrix §四: BI150 'direct launch' unproven-fast); if python-side launch cost exceeds the aten sdpa C++ dispatch it replaces, the prize shrinks by that delta — visible only through the wall observable; declared expected 8.0% already prices in a mid-size launch penalty.
- Numerics: fp32 online softmax with running-max rescaling is the standard flash pattern; -inf masked keys yield exp()=0 EXACTLY (no NaN at finite inputs; equal_nan comparator moot); fp16->fp32 widening is lossless; the 1e-2 fp16-out tolerance dominates residual accumulation-order deltas; tie rules do not exist (no index-carrying reductions anywhere).
- Capability discipline: the fp16-operand dot path (P1/P2) and num_warps>1 (P3) remain UNRUN and are deliberately NOT consumed — the proven fp32 envelope plus widening casts makes the candidate Phase-legal TODAY without probe dependency; P1-P4 stay preflighted for a possible future tuning round only if THIS round wins and post-r002 device share justifies it.
- Capture hazards n/a (no graphs); JIT-compile hazard: first target call compiles the kernel (~100 ms-class) — absorbed by harness warmup 50, outside timed medians, same class as sibling r001-r004 cold-compile precedent.
- DANGER notes for Coder binding statement: torch.compile / TORCHINDUCTOR / reduce-overhead / tf32 strings count 0 REQUIRED; tl.dot call-site audit vs (32,32)@ (32,32) fp32 REQUIRED (any fp16-operand dot or other shape FAILS); num_warps value other than 1 FAILS; any workspace/graph/cache/state attribute on the module FAILS this round.

## Rationale and Evidence

Canonical anchors: last_accepted = `baseline_adapter.py` @`b8ec3458…`, paired-median basis 0.151107 ms (`rounds/report_000.md` @`a90df70d…`); r001 (`rounds/report_001.md` @`8c93d473…`, verdict @`c804df77…`) terminated no-improvement (−1.6873% same-session paired) with mechanism FULLY engaged — the campaign's first falsification, and the source of this round's two load-bearing numbers.

Why THIS family now (evidence-bounded, in order of weight):

1. r001 CENSUS ARITHMETIC (the dispatch-price measurement): candidate aten ops fell 34→6/call (−28) while boundary additions (4 memcpy dispatches + 4 extra submissions + per-call sync + driver-version call) netted −1.6873% (−2.56 us on 151 us scale-class). Solving the identity: removed python dispatch ≈ 17–30 us ⇒ ~0.6–1.0 us per collapsed aten op. The base still carries ~32 collapsible ops (sdpa stack, 20 view routings, 7 allocations) ⇒ remaining python prize ≈ **20–32 us/call = 2.6–4.2x the 7.556 us absolute bar**, and report_001's own bound ("< ~30–60 us") brackets the same number from above. This is the ONLY measured, un-owned host meat left; harness-fixed seed/sync and per-submission costs are not addressable by any candidate-side mechanism (r001 proved the submission route INCREASES them).
2. STRUCTURAL ABSENCE OF THE r001 REGRESSION SOURCE: the wrapper lost because submissions 1→5 and sync appeared. This design launches EXACTLY ONE kernel on the caller's stream with zero copies and zero syncs — the loss mechanism is structurally impossible, not merely tuned around.
3. DEVICE DELTA IS THE ONLY NEW TERM AND IT IS TWO-SIDED BY CONSTRUCTION: the proven-envelope fp32 dot consumption (widened operands) is the profile-legal mechanism class that separates this kernel from the epoch-1 scalar-era naive (0.61x); T_triton is unknown until measured (est. 15–40 us band; failure band pre-declared at >= ~60 us), and the Evaluation Contract carries cn.device-time-delta as a first-class wall edge so a device-eaten outcome is attributed, not hidden.
4. FAMILY FIELD ELIMINATION: H-A falsified by report_001 (boundary economics on a 1-launch base); H-B pre-falsified by the same arithmetic (compile guards ADD host work over 1-launch base; its cudagraph tier hits identical boundary economics/mutation-skip class); H-C died on report_000's census BEFORE coding (the reshape-back is views — zero device copies exist to kill; r002's direct-layout store removes the view OPS themselves instead, which H-C never could); H-E (abort) is rejected because a falsifiable ≥5% hypothesis EXISTS with measured backing (item 1) — the designer contract reserves abort for when none can be justified. γ-field: a view-surgery-only host family cannot touch the sdpa-internal dispatch stack and dies on the same census.
5. STREAK/TERMINAL CALCULUS, STATED HONESTLY: miss streak is 1/3 and a second consecutive no-improvement auto-terminates the campaign. Proceed-vs-abort is therefore: PROCEED = win probability p (assessed moderate: python prize alone 2.6–4.2x bar; device risk the swing term) of banking the only remaining mechanism, else terminal-now with census-grade root cause; ABORT = terminal-deferred (failed_attempt_streak 1, campaign nominally alive) with NOTHING left to dispatch next round (all families exhausted by evidence) — i.e., the same terminal outcome one round later, minus the win probability, plus a wasted round budget slot. Proceeding strictly dominates. If r002 fails, the honest close-out carries: dispatch-price quantification, T_triton measurement, and device-floor documentation — decision-space genuinely exhausted at that point.
6. One-attributable-change compliance: change_scope `mixed` is justified under the inseparability clause — the kernel and the host collapse are ONE mechanism (single-kernel rewrite of the pipeline); their effects are separately observable (device time per call vs aten census), satisfying the mixed-change observability requirement; no other mechanism is touched.

Artifacts consulted: `rounds/report_001.md` @`8c93d473…`, `rounds/decision_001.md` @`fa11b115…`, `rounds/sketch_001.json` @`199275b8…`, `rounds/report_000.md` @`a90df70d…`, `baseline_adapter.py` @`b8ec3458…`, `../base.py` @`dd1359ad…`, `project.md`, `profile_snapshot/triton_cuda.yaml` @`dc8fa4c0…`, `profile_snapshot/capability_claim.json` @`07aa5d48…`, `team-state.md`, `references/invariants.md`, `references/anti-patterns.md`, `references/bottleneck-judgment.md`, `references/decision-template.md`, `auto_bench.py` @`71fb3ad0…`, `kernels/track1-triton/summary_all_backends.md` §四 @`f899c82a…`, sibling `groupedtopk/bi150-round2/{final_summary.md, rounds/report_004.md}` (noncanon priors), archived epoch-1 `bi150/final_summary.md` + `triton_flexattention_001.py` (0.61x naive analog, noncanon device-regression bound).

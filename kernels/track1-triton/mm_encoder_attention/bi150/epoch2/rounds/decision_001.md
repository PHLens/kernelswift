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
  "change_family": "triton-attention-dispatch-collapse",
  "sketch_ref": "rounds/sketch_001.json",
  "sketch_sha256": "a1c27dbae53b1c7a74681510a0d09ced6be58ed8501f86976ce55af1b4772363",
  "implementation_profile_snapshot_ref": "profile_snapshot/triton_cuda.yaml",
  "implementation_profile_snapshot_sha256": "dc8fa4c0c73caecc78c6a886e5ffb18cb97371f3add7cb382dbe30887743b7ae",
  "project_capability_claim_ref": "profile_snapshot/capability_claim.json",
  "project_capability_claim_sha256": "aeba3a87f0494c2bb349b92fe668370c70d77fdebea29eac52824c3556b0d4d8"
}
```

## Optimization Intent

```json
{
  "bottleneck_class": "host-bound",
  "intervention": "replace the entire 33-aten-op base path (SDPA dispatch stack + 8 transpose + 8 as_strided + 7 empty + 4 view + empty_like/empty_strided/reshape) with ONE direct-launched Triton full-attention kernel plus a minimal two-op forward (one torch.empty + one kernel launch): grid = (B*H) x ceil(S/BM) = 16x3 = 48 programs, num_warps=1, BM=BN=32, head-dim D=64 processed as two 32-chunks; fp16 inputs loaded with DIRECT STRIDED ADDRESSING of the [B,S,H*D] views (zero .contiguous() copies — the epoch-1 structural mistake) and WIDENED to fp32 BEFORE tl.dot so every dot call site stays strictly inside the proven (32,32)@(32,32) fp32/fp32->fp32 envelope; online (running-max) softmax over the 3 sequential key-tiles in fp32 with -inf masking ONLY for the S=83 tile-padding columns (bidirectional: no causal skip — every score tile computed); PV accumulation as the same proven-shape dots; results STORED DIRECTLY into the final [2,83,512] fp16 token-major layout so no output view/copy ops exist; run_out(query,key,value,out) writes the caller-provided buffer through the same kernel (bitwise-identical results, zero extra ops); the kernel is the only device work, the launch is the only submission (1 cuLaunchKernel-class), and the module is stateless",
  "allowed_changes": [
    "kernel: one stateless @triton.jit full-attention kernel replacing the vendor SDPA call",
    "forward host collapse: torch.empty([2,83,512], fp16) + single kernel launch (2 python ops)",
    "run_out direct write: same kernel, caller buffer, zero allocation",
    "layout: direct strided addressing of input views + direct [B,S,H*D] output stores (drops all view/transpose/copy host ops)",
    "dtype: fp16->fp32 widening before each tl.dot; fp32 online softmax state; fp16 store",
    "strictly NOT: no torch.compile / no caching launchers / no runtime codegen strings of any kind",
    "strictly NOT: no CUDA graphs, no capture, no replay, no graph pool machinery (graph composition is a SEPARATE future round gated on this round's measurements)",
    "strictly NOT: no manual matmul via tl.sum/reduction loops (reduction.sum substitution BLOCKED — waiver NOT granted)",
    "strictly NOT: no tl.dot call site outside the (32,32) fp32 proven envelope (no fp16-operand dots, no larger tiles, no num_warps != 1) — probe-gated, NOT candidate-ready",
    "strictly NOT: no .contiguous(), no output copies, no extra kernels, no extra host ops beyond empty + launch",
    "strictly NOT: no module state, no caching of tensors or plans, no precision-mode toggles"
  ],
  "invariants": [
    "correctness:pass under the unchanged comparator (allclose atol=1e-2 rtol=1e-2 equal_nan=True, seed 42)",
    "single fp16 output [2,83,512]; bidirectional full-attention semantics with scale=0.125",
    "public API: ModelNew(num_heads, head_size, num_kv_heads=8); forward(q,k,v); run_out(q,k,v,out) 4-arg per project.md public_contract",
    "run_out bitwise==forward on identical inputs (poisoned caller buffers x2, data_ptr preserved)",
    "stateless: zero call-time instance state, zero caches, zero workspace",
    "capability legality: every tl.dot at (32,32)@(32,32) fp32 (operands widened); num_warps=1; num_stages unset",
    "AST-loader-safe module (safe-literal module constants; get_inputs/get_init_inputs retained)",
    "no torch.compile / no graph capture / no DANGER-token constructs anywhere in the candidate"
  ],
  "expected_wall_improvement_pct": 0.0
}
```

## Unified Sketch

```json
{
  "artifact": "rounds/sketch_001.json",
  "sha256": "a1c27dbae53b1c7a74681510a0d09ced6be58ed8501f86976ce55af1b4772363",
  "rendering": "one direct-launched Triton full-attention kernel as the complete computation boundary: 48 programs (batch x head x q-tile parallel, 3 sequential key-tiles per program for the online softmax dependency), fp16 global tensors [B,S,H,D] read with direct strided addressing and widened into fp32 register tiles (D=64 as two 32-chunks), QK^T via proven (32,32)@(32,32) fp32 dots with -inf masking only on S=83 tile-padding columns (bidirectional — no causal skip), running-max online softmax state, PV via the same proven-shape dots, and a single fp16 store directly into the final [B,S,HD] output tensor (forward's fresh buffer or run_out's caller buffer)"
}
```

## Host Plan

```json
{
  "applicability": "required",
  "affected_scope": [
    "ModelNew.forward host path: torch.empty([2,83,512], fp16) + one Triton kernel launch — exactly 2 python-visible ops replacing 33 aten ops",
    "ModelNew.run_out host path: one Triton kernel launch into the caller buffer — zero allocations, zero other ops",
    "kernel launch configuration: grid (48,) = (B*H=16) x (ceil(S=83/BM=32)=3), num_warps=1, constexpr B=2/S=83/H=8/D=64/strides frozen at module-definition time as literals (AST-loader safe)",
    "per-call output allocation: forward allocates one fresh [2,83,512] fp16 tensor; run_out allocates nothing"
  ],
  "state_owner": "NOBODY",
  "lifetime": "stateless module: the @triton.jit function object plus the framework-owned JIT specialization cache live for the module lifetime; per-call buffers live for the call; no cross-call state is created or read",
  "allocation_reuse": "NONE",
  "cache_key": [
    "not-applicable: stateless module; Triton JIT specialization is keyed by the constexpr tuple (fixed literals) and the framework cache lifecycle, not by runtime cache_keys; no cache is owned by the module"
  ],
  "invalidation": "not-applicable (stateless); the kernel recompiles only if the constexpr tuple changed, which it cannot (literals frozen)",
  "concurrency": "stateless and side-effect-free host path; safe under concurrent callers with the same device/stream semantics as the base module",
  "device_stream_behavior": "all work launches onto the caller's current device and current stream via the standard Triton launcher; no side streams, no events, no capture, no synchronize, no cudaMemcpy",
  "unchanged_behavior": [
    "forward returns a fresh [2,83,512] fp16 tensor with identical bidirectional-attention semantics to base (softmax(QK^T*0.125)*V over all 83 keys per (batch,head) pair, no mask, no causal structure)",
    "run_out(query,key,value,out) fills the caller-provided [2,83,512] fp16 buffer bitwise-identically to forward's result and returns None; 4-arg signature per project.md public_contract (kernel-mode arity deviation carries forward)",
    "all host operations launch onto the caller's current device and current stream; no side streams, no events, no capture, no synchronization",
    "module remains JIT-warm after harness warmup 50 (first-call compile absorbed outside timed medians, sibling-campaign precedent)",
    "GQA path untouched: num_kv_heads == num_heads == 8 for this instantiation",
    "output exact fp16 [B,S,H*D] layout preserved — direct store, no view/reshape/transpose/copy chain",
    "the single Triton kernel is the ONLY device work and the ONLY submission; the candidate deliberately contains zero graph machinery"
  ]
}
```

## Evaluation Contract

```json
{
  "hypothesis_id": "H-001",
  "intervention": "replace the entire 33-aten-op base path (SDPA dispatch stack + 8 transpose + 8 as_strided + 7 empty + 4 view + empty_like/empty_strided/reshape) with ONE direct-launched Triton full-attention kernel plus a minimal two-op forward (one torch.empty + one kernel launch): grid = (B*H) x ceil(S/BM) = 16x3 = 48 programs, num_warps=1, BM=BN=32, head-dim D=64 processed as two 32-chunks; fp16 inputs loaded with DIRECT STRIDED ADDRESSING of the [B,S,H*D] views (zero .contiguous() copies — the epoch-1 structural mistake) and WIDENED to fp32 BEFORE tl.dot so every dot call site stays strictly inside the proven (32,32)@(32,32) fp32/fp32->fp32 envelope; online (running-max) softmax over the 3 sequential key-tiles in fp32 with -inf masking ONLY for the S=83 tile-padding columns (bidirectional: no causal skip — every score tile computed); PV accumulation as the same proven-shape dots; results STORED DIRECTLY into the final [2,83,512] fp16 token-major layout so no output view/copy ops exist; run_out(query,key,value,out) writes the caller-provided buffer through the same kernel (bitwise-identical results, zero extra ops); the kernel is the only device work, the launch is the only submission (1 cuLaunchKernel-class), and the module is stateless",
  "expected_causal_chain": [
    "host side: 33 aten cpu_ops/call collapse to <=3 (one torch.empty + launch-visible ops), removing the only compressible host block under the ~133.6 us/call host floor (device_ratio 0.110, report_000)",
    "submission side: base pays exactly 1 cudaLaunchKernel; candidate pays exactly 1 cuLaunchKernel-class launch — submission count UNCHANGED; the sibling replay-family regression structure (extra submissions, boundary memcpys, per-call sync) is structurally ABSENT from this design",
    "MEASURED-PRIOR SWING TERM, this round's falsification target: sibling flexattention r002 (same rig, same CoreX 4.4.0 build, same harness class, bsz=1 causal) measured the Triton python launcher path at +86-89 us/call NET host over the vendor host path it replaces; if that tax transfers to bsz=2, wall lands ~0.235-0.29 ms (a large honest no-improvement) with census-grade attribution; only a material tax collapse opens the win branch — the round is deliberately two-sided",
    "device side: D_cand (proven-envelope Triton kernel at ~4x sibling-kernel work: 16 full-attention pairs vs 8 causal pairs) replaces the 16.537 us/call vendor Ixmma floor; expected band 20-66 us; D_cand is measured directly in the candidate scope and doubles as the F2 composition gate input",
    "regardless of wall outcome the round banks the campaign's PRIMARY contractual product per project.md DELIVERABLE RULE: a correctness-PASS Triton submission (forward + run_out surfaces); adoption is NOT expected (expected_wall_improvement_pct 0.0 declared honestly) — the primary_metric stays the 5.0% adoption bar and the round fails it honestly if the launcher tax transfers"
  ],
  "primary_metric": {
    "name": "wall_time",
    "expected_improvement_pct": 5.0
  },
  "causal_graph": {
    "nodes": [
      "cn.dispatch-collapse",
      "cn.aten-dispatch-time",
      "cn.triton-launcher-tax",
      "cn.device-time-delta",
      "cn.wall-time"
    ],
    "edges": [
      ["cn.dispatch-collapse", "cn.aten-dispatch-time"],
      ["cn.dispatch-collapse", "cn.triton-launcher-tax"],
      ["cn.dispatch-collapse", "cn.device-time-delta"],
      ["cn.aten-dispatch-time", "cn.wall-time"],
      ["cn.triton-launcher-tax", "cn.wall-time"],
      ["cn.device-time-delta", "cn.wall-time"],
      ["cn.dispatch-collapse", "cn.wall-time"]
    ]
  },
  "mechanism_observables": [
    {
      "name": "wall_time_unrounded_paired_median_ms",
      "expectation": "at least 5% below the same-session accepted reference median (0.150149 ms class) across interleaved pairs at warmup 50 / repeat 100; honest expectation per the launcher-tax prior is a large NEGATIVE delta — the observable is declared two-sided, the win branch requires the launcher tax NOT to transfer"
    },
    {
      "name": "aten_cpu_ops_per_call",
      "expectation": "collapse from 33/call (report_000 census) to <=3/call in the candidate forward scope"
    },
    {
      "name": "launch_and_submission_count_per_call",
      "expectation": "exactly 1.00 kernel launch (cuLaunchKernel-class) per call, ZERO cudaMemcpyApi calls, ZERO graph launches, ZERO model-code synchronizations — the direct-family structural guarantee"
    },
    {
      "name": "device_us_per_call",
      "expectation": "TWO-SIDED with pre-declared readings: (a) D_cand <= ~40 us AND wall >= +5% => win branch (launcher tax collapsed — transfer model falsified, re-rank families); (b) D_cand in the 20-66 us band with wall decisively negative and a ~85-90 us net host delta => honest no-improvement #1 with the bsz=2 launcher tax canonically measured; (c) D_cand >= ~66 us => compute-bound kernel regression dominates the device term — all three bands attribute, none hide"
    },
    {
      "name": "triton_launcher_tax_per_call",
      "expectation": "the canonical bsz=2 measurement of THIS campaign: candidate host path vs base host path net delta per call (sibling prior +86-89 us at bsz=1, wall 0.236 ms); derived from paired wall minus device delta and corroborated by the aten census; this number is the F2 arithmetic gate"
    },
    {
      "name": "run_out_bitwise_equals_forward",
      "expectation": "bitwise equality over poisoned caller buffers x2 with data_ptr preserved; forward outputs bitwise-stable across repeated identical-input calls (deterministic kernel, no atomics)"
    },
    {
      "name": "proven_envelope_binding_audit",
      "expectation": "every tl.dot call site uses (32,32)@(32,32) fp32 operands (widened); num_warps=1; count of torch.compile/TORCHINDUCTOR/reduce-overhead strings = 0; zero .contiguous() calls in the forward/run_out host paths"
    }
  ],
  "guardrails": [
    "correctness:pass under the unchanged comparator (allclose atol=1e-2 rtol=1e-2 equal_nan=True, seed 42)",
    "outputs remain single fp16 [2,83,512] tensors with bidirectional full-attention semantics",
    "stateless module: no instance attributes written at call time, no caches, no workspace; caller device and current stream preserved; no added synchronization",
    "run_out bitwise==forward for identical inputs; caller buffers never aliased; returns None",
    "cold first-call JIT compile stays outside timed medians (harness warmup 50 absorbs it)",
    "AST-loader-safe module composition (safe-literal module constants; retained defs)",
    "capability legality: every tl.dot at (32,32)@(32,32) fp32 with widened operands; fp16 loads/stores only at the global memory boundary; zero DANGER tokens (compile/capture) in candidate source"
  ],
  "profiling_level": "targeted"
}
```

## Pitfalls and Anti-pattern Consultation

- anti-patterns.md consulted: no cataloged failure matches (the MLU selection-network entries are selection workloads; this operator is dense, selection-free, tie-free attention). The scalar-FMA lesson class (manual tl.sum matmul losing to vendor tensor cores) is excluded BY DESIGN: all QK^T/PV products go through proven-envelope tl.dot, and the reduction.sum substitution path is BLOCKED (waiver NOT granted) — the candidate cannot silently regress into that pattern.
- epoch-1 naive structural mistakes named and excluded: (i) the 3x .contiguous() + reshape-copy host path (added device copies + aten ops) — replaced by direct strided addressing + direct-layout stores; (ii) BLOCK_S=128 padding (~37% wasted lanes, outside the frozen envelope) — replaced by 32-tiles at the proven (32,32) dot shape.
- sibling r002's unmodeled launcher tax is now a MEASURED prior and this round's falsification target — the expectation is declared honestly (0.0, no adoption expected) instead of priced optimistically; sibling r001's replay-regression structure (extra submissions, boundary memcpys, per-call syncs) is structurally absent: one launch, zero memcpys, zero syncs, zero graph machinery.
- numerics: fp32 online softmax with running max; -inf padded keys -> exp=0 exact; fp16->fp32 widening lossless; 1e-2 fp16-output tolerance dominates accumulation-order deltas; tie-free (no index-carrying reductions anywhere); scale=0.125 exact power of two.
- cold JIT compile is absorbed by harness warmup 50 (sibling precedent); no runtime codegen strings anywhere in the candidate.
- DANGER-token binding notes for Coder: zero compile/capture strings; tl.dot site audit at (32,32) fp32 REQUIRED; num_warps=1; zero .contiguous()/copy_ in host paths; stateless audit; run_out 4-arg signature.

## Rationale and Evidence

**Reference and canonical anchors.** Accepted pair: baseline_adapter.py @ c3980a2c4ae79171c73c7cf85c29c62f6ac98b9e9fb643c249fc9a6e0a921c9f (1832 bytes) and rounds/report_000.md @ 20b21646d9c3ba3abe086d8133799d23a39981dcb4e1cb547e1a3f65b0bf7ffc. Canonical baseline (fingerprint 0c4c7d66…, forward-mode dual-scope, pw=20/pi=100): wall v0 median 0.150149 ms (reference scope) / 0.150147 ms (candidate scope); device 16.537 us/call (reference) / 17.559 us/call (candidate) — ONE fused FlashAttnFwdF16Ixmma<128,128,16,64,64,CausalM_t=0> kernel, bidirectional template arg, bsz handled in runtime params, single launch covers both batches; device_ratio 0.110 / 0.117 — host-bound, host floor ~133.6 us/call, 89% of wall; host census 33 aten cpu_ops/call (8 transpose, 8 as_strided, 7 empty, 4 view, 1 clone, sdpa chain, empty_like, empty_strided, reshape), 1.00 cudaLaunchKernel/call, zero memcpys, zero syncs. The kernel-mode profile attempt failed on run_out arity (make_profile_call passes run_out(inputs[-1], *outputs) = 2 args vs the mandated 4-arg public surface) — exit 1 with stable error, forward-mode dual-scope fallback used; the deviation carries forward and 4-arg run_out remains REQUIRED from candidates. The Phase-0 transfer model SURVIVED verification with minor calibration: this session's vendor device floor is 16.5-17.6 us (epoch-1 noncanon prior said 14.95, +11-18%), wall sits at the low end of the prior band — both below the re-model trigger, so the Phase-0 ranking stands unchanged. Verifier implication line (report_000): a +5% adoption needs a >=43-45% device cut at ZERO added host cost, or a host-floor intervention (~133 us/call).

**Break-even arithmetic with canonical numbers.** Adoption bar 5% = 7.507 us => winning wall <= 142.64 us. For the direct family: wall_cand = 150.149 + T_launcher + (D_cand - 16.537), so the win condition reduces to T_launcher + D_cand <= 9.03 us — impossible while D_cand >= the ~16.5 us vendor-floor class, even at T_launcher = 0. With the sibling prior T_launcher ~ +86-89 us the expected wall is ~0.235-0.29 ms (an honest regression). For the graph family (future round): net = -T_launcher + 69 (R-term) + ~13 (boundary host) + (D_cand + 3.7 copy-out - 16.537); at T_launcher = 87 this is D_cand - 17.8, so parity needs D_cand ~= 18 us and the +5% win needs D_cand <= ~10 us (a 2.4x beat of the vendor kernel at 4x sibling work) — the F2 gate arithmetic below. The Verifier's own implication line is consistent with this: the intervention attacks the host floor with ONE launch, but the Triton python launcher tax is the measured prior against it.

**Why PROCEED with expected_wall_improvement_pct 0.0 (declared honestly).** (1) DELIVERABLE RULE (binding, project.md): the campaign's PRIMARY contractual product is the best correctness-PASS Triton submission even at 0.5x-0.6x — an abort produces NO Triton deliverable at all, stranding the corrected precedent from the sibling campaign; the abort form is reserved for cases where no deliverable-bearing work remains, which is not the case here. (2) Information: F1 canonizes the two numbers every remaining family needs — the bsz=2 launcher tax (never measured; the sibling number is a cross-operator prior) and D_cand (the Triton device floor at 16 full-attention pairs) — at the cost of exactly one round. (3) Falsifiability: the transfer model's launcher-tax line is genuinely two-sided — a material collapse (<= ~22 us combined with the device delta) re-opens the win branch and re-ranks the backlog; the observable bands are pre-declared so either outcome attributes cleanly.

**F2 gating decision (explicit, per dispatch).** STRICT one-attributable-change discipline for the CANDIDATE: zero graph machinery in round 001 — graph composition is a separate future round. The F2 prerequisites (D_cand and T_launcher) are measured as ORDINARY targeted evidence of THIS round (device_us_per_call and triton_launcher_tax_per_call are named mechanism observables; dual-scope profiling plus host census are already mandated) — pre-authorization of the MEASUREMENT, not of candidate composition, so attribution stays clean at zero extra mechanism cost. F2 proceeds to a decision only if BOTH hold: measured T_launcher >= ~50 us/call (composition prize exists) AND D_cand <= ~35 us (device penalty fits the ~+3 us net host prize band); F2 is parity-class only if D_cand <= ~25 us. If the gate fails, F2 is arithmetically dead and the backlog moves to F3 probes (fp16 tl.dot, num_warps>1) or deliverable hardening.

**Family elimination.** F2-first rejected: graph composition needs D_cand and T_launcher measured anyway, and a graph wrapper over an unmeasured kernel is strictly less informative than measuring the kernel first (sibling r003 was built on r002's census for exactly this reason). F3-first rejected: capability probes carry no standalone wall claim; they tune a kernel that does not exist yet in this campaign. F4 dead-ends stay closed: aten-captured graph over a 1-launch base (sibling r001, -1.69% — nothing to compress), math-backend decomposition (anti-pattern: fragments the fused kernel), torch.compile/caching launchers (out of contract, DANGER tokens), harness manipulation (measurement fingerprint — forbidden).

**Streak calculus (honest).** The most likely outcome is no-improvement #1 (streak 1/3) — that outcome banks the deliverable candidate and both canonical physics numbers and leaves two no-improvement slots, exactly the sibling trajectory (which terminated at 3/3 with a 1.00x-parity deliverable). Aborting instead would spend a failed_attempt_streak slot, produce no deliverable, and leave the identical decision to face next round with zero new information.

**Mixed-scope justification.** The kernel rewrite and the host collapse are ONE inseparable mechanism (a single-kernel rewrite of the whole pipeline); they are separately observable (device_us_per_call vs aten_cpu_ops_per_call vs wall_time), satisfying the mixed-change observability requirement.

**Artifacts consulted.** project.md (identity, DELIVERABLE RULE, public_contract, runtime fingerprint 0c4c7d66…); rounds/report_000.md (canonical baseline + verifier implication line); baseline_adapter.py (via report evidence); ../../base.py (immutable reference semantics); profile_snapshot/triton_cuda.yaml @dc8fa4c0… and capability_claim.json @aeba3a87… (frozen envelope; reduction.sum waiver NOT granted); epoch-1 noncanon archive ../final_summary.md + ../triton_mm_encoder_attention_001.py + ../rounds/report_000.md + ../rounds/coder_result_deliverable.md (0.547x naive prior; .contiguous() and BLOCK_S=128 mistakes excluded by design); sibling campaign flexattention/bi150/epoch2 final_summary.md + rounds/report_000-003.md + rounds/decision_002.md + rounds/sketch_002.json (launcher tax +86-89 us, R-term 69 us, proven-envelope kernel architecture); skills references invariants.md, anti-patterns.md, bottleneck-judgment.md, decision-template.md; auto_bench.py (harness/AST-loader/arity behavior); state/designer_context.md (Phase-0 model this decision executes).

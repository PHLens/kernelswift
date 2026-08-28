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
  "backend": "gcu",
  "target_profile": "triton_gcu",
  "runtime_fingerprint_ref": "project.md#runtime-fingerprint",
  "change_scope": "mixed",
  "change_family": "triton-attention-dispatch-collapse",
  "sketch_ref": "rounds/sketch_001.json",
  "sketch_sha256": "ef71920a8a856c633bf8ef5fcebe733bcda6f0fd026210691b1cc8e94aad8f70",
  "implementation_profile_snapshot_ref": "profile_snapshot/triton_gcu.yaml",
  "implementation_profile_snapshot_sha256": "8dfabd0af59b8f6640b47179fee19bca2f5fe35b18535a3db24f60c842e42b70",
  "project_capability_claim_ref": "profile_snapshot/capability_claim.json",
  "project_capability_claim_sha256": "a175f2727b9198a92da978aca9e8f87834a74884372746699412931890d9748e"
}
```

## Optimization Intent

```json
{
  "bottleneck_class": "mixed",
  "intervention": "replace the entire base path (vendor SDPA -> _scaled_dot_product_flash_attention, 2 topsLaunchKernel launches, 28 aten cpu_ops: 8 transpose + 8 as_strided + 4 view + 3 empty + sdpa chain + empty_like + empty_strided + reshape per report_000 census) with ONE direct-launched Triton full-attention kernel: grid = B*H = 16 programs (one program per (batch,head) pair, S=83 padded to TP=128 power-of-2 for tl.arange and mult-of-16 for tl.dot), fp16 q/k/v widened to fp32 BEFORE tl.dot (same-dtype operands required on triton_gcu), QK^T via tl.dot(128x64, 64x128) fp32, mask tile-padding keys to -inf (bidirectional: NO causal skip), softmax via tl.max/tl.sum WITHOUT keepdim (broadcast via [:,None]), PV via tl.dot(128x128, 128x64) fp32, num_warps=2, store fp16 directly into the final [2,83,512] layout; forward = one torch.empty + one kernel launch (2 python ops), run_out writes the caller buffer through the same kernel (zero allocation)",
  "allowed_changes": [
    "kernel: one stateless @triton.jit full-attention kernel replacing the vendor SDPA call",
    "forward host collapse: torch.empty([2,83,512], fp16) + single kernel launch (2 python ops)",
    "run_out direct write: same kernel, caller buffer, zero allocation",
    "layout: direct strided addressing of the [B,S,HD] head-major input (head-stride arithmetic, zero .contiguous() host copies) and direct [B,S,HD] output store",
    "dtype: fp16->fp32 widening before each tl.dot (same-dtype constraint); fp32 softmax state; fp16 store",
    "strictly NOT: no torch.compile / no caching launchers / no runtime codegen strings of any kind",
    "strictly NOT: no CUDA/GCU graphs, no capture, no replay, no graph pool machinery (graph composition is a SEPARATE future round gated on this round's measurements)",
    "strictly NOT: no manual matmul via tl.sum/reduction loops (reduction.sum substitution BLOCKED — waiver NOT granted; tl.dot is constrained-not-unknown so no fallback applies)",
    "strictly NOT: no tl.dot call site outside the mult-of-16 envelope (M/N/K all mult-of-16; TP=128, D=64) with same-dtype operands (fp16 widened to fp32)",
    "strictly NOT: no tl.arange extent that is not a power of 2 (TP=128, never 96); no tl.max/tl.sum keepdim",
    "strictly NOT: no .contiguous(), no output copies, no extra kernels, no extra host ops beyond empty + launch",
    "strictly NOT: no module state, no caching of tensors or plans, no precision-mode toggles"
  ],
  "invariants": [
    "correctness:pass under the unchanged comparator (allclose atol=1e-2 rtol=1e-2 equal_nan=True, seed 42)",
    "single fp16 output [2,83,512]; bidirectional full-attention semantics with scale=0.125",
    "public API: ModelNew(num_heads, head_size, num_kv_heads=8); forward(q,k,v); run_out(q,k,v,out) 4-arg per project.md public_contract",
    "run_out bitwise==forward on identical inputs (poisoned caller buffers x2, data_ptr preserved)",
    "stateless: zero call-time instance state, zero caches, zero workspace",
    "capability legality: every tl.dot at mult-of-16 (TP=128, D=64) fp32 with widened operands; num_warps=2; tl.max/tl.sum without keepdim; tl.arange power-of-2 only",
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
  "sha256": "ef71920a8a856c633bf8ef5fcebe733bcda6f0fd026210691b1cc8e94aad8f70",
  "rendering": "one direct-launched Triton full-attention kernel as the complete computation boundary: 16 programs (B*H, one per (batch,head)), S=83 padded to TP=128 (power-of-2 for tl.arange, mult-of-16 for tl.dot), fp16 global tensors [B,S,HD] read with head-major strided addressing and widened into fp32 register tiles [TP,D], QK^T via tl.dot(128x64,64x128) fp32 with -inf masking on the S=83..127 tile-padding columns (bidirectional — no causal skip), softmax via tl.max/tl.sum (no keepdim) + [:,None] broadcast, PV via tl.dot(128x128,128x64) fp32, num_warps=2, and a single fp16 store directly into the final [B,S,HD] output (forward's fresh buffer or run_out's caller buffer)"
}
```

## Host Plan

```json
{
  "applicability": "required",
  "affected_scope": [
    "ModelNew.forward host path: torch.empty([2,83,512], fp16) + one Triton kernel launch — exactly 2 python-visible ops replacing 28 aten ops + 2 vendor launches",
    "ModelNew.run_out host path: one Triton kernel launch into the caller buffer — zero allocations, zero other ops",
    "kernel launch configuration: grid (16,) = B*H = 2*8, num_warps=2, constexpr B=2/S=83/H=8/D=64/TP=128/scale=0.125 frozen at module-definition time as literals (AST-loader safe)",
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
  "device_stream_behavior": "all work launches onto the caller's current device and current stream via the standard Triton launcher; no side streams, no events, no capture, no synchronize, no memcpy",
  "unchanged_behavior": [
    "forward returns a fresh [2,83,512] fp16 tensor with identical bidirectional-attention semantics to base (softmax(QK^T*0.125)*V over all 83 keys per (batch,head) pair, no mask, no causal structure)",
    "run_out(query,key,value,out) fills the caller-provided [2,83,512] fp16 buffer bitwise-identically to forward's result and returns None; 4-arg signature per project.md public_contract (kernel-mode arity deviation carries forward)",
    "all host operations launch onto the caller's current device and current stream; no side streams, no events, no capture, no synchronization",
    "module remains JIT-warm after harness warmup 50 (first-call compile absorbed outside timed medians)",
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
  "intervention": "replace the entire base path (2-launch vendor SDPA + 28 aten cpu_ops) with ONE direct-launched Triton full-attention kernel (grid=16, num_warps=2, S padded to TP=128): fp16->fp32 widening, QK^T and PV via mult-of-16 tl.dot, no-causal masked softmax without keepdim, direct fp16 store into [2,83,512]",
  "expected_causal_chain": [
    "host side: 28 aten cpu_ops/call + 2 vendor launches collapse to <=2 (one torch.empty + one Triton launch); the GCU launch tax is 17.4us/call (launcher probe) — 5x smaller than the BI150 sibling's 84.77us, so dispatch collapse is a materially narrower but REAL lever on S60",
    "submission side: base pays 2 topsLaunchKernel launches (21.99us/call launch-API); candidate pays exactly 1 direct Triton launch — submission count HALVED",
    "device side: D_cand replaces the base ~118us device floor (base wall 139.9us - 21.99us launch); probe measured D_cand at num_warps=2 = 148.6us wall (candidate -6.2% pre-adoption), nw sweep {1:175.9, 2:148.6, 4:159.0, 8:286.1} — nw=2 is optimal; the honest risk is the higher device floor from 16-program grid + TP=128 padding waste",
    "regardless of wall outcome the round banks the campaign's PRIMARY contractual product per project.md DELIVERABLE RULE: a correctness-PASS Triton submission (forward + run_out surfaces); adoption is NOT expected (expected_wall_improvement_pct 0.0 declared honestly) — the primary_metric stays the 5.0% adoption bar"
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
      "expectation": "honest two-sided: probe measured candidate -6.2% pre-adoption (148.6 vs 139.9us), so the round is likely no-improvement; the win branch requires the device floor to come down (grid-split parallelism, padding reduction) in a follow-up round"
    },
    {
      "name": "aten_cpu_ops_per_call",
      "expectation": "collapse from 28/call (report_000 census) to <=2/call in the candidate forward scope"
    },
    {
      "name": "runtime_launch_count_per_call",
      "expectation": "exactly 1.00 kernel launch per call (vs base 2.0), ZERO extra submissions, ZERO graph launches, ZERO model-code synchronizations — the direct-family structural guarantee"
    },
    {
      "name": "device_us_per_call",
      "expectation": "TWO-SIDED: (a) D_cand <= ~120us => candidate is device-comparable to base and round-2 grid-split parallelism is the win lever; (b) D_cand >= ~150us => 16-program grid is under-parallel and the follow-up must split the token dimension; GCU device-duration is unavailable (launch-only trace) so D_cand is inferred from wall - launch-API-time"
    },
    {
      "name": "triton_launcher_tax_per_call",
      "expectation": "the canonical S60 measurement: candidate host path vs base host path net delta per call (launcher probe prior 17.4us launch-only); this number is the round-2 graph-composition arithmetic gate"
    },
    {
      "name": "run_out_bitwise_equals_forward",
      "expectation": "bitwise equality over poisoned caller buffers x2 with data_ptr preserved; forward outputs bitwise-stable across repeated identical-input calls (deterministic kernel, no atomics)"
    },
    {
      "name": "mult_of_16_envelope_binding_audit",
      "expectation": "every tl.dot call site uses mult-of-16 operands (TP=128, D=64) with same-dtype (fp32) operands; num_warps=2; tl.max/tl.sum called WITHOUT keepdim; tl.arange extents are power-of-2; count of torch.compile/TORCHINDUCTOR/reduce-overhead strings = 0; zero .contiguous() calls in the forward/run_out host paths"
    }
  ],
  "guardrails": [
    "correctness:pass under the unchanged comparator (allclose atol=1e-2 rtol=1e-2 equal_nan=True, seed 42)",
    "outputs remain single fp16 [2,83,512] tensors with bidirectional full-attention semantics",
    "stateless module: no instance attributes written at call time, no caches, no workspace; caller device and current stream preserved; no added synchronization",
    "run_out bitwise==forward for identical inputs; caller buffers never aliased; returns None",
    "cold first-call JIT compile stays outside timed medians (harness warmup 50 absorbs it)",
    "AST-loader-safe module composition (safe-literal module constants; retained defs)",
    "capability legality: every tl.dot at mult-of-16 (TP=128, D=64) fp32 with widened operands; fp16 loads/stores only at the global memory boundary; tl.max/tl.sum without keepdim; tl.arange power-of-2 only; zero DANGER tokens (compile/capture) in candidate source"
  ],
  "profiling_level": "targeted"
}
```

## Pitfalls and Anti-pattern Consultation

- anti-patterns.md consulted: no cataloged failure matches (the MLU selection-network entries are selection workloads; this operator is dense, selection-free, tie-free attention). The scalar-FMA lesson class (manual tl.sum matmul losing to vendor tensor cores) is excluded BY DESIGN: all QK^T/PV products go through mult-of-16 tl.dot, and the reduction.sum substitution path is BLOCKED (waiver NOT granted).
- epoch-1 naive structural mistakes named and excluded: (i) the .contiguous() + reshape-copy host path (added device copies + aten ops) — replaced by head-major strided addressing + direct-layout stores; (ii) BLOCK_S=128 padding with naive tl.sum dot (the epoch-1 0.27x root cause) — replaced by mult-of-16 tl.dot at TP=128.
- S60-SPECIFIC capability constraints (probe-backed, MUST be honored by Coder — these differ from the CUDA/MLU siblings and are the round's key novelty):
  1. `tl.arange` requires POWER-OF-2 extent → TP=128, never 96.
  2. `tl.max` / `tl.sum` do NOT support `keepdim` on triton_gcu 3.6.0 → use `axis=1` then broadcast with `[:, None]`.
  3. `tl.dot` requires SAME-dtype operands → widen fp16 to fp32 (`.to(tl.float32)`) before every dot.
  4. `tl.dot` requires M/N/K multiples of 16 → TP=128 (mult of 16) and D=64 (mult of 16).
  5. num_warps=8 degrades severely (286us) → stay in {1,2,4}, prefer 2 (probe: nw2=148.6us optimal).
- sibling BI150's launcher-tax prior (+84.77us) does NOT transfer to S60: the GCU launcher tax is 17.4us/call (measured), so the direct family is materially closer to the win branch on S60 than on BI150.
- numerics: fp32 softmax with max subtraction; -inf padded keys -> exp=0 exact; fp16->fp32 widening lossless; 1e-2 fp16-output tolerance dominates accumulation-order deltas; tie-free (no index-carrying reductions); scale=0.125 exact power of two.
- cold JIT compile is absorbed by harness warmup 50; no runtime codegen strings anywhere in the candidate.
- DANGER-token binding notes for Coder: zero compile/capture strings; tl.dot site audit at mult-of-16 fp32 REQUIRED; num_warps=2; tl.arange power-of-2; tl.max/tl.sum no-keepdim; zero .contiguous()/copy_ in host paths; stateless audit; run_out 4-arg signature.

## Rationale and Evidence

**Reference and canonical anchors.** Accepted pair: baseline_adapter.py @ 1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e (2331 bytes) and rounds/report_000.md. Canonical baseline (fingerprint c335b39c…, forward-mode dual-scope, pw=20/pi=100): wall v0 median 0.227986 ms / v1 0.230700 ms (identity); census: base SDPA -> `_scaled_dot_product_flash_attention`, 2 `topsLaunchKernel`/call (21.99us/call launch-API), 28 aten cpu_ops/call (8 transpose + 8 as_strided + 4 view + 3 empty + sdpa chain + empty_like + empty_strided + reshape); GCU device-duration UNAVAILABLE (launch-only trace), so device time is inferred from wall - launch-API-time ≈ 139.9 - 21.99 ≈ 118us/call. This is materially DIFFERENT from the BI150 sibling (single Ixmma kernel, device 16.5us, host-bound at device_ratio 0.110): S60 base is a WEAKER 2-launch path with a much higher device floor, which widens the direct-family window.

**Capability preflight (probe-backed, live on S60).** Direct-Triton MHA (tl.dot, TP=128, nw2) is correctness-PASS (max_abs_diff 9.77e-4 < 1e-2) and measures 148.6us vs base 139.9us (-6.2% pre-adoption). num_warps sweep {1:175.9, 2:148.6, 4:159.0, 8:286.1} — nw=2 optimal. Launcher tax: single-launch 17.4us (launch-only), sync-adds ~83us; 5x smaller than BI150's 84.77us.

**Break-even arithmetic with S60 numbers.** Adoption bar 5% = 6.995us (of 139.9us) => winning wall <= ~132.9us. For the direct family: wall_cand = 139.9 + T_launcher + (D_cand - 118); with T_launcher ≈ 17.4 and probe D_cand ≈ 131us (148.6 - 17.4), wall_cand ≈ 170us — an honest no-improvement. The win branch requires D_cand to come DOWN below ~115us (i.e. beat base's device floor), which is the round-2 target: grid-split parallelism (split the token dimension to raise 16 -> 48+ programs) and padding reduction (TP=128 -> closer to 83 via 96 is impossible for tl.arange power-of-2, but BM/BN tiling can cut wasted lanes).

**Why PROCEED with expected_wall_improvement_pct 0.0 (declared honestly).** (1) DELIVERABLE RULE (binding, project.md): the campaign's PRIMARY contractual product is the best correctness-PASS Triton submission even below 1.0x — an abort produces NO Triton deliverable at all. (2) Information: round 1 canonizes the two numbers every remaining family needs — the S60 launcher tax (17.4us class, never canonically measured) and D_cand at 16-program grid (the Triton device floor to beat) — at the cost of exactly one round. (3) Falsifiability: the device-floor line is genuinely two-sided — if grid-split parallelism in round 2 brings D_cand below base's floor, the win branch opens cleanly; the observable bands are pre-declared so either outcome attributes.

**Family elimination.** Graph-composition-first rejected: it needs D_cand and T_launcher measured anyway, and a graph wrapper over an unmeasured kernel is strictly less informative than measuring the kernel first. Capability-probe-first rejected: probes carry no standalone wall claim; the S60-specific constraints (tl.arange power-of-2, no-keepdim, same-dtype dot, mult-of-16) are ALREADY probe-backed during Phase A onboarding + this round's preflight. torch.compile/caching launchers out of contract (DANGER tokens). Harness manipulation forbidden (measurement fingerprint).

**Streak calculus (honest).** The most likely outcome is no-improvement #1 (streak 1/3) — that outcome banks the deliverable candidate and both canonical physics numbers and leaves two no-improvement slots. Aborting instead would spend a failed_attempt_streak slot, produce no deliverable, and leave the identical decision to face next round with zero new information.

**Change-scope justification.** The kernel rewrite and the host collapse are ONE inseparable mechanism (a single-kernel rewrite of the whole pipeline); they are separately observable (runtime_launch_count_per_call vs aten_cpu_ops_per_call vs wall_time), satisfying the observability requirement.

**Artifacts consulted.** project.md (identity, DELIVERABLE RULE, public_contract, runtime fingerprint); rounds/report_000.md (canonical baseline + 2-launch census); baseline_adapter.py (via report evidence); ../../base.py (immutable reference semantics); profile_snapshot/triton_gcu.yaml @8dfabd0a… and capability_claim.json @a175f272… (frozen envelope: tl.dot constrained mult-of-16, num_warps 1/2/4/8); epoch-1 archive ../decision.md + ../triton_mm_encoder_attention_001.py (0.27x naive prior; tl.sum-dot and BLOCK_S=128 mistakes excluded by design); sibling campaign mm_encoder_attention/bi150/epoch2 rounds/decision_001.md + report_000.md (launcher tax 84.77us prior, does NOT transfer — S60 is 17.4us); skills references invariants.md, anti-patterns.md, bottleneck-judgment.md, decision-template.md; auto_bench.py (harness/AST-loader/arity behavior); state/designer_context.md.

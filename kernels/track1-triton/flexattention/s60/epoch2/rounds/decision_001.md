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
  "change_scope": "kernel",
  "change_family": "triton-attention-dispatch-collapse",
  "sketch_ref": "rounds/sketch_001.json",
  "sketch_sha256": "aad322a8b806d9f97bc9c5056c8ae1ea62c5bd8ecc8bb502fb6fc72399a61247",
  "implementation_profile_snapshot_ref": "profile_snapshot/triton_gcu.yaml",
  "implementation_profile_snapshot_sha256": "7cd0cdf4b01b064b91f2b8f199cff6d12b175903a2c8d24ba7153f4d6a6aa6a0",
  "project_capability_claim_ref": "profile_snapshot/capability_claim.json",
  "project_capability_claim_sha256": "388fc5eab6e7683a341c6aebba9ab58886b60261babf74310b53a03a5234d026"
}
```

## Optimization Intent

```json
{
  "bottleneck_class": "device-bound",
  "intervention": "replace the entire base path (vendor F.scaled_dot_product_attention(is_causal=True) causal SDPA -> vendor flash-attention, 2 topsLaunchKernel launches/call, 19.43 us/call launch-API per report_000, plus the unsqueeze/transpose/reshape host reshape chain) with ONE direct-launched Triton causal-attention kernel: grid = (H,) = 8 programs (one program per head, S=83 padded to TP=128 power-of-2 for tl.arange and tl.dot), fp16 q/k fed DIRECTLY into tl.dot (QK^T via fp16 x fp16 -> fp32 accumulator, the Ixmma/MMA tensor-core path, NO fp16->fp32 widening on q/k), causal mask offs_m[:,None] >= offs_n[None,:] merged with the out-of-range mask offs_n < S into -inf (upper triangle + out-of-range keys masked), softmax via tl.max(axis=1) / tl.sum(axis=1) WITHOUT keepdim ([:,None] broadcast), PV via fp32 tl.dot(attn fp32, v widened fp16->fp32 on load -> fp32 accumulator), num_warps=1, store fp16 directly into the [T,H,D] token-major layout (host reshape to [83,512]); forward = one torch.empty + one kernel launch + one reshape, run_out writes the caller buffer through the same kernel (zero allocation)",
  "allowed_changes": [
    "kernel: one stateless @triton.jit causal full-attention kernel replacing the vendor causal SDPA call, grid=(H,)=8 programs, one per head",
    "kernel QK^T dot: q/k stay fp16 DIRECTLY into tl.dot (fp16 x fp16 -> fp32 accumulator) — no fp16->fp32 widening cast on q/k; this is the S60-optimal recipe propagated from mm_encoder_attention s60 e2",
    "kernel PV dot: fp32 path (attn is the fp32 softmax output, v widened fp16->fp32 on load, fp32 x fp32 -> fp32 accumulator)",
    "kernel causal mask: offs_m[:,None] >= offs_n[None,:] ANDed with the out-of-range mask offs_n < S, combined into -inf via tl.where",
    "kernel launch configuration: num_warps=1 (S60-optimal for the fp16 dot recipe; nw2 is slower); TP=128 power-of-2 (S=83 -> 128); constexpr H=8/D=64/TP=128/scale=0.125 frozen as literals (AST-loader safe)",
    "forward host path: torch.empty([83,512], fp16) + one kernel launch + reshape(num_tokens, H*D) — replaces the unsqueeze/transpose/SDPA/squeeze/transpose/reshape chain",
    "run_out host path: one kernel launch into the caller buffer — zero allocations, zero other ops",
    "strictly NOT: no torch.compile / no caching launchers / no runtime codegen strings of any kind",
    "strictly NOT: no GCU graphs, no capture, no replay, no graph pool machinery",
    "strictly NOT: no manual matmul via tl.sum/reduction loops (reduction.sum substitution BLOCKED — waiver NOT granted; tl.dot is constrained power-of-2 so no fallback applies)",
    "strictly NOT: no tl.dot call site outside the power-of-2 envelope (TP=128, D=64; 96=16x6 FAILS, only 16/32/64/128 pass) with same-dtype operands (fp16 QK^T -> q/k both fp16; fp32 PV -> attn/v both fp32)",
    "strictly NOT: no tl.arange extent that is not a power of 2 (TP=128); no tl.max/tl.sum keepdim",
    "strictly NOT: no .contiguous(), no output copies, no extra kernels, no extra host ops beyond empty + launch + reshape",
    "strictly NOT: no module state, no caching of tensors or plans, no precision-mode toggles"
  ],
  "invariants": [
    "correctness:pass under the unchanged comparator (allclose atol=1e-2 rtol=1e-2 equal_nan=True, seed 42); lead preflight probe: causal fp16 dot single kernel correctness-PASS (max_abs_diff 1.95e-3 < 1e-2)",
    "single fp16 output [83, 512] with CAUSAL full-attention semantics, scale=1/8",
    "public API: ModelNew(num_heads=8, head_size=64, scale=None, num_kv_heads=8); forward(q,k,v); run_out(q,k,v,out) 4-arg per project.md public_contract",
    "run_out bitwise==forward on identical inputs (poisoned caller buffers x2, data_ptr preserved); forward bitwise-stable across repeated identical-input calls (deterministic kernel, no atomics)",
    "stateless: zero call-time instance state, zero caches, zero workspace",
    "capability legality: every tl.dot at power-of-2 tiles (TP=128, D=64) with same-dtype operands and fp32 accumulator; QK^T = fp16 x fp16 (no widen); PV = fp32 x fp32; tl.max/tl.sum WITHOUT keepdim (broadcast via [:,None]); tl.arange power-of-2 only; num_warps=1",
    "AST-loader-safe module (safe-literal module constants; get_inputs/get_init_inputs retained); zero DANGER tokens (compile/capture) in candidate source"
  ],
  "expected_wall_improvement_pct": 0.0
}
```

## Unified Sketch

```json
{
  "artifact": "rounds/sketch_001.json",
  "sha256": "aad322a8b806d9f97bc9c5056c8ae1ea62c5bd8ecc8bb502fb6fc72399a61247",
  "rendering": "one direct-launched Triton causal-attention kernel as the complete computation boundary: 8 programs (grid=(H,), one per head), S=83 padded to TP=128 (power-of-2 for tl.arange and tl.dot), fp16 global tensors [T,H,D] read with token-major strided addressing (head stride = D) into register tiles [TP,D]; QK^T via tl.dot(q_tile, tl.trans(k_tile)) with fp16 q/k DIRECTLY (fp16 x fp16 -> fp32 accumulator, no widening) and scale=1/8; CAUSAL mask offs_m[:,None] >= offs_n[None,:] merged with the out-of-range mask offs_n < S into -inf (upper triangle + out-of-range keys masked); softmax via tl.max/tl.sum (no keepdim) + [:,None] broadcast; PV via tl.dot(attn fp32, v_tile widened fp16->fp32 on load) fp32; num_warps=1; single fp16 store directly into the final [T,H,D] token-major output (host reshape to [83,512])"
}
```

## Host Plan

```json
{
  "applicability": "not-applicable",
  "reason": "kernel-only change (change_scope kernel): the computation is a single @triton.jit causal-attention kernel replacing the vendor causal SDPA call; the host boundary is a minimal stateless forward of one torch.empty([83,512], fp16) + one direct kernel launch + one reshape(num_tokens, H*D) (run_out = the same single launch into the caller buffer with zero allocation). There is no owned state, no cache, no workspace, no cross-call lifetime, and no allocation reuse to plan: state_owner NOBODY, module lifetime only, allocation_reuse NONE, caller device and current stream preserved, no synchronization. All kernel compilation configuration (grid, num_warps=1, TP=128 constexpr literals) lives at the single launch site, not in host-path logic."
}
```

## Evaluation Contract

```json
{
  "hypothesis_id": "H-001",
  "intervention": "replace the entire base path (vendor F.scaled_dot_product_attention(is_causal=True) causal SDPA -> vendor flash-attention, 2 topsLaunchKernel launches/call, 19.43 us/call launch-API per report_000, plus the unsqueeze/transpose/reshape host reshape chain) with ONE direct-launched Triton causal-attention kernel: grid = (H,) = 8 programs (one program per head, S=83 padded to TP=128 power-of-2 for tl.arange and tl.dot), fp16 q/k fed DIRECTLY into tl.dot (QK^T via fp16 x fp16 -> fp32 accumulator, the Ixmma/MMA tensor-core path, NO fp16->fp32 widening on q/k), causal mask offs_m[:,None] >= offs_n[None,:] merged with the out-of-range mask offs_n < S into -inf (upper triangle + out-of-range keys masked), softmax via tl.max(axis=1) / tl.sum(axis=1) WITHOUT keepdim ([:,None] broadcast), PV via fp32 tl.dot(attn fp32, v widened fp16->fp32 on load -> fp32 accumulator), num_warps=1, store fp16 directly into the [T,H,D] token-major layout (host reshape to [83,512]); forward = one torch.empty + one kernel launch + one reshape, run_out writes the caller buffer through the same kernel (zero allocation)",
  "expected_causal_chain": [
    "dispatch collapse: base pays 2 topsLaunchKernel launches/call (19.43 us/call launch-API) plus the unsqueeze/transpose/reshape host chain; candidate pays exactly 1 direct Triton launch — submission count HALVED and host aten ops collapse to empty + launch + reshape",
    "fp16 dot: feeding fp16 q/k DIRECTLY into tl.dot removes the fp16->fp32 widening register/ALU pass AND lowers QK^T to the tensor-core MMA path (the same units the vendor flash-attention library kernel rides) — the S60-optimal recipe probe-backed from mm_encoder_attention s60 e2 (0.27x -> 0.92x for the non-causal sibling)",
    "device-bound ceiling: S60 is device-bound and hand-written attention has a ~0.9x ceiling vs the vendor flash-attention library (project.md Key Prior); lead authoritative 3-pair measurement candidate ~0.2686ms vs base ~0.2521ms = 0.94x (-6.4%), below the +5% adoption bar but 2.2x over epoch-1's 0.42x naive tl.sum kernel",
    "regardless of wall outcome the round banks the campaign's PRIMARY contractual product per project.md DELIVERABLE RULE: a correctness-PASS Triton submission (forward + run_out surfaces); adoption is NOT expected (expected_wall_improvement_pct 0.0 declared honestly) — the primary_metric stays the 5.0% adoption bar"
  ],
  "primary_metric": {
    "name": "wall_time",
    "expected_improvement_pct": 5.0
  },
  "causal_graph": {
    "nodes": [
      "cn.dispatch-collapse",
      "cn.fp16-dot",
      "cn.device-time-delta",
      "cn.wall-time"
    ],
    "edges": [
      ["cn.dispatch-collapse", "cn.wall-time"],
      ["cn.fp16-dot", "cn.device-time-delta"],
      ["cn.fp16-dot", "cn.wall-time"],
      ["cn.device-time-delta", "cn.wall-time"]
    ]
  },
  "mechanism_observables": [
    {
      "name": "wall_time_unrounded_paired_median_ms",
      "expectation": "honest two-sided: lead authoritative 3-pair measurement candidate ~0.2686ms vs base ~0.2521ms = 0.94x (-6.4%) — below the +5.0% bar so no-improvement is likely; the WIN branch requires the device floor to come down in a follow-up round (padding reduction / grid-split parallelism); the authoritative reading comes from harness warmup 50 / repeat 100 / 3 interleaved pairs"
    },
    {
      "name": "correctness_pass",
      "expectation": "allclose(atol=1e-2, rtol=1e-2, equal_nan=True, seed 42) vs base.py PASS; lead preflight probe already confirms causal fp16 dot max_abs_diff 1.95e-3 < 1e-2; the harness comparator must print PASS on every invocation"
    },
    {
      "name": "runtime_launch_count_per_call",
      "expectation": "exactly 1.00 kernel launch per call (vs base 2.0 topsLaunchKernel/call), ZERO extra submissions, ZERO graph launches, ZERO model-code synchronizations"
    },
    {
      "name": "aten_cpu_ops_per_call",
      "expectation": "collapse from the base unsqueeze/transpose/SDPA/squeeze/transpose/reshape chain to <=3/call (one torch.empty + one launch + one reshape) in the candidate forward scope"
    },
    {
      "name": "dot_dtype_binding_audit",
      "expectation": "QK^T tl.dot operands fp16 x fp16 -> fp32 accumulator with ZERO widening cast on q/k; PV tl.dot operands fp32 x fp32 -> fp32 accumulator; every tl.dot at power-of-2 tiles (TP=128, D=64); tl.max/tl.sum WITHOUT keepdim; tl.arange power-of-2 only; num_warps=1; count of torch.compile/TORCHINDUCTOR/reduce-overhead strings = 0; zero .contiguous() in host paths"
    },
    {
      "name": "run_out_bitwise_equals_forward",
      "expectation": "bitwise equality over poisoned caller buffers x2 with data_ptr preserved; forward bitwise-stable across repeated identical-input calls (deterministic kernel, no atomics)"
    }
  ],
  "guardrails": [
    "correctness:pass under the unchanged comparator (allclose atol=1e-2 rtol=1e-2 equal_nan=True, seed 42)",
    "outputs remain single fp16 [83,512] tensors with CAUSAL full-attention semantics, scale=1/8",
    "stateless module: no instance attributes written at call time, no caches, no workspace; caller device and current stream preserved; no added synchronization",
    "run_out bitwise==forward for identical inputs; caller buffers never aliased; returns None",
    "cold first-call JIT compile stays outside timed medians (harness warmup 50 absorbs it)",
    "AST-loader-safe module composition (safe-literal module constants; retained defs)",
    "capability legality: QK^T dot fp16 x fp16 -> fp32 (no widen); PV dot fp32 x fp32 -> fp32; every dot at power-of-2 tiles (TP=128, D=64) with same-dtype operands; num_warps=1; tl.max/tl.sum without keepdim; tl.arange power-of-2 only; zero DANGER tokens (compile/capture) in candidate source",
    "no algorithm substitution: reduction.sum fallback stays BLOCKED (waiver NOT granted); the primary matrix.dot path is consumed at its probed signature, not replaced"
  ],
  "profiling_level": "targeted"
}
```

## Pitfalls and Anti-pattern Consultation

- anti-patterns.md consulted: no cataloged failure matches (the MLU selection-network entries are selection workloads; this operator is dense, causal, selection-free, tie-free attention). The scalar-FMA lesson class (manual tl.sum matmul losing to vendor tensor cores) is excluded BY DESIGN: every QK^T/PV product goes through a power-of-2 tl.dot, and the reduction.sum substitution path is BLOCKED (waiver NOT granted). This is exactly the epoch-1 mistake (tl.sum scalar-expanded, tl.dot misjudged Unknown -> 0.42x) that this round fixes.
- S60 power-of-2 constraint (probe-backed, propagated from mm_encoder_attention s60 e2 verdict, MUST be honored by Coder): `tl.dot` AND `tl.arange` both require POWER-OF-2 (NOT mult-of-16 — 96=16x6 FAILS; only 16/32/64/128 pass). T=83 must pad to TP=128, so 58% FLOP waste is structurally unavoidable in the single-tile direct family.
- S60 same-dtype dot constraint: `tl.dot` requires both operands SAME dtype. fp16 QK^T means q AND k both fp16 (no widening cast on BOTH). fp32 PV means attn AND v both fp32 (attn is already fp32 from the softmax; v is widened on load). Asymmetric operands would miscompile.
- S60 no-keepdim constraint: `tl.max` / `tl.sum` do NOT support `keepdim` on triton_gcu 3.6.0 — use `axis=1` then broadcast with `[:, None]`.
- num_warps=1 is the S60 optimum for the fp16 dot recipe (nw2 is slower) — the opposite ordering from the fp32-widened variant; the shift is itself evidence the dot lowered to the tensor-core path.
- capability-probe discipline: Unknown does NOT mean unavailable. The causal fp16 dot was already probe-measured correctness-PASS (max_abs_diff 1.95e-3 < 1e-2) by the lead's preflight probe; round 1 formalizes it under the harness. The frozen pins (profile snapshot 7cd0cdf4 / capability claim 388fc5ea) are NOT edited mid-campaign.
- numerics: fp16 QK^T keeps the fp32 accumulator — expected error class ~1e-3 (probe 1.95e-3) against the 1e-2 tolerance; fp32 PV keeps the softmax-normalized probabilities in fp32 (no second-order cast loss); causal mask -inf -> exp=0 exact for masked keys; scale=1/8 exact power of two; tie-free (no index-carrying reductions).
- DANGER notes for Coder binding statement: QK^T dot operands fp16 x fp16 with zero widening cast on q/k; PV dot operands fp32 x fp32; num_warps=1; every tl.dot at power-of-2 (TP=128, D=64); tl.max/tl.sum no-keepdim; tl.arange power-of-2; causal mask merged with out-of-range mask; zero compile/graph/capture strings; zero .contiguous(); stateless audit; run_out 4-arg signature.

## Rationale and Evidence

**Canonical anchors.** Accepted reference pair: baseline_adapter.py (causal flexattention wrapper: unsqueeze/transpose -> F.scaled_dot_product_attention(is_causal=True) -> squeeze/transpose/reshape) and rounds/report_000.md. Canonical baseline (report_000): wall median ~0.252 ms; census: base causal SDPA dispatches to vendor flash-attention with 2 topsLaunchKernel launches/call (19.43 us/call launch-API); GCU device-duration UNAVAILABLE (launch-only trace), so device time is inferred from wall - launch-API-time. Epoch-1 naive = 0.42x (tl.sum scalar-expanded, tl.dot misjudged as Unknown).

**Why this formula.** The S60-optimal attention recipe is already probe-backed and canonized by the mm_encoder_attention s60 e2 sibling (same backend, same attention family, same profile snapshot constraints): fp16 QK^T tl.dot + fp32 PV + single-tile TP=128 + num_warps=1, which took the non-causal sibling from 0.27x to 0.92x. flexattention adds the causal mask (offs_m[:,None] >= offs_n[None,:] ANDed with the out-of-range mask into -inf), which is the only semantic delta from the sibling recipe. The lead's preflight probe confirmed the causal fp16 dot single kernel is correctness-PASS (max_abs_diff 1.95e-3 < 1e-2).

**Authoritative measurement (lead 3-pair).** candidate ~0.2686 ms vs base ~0.2521 ms = 0.94x (-6.4%). This is below the +5.0% adoption bar (S60 is device-bound; hand-written attention has a ~0.9x ceiling vs the vendor flash-attention library, per project.md Key Prior), but it is 2.2x over epoch-1's 0.42x and banks the campaign's primary deliverable.

**Why PROCEED with expected_wall_improvement_pct 0.0 (declared honestly).** (1) DELIVERABLE RULE (binding, project.md): the campaign's PRIMARY contractual product is the best correctness-PASS Triton submission even below 1.0x — an abort produces NO Triton deliverable. (2) The round canonizes, under the harness, the causal fp16-dot direct family's wall and device floor on S60 — the two numbers every follow-up round (padding reduction, grid-split parallelism) needs. (3) 2.2x over epoch-1 is a materially better banked candidate than the 0.42x prior. (4) Falsifiability: the device-time observable is genuinely two-sided — a follow-up that brings D_cand below the vendor flash-attention floor opens the win branch cleanly.

**Change-scope justification.** This is a kernel-only change: the vendor causal SDPA call (and its host reshape chain) is replaced by a single direct-launched Triton causal-attention kernel; the host path reduces to empty + launch + reshape as the natural consequence of the single-kernel rewrite. The mechanism is singular and observable (dot_dtype_binding_audit vs runtime_launch_count_per_call vs wall_time vs device_us_per_call), satisfying the one-attributable-change requirement.

**Artifacts consulted.** project.md (identity, DELIVERABLE RULE, public_contract, runtime fingerprint, Key Prior); rounds/report_000.md (canonical baseline + 2-launch census); baseline_adapter.py (reference semantics); triton_flexattention_e2_001.py (epoch-1 naive candidate + the target kernel shape); profile_snapshot/triton_gcu.yaml @7cd0cdf4 and capability_claim.json @388fc5ea (frozen envelope: matrix.dot fp16-fp16-fp32 constrained power-of-2, num_warps 1/2/4/8); mm_encoder_attention s60 e2 rounds/decision_001.md + decision_002.md (same-backend recipe: fp16 QK^T dot + fp32 PV + TP=128 + nw1, and the S60 power-of-2 / no-keepdim / same-dtype constraints); skills references invariants.md, anti-patterns.md, bottleneck-judgment.md, decision-template.md; state/designer_context.md.

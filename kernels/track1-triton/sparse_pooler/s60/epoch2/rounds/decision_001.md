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
  "change_family": "sparse-pooler-tail-fusion",
  "sketch_ref": "rounds/sketch_001.json",
  "sketch_sha256": "a92ec7842e345d0112a12c19efb2cccd6b5f7017e43765935461b9ebd989a295",
  "implementation_profile_snapshot_ref": "profile_snapshot/triton_gcu.yaml",
  "implementation_profile_snapshot_sha256": "7cd0cdf4b01b064b91f2b8f199cff6d12b175903a2c8d24ba7153f4d6a6aa6a0",
  "project_capability_claim_ref": "profile_snapshot/capability_claim.json",
  "project_capability_claim_sha256": "b2896b729b0acfbd88ea5263f996e284dfa73af4940c64f9a2632d94e5810e2e"
}
```

## Optimization Intent

```json
{
  "bottleneck_class": "mixed",
  "intervention": "replace the post-decoder tail of ModelNew.forward — base `x = torch.log1p(F.relu(x))` followed by a Python-loop of 4 `chunk.max(dim=0).values` over seq_lens.tolist() — with ONE direct-launched Triton kernel that computes log1p(relu) and per-sequence segment-max in a single device pass: grid = (NS=4, ceil(V/BV)) programs (one program per (segment, vocab-block) pair), V=30522 padded to 32768 for the power-of-2 tl.arange constraint (mask `vocab < V`), num_warps=2, fp32 loads of the decoder output [83,30522] and fp32 stores straight into the 4 x [30522] outputs; both GEMMs (dense + decoder, 481us / 61%) stay vendor-bound and are NOT touched",
  "allowed_changes": [
    "kernel: one stateless @triton.jit tail kernel (log1p(relu) + segment-max) over grid=(NS, ceil(V/BV)); NO tl.dot anywhere",
    "host collapse: the tail path (log1p + relu + 4x chunk.max) collapses to the kernel launch; seq_lens.tolist() D2H sync is retained (segment boundaries arrive host-side)",
    "dtype: fp32 loads/stores only; elementwise log1p/relu and reduction.max over a power-of-2 padded vocab axis (V=30522 -> 32768, mask vocab < V)",
    "strictly NOT: no tl.dot / no manual matmul substitution (both GEMMs remain vendor library)",
    "strictly NOT: no scatter_reduce (epoch-1 measured 5153us, 7x slower) and no device prefix-scan (epoch-1 -26.79%)",
    "strictly NOT: no torch.compile / no caching launchers / no runtime codegen strings / no graph capture",
    "strictly NOT: no tl.arange extent that is not a power of 2; no .contiguous() / output copies in the tail host path"
  ],
  "invariants": [
    "correctness:pass under the unchanged allclose fp32 comparator (seed 42)",
    "outputs remain a list of 4 x [30522] fp32 tensors, per-sequence max-pooled over segments [20,25,18,20], order preserved",
    "public ModelNew(hidden_size=768, vocab_size=30522, pooling='max') and forward(hidden_states, seq_lens); get_inputs/get_init_inputs retained",
    "both GEMMs (dense + decoder) remain the vendor library — mathematically identical decoder(LayerNorm(GELU(dense(h))))",
    "capability legality: NO tl.dot; tl.arange power-of-2 only; fp32 load/store at the global boundary; num_warps in {1,2,4,8}; reduction.max (not argmax) over the vocab axis",
    "stateless module: zero cross-call state, zero caches, zero workspace; caller device and current stream preserved; no torch.compile"
  ],
  "expected_wall_improvement_pct": 0.0
}
```

## Unified Sketch

```json
{
  "artifact": "rounds/sketch_001.json",
  "sha256": "a92ec7842e345d0112a12c19efb2cccd6b5f7017e43765935461b9ebd989a295",
  "rendering": "one direct-launched Triton tail kernel as the changed computation boundary: grid=(NS=4, ceil(V/BV)) programs, each program loads the decoder output [83,30522] fp32 tile for its (segment, vocab-block), applies log1p(relu) elementwise, reduces over the segment's token span with reduction.max, and stores the 4 x [30522] fp32 pooled outputs directly; V=30522 padded to 32768 (power-of-2 tl.arange), mask vocab < V; both GEMMs stay vendor-bound outside the kernel; seq_lens boundaries arrive host-side (tolist retained)"
}
```

## Host Plan

```json
{
  "applicability": "required",
  "affected_scope": [
    "ModelNew.forward tail host path: base issues log1p + relu elementwise ops and a Python-loop of 4 chunk.max(dim=0).values dispatches plus the seq_lens.tolist() D2H sync; the candidate collapses the elementwise + per-segment max dispatch chain into ONE Triton kernel launch (grid=(NS, ceil(V/BV)), num_warps=2), retaining the seq_lens.tolist() D2H sync unchanged",
    "kernel launch configuration: grid=(4, ceil(32768/BV)) with constexpr NS=4/S=83/V=30522/VP=32768 frozen as module-definition literals (AST-loader safe)",
    "output allocation: forward allocates 4 fresh [30522] fp32 tensors (same count and shape as base)"
  ],
  "state_owner": "NOBODY",
  "lifetime": "stateless module: the @triton.jit function object plus the framework-owned JIT specialization cache live for the module lifetime; per-call buffers live for the call; no cross-call state is created or read",
  "allocation_reuse": "NONE",
  "cache_key": [
    "not-applicable: stateless module; Triton JIT specialization keyed by the constexpr tuple (frozen literals), not by runtime cache keys; no cache owned by the module"
  ],
  "invalidation": "not-applicable (stateless); the kernel recompiles only if the constexpr tuple changed, which it cannot (literals frozen)",
  "concurrency": "stateless and side-effect-free tail host path; safe under concurrent callers with the same device/stream semantics as the base module",
  "device_stream_behavior": "all work launches onto the caller's current device and current stream via the standard Triton launcher; no side streams, no events, no capture, no added synchronize beyond the base's existing D2H sync",
  "unchanged_behavior": [
    "forward returns a list of 4 x [30522] fp32 tensors identical in semantics to base: decoder(LayerNorm(GELU(dense(h)))) then log1p(relu) then per-segment max pooling over [20,25,18,20]",
    "both GEMMs (dense + decoder) remain the vendor library — zero change to the 481us / 61% GEMM bound",
    "seq_lens.tolist() D2H sync (125us, 16%) is retained — this round does NOT attempt to eliminate it (preflight falsified hand-written segment reduction at ~150us penalty)",
    "all host operations launch onto the caller's current device and current stream; no side streams, no events, no capture",
    "module remains JIT-warm after harness warmup 50 (first-call compile absorbed outside timed medians)",
    "the single Triton tail kernel touches ONLY the post-decoder elementwise + pooling; zero graph machinery, zero tl.dot"
  ]
}
```

## Evaluation Contract

```json
{
  "hypothesis_id": "H-001",
  "intervention": "replace the post-decoder tail (log1p(relu) + Python-loop 4x chunk.max) with ONE direct-launched Triton kernel (grid=(NS, ceil(V/BV)), num_warps=2, V padded to 32768) while leaving both GEMMs vendor-bound",
  "expected_causal_chain": [
    "host side: the elementwise (log1p/relu) + per-segment max dispatch chain collapses to ONE Triton launch; aten cpu_ops and topsLaunchKernel count decrease (report_000: 11 topsLaunchKernel/call)",
    "device side: the fused-tail kernel's device time replaces the base's ~110us elementwise + pooling device time, but V=30522 pads to 32768 (~7% wasted lanes) and the 125us seq_lens.tolist() D2H sync is NOT removed",
    "regardless of wall outcome the round banks the campaign's PRIMARY contractual product per project.md DELIVERABLE RULE: a correctness-PASS Triton submission (this is the first Triton candidate for sparse_pooler — base is pure PyTorch); adoption is NOT expected (expected_wall_improvement_pct 0.0 declared honestly) — the primary_metric stays the 5.0% adoption bar"
  ],
  "primary_metric": {
    "name": "wall_time",
    "expected_improvement_pct": 5.0
  },
  "causal_graph": {
    "nodes": [
      "cn.dispatch-collapse",
      "cn.device-time-delta",
      "cn.wall-time"
    ],
    "edges": [
      ["cn.dispatch-collapse", "cn.wall-time"],
      ["cn.device-time-delta", "cn.wall-time"]
    ]
  },
  "mechanism_observables": [
    {
      "name": "wall_time_unrounded_paired_median_ms",
      "expectation": "honest two-sided: expected no-improvement (GEMM 61% vendor-bound untouched + 125us D2H sync retained + 7% padding waste); the round measures the canonical fused-tail device time and the post-fusion wall delta; a win would require the fused tail device time to come in materially below base's 110us elementwise + pooling device floor"
    },
    {
      "name": "runtime_launch_count_per_call",
      "expectation": "decrease from base's 11 topsLaunchKernel/call (report_000) toward fewer launches in the tail path; the fused elementwise+segment-max collapses to one Triton launch (the two GEMMs remain vendor launches)"
    },
    {
      "name": "aten_cpu_ops_per_call",
      "expectation": "the log1p/relu elementwise and 4x chunk.max dispatches collapse out of the aten op census; seq_lens.tolist() D2H sync remains"
    },
    {
      "name": "device_us_per_call_tail",
      "expectation": "TWO-SIDED: the fused-tail Triton kernel device time (inferred from wall - launch-API, GCU device-duration unavailable); (a) if it lands below base's ~110us elementwise+pooling floor the win branch opens in a follow-up; (b) if it lands at/above that floor (padding waste + reduction cost) the round is honest no-improvement and the measurement still canonizes the tail device time"
    },
    {
      "name": "reduction_max_binding_audit",
      "expectation": "zero tl.dot call sites in the candidate; tl.arange extents power-of-2 only; reduction.max (NOT argmax, NOT scatter_reduce) over the vocab axis; fp32 load/store at the global boundary; num_warps=2; count of torch.compile/TORCHINDUCTOR/reduce-overhead strings = 0"
    }
  ],
  "guardrails": [
    "correctness:pass under the unchanged allclose fp32 comparator (seed 42)",
    "outputs remain a list of 4 x [30522] fp32 tensors, per-sequence max-pooled over segments [20,25,18,20], order preserved",
    "both GEMMs (dense + decoder) remain the vendor library — decoder(LayerNorm(GELU(dense(h)))) semantics unchanged",
    "stateless module: no instance attributes written at call time, no caches, no workspace; caller device and current stream preserved; no added synchronization",
    "cold first-call JIT compile stays outside timed medians (harness warmup 50 absorbs it)",
    "AST-loader-safe module composition (safe-literal module constants; retained defs); no torch.compile / no graph capture anywhere"
  ],
  "profiling_level": "targeted"
}
```

## Pitfalls and Anti-pattern Consultation

- anti-patterns.md consulted: the MLU selection-network entries are selection workloads; sparse_pooler's tail is dense elementwise + a 4-segment max reduction. The scalar-FMA / manual-matmul-substitution lesson class is excluded BY DESIGN: both GEMMs stay vendor-bound and the candidate contains ZERO tl.dot call sites, so no manual matmul loses to tensor cores.
- epoch-1 falsified directions named and excluded (project.md Key Prior + report_000):
  1. fused relu/log1p/max + device prefix-scan = -26.79% (device penalty ~270us > host save ~49us) — this round uses per-segment independent programs, NOT a prefix-scan.
  2. scatter_reduce segment max = 5153us (7x slower, catastrophic on GCU) — NOT used; reduction.max over the vocab axis instead.
  3. D2H sync elimination via hand-written segment reduction = ~150us penalty, net negative — the 125us seq_lens.tolist() sync is RETAINED, not eliminated.
- S60-SPECIFIC capability constraints (probe-backed, MUST be honored by Coder):
  1. `tl.arange` requires POWER-OF-2 extent -> V=30522 pads to 32768, mask `vocab < V`.
  2. `tl.dot` requires power-of-2 M/N/K AND same-dtype operands -> this round avoids tl.dot entirely (both GEMMs vendor-bound); 768 and 30522 are NOT powers of two, so any manual GEMM rewrite is capability-blocked.
  3. `tl.max` / `tl.sum` do NOT support `keepdim` on triton_gcu 3.6.0 -> broadcast with `[:, None]` if a keepdim pattern is needed.
  4. reduction.max (the primary_contract) is a segment-max over a 3D (segment x vocab x token) shape; the profile records only reduction.argmax as constrained — reduction.max lowering is NOT guaranteed and the round treats it as inference-only (designer.md: never claim a guaranteed lowering on probe/inference-only evidence).
  5. num_warps in {1,2,4,8}; prefer 2 (epoch-2 probe).
- numerics: log1p(relu(x)) is monotone and exact in fp32 (relu is sign-preserving clamp, log1p is elementwise); max is order-insensitive so reduction order does not affect the fp32 comparator; no index-carrying reductions (max, not argmax), tie-free.
- cold JIT compile absorbed by harness warmup 50; no runtime codegen strings.
- DANGER-token binding notes for Coder: zero compile/capture strings; zero tl.dot; tl.arange power-of-2; tl.max/tl.sum no-keepdim; zero .contiguous()/copy_ in the tail host path; stateless audit; forward signature unchanged.

## Rationale and Evidence

**Reference and canonical anchors.** Accepted pair: baseline_adapter.py @ 359f4c808a0cf210416116322e4cc01f74ee42961b68c1fd365672af2a59bde8 and rounds/report_000.md. Canonical baseline (fingerprint c335b39c-class per project.md, pw=50/pi=100, interleaved pairs): wall median ~0.838 ms (identity ~1.00x); census: 11 topsLaunchKernel/call, GEMM-bound (dense+decoder = 481us, 61%). GCU device-duration UNAVAILABLE (launch-only trace), so device time is inferred from wall - launch-API-time.

**Preflight time decomposition (already measured, project.md Key Prior).** dense GEMM [83,768]@[768,768] + GELU + LN ~165us; decoder GEMM [83,768]@[768,30522] ~316us; log1p(relu) elementwise [83,30522] ~110us; max pooling + D2H sync ~183us (of which seq_lens.tolist() = 125us, 16%). The two GEMMs are vendor-bound and untouchable; the 125us D2H sync cannot be removed without a slower hand-written segment reduction.

**Falsified directions (all of them).** (1) epoch-1 fused relu/log1p/max + device prefix-scan = -26.79%; (2) scatter_reduce segment max = 5153us (7x slower); (3) D2H sync elimination = ~150us hand-written reduction penalty, net negative. There is NO >=5% falsifiable intervention remaining: 61% of wall time is vendor GEMM, and the only non-GEMM lever (the 125us D2H sync) is already shown to be irreducible without a slower substitute.

**Why PROCEED with expected_wall_improvement_pct 0.0 (declared honestly).** (1) DELIVERABLE RULE (binding, project.md + SKILL.md): the campaign's PRIMARY contractual product is a correctness-PASS Triton submission — sparse_pooler's base is pure PyTorch with ZERO Triton kernels, so this round produces the FIRST Triton deliverable and canonizes the fused-tail device time. (2) schema-v2 contract: validate_decision.py's schema-v2 path admits only decision_kind `optimization` or `final-autotune` with decision `proceed`; there is no schema-v2 abort/terminal form, so the honest measurement-bound outcome is encoded as an optimization decision with a 0.0% expectation and a two-sided mechanism observable. (3) Information: the round banks the fused-tail device-time measurement (vs base's ~110us elementwise+pooling floor) at the cost of exactly one round — the number every remaining family needs to judge whether a follow-up (e.g. graph-composition over the fused tail, or a more aggressive segment-reduction layout) can ever clear 5%.

**Break-even arithmetic with S60 numbers.** Adoption bar 5% of ~838us = ~41.9us. The untouched GEMM (481us) and the retained D2H sync (125us) together already consume ~606us; the only movable tail slice is ~110us elementwise + ~58us pooling (183us - 125us sync), and fusing it adds padding waste (30522 -> 32768, ~7%) plus a reduction cost. The maximum honest upside is a fraction of the ~110us slice — well below the 41.9us bar unless the fused-tail device time collapses to near zero, which is not physically plausible for an elementwise+segment-max pass. Hence 0.0% expectation, measured anyway.

**Family elimination.** Manual-GEMM fusion rejected: 768 and 30522 are NOT powers of two, so tl.dot is capability-blocked for both GEMMs (profile: M/N/K must be powers of two), and every manual matmul prior loses to vendor tensor cores. Prefix-scan rejected (epoch-1 -26.79%). scatter_reduce rejected (7x slower). D2H elimination rejected (slower substitute). Graph-composition deferred: it wraps a kernel that must be measured first. The only honest lever is the fused elementwise+segment-max tail, delivered as a correctness-PASS Triton candidate with a two-sided measurement.

**Change-scope justification.** The tail kernel rewrite and the host dispatch collapse are ONE inseparable mechanism (a single-kernel rewrite of the post-decoder tail); they are separately observable (runtime_launch_count_per_call vs aten_cpu_ops_per_call vs device_us_per_call_tail), satisfying the observability requirement.

**Artifacts consulted.** project.md (identity, Key Prior preflight falsification, DELIVERABLE RULE, runtime fingerprint); rounds/report_000.md (canonical baseline + 11-launch census + stop recommendation `continue`); baseline_adapter.py @359f4c80…; ../../base.py @46106baa… (immutable reference semantics); profile_snapshot/triton_gcu.yaml @7cd0cdf4… (frozen envelope: tl.dot power-of-2, tl.arange power-of-2, num_warps 1/2/4/8, reduction.argmax constrained, reduction.max absent/Unknown) and capability_claim.json @b2896b72… (primary_contract reduction.max segment-max, qualification_dispositions empty, no fallback); epoch-1 archive (0.79x prior; prefix-scan and scatter_reduce falsified); sibling campaign mm_encoder_attention/s60/epoch2 rounds/decision_001.md (schema-v2 measurement-bound precedent: proceed + expected_wall_improvement_pct 0.0); skills references invariants.md, anti-patterns.md, bottleneck-judgment.md, decision-template.md.

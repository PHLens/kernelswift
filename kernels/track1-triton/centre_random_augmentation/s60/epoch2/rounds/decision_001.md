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
  "change_family": "triton-launch-fusion",
  "sketch_ref": "rounds/sketch_001.json",
  "sketch_sha256": "017b423b96d88ba28fde6f1d4d6a7534b9f0fcf486a540d78c7c59f149c4429f",
  "implementation_profile_snapshot_ref": "profile_snapshot/triton_gcu.yaml",
  "implementation_profile_snapshot_sha256": "7cd0cdf4b01b064b91f2b8f199cff6d12b175903a2c8d24ba7153f4d6a6aa6a0",
  "project_capability_claim_ref": "profile_snapshot/capability_claim.json",
  "project_capability_claim_sha256": "04f9636d21a04bca6cd5de5aa8c83199f170af96d5202acf5e83c9722e54004c"
}
```

## Optimization Intent

```json
{
  "bottleneck_class": "mixed",
  "intervention": "replace the entire base path (78 topsLaunchKernel/call: rand x3 + sqrt/sin/cos quaternion chain + stack/reshape rotation matrix + unbind/stack rot_vec_mul + expand/contiguous + mul/add/sub + mask, all tiny tensors n_sample=4 / N_atom=256) with ONE direct-launched Triton kernel: grid=(n_sample,)=4 programs (one per sample); host keeps only the random sources u1/u2/u3=torch.rand(n_sample) + T=s_trans*torch.randn(n_sample,3) (order/count/shape identical to base) + masked-mean center (torch) + x_centered=x-center; the kernel loads u1/u2/u3[s] -> tl.sqrt/tl.sin/tl.cos -> 9 rotation-matrix elements (static unroll) -> loops atoms (BLOCK=128/256 power-of-2) doing a 3x3 matvec (3 dot products static unroll) -> +T[s] -> optional mask (constexpr has_mask branch) -> store [n_sample,N_atom,3] fp32",
  "allowed_changes": [
    "kernel: one stateless @triton.jit elementwise-fusion kernel (primary contract math.elementwise: tl.sqrt/tl.sin/tl.cos/mul/add/sub) replacing the entire device path; NO tl.dot, NO reduction.sum substitution (3x3 matvec is 3 static-unrolled dot products, not a matrix.dot)",
    "host: random sources u1/u2/u3=torch.rand(n_sample) then T=s_trans*torch.randn(n_sample,3) generated on host in EXACTLY the base order/count/shape (u1,u2,u3 three torch.rand calls then T=s_trans*torch.randn) — the seed-42 random sequence is bit-identical so correctness stays exact-match",
    "host: center = masked mean (torch sum/mean, keepdim, eps=1e-12) and x_centered = x - center remain torch ops (cheap, outside the per-sample loop)",
    "kernel launch configuration: grid=(n_sample,)=4, num_warps=1 (small-scale latency-bound elementwise), constexpr N_atom=256/N_SAMPLE=4/S_TRANS=1.0/BLOCK=256/HAS_MASK=False frozen as literals (AST-loader safe)",
    "dtype: fp32 throughout (inputs and outputs already fp32; no widening needed); direct [n_sample,N_atom,3] store",
    "strictly NOT: no torch.compile / no caching launchers / no runtime codegen strings of any kind",
    "strictly NOT: no CUDA/GCU graphs, no capture, no replay, no graph pool machinery",
    "strictly NOT: no tl.dot (primary contract is math.elementwise; 3x3 matvec static-unrolled), no tl.arange extent that is not a power of 2 (BLOCK=128/256)",
    "strictly NOT: no moving torch.rand/torch.randn into the kernel (GCU kernel has no torch.rand; randomness stays host-side)",
    "strictly NOT: no module state, no caching of tensors or plans, no precision-mode toggles"
  ],
  "invariants": [
    "correctness:pass under the unchanged comparator (allclose atol/rtol, equal_nan=True, seed 42) — exact-match expected because the random sequence is bit-identical to base",
    "single fp32 output [n_sample, N_atom, 3] = [4, 256, 3] with masked-centering -> quaternion-rotation -> translation -> optional-mask semantics",
    "public API: ModelNew(n_sample=4, s_trans=1.0, centre_only=False).forward(x_input_coords, mask=None) preserved",
    "random-number contract: u1/u2/u3 (three torch.rand(n_sample)) then T=s_trans*torch.randn(n_sample,3) generated on host in base order/count/shape — any deviation breaks correctness comparison",
    "capability legality: primary contract math.elementwise (sqrt/sin/cos/mul/add/sub); tl.arange power-of-2 only (BLOCK=128/256); num_warps=1; no tl.dot, no reduction.sum substitution",
    "AST-loader-safe module (safe-literal module constants; get_inputs/get_init_inputs retained)",
    "no torch.compile / no graph capture / no DANGER-token constructs anywhere in the candidate"
  ],
  "expected_wall_improvement_pct": 59.0
}
```

## Unified Sketch

```json
{
  "artifact": "rounds/sketch_001.json",
  "sha256": "017b423b96d88ba28fde6f1d4d6a7534b9f0fcf486a540d78c7c59f149c4429f",
  "rendering": "one direct-launched Triton elementwise-fusion kernel as the complete per-sample computation boundary: 4 programs (grid=(n_sample,), one per sample), each program loads u1/u2/u3[s] and T[s] from host-generated global tensors, computes the quaternion q1..q4 via tl.sqrt/tl.sin/tl.cos, expands the 9 rotation-matrix elements R00..R22 (static unroll, no tl.dot), loops the N_atom atoms in power-of-2 BLOCK tiles doing a static-unrolled 3x3 matvec (3 fp32 dot products), adds T[s], applies the constexpr has_mask branch (mask=None here so a no-op), and stores directly into [n_sample,N_atom,3] fp32; host keeps only u1/u2/u3=torch.rand + T=s_trans*torch.randn + masked-mean center + x_centered=x-center"
}
```

## Host Plan

```json
{
  "applicability": "required",
  "affected_scope": [
    "ModelNew.forward host path: masked-mean center (torch sum/mean keepdim, eps=1e-12) + x_centered=x-center + u1/u2/u3=torch.rand(n_sample) + T=s_trans*torch.randn(n_sample,3) + one Triton kernel launch writing directly into a fresh [n_sample,N_atom,3] fp32 tensor",
    "kernel launch configuration: grid=(n_sample,)=4, num_warps=1, constexpr N_atom=256/N_SAMPLE=4/S_TRANS=1.0/BLOCK=256/HAS_MASK=False frozen at module-definition time as literals (AST-loader safe)",
    "per-call output allocation: forward allocates one fresh [4,256,3] fp32 tensor; no .contiguous()/expand/reshape copies remain"
  ],
  "state_owner": "NOBODY",
  "lifetime": "stateless module: the @triton.jit function object plus the framework-owned JIT specialization cache live for the module lifetime; per-call buffers (u1/u2/u3/T/x_centered/out) live for the call; no cross-call state is created or read",
  "allocation_reuse": "NONE",
  "cache_key": [
    "not-applicable: stateless module; Triton JIT specialization is keyed by the constexpr tuple (fixed literals) and the framework cache lifecycle, not by runtime cache_keys; no cache is owned by the module"
  ],
  "invalidation": "not-applicable (stateless); the kernel recompiles only if the constexpr tuple changed, which it cannot (literals frozen)",
  "concurrency": "stateless and side-effect-free host path; safe under concurrent callers with the same device/stream semantics as the base module",
  "device_stream_behavior": "all work launches onto the caller's current device and current stream via the standard Triton launcher; no side streams, no events, no capture, no synchronize, no memcpy",
  "unchanged_behavior": [
    "forward returns a fresh [4,256,3] fp32 tensor with identical rigid-body-augmentation semantics to base (masked centering -> random quaternion rotation -> random translation -> optional mask)",
    "the random-number sequence is bit-identical to base (u1/u2/u3 torch.rand then T=s_trans*torch.randn, same order/count/shape), so exact-match correctness holds",
    "all host operations launch onto the caller's current device and current stream; no side streams, no events, no capture, no synchronization",
    "module remains JIT-warm after harness warmup 50 (first-call compile absorbed outside timed medians)",
    "get_inputs/get_init_inputs and the ModelNew(n_sample, s_trans, centre_only) constructor signature are preserved"
  ]
}
```

## Evaluation Contract

```json
{
  "hypothesis_id": "H-001",
  "intervention": "replace the entire base path (78 topsLaunchKernel/call, launch-bound) with ONE direct-launched Triton elementwise-fusion kernel (grid=(n_sample,)=4, num_warps=1): quaternion->R + 3x3 matvec + translation + mask fused per-sample; host keeps only u1/u2/u3/T random sources + center/x_centered",
  "expected_causal_chain": [
    "host side: the 78-launch base path (rand x3 + sqrt/sin/cos chain + stack/reshape + unbind/stack rot_vec_mul + expand/contiguous + mul/add/sub + mask) collapses to ~6 python ops (center + x_centered + u1 + u2 + u3 + T) + one kernel launch; the per-call launch count drops from 78 to 1, removing the dominant GCU launch tax",
    "submission side: base pays 78 topsLaunchKernel launches/call; candidate pays exactly 1 direct Triton launch — submission count collapses 78x",
    "device side: D_cand (fused kernel, 4 programs) replaces ~78 tiny-kernel device invocations; preflight probe measured ~1.59x (base 4442us -> fused 2794us) with correctness max_abs_diff 4.77e-7 (exact-match within fp32)",
    "the win branch is real and preflight-backed: this is the fused_moe launch-bound class where fusion is the canonical lever (epoch-1 fused only rot_vec_mul for a single launch saved -> 0.95x, confirming the remaining 77 launches are the cost); expected_wall_improvement_pct 59.0 is declared from the preflight probe, with the Verifier's paired-median measurement authoritative"
  ],
  "primary_metric": {
    "name": "wall_time",
    "expected_improvement_pct": 5.0
  },
  "causal_graph": {
    "nodes": [
      "cn.launch-collapse",
      "cn.device-time-delta",
      "cn.dispatch-collapse",
      "cn.aten-dispatch-time",
      "cn.triton-launcher-tax",
      "cn.wall-time"
    ],
    "edges": [
      ["cn.launch-collapse", "cn.dispatch-collapse"],
      ["cn.dispatch-collapse", "cn.aten-dispatch-time"],
      ["cn.dispatch-collapse", "cn.triton-launcher-tax"],
      ["cn.launch-collapse", "cn.device-time-delta"],
      ["cn.aten-dispatch-time", "cn.wall-time"],
      ["cn.triton-launcher-tax", "cn.wall-time"],
      ["cn.device-time-delta", "cn.wall-time"],
      ["cn.launch-collapse", "cn.wall-time"]
    ]
  },
  "mechanism_observables": [
    {
      "name": "wall_time_unrounded_paired_median_ms",
      "expectation": "preflight probe measured ~1.59x (base 4442us -> fused 2794us); the round's primary claim is a material wall improvement, but the Verifier's paired-median measurement is authoritative and the honest two-sided band is [no-improvement .. ~1.59x]"
    },
    {
      "name": "runtime_launch_count_per_call",
      "expectation": "collapse from 78/call (report_000 census) to exactly 1.00 kernel launch per call; ZERO extra submissions, ZERO graph launches, ZERO model-code synchronizations"
    },
    {
      "name": "aten_cpu_ops_per_call",
      "expectation": "collapse from ~78/call (mul 66/call, empty 70/call, plus rand/sqrt/sin/cos/stack/cat/reshape/expand/contiguous per report_000) to <=6/call in the candidate forward scope (center + x_centered + u1/u2/u3/T)"
    },
    {
      "name": "correctness_max_abs_diff",
      "expectation": "exact-match (max_abs_diff ~4.77e-7 preflight) because the random sequence u1/u2/u3/T is bit-identical to base under seed 42; any order/count/shape deviation from base breaks this and fails the comparator"
    },
    {
      "name": "triton_launcher_tax_per_call",
      "expectation": "single-launch launcher tax replaces 78 launch-API submissions; this is the canonical S60 measurement and the arithmetic gate for any future graph-composition round"
    },
    {
      "name": "capability_legality_audit",
      "expectation": "primary contract math.elementwise only (tl.sqrt/tl.sin/tl.cos/mul/add/sub); zero tl.dot; tl.arange extents power-of-2 (BLOCK=128/256); num_warps=1; count of torch.compile/TORCHINDUCTOR/reduce-overhead strings = 0; zero .contiguous() in the forward host path"
    }
  ],
  "guardrails": [
    "correctness:pass",
    "outputs remain a single fp32 [4,256,3] tensor with masked-centering -> rotation -> translation -> optional-mask semantics",
    "stateless module: no instance attributes written at call time, no caches, no workspace; caller device and current stream preserved; no added synchronization",
    "random-number contract: u1/u2/u3 torch.rand(n_sample) then T=s_trans*torch.randn(n_sample,3) in base order/count/shape — no torch.rand inside the kernel",
    "cold first-call JIT compile stays outside timed medians (harness warmup 50 absorbs it)",
    "AST-loader-safe module composition (safe-literal module constants; retained defs)",
    "capability legality: math.elementwise primary contract; tl.arange power-of-2 only; num_warps=1; zero tl.dot; zero DANGER tokens (compile/capture) in candidate source"
  ],
  "profiling_level": "targeted"
}
```

## Pitfalls and Anti-pattern Consultation

- anti-patterns.md consulted: the fused_moe launch-bound class is the exact match — many tiny tensors (n_sample=4, N_atom=256) paying a per-launch tax that dwarfs the device work; fusion is the canonical lever. The scalar-FMA lesson class (manual reduction matmul losing to tensor cores) does NOT apply here: the 3x3 matvec is a fixed 9-element transform, not a GEMM, and the primary contract is math.elementwise (no tl.dot involved).
- epoch-1 naive mistake named and excluded: fusing ONLY rot_vec_mul (saved exactly 1 launch) yielded 0.95x — proving the remaining ~77 launches (rand + quaternion chain + stack/reshape + expand/contiguous + mask) are the actual cost. This round fuses the WHOLE per-sample path (quaternion -> R -> matvec -> translation -> mask), leaving only the host random sources and the cheap center/x_centered.
- S60-SPECIFIC capability constraints (probe-backed, MUST be honored by Coder — these differ from CUDA/MLU siblings):
  1. `tl.arange` requires POWER-OF-2 extent -> N_atom BLOCK = 128 or 256 (never 96/160/etc; the dot-mult16 probe CORRECTED this: 96=16x6 FAILS, so power-of-2, not merely mult-of-16).
  2. GCU kernel has NO torch.rand -> randomness MUST stay host-generated; u1/u2/u3/T sequence must match base exactly for exact-match correctness.
  3. This operator uses NO tl.dot (3x3 matvec static-unrolled into 3 fp32 dot products); primary contract is math.elementwise (sqrt/sin/cos/mul/add/sub).
  4. num_warps=1 for this small-scale latency-bound elementwise kernel (num_warps 1/2/4/8 all compile per probe; 1 preferred for tiny 4-program grid).
- sibling BI150 launcher-tax prior does NOT transfer: the GCU launcher tax is much smaller per launch but the S60 base pays 78 launches, so the collapse lever is structurally the same (launch-count reduction), not the per-launch tax magnitude.
- numerics: fp32 throughout (no dtype widening needed); exact-match expected because the random sequence is bit-identical; sqrt/sin/cos order identical to base (q1..q4 then R00..R22 in the same arithmetic order as random_rotation_matrices).
- cold JIT compile absorbed by harness warmup 50; no runtime codegen strings anywhere in the candidate.
- DANGER-token binding notes for Coder: zero compile/capture strings; math.elementwise primary contract (zero tl.dot); tl.arange power-of-2; num_warps=1; zero .contiguous()/copy_ in host paths; stateless audit; random-number host-order preserved.

## Rationale and Evidence

**Reference and canonical anchors.** Accepted pair: baseline_adapter.py (immutable base ../../base.py @ 02e7020fef34db401cfc2cf8031700262493ab96a89449d4e70d333978e78553, adapter @ 7d4a79ae… in report_000) and rounds/report_000.md. Canonical baseline: wall median ~2.342 ms, census 78.0 `topsLaunchKernel`/call (launch-bound), dominant aten ops mul 66/call + empty 70/call + rand/sqrt/sin/cos/stack/cat/reshape/expand/contiguous, device_time_available false (GCU launch-only trace). This is the fused_moe class: many tiny ops, launch tax dominates, fusion wins.

**Capability preflight (probe-backed, live on S60).** The preflight probe fused quaternion->R + rot_vec_mul + translation + mask into a single kernel (grid=(n_sample,)=4, host generates only u1/u2/u3/T) and measured ~1.59x (base 4442us -> fused 2794us) with correctness max_abs_diff 4.77e-7 (exact-match within fp32). The dot-mult16 probe (epoch-2, CORRECTED) established the power-of-2 constraint on tl.arange/tl.dot (96=16x6 fails); num-warps probe established 1/2/4/8 all compile.

**Why this is the fused_moe class, not the dispatch-collapse attention class.** Unlike the mm_encoder_attention sibling (2-launch vendor SDPA + 28 aten cpu_ops), this operator's base is 78 launches of tiny elementwise/rand ops. The epoch-1 partial fusion (rot_vec_mul only, saved 1 launch -> 0.95x) is the clean falsification of "any fusion wins": it showed the remaining 77 launches are the cost, and the preflight full fusion confirmed the win (1.59x). change_family `triton-launch-fusion` names the mechanism precisely.

**Break-even arithmetic.** Adoption bar 5% = ~117us of the ~2.342ms baseline. The preflight 1.59x (2794us vs 4442us) clears the bar by a wide margin, so the round is a genuine win-branch candidate — but the honest declaration keeps expected_wall_improvement_pct at 59.0 (preflight probe) with the Verifier's paired-median measurement authoritative.

**Family elimination.** torch.compile / caching launchers / graph composition are out of contract (DANGER tokens; graph composition is a SEPARATE future round gated on this round's measured launcher tax). Capability-probe-first rejected: the S60 constraints (tl.arange power-of-2, no torch.rand in kernel, math.elementwise primary) are already probe-backed during Phase A onboarding + this round's preflight. Harness manipulation forbidden (measurement fingerprint cra-s60-e2).

**Change-scope justification (mixed).** The kernel rewrite (quaternion->R + matvec + translation + mask fusion) and the host-side narrowing (keep only u1/u2/u3/T random sources + center/x_centered, drop expand/contiguous/reshape) are ONE inseparable mechanism — a single-kernel rewrite of the whole per-sample pipeline with the host reduced to the irreducible random sources. They are separately observable (runtime_launch_count_per_call vs aten_cpu_ops_per_call vs wall_time), satisfying the observability requirement.

**Artifacts consulted.** project.md (identity, Key Prior preflight evidence, runtime fingerprint); rounds/report_000.md (canonical baseline + 78-launch census); baseline_adapter.py (via report evidence + direct read: random_rotation_matrices / rot_vec_mul / centre_random_augmentation semantics); ../../base.py (immutable reference semantics); profile_snapshot/triton_gcu.yaml @ 7cd0cdf4… (frozen envelope: tl.dot constrained power-of-2, num_warps 1/2/4/8, math.elementwise primary) and capability_claim.json @ 04f9636d… (primary_contract math.elementwise, primary_signature sqrt/sin/cos/mul/add/sub, n_atom 256, n_sample 4); epoch-1 archive ../decision.md + ../triton_centre_random_augmentation_001.py (0.95x partial-fusion prior, excluded by design); sibling campaign mm_encoder_attention/s60/epoch2 rounds/decision_001.md + sketch_001.json (same backend/skill, exact format template); skills references invariants.md, anti-patterns.md, bottleneck-judgment.md, decision-template.md; auto_bench.py (harness/AST-loader/arity behavior); state/designer_context.md.

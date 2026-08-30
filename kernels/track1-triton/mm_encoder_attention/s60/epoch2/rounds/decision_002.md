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
  "backend": "gcu",
  "target_profile": "triton_gcu",
  "runtime_fingerprint_ref": "project.md#runtime-fingerprint",
  "change_scope": "kernel",
  "change_family": "triton-attention-dot-dtype",
  "sketch_ref": "rounds/sketch_002.json",
  "sketch_sha256": "c3c585d1f95337f25ac1c9ff5dc3c3591637b1e9a7c906174fb60d0da97695dd",
  "implementation_profile_snapshot_ref": "profile_snapshot/triton_gcu.yaml",
  "implementation_profile_snapshot_sha256": "8dfabd0af59b8f6640b47179fee19bca2f5fe35b18535a3db24f60c842e42b70",
  "project_capability_claim_ref": "profile_snapshot/capability_claim.json",
  "project_capability_claim_sha256": "a175f2727b9198a92da978aca9e8f87834a74884372746699412931890d9748e"
}
```

## Optimization Intent

```json
{
  "bottleneck_class": "device-bound",
  "intervention": "keep the ENTIRE r001-measured boundary unchanged — two-op forward host path (one torch.empty + one kernel launch, 1.00 aten op/call, exactly one topsModuleLaunchKernel submission, zero copies/memcpys/syncs), 4-arg run_out writing the caller buffer through the same kernel, grid (16,) = B*H programs, single-tile TP=128 (S=83 padded to power-of-2 128), bidirectional full attention (no causal), scale=0.125, direct strided [B,S,H*D] addressing, direct [2,83,512] fp16 stores, stateless module — and change ONLY the kernel's dot operand dtype + num_warps: QK^T feeds fp16 q/k DIRECTLY into tl.dot (fp16 x fp16 -> fp32 accumulator, NO fp16->fp32 widening cast — the Ixmma/MMA tensor-core path) with num_warps=1 (probe: nw1=129.0us optimal for the fp16 variant, vs nw2=143.6 / nw4=201.7); PV keeps the fp32 primary path (attn is the fp32 softmax result, v widened fp16->fp32 on load, fp32 x fp32 -> fp32 accumulator) with an fp16 fallback variant (attn .to(fp16), v stays fp16) declared but NOT shipped unless the primary path miscompiles or the fp16 PV variant proves faster under the pre-declared selection rule; diagnosis motivating this: r001's fp32 tl.dot version measured ~166us device (hand-written, TP=128 padding forces 58% FLOP waste) vs the CNNL SDPA library kernel ~158us floor — the fp16 QK^T dot removes the widening-pass register/ALU cost AND lowers the dot to the tensor-core MMA path, probe-measured 129.0us wall @ nw1 vs base 125.7us (only -2.6%, still below the +5% bar but a REAL device direction never formally validated in this campaign)",
  "allowed_changes": [
    "kernel QK^T dot operand dtype: q/k stay fp16 into tl.dot (fp16 x fp16 -> fp32 accumulator) — the widening cast `.to(tl.float32)` on q/k BEFORE the QK^T dot is REMOVED; this is the PRIMARY round-2 change",
    "kernel PV dot: PRIMARY keeps the fp32 path (attn fp32 softmax output, v widened to fp32 on load, fp32 x fp32 -> fp32 accumulator); a fallback fp16 variant (attn .to(fp16) + v kept fp16) is declared and shipped ONLY IF the fp16 PV variant correctness-passes AND beats the fp32 PV primary under the pre-declared selection rule — the shipped PV dtype is fixed as module-level literals (no runtime switching)",
    "kernel launch configuration: num_warps moves from r001's value 2 to 1 (probe: nw1=129.0us is the fp16-variant optimum; nw2=143.6 / nw4=201.7 degrade); num_warps is a fixed module-level literal, no runtime switching",
    "strictly NOT: no change to the host path (forward/run_out structure, op counts, allocation behavior), grid (16,), tile shape (single-tile TP=128), masking semantics, store layout, or module statelessness",
    "strictly NOT: no tl.dot site outside the power-of-2 envelope (TP=128, D=64 — NOT mult-of-16: 96=16x6 FAILS, only 16/32/64/128 pass); every dot keeps a same-dtype operand pair",
    "strictly NOT: no algorithm substitution (reduction.sum fallback BLOCKED — waiver NOT granted)",
    "strictly NOT: no torch.compile / no graphs / no capture / no runtime codegen / no precision-mode toggles",
    "strictly NOT: no runtime configuration switching in candidate source — one fixed shipped configuration, selected by the pre-declared rule and recorded in the binding statement"
  ],
  "invariants": [
    "correctness:pass under the unchanged comparator (allclose atol=1e-2 rtol=1e-2 equal_nan=True, seed 42) for the SHIPPED configuration; fp16 QK^T probe already correctness-PASS (max_abs_diff 1.46e-3 < 1e-2); mixed fp16-QK^T + fp32-PV also PASS (1.95e-3)",
    "single fp16 output [2,83,512]; bidirectional full-attention semantics with scale=0.125 unchanged",
    "public API unchanged: ModelNew(num_heads, head_size, num_kv_heads=8); forward(q,k,v); run_out(q,k,v,out) 4-arg per project.md public_contract",
    "run_out bitwise==forward on identical inputs (poisoned caller buffers x2, data_ptr preserved); forward bitwise-stable across repeat identical-input calls (deterministic per configuration)",
    "stateless: zero call-time instance state, zero caches, zero workspace",
    "capability legality: every tl.dot at power-of-2 tiles (TP=128, D=64) with same-dtype operands and fp32 accumulator; QK^T = fp16 x fp16 (no widen); PV = fp32 x fp32 (primary) or fp16 x fp16 (fallback, only if selected); tl.max/tl.sum WITHOUT keepdim (broadcast via [:,None]); tl.arange power-of-2 only; num_warps=1",
    "host-path invariance: aten cpu_ops stays <=2/call with exactly 1.00 topsModuleLaunchKernel submission, zero memcpys, zero graph submissions, zero model-code synchronizations (the r001-measured structure)",
    "AST-loader-safe module (safe-literal module constants; get_inputs/get_init_inputs retained); zero DANGER tokens (compile/capture) in candidate source"
  ],
  "expected_wall_improvement_pct": 0.0
}
```

## Unified Sketch

```json
{
  "artifact": "rounds/sketch_002.json",
  "sha256": "c3c585d1f95337f25ac1c9ff5dc3c3591637b1e9a7c906174fb60d0da97695dd",
  "rendering": "the computation boundary is structurally IDENTICAL to round 001 (16-program direct single-launch Triton full attention: batch x head parallel, single-tile TP=128 = S=83 padded to power-of-2, direct strided fp16 global loads, -inf masking on tile-padding columns for bidirectional no-causal softmax, direct [B,S,HD] fp16 stores, two-op host path, 4-arg run_out) — round 002 changes ONLY the kernel's dot operand dtype and num_warps: QK^T feeds fp16 q/k DIRECTLY into tl.dot (fp16 x fp16 -> fp32 accumulator, the fp16->fp32 widening cast REMOVED) with num_warps=1; PV keeps the fp32 primary path (attn fp32, v widened to fp32) with an fp16 fallback variant declared but not shipped unless it wins under the pre-declared selection rule"
}
```

## Host Plan

```json
{
  "applicability": "not-applicable",
  "reason": "kernel-only change (change_scope kernel): the host boundary is behaviorally identical to the r001-measured candidate — forward = one torch.empty + one kernel launch (1.00 aten op/call, exactly one topsModuleLaunchKernel submission), run_out = the same single launch into the caller buffer with zero allocations, stateless module, caller device and current stream preserved, no synchronization; every decision_001 Host Plan invariant (state_owner NOBODY, lifetime module-scoped, allocation_reuse NONE, no caches, concurrency-safe, default-stream discipline) carries forward unchanged; the only source difference is the kernel body's QK^T dot operand dtype (fp16, widening cast removed), the PV dot dtype (fp32 primary), and the num_warps constant (2 -> 1) at the single launch site — all kernel compilation configuration, not host-path logic"
}
```

## Evaluation Contract

```json
{
  "hypothesis_id": "H-002",
  "intervention": "keep the ENTIRE r001-measured boundary unchanged — two-op forward host path (one torch.empty + one kernel launch, 1.00 aten op/call, exactly one topsModuleLaunchKernel submission, zero copies/memcpys/syncs), 4-arg run_out writing the caller buffer through the same kernel, grid (16,) = B*H programs, single-tile TP=128 (S=83 padded to power-of-2 128), bidirectional full attention (no causal), scale=0.125, direct strided [B,S,H*D] addressing, direct [2,83,512] fp16 stores, stateless module — and change ONLY the kernel's dot operand dtype + num_warps: QK^T feeds fp16 q/k DIRECTLY into tl.dot (fp16 x fp16 -> fp32 accumulator, NO fp16->fp32 widening cast — the Ixmma/MMA tensor-core path) with num_warps=1 (probe: nw1=129.0us optimal for the fp16 variant, vs nw2=143.6 / nw4=201.7); PV keeps the fp32 primary path (attn is the fp32 softmax result, v widened fp16->fp32 on load, fp32 x fp32 -> fp32 accumulator) with an fp16 fallback variant (attn .to(fp16), v stays fp16) declared but NOT shipped unless the primary path miscompiles or the fp16 PV variant proves faster under the pre-declared selection rule; diagnosis motivating this: r001's fp32 tl.dot version measured ~166us device (hand-written, TP=128 padding forces 58% FLOP waste) vs the CNNL SDPA library kernel ~158us floor — the fp16 QK^T dot removes the widening-pass register/ALU cost AND lowers the dot to the tensor-core MMA path, probe-measured 129.0us wall @ nw1 vs base 125.7us (only -2.6%, still below the +5% bar but a REAL device direction never formally validated in this campaign)",
  "expected_causal_chain": [
    "kernel-only scope: the host path is behaviorally identical to the r001-measured candidate (aten <=2/call, 1.00 topsModuleLaunchKernel, zero memcpys/syncs/graphs) so the S60 launcher tax ~17.4us/call and host chain ~11us are INVARIANT to this round (observable triton_launcher_tax_per_call + aten_cpu_ops_per_call verify); the r001 device-bound diagnosis is unchanged: the wall is decided by whether D_cand (device) beats the CNNL SDPA ~158us floor",
    "fp16-dot mechanism: dropping the fp16->fp32 widening cast on q/k removes the register/ALU widening pass AND lowers the QK^T dot from the FFMA fp32 path to the tensor-core MMA path (the same units the CNNL SDPA library kernel rides); probe: fp16 QK^T 129.0us @ nw1 vs r001 fp32 166us device — a ~37us device cut that is REAL but still leaves D_cand ~119us, which does NOT clear the +5% wall bar (needs D_cand <= ~115us to beat base's device floor)",
    "num_warps=1: the fp16 variant is optimal at nw1 (129.0us), degrading at nw2 (143.6) and nw4 (201.7) — the opposite ordering from r001's fp32 variant (nw2 optimal); the shift is itself evidence the dot lowered to a different (tensor-core) execution path",
    "PV dtype: the primary keeps fp32 PV (attn is the fp32 softmax output; widening v is a single cast) because the fp16 PV fallback requires attn .to(fp16) which adds a cast AND risks a second-order precision loss on the softmax-normalized probabilities; the fallback is declared to guard against the primary path miscompiling or the fp16 PV variant measuring faster — selection is by the pre-declared rule, one fixed shipped config",
    "honest wall expectation: the probe shows -2.6% (129.0us vs 125.7us base) — still no-improvement territory; the S60 wall is noise-dominated (base fluctuates 125-171us), so the authoritative conclusion comes from harness 100x median + 3 interleaved pairs, and the round's REAL product is the first formal validation of the fp16-dot device direction in this campaign (correctness already probe-passed at 1.46e-3 / 1.95e-3 < 1e-2)"
  ],
  "primary_metric": {
    "name": "wall_time",
    "expected_improvement_pct": 5.0
  },
  "causal_graph": {
    "nodes": [
      "cn.fp16-dot",
      "cn.device-time-delta",
      "cn.wall-time"
    ],
    "edges": [
      ["cn.fp16-dot", "cn.device-time-delta"],
      ["cn.fp16-dot", "cn.wall-time"],
      ["cn.device-time-delta", "cn.wall-time"]
    ]
  },
  "mechanism_observables": [
    {
      "name": "wall_time_unrounded_paired_median_ms",
      "expectation": "honest two-sided: probe shows fp16 QK^T at -2.6% (129.0us vs 125.7us base) — below the +5.0% bar, so no-improvement is likely; the WIN branch requires the fp16 dot (plus PV-dtype selection) to bring D_cand below ~115us (beat base's ~158us device floor minus the TP=128 58% padding waste), which is arithmetically improbable; the authoritative reading comes from harness warmup 50 / repeat 100 / 3 interleaved pairs because S60 wall noise (base 125-171us) makes single-shot probes unreliable"
    },
    {
      "name": "dot_dtype_binding_audit",
      "expectation": "QK^T tl.dot operands are fp16 x fp16 with fp32 accumulator and ZERO widening cast on q/k; PV tl.dot operands are fp32 x fp32 (primary) or fp16 x fp16 (fallback, only if selected) with fp32 accumulator; every tl.dot at power-of-2 tiles (TP=128, D=64); tl.max/tl.sum WITHOUT keepdim; tl.arange power-of-2 only; num_warps=1; count of torch.compile/TORCHINDUCTOR/reduce-overhead strings = 0; zero .contiguous() in host paths"
    },
    {
      "name": "correctness_pass",
      "expectation": "allclose(atol=1e-2, rtol=1e-2, equal_nan=True, seed 42) vs base.py PASS for the SHIPPED configuration; probe already confirms fp16 QK^T max_abs 1.46e-3 and mixed fp16-QK^T+fp32-PV 1.95e-3 (both < 1e-2); the harness comparator must print PASS on every invocation"
    },
    {
      "name": "run_out_bitwise_equals_forward",
      "expectation": "bitwise equality over poisoned caller buffers x2 with data_ptr preserved; forward bitwise-stable across repeat identical-input calls (deterministic kernel, no atomics)"
    },
    {
      "name": "aten_cpu_ops_per_call",
      "expectation": "<= 2/call (one torch.empty + one Triton launch) — unchanged from r001; a material change falsifies the kernel-only scope claim"
    },
    {
      "name": "triton_launcher_tax_per_call",
      "expectation": "stays in the ~17.4us/call band (r001 measured) — confirming the dtype/num_warps change touched only the device term, not the host/launcher path"
    },
    {
      "name": "device_us_per_call",
      "expectation": "TWO-SIDED: D_cand inferred from wall - launch-API-time (GCU device-duration unavailable); the fp16 QK^T dot should move D_cand from r001's ~166us toward ~119us; a reading >= ~158us (base floor) confirms no headroom from dtype alone; a reading <= ~150us banks the fp16-dot device cut as a canonical, cross-operator fact even if the wall still loses to noise + padding waste"
    }
  ],
  "guardrails": [
    "correctness:pass under the unchanged comparator (allclose atol=1e-2 rtol=1e-2 equal_nan=True, seed 42) for the shipped configuration",
    "outputs remain single fp16 [2,83,512] tensors with bidirectional full-attention semantics",
    "stateless module: no instance attributes written at call time, no caches, no workspace; caller device and current stream preserved; no added synchronization",
    "run_out bitwise==forward for identical inputs; caller buffers never aliased; returns None",
    "cold first-call JIT compile stays outside timed medians (harness warmup 50 absorbs it)",
    "AST-loader-safe module composition (safe-literal module constants; retained defs)",
    "capability legality: QK^T dot fp16 x fp16 -> fp32 (no widen); PV dot fp32 x fp32 (primary) with fp32 accumulator; every dot at power-of-2 tiles (TP=128, D=64) with same-dtype operands; num_warps=1; tl.max/tl.sum without keepdim; tl.arange power-of-2 only; zero DANGER tokens (compile/capture) in candidate source",
    "no algorithm substitution: reduction.sum fallback stays BLOCKED (waiver NOT granted); the primary matrix.dot path is consumed at its probed signature, not replaced"
  ],
  "profiling_level": "targeted"
}
```

## Pitfalls and Anti-pattern Consultation

- anti-patterns.md consulted: no cataloged failure matches (the MLU selection-network entries are selection workloads; this operator is dense, selection-free, tie-free attention). The scalar-FMA lesson class (manual tl.sum matmul losing to vendor tensor cores) is excluded BY DESIGN: every QK^T/PV product stays a tl.dot at power-of-2 tiles — the round changes only the dot OPERAND dtype and num_warps, never the dot primitive; the reduction.sum substitution path stays BLOCKED (waiver NOT granted).
- S60 power-of-2 constraint (probe-backed, propagated from r001's verdict, MUST be honored by Coder): `tl.dot` AND `tl.arange` both require POWER-OF-2 (NOT mult-of-16 — 96=16x6 FAILS; only 16/32/64/128 pass). T=83 must pad to 128, so 58% FLOP waste is structurally unavoidable in the single-tile direct family. The r001 decision/verdict corrected the profile from the stale "mult-of-16" note; round 2 carries the corrected power-of-2 reading forward verbatim.
- S60 same-dtype dot constraint: `tl.dot` requires both operands SAME dtype. fp16 QK^T means q AND k both fp16 (drop the widening cast on BOTH). fp32 PV means attn AND v both fp32 (attn is already fp32 from the softmax; v is widened on load). The fp16 PV fallback requires attn `.to(tl.float16)` AND v kept fp16 — both casts/retentions must be symmetric or the dot miscompiles.
- S60 no-keepdim constraint: `tl.max` / `tl.sum` do NOT support `keepdim` on triton_gcu 3.6.0 — use `axis=1` then broadcast with `[:, None]`. Unchanged from r001.
- capability-probe discipline: Unknown does NOT mean unavailable. The fp16 QK^T dot was already probe-measured correctness-PASS (1.46e-3) and wall-timed (129.0us @ nw1) by the lead's preflight probe; round 2 formalizes it under the harness. The PV fallback (fp16 attn + fp16 v) is declared but the PRIMARY is fp32 PV — the fallback ships ONLY under the pre-declared selection rule, and its evidence is retained in the round log before any adoption. The frozen pins (profile 8dfabd0a / claim a175f272) are NOT edited mid-campaign.
- numerics: fp16 QK^T keeps the fp32 accumulator — expected error class ~1e-3 (probe 1.46e-3) against the 1e-2 tolerance; the fp32 PV primary keeps the softmax-normalized probabilities in fp32 (no second-order cast loss); the fp16 PV fallback would introduce a ~1e-3 additional cast on attn, which is why it is fallback-only. num_warps does not change the mathematics — each configuration is deterministic and bitwise-stable.
- same-family discipline (contract rule 6): round 2 is a DISTINCT change family (`triton-attention-dot-dtype`) from round 1's `triton-attention-dispatch-collapse` — the mechanism layer is the kernel's dot operand dtype + num_warps (device term; causal nodes cn.fp16-dot / cn.device-time-delta), while the host path is untouched; the new Verifier-backed observation naming this lever is report_001's device-bound diagnosis (hand fp32 tl.dot ~166us vs CNNL SDPA ~158us floor) plus the lead's probe (fp16 dot correctness-PASS and 129.0us wall) — a genuinely new device direction never formally validated in this campaign.
- DANGER notes for Coder binding statement: the shipped configuration is FIXED module-level literals (no runtime switching); QK^T dot operands fp16 x fp16 with zero widening cast on q/k; PV dot operands fp32 x fp32 (primary); num_warps=1; every tl.dot at power-of-2 (TP=128, D=64); tl.max/tl.sum no-keepdim; tl.arange power-of-2; zero compile/graph/capture strings; zero .contiguous(); stateless audit; run_out 4-arg signature.

## Rationale and Evidence

**Canonical anchors.** Reference pair unchanged (no accepted round since r000): baseline_adapter.py @1127e8d9… + rounds/report_000.md (Phase-0 baseline). Round-001 history (Verifier-grade, rounds/report_001.md @4ae2b613…-decision / verdict_001.json): wall −10.5% (0.276584 vs 0.305640 ms paired median — decisively below the +5.0% bar); diagnosis DEVICE-BOUND: hand-written fp32 tl.dot kernel ~166us device (TP=128 padding forces 58% FLOP waste) is SLOWER than the CNNL SDPA library kernel ~158us floor; S60 launcher tax ~17.4us/call (5x smaller than BI150's 84.77us — graph-replay has no prize here); num_warps sweep {1:167.9, 2:166.2, 4:193.2, 8:286.1} — nw2 optimal for the fp32 variant; capability correction written back: tl.dot AND tl.arange both POWER-OF-2 (not mult-of-16). Deliverable banked @f2f8b9b6… at ~0.906x. Streak 1/3; round budget 1/20.

**Why the fp16-dot family now.** Round 1 closed the dispatch-collapse family: the host collapse ENGAGED (aten 28 → 2) but the wall is decided by the DEVICE, and the fp32 tl.dot device time (~166us) loses to the CNNL SDPA library kernel (~158us). The remaining live device levers are: (a) dot operand dtype (fp16 unlocks the tensor-core MMA path and removes the widening pass), and (b) padding reduction (structurally blocked: T=83 → TP=128 is forced by the power-of-2 constraint; 96=16×6 FAILS). The lead's preflight probe established the fp16 QK^T dot is correctness-PASS (1.46e-3 < 1e-2) and faster (129.0us @ nw1 vs 166us fp32), a ~37us device cut that is REAL and never formally validated in this campaign. Round 2 makes that validation authoritative under the harness.

**Why PROCEED with expected_wall_improvement_pct 0.0 (declared honestly).** (1) The probe shows −2.6% (129.0 vs 125.7us base) — still below the +5.0% adoption bar; the honest expectation is no-improvement #2. (2) DELIVERABLE RULE (binding, project.md): the campaign's PRIMARY contractual product is the best correctness-PASS Triton submission — round 2 can produce an improved deliverable (fp16 dot) that, even at ~0.93-0.97x, is a strictly better banked candidate than r001's fp32 ~0.906x. (3) Information: the round canonizes the fp16-dot device cut (or its absence) under harness 100× median + 3 interleaved pairs — a cross-operator fact for the flexattention/groupedtopk lineage, and the definitive answer on whether the dot dtype lever can beat the CNNL SDPA floor on S60. (4) Falsifiability: the device-time observable is genuinely two-sided — if the fp16 dot (plus PV-dtype selection) brings D_cand below ~150us, the device cut is banked even if the wall still loses to noise + padding waste; either outcome attributes.

**Change-scope justification.** This is a kernel-only change: the host path, grid, tile shape, masking, store layout, and module statelessness are byte-identical to the r001-measured candidate; the ONLY differences are the QK^T dot operand dtype (fp16, widening removed), the PV dot dtype (fp32 primary), and num_warps (2 → 1). The mechanism is singular and observable (dot_dtype_binding_audit vs wall_time vs device_us_per_call), satisfying the one-attributable-change requirement; host invariance is itself an observable (aten count, launch census, launcher-tax band).

**Artifacts consulted.** project.md (identity, DELIVERABLE RULE, public_contract, runtime fingerprint); rounds/report_000.md (baseline + 2-launch census); rounds/report_001.md + verdict_001.json + decision_001.md + sketch_001.json (r001 canonical evidence: device-bound diagnosis, launcher tax 17.4us, D_cand ~166us, power-of-2 correction); baseline_adapter.py @1127e8d9…; ../../base.py @86ac5703…; profile_snapshot/triton_gcu.yaml @8dfabd0a… and capability_claim.json @a175f272… (frozen envelope: matrix.dot fp16-fp16-fp32 constrained power-of-2, num_warps 1/2/4/8); the lead's round-2 preflight probe conclusions (fp16 QK^T correctness-PASS 1.46e-3 / wall 129.0us @ nw1 / mixed 1.95e-3; num_warps fp16 {1:129.0, 2:143.6, 4:201.7}); skills references invariants.md, anti-patterns.md, bottleneck-judgment.md, decision-template.md; state/designer_context.md.

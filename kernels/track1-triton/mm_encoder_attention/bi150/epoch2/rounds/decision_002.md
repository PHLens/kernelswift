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
  "change_scope": "kernel",
  "change_family": "triton-attention-kernel-config",
  "sketch_ref": "rounds/sketch_002.json",
  "sketch_sha256": "c16b1528b25ae1a3bbfc72b3e459462505d940677e62b30a0585e3b41b46e9e9",
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
  "intervention": "keep the ENTIRE r001-measured boundary unchanged — two-op forward host path (one torch.empty + one kernel launch, 1.00 aten op/call, exactly one cuLaunchKernel-class submission, zero copies/memcpys/syncs), 4-arg run_out writing the caller buffer through the same kernel, grid (48,) = 16 pairs x 3 q-tiles of 32, BM=BN=32 tiles with D=64 as two 32-chunks, online running-max fp32 softmax over 3 sequential key-tiles with -inf masking only on S=83 tile padding, direct strided addressing of the [B,S,H*D] views, direct [2,83,512] fp16 stores, stateless module — and change ONLY the kernel's EXECUTION CONFIGURATION, probe-gated BEFORE adoption: a Verifier-owned capability sweep over num_warps in {1,2,4} x dot-operand in {fp32-widened (r001-proven envelope), fp16-with-fp32-accumulator (Ixmma MMA path, currently unproven on this rig)} at the unchanged (32,32)@(32,32) tile shape measures per-config compilation, numerical exactness vs an fp32 reference on the seed42 target suite + fp16-extreme suite + non-target shapes B1S41/B2S96, bitwise-stability under repeat calls, and device time at the target shape; the candidate SHIPS the fastest exactness-passing configuration under the pre-declared selection rule (ties broken toward fewer new capabilities; only-the-r001-config-passes => ship it and record the no-headroom reading); diagnosis motivating the sweep: at num_warps=1 the kernel holds ~9 live (32,32) fp32 register tiles ~= 288 registers/thread — beyond the 255 budget, spill-class — while num_warps=4 quarters per-thread pressure and adds 4x warp-level latency hiding at unchanged grid, and fp16 operands halve global load traffic while unlocking the same Ixmma MMA units the 17.39 us/call vendor kernel rides; the round's product is the D_cand cut from the measured 28.203 us/call baseline that re-prices the F2 graph-composition gate (parity at D <= ~16.5, win at D <= ~9.2) and the probe-qualified capability evidence — NOT a wall win: with T_launcher = +84.765 us/call measured, the direct family needs D_cand <= -75.7 us to clear +5%, arithmetically impossible",
  "allowed_changes": [
    "kernel launch configuration: num_warps moves from the r001-proven value 1 to a value in {2,4} ONLY IF the pre-adoption capability sweep shows that configuration exactness-passing; the shipped value is fixed as a module-level literal (no runtime switching)",
    "dot operand dtype: tl.dot operands at the unchanged (32,32)@(32,32) sites move from fp32-widened to fp16 ONLY IF the pre-adoption capability sweep shows fp16-operand dots exactness-passing with the fp32 accumulator; loads may then stay fp16 to the dot (halved load traffic); the fp32 accumulator and all softmax state stay fp32",
    "strictly NOT: no change to the host path (forward/run_out structure, op counts, allocation behavior), grid, tile shapes, masking semantics, store layout, or module statelessness",
    "strictly NOT: no dot site outside the (32,32) tile shape, no non-fp32 accumulator, no num_stages, no larger tiles",
    "strictly NOT: no algorithm substitution (reduction.sum BLOCKED — waiver NOT granted)",
    "strictly NOT: no torch.compile / no graphs / no capture / no runtime codegen / no precision-mode toggles",
    "strictly NOT: no runtime configuration switching in candidate source — one fixed shipped config, selected by the pre-declared probe rule and recorded in the binding statement"
  ],
  "invariants": [
    "correctness:pass under the unchanged comparator (allclose atol=1e-2 rtol=1e-2 equal_nan=True, seed 42) for the SHIPPED configuration on every suite: seed42 canonical, fp16-extreme (vs fp32 ground truth, r001-established basis), non-target shapes B1S41/B2S96",
    "single fp16 output [2,83,512]; bidirectional full-attention semantics with scale=0.125 unchanged",
    "public API unchanged: ModelNew(num_heads, head_size, num_kv_heads=8); forward(q,k,v); run_out(q,k,v,out) 4-arg per project.md public_contract",
    "run_out bitwise==forward on identical inputs (poisoned caller buffers x2, data_ptr preserved); forward bitwise-stable across repeat identical-input calls (deterministic per configuration)",
    "stateless: zero call-time instance state, zero caches, zero workspace",
    "capability legality: every tl.dot at (32,32)@(32,32) with fp32 accumulator; operand dtype = fp32-widened OR probe-qualified fp16 (sweep evidence in the round log BEFORE adoption); num_warps = the probe-qualified shipped value; num_stages unset",
    "host-path invariance: aten cpu_ops stays <=3/call with exactly 1.00 cuLaunchKernel-class submission, zero memcpys, zero graph submissions, zero model-code synchronizations (the r001-measured structure)",
    "AST-loader-safe module (safe-literal module constants; get_inputs/get_init_inputs retained); zero DANGER tokens (compile/capture) in candidate source"
  ],
  "expected_wall_improvement_pct": 0.0
}
```

## Unified Sketch

```json
{
  "artifact": "rounds/sketch_002.json",
  "sha256": "c16b1528b25ae1a3bbfc72b3e459462505d940677e62b30a0585e3b41b46e9e9",
  "rendering": "the computation boundary is structurally IDENTICAL to round 001 (48-program direct single-launch Triton full attention: batch x head x q-tile parallel, 3 sequential key-tiles for the online softmax dependency, direct strided fp16 global loads, two-half D processing, -inf masking only on S=83 tile padding, direct [B,S,HD] fp16 stores, two-op host path, 4-arg run_out) — round 002 changes ONLY the kernel's execution configuration: the dot sites' operand dtype (fp32-widened or probe-qualified fp16, fp32 accumulator, (32,32) tiles unchanged) and num_warps (1, 2, or 4, probe-qualified), selected by the pre-adoption capability sweep under the pre-declared fastest-exactness-passing rule"
}
```

## Host Plan

```json
{
  "applicability": "not-applicable",
  "reason": "kernel-only change (change_scope kernel): the host boundary is behaviorally identical to the r001-measured candidate — forward = one torch.empty + one kernel launch (1.00 aten op/call, exactly one cuLaunchKernel-class submission), run_out = the same single launch into the caller buffer with zero allocations, stateless module, caller device and current stream preserved, no synchronization; every decision_001 Host Plan invariant (state_owner NOBODY, lifetime module-scoped, allocation_reuse NONE, no caches, concurrency-safe, default-stream discipline) carries forward unchanged; the only source difference is the kernel body's dot operand dtype and the num_warps constant at the single launch site — both are kernel compilation configuration, not host-path logic"
}
```

## Evaluation Contract

```json
{
  "hypothesis_id": "H-002",
  "intervention": "keep the ENTIRE r001-measured boundary unchanged — two-op forward host path (one torch.empty + one kernel launch, 1.00 aten op/call, exactly one cuLaunchKernel-class submission, zero copies/memcpys/syncs), 4-arg run_out writing the caller buffer through the same kernel, grid (48,) = 16 pairs x 3 q-tiles of 32, BM=BN=32 tiles with D=64 as two 32-chunks, online running-max fp32 softmax over 3 sequential key-tiles with -inf masking only on S=83 tile padding, direct strided addressing of the [B,S,H*D] views, direct [2,83,512] fp16 stores, stateless module — and change ONLY the kernel's EXECUTION CONFIGURATION, probe-gated BEFORE adoption: a Verifier-owned capability sweep over num_warps in {1,2,4} x dot-operand in {fp32-widened (r001-proven envelope), fp16-with-fp32-accumulator (Ixmma MMA path, currently unproven on this rig)} at the unchanged (32,32)@(32,32) tile shape measures per-config compilation, numerical exactness vs an fp32 reference on the seed42 target suite + fp16-extreme suite + non-target shapes B1S41/B2S96, bitwise-stability under repeat calls, and device time at the target shape; the candidate SHIPS the fastest exactness-passing configuration under the pre-declared selection rule (ties broken toward fewer new capabilities; only-the-r001-config-passes => ship it and record the no-headroom reading); diagnosis motivating the sweep: at num_warps=1 the kernel holds ~9 live (32,32) fp32 register tiles ~= 288 registers/thread — beyond the 255 budget, spill-class — while num_warps=4 quarters per-thread pressure and adds 4x warp-level latency hiding at unchanged grid, and fp16 operands halve global load traffic while unlocking the same Ixmma MMA units the 17.39 us/call vendor kernel rides; the round's product is the D_cand cut from the measured 28.203 us/call baseline that re-prices the F2 graph-composition gate (parity at D <= ~16.5, win at D <= ~9.2) and the probe-qualified capability evidence — NOT a wall win: with T_launcher = +84.765 us/call measured, the direct family needs D_cand <= -75.7 us to clear +5%, arithmetically impossible",
  "expected_causal_chain": [
    "kernel-only scope: the host path is behaviorally identical to the r001-measured candidate (aten 1.00/call, 1.00 cuLaunchKernel, zero memcpys/syncs/graphs) so T_launcher = +84.765 us/call is INVARIANT to this round (observable triton_launcher_tax_invariance verifies); wall arithmetic stays wall_cand = 145.375 + 84.765 + (D_new - 17.39) = 212.75 + D_new us — a wall win needs D_new <= -75.7 us and is arithmetically impossible; the declared expectation is honestly 0.0 and the expected wall band at D_new in [9.2, 28.2] is 222-241 us (-53% to -66%)",
    "register-occupancy edge: num_warps=1 holds ~9 live (32,32) fp32 register tiles (q_lo/hi, k_lo/hi, v_lo/hi, s, acc_lo/hi) ~= 288 regs/thread against the 255 budget — spill-class; num_warps=2/4 drops per-thread tile pressure to ~144/~72 regs (spills gone) and multiplies warp-level latency hiding 2-4x at unchanged grid (48 programs -> 96/192 warps over 16 SMs, healthy occupancy)",
    "mma-path edge: fp32 tl.dot lowers to FFMA (no tensor core) — 37.7 MFLOP of dot work at ~1.3 TFLOP/s effective is single-digit percent of peak; fp16-operand tl.dot with fp32 accumulator lowers to the Ixmma MMA path — the SAME device units the 17.39 us/call vendor kernel rides — plus fp16 operands kept to the dot halve Q/K/V global load traffic",
    "pre-declared device bands (two-sided): num_warps=4 + fp32 dots: D 28.2 -> ~14-18 us; num_warps 1-2 + fp16 dots: ~18-24 us; combined num_warps=4 + fp16 dots: ~10-16 us; NO configuration passing below 25 us => the no-headroom reading (F3 closed for good; r003 = F2 composition at the 0.93x-class projection or honest close-out)",
    "the round's products: (a) probe-qualified capability evidence for num_warps>1 and fp16-operand dots on this rig — never measured anywhere in this operator lineage, cross-operator value for the flexattention/groupedtopk siblings; (b) the D_cand cut that re-prices the F2 composition: net_F2 = D_new - 16.455 us on the report_001 basis (parity-class deliverable at D_new <= ~16.5, adoption-grade win at D_new <= ~9.2); (c) an improved direct deliverable (0.603x -> ~0.64x at D=15); the F2 composition follows as round 003 under the staged sequencing unless the sweep closes everything"
  ],
  "primary_metric": {
    "name": "wall_time",
    "expected_improvement_pct": 5.0
  },
  "causal_graph": {
    "nodes": [
      "cn.kernel-config",
      "cn.register-occupancy",
      "cn.mma-path",
      "cn.device-time-delta",
      "cn.wall-time"
    ],
    "edges": [
      ["cn.kernel-config", "cn.register-occupancy"],
      ["cn.kernel-config", "cn.mma-path"],
      ["cn.kernel-config", "cn.device-time-delta"],
      ["cn.register-occupancy", "cn.device-time-delta"],
      ["cn.mma-path", "cn.device-time-delta"],
      ["cn.device-time-delta", "cn.wall-time"],
      ["cn.kernel-config", "cn.wall-time"]
    ]
  },
  "mechanism_observables": [
    {
      "name": "wall_time_unrounded_paired_median_ms",
      "expectation": "at least 5% below the same-session accepted reference median (0.145375 ms class) across interleaved pairs at warmup 50 / repeat 100 — honestly declared unreachable: expected band 222-241 us (-53% to -66%) at D_new in [9.2, 28.2]; the observable stays in the contract as the adoption criterion the F2 composition round will re-attack with the improved D_cand"
    },
    {
      "name": "kernel_config_capability_probe",
      "expectation": "the pre-adoption qualification sweep (Verifier-owned, runs BEFORE the candidate is adopted): 6 configurations (num_warps {1,2,4} x dot-operand {fp32-widened, fp16@fp32acc}) at the unchanged (32,32) tile shape, each measured for compilation, exactness vs fp32 reference (<= 1e-2 on seed42 target + fp16-extreme + B1S41/B2S96; expect <= ~1e-3 for passing configs), bitwise-stability under repeat calls, and device time at the target shape; pre-declared readings: (i) any out-of-envelope config exactness-passes AND beats 28.203 us => ship the fastest (selection rule honored); (ii) only the r001 config passes => no-headroom reading, ship r001 config, F3 permanently closed; (iii) fp16 dots fail compilation or exactness => Ixmma-MMA-via-Triton recorded capability-negative for this lineage, num_warps sweep still ships if it passes; probe evidence retained in the round log (log/probes/) — the frozen profile pins (dc8fa4c0/aeba3a87) are NOT edited mid-campaign; profile promotion is an end-of-campaign act"
    },
    {
      "name": "device_us_per_call",
      "expectation": "D_new for the shipped configuration vs the 28.203 us/call r001 baseline: >= 25 us no headroom (F3 closed); 18-25 us partial (F2 net = D - 16.455 stays +1.5 to +8.5 us WORSE — sub-parity 0.94-0.96x composed deliverable); <= ~16.5 us parity-class unlocked (F2 net <= 0); <= ~9.2 us win-class unlocked (F2 net <= -7.27 = the +5% adoption bar); all bands attribute via the sweep's per-config device times"
    },
    {
      "name": "aten_cpu_ops_per_call",
      "expectation": "<= 3/call (single aten::empty expected) — unchanged from r001; a material change falsifies the kernel-only scope claim"
    },
    {
      "name": "launch_and_submission_count_per_call",
      "expectation": "exactly 1.00 kernel launch (cuLaunchKernel-class) per call, ZERO memcpys, ZERO graph submissions, ZERO model-code synchronizations — the r001 structural guarantee carries over unchanged"
    },
    {
      "name": "run_out_bitwise_equals_forward",
      "expectation": "bitwise equality over poisoned caller buffers x2 with data_ptr preserved, both surfaces, for the SHIPPED configuration; forward bitwise-stable across repeat identical-input calls (deterministic per configuration)"
    },
    {
      "name": "capability_legality_binding_audit",
      "expectation": "every tl.dot call site at (32,32)@(32,32) with fp32 accumulator and the SHIPPED operand dtype (fp32-widened or probe-qualified fp16 — sweep evidence hash-linked in the round log); num_warps equals the shipped probe-qualified value; num_stages absent; zero torch.compile/TORCHINDUCTOR/reduce-overhead/graph strings; zero .contiguous() calls; stateless audit; the binding statement records the shipped configuration and its probe evidence pointer"
    },
    {
      "name": "triton_launcher_tax_invariance",
      "expectation": "T_launcher for the shipped candidate stays in the +80 to +90 us/call band (r001 measured +84.765; sibling prior +86-89) — confirming the configuration change touched only the device term; a material shift falsifies the kernel-only scope claim and re-opens host-side attribution"
    }
  ],
  "guardrails": [
    "correctness:pass under the unchanged comparator (allclose atol=1e-2 rtol=1e-2 equal_nan=True, seed 42) for the shipped configuration on every suite (seed42 canonical, fp16-extreme vs fp32 ground truth, non-target shapes)",
    "outputs remain single fp16 [2,83,512] tensors with bidirectional full-attention semantics",
    "the capability sweep runs BEFORE adoption and its evidence is retained in the round log; only sweep-exactness-passing configurations may ship; the selection rule (fastest exactness-passing, ties toward fewer new capabilities, all-fail => r001 config) is pre-declared and honored",
    "run_out bitwise==forward for identical inputs; caller buffers never aliased; returns None",
    "stateless module: no instance attributes written at call time, no caches, no workspace; caller device and current stream preserved; no added synchronization",
    "host-path invariance: aten <= 3/call, exactly one kernel submission, zero memcpys/graphs/syncs (r001 structure)",
    "cold first-call JIT compile stays outside timed medians (harness warmup 50 absorbs it)",
    "AST-loader-safe module composition (safe-literal module constants; retained defs)",
    "no algorithm substitution: reduction.sum stays BLOCKED (waiver NOT granted); the primary matrix.dot path is consumed at its probed signature, not replaced",
    "zero DANGER tokens (compile/capture) in candidate source; frozen profile/claim pins unchanged mid-campaign"
  ],
  "profiling_level": "targeted"
}
```

## Pitfalls and Anti-pattern Consultation

- anti-patterns.md consulted: no cataloged failure matches (selection-network entries are selection workloads; this operator is dense, selection-free, tie-free). The scalar-FMA lesson class is excluded BY DESIGN: every QK^T/PV product stays a tl.dot at (32,32) — the round changes only dot OPERAND dtype and warp count, never the dot primitive; the reduction.sum substitution path stays BLOCKED (waiver NOT granted).
- capability-probe discipline (the core risk of this round): Unknown does NOT mean unavailable — the sweep tests before use, and every failure mode is pre-declared: (i) all out-of-envelope configs fail => ship the r001 config, record the no-headroom reading, F3 permanently closed (the round still banks the capability resolution); (ii) fp16 dots compile-fail or miss exactness => Ixmma-MMA-via-Triton recorded capability-negative for this lineage (cross-operator fact), num_warps sweep still ships if it passes; (iii) num_warps>1 miscompiles at this tile shape => same closure on the occupancy lever. The probe runs BEFORE the candidate is adopted (candidate-ready gating per the frozen profile's constrained/Unknown statuses); the frozen pins (profile dc8fa4c0 / claim aeba3a87) are NOT edited mid-campaign — probe evidence lives in the round log and any profile promotion is an end-of-campaign Orchestrator act.
- numerics: fp16-operand dots keep the fp32 accumulator — expected error class ~1e-3 against the 1e-2 tolerance; the extreme-suite check uses the fp32-ground-truth basis established in r001 (where the fp32-widened candidate matched to 3.052e-05 while the VENDOR kernel itself diverges to 1457 — the fp16-dot candidate is expected to sit between; the 1e-2 exactness threshold governs qualification, and the r001 fp32-path reading is the reference floor). num_warps does not change the mathematics — each configuration is deterministic and bitwise-stable; the sweep checks this per config.
- same-family discipline (contract rule 6): round 002 is a DISTINCT change family from round 001's triton-attention-dispatch-collapse — the mechanism layer is the kernel's execution configuration (device term; causal nodes cn.register-occupancy / cn.mma-path / cn.device-time-delta) while the host path is untouched; the new Verifier-backed observation naming this lever is report_001's own gate table (D_cand = 28.203 us/call with F2 parity at ~16.5 / win at ~9.2 — the device term is the only live lever on every remaining outcome).
- register-pressure arithmetic for Coder awareness (diagnosis, not a guarantee): ~9 live (32,32) fp32 tiles at num_warps=1 is ~288 regs/thread vs the 255 budget — spill-class; at num_warps=4 it is ~72 regs/thread. The sweep MEASURES whether the diagnosis converts to device time; no expected lowering is claimed as guaranteed (the profile only supports a probe).
- DANGER notes for Coder binding statement: the shipped configuration is FIXED module-level literals (no runtime switching); zero compile/graph/capture strings; dot-site audit vs the SHIPPED operand dtype REQUIRED; num_warps equals exactly the shipped value; num_stages absent; zero .contiguous(); stateless audit; run_out 4-arg signature.

## Rationale and Evidence

**Canonical anchors.** Reference pair unchanged (no accepted round since r000): baseline_adapter.py @c3980a2c… + rounds/report_000.md @20b21646…. Round-001 history (Verifier-grade, rounds/report_001.md @13adafe9…, verdict_001.json @b6e62fdb…): wall −65.7458% (0.240953 vs 0.145375 ms paired median — inside the decision's pre-declared 0.235–0.29 band, reading (b)); T_launcher = +84.7651 µs/call net at bsz=2 (the sibling +86–89 prior transfers in full); D_cand = 28.2030 µs/call (_mm_encoder_attn_fwd; vendor Ixmma 17.3901 whole-trace / 15.6853 attributed); aten 33 → 1.00/call; exactly 1.00 cuLaunchKernel; correctness PASS on every suite incl. the extreme suite where the candidate beats the vendor's own numerics (3.052e-05 vs fp32 ground truth vs vendor divergence 1457); deliverable banked @4171de8d at 0.603x per the project DELIVERABLE RULE. Streak 1/3; round budget 1/20 — rounds are plentiful, the streak is the binding budget.

**The in-envelope question (dispatch's explicit ask), answered NO.** No in-envelope-legal path plausibly halves D_cand: (i) grid (48,) is already maximal in-envelope — 16 (batch,head) pairs x 3 q-tiles is every parallelizable index; the kv-tile loop cannot parallelize (online-softmax running-state dependency) and the D-split cannot either (softmax couples the halves); (ii) tile shapes are pinned at (32,32) by the frozen fp32 dot envelope; (iii) num_warps is pinned at 1 by policy; (iv) every in-envelope restructuring retains ~288 regs/thread spill pressure and the FFMA lowering. The only levers on D_cand are out-of-envelope — num_warps>1 (occupancy) and fp16-operand dots (Ixmma MMA path + halved load traffic) — hence the probe-gated sweep.

**Why the F3 kernel-config round and not F2-now.** The report_001 projection at measured numbers: graph-family net = D_cand − 16.455 µs/call ⇒ +11.75 µs WORSE at D=28.203 (0.93x-class deliverable). Composing BEFORE the device cut forfeits the upside one configuration round can unlock: parity-class at D ≤ ~16.5 (a ~1.00x submission — the sibling campaign's terminal deliverable class) and adoption-grade win at D ≤ ~9.2 (+5% with streak reset). Staged sequencing (config round -> composition round) strictly dominates on terminal-deliverable quality in every branch: no-cut (0.93x either way), cut-to-16.5 (1.00x vs 0.93x), cut-to-9.2 (WIN vs 0.93x). It also banks the capability resolution (num_warps>1, fp16 dots) that no other round would produce — evidence with cross-operator value for the flexattention/groupedtopk lineage.

**Why PROCEED with expected_wall_improvement_pct 0.0 (declared honestly).** Same three-part justification as round 001, now sharpened by measurement: (1) the wall win is arithmetically closed for the direct family (T_launcher + D_cand ≤ 9.03 µs impossible; this round needs D_new ≤ −75.7 µs) — the honest expectation is a large regression band (222–241 µs) and the primary_metric stays the 5.0% bar the NEXT round re-attacks; (2) the round's products are real and measured: the D_cand cut that re-prices F2, the capability qualification, and an improved direct deliverable (0.603x → ~0.64x at D=15); (3) abort would strand the deliverable at 0.603x while a 0.93–1.05x submission is 1–2 rounds away — the DELIVERABLE RULE makes the submission the campaign's primary contractual product, and the abort form is reserved for "no deliverable-bearing work remains", which is false here.

**Streak calculus (honest).** Round 002 is expected no-improvement #2 (direct-family wall arithmetic is closed regardless of D_new). The final bullet (round 003) is then the F2 graph composition — which improves the deliverable in EVERY branch (0.93x at D=28.2 unchanged; ~1.00x at D≈16.5; ≥+5% adoption-grade WIN at D ≤ 9.2 with streak reset) — or, if the sweep closes F3 AND the composition is declined on its projection, an honest close-out with the banked deliverable and the complete physics map (T_launcher, D_cand, R-term transfer, capability matrix — all canonically measured). This is the sibling trajectory with strictly better instrumentation.

**One-attributable-change compliance.** Single mechanism: the kernel execution configuration (one knob-set: num_warps x dot-operand dtype), selected by a pre-declared rule from probe evidence; the host path is untouched and its invariance is itself an observable (aten count, submission census, T_launcher band); the probe is qualification evidence, not a second mechanism — per-config device times attribute the D_cand delta knob-by-knob.

**Artifacts consulted.** rounds/report_001.md @13adafe9… (canonical r001 evidence: T_launcher, D_cand, F2 gate table, +11.75 projection); rounds/verdict_001.json @b6e62fdb…; rounds/report_000.md @20b21646… + baseline_adapter.py @c3980a2c… (reference pair); triton_mm_encoder_attention_e2_001.py @4171de8d… (banked deliverable = the config-sweep base kernel); rounds/decision_001.md @67b96739… + rounds/sketch_001.json @a1c27dba… (the unchanged boundary this round configures); ../../base.py @86ac5703…; profile_snapshot/triton_cuda.yaml @dc8fa4c0… (constrained matrix.dot (32,32) fp32; resource.num-warps policy-only-1; EMPTY probe_catalog — qualification runs as Verifier round evidence, not a catalog probe); profile_snapshot/capability_claim.json @aeba3a87… (primary matrix.dot fp16 signature at this exact operator shape; fallback reduction.sum before-fallback, waiver NOT granted); skills/kernel-opt-loop/scripts/run_profile_probe.py + validate_probe.py (probe mechanics read); auto_bench.py @71fb3ad0…; team-state.md @5e22c0e2… (streak 1/3, budget 1/20, dispatch_next_round true); references/invariants.md, anti-patterns.md, bottleneck-judgment.md, decision-template.md; sibling flexattention/bi150/epoch2 decision_003.md @<read> (staged-composition precedent and graph-round template for the planned round 003); state/designer_context.md (Phase-0 model + r001 updates).

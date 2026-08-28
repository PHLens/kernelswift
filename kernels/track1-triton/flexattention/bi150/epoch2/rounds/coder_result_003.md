# Coder Result 003

Result: candidate-ready

## Identity

- Round: `003`
- Decision: `rounds/decision_003.md` @`d4f7203e9a032a40eb0164eeb515a8a0be31c9e5067e2a80036af4344affb203` (hash re-verified)
- Sketch: `rounds/sketch_003.json` @`4ef267b9bb67f8abc52889684412336785b4281612647f55efbacdc29f8dc6f0` (hash re-verified)
- Candidate: `triton_flexattention_e2_003.py` @`6ffb0c94bf6b126317acddcf14119bfd27fab5709c20a1f33cfdf8883d58bf1e` (15316 bytes; unchanged after first gate across ALL gates/probes)
- Canonical kernel source: `triton_flexattention_e2_002.py` @`570bc2be…12b1` — the `@triton.jit` kernel block is BYTE-IDENTICAL (machine-verified extraction-diff); only `_RECAPTURE_BUDGET = 4` added at module level
- Harness: `auto_bench.py` @`71fb3ad0…` (untouched; phase structure source-verified: `compare_case` L715-739 — ONE correctness call via `run_forward` L449-456 (clones per call), then `time_forward` L459-475 reusing ONE inputs list; `v1_inputs = clone_value(v0_inputs)` L732 makes the timing pointer set distinct from the correctness clone)
- Binding statement: `log/probes/binding_statement_report_003.json` @`f8be3a6b68f080e39f5a0b772b82f541fc590e37708df0aa8a2dfe04e956a7c1`
- Runtime: torch 2.7.1, BI-V150 sm71 mp16, CoreX bootstrap in every shell, cuda:0

## Implementation Summary (decision-003 exact)

- tier-1 direct-address replay: the r002 kernel launch captured ONCE per first-seen pointer set as a manual `torch.cuda.CUDAGraph` bound to the CALLER'S OWN q/k/v pointers (ZERO copy-ins — pointer match means the replayed kernel reads live caller bytes) writing static `out_ws`; warmup-3-then-capture order on a dedicated side stream (JIT specialization frozen BEFORE the capture window); captured region = exactly ONE kernel launch (no branches/prints/host reads/allocations).
- Per served call: 3× data_ptr guard + ONE replay + one copy-out into a fresh invocation-owned buffer (forward) or the caller buffer (run_out) — 2 GPU submissions/call, zero model-code sync/query beyond data_ptr reads.
- Recapture bounded: initial binding budget-free; rebinds consume an irreversible budget of 4; first-seen sets only (`bound_sets` ≤5-entry cache-key history, guard-only use); same-set revisits route tier-2; overflow routes tier-2.
- tier-2 copy-in replay: r001-proven machinery (q_in/k_in/v_in + 3 copy-ins + ONE replay + copy-out), lazy-built, bitwise-identical.
- tier-3 eager: r002 direct-launch path for non-target regimes and any replay failure.
- Permanence: `direct_replay_failed`/`copyin_replay_failed` monotone ≤once each; triggering call always served correctly by the next tier; all tiers bitwise-equal for identical bits.
- Results NEVER returned from graph-resident memory (forward = fresh `torch.empty_like`; run_out fills the caller buffer; `return self.out_ws`-pattern machine count 0).

## Attempt Ledger

| Attempt | Command (abridged) | Exit | Defect | Candidate SHA |
|---|---|---|---|---|
| 1 | authoring; kernel docstring restored to r002 byte-identity pre-gate | — | none (candidate) | `6ffb0c94…bf1e` final |
| 1 | ast.parse + DANGER/sync/count audits + kernel byte-identity | 0 | none — DANGER 0; sync/query 0; graph-resident-return 0; num_warps 1 site =1; 4 dot sites; JIT kernel identical | `6ffb0c94…bf1e` |
| 1 | harness smoke `--warmup 5 --repeat 10` | 0 | none — `PASS accuracy; v0=0.156259, v1=0.149196, 1.047x` | `6ffb0c94…bf1e` |
| 1–3 | probes p13–p19 | 0 | PROBE-SIDE only: p14 lambda-syntax + f4 expectations collected outside mutation loop; p15 revisit vector fed CLONES (new pointer set) instead of original tensors; p16/p17/p18 CUDAGraph fault wrappers lacked capture protocol (must SUBCLASS) and p16's replay patch was class-wide — fixed to instance-identity-scoped; p19 missing helper. CANDIDATE UNTOUCHED | `6ffb0c94…bf1e` |

No Verifier repair requests yet; zero same-round candidate repairs.

## Decision-scoped Checks (log/probes/ — non-authoritative, no timing/profilers)

| Probe | Verdict | Key evidence |
|---|---|---|
| p13 cold-capture smoke + warmup-absorption accounting | PASS 0 | harness-like sequence (1 correctness call on clone pointers → 5 warmup → 10 timed): initial binding budget-free; EXACTLY one recapture in warmup (budget 4→3); timed segment 10/10 replays, ZERO python launcher executions, zero captures inside timed segment; bitwise==r002 twin on live bytes every call; tier-2 never needed |
| p14 capture-fired multi-fact | PASS 0 | f1 live handle; f2 static out_ws address; f3 direct-address freshness (in-place mutation tracked bitwise WITHOUT recapture); f4 lower-tier absence (replay+5/launcher+0/tier-2 never built); f5 stale-address impossibility (new set → bounded recapture → own bytes); f6 ZERO python-visible sync/query across 10 replay serves; ALL True |
| p15 bounded-recapture exercise | PASS 0 | 7 distinct sets: ledger `[(1,4),(2,3),(3,2),(4,1),(5,0),(5,0+t2),(5,0+t2)]`; ≤4 recaptures, overflow→tier-2; same-set revisit never re-binds (tier-2 from live A bytes); budget never negative |
| p16 three tier edges permanent-once | PASS 0 | A (capture denied): same-call bitwise via tier-2, monotone flag, permanence; A2 (total capture failure): same-call via tier-3, both flags down; B (tier-1 replay denied, identity-scoped): same-call via tier-2; C (tier-2 replay denied): same-call via tier-3; chain exhaustion still bitwise-correct |
| p17 run_out poisoned ×2 × three tiers | PASS 0 | both orderings bitwise on tier-1/2/3; data_ptr preserved; zero aliasing; forward results FRESH (never graph-resident); cross-tier run_out bits identical |
| p18 cross-instance alternation | PASS 0 | 12 interleaved serves × 4 instances (2× tier-1, tier-2-pinned, tier-3-pinned) × 3 mutation rounds bitwise-correct per instance; tier state stable |
| p19 bitwise/allclose sweep | PASS 0 | T83 {seed42, causal, extreme}: tier-1/2/3 + run_out + r002-twin 5-way bitwise equal; allclose vs base (max_abs 9.766e-04 / 2.441e-04 / 2.0=1 fp16 ULP); non-target T=41/82/96: tier-3, ZERO graph artifacts, bitwise vs twin, allclose vs base; machine table `p19_r003_sweep_result.json` |

## Binding Statement

- Kernel legality carried verbatim from r002: byte-identical jit block; 4 dot sites all (32,32)@(32,32) fp32 widened; num_warps=1 single site; num_stages 0.
- DANGER list per decision: torch.compile/reduce-overhead/TORCHINDUCTOR/tf32 strings 0; model-code synchronize/query/driver-query 0 (behaviorally re-proved p14 f6 with patched sync/query APIs); graph-resident-tensor-return 0; unbounded-recapture 0 (budget present, irreversible, gated — p15).
- Capture discipline: warmup-3-then-capture, side stream, region = one launch reading caller pointers + writing static out_ws; copy-out outside the boundary; capture once per pointer-set binding.
- Stale-address impossibility: tier-1 fires only on guard match + target regime; mismatches recompute from live bits (p14 f5, p13).
- R-term: pre-declared unattributed swing; python-visible sync/query = 0 in serving windows; driver-intrinsic launch sync reserved for Verifier census (reading (c)).

## Deviations

- D1 (conformance note): budget counting — "recapture max 4 lifetime" implemented as initial-binding-free + ≤4 recaptures (≤5 captures). Alternative reading (all captures count) leaves 3 slots; chosen reading matches "recapture" wording and harness fit. Flagged.
- D2 (conformance note): `bound_sets` (≤5 pointer-triple history) is persistent state beyond the literal Host Plan enumeration — the minimal mechanism for the decision's own "first-seen only / revisits→tier-2" rule; cache-key state, guard-only use, never dereferenced/returned. Flagged.
- D3 (conformance note): non-fp16/non-cuda inputs and kv≠h configs out of scope exactly as r002 (tier-3 assumes decision-pinned contiguous fp16 [T,H,D]; GQA absent-by-construction).
- D4 (probe-side only): fault-injection must subclass real CUDAGraph (wrappers lack capture protocol); class-wide replay patches kill both tiers → instance-identity-scoped faults. Candidate untouched throughout.

## Evidence for Verifier

- Candidate @`6ffb0c94…bf1e`; canonical route: unchanged harness, default stream, seed 42, warmup 50/repeat 100. Expected tier-1 hit-rate 100% in timed regime with exactly one warmup-time recapture (p13 end-to-end reproduction).
- Kernel-mode note: r001's D2 harness-arity fact still applies (make_profile_call passes inputs[-1]+outputs); forward-mode dual-scope fallback or direct-call lambda per Verifier choice.

### Artifact hash ledger

```text
6ffb0c94bf6b126317acddcf14119bfd27fab5709c20a1f33cfdf8883d58bf1e  triton_flexattention_e2_003.py
d4f7203e9a032a40eb0164eeb515a8a0be31c9e5067e2a80036af4344affb203  rounds/decision_003.md
4ef267b9bb67f8abc52889684412336785b4281612647f55efbacdc29f8dc6f0  rounds/sketch_003.json
570bc2be2cb8e79a06ebb32e5e8bf4f79aa62a38d5382b9a1a5f12426f3512b1  triton_flexattention_e2_002.py
b8ec3458bf810e6f3e81f759a716810b2046e5b32e22fc1339ec526e319445d1  baseline_adapter.py
71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29  auto_bench.py
f8be3a6b68f080e39f5a0b772b82f541fc590e37708df0aa8a2dfe04e956a7c1  log/probes/binding_statement_report_003.json
efc2b807d3cd022bd3a7a572201495b716fd1eba5f62b0c5b97cc18334796fef  log/probes/p13_r003_cold_capture_smoke.py
0a9a6bfb8b9c9b1b582cd7a7e8a2b094cb4a25752e62f5ae42039227ba7c4ab2  log/probes/p13_r003_cold_capture_smoke.log
09a913b116c63b5cfc1638347ca402882d39da5a9845bcd252a7750c3361b049  log/probes/p14_r003_capture_fired_multifact.py
d1615f3fab7c8959ba5482edaa22720eed8e9cfdc7e977138358ef4665584e50  log/probes/p14_r003_capture_fired_multifact.log
e1f1e58f3663fdef599623e483d7683ab0416d850768d3b7eb7be3ddfcbc3f90  log/probes/p15_r003_recapture_exercise.py
866605a11a2bfd245847c44dd04a97177245329c8f85ee0d1878685a8d716379  log/probes/p15_r003_recapture_exercise.log
aa7e1217f7acb016be27f29813da5765721cabf3b8a6db3fff341599f462e7e7  log/probes/p16_r003_tier_edges.py
92235ef6ce4243b3642f8a8f31104de3f8da53d0e52a653f9fc4540cac03f08e  log/probes/p16_r003_tier_edges.log
5f20224dc25dac0eebcf4bce5b5d76abee01d114b9e0685b340dadd326fe16e5  log/probes/p17_r003_run_out_poison_altsurface.py
b71ceb0c14bbea3d6892ad6f9dcf0f6934dee8c39206fc8e4f4ab269752ba6de  log/probes/p17_r003_run_out_poison_altsurface.log
49a3be3badc33669cc72dc79d4f15fe68ccde00b4c19c58d41bf71b10ba5039a  log/probes/p18_r003_cross_instance_alternation.py
a38e5762ec0899637ea117eb053eff04da641971f23d9d9c8bc95d07c98e0828  log/probes/p18_r003_cross_instance_alternation.log
1a3fc6e9da67ecc9e7111fc0ed640b3cad300811fbdfe8b9a77c1d43757b5d3c  log/probes/p19_r003_bitwise_sweep.py
393cea133c2b9a57aed6d1b8d54557826c6117ba074eece5bcc5c7ea24b73ac7  log/probes/p19_r003_bitwise_sweep.log
55e1b56993c0164e9c394c8b34e2a7a80f193d0eb87585e8c36fe22812016d30  log/probes/p19_r003_sweep_result.json
```

## Exact Commands (prefix: `cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable;`)

```bash
/usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py \
  --v1_file kernels/track1-triton/flexattention/bi150/epoch2/triton_flexattention_e2_003.py \
  --warmup 5 --repeat 10 --full-traceback        # exit 0, PASS accuracy
/usr/local/bin/python3 kernels/track1-triton/flexattention/bi150/epoch2/log/probes/<p13..p19>.py   # each exit 0
```

Coder claims no measurement and no verdict; classification is candidate-ready. Orchestrator owns the verification dispatch.

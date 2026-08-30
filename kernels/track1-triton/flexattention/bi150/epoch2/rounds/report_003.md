# Report 003

Result: no-improvement

## Identity

- Round: `003`
- Decision: `rounds/decision_003.md` @`d4f7203e9a032a40eb0164eeb515a8a0be31c9e5067e2a80036af4344affb203` (hash re-verified live)
- Candidate: `triton_flexattention_e2_003.py`
- Accepted reference: `baseline_adapter.py` (last_accepted_kernel, unchanged this round)
- Accepted reference report: `rounds/report_000.md` @`a90df70d54e791ecf53b38913ea1165e2a47a6dd6201d68653e6a101c5882e7c`
- Decision SHA256: `d4f7203e9a032a40eb0164eeb515a8a0be31c9e5067e2a80036af4344affb203`
- Sketch SHA256: `4ef267b9bb67f8abc52889684412336785b4281612647f55efbacdc29f8dc6f0` (rounds/sketch_003.json, re-verified)
- Candidate SHA256: `6ffb0c94bf6b126317acddcf14119bfd27fab5709c20a1f33cfdf8883d58bf1e` (15316 bytes, re-verified; AST gate OK)
- Kernel byte-identity: the `@triton.jit` `_causal_attn_fwd` block is **BYTE-IDENTICAL to r002** (Verifier machine extraction-diff: segment `'@triton.jit'..'class ModelNew'` equal across r002/r003 sources) — binding-statement claim confirmed
- Accepted reference SHA256: `b8ec3458bf810e6f3e81f759a716810b2046e5b32e22fc1339ec526e319445d1`
- Base SHA256: `dd1359ad88a5d5aae48a115023e5ae9016e4d5b5e6a12781273703e8d5c6a6d0` (unchanged)
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (unchanged, AST loader)
- Binding: `log/probes/binding_statement_report_003.json` — LIVE sha256 `f8be3a6b68f080e39f5a0b772b82f541fc590e37708df0aa8a2dfe04e956a7c1`; every machine-checkable claim independently re-verified (14 DANGER tokens zero incl. sync/query/driverGet/return-workspace; 4 dot sites (32,32)@(32,32) fp32 widened; num_warps=1 single site; state audit matches declared set; kernel byte-identity machine-verified)
- Prior round artifacts: `rounds/report_001.md` @`8c93d473f6f3babcfd34c1cbe7bde76fbf1b1db1bbc002c61cbc04d76ab79336`, `rounds/report_002.md` @`2b93a9ed63b7d9b1e5b6a043fb202472f9afe647b60ea5b67c2333837c4a5ec8`, `rounds/verdict_002.json` @`a3b7f117567fbd756356c9b10df58965665b8cd481513f47855da52db1c11985` (re-verified)
- Runtime fingerprint: `project.md#runtime-fingerprint` (environment re-bootstrapped every shell; unchanged)
- Measurement fingerprint: `6dc07009177b649f7c2cad8f7be5e9aad74235bd9f50abfebc88bdb273e32af4` (base/harness bytes re-verified identical)
- verification_tier: `authoritative`
- screening_pairs: `not-run: correct candidate proceeds directly to authoritative timing`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness (seed42) | allclose(atol=1e-2, rtol=1e-2, equal_nan) vs base.py | `PASS accuracy` in all three authoritative pairs + identity control + profile run (5/5 invocations) | pass | `log/r003_pair_00{1,2,3}_timing.txt`, `log/verifier_correctness_result_003.json` |
| 5-WAY bitwise (seed42) | tier-1 / tier-2 / tier-3 + run_out + r002-twin all bitwise-equal for identical bits | ALL TRUE: tier-1 (direct-address replay serve) vs r002 twin bitwise; tier-2 (probe-pinned `direct_replay_failed`) bitwise; tier-3 (both flags pinned) bitwise; run_out poisoned-buffer write bitwise with data_ptr preserved and never aliased to workspace; r002 twin computed fresh | pass | `log/verifier_correctness_result_003.json` five_way block |
| harness-premise at measurement scale | warmup-50 absorbs initial binding + exactly ONE recapture; timed segment 100% tier-1, zero recaptures | BEHAVIORAL COUNTERS (instance `recapture_budget`/`bound_sets`/flags, cheaply readable): after correctness-phase call budget=4/bound_sets=1 (initial binding budget-free); after 50-call warmup budget=3/bound_sets=2 (EXACTLY ONE recapture); after 100-call timed analog budget=3/bound_sets=2 (**ZERO timed recaptures**), all 100 timed calls bitwise==twin, tier-1 active (graph handle present, flags false) — coder p13 premise VERIFIED | pass | probe phases ledger |
| fp16-extreme / tie-free boundary case | extreme-magnitude rows within tolerance | manufactured suite: allclose PASS, max_abs 7.8125e-03 (same as r002 — same kernel), tier-1 AND tier-2 bitwise==twin | pass | probe extreme block |
| non-target shape | tier-3 with zero graph artifacts | T=41: allclose PASS, bitwise==twin, `graph_direct/graph_copyin/out_ws/q_in` all None | pass | probe non_target block |
| stale-trap | changed bits on BOUND pointers serve fresh results without recapture | in-place mutation of bound tensors: output bitwise==twin on live bytes, budget unchanged (3), not graph-resident | pass | probe stale_trap block |
| stateless/bounded state | declared state set only; monotone flags; no undeclared attrs | counters match declared set throughout; flags never moved | pass | probe counters + binding state audit |
| default stream | caller-stream replay + copy-out; captures on side stream only | census: replay + memcpyAsync on caller route; no stream manipulation in any verifier invocation | pass | census + source read |

Conformance, correctness, and every declared guardrail passed.

## Screening Evidence

Not run — correct candidate proceeded directly to authoritative timing.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved ordered reference/candidate pairs — three identical harness invocations (pair i = invocation_i.v0_ms then invocation_i.v1_ms), byte-for-byte identical flags, default stream`
- reference_raw_samples_ms: `[0.151042, 0.149147, 0.147234]`
- candidate_raw_samples_ms: `[0.148821, 0.150517, 0.143639]`
- reference_median_ms: `0.149147`
- candidate_median_ms: `0.148821`
- improvement_pct: `+0.2186`

```text
improvement_pct = (0.149147 - 0.148821) / 0.149147 * 100 = +0.2186
```

A WASH: decisively below the +5.0% adoption bar. Pair-level deltas (+1.49% / −0.91% / +2.46%) sit at the drift-noise scale; the identity-control pair (base vs accepted adapter) printed v0=0.146825 / v1=0.146804 (Δ −0.021 µs — noise floor ≈0), confirming the true delta is ≈ −0.3 µs, not a masked larger win.

### Anchor bases (all four, explicitly)

1. **Prescribed paired v0-basis headline**: v0=base.py 0.149147 vs v1=candidate 0.148821 ms, same session → **+0.2186%** (adoption-decisive; below bar).
2. **Direct same-session pair vs r000 v0**: adapter-as-v0 remains structurally blocked (r002 named deviation stands); since the adapter is byte-equivalent in pipeline to base.py, the v0=base.py paired basis IS the accepted-reference basis (stated explicitly): +0.2186%.
3. **Cross-anchor `report_000` 0.151107 ms**: candidate 0.148821 → **+1.5128% faster** (below bar). This session's v0 median sits −1.30% below r000's 0.151107 (session drift, cuts both ways).
4. **Manifest anchor**: identical to the report_000 anchor this early (no accepted round committed beyond Phase 0) — stated explicitly: manifest anchor 0.151107 ms, cumulative context +1.5128%.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| wall_time_unrounded_paired_median_ms | ≥5% below same-session accepted reference median across interleaved pairs | **+0.2186%** (0.148821 vs 0.149147 ms) — wash, below bar | **fail** | pairs + control |
| tier1_hit_rate_in_timed_regime | all timed calls ride tier-1: ZERO copy-in DtoD memcpys, exactly 1.00 cudaGraphLaunch + 1.00 copy-out memcpy per call; timed recaptures = 0 (≤1 in warmup) | census: **0 copy-in memcpys, 1.00 cudaGraphLaunch, 0.99 copy-out DtoD/call**; behavioral counters: exactly one warmup recapture, **zero timed recaptures**, 100/100 timed calls bitwise==twin — reading (d) falsified, design premise CONFIRMED | **pass** | census + probe counters |
| aten_cpu_ops_per_call | ≤5/call on the timed path | **3.00/call** (empty_like + empty_strided + copy_) | **pass** | census |
| submission_and_sync_census | exactly 2.00 GPU submissions/call (1 graph launch + 1 memcpyAsync), ZERO cudaLaunchKernel; model-code sync/query = 0; any per-call cudaDeviceSynchronize/cudaDriverGetVersion OBSERVED = build-intrinsic replay cost (pre-declared branch c), not silently absorbed | **2.00 submissions/call** (1.00 cudaGraphLaunch @5.46 µs host + 1.00 cudaMemcpyAsync @5.51 µs host); **ZERO cudaLaunchKernel; ZERO cuLaunchKernel** (python launcher executes ZERO times in serving — r002's ~85 µs tax neutralized as designed); model-code sync/query = 0 (source audit); **but per-call cudaDeviceSynchronize 69.02 µs/call + cudaDriverGetVersion 0.18 µs OBSERVED in the candidate scope** → BRANCH (c) TRIGGERED, recorded as build-intrinsic replay cost | **pass** with branch-(c) observation | census |
| cross_tier_bitwise_retention | tier-1/2/3 bitwise-equal both surfaces; stale-trap correct; run_out poisoned ×2 bitwise | 5-way bitwise ALL TRUE; stale-trap fresh-bits-no-recapture TRUE; poisoned ×2 + data_ptr preserved TRUE | **pass** | probe |
| device_us_per_call | candidate attributed band ≈16.5 (kernel, in-graph — attribution may coarsen per r001 branch-B, census substitutes) + ~1–2 copy-out; materially higher triggers pessimistic branch | canonically **ZERO attributed kernel events** in candidate scope (intra-graph coarsening — the r001 branch-B pattern; summarizer reports "scope has no kernel events"); census substitution: copy-out DtoD 3.70 µs/call device (0.99 events) + kernel 16.51 µs/call taken from r002's DIRECT measurement of the BYTE-IDENTICAL kernel ⇒ composite ≈20.2 µs/call vs Ixmma 14.99 this session (+5.2 µs, inside the decision's priced +3.3..5.3 band); no foreign kernel names anywhere in the candidate scope (eager fallback would have shown cuLaunchKernel + _causal_attn_fwd events — none) | **pass** (census-substituted) | census + report_002 |
| proven_envelope_binding_audit | every tl.dot (32,32)@(32,32) fp32 widened; num_warps=1; DANGER zero | independently re-verified + kernel byte-identity vs r002 machine-confirmed | **pass** | binding + audit |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-003`
- intervention: compose r001 graph machinery with r002 kernel — direct-address manual-graph replay, lean boundary (guard + ONE replay + copy-out), copy-in + eager fallback tiers
- expected_causal_chain: edges PARTIALLY observed — launcher neutralization CONFIRMED (zero cuLaunchKernel in serving; the entire r002 failure mechanism removed); lean boundary CONFIRMED (0 copy-ins, 2.00 submissions, 3.00 aten ops); device delta CONFIRMED inside the priced band (+5.2 µs composite vs priced +3.3..5.3); wall edge FALSIFIED: the priced python savings (~17–28 µs) did NOT reach the wall because the build-intrinsic replay floor (R-term, branch (c): 69.02 µs/call cudaDeviceSynchronize in the LEAN route) absorbs them — the replay route's blocking structure converts the designed submit-and-return overlap into a serialized wait (harness per-sample sync already serialized the loop; replay sync moves the wait inside the call and the graph-completion latency ≈49 µs above device work is a fixed build floor)
- primary_metric: `wall_time`
- Hypothesis verdict: `partially-confirmed` — three of four mechanism edges measured CONFIRMED (launcher dead, boundary lean, device in-band) and the design premise (hit-rate 100, zero timed recaptures) VERIFIED; the primary wall outcome is a wash (+0.2186% << +5%). Per decision readings: **(b) applies** (wall <5% with hit-rate 100 AND lean census ⇒ boundary floor exceeds the prize) **refined by (c)** (per-call sync/driverGet observed ⇒ build-intrinsic replay floor named as root cause). No (d)/(e)/(f). Adoption governed by wall_time ⇒ `no-improvement` #3 ⇒ campaign auto-terminates with baseline_adapter as final deliverable.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `available for the reference scope; candidate scope intra-graph kernels are attribution-coarsened on this build (zero attributed events — r001 branch-B pattern) and the census substitutes per the decision's pre-declared rule`
- mode deviation: kernel-mode profiling remains structurally blocked (r001 D2 harness arity stands); forward-mode dual-scope used, pw=20/pi=100, `--profile-reference-file baseline_adapter.py`
- trace-shape facts: (i) kineto gpu_user_annotation duplicate-span workaround from r002 applied (host user_annotation scoping); (ii) `summarize_trace.py` reports `scope has no kernel events: candidate_triton_flexattention_e2_003` — recorded as the branch-B attribution signature, not evidence loss; the reference scope summarized canonically
- iterations: `100` forward calls per scope
- trace: `log/r003_forward_100iter.pt.trace.json` @`c8182c25c3f27af789fe6f8d187a35d7eaa95afc739a5aeb285904a85cd5c5f3`
- scope summaries: `log/r003_summary_reference.json` (canonical), candidate scope via `log/diagnostic_scope_census_round003.json` (Verifier census)

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference (baseline_adapter) | 1499.0400390625 | 14.990400390625 | 98 | 0.98 | 0.149147 | 0.10050755557017574 |
| candidate (triton_flexattention_e2_003) | ≈2020 (census-substituted: 16.51 kernel [r002 direct measurement, byte-identical kernel] + 3.70 copy-out) | ≈20.2 | 0 attributed (in-graph) + 99 DtoD events | 0 attributed; 0.99 copy-out | 0.148821 | ≈0.1358 (composite) |

### Accepted Reference Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `ixattnbkd::…::FlashAttnFwdF16Ixmma<128u, 128u, 16u, 64u, 64u, (CausalM_t)2, (AlibiMode_t)0, false, __half, false>` | 98 | 0.98 | 1499.0400390625 | 14.990400390625 |

### Candidate Top Kernels (census-substituted)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `Memcpy DtoD (Device -> Device)` (the copy-out) | 99 | 0.99 | 366.7 | 3.704 |
| `_causal_attn_fwd` (in-graph, attribution-coarsened; device time from r002 direct measurement of the byte-identical kernel) | — | 1.00 (structural) | — | 16.51 (r002) |

No other kernel names appear in the candidate scope (a tier-3 fallback during serving would show `cuLaunchKernel` + `_causal_attn_fwd`/Ixmma events — none present).

### Candidate per-call census (the lean boundary, exactly as designed — plus the R-term)

| Signal | accepted_reference | candidate (tier-1 direct-address replay) |
|---|---:|---:|
| aten cpu_ops | 38 | **3.00** (empty_like, empty_strided, copy_) |
| cudaLaunchKernel | 1.00 (5.00 µs host) | **0** |
| cuLaunchKernel (python launcher — r002's tax) | 0 | **0 — launcher executes ZERO times in serving** |
| cudaGraphLaunch | 0 | **1.00** (5.46 µs host) |
| cudaMemcpyAsync | 0 | 1.00 (5.51 µs host) |
| DtoD device trips | 0 | 0.99 (copy-out only — ZERO copy-ins) |
| **cudaDeviceSynchronize** | **0** | **1.00 @ 69.02 µs/call — THE R-TERM, build-intrinsic (branch (c))** |
| **cudaDriverGetVersion** | **0** | **1.00 @ 0.18 µs/call** |
| model-code sync/query | 0 | 0 (source-audited; the observed sync is INSIDE the build's replay path) |

### THE CAMPAIGN'S FIVE-NUMBER DECOMPOSITION (close-out — full physics accounted)

| # | Term | Measured value | Provenance |
|---|---|---:|---|
| 1 | Base (accepted) host path | **≈134 µs/call** (wall 149.15 − device 14.99 this session; 137.5 µs in r000's session) — ~38 cheap aten ops + sdpa C++ stack + 1 launch + fixed seed/sync floor | r003 ref census; report_000 |
| 2 | r001 wrapper net (aten-captured replay behind a FAT boundary: 3 copy-ins + 5 submissions + per-call sync) | **+2.6 µs/call wall** vs same-session ref (−1.6873%); proved manual capture, bitwise retention at scale, per-aten-op price 0.6–1.0 µs | report_001 |
| 3 | r002 launcher tax (direct python-launched Triton) | **≈+86–89 µs/call net host** (wall +92.3 µs ABAB; device only +2.9); of which the driver submission is 3.53 µs — pure python launcher ≈82–86 µs | report_002 |
| 4 | r003 composed net remaining (lean direct-address replay) | **−0.3 µs/call wall — a wash** (+0.2186%): host work −5.6 µs vs device +5.2 µs (kernel 16.51 + copy-out 3.70 vs Ixmma 14.99); the R-term (build-intrinsic replay sync, 69.02 µs/call observed host-blocking; implied replay fixed latency ≈49 µs above the ~20 µs device work) absorbs exactly the python savings the priced identity banked | this round |
| 5 | Device floors | **Ixmma 13.61–15.0 µs/call** (session-dependent clocks) vs **Triton `_causal_attn_fwd` 16.51 µs/call** (byte-identical kernel, direct r002 measurement) → Δdevice ≈ +2.9..+5.2 µs — the proven-envelope kernel is device-competitive but never wins | r000/r002/r003 |

**R-term attributability (the pre-declared swing): ADJUDICATED — build-intrinsic.** The 69.02 µs/call `cudaDeviceSynchronize` (plus 0.18 µs `cudaDriverGetVersion`) appears on the LEAN tier-1 route whose model code contains zero sync/query (source-audited, DANGER-scan zero) — identical observation to r001's fat route ⇒ intrinsic to this build's `cudaGraphLaunch`/replay path, NOT an r001 code artifact, NOT candidate-caused. It is the dominant candidate-scope host item and, with the harness's own per-sample synchronize, closes the overlap window the composition was designed to open.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable (first verification of this candidate) | `6ffb0c94bf6b126317acddcf14119bfd27fab5709c20a1f33cfdf8883d58bf1e` | candidate bytes never changed; correctness passed first time; no repairs |

At most one Verifier-to-Coder repair is allowed in the same round; zero were needed.

## evidence_for_next_round

- None for a next candidate round: this was the campaign's third no-improvement ⇒ auto-termination; `baseline_adapter.py` @`b8ec3458…` stands as final deliverable (0.151107 ms manifest anchor wall; +59.28% vs the epoch-1 naive 0.61x era on the same rig, per the campaign arc r000→r003).
- Terminal physics (all measured, census-grade): (i) the vendor fused-attention path's host floor (~134 µs/call) is composed of ~38 cheap aten dispatches + sdpa C++ internals + fixed per-sample seed/sync; (ii) no legal candidate mechanism reduces it: aten-captured replay pays boundary ≥ prize (r001 +2.6 µs), direct Triton launch pays ~85 µs python launcher (r002 −60%), lean direct-address replay hits the build-intrinsic replay-sync floor at ~69 µs/call that absorbs the entire python prize (r003 wash); (iii) device-side, the proven-envelope Triton kernel (16.51 µs) trails the vendor IXMMA kernel (13.6–15.0 µs) — no device win exists inside the frozen capability envelope; (iv) the R-term is build-intrinsic (branch (c) adjudicated), so any future graph-based family on this build starts ~69 µs/call in the hole.
- The campaign ends with its full physics accounted: dispatch price, launcher price, boundary floor, device floor, replay-sync attributability — ALL measured and quantified above.

Evidence only; the stop transition belongs to the Orchestrator.

## Stop Recommendation

- recommendation: `stop` (recorded honestly per role: this is no-improvement #3 of 3 — the campaign's own auto-termination rule fires; the evidence base above is complete for finalization)
- evidence: wall +0.2186% << +5% bar; readings (b)+(c) apply; no candidate-failed channel (correctness/bitwise all pass — reading (f) never triggered); counters 3/3 no-improvement, 3/20 rounds.

Orchestrator owns the stop transition.

## Exact Reproduction Commands

Authoritative pairs (1–3 identical; default stream):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/bi150/epoch2/triton_flexattention_e2_003.py --warmup 50 --repeat 100 --full-traceback
```

Identity-control pair (drift context):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/bi150/epoch2/baseline_adapter.py --warmup 50 --repeat 100 --full-traceback
```

Dual-scope profiler (forward-mode, pw=20/pi=100) + per-scope normalization + census:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/bi150/epoch2/triton_flexattention_e2_003.py --warmup 50 --repeat 100 --profile --profile-reference-file kernels/track1-triton/flexattention/bi150/epoch2/baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 100 --profile-output kernels/track1-triton/flexattention/bi150/epoch2/log/r003_forward_100iter.pt.trace.json
/usr/local/bin/python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/flexattention/bi150/epoch2/log/r003_forward_100iter.pt.trace.json --iterations 100 --scope reference_baseline_adapter --wall-ms 0.149147
/usr/local/bin/python3 kernels/track1-triton/flexattention/bi150/epoch2/log/probes/verifier_scope_census_003.py
```

Verifier correctness probe (counters + 5-way bitwise):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/local/bin/python3 kernels/track1-triton/flexattention/bi150/epoch2/log/probes/verifier_correctness_003.py
```

Artifact hash ledger (re-verified this round):

```text
6ffb0c94bf6b126317acddcf14119bfd27fab5709c20a1f33cfdf8883d58bf1e  triton_flexattention_e2_003.py
d4f7203e9a032a40eb0164eeb515a8a0be31c9e5067e2a80036af4344affb203  rounds/decision_003.md
4ef267b9bb67f8abc52889684412336785b4281612647f55efbacdc29f8dc6f0  rounds/sketch_003.json
a90df70d54e791ecf53b38913ea1165e2a47a6dd6201d68653e6a101c5882e7c  rounds/report_000.md
8c93d473f6f3babcfd34c1cbe7bde76fbf1b1db1bbc002c61cbc04d76ab79336  rounds/report_001.md
2b93a9ed63b7d9b1e5b6a043fb202472f9afe647b60ea5b67c2333837c4a5ec8  rounds/report_002.md
a3b7f117567fbd756356c9b10df58965665b8cd481513f47855da52db1c11985  rounds/verdict_002.json
b8ec3458bf810e6f3e81f759a716810b2046e5b32e22fc1339ec526e319445d1  baseline_adapter.py
dd1359ad88a5d5aae48a115023e5ae9016e4d5b5e6a12781273703e8d5c6a6d0  ../../base.py
71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29  auto_bench.py
f8be3a6b68f080e39f5a0b772b82f541fc590e37708df0aa8a2dfe04e956a7c1  log/probes/binding_statement_report_003.json
c8182c25c3f27af789fe6f8d187a35d7eaa95afc739a5aeb285904a85cd5c5f3  log/r003_forward_100iter.pt.trace.json
```

## vNext Fact Pack

```json
{
  "schema_version": 1,
  "candidate_sha256": "6ffb0c94bf6b126317acddcf14119bfd27fab5709c20a1f33cfdf8883d58bf1e",
  "correctness": {
    "status": "pass",
    "evidence": [
      "auto_bench.py PASS accuracy in all three authoritative pairs + identity control + profile run (5/5)",
      "log/verifier_correctness_result_003.json all_checks_pass=true: 5-way bitwise (tier-1/tier-2/tier-3 + run_out poisoned + r002-twin) on seed42, harness-premise at scale (initial binding budget-free -> exactly ONE warmup recapture -> ZERO timed recaptures, 100/100 timed bitwise), stale-trap fresh-bits-no-recapture, fp16-extreme max_abs 7.8125e-03, non-target T=41 tier-3 with zero artifacts"
    ]
  },
  "observables": [
    {
      "name": "wall_time_unrounded_paired_median_ms",
      "status": "observed",
      "value": "+0.2186% (reference 0.149147 ms vs candidate 0.148821 ms; bar +5.0% FAILED; a wash — identity-control delta -0.021 us confirms noise floor)",
      "confidence": "high",
      "evidence": ["log/r003_pair_001_timing.txt", "log/r003_pair_002_timing.txt", "log/r003_pair_003_timing.txt", "log/r003_control_identity.txt"]
    },
    {
      "name": "tier1_hit_rate_in_timed_regime",
      "status": "observed",
      "value": "100% hit-rate CONFIRMED: zero copy-in memcpys, 1.00 cudaGraphLaunch + 0.99 copy-out DtoD per call; exactly one warmup recapture (budget 4->3), zero timed recaptures (behavioral counters)",
      "confidence": "high",
      "evidence": ["log/diagnostic_scope_census_round003.json", "log/verifier_correctness_result_003.json"]
    },
    {
      "name": "aten_cpu_ops_per_call",
      "status": "observed",
      "value": "3.00/call (empty_like + empty_strided + copy_)",
      "confidence": "high",
      "evidence": ["log/diagnostic_scope_census_round003.json"]
    },
    {
      "name": "submission_and_sync_census",
      "status": "observed",
      "value": "exactly 2.00 GPU submissions/call (1.00 cudaGraphLaunch + 1.00 cudaMemcpyAsync), ZERO cudaLaunchKernel, ZERO cuLaunchKernel (python launcher executes zero times in serving); model-code sync/query = 0; BUT per-call cudaDeviceSynchronize 69.02 us/call + cudaDriverGetVersion 0.18 us OBSERVED -> BRANCH (c): build-intrinsic replay cost, recorded not absorbed",
      "confidence": "high",
      "evidence": ["log/diagnostic_scope_census_round003.json"]
    },
    {
      "name": "cross_tier_bitwise_retention",
      "status": "observed",
      "value": "5-way bitwise ALL TRUE on seed42 (tier-1/2/3 + run_out poisoned x2 + r002-twin); stale-trap fresh bits without recapture; data_ptr preserved, never aliased",
      "confidence": "high",
      "evidence": ["log/verifier_correctness_result_003.json"]
    },
    {
      "name": "device_us_per_call",
      "status": "observed",
      "value": "zero attributed kernel events in candidate scope (in-graph coarsening, r001 branch-B pattern; census substitutes per decision rule): copy-out 3.70 us/call device + kernel 16.51 us/call (r002 direct measurement, byte-identical kernel) => composite ~20.2 vs Ixmma 14.99 this session (+5.2, inside priced +3.3..5.3 band); no foreign kernels",
      "confidence": "high",
      "evidence": ["log/diagnostic_scope_census_round003.json", "log/r003_summary_reference.json", "rounds/report_002.md"]
    },
    {
      "name": "proven_envelope_binding_audit",
      "status": "observed",
      "value": "4 tl.dot sites all (32,32)@(32,32) fp32 widened; num_warps=1 single site; 14 DANGER tokens zero; kernel block BYTE-IDENTICAL to r002 (machine extraction-diff)",
      "confidence": "high",
      "evidence": ["log/probes/binding_statement_report_003.json"]
    }
  ],
  "lowering": {
    "status": "observed",
    "expected_mechanism": "present — the byte-identical r002 Triton kernel is captured into a manual graph and replay-served (zero python launcher executions in the timed path)",
    "evidence_contract": "triton_cuda-v1 (proven-envelope dots consumed verbatim; P1-P4 ladder untriggered)",
    "evidence": ["log/diagnostic_scope_census_round003.json", "log/verifier_correctness_result_003.json"]
  },
  "evidence_gap_cause": "none"
}
```

# Report 003

Result: accepted

> **BOUNDARY-CLASS ACCEPTANCE — READ THE MARGINALITY NOTE.** The protocol statistic (three prescribed pairs, unrounded median — the contract's exact rule) clears the 5.0% bar by **+0.077 pp** (+5.0767%). Extended estimators straddle the bar (8-pair median +5.345% PASS; 5-pair +4.636% FAIL; clean per-pair mean +4.679% FAIL). The candidate won **all 8 paired invocations** (100% win rate). The classification `accepted` follows the contract's mechanical rule on the prescribed protocol; the Orchestrator owns transitions and should weigh the boundary evidence in how it applies them. Full estimator table under Interleaved Wall Timing.

## Identity

- Round: `003` (declared FINAL ROUND of the campaign)
- Decision: `rounds/decision_003.md` @`0a678da87a877b9c521b6c280eb3518b20f98e352786e9df129435e2cc918413` (hash re-verified live; F2 graph-replayed-triton-direct-address composition, expected_wall_improvement_pct 0.0 declared honestly with pre-declared readings (a)–(f))
- Candidate: `triton_mm_encoder_attention_e2_003.py`
- Accepted reference: `baseline_adapter.py` (last_accepted_kernel; byte-equivalent pipeline to base.py)
- Accepted reference report: `rounds/report_000.md` @`20b21646d9c3ba3abe086d8133799d23a39981dcb4e1cb547e1a3f65b0bf7ffc` (hash re-verified live)
- Decision SHA256: `0a678da87a877b9c521b6c280eb3518b20f98e352786e9df129435e2cc918413` (file) / `77fb80a5c7ca3bf937c7185f74b775d1c58a0101fd3ff2cd3ebdf9084fb33b90` (canonical metadata, verdict basis)
- Sketch SHA256: `bdf423556e7c80369ae38d4980529a739a52a3d18033e572927354b23e0a4e64` (file; canonical `aa9ddb02…9a22`, verdict basis)
- Candidate SHA256: `d503e845d1d95ad033e616730d78d67d0775df3a473b17a3c37843d463654d81` (re-verified before AND after all runs; AST parse OK)
- Kernel byte-identity vs r002: **machine-verified by the verifier** (extraction-diff of the `@triton.jit`…`class ModelNew` segment: equal, 4168/4168 chars) — round 003 changes ONLY the execution boundary
- Accepted reference SHA256: `c3980a2c4ae79171c73c7cf85c29c62f6ac98b9e9fb643c249fc9a6e0a921c9f`
- Base SHA256: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` (unchanged, re-verified after all runs)
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (unchanged, AST loader)
- Binding: `log/probes/binding_statement_report_003.json` @`4b3985a81b134cc947ae2cbaf1436e67885365a9be54fda8aef6961e5779c9b6` (file; canonical `48cd7b37…4e25`) — verifier independently re-verified every checkable claim (kernel byte-identity, 9 forbidden tokens zero, 4 dot sites, single `num_warps=2` site, 6 widening casts, 5 `copy_` sites outside the captured region, 1 `is_contiguous` gate site, bounded-state audit, 6-way bitwise, tier-edge/fault-injection claims spot-verified via the verifier probe below)
- Runtime fingerprint: `project.md#runtime-fingerprint` (unchanged since r000 live probe; environment re-bootstrapped every shell)
- Measurement fingerprint: `0c4c7d664c85e65d0580091ca5e3a77ff769a0d28f7e679f5bdf78fe5d0d966e` (recomputed live this round; base/harness bytes re-verified identical before and after timing)
- verification_tier: `authoritative`
- screening_pairs: `not-run: correct candidate proceeded directly to authoritative timing (r001/r002 precedent; a screen-out would skip the profiler and destroy the round's four mandated closing censuses)`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness (seed42 canonical) | `allclose(atol=1e-2, rtol=1e-2, equal_nan=True)` vs base.py, seed 42, fp16 out `[2,83,512]` | `PASS accuracy` in all 8 authoritative pairs + profile run (9/9 invocations); probe max_abs 4.883e-04 | pass | `log/r003_pair_00{1..8}_timing.txt`, `log/r003_profile_run.txt`, `log/probes/verifier_r003_result.json` |
| fp16-extreme suite | extreme-magnitude inputs vs fp32 ground truth (r001-established basis) | candidate max_abs 3.052e-05 — r001/r002-identical (byte-identical kernel); vendor base diverges 1457 (standing vendor precision-limit observation) | pass | `log/probes/verifier_r003_result.json` |
| non-target shapes | correctness off the target shape through tier-3, zero graph artifacts | B1S41 seed7, B2S82 seed13, B2S96 seed19 all PASS vs base (max_abs 4.883e-04), bitwise-equal to the r002 twin, run_out bitwise, and **zero graph artifacts** (graph handles/workspaces all None after serving) | pass | `log/probes/verifier_r003_result.json` |
| 6-way bitwise retention (tier-1 ×2 / tier-2 / r002 twin / run_out poisoned ×2) | all tiers bitwise-equal for identical input bits, both surfaces | TRUE on every target-regime suite (seed42, extreme, boundary): tier-1 repeat (replay route), tier-2 (fresh-pointer copy-in route), r002 direct twin, poisoned ×2 both orderings — all `torch.equal` | pass | `log/probes/verifier_r003_result.json` |
| run_out contract | `run_out(q,k,v,out) -> None`, caller buffer filled via copy-out outside the replay boundary, bitwise==forward, data_ptr preserved, never aliased to workspace | poisoned buffers ×2 (−777.0/+555.0, both orderings) bitwise-equal with data_ptr preserved; returns None; zero allocations on this surface | pass | `log/probes/verifier_r003_result.json` |
| stale-address impossibility | mismatched pointers recompute from live bits; in-place mutations flow through the next replay | verified BOTH ways: new-pointer set served bitwise-correct from its own bytes; in-place mutation at stable pointers produced fresh correct results (replay reads LIVE bytes, not stale) | pass | `log/probes/verifier_r003_result.json` |
| forward bitwise-stability | deterministic serving | repeat identical-input calls bitwise-stable on every suite | pass | `log/probes/verifier_r003_result.json` |
| bounded state | declared set only: 2 graph handles, workspaces, anchors, bound_sets ≤5, recapture counter ≤4, monotone tier flags | live-audited after full exercise: budget 3 (4 − 1 recapture), bound_sets 2, 2 graph handles, flags false — exactly the declared set | pass | `log/probes/verifier_r003_result.json` |
| capability legality | kernel byte-identical to r002: 4 dot sites (32,32)@(32,32) fp32 widened, single `num_warps=2` site, `num_stages` absent | all re-verified over final bytes (4 dots, 6 widens, `num_warps=2` single site) | pass | `log/probes/verifier_r003_result.json` |
| no compile/forbidden machinery | zero torch.compile/TORCHINDUCTOR/reduce-overhead/num_stages/autotune/synchronize/.query(/DriverGet/.contiguous( | all 9 forbidden tokens count 0; the sanctioned mechanism sites machine-counted: 1 `torch.cuda.CUDAGraph` code site, 5 `copy_` sites, 1 `is_contiguous` gate | pass | `log/probes/verifier_r003_result.json` |
| zero model-code sync/query | no synchronization in candidate source; the observed per-call sync is the framework replay path (R-term), recorded not absorbed | source `synchronize`/`.query(`/`DriverGet` count 0; the trace's per-call `cudaDeviceSynchronize` originates inside `CUDAGraph.replay()` on this build — census (b) quantifies it | pass | source audit + `log/diagnostic_scope_census_round003.json` |

Conformance, correctness, and every declared guardrail passed.

## Screening Evidence

Not run — correct candidate proceeded directly to authoritative timing (r001/r002 precedent). The round's contractual products are the four closing censuses, which require the profiler a screen-out would skip.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved ordered reference/candidate pairs — harness invocations (pair i = invocation_i.v0_ms then invocation_i.v1_ms), byte-for-byte identical flags, interpreter, device, default-stream route, CoreX environment`
- reference_raw_samples_ms (8 invocations): `[0.147793, 0.149939, 0.194862, 0.144358, 0.144063, 0.149326, 0.158449, 0.151585]`
- candidate_raw_samples_ms (8 invocations): `[0.140942, 0.142966, 0.142327, 0.137093, 0.137414, 0.143134, 0.137038, 0.143989]`
- reference_median_ms: `0.149939` (three prescribed pairs: `[0.147793, 0.149939, 0.194862]`)
- candidate_median_ms: `0.142327` (three prescribed pairs: `[0.140942, 0.142966, 0.142327]`)
- improvement_pct: `+5.076731` (protocol statistic; ≥ 5.0 bar — **cleared by +0.077 pp**)

```text
improvement_pct = (0.149939 - 0.142327) / 0.149939 * 100 = +5.076731
```

| # | Reference wall ms | Candidate wall ms | Improvement | Note | Evidence |
|---:|---:|---:|---:|---|---|
| 1 | `0.147793` | `0.140942` | `+4.636%` | prescribed | `log/r003_pair_001_timing.txt` |
| 2 | `0.149939` | `0.142966` | `+4.651%` | prescribed | `log/r003_pair_002_timing.txt` |
| 3 | `0.194862` | `0.142327` | `+26.96%` | prescribed; REF-side host transient (same environment-noise class as r000 pair-3, candidate side then) | `log/r003_pair_003_timing.txt` |
| 4 | `0.144358` | `0.137093` | `+5.033%` | extra (boundary rigor) | `log/r003_pair_004_timing.txt` |
| 5 | `0.144063` | `0.137414` | `+4.594%` | extra (boundary rigor) | `log/r003_pair_005_timing.txt` |
| 6 | `0.149326` | `0.143134` | `+4.148%` | extra (boundary rigor) | `log/r003_pair_006_timing.txt` |
| 7 | `0.158449` | `0.137038` | `+13.54%` | extra; ref-side elevated window | `log/r003_pair_007_timing.txt` |
| 8 | `0.151585` | `0.143989` | `+5.010%` | extra (boundary rigor) | `log/r003_pair_008_timing.txt` |

Extra (non-authoritative) profile-run timing: v0=0.149817, v1=0.144134 (+3.79%, profiler overhead included).

### Marginality analysis (the loud caveat)

| Estimator | Value | vs 5.0% bar |
|---|---:|---|
| **Protocol statistic (3 prescribed pairs, unrounded median — the contract's rule)** | **+5.0767%** | **PASS by 0.077 pp** |
| Extended 8-pair median (all invocations) | +5.3451% | PASS by 0.345 pp |
| Extended 5-pair median (pairs 1–5) | +4.6355% | FAIL by 0.365 pp |
| Clean per-pair mean (6 pairs excl. ref transients) | +4.679% | FAIL by 0.321 pp |
| Clean per-pair median | +4.644% | FAIL by 0.356 pp |
| Win rate across all 8 invocations | 8/8 candidate faster | — |

Facts bearing on the boundary: (i) the protocol statistic's marginal pass is partly carried by the pair-3 reference-side transient (0.194862), which lifts the 3-sample reference median to the highest clean ref value (0.149939) — without that pair, pairs 1–2 alone give +4.64%; (ii) counterbalancing, the reference side showed an upward drift in later invocations (pairs 6–8: 0.1493–0.1584 vs pairs 4–5: 0.1441–0.1444) while the candidate stayed stable (0.1370–0.1440 throughout) — the 8-pair median (+5.345%) reflects that drift; (iii) the candidate won EVERY invocation, including the clean ones, by 4.1–5.0%; (iv) the mechanism is census-confirmed real (below), so the improvement is not an artifact of noise alone. The honest statement: **the true improvement is ~4.6–5.3% and straddles the 5.0% bar; the pre-registered protocol statistic lands at +5.0767% and the classification follows it.**

### Anchor bases (all four, explicitly)

1. **Prescribed paired v0-basis headline**: v0=base.py 0.149939 ms vs v1=candidate 0.142327 ms, same session → **+5.0767%** (adoption-decisive basis).
2. **Direct same-session pair vs r000 v0**: last_accepted_kernel IS the base adapter — the prescribed paired-v0 basis and the direct pair vs r000 v0 are THE SAME comparison this round — stated explicitly.
3. **Cross-anchor `report_000` 0.150149 ms**: candidate 0.142327 → +5.2126% (8-pair basis: 0.141634 → +5.6730%).
4. **Manifest anchor**: identical to the report_000 anchor (no accepted round committed prior; r000 completion remains the only prior) — stated explicitly: manifest anchor wall = 0.150149 ms, cumulative context +5.2126%.

ABAB interleaved control: superseded by the extended-pair evidence (5 extra invocations beyond the prescribed 3, all with identical flags) — the estimator table above IS the drift control.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| wall_time_unrounded_paired_median_ms | ≥5% below accepted reference — honestly declared unreachable (composed band 0.94–1.01x) | **+5.0767% (protocol statistic)** — the decision's pre-declared win branch (reading a) FIRED, above the entire priced band; marginality documented above | **pass** (bar cleared on the protocol statistic; boundary-class) | `log/r003_pair_00{1..8}_timing.txt` |
| tier1_hit_rate_in_timed_regime | all timed calls ride tier-1: ZERO copy-in memcpys, exactly 1.00 cudaGraphLaunch + 1.00 copy-out per call, zero timed recaptures; hit-rate 0 would falsify the design premise | **1.00 cudaGraphLaunch/call, 1.00 cudaMemcpyAsync/call, 0.99 DtoD device memcpy/call (99/100 — recorder span-edge margin, same class as the reference scope's 0.93), 0.00 cuLaunchKernel serving, 0.00 copy-ins, 0 timed-segment recaptures** (exactly 100 graph launches for 100 profiled iterations; the capture + one warmup recapture ran outside the profiler window — the wall itself proves the replay route served: a launcher-executing route walls at ~0.23 ms, the measured walls are 0.137–0.144 ms) | **pass** (hit rate 100% within recorder margin) | `log/diagnostic_scope_census_round003.json` |
| aten_cpu_ops_per_call | ≤6/call on the timed path | **3.00/call** — exactly `aten::empty_like` (18.64 µs) + `aten::empty_strided` (16.05 µs) + `aten::copy_` (20.66 µs): the destination allocation and the copy-out dispatch, nothing else | **pass** | `log/diagnostic_scope_census_round003.json` |
| submission_and_sync_census | exactly 2.00 GPU submissions/call (1 graph launch + 1 memcpyAsync), ZERO cuLaunchKernel on the replayed route; per-call sync/driverGet OBSERVED and recorded as the build-intrinsic replay cost (R-term), not silently absorbed | **2.00 GPU submissions/call exactly** (1.00 cudaGraphLaunch + 1.00 cudaMemcpyAsync); **0.00 cuLaunchKernel serving**; **1.00 cudaDeviceSynchronize (65.595 µs/call) + 1.00 cudaDriverGetVersion (0.1683 µs/call) per call — RECORDED as the R-term** (pre-declared pessimistic branch (c) materialized and quantified; the source contains zero sync calls — the sync is inside the framework's `CUDAGraph.replay()` on this build) | **pass** | `log/diagnostic_scope_census_round003.json` |
| rterm_transfer_at_bsz2 | canonical bsz=2 R-term vs sibling 69.02 µs/call (bsz=1); >±5 µs deviation re-prices future graph decisions | **R-term(bsz=2) = 65.76 µs/call API-sum** (sync 65.595 + driverGet 0.168) vs sibling 69.02 → **TRANSFERS within 3.42 µs** (inside the ±5 µs materiality band). Decomposition now measured: idle-device sync overhead 15.21 µs + wait absorbing the in-graph round-trip (64.47 µs event-timed) | **observed** (transfer confirmed) | `log/diagnostic_scope_census_round003.json`, `log/probes/verifier_r003_kernel_in_graph.json` |
| device_us_per_call | composed attributed band: kernel-in-graph ~15.3–19.6 + copy-out ~3.7; attribution may coarsen per branch-B | attribution coarsened to the EXTREME: graph-interior kernels emit NO cat=kernel events on this build (whole trace = exactly 100 kernel events, all vendor/reference). Census substitutes: **DtoD copy-out 5.4468 µs/call visible; in-graph single-launch replay round-trip 64.4673 µs/call event-timed** (min 63.81 / max 86.72); same-session direct-launch event-timed control 18.3654 µs | **observed** (census substitution) | `log/diagnostic_scope_census_round003.json`, `log/probes/verifier_r003_kernel_in_graph.json` |
| cross_tier_bitwise_retention | tier-1/2/3 bitwise-equal through both surfaces; stale-trap correctness; poisoned ×2; composed == r002 kernel bits | all verified (6-way bitwise on every suite; freshness both directions; tier-3 zero-artifacts) | **pass** | `log/probes/verifier_r003_result.json` |
| proven_envelope_binding_audit | dot sites (32,32) fp32; num_warps=2 single site; bounded state; zero forbidden strings | all re-verified over final bytes (4 dots, 6 widens, single nw=2 site, 9 forbidden tokens zero, bounded-state audit exact) | **pass** | `log/probes/verifier_r003_result.json`, `log/probes/binding_statement_report_003.json` |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-003`
- intervention: F2 composition — three-tier graph-replay chain (direct-address replay → copy-in replay → eager) around the byte-identical r002 kernel; per timed call: data_ptr guard + ONE cudaGraphLaunch + ONE DtoD copy-out; the r002 python launcher never executes on the serving route
- expected_causal_chain: every link measured — cn.graph-replay-direct-address → cn.launcher-python-time CONFIRMED (launcher neutralized: 0.00 cuLaunchKernel serving vs r002's 1.00; the +84.57 µs launcher tax is absent from the route); cn.graph-replay-direct-address → cn.boundary-cost-delta measured (aten 55.36 + graphLaunch 6.06 + memcpyAsync 4.88 + DtoD 5.45 + sync absorbing the graph round-trip); cn.launcher-python-time → cn.wall-time CONFIRMED (the composition converts the r002 direct wall 0.231689 ms to 0.142327 ms — the launcher removal transfers ~1:1 into wall); cn.device-time-delta measured (in-graph round-trip 64.47 event-timed, hidden inside the sync wait; DtoD 5.45 visible); net wall −7.612 µs/call vs base (protocol basis)
- primary_metric: `wall_time`
- Hypothesis verdict: `confirmed` — the mechanism chain is measured end-to-end and the intervention achieved the win branch the decision priced as unreachable (adoption bar cleared on the protocol statistic at +5.0767%, vs the priced band 0.94–1.01x). The decision's honest 0.0 expectation was falsified in the FAVORABLE direction (see the falsification statement under evidence_for_next_round).

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted` (dual-scope forward trace + the four closing censuses)
- profiler_device_time: `available for the reference scope and the copy-out; graph-interior kernels are invisible to the kineto kernel tracer on this build — census substitution (branch-B extreme)`
- mode deviation (D1, standing): canonical `profile_mode=kernel` fails inside harness `make_profile_call` — `TypeError: ModelNew.run_out() missing 2 required positional arguments: 'value' and 'out'`, exit 1 (`log/r003_kernel_mode_attempt.txt`, attempted this round). Forward-mode dual-scope fallback used (pw=20/pi=100 at regime values).
- summarizer deviation (D2′, NEW FORM): `summarize_trace.py` SUCCEEDED on the candidate scope this round (no overlap error) but attributes only 0.07/call vendor span-edge leakage — the graph-replayed Triton kernel emits no kernel events at all. Census substitution: `log/probes/verifier_r003_scope_census.py` → `log/diagnostic_scope_census_round003.json` + the dedicated in-graph probe `log/probes/verifier_r003_kernel_in_graph.py`.
- iterations: `100` forward calls per scope
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- trace: `log/r003_forward_100iter.pt.trace.json`
- trace_sha256: `1c47a5f91b7eb5f433ec3884b2a04511219b2af794ca3ead2dcbcf2919e9df47`
- scope summaries: `log/r003_summary_reference.json` (canonical), `log/r003_summary_candidate.json` (canonical output — attribution-coarsened, superseded by the census)

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference (canonical summarizer) | 1584.7100 | 15.8471 | 93 | 0.93 | 0.149939 | 0.10569 |
| candidate (census substitution) | 544.68 (DtoD visible) + 64.47/call in-graph (event-timed, tracer-invisible) | 5.4468 + 64.4673 | 99 DtoD + 100 graph-interior | 0.99 + 1.00 | 0.142327 | 0.0383 (visible) / 0.4903 (incl. in-graph) |

### Accepted Reference Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `void ixattnbkd::src::impl::MR::FlashAttnFwdF16Ixmma<128u, 128u, 16u, 64u, 64u, (CausalM_t)0, (AlibiMode_t)0, false, __half, false>(FlashAttnFwdParams, ...)` | 93 | 0.93 | 1584.7100 | 15.8471 |

### Candidate Top Kernels (census substitution — the tracer sees none of them)

| Kernel | Count total | Count/call | Total us | Us/call | Measurement basis |
|---|---:|---:|---:|---:|---|
| `_mm_encoder_attn_fwd` (in-graph, single-launch replay round-trip) | 100 | 1.00 | 6446.73 | 64.4673 | event-timed replays ×100, median (graph processing ~46 µs + kernel math ~18 µs) |
| `Memcpy DtoD (Device -> Device)` (copy-out) | 99 | 0.99 | 544.68 | 5.4468 | gpu_memcpy events (visible) |

### Host census — the composed serving route (per call)

| Signal | accepted_reference | candidate (r003 composed) |
|---|---:|---:|
| aten cpu_ops | 33.00 | **3.00** (`empty_like` 18.64 + `empty_strided` 16.05 + `copy_` 20.66 µs) |
| GPU submissions | 1.00 `cudaLaunchKernel` | **2.00** (1.00 `cudaGraphLaunch` 6.06 µs + 1.00 `cudaMemcpyAsync` 4.88 µs) |
| cuLaunchKernel (python launcher) | — | **0.00** (the +84.57 µs launcher tax never executes) |
| sync/driver queries | 0.00 model-code | **1.00 `cudaDeviceSynchronize` (65.60 µs) + 1.00 `cudaDriverGetVersion` (0.17 µs)** — framework-intrinsic replay cost (R-term) |
| memcpys | 0 | **1.00 DtoD** (copy-out; ZERO copy-ins — tier-1 direct-address) |
| graph submissions | 0 | **1.00** |

### The four closing censuses

**(a) tier1_hit_rate + submission census — ENGAGED.** 1.00 `cudaGraphLaunch` + 1.00 `cudaMemcpyAsync` + 0.99 DtoD (recorder margin) per call; 0.00 `cuLaunchKernel` serving; 3.00 aten ops; 0 copy-ins; 0 timed-segment recaptures (exactly 100 graph launches for 100 profiled iterations). The wall itself corroborates at scale: 0.137–0.144 ms across all 8 invocations (a launcher-executing route walls at ~0.231 ms; a tier-2 route would show 3 copy-ins).

**(b) rterm_transfer_at_bsz2 — TRANSFERS.** R-term(bsz=2) = 65.76 µs/call (sync 65.595 + driverGet 0.168) vs sibling bsz=1 69.02 → transfer within 3.42 µs, inside the decision's ±5 µs materiality band. Decomposition newly measured: idle-device sync overhead 15.21 µs; the remainder of the sync duration is the WAIT for the in-graph round-trip (64.47 µs event-timed). The R-term is not a pure host overhead — it absorbs the graph's device-side execution.

**(c) kernel-in-graph regime adjudication — RESOLVED, and the p13 bias question with it.** The single-launch graph replay round-trip = **64.4673 µs/call** (event-timed, ×100 median) vs the same-session direct-launch control 18.3654 µs (event-timed) vs r002's attributed 19.5550. Findings: (i) the single-launch graph's device-side processing costs ~46 µs OVER the kernel math — the graph frontend dominates; (ii) the p13 probe's 15.317 µs measured a 100-launch graph (per-kernel amortization), NOT the candidate's single-launch serving regime — the "+4.2–4.7 µs probe-vs-attributed bias" resolves as: for DIRECT launches event-timed ≈ attributed (18.37 vs 19.555, ~1.2 µs), and the replay regime is a different cost structure entirely; (iii) the in-graph kernel math itself is unchanged (~18 µs — the byte-identical kernel).

**(d) deliverable ledger decision — SUBMISSION = r003 composed.** The composed candidate's speedup vs base = **1.0486x–1.0565x** (every estimator: 5-pair 1.0486, 3-pair protocol 1.0535, 8-pair 1.0565) vs the r002 direct deliverable's 0.6258x — domination on every estimator, correctness-PASS with full 6-way bitwise retention and the r001/r002 numerics pedigree. Per the project DELIVERABLE RULE the campaign's submission is `triton_mm_encoder_attention_e2_003.py` @`d503e845d1d95ad033e616730d78d67d0775df3a473b17a3c37843d463654d81` — a **+5.08% faster-than-base** Triton submission (the first candidate in this campaign to beat base at all).

### Wall decomposition of the composed route (measured, µs/call)

| Component | Value |
|---|---:|
| aten host (empty_like 18.64 + empty_strided 16.05 + copy_ 20.66) | 55.36 |
| cudaGraphLaunch host API | 6.06 |
| cudaDeviceSynchronize (R-term: 15.21 overhead + wait absorbing the 64.47 round-trip) | 65.60 |
| cudaMemcpyAsync host API | 4.88 |
| DtoD copy-out device | 5.45 |
| cudaDriverGetVersion + cudaStreamIsCapturing + python glue | ~5 |
| **Total (≈ candidate wall)** | **≈ 142.3** ✓ (measured 0.142327 ms) |

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable (first verification of this candidate) | `d503e845d1d95ad033e616730d78d67d0775df3a473b17a3c37843d463654d81` | all gates passed on the first verifier attempt (correctness probe ALL_OK first try; census clean; the in-graph probe resolved census (c)); candidate bytes never changed |

At most one Verifier-to-Coder repair is allowed in the same round; zero were needed — no candidate defect was found at any point (candidate hash constant end-to-end, matching the coder ledger).

## evidence_for_next_round

### The falsification statement (loud, as dispatched)

**The decision's priced identity was falsified in the FAVORABLE direction.** Predicted composed net vs base: **+3.28 µs WORSE** (attributed-D basis, 0.978x) to **−0.95 µs** (graph-paced basis, 1.007x); the pre-declared honest band was 0.94–1.01x and the win branch was priced unreachable ("needs device ≤ 9.2 µs, 10.36 µs below the authoritative floor"). Measured: **net = −7.612 µs/call (protocol basis) / −7.998 µs (8-pair basis)** → 1.0508x–1.0565x. The composition beat the decision's optimistic branch by ~6.6–7.1 µs/call and CLEARED the +5% adoption bar on the protocol statistic. Root causes of the mispricing, now measured: (i) the identity treated R(69.02) and D_kernel-in-graph as ADDITIVE, but the sync API ABSORBS the graph round-trip wait (measured: sync 65.60 = overhead 15.21 + wait ~50–64); (ii) the boundary's aten component is 55.36 µs/call (empty_like + empty_strided + copy_ dispatch), not ~2 µs; (iii) the replaceable base host stack is ~131 µs/call (33-op aten + launch), larger than T_launcher alone — error cancellation landed the optimistic branch ~7 µs low. NOT falsified: T_launcher invariance (84.77/84.57 across two rounds — CONFIRMED, and the launcher is verifiably ABSENT on the serving route: 0.00 cuLaunchKernel) and the R-term transfer (65.76 vs sibling 69.02 — TRANSFERS).

### Campaign physics closure (the dispatch's four closing lines)

- **T_launcher invariance**: +84.7651 (r001) / +84.5712 (r002) µs/call — measured twice, stable within 0.19 µs; this round the launcher is REMOVED from the serving route (0.00 cuLaunchKernel, census-verified). The launcher tax was the entire direct-family failure mode; the composition is the only route that removes it.
- **D_cand floor trajectory**: 28.2030 (r001 nw1 direct) → 19.5550 (r002 nw2 direct) → **64.4673 in-graph single-launch round-trip** (of which kernel math ~18.37 unchanged; graph frontend ~46 build-intrinsic). The direct-launch floor stands at 19.555; the replay regime does NOT run the kernel faster — its value is the host-side saving, not device speed.
- **R-term transfer verdict**: TRANSFERS (65.76 at bsz=2 vs 69.02 sibling bsz=1, within 3.42 µs); decomposition measured for the first time in this lineage (idle overhead 15.21 + wait absorbing the graph round-trip).
- **Capability matrix (final)**: fp16-operand tl.dot @fp32acc — COMPILES, runs 8.6–11.7 µs/call kernel-only, EXACTNESS-NEGATIVE on extreme suites at every warp count (vendor-class one-hot tie-flip); num_warps — 2 optimal, 4 no gain, warp-count output-invariant; Triton-launch capturability — PROVEN (manual torch.cuda.CUDAGraph capture of the Triton launch works on this build, p13 + this round at scale); kineto kernel tracer — BLIND to graph-interior kernels (attribution via API census + CUDA events); reduction.sum — BLOCKED throughout (waiver never granted).

### Remaining observed levers (evidence only; selection belongs to Designer/Orchestrator)

- The composed route's aten boundary is 55.36 µs/call, of which ~34.7 µs is the fresh-destination allocation (empty_like + empty_strided). A future family revisiting the allocation strategy (within its own contract) attacks ~35 µs of the 142.3 µs wall — the largest remaining attributable host term.
- The sync-absorbed graph round-trip (64.47 µs) and the idle sync overhead (15.21 µs) are build-intrinsic on this rig; the 5-tile kernel math (~18 µs) is at its proven-envelope floor.
- Marginality warning for any successor round: the +5.08% protocol pass has a 0.077 pp margin; a successor must not assume a large cushion over the bar.

## Stop Recommendation

- recommendation: `continue`
- evidence: accepted round (streak resets; round budget 3/20; counters alive); the adoption bar was cleared on the protocol statistic with the mechanism census-confirmed and the candidate winning all 8 invocations. The Orchestrator declared this the final round and owns the transition: closing NOW at the 1.05x deliverable with the complete physics map is fully defensible (the campaign's stated terminal product is banked and improved from 0.6258x to ~1.05x); continuing attacks the ~35 µs allocation boundary with a fresh streak. Note the boundary-class nature of the acceptance (marginality table above) when weighing.

Orchestrator owns the stop transition.

## Exact Reproduction Commands

Correctness + first authoritative pair (pairs 2–8 identical):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/bi150/epoch2/triton_mm_encoder_attention_e2_003.py --warmup 50 --repeat 100 --full-traceback
```

Kernel-mode attempt (records the D1 arity deviation verbatim, exit 1):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/bi150/epoch2/triton_mm_encoder_attention_e2_003.py --warmup 5 --repeat 10 --profile --profile-mode kernel --profile-warmup 20 --profile-iterations 100 --profile-output kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/r003_kernel_mode_attempt.pt.trace.json --full-traceback
```

Dual-scope profiler (forward-mode fallback, pw=20/pi=100) + per-scope normalization + censuses:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/bi150/epoch2/triton_mm_encoder_attention_e2_003.py --warmup 50 --repeat 100 --profile --profile-reference-file kernels/track1-triton/mm_encoder_attention/bi150/epoch2/baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 100 --profile-output kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/r003_forward_100iter.pt.trace.json
/usr/local/bin/python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/r003_forward_100iter.pt.trace.json --iterations 100 --scope reference_baseline_adapter --wall-ms 0.149939
/usr/local/bin/python3 kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/probes/verifier_r003_scope_census.py
/usr/local/bin/python3 kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/probes/verifier_r003_kernel_in_graph.py
```

Verifier correctness probe:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/local/bin/python3 kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/probes/verifier_r003_correctness.py
```

Artifact hash ledger (re-verified this round, before and after all measurement):

```text
d503e845d1d95ad033e616730d78d67d0775df3a473b17a3c37843d463654d81  triton_mm_encoder_attention_e2_003.py
0a678da87a877b9c521b6c280eb3518b20f98e352786e9df129435e2cc918413  rounds/decision_003.md
bdf423556e7c80369ae38d4980529a739a52a3d18033e572927354b23e0a4e64  rounds/sketch_003.json
cc98318bbfed0d9cdbf7a99769c4a44077dc15590496a4e8410ba9eda99f3078  triton_mm_encoder_attention_e2_002.py (kernel byte-identical segment)
20b21646d9c3ba3abe086d8133799d23a39981dcb4e1cb547e1a3f65b0bf7ffc  rounds/report_000.md
c3980a2c4ae79171c73c7cf85c29c62f6ac98b9e9fb643c249fc9a6e0a921c9f  baseline_adapter.py
86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2  ../../base.py
71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29  auto_bench.py
4b3985a81b134cc947ae2cbaf1436e67885365a9be54fda8aef6961e5779c9b6  log/probes/binding_statement_report_003.json
1c47a5f91b7eb5f433ec3884b2a04511219b2af794ca3ead2dcbcf2919e9df47  log/r003_forward_100iter.pt.trace.json
```

## vNext Fact Pack

```json
{
  "schema_version": 1,
  "candidate_sha256": "d503e845d1d95ad033e616730d78d67d0775df3a473b17a3c37843d463654d81",
  "correctness": {
    "status": "pass",
    "evidence": [
      "auto_bench.py PASS accuracy in all 8 authoritative pairs + profile run (9/9 invocations, seed42 canonical regime)",
      "log/probes/verifier_r003_result.json all_ok=true: seed42 max_abs 4.883e-04, fp16-extreme vs fp32 ground truth 3.052e-05 (byte-identical kernel pedigree), boundary suite, non-target shapes B1S41/B2S82/B2S96 through tier-3 with ZERO graph artifacts, 6-way bitwise retention (tier-1 x2 / tier-2 / r002 twin / run_out poisoned x2) on every target-regime suite, stale-address impossibility verified both directions, bounded-state audit exact (budget 3, bound_sets 2, 2 graph handles), 9 forbidden tokens zero, 4 dot sites (32,32) fp32, single num_warps=2 site, 6 widening casts, kernel byte-identity vs r002 machine-verified 4168/4168 chars"
    ]
  },
  "observables": [
    {
      "name": "wall_time_unrounded_paired_median_ms",
      "status": "observed",
      "value": "+5.076731% (protocol statistic: reference 0.149939 ms vs candidate 0.142327 ms over the three prescribed pairs; bar +5.0% CLEARED by 0.077pp — BOUNDARY-CLASS; extended estimators straddle: 8-pair median +5.3451% PASS, 5-pair +4.6355% FAIL, clean per-pair mean +4.679% FAIL; win rate 8/8 invocations; the decision's priced band 0.94-1.01x was exceeded — favorable-direction falsification)",
      "confidence": "medium-high (mechanism census-confirmed; estimator spread documented)",
      "evidence": ["log/r003_pair_001_timing.txt", "log/r003_pair_002_timing.txt", "log/r003_pair_003_timing.txt", "log/r003_pair_004_timing.txt", "log/r003_pair_005_timing.txt", "log/r003_pair_006_timing.txt", "log/r003_pair_007_timing.txt", "log/r003_pair_008_timing.txt"]
    },
    {
      "name": "tier1_hit_rate_in_timed_regime",
      "status": "observed",
      "value": "100% within recorder margin: 1.00 cudaGraphLaunch + 1.00 cudaMemcpyAsync + 0.99 DtoD (99/100 span-edge) per call, 0.00 cuLaunchKernel serving, 0.00 copy-ins, 0 timed-segment recaptures (100 graph launches = 100 profiled iterations); wall corroborates at scale (0.137-0.144 ms vs the launcher route's 0.231 ms)",
      "confidence": "high",
      "evidence": ["log/diagnostic_scope_census_round003.json"]
    },
    {
      "name": "aten_cpu_ops_per_call",
      "status": "observed",
      "value": "3.00/call (aten::empty_like 18.64us + aten::empty_strided 16.05us + aten::copy_ 20.66us) — well under the <=6 bound; the ~34.7us fresh-destination allocation is the largest remaining attributable host term",
      "confidence": "high",
      "evidence": ["log/diagnostic_scope_census_round003.json"]
    },
    {
      "name": "submission_and_sync_census",
      "status": "observed",
      "value": "exactly 2.00 GPU submissions/call (1.00 cudaGraphLaunch 6.06us + 1.00 cudaMemcpyAsync 4.88us), ZERO cuLaunchKernel on the serving route; per-call cudaDeviceSynchronize 65.595us + cudaDriverGetVersion 0.168us OBSERVED and RECORDED as the build-intrinsic replay cost (framework-internal to CUDAGraph.replay on this build; source contains zero sync strings)",
      "confidence": "high",
      "evidence": ["log/diagnostic_scope_census_round003.json"]
    },
    {
      "name": "rterm_transfer_at_bsz2",
      "status": "observed",
      "value": "R-term(bsz=2) = 65.76 us/call API-sum (sync 65.595 + driverGet 0.168) vs sibling bsz=1 69.02 -> TRANSFERS within 3.42 us (inside the +/-5 us materiality band); decomposition measured for the first time: idle-device sync overhead 15.21 us + wait absorbing the in-graph round-trip (64.47 us event-timed)",
      "confidence": "high",
      "evidence": ["log/diagnostic_scope_census_round003.json", "log/probes/verifier_r003_kernel_in_graph.json"]
    },
    {
      "name": "device_us_per_call",
      "status": "observed",
      "value": "attribution coarsened to the extreme (graph-interior kernels emit NO cat=kernel events on this build); census substitution: DtoD copy-out 5.4468 us/call visible + in-graph single-launch replay round-trip 64.4673 us/call (event-timed x100 median; graph frontend ~46 us over unchanged ~18.37 us kernel math — same-session direct-launch control) vs r002 direct attributed 19.5550",
      "confidence": "high",
      "evidence": ["log/diagnostic_scope_census_round003.json", "log/probes/verifier_r003_kernel_in_graph.json"]
    },
    {
      "name": "cross_tier_bitwise_retention",
      "status": "observed",
      "value": "6-way bitwise (tier-1 x2 / tier-2 / r002 twin / run_out poisoned x2) TRUE on every target-regime suite; stale-address impossibility verified both directions (new-pointer set + in-place mutation freshness); non-target shapes through tier-3 with zero graph artifacts; composed outputs bitwise-equal to the r002 kernel on identical bits",
      "confidence": "high",
      "evidence": ["log/probes/verifier_r003_result.json"]
    },
    {
      "name": "proven_envelope_binding_audit",
      "status": "observed",
      "value": "kernel byte-identical to r002 (machine-verified 4168/4168): 4 tl.dot sites (32,32)@(32,32) fp32 widened (6 casts), single num_warps=2 site, num_stages absent, 9 forbidden tokens zero (incl. torch.compile/synchronize/DriverGet/.contiguous(), bounded-state audit exact (2 graph handles, workspaces, anchors, counter 3 remaining, monotone flags)",
      "confidence": "high",
      "evidence": ["log/probes/verifier_r003_result.json", "log/probes/binding_statement_report_003.json"]
    }
  ],
  "lowering": {
    "status": "observed",
    "expected_mechanism": "present — the composed route device-executes the byte-identical r002 kernel via graph replay (1.00 cudaGraphLaunch/call; in-graph round-trip 64.4673 us/call event-timed; kernel math ~18.37 us unchanged) plus one DtoD copy-out (5.4468 us/call); the kineto kernel tracer is blind to graph-interior kernels — attribution via API census + CUDA events (branch-B extreme)",
    "evidence_contract": "triton_cuda-v1 (proven-envelope dots consumed exactly as declared; manual CUDAGraph capture of the Triton launch PROVEN at scale; P1-P4 ladder untriggered)",
    "evidence": ["log/diagnostic_scope_census_round003.json", "log/probes/verifier_r003_kernel_in_graph.json"]
  },
  "evidence_gap_cause": "none"
}
```

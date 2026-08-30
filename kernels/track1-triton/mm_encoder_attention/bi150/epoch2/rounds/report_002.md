# Report 002

Result: no-improvement

## Identity

- Round: `002`
- Decision: `rounds/decision_002.md` @`20b360ac936bf4d9d41afadac90c40578f0a758e628ec40af2d3c759eb22d3fb` (hash re-verified live; F3 triton-attention-kernel-config, expected_wall_improvement_pct 0.0 declared honestly)
- Candidate: `triton_mm_encoder_attention_e2_002.py`
- Accepted reference: `baseline_adapter.py` (last_accepted_kernel; byte-equivalent pipeline to base.py)
- Accepted reference report: `rounds/report_000.md` @`20b21646d9c3ba3abe086d8133799d23a39981dcb4e1cb547e1a3f65b0bf7ffc` (hash re-verified live)
- Decision SHA256: `20b360ac936bf4d9d41afadac90c40578f0a758e628ec40af2d3c759eb22d3fb` (file) / `a8ff185bcc6fdea5b3a6ac60a356ab4e7ef32fd1f6c8c36987548f0a572e5f0c` (canonical metadata, verdict basis)
- Sketch SHA256: `c16b1528b25ae1a3bbfc72b3e459462505d940677e62b30a0585e3b41b46e9e9` (file; canonical `4defe199…b36c2`, verdict basis)
- Candidate SHA256: `cc98318bbfed0d9cdbf7a99769c4a44077dc15590496a4e8410ba9eda99f3078` (re-verified before AND after all runs; AST parse OK)
- Accepted reference SHA256: `c3980a2c4ae79171c73c7cf85c29c62f6ac98b9e9fb643c249fc9a6e0a921c9f`
- Base SHA256: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` (unchanged, re-verified after all runs)
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (unchanged, AST loader)
- Binding: `log/probes/binding_statement_report_002.json` @`322d932902fb161d7f4529576db972f1a858b2a731ff31a4f8b9f2ca1bde3863` (file; canonical `394964fd…8ea47`) — verifier independently re-verified every checkable claim: diff-verified the ONLY functional delta vs r001 @`4171de8d…` is `num_warps=1,` → `num_warps=2,` at the single launch site (kernel arithmetic textually identical; remaining deltas are docstrings/comments); 9-token DANGER scan all-zero over final bytes; 4 dot sites; single `num_warps=2` site; 6 widening casts; zero `contiguous` occurrences; stateless audit; run_out 4-arg + returns None; r002-vs-r001 outputs BITWISE-equal (spot-verified on every verifier suite)
- Runtime fingerprint: `project.md#runtime-fingerprint` (unchanged since r000 live probe; environment re-bootstrapped every shell)
- Measurement fingerprint: `0c4c7d664c85e65d0580091ca5e3a77ff769a0d28f7e679f5bdf78fe5d0d966e` (recomputed live this round; base/harness bytes re-verified identical before and after timing)
- verification_tier: `authoritative`
- screening_pairs: `not-run: correct candidate proceeded directly to authoritative timing (r001 precedent; a screen-out would skip the profiler and destroy the round's mandated D_cand/host-invariance measurement duties)`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness (seed42 canonical) | `allclose(atol=1e-2, rtol=1e-2, equal_nan=True)` vs base.py, seed 42, fp16 out `[2,83,512]` | `PASS accuracy` in all three authoritative pairs + profile run (4/4 invocations); probe max_abs 4.883e-04 | pass | `log/r002_pair_00{1,2,3}_timing.txt`, `log/r002_profile_run.txt`, `log/probes/verifier_r002_result.json` |
| fp16-extreme suite | extreme-magnitude inputs vs fp32 ground truth (r001-established basis) | candidate max_abs 3.052e-05 — bit-for-bit the r001 reading (identical arithmetic, warp-count invariant); vendor base diverges 1457 on the same inputs (standing vendor precision-limit observation) | pass | `log/probes/verifier_r002_result.json` |
| non-target shapes | correctness off the target shape (stateless recompile routing) | B1S41 seed7, B2S96 seed19, B2S82 seed13 all PASS vs base (max_abs 4.883e-04), run_out bitwise, shape/dtype ok | pass | `log/probes/verifier_r002_result.json` |
| r002 ≡ r001 bitwise (coder claim, spot-verified) | kernel arithmetic identical ⇒ outputs bitwise-equal at nw=2 | BITWISE-EQUAL on every verifier suite (seed42, extreme, B1S41, B2S96, B2S82): `torch.equal` true on all five — the config change is output-invariant exactly as claimed | pass | `log/probes/verifier_r002_result.json` (`r002_equals_r001_bitwise: true` per suite) |
| run_out contract | `run_out(q,k,v,out) -> None`, caller buffer filled directly, bitwise==forward | poisoned caller buffers ×2 (−777.0 and +555.0, both orderings) bitwise-equal to forward with data_ptr preserved; returns None; single launch | pass | `log/probes/verifier_r002_result.json` |
| forward bitwise-stability | deterministic kernel, no atomics | repeat identical-input calls bitwise-stable on every suite (fresh allocations) | pass | `log/probes/verifier_r002_result.json` |
| stateless module | zero call-time instance state | `__dict__` delta across forward+run_out calls = empty | pass | `log/probes/verifier_r002_result.json` |
| capability legality | every tl.dot at (32,32)@(32,32) fp32 (widened); num_warps=2 at the single launch site (probe-qualified); num_stages absent | 4 dot sites, 6 widening casts, single launch site `num_warps=2`, `num_stages` absent — identical envelope to r001; the ONLY new capability is num_warps=2, qualified by the pre-adoption sweep (p13, reviewed) and independently exactness-verified by this probe | pass | `log/probes/verifier_r002_result.json`, `log/probes/p13_r002_sweep_result.json` |
| no compile/graph machinery | zero compile/capture/graph/autotune tokens | all re-scanned tokens count 0 over final candidate bytes | pass | `log/probes/verifier_r002_result.json` |
| pre-adoption sweep discipline (decision guardrail) | sweep BEFORE adoption; only exactness-passing configs selectable; pre-declared selection rule honored | sweep evidence reviewed: 6 configs × {compile, exactness, bitwise-stability, device time}; exactness-passing = {nw1_fp32widen 23.492, nw2_fp32widen 15.317, nw4_fp32widen 15.441}; fastest-first eliminates nw1 (outside 0.5 µs tie band); nw2/nw4 tied → fewer-new-capabilities (both 1) → lower num_warps → nw2 SELECTED — rule trace honored exactly; fp16-operand dots failed exactness on the extreme suite (max_abs 1459, vendor-class one-hot tie-flip signature) at every warp count ⇒ capability-negative, not selectable | pass | `log/probes/p13_r002_sweep_result.json` @`9472cd8b…84f` |
| default-stream discipline | all invocations on the harness default route | every measurement ran the unchanged harness default path; zero stream manipulation | pass | command history |

Conformance, correctness, and every declared guardrail passed.

## Screening Evidence

Not run — correct candidate proceeded directly to authoritative timing (r001 precedent). The round's contractual products are the authoritative D_cand(nw2) (the number that gates r003) and the host-invariance census, which require the profiler a screen-out would skip.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved ordered reference/candidate pairs — three identical harness invocations (pair i = invocation_i.v0_ms then invocation_i.v1_ms), byte-for-byte identical flags, interpreter, device, default-stream route, CoreX environment`
- reference_raw_samples_ms: `[0.146358, 0.144984, 0.144069]`
- candidate_raw_samples_ms: `[0.232103, 0.231037, 0.231689]`
- reference_median_ms: `0.144984`
- candidate_median_ms: `0.231689`
- improvement_pct: `-59.803150692490206`

```text
improvement_pct = (0.144984 - 0.231689) / 0.144984 * 100 = -59.803151
```

| Independent invocation | Reference wall ms | Candidate wall ms | Speedup | Evidence |
|---:|---:|---:|---:|---|
| 1 | `0.146358` | `0.232103` | `0.631x` | `log/r002_pair_001_timing.txt` |
| 2 | `0.144984` | `0.231037` | `0.628x` | `log/r002_pair_002_timing.txt` |
| 3 | `0.144069` | `0.231689` | `0.622x` | `log/r002_pair_003_timing.txt` |

BELOW the 5.0% adoption bar with a decisively NEGATIVE sign: candidate wall 0.231689 ms lands INSIDE the dispatch's expected honest band (~0.225–0.245 ms) and the decision's declared band (222–241 µs). Extra (non-authoritative) profile-run timing: v0=0.145125, v1=0.231593 ms.

### Anchor bases (all four, explicitly)

1. **Prescribed paired v0-basis headline**: v0=base.py 0.144984 ms vs v1=candidate 0.231689 ms, same session → **−59.8032%** (adoption-decisive basis).
2. **Direct same-session pair vs r000 v0**: last_accepted_kernel IS the base adapter, so the prescribed paired-v0 basis and the direct pair vs r000 v0 are THE SAME comparison this round — stated explicitly.
3. **Cross-anchor `report_000` 0.150149 ms**: candidate 0.231689 → −54.3061%. Session-drift context: this session's v0 median sits −3.4399% below r000's (and −0.2690% below r001's 0.145375); same-session pairing absorbs it.
4. **Manifest anchor**: identical to the report_000 anchor (no accepted round committed; r000 completion remains the only prior) — stated explicitly: manifest anchor wall = 0.150149 ms, cumulative context −54.3061%.

Deliverable-ledger note (per dispatch): the banked direct deliverable IMPROVES from r001 — 0.240953 → 0.231689 ms (−9.264 µs wall, consistent with the −8.648 µs attributed device cut), speedup vs base 0.6033x → 0.6258x, with outputs bitwise-equal to r001 (zero behavioral change) — the deliverable pointer should move to the nw2 config in the deliverable ledger (canonical performance pointer unchanged: no-improvement does not move last_accepted).

ABAB interleaved control: not run — measured delta (−59.80%) is ~12× the 5.0% bar; v0 session drift −3.44% vs r000 / −0.27% vs r001, both absorbed by same-session pairing; no plausible drift magnitude affects the classification (dispatch made ABAB optional).

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| wall_time_unrounded_paired_median_ms | ≥5% below accepted reference — honestly declared unreachable (expected band 222–241 µs) | −59.8032% (0.231689 vs 0.144984 ms); wall 0.231689 ms INSIDE the pre-declared band (222–241 µs / dispatch 0.225–0.245) | **fail** (vs the ≥5% adoption criterion; the honest 0.0 expectation realized exactly) | `log/r002_pair_00{1,2,3}_timing.txt` |
| kernel_config_capability_probe | pre-adoption sweep over num_warps {1,2,4} × dot-operand {fp32-widened, fp16@fp32acc}; pre-declared readings (i)/(ii)/(iii) | Reading (i) realized: out-of-envelope configs DID exactness-pass and beat 28.203 µs — nw2_fp32widen (15.317 µs probe-method) SHIPPED under the pre-declared fastest-exactness-passing rule (tie band 0.5 µs → nw2/nw4 tied → fewer-new-capabilities → lower nw → nw2; rule trace honored exactly). Capability resolutions banked: fp16-operand dots COMPILE but FAIL exactness on the extreme suite (max_abs 1459, vendor-class one-hot tie-flip signature) at every warp count ⇒ Ixmma-MMA-via-Triton capability-NEGATIVE for this lineage (cross-operator fact); nw4 buys nothing over nw2 (15.441 vs 15.317, within noise) | **pass** (evidence reviewed; selected config independently exactness-verified) | `log/probes/p13_r002_sweep_result.json`, `log/probes/verifier_r002_result.json` |
| device_us_per_call | D_new bands: ≥25 no headroom; 18–25 partial (F2 net +1.5 to +8.5 µs WORSE, sub-parity); ≤~16.5 parity-class; ≤~9.2 win-class | **D_cand(nw2) = 19.5550 µs/call AUTHORITATIVE** (attributed GPU-span projection, 100/100; whole-trace cross-check identical; vendor Ixmma 17.4212 whole-trace / 15.3582 attributed) — **18–25 partial band**: net_F2 = 19.555 − 16.455 = **+3.10 µs/call WORSE than base** (sub-parity 0.94–0.96x composed deliverable class, inside the decision's +1.5 to +8.5 range). The dispatch's ~15–16 branch did NOT materialize on the attributed basis; parity-class (D ≤ 16.5) NOT unlocked (+3.06 µs short); win-class NOT unlocked (+10.36 µs short) | **observed** (band: partial) | `log/diagnostic_scope_census_round002.json` |
| aten_cpu_ops_per_call | ≤3/call, single aten::empty expected; a material change falsifies kernel-only scope | **1.00/call (single `aten::empty`)** — unchanged from r001 | **pass** | `log/diagnostic_scope_census_round002.json` |
| launch_and_submission_count_per_call | exactly 1.00 kernel launch, zero memcpys/graphs/syncs | **1.00 `cuLaunchKernel`/call** (cuda_driver API, 3.66 µs/call host); **0 memcpys, 0 graph submissions, 0 model-code syncs** — the config change is host-INVISIBLE exactly as the kernel-only scope claims | **pass** | `log/diagnostic_scope_census_round002.json` |
| run_out_bitwise_equals_forward | bitwise over poisoned buffers ×2, data_ptr preserved; forward bitwise-stable | true on all five suites; returns None | **pass** | `log/probes/verifier_r002_result.json` |
| capability_legality_binding_audit | dot sites (32,32) fp32; num_warps = shipped value; num_stages absent; zero DANGER tokens; stateless | 4 dot sites (32,32) fp32 widened (6 casts); single `num_warps=2` site; `num_stages` absent; 9 DANGER tokens zero; zero `.contiguous`; stateless — re-verified over final bytes | **pass** | `log/probes/verifier_r002_result.json`, `log/probes/binding_statement_report_002.json` |
| triton_launcher_tax_invariance | T_launcher stays in the +80 to +90 µs/call band (r001 +84.765) | **+84.5712 µs/call net** (whole-trace basis; Δwall 86.705 − Δdevice 2.134) — INSIDE the band, −0.19 µs vs r001: the configuration change touched ONLY the device term, exactly as the kernel-only scope claims | **pass** | `log/diagnostic_scope_census_round002.json` |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-002`
- intervention: kernel execution-configuration change on the unchanged r001 boundary — num_warps 1→2 at the single launch site (the only functional delta, diff-verified), selected by the pre-adoption capability sweep under the pre-declared fastest-exactness-passing rule
- expected_causal_chain: every link measured — cn.kernel-config → cn.register-occupancy CONFIRMED (spill-class nw1 → 2-warp occupancy: device 28.203 → 19.555 µs/call attributed, −30.66%); cn.kernel-config → cn.mma-path resolved capability-NEGATIVE (fp16-operand dots fail exactness; not shipped); cn.kernel-config → cn.device-time-delta measured (−8.648 µs attributed / −8.175 µs probe-method); cn.device-time-delta → cn.wall-time CONFIRMED (wall 0.240953 → 0.231689 ms, −9.264 µs — device cut transfers ~1:1 to wall since T_launcher is invariant); cn.kernel-config → cn.wall-time (host-invisible: aten 1.00, launches 1.00, T_launcher invariance band pass)
- primary_metric: `wall_time`
- Hypothesis verdict: `partially-confirmed` — the occupancy mechanism is real and measured end-to-end (device −30.66%, wall −9.26 µs, bitwise-identical outputs, host fully invariant); the ≥5% wall criterion FAILED exactly as the honest 0.0 expectation predicted (direct-family wall arithmetic stays closed: T_launcher +84.57 µs dominates). Adoption is governed by wall_time alone ⇒ `no-improvement`. The round's declared products — capability qualification + the D_cand cut — were both banked.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted` (dual-scope forward trace + host-invariance census + authoritative D_cand)
- profiler_device_time: `available: candidate kernel attributed 100/100 under the GPU-span projection; whole-trace cross-check identical`
- mode deviation (D1, standing): canonical `profile_mode=kernel` fails inside harness `make_profile_call` — it invokes `run_out(gating_output, *output_args)` producing `TypeError: ModelNew.run_out() missing 2 required positional arguments: 'value' and 'out'`, exit 1 (`log/r002_kernel_mode_attempt.txt`, attempted this round). The 4-arg run_out is project.md public_contract; no accommodation invented. Fallback: `--profile-mode forward` dual-scope via `--profile-reference-file`, pw=20/pi=100 at regime values.
- summarizer deviation (D2, standing): `summarize_trace.py` succeeds on the reference scope (`log/r002_summary_reference.json`) but errors `overlapping scope events` on the candidate scope (kineto dual-span shape; `log/r002_summary_candidate.json` holds the failure marker, exit 2). Census-substitution: `log/probes/verifier_r002_scope_census.py` → `log/diagnostic_scope_census_round002.json` (host ops on the host span, device kernels on the GPU-span projection, whole-trace cross-check; both bases agree exactly).
- iterations: `100` forward calls per scope
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- trace: `log/r002_forward_100iter.pt.trace.json`
- trace_sha256: `1eb734f91a945fde7b031be50fc22de0e4693342496a22eb00e865b33a7c50e4`
- scope summaries: `log/r002_summary_reference.json` (canonical) + `log/diagnostic_scope_census_round002.json` (candidate-scope substitution)

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference (canonical summarizer) | 1535.8193 | 15.3582 | 88 | 0.88 | 0.144984 | 0.10593 |
| candidate (census GPU-projection) | 1955.50 | 19.5550 | 100 | 1.00 | 0.231689 | 0.08439 |

Attribution notes: (i) the reference scope's 0.88/call attributed count is the stable recorder span-edge margin (88 of 100 vendor kernels fully inside the span; whole-trace 17.4212 µs/call is the paired-decomposition basis); (ii) the candidate host span shows 0.12/call stray vendor-kernel events — span-edge leakage of the PRECEDING reference scope's trailing kernels (the candidate never issues the vendor kernel; 88+12 = the reference's 100); (iii) the candidate kernel is attributed 100/100 under the GPU-span projection at 19.5550 µs/call, identical to the whole-trace total.

### Accepted Reference Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `void ixattnbkd::src::impl::MR::FlashAttnFwdF16Ixmma<128u, 128u, 16u, 64u, 64u, (CausalM_t)0, (AlibiMode_t)0, false, __half, false>(FlashAttnFwdParams, ...)` | 88 | 0.88 | 1535.8193 | 15.3582 (whole-trace: 17.4212) |

### Candidate Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `_mm_encoder_attn_fwd` (num_warps=2) | 100 | 1.00 | 1955.50 | 19.5550 |

### Host census — invariance check vs r001 (the decision observable)

| Signal | accepted_reference | candidate (r002 nw2) | r001 (prior round) | Invariant? |
|---|---:|---:|---:|---|
| aten cpu_ops total/call | 33.00 | **1.00** (single `aten::empty`) | 1.00 | **YES** |
| kernel launches | 1.00 `cudaLaunchKernel` (cuda_runtime, 5.16 µs/call) | **1.00 `cuLaunchKernel`** (cuda_driver, 3.66 µs/call) | 1.00 `cuLaunchKernel` (3.64) | **YES** |
| memcpys (any class) | 0 | **0** | 0 | **YES** |
| graph submissions | 0 | **0** | 0 | **YES** |
| model-code syncs | 0 | **0** | 0 | **YES** |

The configuration change is host-INVISIBLE — the kernel-only scope claim is verified on every host census signal.

### AUTHORITATIVE D_cand(nw2) and the F2 projection (the number that gates r003)

| Quantity | Value (µs/call) | Basis |
|---|---:|---|
| **D_cand(nw2) — AUTHORITATIVE** | **19.5550** | attributed GPU-span projection, 100/100 (whole-trace identical) |
| D_cand(nw1) — r001 attributed | 28.2030 | r001 census, same method |
| device cut (attributed) | −8.648 (−30.66%) | 28.2030 → 19.5550 |
| device cut (probe-method) | −8.175 (−34.80%) | 23.492 → 15.317 (p13 sweep) |
| wall cut (candidate side) | −9.264 | 0.240953 → 0.231689 ms |
| vendor Ixmma (this session) | 17.4212 whole-trace / 15.3582 attributed | reference scope |
| T_launcher (net, whole-trace) | **+84.5712** | Δwall 86.705 − Δdevice 2.134; r001 was +84.7651 (−0.19 µs; +80–90 invariance band PASS) |

Method-delta note (dispatch's explicit ask): the p13 probe-method (graph-assisted kernel-only, back-to-back replay) reads nw1 = 23.492 and nw2 = 15.317 µs; the verifier attributed method reads nw1 = 28.203 (r001) and nw2 = 19.555 µs — a consistent +4.71/+4.24 µs method bias (replay regime compresses kernel time vs profiler-attributed regime). The ATTRIBUTED number is canonical per the report_001 basis the decision's F2 arithmetic uses.

**F2 composition projection (decision-002 arithmetic, net_F2 = D_new − 16.455 µs on the report_001 basis): net_F2 = 19.555 − 16.455 = +3.10 µs/call WORSE than base.** The dispatch's ~15–16 branch (net ≈ −1.1 µs, parity-plus) did NOT materialize on the attributed basis; the projection reverts to WORSE-THAN-BASE (sub-parity): parity-class (D ≤ 16.5) is 3.06 µs short, win-class (D ≤ 9.2) is 10.36 µs short. The composed-graph deliverable at these numbers is the decision's own 0.94–0.96x class — NOT adoption-grade. Evidence only; the F2 proceed/kill decision belongs to the Designer/Orchestrator.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable (first verification of this candidate) | `cc98318bbfed0d9cdbf7a99769c4a44077dc15590496a4e8410ba9eda99f3078` | all gates passed on the first verifier attempt (correctness probe ALL_OK, census clean); no probe-side fixes needed this round; candidate bytes never changed |

At most one Verifier-to-Coder repair is allowed in the same round; zero were needed — no candidate defect was found at any point (candidate hash constant end-to-end, matching the coder ledger).

## evidence_for_next_round

- Observed fact (the r003-gating number): **D_cand(nw2) = 19.5550 µs/call attributed** — the 18–25 µs partial band, NOT the ~15–16 parity-plus branch. F2 graph composition at these numbers: net_F2 = **+3.10 µs/call worse than base** (0.94–0.96x composed deliverable class). Parity-class needs a further −3.06 µs device cut (to ≤16.5); win-class needs −10.36 µs (to ≤9.2).
- Observed fact: the occupancy mechanism is REAL and cleanly measured — nw1→nw2 cuts attributed device time −30.66% (28.203 → 19.555 µs/call) with bitwise-identical outputs, and the cut transfers ~1:1 to wall (−9.26 µs) because T_launcher is invariant (+84.57 vs +84.77; +80–90 band). Host census fully unchanged (1.00 aten::empty, 1.00 cuLaunchKernel, zero memcpys/graphs/syncs).
- Observed fact (capability resolutions banked, cross-operator value): (a) fp16-operand tl.dot @fp32-acc COMPILES on BI150 Triton 3.1.0 and runs 8.6–11.7 µs/call kernel-only — but FAILS exactness on the fp16-extreme suite (max_abs 1459, one-hot tie-flip, vendor-class signature) at EVERY warp count ⇒ Ixmma-MMA-via-Triton is capability-NEGATIVE for exactness-sensitive lineages; (b) num_warps=4 buys nothing over 2 at this tile shape (15.441 vs 15.317 probe-method, within noise); (c) num_warps=2 is output-invariant (bitwise-equal to nw1 on all suites).
- Observed fact (method bias, now measured twice): graph-assisted kernel-only replay reads ~4.2–4.7 µs/call FASTER than profiler-attributed device time for the same kernel (nw1: 23.492 vs 28.203; nw2: 15.317 vs 19.555). Cross-method deltas (−8.175 vs −8.648 µs) agree on the CUT magnitude; absolute levels differ by the replay-regime bias. r003 projections must use the attributed basis (the decision's own arithmetic does).
- Deliverable ledger: banked direct deliverable improves to `triton_mm_encoder_attention_e2_002.py` @`cc98318b…` — 0.6258x vs base (was 0.6033x), outputs bitwise-equal to r001, strictly faster device time, host unchanged. Canonical performance pointer does NOT move (no-improvement).
- Standing build facts carried: D1 kernel-mode arity; D2 kineto dual-span census-substitution; reference-scope attribution margin (0.88–0.90/call); Triton launches via cuda_driver cuLaunchKernel (~3.6 µs/call host API cost); vendor precision limit on extreme inputs (1457 divergence, labeled non-blocking).
- Session drift note: v0 −3.44% vs r000 / −0.27% vs r001 — paired same-session basis absorbs; cross-session anchors are context only.

Evidence only; selection of the next optimization belongs to the Designer.

## Stop Recommendation

- recommendation: `continue`
- evidence: no-improvement #2 (streak 2/3 vs valid_no_improvement_limit 3); round budget 2/20; the round banked its declared products (capability qualification + the authoritative D_cand cut 28.203 → 19.555 µs/call + the improved 0.6258x deliverable); per the decision's staged sequencing the final bullet r003 = the F2 graph composition — whose measured-number projection is now sub-parity (+3.10 µs worse than base, 0.94–0.96x composed class), an honest input the Designer must weigh against an honest close-out.

Orchestrator owns the stop transition.

## Exact Reproduction Commands

Correctness + first authoritative pair (pairs 2 and 3 identical):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/bi150/epoch2/triton_mm_encoder_attention_e2_002.py --warmup 50 --repeat 100 --full-traceback
```

Kernel-mode attempt (records the D1 arity deviation verbatim, exit 1):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/bi150/epoch2/triton_mm_encoder_attention_e2_002.py --warmup 5 --repeat 10 --profile --profile-mode kernel --profile-warmup 20 --profile-iterations 100 --profile-output kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/r002_kernel_mode_attempt.pt.trace.json --full-traceback
```

Dual-scope profiler (forward-mode fallback, pw=20/pi=100) + per-scope normalization + census:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/bi150/epoch2/triton_mm_encoder_attention_e2_002.py --warmup 50 --repeat 100 --profile --profile-reference-file kernels/track1-triton/mm_encoder_attention/bi150/epoch2/baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 100 --profile-output kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/r002_forward_100iter.pt.trace.json
/usr/local/bin/python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/r002_forward_100iter.pt.trace.json --iterations 100 --scope reference_baseline_adapter --wall-ms 0.144984
/usr/local/bin/python3 kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/probes/verifier_r002_scope_census.py
```

Verifier correctness probe:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/local/bin/python3 kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/probes/verifier_r002_correctness.py
```

Artifact hash ledger (re-verified this round, before and after all measurement):

```text
cc98318bbfed0d9cdbf7a99769c4a44077dc15590496a4e8410ba9eda99f3078  triton_mm_encoder_attention_e2_002.py
20b360ac936bf4d9d41afadac90c40578f0a758e628ec40af2d3c759eb22d3fb  rounds/decision_002.md
c16b1528b25ae1a3bbfc72b3e459462505d940677e62b30a0585e3b41b46e9e9  rounds/sketch_002.json
20b21646d9c3ba3abe086d8133799d23a39981dcb4e1cb547e1a3f65b0bf7ffc  rounds/report_000.md
13adafe951df94bb7bb74294e195cfffc6992057d36e958801b293ab292f449c  rounds/report_001.md
c3980a2c4ae79171c73c7cf85c29c62f6ac98b9e9fb643c249fc9a6e0a921c9f  baseline_adapter.py
86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2  ../../base.py
71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29  auto_bench.py
4171de8dcb1df6478942b0861756d660ef7955c5741e60072f03476a5fab2fc2  triton_mm_encoder_attention_e2_001.py
322d932902fb161d7f4529576db972f1a858b2a731ff31a4f8b9f2ca1bde3863  log/probes/binding_statement_report_002.json
9472cd8bf7fc17fda24155f56474c8dbf386b233f033e4ebac5c3f2e7dd58c4f  log/probes/p13_r002_sweep_result.json
f137d87f31ced2ca64c5485abb3772a8d6f28ff2cec5d559521e2389eac52721  log/probes/p14_r002_gates_result.json
1eb734f91a945fde7b031be50fc22de0e4693342496a22eb00e865b33a7c50e4  log/r002_forward_100iter.pt.trace.json
```

## vNext Fact Pack

```json
{
  "schema_version": 1,
  "candidate_sha256": "cc98318bbfed0d9cdbf7a99769c4a44077dc15590496a4e8410ba9eda99f3078",
  "correctness": {
    "status": "pass",
    "evidence": [
      "auto_bench.py PASS accuracy in all three authoritative pairs + profile run (4/4 invocations, seed42 canonical regime)",
      "log/probes/verifier_r002_result.json all_ok=true: seed42 max_abs 4.883e-04, fp16-extreme vs fp32 ground truth max_abs 3.052e-05 (r001-identical), B1S41/B2S96/B2S82 non-target shapes, run_out poisoned x2 bitwise both orderings with data_ptr preserved, forward bitwise-stable, stateless attr audit, 9 DANGER tokens zero, 4 dot sites (32,32) fp32, single num_warps=2 site, 6 widening casts",
      "r002 outputs BITWISE-equal to r001 on every verifier suite (warp-count output-invariance verified)"
    ]
  },
  "observables": [
    {
      "name": "wall_time_unrounded_paired_median_ms",
      "status": "observed",
      "value": "-59.8032% (reference 0.144984 ms vs candidate 0.231689 ms; bar +5.0% FAILED with negative sign; wall INSIDE the pre-declared honest band 0.222-0.241 ms)",
      "confidence": "high",
      "evidence": ["log/r002_pair_001_timing.txt", "log/r002_pair_002_timing.txt", "log/r002_pair_003_timing.txt"]
    },
    {
      "name": "kernel_config_capability_probe",
      "status": "observed",
      "value": "pre-adoption sweep evidence reviewed and honored: exactness-passing configs nw1_fp32widen 23.492 / nw2_fp32widen 15.317 / nw4_fp32widen 15.441 us probe-method; selection rule applied exactly (fastest, 0.5us tie band, fewer-new-caps, lower-nw) -> nw2_fp32widen SHIPPED; capability negatives banked: fp16-operand dots fail exactness on extreme suite (max_abs 1459, vendor-class tie-flip) at every warp count; nw4 no gain over nw2",
      "confidence": "high",
      "evidence": ["log/probes/p13_r002_sweep_result.json", "log/probes/verifier_r002_result.json"]
    },
    {
      "name": "device_us_per_call",
      "status": "observed",
      "value": "D_cand(nw2) = 19.5550 us/call AUTHORITATIVE (attributed GPU-span projection 100/100; whole-trace identical; nw1 r001 same-method 28.2030 -> cut -8.648 us / -30.66%; vendor Ixmma 17.4212 whole-trace) — 18-25 partial band: net_F2 = 19.555 - 16.455 = +3.10 us/call WORSE than base (sub-parity 0.94-0.96x composed class); parity band 16.5 NOT reached (+3.06 short); win band 9.2 NOT reached (+10.36 short)",
      "confidence": "high",
      "evidence": ["log/diagnostic_scope_census_round002.json", "log/r002_summary_reference.json"]
    },
    {
      "name": "aten_cpu_ops_per_call",
      "status": "observed",
      "value": "1.00/call (single aten::empty) — unchanged from r001; kernel-only scope verified",
      "confidence": "high",
      "evidence": ["log/diagnostic_scope_census_round002.json"]
    },
    {
      "name": "launch_and_submission_count_per_call",
      "status": "observed",
      "value": "1.00 cuLaunchKernel/call (cuda_driver API, 3.66 us/call host), zero memcpys, zero graph submissions, zero model-code syncs — host-INVISIBLE config change",
      "confidence": "high",
      "evidence": ["log/diagnostic_scope_census_round002.json"]
    },
    {
      "name": "run_out_bitwise_equals_forward",
      "status": "observed",
      "value": "bitwise over poisoned buffers x2 (both orderings) with data_ptr preserved; returns None; forward bitwise-stable across repeat identical-input calls on all five suites; outputs bitwise-equal to r001",
      "confidence": "high",
      "evidence": ["log/probes/verifier_r002_result.json"]
    },
    {
      "name": "capability_legality_binding_audit",
      "status": "observed",
      "value": "4 tl.dot sites all (32,32)@(32,32) fp32 with widened operands (6 casts); single num_warps=2 site (the probe-qualified shipped value); num_stages absent; 9 DANGER tokens zero; zero .contiguous/layout-copy calls; stateless — re-verified over final bytes",
      "confidence": "high",
      "evidence": ["log/probes/verifier_r002_result.json", "log/probes/binding_statement_report_002.json"]
    },
    {
      "name": "triton_launcher_tax_invariance",
      "status": "observed",
      "value": "+84.5712 us/call net (whole-trace basis; r001 +84.7651; delta -0.19 us) — INSIDE the declared +80 to +90 band: the configuration change touched only the device term",
      "confidence": "high",
      "evidence": ["log/diagnostic_scope_census_round002.json"]
    }
  ],
  "lowering": {
    "status": "observed",
    "expected_mechanism": "present — ONE Triton kernel _mm_encoder_attn_fwd (num_warps=2) lowered and device-executed (100/100 events attributed, 19.5550 us/call)",
    "evidence_contract": "triton_cuda-v1 (proven-envelope dots consumed exactly as declared; num_warps=2 qualified by the pre-adoption sweep; P1-P4 ladder untriggered)",
    "evidence": ["log/diagnostic_scope_census_round002.json"]
  },
  "evidence_gap_cause": "none"
}
```

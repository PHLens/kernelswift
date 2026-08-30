# Report 001

Result: no-improvement

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md` @`67b96739c35adabb713081a1f3a50649193b28eed420dc32dd512572fab26c78` (hash re-verified live; F1 deliverable-grade triton-attention-dispatch-collapse, expected_wall_improvement_pct 0.0 declared honestly)
- Candidate: `triton_mm_encoder_attention_e2_001.py`
- Accepted reference: `baseline_adapter.py` (last_accepted_kernel per r000; byte-equivalent pipeline to base.py)
- Accepted reference report: `rounds/report_000.md` @`20b21646d9c3ba3abe086d8133799d23a39981dcb4e1cb547e1a3f65b0bf7ffc` (hash re-verified live)
- Decision SHA256: `67b96739c35adabb713081a1f3a50649193b28eed420dc32dd512572fab26c78`
- Sketch SHA256: `a1c27dbae53b1c7a74681510a0d09ced6be58ed8501f86976ce55af1b4772363` (rounds/sketch_001.json, re-verified)
- Candidate SHA256: `4171de8dcb1df6478942b0861756d660ef7955c5741e60072f03476a5fab2fc2` (re-verified before AND after all runs; AST parse OK; verifier's own DANGER-token re-scan all-zero)
- Accepted reference SHA256: `c3980a2c4ae79171c73c7cf85c29c62f6ac98b9e9fb643c249fc9a6e0a921c9f`
- Base SHA256: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` (unchanged, re-verified after all runs)
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (unchanged, AST loader)
- Binding: `log/probes/binding_statement_report_001.json` @`623783fd96ecfa90e77e88b59985d433ee8a31097f802d896bb82a9947630b2f` (coder-shaped statement; verifier independently re-verified every checkable claim: 9-token DANGER scan all-zero over final bytes, 4 dot sites, single num_warps=1 site, 6 widening casts, zero `.contiguous`/`contiguous` occurrences, stateless attr audit, run_out 4-arg + returns None)
- Runtime fingerprint: `project.md#runtime-fingerprint` (unchanged since r000 live probe; environment re-bootstrapped every shell)
- Measurement fingerprint: `0c4c7d664c85e65d0580091ca5e3a77ff769a0d28f7e679f5bdf78fe5d0d966e` (recomputed live this round; base/harness bytes re-verified identical before and after timing)
- verification_tier: `authoritative`
- screening_pairs: `not-run: correct candidate proceeded directly to authoritative timing (sibling r001 precedent; this dispatch routes below-bar outcomes to no-improvement with full census, and a screen-out would skip the profiler — destroying the round's mandated T_launcher/D_cand dual-gate measurement duty)`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness (seed42 canonical) | `allclose(atol=1e-2, rtol=1e-2, equal_nan=True)` vs base.py, seed 42, fp16 out `[2,83,512]` | `PASS accuracy` in all three authoritative pairs + profile run (4/4 invocations); probe max_abs 4.883e-04 | pass | `log/r001_pair_00{1,2,3}_timing.txt`, `log/r001_profile_run.txt`, `log/probes/verifier_r001_result.json` |
| fp16-extreme suite | extreme-magnitude inputs stay correct | candidate matches an independent fp32 textbook reference to max 3.052e-05 on the verifier extreme suite (±4096/±2048/±256/subnormal/zero mix). The VENDOR base kernel itself diverges from fp32 ground truth by up to 1457 on the same inputs (score range ±2.4e7 exceeds its fp16 internal envelope; 33 exactly-tied one-hot softmax rows flip argmax under fp16 vs fp32 math) — recorded as a vendor precision-limit observation, NOT a candidate defect; the binding comparator regime (seed42 randn) PASSes | pass | `log/probes/verifier_r001_extreme_diag.log` |
| non-target shape | correctness off the target shape (stateless recompile routing) | B1S41 seed7 and B2S96 seed19 (zero-padding tail) both PASS vs base (max_abs 4.883e-04), run_out bitwise, shape/dtype ok | pass | `log/probes/verifier_r001_result.json` |
| run_out contract | `run_out(q,k,v,out) -> None`, caller buffer filled directly, bitwise==forward | poisoned caller buffers ×2 (−777.0 and +555.0, both call orderings) bitwise-equal to forward with data_ptr preserved; returns None; zero extra ops (single launch) | pass | `log/probes/verifier_r001_result.json` |
| forward bitwise-stability | deterministic kernel, no atomics | repeat identical-input calls bitwise-stable on every suite (fresh allocations) | pass | `log/probes/verifier_r001_result.json` |
| stateless module | zero call-time instance state, no caches/workspace | `__dict__` delta across forward+run_out calls = empty | pass | `log/probes/verifier_r001_result.json` |
| capability legality | every tl.dot at (32,32)@(32,32) fp32 (widened); num_warps=1; fp16 only at global boundary | 4 dot sites, 6 widening casts (q×2, k×2, v×2 load sites — every fp16 tile widened before first dot use), single launch site `num_warps=1`, `num_stages` absent | pass | `log/probes/verifier_r001_result.json`, source audit |
| no compile/graph machinery | zero `torch.compile`/`TORCHINDUCTOR`/`reduce-overhead`/graph/capture/`contiguous`/`num_stages` tokens | all 9 re-scanned tokens count 0 over final candidate bytes | pass | `log/probes/verifier_r001_result.json` |
| default-stream discipline | all invocations on the harness default route | every measurement ran the unchanged harness default path; zero stream manipulation | pass | command history |
| cold JIT outside medians | warmup 50 absorbs first-call compile | harness warmup 50 precedes every timed section; JIT compile absorbed (sibling precedent) | pass | harness behavior |

Conformance, correctness, and every declared guardrail passed.

## Screening Evidence

Not run — correct candidate proceeded directly to authoritative timing (three ordered pairs). Rationale recorded in Identity: this round's contractual products are the Triton deliverable plus the two named mechanism observables (`triton_launcher_tax_per_call`, `device_us_per_call`), which require the profiler census a screen-out would skip.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved ordered reference/candidate pairs — three identical harness invocations (pair i = invocation_i.v0_ms then invocation_i.v1_ms), byte-for-byte identical flags, interpreter, device, default-stream route, CoreX environment`
- reference_raw_samples_ms: `[0.145375, 0.145466, 0.145368]`
- candidate_raw_samples_ms: `[0.238943, 0.242051, 0.240953]`
- reference_median_ms: `0.145375`
- candidate_median_ms: `0.240953`
- improvement_pct: `-65.74582975064487`

```text
improvement_pct = (0.145375 - 0.240953) / 0.145375 * 100 = -65.745830
```

| Independent invocation | Reference wall ms | Candidate wall ms | Speedup | Evidence |
|---:|---:|---:|---:|---|
| 1 | `0.145375` | `0.238943` | `0.608x` | `log/r001_pair_001_timing.txt` |
| 2 | `0.145466` | `0.242051` | `0.601x` | `log/r001_pair_002_timing.txt` |
| 3 | `0.145368` | `0.240953` | `0.603x` | `log/r001_pair_003_timing.txt` |

BELOW the 5.0% adoption bar with a decisively NEGATIVE sign: the candidate wall 0.240953 ms lands squarely inside the decision's pre-declared launcher-tax-transfers band (0.235–0.29 ms). Extra (non-authoritative) profile-run timing: v0=0.144873, v1=0.265107 ms.

### Anchor bases (all four, explicitly)

1. **Prescribed paired v0-basis headline**: v0=base.py 0.145375 ms vs v1=candidate 0.240953 ms, same session → **−65.7458%** (adoption-decisive basis).
2. **Direct same-session pair vs r000 v0**: last_accepted_kernel IS the base adapter (`baseline_adapter.py`, byte-equivalent pipeline to base.py), so the prescribed paired-v0 basis and the direct pair vs r000 v0 are THE SAME comparison this round — stated explicitly.
3. **Cross-anchor `report_000` 0.150149 ms**: candidate 0.240953 → −60.4759%. Session-drift context: this session's v0 median sits −3.1795% below r000's 0.150149 (0.145375/0.150149), so cross-session anchors are context only; the same-session pair is authoritative.
4. **Manifest anchor**: identical to the report_000 anchor this early in the campaign (no accepted round has been committed; r000 completion is the only prior) — stated explicitly: manifest anchor wall = 0.150149 ms, cumulative context −60.4759%.

ABAB interleaved control: not run — the measured delta (−65.75%) is ~13× the 5.0% bar and ~20× the observed session drift (−3.18% on the v0 side, absorbed by same-session pairing); no plausible drift magnitude affects the classification (dispatch made ABAB optional).

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| wall_time_unrounded_paired_median_ms | ≥5% below accepted reference median across interleaved pairs at warmup 50 / repeat 100; two-sided declaration — win branch requires the launcher tax NOT to transfer | −65.7458% (candidate 0.240953 vs reference 0.145375 ms, three ordered pairs); wall lands inside the pre-declared 0.235–0.29 ms launcher-tax-transfers band | **fail** (vs the ≥5% adoption criterion; the two-sided transfers branch is CONFIRMED — the honest expectation 0.0 realized) | `log/r001_pair_00{1,2,3}_timing.txt` |
| aten_cpu_ops_per_call | collapse from 33/call (report_000 census) to ≤3/call in the candidate forward scope | **33 → 1.00/call** (single `aten::empty`; the Triton launch path contributes zero aten ops) — collapse fully engaged | **pass** | `log/diagnostic_scope_census_round001.json` |
| launch_and_submission_count_per_call | exactly 1.00 kernel launch (cuLaunchKernel-class) per call, ZERO cudaMemcpyApi, ZERO graph launches, ZERO model-code synchronizations | **1.00 `cuLaunchKernel`/call** (cuda_driver API, 3.64 µs/call host); 0 memcpys, 0 graph submissions, 0 syncs in the candidate scope — the direct-family structural guarantee holds exactly | **pass** | `log/diagnostic_scope_census_round001.json` |
| device_us_per_call | TWO-SIDED with pre-declared readings: (a) D_cand ≤ ~40 µs AND wall ≥ +5% ⇒ win branch; (b) D_cand in 20–66 µs band with wall decisively negative and ~85–90 µs net host delta ⇒ honest no-improvement #1 with the bsz=2 launcher tax canonically measured; (c) D_cand ≥ ~66 µs ⇒ compute-bound regression | **D_cand = 28.2030 µs/call** (`_mm_encoder_attn_fwd`, 100/100 events attributed in the GPU-span projection; whole-trace cross-check identical) — **band (b) exactly**: 28.20 ∈ [20,66], wall decisively negative (−65.75%), net host delta +84.77 µs ≈ the 85–90 µs band | **pass** (attributed, band (b)) | `log/diagnostic_scope_census_round001.json` |
| triton_launcher_tax_per_call | canonical bsz=2 measurement: candidate host path vs base host path net delta per call (sibling prior +86–89 µs at bsz=1); derived from paired wall minus device delta, corroborated by the aten census; the F2 arithmetic gate | **T_launcher = +84.7651 µs/call net** (Δwall 95.578 − Δdevice 10.813); attributed-basis corroboration +83.0603 µs/call; sibling bsz=1 prior +86–89 µs — **the launcher tax TRANSFERS to bsz=2 essentially in full** | **pass** (measured; falsification target of the transfer model CONFIRMED) | `log/diagnostic_scope_census_round001.json` |
| run_out_bitwise_equals_forward | bitwise equality over poisoned caller buffers ×2 with data_ptr preserved; forward outputs bitwise-stable across repeated identical-input calls | true on all four suites (seed42 canonical, fp16-extreme, B1S41, B2S96): poisoned ×2 both orderings bitwise==forward, data_ptr preserved, forward bitwise-stable | **pass** | `log/probes/verifier_r001_result.json` |
| proven_envelope_binding_audit | every tl.dot call site (32,32)@(32,32) fp32 (widened); num_warps=1; count of torch.compile/TORCHINDUCTOR/reduce-overhead strings = 0; zero `.contiguous()` in forward/run_out host paths | 4/4 dot sites at (32,32) fp32 with widened operands; single `num_warps=1` site; 9 DANGER tokens all zero (incl. `contiguous` and `num_stages`); zero layout-copy calls — independently re-verified over final bytes | **pass** | `log/probes/verifier_r001_result.json`, `log/probes/binding_statement_report_001.json` |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-001`
- intervention: one direct-launched Triton full-attention kernel (48 programs, BM=BN=32, D=64 as two 32-chunks, proven (32,32) fp32 dot envelope, online fp32 softmax, direct strided addressing, direct-layout stores) replacing the whole 33-aten-op base path with a two-op forward (torch.empty + one launch) plus a 4-arg run_out surface
- expected_causal_chain: chain observed with full attribution — cn.dispatch-collapse → cn.aten-dispatch-time CONFIRMED (33 → 1 aten ops/call); cn.dispatch-collapse → cn.triton-launcher-tax CONFIRMED (+84.77 µs/call net, the sibling prior transfers to bsz=2); cn.device-time-delta measured (+10.81 µs/call: 28.20 Triton vs 17.39 vendor, in the pre-declared 20–66 band); cn.dispatch-collapse → cn.wall-time via the launcher-tax edge dominates ⇒ wall −65.75%
- primary_metric: `wall_time`
- Hypothesis verdict: `partially-confirmed` — every mechanism link is measured and census-attributed exactly as designed (dispatch collapse, single submission, envelope-legal kernel, deliverable surfaces), and the round's declared falsification target (launcher-tax transfer) is CONFIRMED on the transfers branch; the primary wall criterion (≥5%) FAILED exactly as the honestly-declared expectation (0.0) predicted. Adoption is governed by wall_time alone ⇒ `no-improvement`. This is precisely the decision's pre-declared reading (b).

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted` (per decision profiling_level; dual-scope forward trace + host census + launcher-tax decomposition)
- profiler_device_time: `available: BI150 trace contains cat=kernel device-duration events; candidate kernel attributed 100/100 under the GPU-span projection`
- mode deviation (D1, standing): canonical settings declare `profile_mode=kernel`; kernel mode on THIS candidate fails inside harness `make_profile_call`: it invokes `run_out(gating_output, *output_args)` — only the last input plus outputs — producing `TypeError: ModelNew.run_out() missing 2 required positional arguments: 'value' and 'out'`, exit 1 (`log/r001_kernel_mode_attempt.txt`, attempted this round). The candidate's 4-arg `run_out` is decision-mandated and correct; no accommodation invented. Fallback used: `--profile-mode forward` dual-scope via `--profile-reference-file`, `--profile-warmup 20 --profile-iterations 100` kept at regime values.
- summarizer deviation (D2): `summarize_trace.py` succeeds on the reference scope (`log/r001_summary_reference.json`) but errors `overlapping scope events` on the candidate scope because kineto emitted BOTH a host `user_annotation` span AND a `gpu_user_annotation` projection with the same record_function name (`log/r001_summary_candidate.json` left as the 0-byte failure marker, sibling r002 convention). Census-substitution: `log/probes/verifier_r001_scope_census.py` scopes host ops on the host span, device kernels on the GPU-span projection, and cross-checks against whole-trace kernel totals (the trace contains exactly 100 vendor + 100 candidate kernel events = the 100 recorded iterations per scope; profile warmup ran outside the profiler context). Whole-trace and GPU-projection attributions agree exactly.
- iterations: `100` forward calls per scope
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- trace: `log/r001_forward_100iter.pt.trace.json`
- trace_sha256: `3ce1e0d857a35d4c75f23a0d362cced11bbc5b502f886c5e5ec2fa7c22174803`
- scope summaries: `log/r001_summary_reference.json` (canonical summarizer) + `log/diagnostic_scope_census_round001.json` (census substitution for the candidate scope; separate scopes, never combined)

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference (baseline_adapter, canonical summarizer) | 1568.53125 | 15.6853 | 90 | 0.90 | 0.145375 | 0.10790 |
| candidate (triton_mm_encoder_attention_e2_001, census GPU-projection) | 2820.30 | 28.2030 | 100 | 1.00 | 0.240953 | 0.11704 |

```text
device_ratio = device_us_per_call / (scope_median_wall_ms * 1000)
```

Attribution notes: (i) the reference scope's 0.90/call attributed count is the recorder's span-edge margin (90 of 100 launched vendor kernels fully inside the span), same class as r000; the whole-trace cross-check gives 1739.01 µs / 100 = 17.3901 µs/call for the vendor kernel — the paired-decomposition below uses the whole-trace basis for both sides and reports the attributed-basis value as corroboration. (ii) The candidate host span additionally shows 0.10/call stray `FlashAttnFwdF16Ixmma` events — span-edge leakage of the PRECEDING reference scope's trailing kernels on the GPU timeline (the candidate never issues this kernel; 90+10 = the reference's 100). (iii) The candidate kernel is attributed 100/100 under the GPU-span projection with duration identical to the whole-trace total.

### Accepted Reference Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `void ixattnbkd::src::impl::MR::FlashAttnFwdF16Ixmma<128u, 128u, 16u, 64u, 64u, (CausalM_t)0, (AlibiMode_t)0, false, __half, false>(FlashAttnFwdParams, ...)` | 90 | 0.90 | 1568.53125 | 15.6853 (whole-trace: 17.3901) |

### Candidate Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `_mm_encoder_attn_fwd` | 100 | 1.00 | 2820.30 | 28.2030 |

Exactly ONE Triton kernel per call, exactly ONE launch per call — the direct-family structure is exactly as designed.

### Host census (dispatch-compression signature, per call)

| Signal | accepted_reference | candidate (direct Triton) |
|---|---:|---:|
| aten cpu_ops total/call | **33.00** (8 transpose, 8 as_strided, 7 empty, 4 view, 1 reshape, 1 empty_like, 1 empty_strided, sdpa chain ×3) | **1.00** (single `aten::empty`) |
| kernel launches | 1.00 `cudaLaunchKernel` (cuda_runtime, 5.20 µs/call) | 1.00 `cuLaunchKernel` (cuda_driver, 3.64 µs/call) |
| memcpys (any class) | 0 | **0** |
| graph submissions | 0 | **0** |
| model-code syncs | 0 | **0** |
| other per-call runtime | 1.00 `cudaStreamIsCapturing` | (none) |

### Launcher-tax and device decomposition (the dual-gate measurement duty)

| Quantity | Value (µs/call) | Basis |
|---|---:|---|
| Δwall (paired, same session) | 95.578 | 0.240953 − 0.145375 ms |
| Δdevice | +10.813 | D_cand 28.2030 − D_ref 17.3901 (whole-trace) |
| **T_launcher (net)** | **+84.7651** | Δwall − Δdevice |
| T_launcher (attributed-basis corroboration) | +83.0603 | Δwall − (28.2030 − 15.6853) |
| Sibling bsz=1 prior | +86–89 | flexattention r002 (cross-operator) |

**T_launcher = +84.77 µs/call at bsz=2: the Triton python launcher tax TRANSFERS essentially in full from the sibling's bsz=1 measurement** — the transfer model's falsification target is CONFIRMED (this is the canonical bsz=2 number this campaign set out to bank).

**F2 gate arithmetic (decision pre-authorized; side-of-threshold statements):**

| Gate | Threshold | Measured | Side |
|---|---|---:|---|
| composition prize exists | T_launcher ≥ ~50 | 84.7651 | **OPEN-side** (≥) — prize is real |
| device penalty fits | D_cand ≤ ~35 | 28.2030 | **OPEN-side** (≤) |
| parity-class | D_cand ≤ ~25 (decision) / ≈ 18 (dispatch band) | 28.2030 | **ABOVE** (+3.2 vs 25; +10.2 vs 18) — parity NOT reached |
| win-class | D_cand ≤ ~10 | 28.2030 | **ABOVE** (+18.2) — win NOT reachable |

F2 OPENS (both opening conditions hold), but at the measured numbers the graph-family projection is: net = −T_launcher + 69 (R-term) + ~13 (boundary host) + (D_cand + 3.7 copy-out − D_ref) = **+11.75 µs/call WORSE than base** — parity-class is not reached with D_cand = 28.2. Evidence only; the F2 proceed/kill decision belongs to the Designer/Orchestrator.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable (first verification of this candidate) | `4171de8dcb1df6478942b0861756d660ef7955c5741e60072f03476a5fab2fc2` | correctness passed; two PROBE-SIDE expected-value defects in the verifier's own probe were found and fixed probe-side (widen-cast threshold 8→6; extreme-suite comparison basis base→fp32 ground truth after the vendor-saturation diagnosis), re-run clean PASS; candidate bytes never changed |
| — | census script | `4171de8d…fc2` | `4171de8d…fc2` | one census-script defect (span list passed where aggregate expected) fixed script-side; no measurement affected |

At most one Verifier-to-Coder repair is allowed in the same round; zero were needed — no candidate defect was found at any point (candidate hash constant end-to-end, matching the coder ledger).

## evidence_for_next_round

- Observed fact: the dispatch collapse ENGAGES EXACTLY — 33 aten ops → 1 (single `torch.empty`), exactly 1.00 `cuLaunchKernel`/call via the cuda_driver API, zero memcpys, zero graph machinery, zero syncs — yet paired wall REGRESSED −65.7458% (0.240953 vs 0.145375 ms). The wall is decided by the Triton python launcher path, not by aten op count.
- Observed fact (canonical, this campaign): **T_launcher = +84.77 µs/call net at bsz=2** (wall−device; attributed-basis 83.06 corroborates) — the sibling's +86–89 µs at bsz=1 transfers essentially in full. The ~133 µs/call host floor of the base is NOT compressible by removing aten ops: the python-side Triton launch path costs more than the entire 33-op aten stack it replaces.
- Observed fact (canonical, this campaign): **D_cand = 28.20 µs/call** for the proven-envelope Triton kernel at 16 full-attention pairs (vs vendor Ixmma 17.39 whole-trace / 15.69 attributed this session; 13.6–17.6 band across campaigns). The 32×32-tile single-warp kernel pays ~1.6× the vendor kernel for ~2× the sibling's per-call work — a reasonable but not competitive device floor inside the frozen (32,32) fp32 envelope with num_warps=1.
- F2 gate status: OPEN (T_launcher 84.77 ≥ 50 AND D_cand 28.20 ≤ 35) but parity-class NOT reached (28.20 > 25 decision bar / ≈18 dispatch band); measured-number projection for the graph family = **+11.75 µs/call worse than base**. A graph round would need D_cand ≈ 18 µs (parity) — i.e., a ~36% device-time cut from the current kernel (F3 probe territory: fp16 tl.dot, num_warps>1) before composition arithmetic can close.
- Deliverable banked: `triton_mm_encoder_attention_e2_001.py` @`4171de8d…` is a correctness-PASS Triton submission (forward + 4-arg run_out surfaces, bitwise-equal, stateless, envelope-legal) at 0.603x — per project.md DELIVERABLE RULE this is the campaign's primary contractual product regardless of adoption; canonical pointer stays `baseline_adapter.py`.
- Vendor precision-limit observation (labeled, non-blocking): on verifier-manufactured extreme inputs (|score| up to 2.4e7), the vendor FlashAttnFwdF16Ixmma kernel diverges from fp32 ground truth by up to 1457 while the candidate matches it to 3.05e-05. Outside the harness comparator regime (seed42 randn, PASS); recorded for numerics provenance, not as a performance claim.
- Confirmed mechanism (harness/build facts carried forward): D1 kernel-mode arity (4-arg run_out vs harness 2-arg call); D2 kineto dual-span shape on direct-Triton candidates (gpu_user_annotation projection overlaps the host span — census-substitution pattern now proven in two campaigns); reference-scope attribution margin (~0.90/call) is stable across rounds on this build.
- Session drift note: v0 medians −3.18% vs r000 (0.145375 vs 0.150149) — paired same-session basis absorbs it; cross-session anchors are context only.

Evidence only; selection of the next optimization belongs to the Designer.

## Stop Recommendation

- recommendation: `continue`
- evidence: no-improvement #1 on the campaign (streak 1/3 vs valid_no_improvement_limit 3); round budget 1/20 consumed; counters alive; the round banked the Triton deliverable plus both canonical physics numbers (T_launcher +84.77 µs/call, D_cand 28.20 µs/call) with census-grade attribution; F2 gate open-but-not-parity-class and F3 probes remain the live levers.

Orchestrator owns the stop transition.

## Exact Reproduction Commands

Correctness + first authoritative pair (pairs 2 and 3 identical; default-stream route):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/bi150/epoch2/triton_mm_encoder_attention_e2_001.py --warmup 50 --repeat 100 --full-traceback
```

Kernel-mode attempt (records the D1 arity deviation verbatim, exit 1):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/bi150/epoch2/triton_mm_encoder_attention_e2_001.py --warmup 5 --repeat 10 --profile --profile-mode kernel --profile-warmup 20 --profile-iterations 100 --profile-output kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/r001_kernel_mode_attempt.pt.trace.json --full-traceback
```

Dual-scope profiler (forward-mode fallback, pw=20/pi=100 per regime) + per-scope normalization + census substitution:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/bi150/epoch2/triton_mm_encoder_attention_e2_001.py --warmup 50 --repeat 100 --profile --profile-reference-file kernels/track1-triton/mm_encoder_attention/bi150/epoch2/baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 100 --profile-output kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/r001_forward_100iter.pt.trace.json
/usr/local/bin/python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/r001_forward_100iter.pt.trace.json --iterations 100 --scope reference_baseline_adapter --wall-ms 0.145375
/usr/local/bin/python3 kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/probes/verifier_r001_scope_census.py
```

Verifier correctness probe:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/local/bin/python3 kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/probes/verifier_r001_correctness.py
```

Artifact hash ledger (re-verified this round, before and after all measurement):

```text
4171de8dcb1df6478942b0861756d660ef7955c5741e60072f03476a5fab2fc2  triton_mm_encoder_attention_e2_001.py
67b96739c35adabb713081a1f3a50649193b28eed420dc32dd512572fab26c78  rounds/decision_001.md
a1c27dbae53b1c7a74681510a0d09ced6be58ed8501f86976ce55af1b4772363  rounds/sketch_001.json
20b21646d9c3ba3abe086d8133799d23a39981dcb4e1cb547e1a3f65b0bf7ffc  rounds/report_000.md
c3980a2c4ae79171c73c7cf85c29c62f6ac98b9e9fb643c249fc9a6e0a921c9f  baseline_adapter.py
86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2  ../../base.py
71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29  auto_bench.py
623783fd96ecfa90e77e88b59985d433ee8a31097f802d896bb82a9947630b2f  log/probes/binding_statement_report_001.json
3ce1e0d857a35d4c75f23a0d362cced11bbc5b502f886c5e5ec2fa7c22174803  log/r001_forward_100iter.pt.trace.json
```

## vNext Fact Pack

```json
{
  "schema_version": 1,
  "candidate_sha256": "4171de8dcb1df6478942b0861756d660ef7955c5741e60072f03476a5fab2fc2",
  "correctness": {
    "status": "pass",
    "evidence": [
      "auto_bench.py PASS accuracy in all three authoritative pairs + profile run (4/4 invocations, seed42 canonical regime)",
      "log/probes/verifier_r001_result.json all_ok=true: seed42 max_abs 4.883e-04, fp16-extreme candidate-vs-fp32-ground-truth max_abs 3.052e-05 (vendor base itself diverges 1457 at |score|<=2.4e7), B1S41/B2S96 non-target shapes, run_out poisoned x2 bitwise both orderings with data_ptr preserved, forward bitwise-stable, stateless attr audit, 9 DANGER tokens zero, 4 dot sites (32,32) fp32, single num_warps=1 site"
    ]
  },
  "observables": [
    {
      "name": "wall_time_unrounded_paired_median_ms",
      "status": "observed",
      "value": "-65.7458% (reference 0.145375 ms vs candidate 0.240953 ms; bar +5.0% FAILED with negative sign; wall inside the pre-declared 0.235-0.29 launcher-tax-transfers band)",
      "confidence": "high",
      "evidence": ["log/r001_pair_001_timing.txt", "log/r001_pair_002_timing.txt", "log/r001_pair_003_timing.txt"]
    },
    {
      "name": "aten_cpu_ops_per_call",
      "status": "observed",
      "value": "33 -> 1.00/call (single aten::empty; dispatch collapse fully engaged)",
      "confidence": "high",
      "evidence": ["log/diagnostic_scope_census_round001.json"]
    },
    {
      "name": "launch_and_submission_count_per_call",
      "status": "observed",
      "value": "1.00 cuLaunchKernel/call (cuda_driver API, 3.64 us/call host), zero memcpys, zero graph submissions, zero model-code syncs",
      "confidence": "high",
      "evidence": ["log/diagnostic_scope_census_round001.json"]
    },
    {
      "name": "device_us_per_call",
      "status": "observed",
      "value": "D_cand = 28.2030 us/call (_mm_encoder_attn_fwd, 100/100 attributed; vendor 17.3901 whole-trace / 15.6853 attributed) — pre-declared band (b) exactly: 20-66 band, wall decisively negative, net host delta in the 85-90 us class",
      "confidence": "high",
      "evidence": ["log/diagnostic_scope_census_round001.json", "log/r001_summary_reference.json"]
    },
    {
      "name": "triton_launcher_tax_per_call",
      "status": "observed",
      "value": "+84.7651 us/call net at bsz=2 (wall 95.578 - device 10.813; attributed-basis +83.0603 corroborates; sibling bsz=1 prior +86-89 transfers in full) — F2 gate: OPEN on both conditions (>=50 and <=35), parity band 18 NOT reached (+10.2), win band 10 NOT reached (+18.2)",
      "confidence": "high",
      "evidence": ["log/diagnostic_scope_census_round001.json"]
    },
    {
      "name": "run_out_bitwise_equals_forward",
      "status": "observed",
      "value": "bitwise over poisoned buffers x2 (both call orderings) with data_ptr preserved; returns None; forward bitwise-stable across repeat identical-input calls on all four suites",
      "confidence": "high",
      "evidence": ["log/probes/verifier_r001_result.json"]
    },
    {
      "name": "proven_envelope_binding_audit",
      "status": "observed",
      "value": "4 tl.dot sites all (32,32)@(32,32) fp32 with widened operands (6 widening casts at the 6 load sites); num_warps=1 at the single shared launch site; 9 DANGER tokens zero; zero .contiguous/layout-copy calls — independently re-verified over final bytes",
      "confidence": "high",
      "evidence": ["log/probes/verifier_r001_result.json", "log/probes/binding_statement_report_001.json"]
    }
  ],
  "lowering": {
    "status": "observed",
    "expected_mechanism": "present — ONE Triton kernel _mm_encoder_attn_fwd lowered and device-executed (100/100 events attributed, 28.2030 us/call)",
    "evidence_contract": "triton_cuda-v1 (proven-envelope dots consumed exactly as declared; P1-P4 ladder untriggered)",
    "evidence": ["log/diagnostic_scope_census_round001.json"]
  },
  "evidence_gap_cause": "none"
}
```

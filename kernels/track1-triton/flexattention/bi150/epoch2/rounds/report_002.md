# Report 002

Result: no-improvement

## Identity

- Round: `002`
- Decision: `rounds/decision_002.md` @`459e8d37219b5534103a82a7a342c61ef04e147158a6851d794b73e2a44f8730` (hash re-verified live)
- Candidate: `triton_flexattention_e2_002.py`
- Accepted reference: `baseline_adapter.py` (last_accepted_kernel, unchanged)
- Accepted reference report: `rounds/report_000.md` @`a90df70d54e791ecf53b38913ea1165e2a47a6dd6201d68653e6a101c5882e7c`
- Decision SHA256: `459e8d37219b5534103a82a7a342c61ef04e147158a6851d794b73e2a44f8730`
- Sketch SHA256: `fb5bec0b957a04ffa19d20edb2f0fdb92de156c0aea6429b1c796a86b89bd87c` (rounds/sketch_002.json, re-verified)
- Candidate SHA256: `570bc2be2cb8e79a06ebb32e5e8bf4f79aa62a38d5382b9a1a5f12426f3512b1` (6445 bytes, re-verified; AST gate OK)
- Accepted reference SHA256: `b8ec3458bf810e6f3e81f759a716810b2046e5b32e22fc1339ec526e319445d1`
- Base SHA256: `dd1359ad88a5d5aae48a115023e5ae9016e4d5b5e6a12781273703e8d5c6a6d0` (unchanged)
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (unchanged, AST loader)
- Binding: `log/probes/binding_statement_report_002.json` — LIVE sha256 `331859628ad6e891a835be7fa71baa99b69999f744c947293d8463e8b02c3278`. LEDGER DEFECT NOTED: coder_result_002's hash ledger lists the binding as `ad4d4ba7…`, which is actually `p12_r002_sweep_result.json`'s hash (same value assigned to two different files — copy-paste defect). The live binding content was independently verified by Verifier: 14 DANGER tokens all-zero (incl. r001 machinery tokens), 4 tl.dot sites all (32,32)@(32,32) fp32 with fp16→fp32 widening before first dot use, num_warps=1 at exactly one site, num_stages absent, stateless (4 config attr writes in __init__ only). All claims consistent with Verifier's own source audit.
- Prior round artifacts: `rounds/report_001.md` @`8c93d473f6f3babcfd34c1cbe7bde76fbf1b1db1bbc002c61cbc04d76ab79336`, `rounds/verdict_001.json` @`c804df77d0d9ad6cff85c4cfd0b587da76b7ceb06a4843f352eee59ac9e6e362` (re-verified)
- Runtime fingerprint: `project.md#runtime-fingerprint` (environment re-bootstrapped every shell; unchanged)
- Measurement fingerprint: `6dc07009177b649f7c2cad8f7be5e9aad74235bd9f50abfebc88bdb273e32af4` (base/harness bytes re-verified identical)
- verification_tier: `authoritative`
- screening_pairs: `not-run: correct candidate proceeds directly to authoritative timing`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness (seed42) | allclose(atol=1e-2, rtol=1e-2, equal_nan) vs base.py | `PASS accuracy` in all three authoritative pairs + ABAB pairs + profile run (7/7 invocations); max_abs = 9.766e-04 (matches coder p11) | pass | `log/r002_pair_00{1,2,3}_timing.txt`, `log/verifier_correctness_result_002.json` |
| fp16-extreme/tie-free boundary case | extreme-magnitude rows stay within tolerance | manufactured suite (±32/±24 rows, fp16-subnormal-scale entries, zeros, continuous perturbations = tie-free): allclose PASS, **max_abs = 7.8125e-03**, mean_abs 1.946e-04 — under the 1e-2 tolerance; corroborates coder's fp16-ULP characterization (their extreme suite's 2.0 abs diff = 1 fp16 ULP at magnitude 2048–4096, pure output quantization in the rtol regime, comparator-legal) | pass | `log/verifier_correctness_result_002.json` |
| off-regime spot check | shape-specialized recompile correctness | T=41 suite: allclose PASS, max_abs 9.766e-04 | pass | same JSON |
| run_out bitwise == forward | poisoned caller buffers ×2, data_ptr preserved | both orderings bitwise-equal to forward; returns None; data_ptr preserved; repeat identical-input call bitwise-stable (deterministic kernel) on every suite | pass | same JSON |
| stateless module | no instance attrs written at call time; no r001 machinery | post-call `__dict__` diff EMPTY across forward+run_out; zero r001-style attrs (cuda_graph/q_in/k_in/v_in/attn_flat_ws/replay_failed absent); non-underscore attrs = exactly the 4 config attrs | pass | same JSON checks block |
| proven-envelope binding audit | every tl.dot (32,32)@(32,32) fp32; num_warps=1; DANGER zero | Verifier independent re-audit: 4 dot sites (q_lo·k_lo_t, q_hi·k_hi_t, p·v_lo, p·v_hi) all fp32 widened operands at (32,32); num_warps single site value 1; 14 DANGER tokens zero; tl.argmax/tl.trans/atomic/synchronize/.item() zero | pass | binding JSON `33185962…3278` + this audit |
| default stream | caller-stream launches, no stream tricks | kernel launched on caller's current stream; all invocations on harness default route | pass | source read + command history |

Conformance, correctness, and every declared guardrail passed.

## Screening Evidence

Not run — correct candidate proceeded directly to authoritative timing.

## Interleaved Wall Timing

- warmup: `50` (absorbs one-time JIT compile)
- repeat: `100`
- order: `interleaved ordered reference/candidate pairs — three identical harness invocations (pair i = invocation_i.v0_ms then invocation_i.v1_ms), byte-for-byte identical flags, default stream, plus a drift-separated ABAB supplement`
- reference_raw_samples_ms: `[0.200049, 0.147338, 0.143139]`
- candidate_raw_samples_ms: `[0.238962, 0.236245, 0.234321]`
- reference_median_ms: `0.147338`
- candidate_median_ms: `0.236245`
- improvement_pct: `-60.3422`

```text
improvement_pct = (0.147338 - 0.236245) / 0.147338 * 100 = -60.3422
```

DECISIVELY BELOW the 5.0% bar with a large negative sign (candidate ≈ 1.60x SLOWER than base).

### ABAB drift-separated supplement (ordered A-B-B-A executed as A1, B1, A2, B2)

| Position | Invocation | v0 ms | v1 ms | slot delta µs |
|---|---|---:|---:|---:|
| A1 (identity control: base vs accepted adapter) | `log/r002_abab_A1_control.txt` | 0.149233 | 0.148968 | −0.265 |
| B1 (base vs candidate) | `log/r002_abab_B1_candidate.txt` | 0.144823 | 0.237223 | +92.400 |
| A2 (identity control repeat) | `log/r002_abab_A2_control.txt` | 0.150437 | 0.150180 | −0.257 |
| B2 (base vs candidate repeat) | `log/r002_abab_B2_candidate.txt` | 0.144675 | 0.236848 | +92.173 |

Identity-control deltas ≈ **−0.26 µs** (noise floor ~0); candidate deltas **+92.2/+92.4 µs** in both B positions. Session drift is therefore ruled out as the cause: the ~+92 µs/call regression is REAL and stable. (Pair-1 v0=0.200049 was a drift spike; the paired same-invocation basis and the ABAB control both absorb it.)

### Anchor bases (all four, explicitly)

1. **Prescribed paired v0-basis headline**: v0=base.py 0.147338 vs v1=candidate 0.236245 ms, same session → **−60.3422%** (adoption-decisive).
2. **Direct same-session pair vs r000 v0**: a literal adapter-as-v0 pair is STRUCTURALLY BLOCKED by the harness — v0 must define `Model`, and the accepted adapter defines `ModelNew` (`KsCompareError: … baseline_adapter.py must define 'Model'.`, exit 1, `log/r002_abab_B1_direct.txt` — named deviation). Since the adapter is byte-equivalent in pipeline to base.py, the v0=base.py paired basis IS the accepted-reference basis this round (stated explicitly): −60.3414%.
3. **Cross-anchor `report_000` 0.151107 ms**: candidate 0.236245 → **+56.34% slower** (0.236245/0.151107 = 1.5634). This session's v0 median sits −2.49% BELOW r000's (0.147338/0.151107) — session drift cuts both ways; same-session pairing remains authoritative.
4. **Manifest anchor**: identical to the report_000 anchor this early (no accepted round committed beyond Phase 0) — stated explicitly: manifest anchor 0.151107 ms, cumulative context +56.34% slower.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| wall_time_unrounded_paired_median_ms | ≥5% below same-session accepted reference median across interleaved pairs | **−60.3422%** (0.236245 vs 0.147338 ms); ABAB drift-corrected delta ≈ +92.5 µs/call | **fail** | pairs + ABAB logs |
| aten_cpu_ops_per_call | collapse from ~34/call to ≤3/call | **38 → 1.00/call** in the candidate scope census (single `aten::empty`; dispatch-collapse mechanism FULLY ENGAGED on the op-count edge — even deeper than designed) | **pass** | `log/diagnostic_scope_census_round002.json` |
| launch_and_submission_count_per_call | exactly 1.00 kernel launch/call, ZERO memcpys, ZERO graph submissions, ZERO model-code syncs | **1.00 cuLaunchKernel/call** (driver-API cat — see trace-shape note), ZERO memcpys, ZERO graph submissions, ZERO model-code syncs; r001's regression structure (5 submissions + per-call sync) absent as designed | **pass** | same census |
| device_us_per_call | TWO-SIDED: (a) ≤~40 µs with wall ≥+5%; (b) ≥~60 µs with wall flat/negative; (c) between → wall decides, census attributes | T_triton(**_causal_attn_fwd**) = **16.510 µs/call** (66 attributed events) vs base Ixmma 13.614 µs/call → Δdevice ≈ **+2.9 µs/call** — device lands in band (a)'s range (fine), but wall is decisively negative ⇒ reading (c) applies: wall decides, census attributes — the failure is HOST-side launcher overhead, NOT device regression (falsifies the decision's device-fear branch) | **pass** (observed, reading (c)) | census + summaries |
| run_out_bitwise_equals_forward | bitwise over poisoned buffers ×2 with data_ptr preserved; repeat stability | all true on every suite incl. repeat-call bitwise stability | **pass** | `log/verifier_correctness_result_002.json` |
| proven_envelope_binding_audit | every tl.dot (32,32)@(32,32) fp32 widened; num_warps=1; DANGER zero | independently re-verified (see Correctness table) | **pass** | binding JSON + audit |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-002`
- intervention: ONE direct-launched Triton blocked-attention kernel + two-op forward replacing the ~34-38-aten-op SDPA path; stateless; no graphs/compile
- expected_causal_chain: edges PARTIALLY observed — cn.dispatch-collapse → cn.aten-dispatch-time CONFIRMED (38→1 ops/call); cn.dispatch-collapse → cn.device-time-delta CONFIRMED and SMALL (+2.9 µs; proven-envelope dots kept the kernel in the good band); cn.dispatch-collapse → cn.wall-time FALSIFIED by an UNMODELED host term: the Triton python launcher path costs ~+86–89 µs/call more than the entire base host path it replaced (quantified below), overwhelming the ~20–32 µs python prize
- primary_metric: `wall_time`
- Hypothesis verdict: `partially-confirmed` — the dispatch-collapse mechanism and the device-band fear are both MEASURED (collapse real and deep; device fine at 16.5 µs), but the primary wall outcome is decisively falsified by the launcher-overhead term. Adoption is governed by wall_time alone ⇒ `no-improvement`.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted` (dual-scope forward traces + host census)
- profiler_device_time: `available: BI150 trace contains cat=kernel device-duration events scoped under per-target record_function spans`
- mode deviation: kernel-mode profiling remains structurally blocked by the harness arity fact (r001 D2 stands, unchanged this round — `make_profile_call` passes `run_out(<inputs[-1]>, *outputs)` only); forward-mode dual-scope used with regime pw=20/pi=100, `--profile-reference-file baseline_adapter.py`
- CANONICAL-TOOL TRACE-SHAPE LIMITATION (named): `summarize_trace.py` errors `overlapping scope events: candidate_triton_flexattention_e2_002` on this trace because kineto emitted BOTH a host `user_annotation` span AND a `gpu_user_annotation` projection with the same name for the candidate scope (device-correlated annotation; the reference scope got only the host span). The reference scope summarized canonically; the candidate scope was summarized by the Verifier census scoping on the host `user_annotation` span with identical attribution rules — recorded as a tool limitation, not evidence loss.
- iterations: `100` forward calls per scope
- trace: `log/r002_forward_100iter.pt.trace.json` @`d1e91bbd642584730d98b21d19c9b32e16b4cb0aa632062fcd3b01deb9a95686`
- scope summaries: `log/r002_summary_reference.json` (canonical tool), candidate scope via `log/diagnostic_scope_census_round002.json` (Verifier census; canonical tool blocked by the duplicate-span artifact)

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference (baseline_adapter) | 1361.3935546875 | 13.613935546875 | 99 | 0.99 | 0.147338 | 0.09239935079120798 |
| candidate (triton_flexattention_e2_002) | 1651.124 | 16.511 | 67 | 0.67 | 0.236245 | 0.069868 |

```text
device_ratio = device_us_per_call / (scope_median_wall_ms * 1000)
```

Attribution note: candidate kernel events = `_causal_attn_fwd` (66 events, 16.5098 µs/call; 1650.979 µs total) + 1 span-edge stray Ixmma event (0.145 µs — margin artifact). The 0.67/call attribution rate mirrors the r000/r001 margin behavior; per-launch device time is computed over the declared iteration count per contract.

### Accepted Reference Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `ixattnbkd::…::FlashAttnFwdF16Ixmma<128u, 128u, 16u, 64u, 64u, (CausalM_t)2, (AlibiMode_t)0, false, __half, false>` | 99 | 0.99 | 1361.3935546875 | 13.613935546875 |

### Candidate Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `_causal_attn_fwd` (candidate's own Triton kernel) | 66 | 0.66 | 1650.979 | 16.5098 |

### Host census and the D1 launcher-overhead adjudication

Per-call structure (host `user_annotation` scope, pi=100):

| Signal | accepted_reference | candidate (Triton direct launch) |
|---|---:|---:|
| aten cpu_ops | 38 (12 as_strided, 8 transpose, 7 empty, 3 unsqueeze, sdpa stack, empty_like/empty_strided/squeeze) | **1** (aten::empty only) |
| kernel submission | 1.00 cudaLaunchKernel (runtime API) | **1.00 cuLaunchKernel** (driver API; 3.53 µs/call host duration) |
| memcpys / graph submissions / model syncs | 0 / 0 / 0 | 0 / 0 / 0 |

D1 ADJUDICATION (coder's suspicion → measured verdict, **CONFIRMED**):

- Wall delta vs accepted pipeline: **+88.9 µs/call** (paired medians; +92.2/+92.4 µs in both ABAB B positions against a −0.26 µs control floor).
- Device delta: **+2.9 µs/call** (16.510 Triton vs 13.614 Ixmma) — the kernel is NOT the problem (decision's feared ≥60 µs band did not materialize).
- Residual host delta ≈ **+86.0 µs/call** (ABAB: ≈+89.3 µs/call) — the Triton python launcher path (JIT dispatch, arg binding, grid computation, cache lookup) costs ~86–89 µs MORE per call than the ENTIRE base host path it replaced (~134 µs wall-less-device for ~38 ops + sdpa C++ dispatch), of which only **3.53 µs** is the actual driver submission. Pure python-side launcher overhead ≈ **82–86 µs/call**.
- Conclusion: on this build the single-Triton-launch host path (~220 µs/call host) is ~1.6x the whole vendor-stack host path (~134 µs/call). The dispatch prize (~20–32 µs, decision's r001-arithmetic estimate) is real but dwarfed: the launcher overhead SWAMPS it. No legal candidate-side mechanism removes python launcher overhead while keeping the direct-launch design (decision forbids caching/fast-launch workarounds by construction).
- Caveat: kineto cpu_op DURATIONS on this trace are nesting-inflated (sdpa stack shows 166 µs/call > whole-call wall) and are NOT used as absolute host cost — counts and wall/device deltas carry the quantification.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable (first verification of this candidate) | `570bc2be2cb8e79a06ebb32e5e8bf4f79aa62a38d5382b9a1a5f12426f3512b1` | candidate bytes never changed; correctness passed first time; no repairs |

At most one Verifier-to-Coder repair is allowed in the same round; zero were needed.

## evidence_for_next_round

- Observed fact: the dispatch-collapse mechanism WORKS as designed — aten ops 38→1/call, single driver-API launch (3.53 µs), zero copies/graphs/syncs, stateless — and the Triton kernel itself is healthy at 16.51 µs/call device (+2.9 µs vs Ixmma; proven-envelope dots kept it far from the feared 60 µs band).
- Falsified mechanism (with root cause sized): on this CoreX 4.4.0/torch 2.7.1/BI-V150 build, the Triton python launcher path costs ~86–89 µs/call MORE than the entire ~38-op vendor host path — wall −60.34% — so ANY direct-python-launch Triton candidate at this call granularity loses to the vendor stack regardless of device time, unless the per-call python launcher cost is eliminated (graphs fail per r001 economics; caching/compile machinery is out of contract scope).
- Observed fact: both campaign host levers are now measured dead-ends at this operator's granularity — (r001) graph replay adds boundary economics on a 1-launch base; (r002) direct Triton launch adds ~86 µs python overhead. Device-side: vendor Ixmma 13.6 µs floor vs best legal Triton attempt 16.5 µs (+21%) — no device win available within the proven envelope either.
- Campaign-level implication (evidence only): the remaining measurable levers under the current contract set are exhausted at this operator size; both no-improvement root causes are now census-grade documented.
- Labeled noncanon priors: epoch-1 naive 0.61x (device-dominated scalar era) and groupedtopk-r2 graph success (6.9-launch base) bracket WHY this operator differs: it has ONE launch and ~134 µs of cheap-C++ host path.

Evidence only; selection of the next optimization belongs to the Designer.

## Stop Recommendation

- recommendation: `continue` (Verifier evidence role only — but recorded honestly: this is the campaign's 2nd no-improvement; per team-state rules a 3rd terminates. The decision-space documentation above is the input to that judgment; Orchestrator owns the transition.)
- evidence: no-improvement #2 (streak 2/3); round budget 2/20; both host families measured to root cause; device family measured within proven envelope.

Orchestrator owns the stop transition.

## Exact Reproduction Commands

Authoritative pairs (1–3 identical; default stream):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/bi150/epoch2/triton_flexattention_e2_002.py --warmup 50 --repeat 100 --full-traceback
```

ABAB supplement (A = identity control, B = candidate pair):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/bi150/epoch2/baseline_adapter.py --warmup 50 --repeat 100 --full-traceback
/usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/bi150/epoch2/triton_flexattention_e2_002.py --warmup 50 --repeat 100 --full-traceback
```

Adapter-as-v0 direct pair (records the v0-contract format fact, exit 1):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/bi150/epoch2/baseline_adapter.py --v1_file kernels/track1-triton/flexattention/bi150/epoch2/triton_flexattention_e2_002.py --warmup 50 --repeat 100 --full-traceback
```

Dual-scope profiler (forward-mode, pw=20/pi=100) + per-scope normalization + census:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/bi150/epoch2/triton_flexattention_e2_002.py --warmup 50 --repeat 100 --profile --profile-reference-file kernels/track1-triton/flexattention/bi150/epoch2/baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 100 --profile-output kernels/track1-triton/flexattention/bi150/epoch2/log/r002_forward_100iter.pt.trace.json
/usr/local/bin/python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/flexattention/bi150/epoch2/log/r002_forward_100iter.pt.trace.json --iterations 100 --scope reference_baseline_adapter --wall-ms 0.147338
/usr/local/bin/python3 kernels/track1-triton/flexattention/bi150/epoch2/log/probes/verifier_scope_census_002.py
```

Verifier correctness probe:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/local/bin/python3 kernels/track1-triton/flexattention/bi150/epoch2/log/probes/verifier_correctness_002.py
```

Artifact hash ledger (re-verified this round):

```text
570bc2be2cb8e79a06ebb32e5e8bf4f79aa62a38d5382b9a1a5f12426f3512b1  triton_flexattention_e2_002.py
459e8d37219b5534103a82a7a342c61ef04e147158a6851d794b73e2a44f8730  rounds/decision_002.md
fb5bec0b957a04ffa19d20edb2f0fdb92de156c0aea6429b1c796a86b89bd87c  rounds/sketch_002.json
a90df70d54e791ecf53b38913ea1165e2a47a6dd6201d68653e6a101c5882e7c  rounds/report_000.md
8c93d473f6f3babcfd34c1cbe7bde76fbf1b1db1bbc002c61cbc04d76ab79336  rounds/report_001.md
c804df77d0d9ad6cff85c4cfd0b587da76b7ceb06a4843f352eee59ac9e6e362  rounds/verdict_001.json
b8ec3458bf810e6f3e81f759a716810b2046e5b32e22fc1339ec526e319445d1  baseline_adapter.py
dd1359ad88a5d5aae48a115023e5ae9016e4d5b5e6a12781273703e8d5c6a6d0  ../../base.py
71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29  auto_bench.py
331859628ad6e891a835be7fa71baa99b69999f744c947293d8463e8b02c3278  log/probes/binding_statement_report_002.json (LIVE; coder ledger's ad4d4ba7… is p12_r002_sweep_result.json's hash — bookkeeping defect)
d1e91bbd642584730d98b21d19c9b32e16b4cb0aa632062fcd3b01deb9a95686  log/r002_forward_100iter.pt.trace.json
```

## vNext Fact Pack

```json
{
  "schema_version": 1,
  "candidate_sha256": "570bc2be2cb8e79a06ebb32e5e8bf4f79aa62a38d5382b9a1a5f12426f3512b1",
  "correctness": {
    "status": "pass",
    "evidence": [
      "auto_bench.py PASS accuracy in all three authoritative pairs + ABAB pairs + profile run (7/7)",
      "log/verifier_correctness_result_002.json all_checks_pass=true: seed42 max_abs 9.766e-04, fp16-extreme tie-free suite max_abs 7.8125e-03 (within 1e-2), T=41 spot 9.766e-04, run_out poisoned x2 bitwise==forward, repeat bitwise stability, stateless attr audit"
    ]
  },
  "observables": [
    {
      "name": "wall_time_unrounded_paired_median_ms",
      "status": "observed",
      "value": "-60.3422% (reference 0.147338 ms vs candidate 0.236245 ms; bar +5.0% FAILED; ABAB drift-corrected +92.5 us/call)",
      "confidence": "high",
      "evidence": ["log/r002_pair_001_timing.txt", "log/r002_pair_002_timing.txt", "log/r002_pair_003_timing.txt", "log/r002_abab_A1_control.txt", "log/r002_abab_B1_candidate.txt", "log/r002_abab_A2_control.txt", "log/r002_abab_B2_candidate.txt"]
    },
    {
      "name": "aten_cpu_ops_per_call",
      "status": "observed",
      "value": "38 -> 1.00/call (single aten::empty; dispatch collapse fully engaged)",
      "confidence": "high",
      "evidence": ["log/diagnostic_scope_census_round002.json"]
    },
    {
      "name": "launch_and_submission_count_per_call",
      "status": "observed",
      "value": "1.00 cuLaunchKernel/call (driver API, 3.53 us host), zero memcpys, zero graph submissions, zero model-code syncs",
      "confidence": "high",
      "evidence": ["log/diagnostic_scope_census_round002.json"]
    },
    {
      "name": "device_us_per_call",
      "status": "observed",
      "value": "T_triton 16.510 us/call vs base Ixmma 13.614 (+2.9 us) — band (a) device range; reading (c) applied: wall decides, census attributes failure to host launcher not device",
      "confidence": "high",
      "evidence": ["log/diagnostic_scope_census_round002.json", "log/r002_summary_reference.json"]
    },
    {
      "name": "run_out_bitwise_equals_forward",
      "status": "observed",
      "value": "bitwise over poisoned buffers x2 with data_ptr preserved; repeat identical-input calls bitwise-stable on every suite",
      "confidence": "high",
      "evidence": ["log/verifier_correctness_result_002.json"]
    },
    {
      "name": "proven_envelope_binding_audit",
      "status": "observed",
      "value": "4 tl.dot sites all (32,32)@(32,32) fp32 widened; num_warps=1 single site; 14 DANGER tokens zero — independently re-verified against live binding 33185962…3278 (coder ledger hash defect noted)",
      "confidence": "high",
      "evidence": ["log/probes/binding_statement_report_002.json"]
    }
  ],
  "lowering": {
    "status": "observed",
    "expected_mechanism": "present — ONE Triton kernel _causal_attn_fwd lowered and device-executed (66 attributed events, 16.51 us/call)",
    "evidence_contract": "triton_cuda-v1 (proven-envelope dots consumed exactly as declared; P1-P4 ladder untriggered)",
    "evidence": ["log/diagnostic_scope_census_round002.json"]
  },
  "evidence_gap_cause": "none"
}
```

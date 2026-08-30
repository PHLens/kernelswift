# Report 001

Result: no-improvement

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md` @`fa11b1152306e4cc4b33a02e31bc52d4c76de210c79385f41e02ee25c3bc7b1d` (hash re-verified live)
- Candidate: `triton_flexattention_e2_001.py`
- Accepted reference: `baseline_adapter.py` (last_accepted_kernel per r000; byte-equivalent pipeline to base.py)
- Accepted reference report: `rounds/report_000.md` @`a90df70d54e791ecf53b38913ea1165e2a47a6dd6201d68653e6a101c5882e7c` (hash re-verified live)
- Decision SHA256: `fa11b1152306e4cc4b33a02e31bc52d4c76de210c79385f41e02ee25c3bc7b1d`
- Sketch SHA256: `199275b85e831238c2f0c9c694d3c4c03550c6681bd7a8e87f3474642b3c1fce` (rounds/sketch_001.json, re-verified)
- Candidate SHA256: `b490acc674ef5570900e8273bd6e3ab2a10102612b8c6fc6da63271a2dfcadec` (9744 bytes, re-verified; AST gate OK; verifier's own DANGER token re-scan all-zero)
- Accepted reference SHA256: `b8ec3458bf810e6f3e81f759a716810b2046e5b32e22fc1339ec526e319445d1`
- Base SHA256: `dd1359ad88a5d5aae48a115023e5ae9016e4d5b5e6a12781273703e8d5c6a6d0` (unchanged)
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (unchanged, AST loader)
- Binding: `log/probes/binding_statement_report_001.json` @`916058cb682f65a65908fbe5bc3c0c8e4a397067eec7b4dfc7d6737b7cb8dc5b` (D3: coder-shaped statement is the norm this round — no non-Triton symbols in the profile matrix; verifier independently re-verified every checkable claim: all nine DANGER tokens = 0 over final bytes, segment-freeze claims consistent with source read-through, hashes match)
- Runtime fingerprint: `project.md#runtime-fingerprint` (unchanged since r000 live probe; environment re-bootstrapped every shell)
- Measurement fingerprint: `6dc07009177b649f7c2cad8f7be5e9aad74235bd9f50abfebc88bdb273e32af4` (unchanged; base/harness bytes re-verified identical before timing)
- verification_tier: `authoritative`
- screening_pairs: `not-run: correct candidate proceeds directly to authoritative timing`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness (seed42, replayed route) | `allclose(atol=1e-2, rtol=1e-2, equal_nan)` vs base.py through the ACTIVE tier | `PASS accuracy` in all three authoritative pairs + profile run (4/4 invocations); ACTIVE TIER behaviorally determined = **manual-replay** (verifier probe: real graph handle + workspace present + `replay_failed=False` on first call; 150/150 distinct-input calls bitwise==eager reference at harness-like scale with flags unchanged) | pass | `log/r001_pair_00{1,2,3}_timing.txt`, `log/verifier_tier_result_001.json` |
| ACTIVE-tier determination | replay tier must actually serve timed calls, not silent eager | probe: `active_tier_is_replay=true` (graph handle present, workspace present, flag false); trace corroboration: candidate scope shows 1.00 `cudaGraphLaunch`/call + 3.99 DtoD memcpys/call and aten census 34→6/call — eager route would show ~1 kernel + ~34 aten ops/call | pass | probe + `log/diagnostic_scope_census_round001.json` |
| bitwise tier-retention (replay vs eager twin) | replayed-tier outputs BITWISE-EQUAL to eager tier for identical bits, both surfaces | seed42 forward bitwise==eager-staged twin AND bitwise==baseline_adapter reference; fp16-extreme suite bitwise on forward+run_out; run_out poisoned ×2 bitwise with data_ptr preserved and never aliased to workspace | pass | `log/verifier_tier_result_001.json` (all 14 boolean checks true) |
| fp16-extreme / tie-free boundary case | extreme-magnitude rows stay correct and bitwise-retained | manufactured suite (±32 rows, ±24 rows, fp16-subnormal-scale entries, zeros, continuous perturbations = tie-free): replay forward bitwise==eager twin AND ==reference adapter; `allclose vs base` pass; run_out bitwise==forward | pass | `log/verifier_tier_result_001.json` |
| at-scale retention sweep | harness-like load through replay route stays bitwise-correct | 150 calls, distinct seeds 1000–1149: **150/150 bitwise==reference adapter**, flags never moved, graph handle retained | pass | probe JSON `at_scale_150_replay_bitwise_vs_ref` |
| fallback_tier_selectivity_and_recovery | non-target first call (T=41 fp16) = eager with ZERO artifacts; following target call captures and serves replay again | confirmed: `zero_artifacts=true` (no graph handle, no workspace attrs), outputs base-consistent bitwise==reference; recovery: captured ONCE on target call, replay route bitwise-correct after; T=41 calls still correct afterward; flags moved only downward-by-exception (never observed moving) | pass | probe JSON selectivity/recovery blocks |
| run_out contract | `run_out(q,k,v,out) -> None`, caller buffer filled via out-of-boundary copy-out, never aliased | returns None; poisoned ×2 both bitwise-equal to forward; data_ptr preserved; ≠ workspace pointer; eager-tier surface parity | pass | probe JSON run_out_poisoned_x2 |
| default-stream discipline (D1) | all invocations on harness default route; no stream tricks | every measurement ran the unchanged harness default path; no stream manipulation in any verifier invocation | pass | command history |
| no compile machinery / no tl.dot | zero `torch.compile`/`reduce-overhead`/`TORCHINDUCTOR`/tf32 strings; zero Triton | verifier's independent re-scan over final bytes: all nine tokens count 0; AST parse OK | pass | this report Identity block |

Conformance, correctness, and every declared guardrail passed.

## Screening Evidence

Not run — correct candidate proceeded directly to authoritative timing (three ordered pairs).

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved ordered reference/candidate pairs — three identical harness invocations (pair i = invocation_i.v0_ms then invocation_i.v1_ms), byte-for-byte identical flags, interpreter, device, default-stream route, CoreX environment`
- reference_raw_samples_ms: `[0.155043, 0.153720, 0.156726]`
- candidate_raw_samples_ms: `[0.161593, 0.154599, 0.157659]`
- reference_median_ms: `0.155043`
- candidate_median_ms: `0.157659`
- improvement_pct: `-1.687325`

```text
improvement_pct = (0.155043 - 0.157659) / 0.155043 * 100 = -1.687325
```

BELOW the 5.0% adoption bar with a NEGATIVE sign (candidate slower than its own same-session reference). Adoption requires ≥+5.0%; this is a measured regression of −1.69% on the paired basis.

### Anchor bases (all four, explicitly)

1. **Prescribed paired v0-basis headline**: v0=base.py 0.155043 ms vs v1=candidate 0.157659 ms, same session → **−1.6873%** (adoption-decisive basis).
2. **Direct same-session pair vs r000 v0**: last_accepted_kernel IS the base adapter (`baseline_adapter.py`, byte-equivalent pipeline to base.py), so the prescribed "paired v0 basis" and the "direct pair vs r000 v0" are THE SAME comparison this round — stated explicitly. Candidate 0.157659 vs its own session's reference 0.155043 = −1.6873%.
3. **Cross-anchor `report_000` 0.151107 ms**: candidate 0.157659 → −4.3355% (slower). Session-drift context: this session's v0 median sits +2.604% above r000's 0.151107 (0.155043/0.151107), so cross-session anchors are context only; the same-session pair is authoritative.
4. **Manifest anchor**: identical to the report_000 anchor this early in the campaign (no accepted round has been committed yet; r000 completion is the only prior) — stated explicitly: manifest anchor wall = 0.151107 ms, cumulative context −4.3355%.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| wall_time_unrounded_paired_median_ms | ≥5% below accepted reference median across interleaved pairs at warmup 50 / repeat 100 | −1.6873% (negative; candidate 0.157659 vs reference 0.155043 ms, three ordered pairs) | **fail** | `log/r001_pair_00{1,2,3}_timing.txt` |
| bitwise_tier_retention_equivalence | replayed outputs bitwise==eager for identical bits on seed42 AND ≥2 manufactured suites (causal-boundary rows, extreme-magnitude), through BOTH forward and run_out (incl. poisoned ×2); allclose PASS everywhere | verifier probe all 14 checks true: seed42 bitwise vs eager-staged twin AND vs reference adapter (forward), fp16-extreme suite bitwise (forward+run_out), run_out poisoned ×2 bitwise with data_ptr preserved, 150/150 at-scale distinct-input sweep, allclose vs base everywhere — consistent with coder sweep 12/12 | **pass** | `log/verifier_tier_result_001.json` |
| fallback_tier_selectivity_and_recovery | non-target first call executes eager with ZERO replay artifacts; following target call captures and serves replayed tier again; flags move downward only on failure | T=41-first: zero artifacts (no graph handle, no workspace attributes), flag intact, outputs bitwise==reference; recovery: captured once on target call, replay route bitwise-correct; T=41 still correct afterward | **pass** | `log/verifier_tier_result_001.json` |
| kernel_count_per_call | TWO-BRANCH: (A) attributed launches collapse far below 0.88/call toward ≤~0.1/call; OR (B) intra-replay launches explicitly unattributable AND host census shows ≤~6 memcpy-class calls/call with no other GPU submissions; FAIL requires attributed ≈0.88/call WITH flat wall | **BRANCH A taken, corroborated by branch-B-grade host census**: attributed cat=kernel = **0.14/call** (14/100 events; vs base 0.86/call — collapsed far below; the 14 stray attributed events are span-edge attribution-margin artifacts, not per-call launches). Positive single-submission evidence at measurement scale: **1.00 cudaGraphLaunch/call**, 4.00 cudaMemcpyAsync + 3.99 Memcpy DtoD per call, NO other GPU submissions. Failure clause moot: capture demonstrably PRESENT (graph handle + replay behavior + bitwise retention at scale) and wall is not flat-but-correct — it regressed with the mechanism fully engaged | **pass** | `log/r001_summary_candidate.json`, `log/diagnostic_scope_census_round001.json` |
| host_dispatch_compression_signature | candidate census shows python/aten op counts dropping toward designed minimum (≈3-4 copy-class ops), consistent with ONE replay submission; absence of BOTH compression signatures AND ≥5% wall gain fails the hypothesis | aten cpu_ops per call **34 → 6** (12 as_strided + 8 transpose + 7 empty + 3 unsqueeze + sdpa stack + empty_like → 4 copy_ + empty_like + empty_strided); GPU submissions restructured 1 cudaLaunchKernel → 1 cudaGraphLaunch + 4 cudaMemcpyAsync. Compression signatures PRESENT while wall <5% ⇒ decision's pre-declared reading (a) applies: host floor harder than sibling analog, honest no-improvement | **pass** | `log/diagnostic_scope_census_round001.json` |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-001`
- intervention: one-time manual CUDA-graph workspace capture of the byte-frozen base pipeline; per-call guard + 3 copy-ins + ONE replay submission + 1 copy-out
- expected_causal_chain: chain PARTIALLY observed — cn.workspace-manual-replay → cn.host-dispatch-time CONFIRMED (aten dispatch 34→6/call; single graph submission); cn.host-dispatch-time → cn.wall-time FALSIFIED (wall regressed −1.69%; per-submission costs and harness-fixed floor dominate the residual window; the replay route pays 5 GPU submissions/call where eager pays 1, plus per-call `cudaDeviceSynchronize`/`cudaDriverGetVersion` observed inside the candidate scope census that the eager route does not pay)
- primary_metric: `wall_time`
- Hypothesis verdict: `partially-confirmed` — mechanism engagement (capture + dispatch compression + retention) is CONFIRMED by measurement; the expected ≥5% wall outcome is FALSIFIED (measured −1.6873%). Adoption is governed by wall_time alone ⇒ `no-improvement`. This is exactly the decision's pre-declared two-sided reading: "wall <5% WITH compression signatures present ⇒ host floor harder than sibling analog ⇒ honest no-improvement #1".

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted` (per decision profiling_level; dual-scope forward traces + host census)
- profiler_device_time: `available: BI150 trace contains cat=kernel device-duration events scoped under per-target record_function spans; intra-graph replay kernels are attribution-coarsened per the decision's ATTRIBUTION SCOPING CONTRACT (documented, not treated as mechanism failure)`
- mode deviation: canonical settings declare `profile_mode=kernel`; kernel mode on THIS candidate fails inside harness `make_profile_call` (D2): it invokes `run_out(gating_output, *output_args)` — only the last input plus outputs — producing `TypeError: ModelNew.run_out() missing 2 required positional arguments: 'value' and 'out'`, exit 1 (`log/r001_kernel_mode_attempt.txt`, attempt recorded this round). The candidate's 4-arg `run_out` signature is decision-mandated and correct; no accommodation invented. Fallback used: `--profile-mode forward` dual-scope via `--profile-reference-file`, `--profile-warmup 20 --profile-iterations 100` kept at regime values — same deviation class as report_000.
- iterations: `100` forward calls per scope
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- trace: `log/r001_forward_100iter.pt.trace.json`
- trace_sha256: `b6732432d642a79ea1ca55f6d5dccea26150f41ae64230d9c05dac0e2bad3271`
- scope summaries: `log/r001_summary_reference.json`, `log/r001_summary_candidate.json` (separate scopes, never combined)
- host census: `log/diagnostic_scope_census_round001.json` (per-scope cpu_op / cuda_runtime / gpu_memcpy / kernel tallies)

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference (baseline_adapter) | 1325.4453125 | 13.254453125 | 86 | 0.86 | 0.155043 | 0.08548888453525796 |
| candidate (triton_flexattention_e2_001) | 213.0546875 | 2.130546875 | 14 | 0.14 | 0.157659 | 0.013513639405298778 |

```text
device_ratio = device_us_per_call / (scope_median_wall_ms * 1000)
```

Attribution note (per the decision's ATTRIBUTION SCOPING CONTRACT): the candidate's 0.14/call attributed count is the trace-visible residue of a fully-engaged replay route — intra-graph kernels are coarsened/unattributable on this build; the 14 attributed FlashAttnFwdF16Ixmma events across 100 calls are span-edge margin, not per-call launches. Device work per call is identical-by-construction to the reference pipeline (captured verbatim) plus 4 small DtoD boundary copies (3.99 Memcpy DtoD events/call, device-side cost ≈ 2.13 µs/call attributed band). Reference scope remains the honest device-floor measurement: 13.25 µs/call, 0.86/call.

### Accepted Reference Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `ixattnbkd::src::impl::MR::FlashAttnFwdF16Ixmma<128u, 128u, 16u, 64u, 64u, (CausalM_t)2, (AlibiMode_t)0, false, __half, false>` | 86 | 0.86 | 1325.4453125 | 13.254453125 |

### Candidate Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `ixattnbkd::src::impl::MR::FlashAttnFwdF16Ixmma<128u, 128u, 16u, 64u, 64u, (CausalM_t)2, (AlibiMode_t)0, false, __half, false>` | 14 | 0.14 | 213.0546875 | 2.130546875 |

### Host census (dispatch-compression signature, per call)

| Signal | accepted_reference | candidate (manual replay) |
|---|---:|---:|
| aten cpu_ops total/call | ≈34 (12 as_strided, 8 transpose, 7 empty, 3 unsqueeze, 1 empty_like, sdpa stack ×3) | **6** (4 copy_, 1 empty_like, 1 empty_strided) |
| kernel launches (cuda_runtime) | 1.00 cudaLaunchKernel | 0 cudaLaunchKernel |
| graph submissions | — | **1.00 cudaGraphLaunch** |
| memcpys (host API) | 0 | 4.00 cudaMemcpyAsync |
| DtoD device trips | 0 | 3.99 Memcpy DtoD |
| other per-call runtime | 1.00 cudaStreamIsCapturing | 1.00 cudaStreamIsCapturing, 1.00 cudaDriverGetVersion, 1.00 cudaDeviceSynchronize |

Designed trip structure (3 copy-ins + 1 replay + 1 copy-out) is EXACTLY what the trace shows. The per-call `cudaDeviceSynchronize` and `cudaDriverGetVersion` inside the candidate replay route are observed build costs the eager route does not pay — recorded as evidence for the wall regression, not as a prescribed fix.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable (first verification of this candidate) | `b490acc674ef5570900e8273bd6e3ab2a10102612b8c6fc6da63271a2dfcadec` | candidate bytes never changed; correctness passed first time; no repairs |

At most one Verifier-to-Coder repair is allowed in the same round; zero were needed (candidate hash unchanged end-to-end, matching the coder ledger).

## evidence_for_next_round

- Observed fact: the manual-replay mechanism ENGAGES and COMPRESSES on this build — 1.00 cudaGraphLaunch/call, aten dispatch 34→6/call, designed trip structure exact, bitwise retention 150/150 at scale — yet paired wall REGRESSED −1.6873% (0.157659 vs 0.155043 ms). The report_000 host window (~91%) is therefore NOT python-dispatch-count-bound: the residual floor is per-submission/sync-class cost that the replay route INCREASES (5 GPU submissions/call vs eager 1, plus per-call cudaDeviceSynchronize/cudaDriverGetVersion observed in the candidate route).
- Falsified mechanism (this round's family on THIS operator): boundary-copy+replay does NOT collapse the host floor for a single-kernel eager pipeline — the graph wrapper's boundary traffic and submission count exceed what it removes when the base path already issues only ONE kernel launch and ~34 cheap host-side dispatches.
- Confirmed mechanism: workspace capture/replay is functionally sound on CoreX 4.4.0/BI-V150 for default-stream use (library-op capturability, static addresses, permanent fallback edges, selectivity) — the family remains viable ONLY where the captured region amortizes MULTIPLE launches (noncanon sibling r004 precedent: 6.9 launches/call collapsed to 1 → +42.5%); here the launch count was already 1.
- Observed fact: base device floor confirmed again at 13.25 µs/call (0.86/call attributed), device_ratio 0.0855 — the ~85 µs of wall ABOVE the r000 numbers is session drift, not device change (same-session pairing absorbed it).
- Level-2 host decomposition observation: harness-fixed components (set_seed + sync per sample) remain inside every timed sample; the wall floor is now bounded below by per-call submission/sync costs rather than aten dispatch count.

Evidence only; selection of the next optimization belongs to the Designer.

## Stop Recommendation

- recommendation: `continue`
- evidence: no-improvement #1 on the campaign (streak 1/3 vs valid_no_improvement_limit 3); round budget 1/20 consumed; counters alive; evidence points to device-side families (sub-branches (ii)/(iii) backlog, preconditions P1–P4 unrun) or next-round Level-2 host decomposition as the remaining levers.

Orchestrator owns the stop transition.

## Exact Reproduction Commands

Correctness + first authoritative pair (pairs 2 and 3 identical; default-stream route):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/bi150/epoch2/triton_flexattention_e2_001.py --warmup 50 --repeat 100 --full-traceback
```

Kernel-mode attempt (records the D2 arity deviation verbatim, exit 1):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/bi150/epoch2/triton_flexattention_e2_001.py --warmup 5 --repeat 10 --profile --profile-mode kernel --profile-warmup 20 --profile-iterations 100 --profile-output kernels/track1-triton/flexattention/bi150/epoch2/log/r001_kernel_mode_attempt.pt.trace.json --full-traceback
```

Dual-scope profiler (forward-mode fallback, pw=20/pi=100 per regime) + per-scope normalization + host census:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/bi150/epoch2/triton_flexattention_e2_001.py --warmup 50 --repeat 100 --profile --profile-reference-file kernels/track1-triton/flexattention/bi150/epoch2/baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 100 --profile-output kernels/track1-triton/flexattention/bi150/epoch2/log/r001_forward_100iter.pt.trace.json
/usr/local/bin/python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/flexattention/bi150/epoch2/log/r001_forward_100iter.pt.trace.json --iterations 100 --scope reference_baseline_adapter --wall-ms 0.155043
/usr/local/bin/python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/flexattention/bi150/epoch2/log/r001_forward_100iter.pt.trace.json --iterations 100 --scope candidate_triton_flexattention_e2_001 --wall-ms 0.157659
/usr/local/bin/python3 kernels/track1-triton/flexattention/bi150/epoch2/log/probes/verifier_scope_census_001.py
```

Verifier correctness/active-tier probe:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/local/bin/python3 kernels/track1-triton/flexattention/bi150/epoch2/log/probes/verifier_tier_check_001.py
```

Artifact hash ledger (re-verified this round):

```text
b490acc674ef5570900e8273bd6e3ab2a10102612b8c6fc6da63271a2dfcadec  triton_flexattention_e2_001.py
fa11b1152306e4cc4b33a02e31bc52d4c76de210c79385f41e02ee25c3bc7b1d  rounds/decision_001.md
199275b85e831238c2f0c9c694d3c4c03550c6681bd7a8e87f3474642b3c1fce  rounds/sketch_001.json
a90df70d54e791ecf53b38913ea1165e2a47a6dd6201d68653e6a101c5882e7c  rounds/report_000.md
b8ec3458bf810e6f3e81f759a716810b2046e5b32e22fc1339ec526e319445d1  baseline_adapter.py
dd1359ad88a5d5aae48a115023e5ae9016e4d5b5e6a12781273703e8d5c6a6d0  ../../base.py
71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29  auto_bench.py
916058cb682f65a65908fbe5bc3c0c8e4a397067eec7b4dfc7d6737b7cb8dc5b  log/probes/binding_statement_report_001.json
b6732432d642a79ea1ca55f6d5dccea26150f41ae64230d9c05dac0e2bad3271  log/r001_forward_100iter.pt.trace.json
```

## vNext Fact Pack

```json
{
  "schema_version": 1,
  "candidate_sha256": "b490acc674ef5570900e8273bd6e3ab2a10102612b8c6fc6da63271a2dfcadec",
  "correctness": {
    "status": "pass",
    "evidence": [
      "auto_bench.py PASS accuracy in all three authoritative pairs + profile run (default-stream replayed route)",
      "log/verifier_tier_result_001.json all_checks_pass=true (14/14): seed42 allclose vs base + bitwise vs eager twins, fp16-extreme suite, run_out poisoned x2, 150/150 at-scale replay sweep, selectivity/recovery"
    ]
  },
  "observables": [
    {
      "name": "wall_time_unrounded_paired_median_ms",
      "status": "observed",
      "value": "-1.6873% (reference 0.155043 ms vs candidate 0.157659 ms; bar +5.0% FAILED, negative sign)",
      "confidence": "high",
      "evidence": ["log/r001_pair_001_timing.txt", "log/r001_pair_002_timing.txt", "log/r001_pair_003_timing.txt"]
    },
    {
      "name": "bitwise_tier_retention_equivalence",
      "status": "observed",
      "value": "bitwise-equal replay vs eager on seed42 + fp16-extreme suite through forward and run_out (poisoned x2, data_ptr preserved); 150/150 at-scale",
      "confidence": "high",
      "evidence": ["log/verifier_tier_result_001.json"]
    },
    {
      "name": "fallback_tier_selectivity_and_recovery",
      "status": "observed",
      "value": "T=41-first: zero artifacts, eager correct; recovery captured once and served replay bitwise-correct; flags never moved",
      "confidence": "high",
      "evidence": ["log/verifier_tier_result_001.json"]
    },
    {
      "name": "kernel_count_per_call",
      "status": "observed",
      "value": "BRANCH A: attributed 0.14/call vs base 0.86/call; host census 1.00 cudaGraphLaunch/call + 3.99 DtoD memcpys/call; failure clause moot (capture demonstrably present)",
      "confidence": "high",
      "evidence": ["log/r001_summary_candidate.json", "log/diagnostic_scope_census_round001.json"]
    },
    {
      "name": "host_dispatch_compression_signature",
      "status": "observed",
      "value": "aten cpu_ops 34 -> 6 per call; single graph submission + 4 cudaMemcpyAsync; signatures PRESENT while wall <5% (decision reading (a) applies)",
      "confidence": "high",
      "evidence": ["log/diagnostic_scope_census_round001.json"]
    }
  ],
  "lowering": {
    "status": "observed",
    "expected_mechanism": "absent (zero tl.dot / zero Triton kernels by design; host-only manual-graph family)",
    "evidence_contract": "triton_cuda-v1 (no Triton symbols consumed this round; P1-P4 ladder untriggered)",
    "evidence": ["log/diagnostic_scope_census_round001.json"]
  },
  "evidence_gap_cause": "none"
}
```

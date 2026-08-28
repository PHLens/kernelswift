# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `../base.py` (`kernels/track1-triton/flexattention/base.py`; Phase 0 accepted reference == base)
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `b8ec3458bf810e6f3e81f759a716810b2046e5b32e22fc1339ec526e319445d1` (2090 bytes)
- Accepted reference SHA256: `dd1359ad88a5d5aae48a115023e5ae9016e4d5b5e6a12781273703e8d5c6a6d0` (2479 bytes)
- Base SHA256: `dd1359ad88a5d5aae48a115023e5ae9016e4d5b5e6a12781273703e8d5c6a6d0` (2479 bytes, equals project.md declaration, unchanged)
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (29428 bytes, AST loader, unchanged)
- Runtime fingerprint: `project.md#runtime-fingerprint` (re-probed live: triton 3.1.0 `/usr/local/corex-4.4.0/lib64/python3/dist-packages/triton`, torch 2.7.1, CoreX 4.4.0 nvcc V10.2.89 bootstrap `COREX_VERSION=4.4.0`, Iluvatar BI-V150 capability major=7 minor=1 multi_processor_count=16 total_memory=17179869184, interpreter `/usr/local/bin/python3`, device `cuda:0` — match)
- Measurement fingerprint: `6dc07009177b649f7c2cad8f7be5e9aad74235bd9f50abfebc88bdb273e32af4` (recomputed live as `sha256(base_bytes ‖ NUL ‖ harness_bytes ‖ NUL ‖ canonical_json_settings)` with `sort_keys=True, separators=(',',':')`; formula cross-validated by reproducing the sibling campaign fingerprint `8deb1b01…431` from its own artifacts — match with project.md)
- verification_tier: `baseline`
- screening_pairs: `not-applicable: Phase 0 baseline`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | pass vs base.py semantics; fp16 out `[83,512]`, `allclose(atol=1e-2, rtol=1e-2)`; seed default 42 | base.py (v0) and baseline_adapter.py (v1) outputs compared equal under harness comparator in all timed runs and the profile run: `PASS accuracy` printed, exit 0 | pass | `log/pair_001_timing.txt`, `log/pair_002_timing.txt`, `log/pair_003_timing.txt`, `log/profile_forward_run.txt` |
| runtime bootstrap | CoreX before any torch/triton use; BI-V150 cuda:0 | `export COREX_VERSION=4.4.0; . /usr/local/corex/enable` in every shell; live probe matched project.md runtime fingerprint exactly | pass | round_status_000.md probe entry |
| harness immutability | auto_bench.py loaded through AST loader, bytes unchanged | sha256 re-verified during verification: `71fb3ad0…fe29` (29428 bytes) | pass | sha256 ledger below |
| immutable base | `../base.py` bytes unchanged after adapter generation/verification | sha256 `dd1359ad…a6d0` (2479 bytes) equals project.md declaration; re-verified post-run | pass | sha256 ledger below |
| measurement fingerprint equality | live recompute must equal project.md value | recomputed `6dc07009177b649f7c2cad8f7be5e9aad74235bd9f50abfebc88bdb273e32af4` == declared value | pass | recompute transcript in round_status_000.md |
| canonical profile_mode=kernel applicability | regime declares `profile_mode=kernel` | kernel mode requires a callable `ModelNew.run_out`; Phase-0 baseline adapter defines only `forward`. Empirical attempt failed inside harness `make_profile_call`: `KsCompareError: candidate_baseline_adapter: kernel profiling requires a callable ModelNew.run_out`, exit 1 | observed-limitation | kernel-mode attempt logged under Exact Reproduction Commands; fallback `--profile-mode forward` used, `--profile-warmup 20 --profile-iterations 100` kept at regime values |

Conformance, correctness, and every declared guardrail passed; the single observed limitation is profiler-mode coverage (see Profiler Evidence deviation).

## Screening Evidence

Not applicable to Phase 0 baseline; the candidate is the baseline adapter itself generated from immutable base by renaming `Model` → `ModelNew`. No screen decision exists for this round.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved ordered reference/candidate pairs — three identical harness invocations (pair i = invocation_i.v0_ms then invocation_i.v1_ms) with byte-for-byte identical flags, interpreter, device, CoreX environment`
- reference_raw_samples_ms: `[0.151079, 0.151107, 0.151336]`
- candidate_raw_samples_ms: `[0.151440, 0.150994, 0.150791]`
- reference_median_ms: `0.151107`
- candidate_median_ms: `0.150994`
- improvement_pct: `+0.074778`

```text
improvement_pct = (0.151107 - 0.150994) / 0.151107 * 100 = +0.074778
```

Identity-level (<1%) delta between base.py and its adapter-of-base is the expected ~1.00x and is recorded as evidence, not an optimization claim. Within each invocation the harness medians over 100 individually synchronized repeats (seed fixed at 42 before each sample).

## Evaluation Contract Mirror

- Evaluation Contract applicability: `not-applicable: Phase 0`
- hypothesis_id: `not-applicable: Phase 0`
- intervention: `not-applicable: Phase 0`
- expected_causal_chain: `not-applicable: Phase 0`
- primary_metric: `wall_time`
- Hypothesis verdict: `not-applicable: Phase 0`

No decision_000.md exists; there are no declared mechanism observables to mirror.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `summary`
- profiler_device_time: `available: BI150 trace contains cat=kernel device-duration events scoped under per-target record_function spans`
- mode deviation: canonical settings declare `profile_mode=kernel`, but kernel mode requires `ModelNew.run_out` which the Phase-0 baseline adapter does not implement (harness `make_profile_call` raises `KsCompareError: candidate_baseline_adapter: kernel profiling requires a callable ModelNew.run_out`, exit 1). Fallback used: `--profile-mode forward`, dual-scope trace via `--profile-reference-file`, keeping regime-declared `--profile-warmup 20 --profile-iterations 100`. Same structural deviation and wording class as groupedtopk-r2 Phase-0 (labeled noncanon prior).
- iterations: `100` forward calls per scope
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- trace: `log/flexattention_baseline_forward_100iter.pt.trace.json`
- trace_sha256: `1185fa8de04fb094d7a099f6bd002d843a90d7532a539a7f33501d35d66828a5`
- scope summaries: `log/summary_reference_baseline_adapter.json`, `log/summary_candidate_baseline_adapter.json` (separate scopes, never combined)

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference (reference_baseline_adapter = base.py) | 1356.029296875 | 13.56029296875 | 88 | 0.88 | 0.151107 | 0.08973967432845599 |
| candidate (candidate_baseline_adapter) | 1496.966796875 | 14.96966796875 | 98 | 0.98 | 0.150994 | 0.09914081333529809 |

```text
device_ratio = device_us_per_call / (scope_median_wall_ms * 1000)
host_share = 1 - device_ratio   # reference ≈ 0.91026, candidate ≈ 0.90086
```

Attribution note: both scopes are structurally a SINGLE fused attention kernel per call; fractional counts-per-call (0.88 / 0.98 instead of 1.00) mean the traced cat=kernel events captured 88 and 98 of 100 launched instances respectively — an event-attribution margin of the recorder around span boundaries, not additional or missing kernel diversity. Per-call device time above is therefore normalized over the declared iteration count per the contract.

Host-vs-device decomposition (authoritative medians):

| Scope | Median wall us | Device us/call | Estimated host+dispatch share |
|---|---:|---:|---:|
| accepted_reference | 151.107 | 13.560 | ≈ 91.0% |
| candidate | 150.994 | 14.970 | ≈ 90.1% |

### Accepted Reference Top Kernels (BASE Ixmma launch census)

Full censused path of `F.scaled_dot_product_attention(is_causal=True)` on BI-V150 — exactly one user-visible device kernel per call:

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `ixattnbkd::src::impl::MR::FlashAttnFwdF16Ixmma<128u, 128u, 16u, 64u, 64u, (CausalM_t)2, (AlibiMode_t)0, false, __half, false>` | 88 | 0.88 | 1356.029296875 | 13.56029296875 |

No other kernel names appear in the reference scope. (Flanking `unsqueeze/transpose/reshape` are views — zero-device-cost; no copy/cat/reduce kernels present.)

### Candidate Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `ixattnbkd::src::impl::MR::FlashAttnFwdF16Ixmma<128u, 128u, 16u, 64u, 64u, (CausalM_t)2, (AlibiMode_t)0, false, __half, false>` | 98 | 0.98 | 1496.966796875 | 14.96966796875 |

Identical single-kernel structure in both scopes confirms the adapter executes exactly the base pipeline (identity of code path; the 1.41 µs/call difference between scopes is noise-level on identical work).

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable (first run) | `b8ec3458bf810e6f3e81f759a716810b2046e5b32e22fc1339ec526e319445d1` | pass on first attempt; no repairs |

At most one Verifier-to-Coder repair is allowed in the same round; zero were needed.

## evidence_for_next_round

- Observed fact: BASE attention path issues exactly ONE fused device kernel per forward — `FlashAttnFwdF16Ixmma<(128,128,16),(64,64),Causal=2,f16>` — at 13.56–14.97 µs/call (two scopes), confirming the ~13–15 µs device claim.
- Observed fact: median wall is ~0.1510–0.1511 ms ⇒ device_ratio ≈ 0.09–0.10; ≈ 90% of every forward call (~136 µs) is host-side floor outside kernel execution (SDPA Python dispatch, view ops, launch latency, seed/sync per sample in canonical regime).
- Campaign-shaping implication (evidence only): any single-kernel Triton replacement inherits essentially the same host floor unless it reduces host-side dispatch cost itself; ceiling for pure device-time reduction is bounded by ~15 µs/call (~10% of wall) absent a host-path intervention (graphs/lowering of dispatch).
- Confirmed mechanism (Phase-0 gate): harness AST loading, fp16 allclose comparators, seeding, BI150 cat=kernel device timing, and dual-scope forward profiling all work end-to-end on cuda:0.
- Limitation carried forward: `profile_mode=kernel` requires a candidate-side `ModelNew.run_out(query,key,value,out)` preallocated-output surface (project.md public_contract already mandates it); until a candidate provides it, forward-mode scoping remains the profiler regime.
- Labeled noncanon priors (different fingerprints; historical evidence only): epoch-1 archive `../rounds/report_000.md` (naive causal deliverable, matrix 0.61x, campaign stopped/user-intervention); sibling `../../../../groupedtopk/bi150-round2/rounds/report_000.md` established this deviation wording and manual-CUDA-graph architecture family.

Evidence only; selection of the next optimization belongs to the Designer.

## Stop Recommendation

- recommendation: `continue`
- evidence: baseline established (correctness PASS across all invocations, wall medians 0.151107/0.150994 ms, single-kernel census 13.56/14.97 µs/call, fingerprints verified); no candidate round evaluated yet.

Orchestrator owns the stop transition.

## Exact Reproduction Commands

Correctness + first timing pair (pairs 2 and 3 use the identical command; interpreter `/usr/local/bin/python3`, device `cuda:0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/bi150/epoch2/baseline_adapter.py --warmup 50 --repeat 100 --full-traceback
```

Canonical kernel-mode attempt (records the run_out limitation verbatim, exit 1):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/bi150/epoch2/baseline_adapter.py --warmup 5 --repeat 10 --profile --profile-mode kernel --profile-warmup 20 --profile-iterations 100 --profile-output kernels/track1-triton/flexattention/bi150/epoch2/log/kernel_mode_attempt.pt.trace.json --full-traceback
```

Dual-scope profiler (forward-mode fallback, pw=20/pi=100 per regime) + per-scope normalization:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/bi150/epoch2/baseline_adapter.py --warmup 50 --repeat 100 --profile --profile-reference-file kernels/track1-triton/flexattention/bi150/epoch2/baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 100 --profile-output kernels/track1-triton/flexattention/bi150/epoch2/log/flexattention_baseline_forward_100iter.pt.trace.json
/usr/local/bin/python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/flexattention/bi150/epoch2/log/flexattention_baseline_forward_100iter.pt.trace.json --iterations 100 --scope reference_baseline_adapter --wall-ms 0.151107
/usr/local/bin/python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/flexattention/bi150/epoch2/log/flexattention_baseline_forward_100iter.pt.trace.json --iterations 100 --scope candidate_baseline_adapter --wall-ms 0.150994
```

Artifact hash ledger:

```text
sha256sum kernels/track1-triton/flexattention/base.py auto_bench.py kernels/track1-triton/flexattention/bi150/epoch2/baseline_adapter.py
dd1359ad88a5d5aae48a115023e5ae9016e4d5b5e6a12781273703e8d5c6a6d0  kernels/track1-triton/flexattention/base.py
71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29  auto_bench.py
b8ec3458bf810e6f3e81f759a716810b2046e5b32e22fc1339ec526e319445d1  kernels/track1-triton/flexattention/bi150/epoch2/baseline_adapter.py
sha256sum log/flexattention_baseline_forward_100iter.pt.trace.json
1185fa8de04fb094d7a099f6bd002d843a90d7532a539a7f33501d35d66828a5
```

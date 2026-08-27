# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `../base.py`
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `ecce4dacee211a86ba38584b6b78fc2f575ba60cedccdc6f79ac4f6fb0139fa5`
- Accepted reference SHA256: `12f3324896d6b72bd4eac556839db53fb7045b2965b7f8caf2734551daad0f58`
- Base SHA256: `12f3324896d6b72bd4eac556839db53fb7045b2965b7f8caf2734551daad0f58` (3541 bytes, unchanged)
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (29428 bytes, AST loader)
- Runtime fingerprint: `project.md#runtime-fingerprint` (re-probed live: triton 3.1.0 `/usr/local/corex-4.4.0/lib64/python3/dist-packages/triton`, torch 2.7.1, CoreX 4.4.0 nvcc V10.2.89, Iluvatar BI-V150 capability major=7 minor=1 multi_processor_count=16 total_memory=17179869184 — match)
- Measurement fingerprint: `8deb1b012de31b18887562e736c7b9e120b9d9f9500230e237ee003c5fa5a431` (recomputed as `sha256(base_bytes ‖ NUL ‖ harness_bytes ‖ NUL ‖ canonical_json_settings)` — match)
- verification_tier: `baseline`
- screening_pairs: `not-applicable: Phase 0 baseline`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | pass vs base.py semantics; fp32/fp16 outputs `allclose(atol=1e-2, rtol=1e-2)`, int32 ids exactly equal; seed default 42 | `base.py` and `baseline_adapter.py` produced identical output tuples under the harness comparator in every invocation (4/4 runs incl. profile run) | pass | `PASS accuracy; v0=…, v1=…` printed by `auto_bench.py`; exit 0 in all runs |
| runtime bootstrap | CoreX bootstrap before import/use; device `cuda:0` BI-V150 | `export COREX_VERSION=4.4.0; . /usr/local/corex/enable` succeeded; ixsmi shows Iluvatar BI-V150 @ 54:00.0; probes match fingerprint | pass | live ixsmi + torch/triton probe (round_status_000.md) |
| harness immutability | auto_bench.py loaded through AST loader, bytes unchanged | sha256 re-verified after all runs: `71fb3ad0…fe29`, 29428 bytes | pass | `sha256sum` ledger in round_status_000.md |
| immutable base | `../base.py` bytes unchanged after adapter generation/verification | sha256 `12f33248…d0f58` (3541 bytes) equals project.md declaration | pass | `sha256sum ../base.py` |
| canonical profile_mode=kernel applicability | regime declares `profile_mode=kernel` | kernel mode requires a callable `ModelNew.run_out`; Phase-0 baseline adapter defines only `forward`. Empirical attempt failed inside harness `make_profile_call`: `KsCompareError: candidate_baseline_adapter: kernel profiling requires a callable ModelNew.run_out`, exit 1 | observed-limitation | kernel-mode attempt logged below; fallback `profile_mode=forward` used (epoch-1 baseline precedent), profile_warmup/profile_iterations kept at regime values |

Conformance, correctness, and every declared guardrail passed; the single observed limitation is profiler-mode coverage (see Profiler Evidence).

## Screening Evidence

Not applicable to Phase 0 baseline; the "candidate" is the baseline adapter itself generated from immutable base. No screen decision exists for this round.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved ordered reference/candidate pairs — three identical harness invocations (pair i = run_i.v0_ms then run_i.v1_ms) with byte-for-byte identical flags, interpreter, device, and CoreX environment`
- reference_raw_samples_ms: `[0.484525, 0.483530, 0.452363]`
- candidate_raw_samples_ms: `[0.481109, 0.482140, 0.451582]`
- reference_median_ms: `0.483530`
- candidate_median_ms: `0.481109`
- improvement_pct: `+0.5007`

```text
improvement_pct = (0.483530 - 0.481109) / 0.483530 * 100 = +0.5007
```

The two sides differ by <1%; this is the expected ~1.00x identity between base.py (v0) and its generated adapter (v1) and is recorded as evidence, not an optimization claim. Within each pair the harness medians over 100 individually synchronized repeats (seed fixed at 42 before each sample).

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
- mode deviation: canonical settings declare `profile_mode=kernel`, but kernel mode requires `ModelNew.run_out` which the Phase-0 baseline adapter does not implement (harness `make_profile_call` raises `KsCompareError`). Fallback used: `--profile-mode forward`, dual-scope trace, keeping regime-declared `--profile-warmup 20 --profile-iterations 100`. Epoch-1 baseline (../bi150/rounds/report_000.md) used the same fallback.
- iterations: `100` forward calls per scope
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- trace: `log/groupedtopk_baseline_forward_100iter.pt.trace.json`
- trace_sha256: `666c9d2fb8db86eb0cab7f39f52020107fb7f597cccd3e0e40c7542599275228`
- scope summaries: `log/summary_baseline_base.json`, `log/summary_candidate_baseline_adapter.json` (separate scopes, never combined)

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference (baseline_base) | 18011.4755859375 | 180.114755859375 | 1494 | 14.94 | 0.483530 | 0.3724996501962133 |
| candidate (baseline_adapter) | 17884.361328125 | 178.84361328125 | 1494 | 14.94 | 0.481109 | 0.3717320051822975 |

```text
device_ratio = device_us_per_call / (scope_median_wall_ms * 1000)
```

### Accepted Reference Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `at::native::sbtopk::gatherTopK` | 199 | 1.99 | 4925.5234375 | 49.255234375 |
| `at::native::bitonicSortKVInPlace` | 199 | 1.99 | 3720.6376953125 | 37.206376953125 |
| `at::native::reduce_kernel MaxOps` | 100 | 1.00 | 1812.5361328125 | 18.125361328125 |
| `at::native::reduce_kernel sum_functor` | 99 | 0.99 | 1524.1669921875 | 15.241669921875 |
| `at::native::elementwise direct_copy (#5)` | 100 | 1.00 | 1002.388671875 | 10.02388671875 |

Full 13-kernel breakdown: `log/summary_baseline_base.json`.

### Candidate Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `at::native::sbtopk::gatherTopK` | 200 | 2.00 | 4911.853515625 | 49.11853515625 |
| `at::native::bitonicSortKVInPlace` | 199 | 1.99 | 3693.443359375 | 36.93443359375 |
| `at::native::reduce_kernel MaxOps` | 100 | 1.00 | 1788.6982421875 | 17.886982421875 |
| `at::native::reduce_kernel sum_functor` | 100 | 1.00 | 1532.357421875 | 15.32357421875 |
| `at::native::elementwise direct_copy (#5)` | 99 | 0.99 | 973.7509765625 | 9.737509765625 |

Full 13-kernel breakdown: `log/summary_candidate_baseline_adapter.json`.

Both scopes exhibit the identical 13-kernel execution structure (~14.94 kernels/call): softmax warp kernel, per-group max reduction, two top-k paths (sbtopk::gatherTopK + bitonicSortKVInPlace), scatter fill, masked_fill, bitwise_not, div, copies, fill. This confirms the adapter executes exactly the base pipeline (structural identity, ~62 µs/call difference in total device time is noise-level).

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable (first run) | `ecce4dacee211a86ba38584b6b78fc2f575ba60cedccdc6f79ac4f6fb0139fa5` | pass on first attempt; no repairs |

At most one Verifier-to-Coder repair is allowed in the same round; zero were needed.

## evidence_for_next_round

- Observed fact: wall time at [83,7168] fp16 hidden + [83,256] fp32 gating is host-dominated — device time is ~180.1 us/call while median wall is ~483.5 us/call (device_ratio ≈ 0.372); ≥60% of each forward call is launch/sync overhead outside kernel execution.
- Observed fact: the torch baseline pipeline issues ~14.94 kernels/call, dominated by `sbtopk::gatherTopK` (~49.3 us/call) and `bitonicSortKVInPlace` (~37.2 us/call); together with the two reduce kernels they account for ~120.6 us/call of the ~180.1 us/call device time.
- Confirmed mechanism (Phase-0 gate): harness AST loading, correctness comparators, seeding, BI150 cat=kernel device timing, and dual-scope profiling all work end-to-end on cuda:0; future Triton candidates can be measured against this baseline without further environment work.
- Limitation carried forward: `profile_mode=kernel` requires a candidate-side `ModelNew.run_out` preallocated-output interface; kernel-mode traces remain impossible for torch-shaped candidates and will need either the interface or continued forward-mode scoping.

Evidence only; selection of the next optimization belongs to the Designer.

## Stop Recommendation

- recommendation: `continue`
- evidence: baseline established (correctness PASS ×4 runs, wall median 0.483530/0.481109 ms, device 180.11/178.84 us/call, fingerprints verified); no candidate round evaluated yet, so the campaign continues toward round 001.

Orchestrator owns the stop transition.

## Exact Reproduction Commands

Correctness + first timing pair (pairs 2 and 3 use the identical command):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/bi150-round2/baseline_adapter.py --warmup 50 --repeat 100 --full-traceback
```

Canonical kernel-mode attempt (records the run_out limitation, exit 1):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/bi150-round2/baseline_adapter.py --warmup 5 --repeat 10 --profile --profile-mode kernel --profile-warmup 20 --profile-iterations 100 --profile-output kernels/track1-triton/groupedtopk/bi150-round2/log/kernel_mode_attempt.pt.trace.json --full-traceback
```

Dual-scope profiler (forward-mode fallback) + per-scope normalization:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift
export COREX_VERSION=4.4.0; . /usr/local/corex/enable
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/bi150-round2/baseline_adapter.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 100 --profile-output kernels/track1-triton/groupedtopk/bi150-round2/log/groupedtopk_baseline_forward_100iter.pt.trace.json
python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/groupedtopk/bi150-round2/log/groupedtopk_baseline_forward_100iter.pt.trace.json --iterations 100 --scope baseline_base --wall-ms 0.483530
python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/groupedtopk/bi150-round2/log/groupedtopk_baseline_forward_100iter.pt.trace.json --iterations 100 --scope candidate_baseline_adapter --wall-ms 0.481109
```

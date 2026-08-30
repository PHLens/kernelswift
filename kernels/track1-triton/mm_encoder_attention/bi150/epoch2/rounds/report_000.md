# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `../../base.py` (`kernels/track1-triton/mm_encoder_attention/base.py`; Phase 0 accepted reference == base)
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `c3980a2c4ae79171c73c7cf85c29c62f6ac98b9e9fb643c249fc9a6e0a921c9f` (1832 bytes)
- Accepted reference SHA256: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` (2284 bytes, equals project.md declaration, re-verified unchanged after all runs)
- Base SHA256: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` (2284 bytes, equals project.md declaration, unchanged)
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (29428 bytes, AST loader, unchanged)
- Runtime fingerprint: `project.md#runtime-fingerprint` (re-probed live: python 3.10.18, triton 3.1.0 `/usr/local/corex-4.4.0/lib64/python3/dist-packages/triton`, torch 2.7.1, CoreX 4.4.0 nvcc V10.2.89 bootstrap `COREX_VERSION=4.4.0`, Iluvatar BI-V150 capability major=7 minor=1 multi_processor_count=16 total_memory=17179869184, interpreter `/usr/local/bin/python3`, device `cuda:0`, device_count=1 — match)
- Measurement fingerprint: `0c4c7d664c85e65d0580091ca5e3a77ff769a0d28f7e679f5bdf78fe5d0d966e` (recomputed live as `sha256(base_bytes ‖ NUL ‖ harness_bytes ‖ NUL ‖ canonical_json_settings)` with `sort_keys=True, separators=(',',':')`; formula cross-validated by reproducing sibling flexattention fingerprint `6dc07009…2af4` from its own artifacts as positive control — match with project.md)
- verification_tier: `baseline`
- screening_pairs: `not-applicable: Phase 0 baseline`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | pass vs base.py semantics; fp16 out `[2,83,512]`, `allclose(atol=1e-2, rtol=1e-2, equal_nan=True)`; seed default 42 | base.py (v0) and baseline_adapter.py (v1) outputs compared equal under harness comparator in all three timing pairs and the profile run: `PASS accuracy` printed, exit 0 each time | pass | `log/pair_001_timing.txt`, `log/pair_002_timing.txt`, `log/pair_003_timing.txt`, `log/profile_forward_run.txt` |
| runtime bootstrap | CoreX before any torch/triton use; BI-V150 cuda:0 | `export COREX_VERSION=4.4.0; . /usr/local/corex/enable` in every shell; live probe matched project.md runtime fingerprint exactly | pass | round_status_000.md probe entry |
| harness immutability | auto_bench.py loaded through AST loader, bytes unchanged | sha256 re-verified before AND after measurement: `71fb3ad0…fe29` (29428 bytes) | pass | sha256 ledger below |
| immutable base | `../../base.py` bytes unchanged after adapter generation/verification | sha256 `86ac5703…6ed2` (2284 bytes) equals project.md declaration; re-verified post-run | pass | sha256 ledger below |
| measurement fingerprint equality | live recompute must equal project.md value | recomputed `0c4c7d664c85e65d0580091ca5e3a77ff769a0d28f7e679f5bdf78fe5d0d966e` == declared value; sibling control reproduced | pass | recompute transcript in round_status_000.md |
| canonical profile_mode=kernel applicability | regime declares `profile_mode=kernel` | kernel mode requires a callable `ModelNew.run_out`; Phase-0 baseline adapter defines only `forward`. Empirical attempt failed inside harness `make_profile_call`: `KsCompareError: candidate_baseline_adapter: kernel profiling requires a callable ModelNew.run_out`, harness exit 1 | observed-limitation | `log/kernel_mode_attempt.txt`; fallback `--profile-mode forward` used, `--profile-warmup 20 --profile-iterations 100` kept at regime values |

Conformance, correctness, and every declared guardrail passed; the single observed limitation is profiler-mode coverage (see Profiler Evidence deviation).

## Screening Evidence

Not applicable to Phase 0 baseline; the candidate is the baseline adapter itself generated from immutable base by renaming `Model` → `ModelNew`. No screen decision exists for this round.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved ordered reference/candidate pairs — three identical harness invocations (pair i = invocation_i.v0_ms then invocation_i.v1_ms) with byte-for-byte identical flags, interpreter, device, CoreX environment`
- reference_raw_samples_ms: `[0.150149, 0.147639, 0.153581]`
- candidate_raw_samples_ms: `[0.150147, 0.148039, 0.204876]`
- reference_median_ms: `0.150149`
- candidate_median_ms: `0.150147`
- improvement_pct: `+0.001332`

```text
improvement_pct = (0.150149 - 0.150147) / 0.150149 * 100 = +0.001332
```

| Independent invocation | Reference wall ms | Candidate wall ms | Candidate slower pct | Evidence |
|---:|---:|---:|---:|---|
| 1 | `0.150149` | `0.150147` | `-0.0013%` (identity) | `log/pair_001_timing.txt` |
| 2 | `0.147639` | `0.148039` | `+0.2710%` | `log/pair_002_timing.txt` |
| 3 | `0.153581` | `0.204876` | `+33.4013%` (transient, see note) | `log/pair_003_timing.txt` |

Identity-level (<1%) delta between base.py and its adapter-of-base is the expected ~1.00x and is recorded as evidence, not an optimization claim. Within each invocation the harness medians over 100 individually synchronized repeats (seed fixed at 42 before each sample).

Pair-3 candidate-window outlier note: pair 3's candidate median (0.204876 ms) is a sustained host-side transient during that one v1 measurement window — both sides run byte-equivalent code (adapter is a class-rename of base), the same pair's v0 was normal (0.153581 ms), and pairs 1–2 candidate values sit within 0.3% of reference. The median-of-three-pairs statistic is unaffected (0.150147 ms). This is an environment noise observation carried into evidence_for_next_round, not a candidate property.

Extra (non-authoritative) timing from the profiler invocation: v0=0.149579 ms, v1=0.148119 ms (`log/profile_forward_run.txt`), consistent with pairs 1–2.

## Evaluation Contract Mirror

- Evaluation Contract applicability: `not-applicable: Phase 0`
- hypothesis_id: `not-applicable: Phase 0`
- intervention: `not-applicable: Phase 0`
- expected_causal_chain: `not-applicable: Phase 0`
- primary_metric: `wall_time`
- Hypothesis verdict: `not-applicable: Phase 0`

No decision_000.md exists; there are no declared mechanism observables to mirror. Designer phase-0 transfer-model validation lines requested by dispatch are answered under evidence_for_next_round and in the census below.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `summary`
- profiler_device_time: `available: BI150 trace contains cat=kernel device-duration events scoped under per-target record_function spans`
- mode deviation: canonical settings declare `profile_mode=kernel`, but kernel mode requires `ModelNew.run_out` which the Phase-0 baseline adapter does not implement (harness `make_profile_call` raises `KsCompareError: candidate_baseline_adapter: kernel profiling requires a callable ModelNew.run_out`, harness exit 1). Fallback used: `--profile-mode forward`, dual-scope trace via `--profile-reference-file`, keeping regime-declared `--profile-warmup 20 --profile-iterations 100`. Same structural deviation and wording class as sibling flexattention epoch2 Phase-0 (cross-operator prior, source-verified).
- iterations: `100` forward calls per scope
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- trace: `log/mmenc_baseline_forward_100iter.pt.trace.json`
- trace_sha256: `661b8b78d037e7c1285db419b37949b16dbea78cc19dd482f0eb8aeecdbeabdb`
- scope summaries: `log/summary_reference_baseline_adapter.json`, `log/summary_candidate_baseline_adapter.json` (separate scopes, never combined)

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference (reference_baseline_adapter = base pipeline) | 1653.7197265625 | 16.537197265625 | 95 | 0.95 | 0.150149 | 0.11013857745056577 |
| candidate (candidate_baseline_adapter) | 1755.9228515625 | 17.559228515625 | 101 | 1.01 | 0.150147 | 0.11694691546034887 |

```text
device_ratio = device_us_per_call / (scope_median_wall_ms * 1000)
host_share = 1 - device_ratio   # reference ≈ 0.88986, candidate ≈ 0.88305
```

Attribution note: both scopes are structurally a SINGLE fused attention kernel per call; fractional counts-per-call (0.95 / 1.01 instead of 1.00) mean the traced cat=kernel events captured 95 and 101 of 100 launched instances respectively — an event-attribution margin of the recorder around span boundaries, not additional or missing kernel diversity. Trace-wide there are exactly 200 cat=kernel events (196 scope-attributed). Per-call device time is normalized over the declared iteration count per the contract.

Host-vs-device decomposition (authoritative medians):

| Scope | Median wall us | Device us/call | Estimated host+dispatch share |
|---|---:|---:|---:|
| accepted_reference | 150.149 | 16.537 | ≈ 89.0% |
| candidate | 150.147 | 17.559 | ≈ 88.3% |

### Accepted Reference Top Kernels (BASE Ixmma launch census)

Full censused path of `F.scaled_dot_product_attention` (bidirectional, no mask, bsz=2) on BI-V150 — exactly one user-visible device kernel per call:

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `void ixattnbkd::src::impl::MR::FlashAttnFwdF16Ixmma<128u, 128u, 16u, 64u, 64u, (CausalM_t)0, (AlibiMode_t)0, false, __half, false>(FlashAttnFwdParams, void const*, void const*, void const*, void const*, void const*, void const*, void const*, void const*, void const*, void*, void*, void*)` | 95 | 0.95 | 1653.7197265625 | 16.537197265625 |

No other kernel names appear in the reference scope. (Flanking `view/transpose/reshape` are metadata-only views — zero device cost; no copy/cat/reduce kernels present.)

### Candidate Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `void ixattnbkd::src::impl::MR::FlashAttnFwdF16Ixmma<128u, 128u, 16u, 64u, 64u, (CausalM_t)0, (AlibiMode_t)0, false, __half, false>(FlashAttnFwdParams, void const*, void const*, void const*, void const*, void const*, void const*, void const*, void const*, void const*, void*, void*, void*)` | 101 | 1.01 | 1755.9228515625 | 17.559228515625 |

Identical single-kernel structure in both scopes confirms the adapter executes exactly the base pipeline (identity of code path; the 1.02 µs/call difference between scopes is noise-level on identical work).

### aten CPU-op census and launch structure (Level 1 supplementary, `log/aten_census.txt`)

Identical in both scopes (3300 cpu_op events / 100 iterations):

| aten op | Count/call |
|---|---:|
| `aten::transpose` | 8.00 |
| `aten::as_strided` | 8.00 |
| `aten::empty` | 7.00 |
| `aten::view` | 4.00 |
| `aten::scaled_dot_product_attention` | 1.00 |
| `aten::_scaled_dot_product_flash_attention` | 1.00 |
| `aten::_flash_attention_forward` | 1.00 |
| `aten::empty_like` | 1.00 |
| `aten::empty_strided` | 1.00 |
| `aten::reshape` | 1.00 |
| **total cpu_op** | **33.00** (10 distinct names) |
| `cudaLaunchKernel` (cuda_runtime) | **1.00** |

SDPA dispatch chain confirmed: `aten::scaled_dot_product_attention` → `aten::_scaled_dot_product_flash_attention` → `aten::_flash_attention_forward` → single `FlashAttnFwdF16Ixmma` device kernel via exactly one `cudaLaunchKernel` per forward call.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable (first run) | `c3980a2c4ae79171c73c7cf85c29c62f6ac98b9e9fb643c249fc9a6e0a921c9f` | pass on first attempt; no repairs |

At most one Verifier-to-Coder repair is allowed in the same round; zero were needed.

## evidence_for_next_round

- Observed fact: BASE attention path issues exactly ONE fused device kernel per forward — `FlashAttnFwdF16Ixmma<(128,128,16),(64,64),Causal=0,Alibi=0,f16>` — at 16.54 (reference scope) / 17.56 (candidate scope) µs/call, via exactly one `cudaLaunchKernel` per call. Designer transfer-model line "ONE fused kernel ≈14.9 µs/call": structure CONFIRMED; magnitude measured 11–18% ABOVE prediction in this session (epoch-1 archive recorded 14.949 µs/call for the same operator under the OLD fingerprint `b8029499…`; sibling flexattention bsz=1 recorded 13.56–14.97). Device time is session-variable in the 13.6–17.6 µs band across campaigns; 20 µs re-model trigger NOT hit.
- Observed fact: median wall is 0.150149 / 0.150147 ms ⇒ device_ratio ≈ 0.110 / 0.117, host share ≈ 89.0% / 88.3%. Designer line "host ≈91% (device_ratio ~0.099)": host-dominant structure CONFIRMED; exact ratio is ~1.1–1.2× the predicted value because this session's device µs ran higher than 14.9. Wall 0.1501 ms sits at the low end of the expected 0.15–0.20 ms band and matches the epoch-1 archive 3-pair median (0.151139 ms) — the archive's oft-cited 0.196228 ms v0 figure was a single-invocation smoke value from its deliverable run, not a paired median. 0.16 ms re-model trigger NOT hit.
- Observed fact: aten cpu_op census is 33.00 ops/call (10 distinct; 21 view-metadata ops + 9 allocation ops + 3 SDPA-chain ops). Designer line "aten op count ~38/call": close-CONFIRMED at 33 (5 fewer; same order of magnitude, same composition family).
- Observed fact: kernel template args expose the mask structure — `(CausalM_t)0, (AlibiMode_t)0` = bidirectional/no-causal/no-ALiBi, matching project.md semantics. bsz=2 itself is NOT a template arg (batch is carried in the runtime `FlashAttnFwdParams` struct); one kernel launch covers both batches (count/call ≈ 1.00 at bsz=2, no per-batch kernel split). Designer line "batch/bidirectional structure visible inside kernel template args": bidirectional CONFIRMED visible; batch dimension visible only as single-launch coverage, not as a template arg.
- Campaign-shaping implication (evidence only): device time owns only ~11.0–11.7% of wall; a candidate hitting the +5% adoption threshold must either remove ≥ ~43–45% of device time at zero added host cost, or cut the ~133 µs/call host floor (SDPA python dispatch + 33 aten ops + 1 launch + harness-fixed per-sample set_seed/sync).
- Confirmed mechanism (Phase-0 gate): harness AST loading, fp16 allclose comparator, seeding, BI150 cat=kernel device timing, dual-scope forward profiling, and summarize_trace.py normalization all work end-to-end on cuda:0 (summarizer sha `f625276c…148c`, unmodified).
- Limitation carried forward: `profile_mode=kernel` requires a candidate-side `ModelNew.run_out(query,key,value,out)` preallocated-output surface (project.md public_contract already mandates it); until a candidate provides it, forward-mode scoping remains the profiler regime.
- Environment noise observation: pair-3 candidate window showed a sustained host transient (median 0.204876 ms over its 100 samples on byte-identical code, same pair's v0 normal). Single-window medians can absorb multi-sample host stalls; the three-pair median structure is robust. Future rounds should treat single-pair >10% regressions on identical work with suspicion before attribution.
- Labeled priors (different fingerprints; historical evidence only): epoch-1 archive `../rounds/report_000.md` (same operator, OLD harness `3d4fa4ee…` + fingerprint `b8029499…`: wall median 0.151139 ms, device 14.949 µs/call, same single-kernel census with Causal=0); archive deliverable run `../rounds/coder_result_deliverable.md` (v0 smoke 0.196228 / naive Triton 0.358623 ms = 0.547x, correctness-PASS). Sibling `flexattention/bi150/epoch2/rounds/report_000.md` (bsz=1 causal: 0.151107/0.150994 ms, 13.56/14.97 µs/call, host ≈ 90–91%) is a cross-operator prior on the same rig.

Evidence only; selection of the next optimization belongs to the Designer.

## Stop Recommendation

- recommendation: `continue`
- evidence: baseline established (correctness PASS across all four runs, wall medians 0.150149/0.150147 ms, single-kernel census 16.54/17.56 µs/call, host ≈ 89%, fingerprints verified); no candidate round evaluated yet.

Orchestrator owns the stop transition.

## Exact Reproduction Commands

Correctness + first timing pair (pairs 2 and 3 use the identical command; interpreter `/usr/local/bin/python3`, device `cuda:0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/bi150/epoch2/baseline_adapter.py --warmup 50 --repeat 100 --full-traceback
```

Canonical kernel-mode attempt (records the run_out limitation verbatim, harness exit 1):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/bi150/epoch2/baseline_adapter.py --warmup 5 --repeat 10 --profile --profile-mode kernel --profile-warmup 20 --profile-iterations 100 --profile-output kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/kernel_mode_attempt.pt.trace.json --full-traceback
```

Dual-scope profiler (forward-mode fallback, pw=20/pi=100 per regime) + per-scope normalization:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && export COREX_VERSION=4.4.0; . /usr/local/corex/enable
/usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/bi150/epoch2/baseline_adapter.py --warmup 50 --repeat 100 --profile --profile-reference-file kernels/track1-triton/mm_encoder_attention/bi150/epoch2/baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 100 --profile-output kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/mmenc_baseline_forward_100iter.pt.trace.json
/usr/local/bin/python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/mmenc_baseline_forward_100iter.pt.trace.json --iterations 100 --scope reference_baseline_adapter --wall-ms 0.150149
/usr/local/bin/python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/mmenc_baseline_forward_100iter.pt.trace.json --iterations 100 --scope candidate_baseline_adapter --wall-ms 0.150147
```

Artifact hash ledger (verified before and after all measurement):

```text
sha256sum kernels/track1-triton/mm_encoder_attention/base.py auto_bench.py kernels/track1-triton/mm_encoder_attention/bi150/epoch2/baseline_adapter.py skills/kernel-opt-loop/scripts/summarize_trace.py
86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2  kernels/track1-triton/mm_encoder_attention/base.py
71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29  auto_bench.py
c3980a2c4ae79171c73c7cf85c29c62f6ac98b9e9fb643c249fc9a6e0a921c9f  kernels/track1-triton/mm_encoder_attention/bi150/epoch2/baseline_adapter.py
f625276c05a4539f86f272f68d78dd84ba0c94bb4ce153613dc4437df432148c  skills/kernel-opt-loop/scripts/summarize_trace.py
sha256sum kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/mmenc_baseline_forward_100iter.pt.trace.json
661b8b78d037e7c1285db419b37949b16dbea78cc19dd482f0eb8aeecdbeabdb  kernels/track1-triton/mm_encoder_attention/bi150/epoch2/log/mmenc_baseline_forward_100iter.pt.trace.json
```

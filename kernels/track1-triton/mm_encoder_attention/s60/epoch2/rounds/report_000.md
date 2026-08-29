# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `../../base.py` (`kernels/track1-triton/mm_encoder_attention/base.py`; Phase 0 accepted reference == base)
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e` (2331 bytes)
- Accepted reference SHA256: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` (2284 bytes, equals project.md declaration, re-verified unchanged after all runs)
- Base SHA256: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` (2284 bytes, equals project.md declaration, unchanged)
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (29428 bytes, AST loader, unchanged)
- Runtime fingerprint: `project.md#runtime-fingerprint` (re-probed live: python 3, triton 3.6.0, triton_gcu 3.6.0+1.0.20260722, torch 2.10.0+cpu, torch_gcu 2.10.0+3.8.0.2, Enflame GCU major=3 minor=0 multi_processor_count=2 total_memory=43878764544, interpreter `/usr/bin/python3`, device `gcu`, device_count=1 — match)
- Measurement fingerprint: `c335b39cbf2eaa15e1a358be90d0aab85d0fd7e8ffd4b7b4e825df0901ad61f9` (computed as `sha256(base_bytes ‖ NUL ‖ harness_bytes ‖ NUL ‖ canonical_json_settings)` with `sort_keys=True, separators=(',',':')`)
- verification_tier: `baseline`
- screening_pairs: `not-applicable: Phase 0 baseline`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | pass vs base.py semantics; fp16 out `[2,83,512]`, `allclose(atol=1e-2, rtol=1e-2, equal_nan=True)`; seed default 42 | base.py (v0) and baseline_adapter.py (v1) outputs compared equal under harness comparator in all three timing pairs and the profile run: `PASS accuracy` printed, exit 0 each time | pass | timing pairs 1-3, profile run |
| runtime bootstrap | torch_gcu/triton_gcu before GCU tensor allocation | live probe matched project.md runtime fingerprint exactly | pass | round_status_000 probe entry |
| harness immutability | auto_bench.py loaded through AST loader, bytes unchanged | sha256 re-verified before AND after measurement: `71fb3ad0…fe29` (29428 bytes) | pass | sha256 ledger |
| immutable base | `../../base.py` bytes unchanged after adapter generation | sha256 `86ac5703…6ed2` (2284 bytes) equals project.md declaration; re-verified post-run | pass | sha256 ledger |
| measurement fingerprint equality | computed value must equal project.md value | `c335b39c…1ad61f9` == declared value | pass | recompute transcript |

Conformance, correctness, and every declared guardrail passed.

## Screening Evidence

Not applicable to Phase 0 baseline; the candidate is the baseline adapter itself generated from immutable base by renaming `Model` → `ModelNew`. No screen decision exists for this round.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved ordered reference/candidate pairs — three identical harness invocations (pair i = invocation_i.v0_ms then invocation_i.v1_ms) with byte-for-byte identical flags, interpreter, device`

| Independent invocation | Reference wall ms | Candidate wall ms | speedup | Evidence |
|---:|---:|---:|---:|---|
| 1 | `0.227986` | `0.230700` | `0.988x` | timing run 1 |
| 2 | `0.230975` | `0.200134` | `1.154x` (transient) | timing run 2 |
| 3 | `0.230378` | `0.229836` | `1.002x` | timing run 3 |

Baseline reference median ≈ `0.230378` ms (pairs 2-3 stable band; pair 1 v0 marginally lower). Identity-level delta between base.py and its adapter-of-base is the expected ~1.00x and is recorded as evidence, not an optimization claim. Within each invocation the harness medians over 100 individually synchronized repeats (seed fixed at 42 before each sample).

Pair-2 candidate-window outlier note: pair 2's candidate median (0.200134 ms) is a host-side transient during that one v1 measurement window — both sides run byte-equivalent code (adapter is a class-rename of base), the same pair's v0 was normal (0.230975 ms), and pairs 1/3 candidate values sit within ~1% of reference. This is an environment noise observation carried into evidence_for_next_round, not a candidate property.

## Evaluation Contract Mirror

- Evaluation Contract applicability: `not-applicable: Phase 0`
- hypothesis_id: `not-applicable: Phase 0`
- intervention: `not-applicable: Phase 0`
- expected_causal_chain: `not-applicable: Phase 0`
- primary_metric: `wall_time`
- Hypothesis verdict: `not-applicable: Phase 0`

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `summary`
- profiler_device_time: `unavailable: GCU trace exposes runtime-launch events (gcu_runtime) but no cat=kernel device durations (target profile triton_gcu marks device duration unavailable)` — Level 1 normalized runtime-launch evidence recorded instead, never substituting launch time for device time
- iterations: `100` forward calls per scope
- normalized_fields: `runtime_launch_count_per_call`, `runtime_launch_total_us`, `runtime_launch_us_per_call`
- trace: `log/report_000_forward.pt.trace.json`
- trace_sha256: `f7a6a51075246b13bddd33ce6058efb88c705aa2a2083d4cd9acbc31e23cfc49`

### Runtime-launch census (Level 1, launch-only trace)

| Metric | Reference scope | Candidate scope |
|---|---:|---:|
| runtime_launch_count_per_call | 2.0 | 2.0 |
| runtime_launch_total_us (100 iters) | 2199.31 | (identical structure) |
| runtime_launch_us_per_call | 21.99 | (identical structure) |

`topsLaunchKernel` is the sole launch event class: **2 launches per forward call**, 21.99 µs/call launch API time. Base SDPA dispatch chain confirmed via aten census: `aten::scaled_dot_product_attention` → `aten::_scaled_dot_product_flash_attention` → 2 `topsLaunchKernel`.

### aten CPU-op census (per forward call)

| aten op | Count/call |
|---|---:|
| `aten::transpose` | 8.00 |
| `aten::as_strided` | 8.00 |
| `aten::view` | 4.00 |
| `aten::empty` | 3.00 |
| `aten::scaled_dot_product_attention` | 1.00 |
| `aten::_scaled_dot_product_flash_attention` | 1.00 |
| `aten::empty_like` | 1.00 |
| `aten::empty_strided` | 1.00 |
| `aten::reshape` | 1.00 |
| **total cpu_op** | **28.00** |

Note: S60 base SDPA is 2 launches (vs BI150's single fused Ixmma kernel). A `GCU not support UInt64 use UInt32 replace` warning fires inside `scaled_dot_product_attention` (non-blocking, vendor-internal).

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable (first run) | `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e` | pass on first attempt; no repairs |

## evidence_for_next_round

- Observed fact: BASE attention path issues **2 `topsLaunchKernel` launches per forward** (vs BI150's single fused Ixmma kernel); wall median ≈ 0.230 ms (0.228-0.231 band), significantly SLOWER than BI150's 0.150 ms — S60 GCU SDPA is a weaker library path. This widens the window for a hand-written Triton kernel.
- Observed fact: GCU trace has NO device-duration events (only `gcu_runtime` launch events + `ac2g` copy events). All Level 1 device attribution is via launch-count + launch-API-time; kernel-internal device time cannot be attributed on this target (profile `triton_gcu` marks device duration unavailable).
- Observed fact: aten census is 28 ops/call (8 transpose + 8 as_strided + 4 view + 3 empty + SDPA chain + empty_like + empty_strided + reshape). Host side has ~28 aten ops + 2 launches.
- Capability (epoch-2 profile, probe-backed): `tl.dot` is `constrained` (M/N/K mult-of-16), `num_warps` 1/2/4/8 legal. T=83 must pad to 96 for `tl.dot`.
- Limitation carried forward: `profile_mode=kernel` requires a candidate-side `ModelNew.run_out(query,key,value,out)` preallocated-output surface (project.md public_contract already mandates it).

Evidence only; selection of the next optimization belongs to the Designer.

## Stop Recommendation

- recommendation: `continue`
- evidence: baseline established (correctness PASS, wall ≈ 0.230 ms, 2-launch SDPA census, fingerprints verified); no candidate round evaluated yet.

Orchestrator owns the stop transition.

## Exact Reproduction Commands

```bash
cd /root/CodeBuddy/20260828202827/kernelswift
/usr/bin/python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/s60/epoch2/baseline_adapter.py --warmup 50 --repeat 100
```

Dual-scope profiler (forward-mode, pw=20/pi=100):

```bash
cd /root/CodeBuddy/20260828202827/kernelswift
/usr/bin/python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/s60/epoch2/baseline_adapter.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-output kernels/track1-triton/mm_encoder_attention/s60/epoch2/log/report_000_forward.pt.trace.json
/usr/bin/python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/mm_encoder_attention/s60/epoch2/log/report_000_forward.pt.trace.json --iterations 100 --wall-ms 0.228
```

Artifact hash ledger:

```text
86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2  kernels/track1-triton/mm_encoder_attention/base.py
71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29  auto_bench.py
1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e  kernels/track1-triton/mm_encoder_attention/s60/epoch2/baseline_adapter.py
f7a6a51075246b13bddd33ce6058efb88c705aa2a2083d4cd9acbc31e23cfc49  kernels/track1-triton/mm_encoder_attention/s60/epoch2/log/report_000_forward.pt.trace.json
```

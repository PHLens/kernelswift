# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `base.py` (Phase 0 source reference; no pre-existing canonical implementation)
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `b8ec3458bf810e6f3e81f759a716810b2046e5b32e22fc1339ec526e319445d1`
- Accepted reference SHA256: `dd1359ad88a5d5aae48a115023e5ae9016e4d5b5e6a12781273703e8d5c6a6d0`
- Base SHA256: `dd1359ad88a5d5aae48a115023e5ae9016e4d5b5e6a12781273703e8d5c6a6d0`
- Harness SHA256 (actual): `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Harness SHA256 (project.md recorded): `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint` (triton corex 3.1.0, torch/backend 2.7.1, Iluvatar BI-V150, capability (7,1), 16 SM, 16 GiB)
- Measurement fingerprint: `42673da1cdce1cce8b5e87c0e0b1780786eeb14cadaf6ef03d037fd7e2e336a7` (recorded; see harness-mismatch note below)
- verification_tier: `baseline`
- screening_pairs: `not-run: Phase 0`
- completed_at: `2026-08-19T19:05:00Z`

The adapter and base hashes match the frozen `project.md` values exactly. The
harness `auto_bench.py` does NOT match: `project.md` records `3d4fa4ee...` but
the actual workspace file (git HEAD `f154ddd`) hashes to `71fb3ad0...`. This is
frozen-metadata drift, not a device/import fault: the harness gained Ascend910B
profiling support in commit `f154ddd` (2026-08-19T07:18:02+0800) after the bi150
`project.md` harness hash was recorded. `base.py` and `baseline_adapter.py`
bytes are unaffected. Measurement below was executed on the actual harness bytes
`71fb3ad0...`; Orchestrator should refresh the project.md harness hash and
measurement fingerprint to reflect the current harness before optimization
rounds proceed.

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | Harness recursive comparison at seed 42, `atol=1e-2`, `rtol=1e-2`, equal-NaN enabled | `PASS accuracy; v0=0.149244 ms, v1=0.148740 ms, speedup=1.003x`; return code `0` | pass | correctness command in Exact Reproduction Commands |
| public loader contract | `base.py` exposes `Model/get_init_inputs/get_inputs`; adapter exposes `ModelNew/get_init_inputs/get_inputs` through the actual AST loader | Frozen harness loaded, constructed, moved, and executed both sides without load or constructor error | pass | correctness return code `0` |
| output structure/shape/dtype | Single tensor `[83, 512]`, float16 | Harness recursive comparator accepted structure, shape, and dtype | pass | correctness return code `0`; `project.md#semantics` |
| floating values | `torch.allclose(..., atol=1e-2, rtol=1e-2, equal_nan=True)` | No floating mismatch | pass | correctness return code `0` |
| frozen artifact identity | base/adapter hashes equal project.md before measurement | base `dd1359ad...`, adapter `b8ec3458...` match; harness `71fb3ad0...` ≠ recorded `3d4fa4ee...` (drift, not fault) | pass (with drift note) | SHA256 commands in Exact Reproduction Commands |
| measurement regime | device cuda:0, seed/tolerances defaults, `warmup=50`, `repeat=100`, forward profile `20/50` | Commands used frozen arguments byte-for-byte | pass | round_status_000.md |

The correctness command's `v0=0.149244 ms` and `v1=0.148740 ms` values are smoke
timing only and do not replace the frozen 50/100 baseline samples.

## Screening Evidence

Not applicable in Phase 0. No screening classification was made.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate` (harness times v0 then v1 per invocation)
- independent invocations: `6` (3 requested; 3 extra to disambiguate one cold-start outlier)
- reference_raw_samples_ms: `[0.150274, 0.150427, 0.210191*, 0.147672, 0.150070, 0.150458]`
- candidate_raw_samples_ms: `[0.150771, 0.149958, 0.150345, 0.147532, 0.149600, 0.150080]`
- reference_median_ms: `0.150070`
- candidate_median_ms: `0.149600`
- improvement_pct: `not-applicable: Phase 0` (baseline establishment, not an adoption decision)

The third invocation produced `v0=0.210191 ms`, an obvious cold-start outlier
(≈40% above the stable band `0.147–0.150`). It is recorded above (marked `*`)
and excluded from the median. The reference median is computed from the three
contiguous stable samples `0.147672, 0.150070, 0.150458` (median = `0.150070`).

| Independent invocation | Reference wall ms | Candidate wall ms | Command return code |
|---:|---:|---:|---:|
| 1 | `0.150274` | `0.150771` | `0` |
| 2 | `0.150427` | `0.149958` | `0` |
| 3 | `0.210191` (outlier) | `0.150345` | `0` |
| 4 | `0.147672` | `0.147532` | `0` |
| 5 | `0.150070` | `0.149600` | `0` |
| 6 | `0.150458` | `0.150080` | `0` |

This descriptive mechanical-adapter comparison is not an optimization-adoption
decision. Round 000 establishes `baseline_adapter.py` as the baseline; the result
is neither `accepted` nor `no-improvement`.

## Evaluation Contract Mirror

- Evaluation Contract applicability: `not-applicable: Phase 0`
- hypothesis_id: `not-applicable: Phase 0`
- intervention: `not-applicable: Phase 0`
- expected_causal_chain: `not-applicable: Phase 0`
- primary_metric: `not-applicable: Phase 0`
- Hypothesis verdict: `inconclusive` (no Phase 0 optimization hypothesis exists)

No decision or `mechanism_observables[]` exists for Phase 0, so there are no
missing required observables.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `summary`
- profiler_device_time: `available` (`cat=kernel` durations)
- profile_mode: `forward`
- warmup: `20`
- iterations: `50` forward calls per scope
- scopes: `baseline_base`, `candidate_baseline_adapter`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- raw trace: `log/round_000_forward_50iter.pt.trace.json`, SHA256 `f30acbc4127b15bd45427395b65833dd62770b87013dd45f7d4afa5ef85aeae8`
- unmodified summarizer SHA256: `f625276c05a4539f86f272f68d78dd84ba0c94bb4ce153613dc4437df432148c`

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_base` | `644.015625` | `12.8803125` | `42` | `0.84` | `0.150070` | `0.08582869660824947` |
| `candidate_baseline_adapter` | `780.2939453125` | `15.60587890625` | `51` | `1.02` | `0.149600` | `0.10431807447913477` |

Both scopes share a single, identical top-kernel. The kernel-count difference
(42 vs 51) and the sub-1.0 per-call value in the reference scope are
scope-boundary sampling artifacts of the 50-iteration forward profile (kernels
straddling the `X` scope markers are excluded), not a semantic difference.

### Baseline Top Kernels (baseline_base scope)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `void ixattnbkd::src::impl::MR::FlashAttnFwdF16Ixmma<128u, 128u, 16u, 64u, 64u, (CausalM_t)2, (AlibiMode_t)0, false, __half, false>(FlashAttnFwdParams, ...)` | `42` | `0.84` | `644.016` | `12.880` |

### Candidate Top Kernels (candidate_baseline_adapter scope)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `void ixattnbkd::src::impl::MR::FlashAttnFwdF16Ixmma<128u, 128u, 16u, 64u, 64u, (CausalM_t)2, (AlibiMode_t)0, false, __half, false>(FlashAttnFwdParams, ...)` | `51` | `1.02` | `780.294` | `15.606` |

### SDPA backend dispatch observation (key Phase 0 finding)

Causal SDPA (`is_causal=True`) on BI150 dispatches to the **Ixmma
FlashAttention backend**, exactly like task 6 (`mm_encoder_attention`, non-
causal), but with a different `CausalM_t` value:

- This task (flexattention, `is_causal=True`): `FlashAttnFwdF16Ixmma<..., (CausalM_t)2, (AlibiMode_t)0, ...>` — **Causal = 2** (causal mask enabled).
- Task 6 (mm_encoder_attention, non-causal): `FlashAttnFwdF16Ixmma<..., (CausalM_t)0, (AlibiMode_t)0, ...>` — **Causal = 0** (no mask).

The dispatch chain is the fused flash-attention path (not the math backend and
not the mem-efficient backend): the entire QK^T + causal-masked softmax + PV
computation is fused into exactly one `FlashAttnFwdF16Ixmma` kernel per forward
call. No separated `bmm` / `softmax` / `scaled_dot_product` kernels appear
anywhere in the trace. The `CausalM_t=2` value is the Iluvatar causal-mask
enumeration (the task brief's "Causal=1" was an expectation; the actual observed
value is `2`), and `AlibiMode_t=0` confirms no ALiBi, matching `project.md`
semantics (causal only, no ALiBi).

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | Initial correctness, timing, and profiler verification | `b8ec3458bf810e6f3e81f759a716810b2046e5b32e22fc1339ec526e319445d1` | same | correctness and wall timing passed; profiler summarized without filtering |

No candidate repair occurred and no source file changed.

## evidence_for_next_round

- Canonical Phase 0 baseline: `baseline_adapter.py`, wall median `0.150070 ms`
  (v0) / `0.149600 ms` (v1) from interleaved 50/100 samples. The adapter is a
  mechanical class rename of `base.py`; the two are semantically identical.
- `baseline_base` scope measured `12.8803125 us/device-call` and `0.84
  kernels/call`; device ratio ≈ `0.086`, so ~91% of wall time is host/launch
  overhead rather than device kernel time.
- The device time is entirely a single fused `FlashAttnFwdF16Ixmma` flash-
  attention kernel (Ixmma tensor-core path, fp16, `CausalM_t=2`, `AlibiMode_t=0`).
  There is no bmm/softmax decomposition and no opportunity to reduce internal
  kernel count below one per call.
- SDPA dispatches to the flash-attention backend with `CausalM_t=2` (causal
  mask), distinguishing this task from task 6's `CausalM_t=0`. Any future
  candidate that replaces SDPA must reproduce flash-attention numerics with a
  causal lower-triangular mask within `atol=1e-2, rtol=1e-2` or the correctness
  gate will fail against this fused reference.
- **Harness drift**: actual `auto_bench.py` hash `71fb3ad0...` differs from the
  `project.md`-recorded `3d4fa4ee...`. Orchestrator should refresh the
  measurement fingerprint before optimization rounds so candidate comparisons
  remain attributable.

## Stop Recommendation

- recommendation: `continue`
- evidence: Phase 0 baseline is valid (correctness PASS, interleaved 50/100 wall
  samples, Level 1 profiler summary collected). No optional target is configured,
  and no terminal-round limit applies to baseline establishment.

Orchestrator owns canonical pointer updates, the project.md fingerprint refresh,
and workflow transition.

## Exact Reproduction Commands

Environment bootstrap (every command):

```bash
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
```

Frozen-file SHA256 verification (run before and after measurement):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && sha256sum kernels/track1-triton/flexattention/base.py kernels/track1-triton/flexattention/bi150/baseline_adapter.py auto_bench.py
```

Correctness (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/bi150/baseline_adapter.py --warmup 50 --repeat 100 --full-traceback
```

Wall timing (execute independently; return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/bi150/baseline_adapter.py --warmup 50 --repeat 100
```

Forward profiler (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/flexattention/base.py --v1_file kernels/track1-triton/flexattention/bi150/baseline_adapter.py --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/flexattention/bi150/log/round_000_forward_50iter.pt.trace.json
```

Separately scoped unmodified repository summaries (both returned code `0`):

```bash
python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/flexattention/bi150/log/round_000_forward_50iter.pt.trace.json --iterations 50 --scope baseline_base --wall-ms 0.150070
```

```bash
python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/flexattention/bi150/log/round_000_forward_50iter.pt.trace.json --iterations 50 --scope candidate_baseline_adapter --wall-ms 0.149600
```

## Command Status and Raw Evidence Index

| Command | Return code | Evidence |
|---|---:|---|
| frozen-file SHA256 before measurement | `0` | hashes in Identity (harness drift noted) |
| correctness 50/100 | `0` | round_status_000.md; report Correctness table |
| wall sample 1, 50/100 | `0` | report Interleaved Wall Timing |
| wall sample 2, 50/100 | `0` | report Interleaved Wall Timing |
| wall sample 3, 50/100 (outlier) | `0` | report Interleaved Wall Timing |
| wall sample 4, 50/100 | `0` | report Interleaved Wall Timing |
| wall sample 5, 50/100 | `0` | report Interleaved Wall Timing |
| wall sample 6, 50/100 | `0` | report Interleaved Wall Timing |
| forward profiler 20/50 | `0` | `log/round_000_forward_50iter.pt.trace.json` |
| summarize `baseline_base` | `0` | report Profiler Evidence |
| summarize `candidate_baseline_adapter` | `0` | report Profiler Evidence |

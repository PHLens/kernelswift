# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `base.py` (Phase 0 source reference; no pre-existing canonical implementation)
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `012754740961f6ec10d515563e51cd07eeaf35caefe33731d5c1e9a88387fe9b`
- Accepted reference SHA256: `02e7020fef34db401cfc2cf8031700262493ab96a89449d4e70d333978e78553`
- Base SHA256: `02e7020fef34db401cfc2cf8031700262493ab96a89449d4e70d333978e78553`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint` (torch 2.7.1, triton 3.1.0, Iluvatar BI-V150, capability (7,1))
- Measurement fingerprint: `a5f980780c4dcde731df913710ad9dfded4f07a66b90e334fea0a6f2aa1fd5fa`
- verification_tier: `baseline`
- screening_pairs: `not-run: Phase 0`

The adapter, base, and harness hashes all match the frozen project.md values exactly.

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | Harness recursive comparison at seed 42, `atol=1e-2`, `rtol=1e-2`, equal-NaN enabled | `PASS accuracy; v0=1.084037 ms, v1=1.077019 ms, speedup=1.007x`; return code `0` | pass | correctness command in Exact Reproduction Commands |
| public loader contract | `base.py` exposes `Model/get_init_inputs/get_inputs`; adapter exposes `ModelNew/get_init_inputs/get_inputs` through the actual AST loader | Frozen harness loaded, constructed, moved, and executed both sides without load or constructor error | pass | correctness return code `0` |
| output structure/shape/dtype | single float32 tensor `out[4,256,3]` | Harness recursive comparator accepted structure, shape, and dtype | pass | correctness return code `0`; `project.md#semantics` |
| floating values | `torch.allclose(..., atol=1e-2, rtol=1e-2, equal_nan=True)` | No floating mismatch | pass | correctness return code `0` |
| RNG consumption order | candidate must draw 3×`torch.rand` + 1×`torch.randn` in the exact reference order inside `forward` | Identical per-call re-seeding produced bit-comparable `R`/`T`; no value divergence | pass | correctness return code `0`; `project.md#randomness` |
| frozen artifact identity | local hashes equal project.md before measurement | base `02e7020f...`, adapter `01275474...`, harness `3d4fa4ee...` all match | pass | SHA256 commands in Exact Reproduction Commands |
| measurement regime | device cuda:0, seed/tolerances defaults, `warmup=50`, `repeat=100`, forward profile `20/50` | Commands used frozen arguments byte-for-byte | pass | round_status_000.md |

The correctness command's `v0=1.084037 ms` and `v1=1.077019 ms` values are smoke timing only and do not replace the frozen 50/100 baseline samples.

## Screening Evidence

Not applicable in Phase 0. No screening classification was made.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate` (harness times v0 then v1 per invocation)
- independent invocations: `3`
- reference_raw_samples_ms: `[1.070444, 1.073250, 1.078100]`
- candidate_raw_samples_ms: `[1.070630, 1.072018, 1.076946]`
- reference_median_ms: `1.073250`
- candidate_median_ms: `1.072018`
- improvement_pct: `not-applicable: Phase 0` (baseline establishment, not an adoption decision)

| Independent invocation | Reference wall ms | Candidate wall ms | Command return code |
|---:|---:|---:|---:|
| 1 | `1.070444` | `1.070630` | `0` |
| 2 | `1.073250` | `1.072018` | `0` |
| 3 | `1.078100` | `1.076946` | `0` |

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

No decision or `mechanism_observables[]` exists for Phase 0, so there are no missing required observables.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `summary`
- profiler_device_time: `available` (`cat=kernel` durations)
- profile_mode: `forward`
- warmup: `20`
- iterations: `50` forward calls per scope
- scopes: `baseline_base`, `candidate_baseline_adapter`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- raw trace: `kernels/track1-triton/centre_random_augmentation/bi150/log/round_000_forward_50iter.pt.trace.json`, SHA256 `397ecd670561b94933fc2cde22561992078b324c19252cc473c2863835cd8739`

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_base` | `21034.212` | `420.684` | `3940` | `78.8` | `1.073250` | `0.39199` |
| `candidate_baseline_adapter` | `21178.657` | `423.573` | `3951` | `79.02` | `1.072018` | `0.39511` |

```text
device_ratio = device_us_per_call / (wall_median_ms * 1000) = 420.684 / 1073.250 ≈ 0.392
```

Both scopes share an identical kernel sequence and near-identical per-call
totals. The tiny numeric difference between the two scopes (420.684 vs 423.573
us/call; 78.8 vs 79.02 kernels/call) is a measurement artifact of interleaved
scope-boundary sampling, not a semantic difference.

### Baseline Top Kernels (baseline_base scope)

| Kernel (semantic label) | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| elementwise mul (binary, `MulFunctor`) | `545` | `10.90` | `3993.11` | `79.86` |
| elementwise add (`CUDAFunctor_add`, 3-ptr) | `747` | `14.94` | `3115.74` | `62.31` |
| elementwise mul (unary, `AUnaryFunctor<MulFunctor>`) | `700` | `14.00` | `2836.62` | `56.73` |
| elementwise mul (binary, `MulFunctor` variant) | `650` | `13.00` | `2703.47` | `54.07` |
| reduce sum (`sum_functor`) | `100` | `2.00` | `1882.03` | `37.64` |
| elementwise add (other, `CUDAFunctorOnOther_add`) | `250` | `5.00` | `1020.55` | `20.41` |
| sqrt (`sqrt_kernel_cuda`) | `200` | `4.00` | `818.32` | `16.37` |
| rand (uniform) distribution kernel | `150` | `3.00` | `779.00` | `15.58` |
| sin (`sin_kernel_cuda`) | `100` | `2.00` | `580.05` | `11.60` |
| cos (`cos_kernel_cuda`) | `100` | `2.00` | `578.41` | `11.57` |
| cat (stack last-dim contiguous) | `49` | `0.98` | `544.72` | `10.89` |
| cat (batched copy) | `50` | `1.00` | `479.52` | `9.59` |
| elementwise add (`CUDAFunctor_add`) | `49` | `0.98` | `458.20` | `9.16` |
| copy (contiguous/direct) | `50` | `1.00` | `362.82` | `7.26` |
| randn (normal) distribution kernel | `50` | `1.00` | `269.97` | `5.40` |
| elementwise add (`CUDAFunctor_add`) | `50` | `1.00` | `211.90` | `4.24` |
| elementwise add (self, `CUDAFunctorOnSelf_add`) | `50` | `1.00` | `204.78` | `4.10` |
| elementwise div (`DivFunctor`) | `50` | `1.00` | `195.02` | `3.90` |

### Host-Bound Observation

`device_ratio ≈ 0.392` for the baseline scope: only ~39% of wall time is spent in
device kernel execution, with ~61% being host-side launch/overhead. This is the
expected signature of an operator whose output is only `[4,256,3] = 3072`
elements but which launches ~79 kernels per forward call. The dominant cost is
launch-count overhead, not device compute.

The kernel mix reflects the reference dataflow exactly:
- **Random draws**: 3×`torch.rand` (uniform) + 1×`torch.randn` (normal) per
  forward → 3 uniform distribution kernels (150 over 50 calls) + 1 normal
  distribution kernel (50 over 50 calls).
- **Quaternion → rotation matrix**: `sqrt`/`sin`/`cos` transcendental
  elementwise kernels (200/100/100 counts), the `torch.stack`/`cat` kernels
  (49 + 50), and numerous `mul`/`add` elementwise kernels for the
  `1-2*(yy+zz)` etc. matrix entries.
- **Centering**: one reduce-sum (the masked `sum(dim=-2)`) + one elementwise div
  (the `/ (m.sum + eps)`) + a subtract; the `.contiguous()`/`.expand()` produce
  the copy kernels.
- **rot_vec_mul + translation**: the `AUnaryFunctor<MulFunctor>` and `MulFunctor`
  elementwise kernels plus `CUDAFunctor_add`/`OnOther_add` additions for the
  3×3-by-3 vector product and `+ T[:, None, :]`.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | Initial correctness, timing, and profiler verification | `012754740961f6ec10d515563e51cd07eeaf35caefe33731d5c1e9a88387fe9b` | same | correctness and wall timing passed; profiler summarized without filtering |

No candidate repair occurred and no source file changed.

## evidence_for_next_round

- Canonical Phase 0 baseline: `baseline_adapter.py`, wall median `1.073250 ms` (reference) / `1.072018 ms` (candidate) from three independent 50/100 samples under measurement fingerprint `a5f980780c4dcde731df913710ad9dfded4f07a66b90e334fea0a6f2aa1fd5fa`.
- `baseline_base` scope measured `420.684 us/device-call` and `78.8 kernels/call`. Device ratio ≈ `0.392`, so ~39% of wall time is device kernel time and ~61% is host/launch overhead. This operator is **host-bound** at baseline, driven by an extreme launch count relative to the tiny `[4,256,3]` data size.
- Device time is dominated by many tiny elementwise/reduce kernels: ~79 kernels per forward, mostly `mul`/`add` elementwise ops (the quaternion→rotation-matrix construction and the 3×3-by-3 vector product), transcendental `sqrt`/`sin`/`cos`, and the random-number distribution kernels (3 uniform + 1 normal).
- The output is only `[4,256,3]` (3072 elements) yet the forward launches ~79 kernels; there is substantial opportunity to fuse the random-rotation construction, `rot_vec_mul`, centering, and translation into far fewer kernels.
- Base and adapter are semantically equivalent (adapter is a top-level class rename); the small wall/device differences are measurement observations, not an optimization mechanism.

## Stop Recommendation

- recommendation: `continue`
- evidence: Phase 0 baseline is valid (correctness PASS, three 50/100 wall samples, Level 1 profiler summary collected). No optional target is configured, and no terminal-round limit applies to baseline establishment.

Orchestrator owns canonical pointer updates and workflow transition.

## Exact Reproduction Commands

Environment bootstrap (every command):

```bash
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
```

Frozen-file SHA256 verification (run before and after measurement; all returned code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && sha256sum kernels/track1-triton/centre_random_augmentation/base.py kernels/track1-triton/centre_random_augmentation/bi150/baseline_adapter.py auto_bench.py
```

Correctness (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/centre_random_augmentation/base.py --v1_file kernels/track1-triton/centre_random_augmentation/bi150/baseline_adapter.py --warmup 50 --repeat 100 --full-traceback
```

Wall timing (execute independently three times; return codes `0, 0, 0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/centre_random_augmentation/base.py --v1_file kernels/track1-triton/centre_random_augmentation/bi150/baseline_adapter.py --warmup 50 --repeat 100
```

Forward profiler (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/centre_random_augmentation/base.py --v1_file kernels/track1-triton/centre_random_augmentation/bi150/baseline_adapter.py --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/centre_random_augmentation/bi150/log/round_000_forward_50iter.pt.trace.json
```

Separately scoped unmodified repository summaries (both returned code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/centre_random_augmentation/bi150/log/round_000_forward_50iter.pt.trace.json --iterations 50 --scope baseline_base --wall-ms 1.073250
```

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/centre_random_augmentation/bi150/log/round_000_forward_50iter.pt.trace.json --iterations 50 --scope candidate_baseline_adapter --wall-ms 1.072018
```

## Command Status and Raw Evidence Index

| Command | Return code | Evidence |
|---|---:|---|
| frozen-file SHA256 before measurement | `0` | hashes in Identity |
| runtime fingerprint check | `0` | torch 2.7.1, triton 3.1.0, BI-V150 (7,1) |
| correctness 50/100 | `0` | round_status_000.md; report Correctness table |
| wall sample 1, 50/100 | `0` | report Interleaved Wall Timing |
| wall sample 2, 50/100 | `0` | report Interleaved Wall Timing |
| wall sample 3, 50/100 | `0` | report Interleaved Wall Timing |
| forward profiler 20/50 | `0` | `log/round_000_forward_50iter.pt.trace.json` |
| summarize `baseline_base` | `0` | report Profiler Evidence |
| summarize `candidate_baseline_adapter` | `0` | report Profiler Evidence |

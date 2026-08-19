# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `base.py` (Phase 0 source reference; no pre-existing canonical implementation)
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `98cf1e1e5f493ae4ae08c9391f00db83b3cd783a1b194d9c0d63851091133a5d`
- Accepted reference SHA256: `28d4d213638e5adc7f6bc52928c8f7596cd8446e3e0201beab364b5c3c7988fc`
- Base SHA256: `28d4d213638e5adc7f6bc52928c8f7596cd8446e3e0201beab364b5c3c7988fc`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint` (torch 2.7.1, triton 3.1.0, Iluvatar BI-V150, capability (7,1))
- Measurement fingerprint: `a03823074048c8cb5e8199b593c8c19aa3b259180969321015e5a1679461b71a`
- verification_tier: `baseline`
- screening_pairs: `not-run: Phase 0`

The adapter, base, and harness hashes all match the frozen project.md values exactly.

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | Harness recursive comparison at seed 42, `atol=1e-2`, `rtol=1e-2`, equal-NaN enabled | `PASS accuracy; v0=0.351913 ms, v1=0.352298 ms, speedup=0.999x`; return code `0` | pass | correctness command in Exact Reproduction Commands |
| public loader contract | `base.py` exposes `Model/get_init_inputs/get_inputs`; adapter exposes `ModelNew/get_init_inputs/get_inputs` through the actual AST loader | Frozen harness loaded, constructed, moved, and executed both sides without load or constructor error | pass | correctness return code `0` |
| output structure/shape/dtype | 3-tuple of float32 tensors `grad_input_mix[2,1024,4]`, `grad_mhc_scale[1]`, `grad_mhc_base[4]` | Harness recursive comparator accepted structure, shape, and dtype | pass | correctness return code `0`; `project.md#semantics` |
| floating values | `torch.allclose(..., atol=1e-2, rtol=1e-2, equal_nan=True)` | No floating mismatch | pass | correctness return code `0` |
| frozen artifact identity | local hashes equal project.md before measurement | base `28d4d213...`, adapter `98cf1e1e...`, harness `3d4fa4ee...` all match | pass | SHA256 commands in Exact Reproduction Commands |
| measurement regime | device cuda:0, seed/tolerances defaults, `warmup=50`, `repeat=100`, forward profile `20/50` | Commands used frozen arguments byte-for-byte | pass | round_status_000.md |

The correctness command's `v0=0.351913 ms` and `v1=0.352298 ms` values are smoke timing only and do not replace the frozen 50/100 baseline samples.

## Screening Evidence

Not applicable in Phase 0. No screening classification was made.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate` (harness times v0 then v1 per invocation)
- independent invocations: `3`
- reference_raw_samples_ms: `[0.352471, 0.351449, 0.348174]`
- candidate_raw_samples_ms: `[0.352724, 0.351174, 0.348107]`
- reference_median_ms: `0.351449`
- candidate_median_ms: `0.351174`
- improvement_pct: `not-applicable: Phase 0` (baseline establishment, not an adoption decision)

| Independent invocation | Reference wall ms | Candidate wall ms | Command return code |
|---:|---:|---:|---:|
| 1 | `0.352471` | `0.352724` | `0` |
| 2 | `0.351449` | `0.351174` | `0` |
| 3 | `0.348174` | `0.348107` | `0` |

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
- raw trace: `log/round_000_forward_50iter.pt.trace.json`, SHA256 `6b0b555f903e0f61fc23ba90bf14cb9f64fa855d95f55c5e837e58217f54cb97`

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_base` | `9279.9736328125` | `185.59947265625` | `487` | `9.74` | `0.351449` | `0.5280978823563305` |
| `candidate_baseline_adapter` | `9584.8974609375` | `191.69794921875` | `491` | `9.82` | `0.351449` | `0.545450262253556` |

```text
device_ratio = device_us_per_call / (wall_median_ms * 1000) = 185.599 / 351.449 ≈ 0.528
```

Both scopes share an essentially identical kernel sequence (6 distinct kernel
templates). The small per-call numeric difference between the two scopes
(185.599 vs 191.698 us/call, 9.74 vs 9.82 kernels/call) is a measurement
artifact of interleaved scope-boundary sampling, not a semantic difference.

### Baseline Top Kernels (baseline_base scope)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `reduce_kernel<1024,1,ReduceOp<float,sum_functor...>>` | `96` | `1.92` | `7398.848` | `147.977` |
| `modern::elementwise_kernel<BinaryFunctor<float,MulFunctor<float>>>` | `146` | `2.92` | `706.193` | `14.124` |
| `elementwise_kernel<512,4,BinaryFunctor<float,MulFunctor<float>>>` | `98` | `1.96` | `459.241` | `9.185` |
| `modern::elementwise_kernel<sigmoid_kernel_cuda>` | `49` | `0.98` | `260.252` | `5.205` |
| `elementwise_kernel<512,4,CUDAFunctor_add<float>>` (lambda#2) | `49` | `0.98` | `240.207` | `4.804` |
| `modern::elementwise_kernel<CUDAFunctorOnOther_add<float>>` | `49` | `0.98` | `215.233` | `4.305` |

### Candidate Top Kernels (candidate_baseline_adapter scope)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `reduce_kernel<1024,1,ReduceOp<float,sum_functor...>>` | `99` | `1.98` | `7696.228` | `153.925` |
| `modern::elementwise_kernel<BinaryFunctor<float,MulFunctor<float>>>` | `147` | `2.94` | `713.383` | `14.268` |
| `elementwise_kernel<512,4,BinaryFunctor<float,MulFunctor<float>>>` | `98` | `1.96` | `459.059` | `9.181` |
| `modern::elementwise_kernel<sigmoid_kernel_cuda>` | `49` | `0.98` | `259.837` | `5.197` |
| `elementwise_kernel<512,4,CUDAFunctor_add<float>>` (lambda#2) | `49` | `0.98` | `241.220` | `4.824` |
| `modern::elementwise_kernel<CUDAFunctorOnOther_add<float>>` | `49` | `0.98` | `215.171` | `4.303` |

### Kernel Sequence Observation

The sigmoid-backward affine modulation is visible in the top-kernel breakdown.
Per forward call (~9.74 kernels):

- `sum_functor` reduce kernel: ~1.92 per call (2 reduction ops expected: one
  `sum(dim=(0,1))` for `grad_mhc_base`, one `sum(dim=(0,1,2))` for
  `grad_mhc_scale`). This single reduce kernel dominates device time at
  `147.977 us/call` out of `185.599 us/call` total — roughly 80% of device time.
- `MulFunctor` elementwise kernels (two variants): ~2.92 + ~1.96 ≈ 4.88 per call.
  These correspond to the multiplication chain: `z = input_mix * mhc_scale`,
  `grad_out * sigmoid`, `*(1 - sigmoid)`, `grad_z * mhc_scale`, and
  `grad_z * input_mix` (for the scale gradient) — a small set of elementwise
  multiplies over the `[2,1024,4]` = 8192-element tensor.
- `sigmoid_kernel_cuda`: ~0.98 per call (the single `torch.sigmoid(z)`).
- `CUDAFunctor_add<float>` (two variants): ~0.98 + ~0.98 ≈ 1.96 per call,
  corresponding to `z + mhc_base` broadcast-add and the `1 - sigmoid` subtract.

Overall the operator is extremely light: only ~10 kernels per forward call over
an 8192-element tensor. Device time is dominated by the two `sum` reductions
(one full 3-D reduce and one 2-D reduce), which together account for ~80% of
device time, while the elementwise multiply/add/sigmoid chain is individually
tiny but numerous.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | Initial correctness, timing, and profiler verification | `98cf1e1e5f493ae4ae08c9391f00db83b3cd783a1b194d9c0d63851091133a5d` | same | correctness and wall timing passed; profiler summarized without filtering |

No candidate repair occurred and no source file changed.

## evidence_for_next_round

- Canonical Phase 0 baseline: `baseline_adapter.py`, wall median `0.351449 ms` from three independent 50/100 samples under measurement fingerprint `a03823074048c8cb5e8199b593c8c19aa3b259180969321015e5a1679461b71a`.
- `baseline_base` scope measured `185.59947265625 us/device-call` and `9.74 kernels/call`. Device ratio ≈ `0.528`, so roughly 53% of wall time is device kernel time and ~47% is host/launch overhead. This operator is partially host-bound at baseline.
- Device time is dominated by the two `sum` reductions (~1.92 per call, ~147.98 us/call, ~80% of device time): `grad_mhc_base = grad_z.sum(dim=(0,1))` and `grad_mhc_scale = (grad_z * input_mix).sum(dim=(0,1,2))`.
- The elementwise chain (mul/add/sigmoid) is individually tiny but contributes ~10 distinct kernel launches per forward call over the `[2,1024,4]` = 8192-element tensor. There is opportunity to fuse the sigmoid + elementwise + reduction into far fewer kernels, and to avoid materializing intermediate tensors (`z`, `sigmoid`, `grad_z`).
- Base and adapter are semantically equivalent (adapter is a top-level class rename plus an explicit `return (...)` tuple); the small wall/device differences are measurement observations, not an optimization mechanism.

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
cd /root/CodeBuddy/20260818191200/kernelswift && sha256sum kernels/track1-triton/mhc_head_compute_mix_backward/base.py kernels/track1-triton/mhc_head_compute_mix_backward/bi150/baseline_adapter.py auto_bench.py
```

Correctness (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix_backward/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix_backward/bi150/baseline_adapter.py --warmup 50 --repeat 100 --full-traceback
```

Wall timing (execute independently three times; return codes `0, 0, 0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix_backward/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix_backward/bi150/baseline_adapter.py --warmup 50 --repeat 100
```

Forward profiler (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix_backward/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix_backward/bi150/baseline_adapter.py --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/mhc_head_compute_mix_backward/bi150/log/round_000_forward_50iter.pt.trace.json
```

Separately scoped unmodified repository summaries (both returned code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/mhc_head_compute_mix_backward/bi150/log/round_000_forward_50iter.pt.trace.json --iterations 50 --scope baseline_base --wall-ms 0.351449
```

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/mhc_head_compute_mix_backward/bi150/log/round_000_forward_50iter.pt.trace.json --iterations 50 --scope candidate_baseline_adapter --wall-ms 0.351449
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

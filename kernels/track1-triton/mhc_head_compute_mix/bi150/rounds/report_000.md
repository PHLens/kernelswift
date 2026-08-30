# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `base.py` (Phase 0 source reference; no pre-existing canonical implementation)
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `ceebdc6185de4c980156a7833073678a0964fb7ccb5edf74b42be6156652eaed`
- Accepted reference SHA256: `4c5167f6cfe9099786f204fad222f3b780d3ebb814494b9582112422abf84ac5`
- Base SHA256: `4c5167f6cfe9099786f204fad222f3b780d3ebb814494b9582112422abf84ac5`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint` (torch 2.7.1, triton 3.1.0, Iluvatar BI-V150, capability (7,1))
- Measurement fingerprint: `4e4f0575e2251810e9be7667c98eb4923c6910787d4412abc2c4a976c2b26a8e`
- verification_tier: `baseline`
- screening_pairs: `not-run: Phase 0`

The adapter, base, and harness hashes all match the frozen project.md values exactly.

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | Harness recursive comparison at seed 42, `atol=1e-2`, `rtol=1e-2`, equal-NaN enabled | `PASS accuracy; v0=1.519272 ms, v1=1.515536 ms, speedup=1.002x`; return code `0` | pass | correctness command in Exact Reproduction Commands |
| public loader contract | `base.py` exposes `Model/get_init_inputs/get_inputs`; adapter exposes `ModelNew/get_init_inputs/get_inputs` through the actual AST loader | Frozen harness loaded, constructed, moved, and executed both sides without load or constructor error | pass | correctness return code `0` |
| output structure/shape/dtype | 3-tuple of float32 tensors `pre[2,8,4]`, `post[2,8,4]`, `comb[2,8,4,4]` | Harness recursive comparator accepted structure, shape, and dtype | pass | correctness return code `0`; `project.md#semantics` |
| floating values | `torch.allclose(..., atol=1e-2, rtol=1e-2, equal_nan=True)` | No floating mismatch | pass | correctness return code `0` |
| frozen artifact identity | local hashes equal project.md before measurement | base `4c5167f6...`, adapter `ceebdc61...`, harness `3d4fa4ee...` all match | pass | SHA256 commands in Exact Reproduction Commands |
| measurement regime | device cuda:0, seed/tolerances defaults, `warmup=50`, `repeat=100`, forward profile `20/50` | Commands used frozen arguments byte-for-byte | pass | round_status_000.md |

The correctness command's `v0=1.519272 ms` and `v1=1.515536 ms` values are smoke timing only and do not replace the frozen 50/100 baseline samples.

## Screening Evidence

Not applicable in Phase 0. No screening classification was made.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate` (harness times v0 then v1 per invocation)
- independent invocations: `3`
- reference_raw_samples_ms: `[1.517299, 1.515616, 1.523859]`
- candidate_raw_samples_ms: `[1.518374, 1.519256, 1.533459]`
- reference_median_ms: `1.517299`
- candidate_median_ms: `1.518374`
- improvement_pct: `not-applicable: Phase 0` (baseline establishment, not an adoption decision)

| Independent invocation | Reference wall ms | Candidate wall ms | Command return code |
|---:|---:|---:|---:|
| 1 | `1.517299` | `1.518374` | `0` |
| 2 | `1.515616` | `1.519256` | `0` |
| 3 | `1.523859` | `1.533459` | `0` |

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
- raw trace: `log/round_000_forward_50iter.pt.trace.json`, SHA256 `ca2fa8c940184de974e6dc326c4a4cd7f0a0e9322826395aa28ab42ef825b083`

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_base` | `46319.767578125` | `926.3953515625` | `6644` | `132.88` | `1.517299` | `0.6105555672036296` |
| `candidate_baseline_adapter` | `46318.45458984375` | `926.369091796875` | `6644` | `132.88` | `1.517299` | `0.6105382602881008` |

```text
device_ratio = device_us_per_call / (wall_median_ms * 1000) = 926.395 / 1517.299 ≈ 0.6106
```

Both scopes share an identical kernel sequence and near-identical per-call
totals. The tiny numeric difference between the two scopes (926.395 vs 926.369
us/call) is a measurement artifact of interleaved scope-boundary sampling, not a
semantic difference.

### Baseline Top Kernels (baseline_base scope)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `reduce_kernel<1024,1,ReduceOp<float,sum_functor...>>` | `1998` | `39.96` | `22125.758` | `442.515` |
| `elementwise_kernel<512,4,BinaryFunctor<float,DivFunctor<float>>>` | `1998` | `39.96` | `12711.801` | `254.236` |
| `modern::elementwise_kernel<CUDAFunctorOnSelf_add<float>>` | `2048` | `40.96` | `8070.366` | `161.407` |
| `elementwise_kernel<512,4,BinaryFunctor<float,MulFunctor<float>>>` | `150` | `3.0` | `999.736` | `19.995` |
| `reduce_kernel<1024,1,ReduceOp<float,MaxNanFunctor<float>>>` | `50` | `1.0` | `623.524` | `12.470` |
| `elementwise_kernel<512,4,CUDAFunctor_add<float>>` (lambda#2) | `150` | `3.0` | `594.072` | `11.881` |
| `modern::elementwise_kernel<sigmoid_kernel_cuda>` | `100` | `2.0` | `427.417` | `8.548` |
| `elementwise_kernel<512,4,CUDAFunctor_add<float>>` (lambda#5) | `50` | `1.0` | `346.707` | `6.934` |
| `modern::elementwise_kernel<exp_kernel_cuda>` | `50` | `1.0` | `217.217` | `4.344` |
| `modern::elementwise_kernel<AUnaryFunctor<float,MulFunctor<float>>>` | `50` | `1.0` | `203.170` | `4.063` |

### Candidate Top Kernels (candidate_baseline_adapter scope)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `reduce_kernel<1024,1,ReduceOp<float,sum_functor...>>` | `1998` | `39.96` | `22128.544` | `442.571` |
| `elementwise_kernel<512,4,BinaryFunctor<float,DivFunctor<float>>>` | `1998` | `39.96` | `12706.307` | `254.126` |
| `modern::elementwise_kernel<CUDAFunctorOnSelf_add<float>>` | `2048` | `40.96` | `8067.352` | `161.347` |
| `elementwise_kernel<512,4,BinaryFunctor<float,MulFunctor<float>>>` | `150` | `3.0` | `999.518` | `19.990` |
| `reduce_kernel<1024,1,ReduceOp<float,MaxNanFunctor<float>>>` | `50` | `1.0` | `626.360` | `12.527` |
| `elementwise_kernel<512,4,CUDAFunctor_add<float>>` (lambda#2) | `150` | `3.0` | `593.697` | `11.874` |
| `modern::elementwise_kernel<sigmoid_kernel_cuda>` | `100` | `2.0` | `427.458` | `8.549` |
| `elementwise_kernel<512,4,CUDAFunctor_add<float>>` (lambda#5) | `50` | `1.0` | `349.672` | `6.993` |
| `modern::elementwise_kernel<exp_kernel_cuda>` | `50` | `1.0` | `216.730` | `4.335` |
| `modern::elementwise_kernel<AUnaryFunctor<float,MulFunctor<float>>>` | `50` | `1.0` | `202.817` | `4.056` |

### Sinkhorn Kernel Sequence Observation

The Sinkhorn iteration (20 rounds of alternating row/column normalization) is
clearly visible in the top-kernel breakdown:

- `sum_functor` reduce kernel: `1998` total over 50 calls = `39.96` per call.
  Each forward performs the Sinkhorn loop with 20 row-normalizations + 20
  column-normalizations, each requiring one `sum(dim=-1)` and one `sum(dim=-2)`.
  That is 40 sum reductions per forward (plus one `amax` reduction and the
  pre-loop row/column sums). 39.96/call matches the ~40 expected sum reductions.
- `DivFunctor` elementwise kernel: `1998` total = `39.96` per call. Each of the
  40 normalizations is a division of `comb` by the (sum + eps) denominator, so
  ~40 divisions per forward. 39.96/call matches.
- `CUDAFunctorOnSelf_add<float>` elementwise kernel: `2048` total = `40.96` per
  call. These are the `+ eps` floor additions applied to each normalized matrix
  and denominator (40 in-loop add-eps operations, plus the explicit pre-loop
  add-eps), matching ~41 per forward.

The remaining one-off kernels per forward are the non-iterated head-compute
stage: 2 sigmoid kernels (pre and post), 1 exp kernel (softmax), 1 amax reduction
(row_max), and the mul/add affine elementwise kernels (s0/s1/s2 scaling and base
offsets).

Overall the device time is dominated by a very large number of tiny
elementwise/reduce kernels (132.88 kernels per forward call), each operating on
the extremely small tensors (`mixes [2,8,24]`, `comb [16,4,4]`). The Sinkhorn
loop alone contributes roughly 40 sum + 40 div + 40 add ≈ 120 of these ~133
kernels.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | Initial correctness, timing, and profiler verification | `ceebdc6185de4c980156a7833073678a0964fb7ccb5edf74b42be6156652eaed` | same | correctness and wall timing passed; profiler summarized without filtering |

No candidate repair occurred and no source file changed.

## evidence_for_next_round

- Canonical Phase 0 baseline: `baseline_adapter.py`, wall median `1.517299 ms` from three independent 50/100 samples under measurement fingerprint `4e4f0575e2251810e9be7667c98eb4923c6910787d4412abc2c4a976c2b26a8e`.
- `baseline_base` scope measured `926.3953515625 us/device-call` and `132.88 kernels/call`. Device ratio ≈ `0.611`, so roughly 61% of wall time is device kernel time and ~39% is host/launch overhead. This operator is NOT purely host-bound at baseline.
- Device time is dominated by a very large count of tiny elementwise/reduce kernels: ~40 sum reductions + ~40 division elementwise + ~41 add-eps elementwise per forward call, driven by the 20-round Sinkhorn iteration over `[16,4,4]` tensors.
- The `comb` tensors are `[16,4,4]` (256 elements per matrix); each Sinkhorn normalization launches a full separate CUDA kernel for a few hundred elements, producing extreme launch-count overhead relative to the tiny data size. There is substantial opportunity to fuse the Sinkhorn iteration (and the head-compute elementwise ops) into far fewer kernels.
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
cd /root/CodeBuddy/20260818191200/kernelswift && sha256sum kernels/track1-triton/mhc_head_compute_mix/base.py kernels/track1-triton/mhc_head_compute_mix/bi150/baseline_adapter.py auto_bench.py
```

Correctness (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix/bi150/baseline_adapter.py --warmup 50 --repeat 100 --full-traceback
```

Wall timing (execute independently three times; return codes `0, 0, 0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix/bi150/baseline_adapter.py --warmup 50 --repeat 100
```

Forward profiler (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix/bi150/baseline_adapter.py --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/mhc_head_compute_mix/bi150/log/round_000_forward_50iter.pt.trace.json
```

Separately scoped unmodified repository summaries (both returned code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/mhc_head_compute_mix/bi150/log/round_000_forward_50iter.pt.trace.json --iterations 50 --scope baseline_base --wall-ms 1.517299
```

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/mhc_head_compute_mix/bi150/log/round_000_forward_50iter.pt.trace.json --iterations 50 --scope candidate_baseline_adapter --wall-ms 1.517299
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

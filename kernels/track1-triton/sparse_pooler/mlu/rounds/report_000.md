# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `base.py`
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `d7e69ed4b66a4193a4475cc307fc0929cef807f875785652df6cc36fb2c487e5`
- Accepted reference SHA256: `ccccbbefadf1d697341451b542f17392acc8a2b9e4a3a41e50b2f9d58dbf61de`
- Base SHA256: `ccccbbefadf1d697341451b542f17392acc8a2b9e4a3a41e50b2f9d58dbf61de`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `a0208c7da7e371d45c88f82ebddd3850d01669aa5d912f31db9234a7a56ebab7`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | `torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True)` between `base.py` Model and `baseline_adapter.py` ModelNew outputs (list of 4 × Tensor[30522]) | `PASS accuracy` across 3 benchmark runs and 1 profiler run | pass | `python3 auto_bench.py --v0_file sparse_pooler/base.py --v1_file sparse_pooler/baseline_adapter.py --warmup 50 --repeat 100` |
| output structure | list of 4 tensors, each shape [30522], dtype fp32, device mlu:0 | confirmed via direct execution | pass | `out.shape=torch.Size([30522]) dtype=torch.float32 device=mlu:0` × 4 |

Conformance, correctness, and every declared guardrail must pass before adoption.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate`
- reference_raw_samples_ms: `[0.898116, 0.905639, 0.911165]` (base.py Model)
- candidate_raw_samples_ms: `[0.944545, 0.901030, 0.909974]` (baseline_adapter.py ModelNew)
- reference_median_ms: `0.905639`
- candidate_median_ms: `0.909974`
- improvement_pct: `-0.48` (baseline_adapter is ~0.5% slower than base; expected since they are semantically identical and the difference is noise)

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
                = (0.905639 - 0.909974) / 0.905639 * 100
                = -0.48
```

The unrounded improvement controls the 5% adoption threshold. Profiler time does
not replace this benchmark result.

Note: In Phase 0, the "candidate" is `baseline_adapter.py` (a byte-identical
rename of `base.py`'s `Model` to `ModelNew`). The two are semantically the same
operator, so the wall times are within noise. The Phase 0 baseline median used
for future candidate comparison is the **baseline_adapter** median of
`0.909974 ms` (the accepted canonical implementation going forward).

## Evaluation Contract Mirror

- Evaluation Contract applicability: `not-applicable: Phase 0`
- hypothesis_id: `not-applicable: Phase 0`
- intervention: `not-applicable: Phase 0`
- expected_causal_chain: `not-applicable: Phase 0`
- primary_metric: `not-applicable: Phase 0`
- Hypothesis verdict: `not-applicable: Phase 0`

## Profiler Evidence

- profiler_level: `summary`
- iterations: `50` (forward calls per scope)
- normalized_fields: `device_total_us`, `device_us_per_call`,
  `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`

Reference and candidate scopes are collected and summarized independently. All
totals below are normalized by `iterations` before they are compared. The
profiler trace was produced with
`--profile --profile-reference-file sparse_pooler/baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 50`.
Scopes `reference_baseline_adapter` and `candidate_baseline_adapter` were
summarized using their `gpu_user_annotation` device-side intervals.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference | 9002.5 | 180.05 | 500 | 10.0 | 0.909974 | 0.1979 |
| candidate | 9037.3 | 180.75 | 500 | 10.0 | 0.909974 | 0.1986 |

```text
device_ratio = device_us_per_call / (candidate_median_ms * 1000)
             = 180.05 / (0.909974 * 1000)
             = 0.1979
```

### Accepted Reference Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| MLUFusedMatMulGepm (decoder matmul 768→30522) | 50 | 1.00 | 4471.0 | 89.42 |
| reduceKernelMaxIndex (per-sequence max pool) | 200 | 4.00 | 1481.0 | 29.62 |
| MLUBlockKernel5StagePipelineLog1pFast (log1p) | 50 | 1.00 | 1308.4 | 26.17 |
| MLUBlockKernel3StagePipelineClipFast (relu) | 50 | 1.00 | 682.7 | 13.65 |
| MLUFusedMatMulGepdot (dense matmul 768→768) | 50 | 1.00 | 423.0 | 8.46 |
| layerNormForwardKernel (LayerNorm) | 50 | 1.00 | 359.4 | 7.19 |
| MLUBlockKernel3StagePipelineGeluHighAccCubic (GELU) | 50 | 1.00 | 277.0 | 5.54 |

### Candidate Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| MLUFusedMatMulGepm (decoder matmul 768→30522) | 50 | 1.00 | 4510.4 | 90.21 |
| reduceKernelMaxIndex (per-sequence max pool) | 200 | 4.00 | 1477.3 | 29.55 |
| MLUBlockKernel5StagePipelineLog1pFast (log1p) | 50 | 1.00 | 1308.3 | 26.17 |
| MLUBlockKernel3StagePipelineClipFast (relu) | 50 | 1.00 | 681.5 | 13.63 |
| MLUFusedMatMulGepdot (dense matmul 768→768) | 50 | 1.00 | 422.4 | 8.45 |
| layerNormForwardKernel (LayerNorm) | 50 | 1.00 | 360.2 | 7.20 |
| MLUBlockKernel3StagePipelineGeluHighAccCubic (GELU) | 50 | 1.00 | 277.2 | 5.54 |

Reference and candidate kernel breakdowns are within noise (they are the same
operator). The baseline profile is the diagnostic baseline for future rounds.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | Phase 0 baseline verification | not-applicable | d7e69ed4b66a4193a4475cc307fc0929cef807f875785652df6cc36fb2c487e5 | pass |

At most one Verifier-to-Coder repair is allowed in the same round.

## Upbound Gap

- upbound_source: `project.md#upbound (estimated semantic bound)`
- comparable_metric: `wall_time_ms`
- absolute_gap: `null (no measured upbound)`
- ratio_to_upbound: `null`
- interpretation: `The declared upbound is a semantic estimate, not a measured bound. The 5% adoption threshold uses measured wall time only.`

## evidence_for_next_round

- Device ratio is ~0.20 (baseline_adapter median 0.909974 ms, device 180.05 us/call). This sits at the host-bound/mixed boundary per `references/bottleneck-judgment.md`. Roughly 730 us/call (~80%) is host-side: Python loop, per-launch overhead, and the 4 sequential max-pool dispatches.
- 10 device kernels per forward call. The decoder matmul (MLUFusedMatMulGepm) dominates device time at 89.42 us/call (49.6% of device work).
- The 4 per-sequence max-pool kernels (reduceKernelMaxIndex) total 118.5 us/call device time (4 × 29.62) and are launched from a Python `for` loop over `seq_lens.tolist()`. This loop is a host-side bottleneck and a fusion target.
- log1p + relu together cost 39.82 us/call device time and could be fused with the preceding decoder matmul or into a single elementwise kernel.
- A single fused Triton kernel covering the MLM head tail (relu+log1p) and the per-sequence max pooling would eliminate 5 of the 10 device kernels and the Python loop, potentially reducing both device and host time.

Record evidence only; do not select the next optimization.

## Stop Recommendation

- recommendation: `continue`
- evidence: `Phase 0 baseline established. Device ratio 0.20 indicates substantial host-side and Python-loop overhead. Multiple fusion and launcher-reduction opportunities exist that are expected to clear the 5% adoption threshold.`

Orchestrator owns the stop transition.

## Exact Reproduction Commands

```bash
python3 /projs/framework/lipenghui/projects/kernelswift/auto_bench.py \
  --v0_file sparse_pooler/base.py \
  --v1_file sparse_pooler/baseline_adapter.py \
  --warmup 50 --repeat 100
```

```bash
for i in 1 2 3; do
  python3 /projs/framework/lipenghui/projects/kernelswift/auto_bench.py \
    --v0_file sparse_pooler/base.py \
    --v1_file sparse_pooler/baseline_adapter.py \
    --warmup 50 --repeat 100
done
```

```bash
python3 /projs/framework/lipenghui/projects/kernelswift/auto_bench.py \
  --v0_file sparse_pooler/base.py \
  --v1_file sparse_pooler/baseline_adapter.py \
  --profile --profile-reference-file sparse_pooler/baseline_adapter.py \
  --profile-mode forward --profile-warmup 20 --profile-iterations 50 \
  --profile-output sparse_pooler/log/round_000_forward_50iter.pt.trace.json
```

For `Result: baseline`, this report contains correctness, baseline wall
samples, a Level 1 profiler summary, runtime and measurement fingerprints, and
exact reproduction commands. Its Evaluation Contract mirror is
`not-applicable: Phase 0` because no round decision exists.

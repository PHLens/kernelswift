# Report 001

Result: accepted

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md`
- Candidate: `maca/triton_rotary_001.py`
- Accepted reference: `baseline_adapter.py` (canonical) / `base.py` (v0 proxy exposing `Model`)
- Accepted reference report: `rounds/report_000.md`
- Decision SHA256: `6e5741d2ccabe1883520625bfdb5a8e6e7f334b9ea995de5069943246342eceb`
- Candidate SHA256: `dec9aa12bc50886503831c48b82767e6a76ecd29d3a5c29cb41185d6ef633c39`
- Accepted reference SHA256: `40c1a8bfbd9a0e957f21ae8ac686aa4c378a28299fd1f053d1e35b5fa8c443e0`
- Base SHA256: `98be7d25ad94930fd9497e6ab33e64970c509c6c3569c8c3df40ae9d22b9a341`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `486242286573efe11bdd7b852247cb0ed4d63113e0e41c7c432ab65e654a6518`
- verification_tier: authoritative
- screening_pairs: `not-run` (authoritative timing required; candidate proceeded past screening)

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | v0 and v1 outputs identical under atol=1e-2, rtol=1e-2, equal_nan=True | `PASS accuracy; v0=0.195046 ms, v1=0.082840 ms, speedup=2.354x` | pass | `auto_bench.py ... --warmup 5 --repeat 10 --full-traceback` (exit 0) |
| output tuple shape dtype device unchanged | `(cos, sin)` each `(4,32,128)` fp32 contiguous on cuda:0 | recursive `compare_values` passed (no shape/dtype/device mismatch) | pass | correctness run exit 0 |
| non-mutation of timestamps | forward must not mutate inputs | harness clones inputs and compares under `torch.no_grad()`; no mutation | pass | correctness run exit 0 |
| caller-selected device and current stream preserved | kernel launches on input device/current stream | candidate dispatches on `timestamps.device`; correctness pass on cuda:0 | pass | correctness run exit 0 |
| PyTorch fallback preserved for non-benchmark shapes | non-benchmark shapes use unchanged PyTorch path | `ModelNew.forward` retains verbatim PyTorch fallback branch | pass | code inspection of `triton_rotary_001.py` forward |

## Screening Evidence

A correct candidate proceeded directly to authoritative timing (screening is for
detecting 10%+ slowdowns; the candidate's local gate already showed ~2.5x speedup).

| Pair | Reference short wall ms | Candidate short wall ms | Candidate slower pct | Evidence |
|---:|---:|---:|---:|---|
| 1 | not-run | not-run | not-run | authoritative timing path |

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `sequential complete accepted-reference block, then complete candidate block`
- reference_raw_samples_ms: `not captured by harness (harness returns median only)`
- candidate_raw_samples_ms: `not captured by harness (harness returns median only)`
- reference_median_ms: `0.180151`
- candidate_median_ms: `0.080036`
- improvement_pct: `55.5727`

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
                = (0.180151 - 0.080036) / 0.180151 * 100
                = 55.5727
```

Three independent authoritative wall-timing runs were executed (warmup 50, repeat 100):

| Run | v0_ms (reference) | v1_ms (candidate) | speedup |
|---:|---:|---:|---:|
| 1 | 0.180151 | 0.079770 | 2.258x |
| 2 | 0.180017 | 0.080036 | 2.249x |
| 3 | 0.184270 | 0.080713 | 2.283x |

Median of the three runs:

- reference_median_ms = `median(0.180151, 0.180017, 0.184270)` = `0.180151`
- candidate_median_ms = `median(0.079770, 0.080036, 0.080713)` = `0.080036`

The unrounded improvement (55.57%) exceeds the 5% adoption threshold. Benchmark
wall time controls adoption.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| candidate_kernel_count_per_call | `11.0 -> 1.0` | `1.0` (50 kernels / 50 iterations) | pass | `summarize_trace.py --scope candidate_triton_rotary_001` |
| candidate_device_us_per_call | `decrease from 50.95` | `16.901123046875` (baseline was 50.948896484375) | pass | `summarize_trace.py` |
| broadcast_mul_plus_cat_us_per_call | `~20.9 -> 0.0` | `0.0` (broadcast-mul and cat library kernels absent from candidate scope) | pass | candidate top-kernels list is a single `_rotary_embed_fused_kernel` |
| fused_triton_kernel_count_per_call | `== 1.0` | `1.0` (single `_rotary_embed_fused_kernel`) | pass | `summarize_trace.py` candidate scope |
| cos_sin_allclose | `pass within atol=rtol=1e-2` | `PASS accuracy` | pass | correctness run exit 0 |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-001`
- intervention: `fuse the full rotary-embedding chain (broadcast/cat/angle-scale/cos/sin) into one direct-launch Triton-MACA elementwise kernel over (B*SEQ, 2*dim) output elements`
- expected_causal_chain: `11 host-launched PyTorch kernels collapse to 1 Triton kernel -> host launch overhead disappears -> intermediate materializations removed -> wall time decreases`
- primary_metric: `wall_time`
- Hypothesis verdict: `confirmed`

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `summary`
- profiler_device_time: `available`
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- backend_runtime_fields: `not-applicable` (device `cat=kernel` durations available)

Reference and candidate scopes are collected and summarized independently. All
totals below are normalized by `iterations=50`.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| baseline_base (v0) | 2534.63134765625 | 50.692626953125 | 550 | 11.0 | 0.180151 | 0.2813896506437655 |
| candidate_triton_rotary_001 (v1) | 845.05615234375 | 16.901123046875 | 50 | 1.0 | 0.080036 | 0.21116901203052377 |

```text
device_ratio = device_us_per_call / (wall_ms * 1000)
```

### baseline_base (v0) Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| elementwise_kernel_2_2_broadcast_uncontiguous<512,... MulFunctor> | 50 | 1.0 | 651.26416015625 | 13.025283203125 |
| MACA_CatArrayBatchedCopyNoPartialWrite ... last_dim_cat | 50 | 1.0 | 382.20751953125 | 7.644150390625 |
| vectorized_elementwise_kernel<4, AUnaryFunctor MulFunctor> | 100 | 2.0 | 282.61572265625 | 5.652314453125 |
| vectorized_elementwise_kernel<4, sin_kernel_cuda> | 50 | 1.0 | 264.9619140625 | 5.29923828125 |
| vectorized_elementwise_kernel<4, cos_kernel_cuda> | 50 | 1.0 | 259.837890625 | 5.1967578125 |
| elementwise_kernel_2_1<128,... direct_copy_kernel_cuda> | 50 | 1.0 | 163.32763671875 | 3.266552734375 |
| elementwise_kernel_2_2_template<128,... MulFunctor> | 50 | 1.0 | 153.84765625 | 3.076953125 |
| vectorized_elementwise_kernel<4, BUnaryFunctor MulFunctor> | 50 | 1.0 | 136.70263671875 | 2.734052734375 |
| vectorized_elementwise_kernel<4, neg_kernel_cuda> | 50 | 1.0 | 134.654296875 | 2.6930859375 |
| elementwise_kernel_with_index<int, arange_cuda_out> | 50 | 1.0 | 105.2119140625 | 2.10423828125 |

### candidate_triton_rotary_001 (v1) Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| _rotary_embed_fused_kernel | 50 | 1.0 | 845.05615234375 | 16.901123046875 |

The candidate scope contains exactly one kernel (`_rotary_embed_fused_kernel`),
confirming the 11 -> 1 kernel fusion. The broadcast-multiply and `torch.cat`
library kernels are gone.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | - | `dec9aa12bc50886503831c48b82767e6a76ecd29d3a5c29cb41185d6ef633c39` | accepted |

## evidence_for_next_round

- The 11 -> 1 kernel fusion is confirmed: candidate emits a single
  `_rotary_embed_fused_kernel` at 1.0 kernels/call (vs 11.0 baseline).
- Candidate device time fell from 50.95 us/call to 16.90 us/call; the broadcast-mul
  + cat library kernels (~20.7 us/call) are eliminated.
- Wall time improved 55.57% (0.180151 ms -> 0.080036 ms), exceeding the 5% threshold.
- Remaining device time is a single fused elementwise kernel at ~16.9 us/call;
  device_ratio for the candidate is ~0.211 (vs ~0.281 baseline), i.e. host launch
  overhead still contributes the majority of the ~0.08 ms wall time. Further
  host-side reduction (e.g. launch cost, allocation of cos/sin buffers) may be the
  next lever, but this is evidence only.

## Stop Recommendation

- recommendation: `continue`
- evidence: target_mode=null; no stop condition triggered. Improvement is 55.57%
  with correctness pass.

## Exact Reproduction Commands

```bash
cd /root/kernelswift-rotary && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/maca/triton_rotary_001.py --warmup 5 --repeat 10 --full-traceback
```

```bash
cd /root/kernelswift-rotary && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/maca/triton_rotary_001.py --warmup 50 --repeat 100
```

```bash
cd /root/kernelswift-rotary && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/maca/triton_rotary_001.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/music_flamingo_rotary_embedding/maca/log/round_001_forward_50iter.pt.trace.json
```

All commands require `source /root/.profile` first (sets `MACA_PATH=/opt/maca`).

## Profiler Trace Filtering Note

The C500 trace emitted the same duplicate nested `gpu_user_annotation` scope markers
as `report_000.md`. The two CPU-side `user_annotation` markers for
`baseline_base` and `candidate_triton_rotary_001` were removed (the GPU-side
`gpu_user_annotation` markers retained), producing
`log/round_001_forward_50iter.pt.trace.filtered.json`, which `summarize_trace.py`
consumed without the `overlapping scope events` error. Raw trace preserved at
`log/round_001_forward_50iter.pt.trace.json`.

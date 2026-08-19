# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `maca/baseline_adapter.py`
- Accepted reference: `base.py` (Phase 0 baseline reference)
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `40c1a8bfbd9a0e957f21ae8ac686aa4c378a28299fd1f053d1e35b5fa8c443e0`
- Accepted reference SHA256: `98be7d25ad94930fd9497e6ab33e64970c509c6c3569c8c3df40ae9d22b9a341`
- Base SHA256: `98be7d25ad94930fd9497e6ab33e64970c509c6c3569c8c3df40ae9d22b9a341`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `486242286573efe11bdd7b852247cb0ed4d63113e0e41c7c432ab65e654a6518`
- verification_tier: baseline
- screening_pairs: `not-run` (Phase 0 baseline has no accepted reference to screen against)

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | v0 (Model) and v1 (ModelNew) outputs identical under atol=1e-2, rtol=1e-2, equal_nan=True | `PASS accuracy; v0=0.204476 ms, v1=0.198539 ms, speedup=1.030x` | pass | `cd /root/kernelswift-rotary && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/maca/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback` (exit 0) |
| public_contract | v1 exposes `ModelNew`, `get_inputs`, `get_init_inputs`; forward returns `(cos, sin)` tuple | Harness loaded v1 via AST loader and executed `ModelNew` without error | pass | `ModelNew` defined; correctness run returned exit 0 |
| tuple/shape/dtype/device | outputs `(cos, sin)` each `(4,32,128)` fp32 contiguous on cuda:0 | `compare_values` recursive check passed (no shape/dtype/device mismatch raised) | pass | correctness run exit 0 |
| non-mutation / stream | forward must not mutate timestamps, preserve device/current stream | Harness clones inputs and compares under `torch.no_grad()`; no mutation detected | pass | correctness run exit 0 |

## Screening Evidence

Screening is not applicable for a Phase 0 baseline: there is no accepted reference
against which to compute a slowdown, so no `screened-out` determination is possible.

| Pair | Reference short wall ms | Candidate short wall ms | Candidate slower pct | Evidence |
|---:|---:|---:|---:|---|
| 1 | not-run | not-run | not-run | Phase 0 baseline |

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `sequential complete accepted-reference block, then complete candidate block` (harness `time_forward` per side)
- reference_raw_samples_ms: `not captured by harness (harness returns median only)` — see note below
- candidate_raw_samples_ms: `not captured by harness (harness returns median only)` — see note below
- reference_median_ms: `0.191406`
- candidate_median_ms: `0.190557`
- improvement_pct: `0.4435`

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
                = (0.191406 - 0.190557) / 0.191406 * 100
                = 0.4435
```

Three independent authoritative wall-timing runs were executed (warmup 50, repeat 100):

| Run | v0_ms (reference) | v1_ms (candidate) | speedup |
|---:|---:|---:|---:|
| 1 | 0.191406 | 0.190557 | 1.004x |
| 2 | 0.192640 | 0.190242 | 1.013x |
| 3 | 0.191099 | 0.191967 | 0.995x |

Median of the three runs:

- reference_median_ms = `median(0.191406, 0.192640, 0.191099)` = `0.191406`
- candidate_median_ms = `median(0.190557, 0.190242, 0.191967)` = `0.190557`

Note on precision: the harness prints medians rounded to 6 decimal places
(`f"{v0_ms:.6f}"`). The values recorded here are the unrounded harness medians as
reported to 6 decimals, which is the authoritative measurement surface. A
supplementary raw-sample capture (100 samples per side, same warmup/repeat/seed,
same loader and timing methodology) produced unrounded medians
`v0=0.185721553862 ms`, `v1=0.185627955943 ms`, consistent with the harness medians
within run-to-run variance (~3%) for this sub-0.2 ms elementwise operator.

## Evaluation Contract Mirror

- Evaluation Contract applicability: `not-applicable: Phase 0`
- hypothesis_id: `not-applicable: Phase 0`
- intervention: `not-applicable: Phase 0`
- expected_causal_chain: `not-applicable: Phase 0`
- primary_metric: `wall_time`
- Hypothesis verdict: `not-applicable: Phase 0`

No round decision exists in Phase 0, so no mechanism observables are declared and
no hypothesis verdict applies.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `summary`
- profiler_device_time: `available`
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- backend_runtime_fields: `not-applicable` (device `cat=kernel` durations are available; no GCU runtime-only fallback)

Reference and candidate scopes are collected and summarized independently. All
totals below are normalized by `iterations=50`.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| baseline_base (v0) | 2559.98046875 | 51.199609375 | 550 | 11.0 | 0.191406 | 0.26749218611224307 |
| candidate_baseline_adapter (v1) | 2547.44482421875 | 50.948896484375 | 550 | 11.0 | 0.190557 | 0.26736827555206577 |

```text
device_ratio = device_us_per_call / (candidate_median_ms * 1000)
```

### baseline_base (v0) Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| elementwise_kernel_2_2_broadcast_uncontiguous<512,... MulFunctor> (freqs*angle broadcast mul) | 50 | 1.0 | 659.19970703125 | 13.183994140625 |
| MACA_CatArrayBatchedCopyNoPartialWrite ... last_dim_cat (torch.cat) | 50 | 1.0 | 391.68115234375 | 7.833623046875 |
| vectorized_elementwise_kernel<4, AUnaryFunctor MulFunctor> (broadcast mul) | 100 | 2.0 | 280.056640625 | 5.6011328125 |
| vectorized_elementwise_kernel<4, cos_kernel_cuda> | 50 | 1.0 | 267.26416015625 | 5.345283203125 |
| vectorized_elementwise_kernel<4, sin_kernel_cuda> | 50 | 1.0 | 265.98486328125 | 5.319697265625 |
| elementwise_kernel_2_1<128,... direct_copy_kernel_cuda> | 50 | 1.0 | 163.5849609375 | 3.27169921875 |
| elementwise_kernel_2_2_template<128,... MulFunctor> | 50 | 1.0 | 153.84765625 | 3.076953125 |
| vectorized_elementwise_kernel<4, BUnaryFunctor MulFunctor> | 50 | 1.0 | 136.95849609375 | 2.739169921875 |
| vectorized_elementwise_kernel<4, neg_kernel_cuda> | 50 | 1.0 | 135.67822265625 | 2.713564453125 |
| elementwise_kernel_with_index<int, arange_cuda_out> | 50 | 1.0 | 105.724609375 | 2.1144921875 |

### candidate_baseline_adapter (v1) Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| elementwise_kernel_2_2_broadcast_uncontiguous<512,... MulFunctor> (freqs*angle broadcast mul) | 50 | 1.0 | 662.52880859375 | 13.250576171875 |
| MACA_CatArrayBatchedCopyNoPartialWrite ... last_dim_cat (torch.cat) | 50 | 1.0 | 385.0244140625 | 7.70048828125 |
| vectorized_elementwise_kernel<4, AUnaryFunctor MulFunctor> (broadcast mul) | 100 | 2.0 | 275.96240234375 | 5.519248046875 |
| vectorized_elementwise_kernel<4, cos_kernel_cuda> | 50 | 1.0 | 263.93701171875 | 5.278740234375 |
| vectorized_elementwise_kernel<4, sin_kernel_cuda> | 50 | 1.0 | 262.14599609375 | 5.242919921875 |
| elementwise_kernel_2_1<128,... direct_copy_kernel_cuda> | 50 | 1.0 | 164.09912109375 | 3.281982421875 |
| elementwise_kernel_2_2_template<128,... MulFunctor> | 50 | 1.0 | 154.3623046875 | 3.08724609375 |
| vectorized_elementwise_kernel<4, BUnaryFunctor MulFunctor> | 50 | 1.0 | 137.2138671875 | 2.74427734375 |
| vectorized_elementwise_kernel<4, neg_kernel_cuda> | 50 | 1.0 | 136.7021484375 | 2.73404296875 |
| elementwise_kernel_with_index<int, arange_cuda_out> | 50 | 1.0 | 105.46875 | 2.109375 |

Both scopes emit 11 kernels per forward call (elementwise-heavy), with identical
kernel names and near-identical durations — expected because the baseline adapter
is a semantically identical renaming (`Model` -> `ModelNew`) of `base.py`.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification (Phase 0) | not-applicable | 40c1a8bfbd9a0e957f21ae8ac686aa4c378a28299fd1f053d1e35b5fa8c443e0 | baseline |

## evidence_for_next_round

- Correctness passes: v0 and v1 outputs are numerically identical under atol/rtol 1e-2.
- 11 kernels per forward call; the two most expensive kernels are a broadcast
  multiply (`elementwise_kernel_2_2_broadcast_uncontiguous`, ~13.2 us/call) and the
  `torch.cat` (`MACA_CatArrayBatchedCopyNoPartialWrite`, ~7.7 us/call). cos/sin each
  ~5.3 us/call.
- Device time is ~51 us/call while wall time is ~0.19 ms/call; device_ratio ~0.267,
  i.e. the majority of wall time is host launch overhead across 11 small kernels,
  not device execution.
- Baseline is a pure PyTorch elementwise composition (no Triton kernel yet). The
  baseline adapter is semantically identical to base.py.

## Stop Recommendation

- recommendation: `continue`
- evidence: Phase 0 baseline established. No target set (target_mode=null).

## Exact Reproduction Commands

```bash
cd /root/kernelswift-rotary && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/maca/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback
```

```bash
cd /root/kernelswift-rotary && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/maca/baseline_adapter.py --warmup 50 --repeat 100
```

```bash
cd /root/kernelswift-rotary && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/maca/baseline_adapter.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/music_flamingo_rotary_embedding/maca/log/round_000_forward_50iter.pt.trace.json
```

All commands require `source /root/.profile` first (sets `MACA_PATH=/opt/maca`).

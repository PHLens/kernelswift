# Report 001

Result: accepted

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md`
- Candidate: `triton_mhcc_001.py`
- Accepted reference: `baseline_adapter.py`
- Accepted reference report: `rounds/report_000.md`
- Decision SHA256: `<decision_001.md>`
- Candidate SHA256: `f29b71c87712aa3f674c2ec6e448bf1026a81986ecaa070645e691d66e969c39`
- Accepted reference SHA256: `c3cc90deb93c69f092e876ddc0057ddf6c2d7ac28020f1b7fd1ed260bca72fee`
- Base SHA256: `4c5167f6cfe9099786f204fad222f3b780d3ebb814494b9582112422abf84ac5`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `d8f4b63bfbf09ce8a32f3bdcd4d85553f34abce7384e495ba5f66baf49bf795e`
- verification_tier: `authoritative`
- screening_pairs: `not-run` (candidate is a clear speedup; proceeded directly to authoritative timing)

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | pre/post/comb allclose atol=1e-2 rtol=1e-2 | `PASS accuracy; v0=1.732976 ms, v1=0.124444 ms, speedup=13.926x` | pass | correctness command, exit 0 |
| output tuple shape dtype device unchanged | `(pre, post, comb)` with shapes `[2,8,4]`, `[2,8,4]`, `[2,8,4,4]` fp32 on cuda | tuple of 3 fp32 tensors, correct shapes/device (harness `compare_values` recursively allclose'd each) | pass | correctness command |
| inputs not mutated | candidate reads inputs only; outputs are fresh `torch.empty` | harness clones inputs; forward allocates new `pre/post/comb`; no in-place op on inputs | pass | candidate source review |
| caller-selected device and current stream preserved | outputs on `x.device`; launch on current stream | outputs allocated on `x.device`; direct launch on current stream, no stream switch | pass | candidate source review |

Conformance, correctness, and every declared guardrail passed before adoption.

## Screening Evidence

Not run: candidate showed a clear speedup on the correctness pass (~13.9x); proceeded
directly to authoritative timing per the verifier contract (screening is only required
to short-circuit candidates that are at least 10% slower).

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `sequential complete accepted-reference block, then complete candidate block` (harness `time_forward` v0 then v1)
- reference_raw_samples_ms: `[1.365771, 1.665487, 1.673250]` (unrounded median per authoritative run)
- candidate_raw_samples_ms: `[0.118193, 0.118501, 0.118357]`
- reference_median_ms: `1.665487`
- candidate_median_ms: `0.118357`
- improvement_pct: `92.893`

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
               = (1.665487 - 0.118357) / 1.665487 * 100 = 92.893
```

Three authoritative wall runs were executed separately. Each run's value is the
harness-reported unrounded median of 100 repeat samples (`statistics.median`); the
median across the three runs is reported above. Improvement far exceeds the 5%
adoption threshold.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| candidate_kernel_count_per_call | decrease from ~133 to 1 or a tiny constant | 133.0 → 1.0 (exactly one fused Triton kernel per call) | pass | summarize_trace.py candidate scope: 50 kernels / 50 iterations |
| candidate_device_us_per_call | decrease from ~534.685 | 534.014 → 43.791 (12.2x reduction) | pass | summarize_trace.py candidate scope |
| fused_triton_kernel_count_per_call | equal to 1.0 | 1.0 (`_mhc_head_compute_mix_kernel` = 50 events / 50 iterations) | pass | summarize_trace.py + manual trace inspection |
| sinkhorn_sum_div_kernel_us_per_call | decrease from ~500 toward 0 | 0 (all `reduce_kernel_maca` sum and `DivFunctor` library kernels eliminated from candidate scope) | pass | candidate scope has only `_mhc_head_compute_mix_kernel` |
| correctness | pre/post/comb allclose atol=1e-2 rtol=1e-2 | PASS | pass | correctness command, exit 0 |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-001`
- intervention: `fuse the entire MHCHeadComputeMix forward into a single Triton kernel`
- expected_causal_chain: `133-library-kernel chain -> 1 fused kernel -> host launch overhead eliminated -> device time drops (in-kernel reductions) -> wall time drops`
- primary_metric: `wall_time`
- Hypothesis verdict: `confirmed`

All five mechanism_observables met their expectations. The causal chain is confirmed:
the 133-library-kernel chain collapsed to 1 fused kernel, `device_us_per_call` dropped
from 534.014 us to 43.791 us (12.2x), the sum/div library kernels disappeared entirely,
and wall time dropped 92.9% (from 1.665487 ms to 0.118357 ms).

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `available`
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- backend_runtime_fields: `not-applicable`

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| baseline_base | 26700.691 | 534.014 | 6650 | 133.0 | 1.665487 | 0.320635 |
| candidate_triton_mhcc_001 | 2189.568 | 43.791 | 50 | 1.0 | 0.118357 | 0.369996 |

```text
device_ratio = device_us_per_call / (median_ms * 1000)
```

Candidate `kernel_count_total`/`device_total_us` above are the true values from manual
trace inspection (50 `_mhc_head_compute_mix_kernel` events, 2189.568 us total), because
`summarize_trace.py`'s strict `end <= scope_end` filter dropped the 50th kernel whose
GPU interval extended 10.26 us past the CPU scope-close marker (async completion
boundary artifact). The raw `summarize_trace.py` output reported 49 kernels (0.98/call)
and 2145.789 us for this reason. Both are documented here; the true per-call values are
1.0 kernel and 43.791 us.

### Accepted Reference (baseline_base) Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| reduce_kernel_maca (sum, dim -1) | 1000 | 20.0 | 6164.765 | 123.295 |
| InputPerOutputImcontinuousReduceKernel (sum, dim -2) | 1000 | 20.0 | 5865.031 | 117.301 |
| vectorized_elementwise_kernel (CUDAFunctorOnSelf_add, `+eps`) | 2050 | 41.0 | 5556.686 | 111.134 |
| elementwise_kernel_3_2 (DivFunctor) | 1000 | 20.0 | 3950.075 | 79.001 |
| elementwise_kernel_2_2_template (DivFunctor) | 1000 | 20.0 | 3143.828 | 62.877 |
| elementwise_kernel_2_2_template (MulFunctor) | 150 | 3.0 | 438.508 | 8.770 |
| elementwise_kernel_2_2_template (CUDAFunctor_add) | 150 | 3.0 | 421.877 | 8.438 |
| sigmoid_kernel_cuda | 100 | 2.0 | 369.657 | 7.393 |
| reduce_kernel_maca (MaxNanFunctor, amax) | 50 | 1.0 | 321.020 | 6.420 |
| exp_kernel_cuda | 50 | 1.0 | 167.685 | 3.354 |
| elementwise_kernel_2_2_template (CUDAFunctor_add) | 50 | 1.0 | 160.253 | 3.205 |
| vectorized_elementwise_kernel (AUnaryFunctor Mul) | 50 | 1.0 | 141.308 | 2.826 |

### Candidate (candidate_triton_mhcc_001) Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| _mhc_head_compute_mix_kernel | 50 | 1.0 | 2189.568 | 43.791 |

The candidate scope contains a single kernel name: the fused `_mhc_head_compute_mix_kernel`,
launched exactly once per forward call. All 12 library kernel names from the reference
are gone.

### Profiler Trace Note (C500 duplicate nested marker)

Same known C500 issue as report_000: each scope had one `cat=user_annotation` and one
duplicate nested `cat=gpu_user_annotation` X marker, causing `overlapping scope events`.
Preserved the raw trace at
`log/round_001_forward_50iter.pt.trace.json` (36769 events) and produced a filtered copy
at `log/round_001_forward_50iter.filtered.pt.trace.json` (36767 events) dropping the 2
duplicate `gpu_user_annotation` scope markers. Summaries above were computed from the
filtered trace.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial authoritative verification | `f29b71c87712aa3f674c2ec6e448bf1026a81986ecaa070645e691d66e969c39` | `f29b71c87712aa3f674c2ec6e448bf1026a81986ecaa070645e691d66e969c39` | accepted (no repair needed) |

## evidence_for_next_round

- Host-bound bottleneck eliminated: 133 library launches/call → 1 fused Triton kernel/call, wall time 1.665487 ms → 0.118357 ms (92.9% improvement).
- Remaining candidate device_ratio is still ~0.37 (43.791 us device / 118.357 us wall), so ~63% of the now-much-smaller wall time is still host/launch/dispatch overhead — a single 16-program launch still leaves host launch overhead as the dominant cost.
- The fused kernel itself is only ~43.8 us of device time for 16 programs; the operator is now latency-bound on a single launch rather than throughput-bound.
- The `+eps` self-add elementwise kernels (41/call, ~111 us/call) and the sum/div library kernels (~500 us/call total) are fully eliminated in the candidate.
- Candidate fast-path guard correctly falls back to the PyTorch reference path for non-benchmark shapes/dtypes/devices, preserving the public contract.

## Stop Recommendation

- recommendation: `continue`
- evidence: `accepted` with 92.9% wall improvement. No target is set (`target_mode: null`), so no target-reached condition applies. Continue optimization if further rounds are warranted; the remaining bottleneck is single-launch host overhead.

## Exact Reproduction Commands

```bash
cd /root/kernelswift-mhcc && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix/maca/triton_mhcc_001.py --warmup 5 --repeat 10 --full-traceback
```

```bash
cd /root/kernelswift-mhcc && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix/maca/triton_mhcc_001.py --warmup 50 --repeat 100
```

```bash
cd /root/kernelswift-mhcc && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix/maca/triton_mhcc_001.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/mhc_head_compute_mix/maca/log/round_001_forward_50iter.pt.trace.json
```

```bash
cd /root/kernelswift-mhcc && python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/mhc_head_compute_mix/maca/log/round_001_forward_50iter.filtered.pt.trace.json --iterations 50 --scope baseline_base --wall-ms 1.665487
cd /root/kernelswift-mhcc && python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/mhc_head_compute_mix/maca/log/round_001_forward_50iter.filtered.pt.trace.json --iterations 50 --scope candidate_triton_mhcc_001 --wall-ms 0.118357
```

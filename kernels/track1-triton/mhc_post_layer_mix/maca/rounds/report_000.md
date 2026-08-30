# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `maca/baseline_adapter.py` (`ModelNew`)
- Accepted reference: `base.py` (`Model`) — Phase 0 baseline
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `2c0c121e2e4504e791fee3675ac1cd54d1322059fc537555578a91cabd2a24e6`
- Accepted reference SHA256: `e392799f72edcf7210373ac8f2522bcf3d4f740c8f9dd0d1f117bc627939ebf3`
- Base SHA256: `e392799f72edcf7210373ac8f2522bcf3d4f740c8f9dd0d1f117bc627939ebf3`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `17bf289997ea6c7a2961ba2640125464ed046471dbff9261a8dcba7fbfccc17e`
- verification_tier: `baseline`
- screening_pairs: `not-run: Phase 0`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | v0 and v1 outputs allclose (atol=1e-2, rtol=1e-2, equal_nan=True) | `PASS accuracy` | pass | exit 0; `PASS accuracy; v0=7.640548 ms, v1=7.640149 ms, speedup=1.000x` |
| output dtype/shape/device | bf16 `(2,4096,4,1280)` contiguous on cuda:0 | recursive allclose passed | pass | harness `compare_values` recursion |
| input non-mutation | forward must not mutate inputs | harness clones inputs and re-seeds | pass | `clone_value` in `run_forward` |
| device/stream | preserve caller-selected device & current stream | `torch.no_grad()` + same device | pass | `_detect_target_device` = cuda:0 |

Correctness command (exit 0):
```bash
cd /root/kernelswift-mhc && source /root/.profile && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mhc_post_layer_mix/base.py --v1_file kernels/track1-triton/mhc_post_layer_mix/maca/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback
```

## Screening Evidence

Not applicable: Phase 0 baseline (no candidate-vs-accepted comparison).

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `sequential complete accepted-reference block, then complete candidate block` (harness default; v0 timed then v1 timed)
- reference_raw_samples_ms: `[7.639543, 7.634985, 7.635598]`
- candidate_raw_samples_ms: `[7.636774, 7.636353, 7.636740]`
- reference_median_ms: `7.635598`
- candidate_median_ms: `7.636740`
- improvement_pct: `-0.014956` (Phase 0: identical computation, not an adoption decision)

Each sample above is itself the harness median over 100 repeats (harness `time_forward` returns `statistics.median(samples)`). Three independent runs were executed; the reported medians are the median of the three run values.

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
```

Note: two runs executed in parallel on the shared C500 produced contaminated samples (v0=14.59ms / v1=14.55ms) due to GPU contention; those were discarded and re-measured sequentially. The three samples above are all from sequential runs.

## Evaluation Contract Mirror

- Evaluation Contract applicability: `not-applicable: Phase 0`
- hypothesis_id: `not-applicable: Phase 0`
- intervention: `not-applicable: Phase 0`
- expected_causal_chain: `not-applicable: Phase 0`
- primary_metric: `not-applicable: Phase 0`
- Hypothesis verdict: `not-applicable`

No round decision exists for Phase 0.

## Profiler Evidence

- profiler_applicability: `required` (baseline)
- profiler_level: `summary`
- profiler_device_time: `available`
- iterations: `50` forward calls
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- backend_runtime_fields: `not-applicable` (device kernel time available)

Reference and candidate scopes collected and summarized independently, normalized by 50 forward calls.

KNOWN C500 ISSUE encountered: the raw trace contained duplicate nested scope markers — each scope had both a host-side `user_annotation` (CPU, ~7.7ms) and a device-side `gpu_user_annotation` (GPU, ~379ms over 50 iterations) with the same name. For `baseline_base` the host marker's interval overlapped the device marker's interval, so `summarize_trace.py` raised `overlapping scope events: baseline_base`. Resolution: the raw trace is preserved verbatim at `log/round_000_forward_50iter.pt.trace.json`; a filtered copy `log/round_000_forward_50iter.filtered.pt.trace.json` drops the 2 host-side `user_annotation` scope markers (keeping the device-side `gpu_user_annotation` intervals that contain the `cat=kernel` events), and both scopes are summarized from the filtered copy. This is a marker-duplication artifact, not a data loss — kernel events are unchanged.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| baseline_base (v0) | 377959.930664 | 7559.198613 | 300 | 6.0 | 7.635598 | 0.989994 |
| candidate_baseline_adapter (v1) | 378087.673340 | 7561.753467 | 300 | 6.0 | 7.636740 | 0.990086 |

```text
device_ratio = device_us_per_call / (median_ms * 1000)
```

### Baseline (v0) Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `mcblas__Mck_tf32gemm_nt_64x64x128_4m4n1k_256t_fp32_fp32_tf32_sb_0_0` | 50 | 1.0 | 303577.084961 | 6071.541699 |
| `...elementwise_kernel_3_2<128,2,float,...MulFunctor<float>...>` (mul) | 50 | 1.0 | 37053.953613 | 741.079072 |
| `...vectorized_elementwise_kernel<4,...CUDAFunctor_add<float>...>` (add) | 50 | 1.0 | 16997.886719 | 339.957734 |
| `...unrolled_elementwise_kernel_copy_cast<...float,c10::BFloat16,4>` (bf16 cast) | 100 | 2.0 | 11476.734863 | 229.534697 |
| `...unrolled_elementwise_kernel_copy_cast<...c10::BFloat16,float,4>` (bf16→fp32) | 50 | 1.0 | 8854.270508 | 177.085410 |

### Candidate (v1) Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `mcblas__Mck_tf32gemm_nt_64x64x128_4m4n1k_256t_fp32_fp32_tf32_sb_0_0` | 50 | 1.0 | 303572.221680 | 6071.444434 |
| `...elementwise_kernel_3_2<128,2,float,...MulFunctor<float>...>` (mul) | 50 | 1.0 | 37052.416016 | 741.048320 |
| `...vectorized_elementwise_kernel<4,...CUDAFunctor_add<float>...>` (add) | 50 | 1.0 | 17013.245605 | 340.264912 |
| `...unrolled_elementwise_kernel_copy_cast<...float,c10::BFloat16,4>` (bf16 cast) | 100 | 2.0 | 11544.062988 | 230.881260 |
| `...unrolled_elementwise_kernel_copy_cast<...c10::BFloat16,float,4>` (bf16→fp32) | 50 | 1.0 | 8905.727051 | 178.114541 |

Key observation for next round: the `einsum('abmn,abmc->abnc', ...)` (batched `[4,4] x [4,1280]` matmul) lowers to a **single tf32 GEMM** (`mcblas__Mck_tf32gemm_nt_64x64x128_4m4n1k_256t_fp32_fp32_tf32`), ~6071 µs/call = ~80% of total device time. The remaining ~20% is elementwise work: one mul (~741 µs), one add (~340 µs), and two bf16 cast kernels (~230 + ~178 µs).

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial Phase 0 verification | not-applicable | `2c0c121e2e4504e791fee3675ac1cd54d1322059fc537555578a91cabd2a24e6` | correctness pass; baseline recorded |

## evidence_for_next_round

- Correctness passed (exit 0) at atol/rtol 1e-2; wall ~7.64 ms per forward call on the C500.
- Device time dominates wall: `device_ratio ≈ 0.99` for both scopes — this is a compute/memory-dense op, unlike the prior host-bound operators.
- The einsum lowers to a **tf32 GEMM** (`mcblas__Mck_tf32gemm_nt_64x64x128_4m4n1k_256t_fp32_fp32_tf32_sb_0_0`), 1 call per forward, ~6071 µs (~80% of device time). This is the dominant kernel and primary optimization target.
- Elementwise tail: mul (741 µs), add (340 µs), and two bf16 copy_cast kernels (230 + 178 µs) account for the remaining ~20%.
- Kernel count is stable at 6.0 per forward call; both v0 and v1 produce identical kernel breakdowns (v1 is a verbatim adapter of v0's einsum path).
- Wall and device time are essentially identical between v0 and v1 (improvement −0.015%), confirming `baseline_adapter.py` is a faithful executable canonical of `base.py`.

## Stop Recommendation

- recommendation: `continue`
- evidence: Phase 0 baseline established; `baseline_adapter.py` is the canonical accepted reference for Round 1.

## Exact Reproduction Commands

Correctness:
```bash
cd /root/kernelswift-mhc && source /root/.profile && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mhc_post_layer_mix/base.py --v1_file kernels/track1-triton/mhc_post_layer_mix/maca/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback
```

Authoritative wall timing (run sequentially, never in parallel on the shared C500):
```bash
cd /root/kernelswift-mhc && source /root/.profile && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mhc_post_layer_mix/base.py --v1_file kernels/track1-triton/mhc_post_layer_mix/maca/baseline_adapter.py --warmup 50 --repeat 100
```

Profiler (forward):
```bash
cd /root/kernelswift-mhc && source /root/.profile && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mhc_post_layer_mix/base.py --v1_file kernels/track1-triton/mhc_post_layer_mix/maca/baseline_adapter.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/mhc_post_layer_mix/maca/log/round_000_forward_50iter.pt.trace.json
```

Trace summary (filtered copy):
```bash
cd /root/kernelswift-mhc && python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/mhc_post_layer_mix/maca/log/round_000_forward_50iter.filtered.pt.trace.json --iterations 50 --scope baseline_base --wall-ms 7.635598
cd /root/kernelswift-mhc && python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/mhc_post_layer_mix/maca/log/round_000_forward_50iter.filtered.pt.trace.json --iterations 50 --scope candidate_baseline_adapter --wall-ms 7.636740
```

# Report 001

Result: accepted

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md`
- Candidate: `maca/triton_mhc_001.py` (`ModelNew`)
- Accepted reference: `baseline_adapter.py` (last_accepted_kernel)
- Accepted reference report: `rounds/report_000.md`
- Decision SHA256: `9f3795f57808c0ada1ffaa6c02cfea507f9026cd8a09684987f0e30d3074da5a`
- Candidate SHA256: `e54e5b2e553449134eb3b6679d6ed759e30fd2dd42499f9a21716ae57216b944`
- Accepted reference SHA256: `2c0c121e2e4504e791fee3675ac1cd54d1322059fc537555578a91cabd2a24e6`
- Base SHA256: `e392799f72edcf7210373ac8f2522bcf3d4f740c8f9dd0d1f117bc627939ebf3`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `17bf289997ea6c7a2961ba2640125464ed046471dbff9261a8dcba7fbfccc17e`
- verification_tier: `authoritative`
- screening_pairs: `not-run: correct candidate proceeds directly to authoritative timing`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | v0 and v1 outputs allclose (atol=1e-2, rtol=1e-2, equal_nan=True) | `PASS accuracy` | pass | exit 0; `PASS accuracy; v0=7.666721 ms, v1=0.247846 ms, speedup=30.933x` |
| output shape dtype device | bf16 `(2,4096,4,1280)` contiguous on cuda:0 | recursive allclose passed | pass | harness `compare_values` recursion |
| input non-mutation | forward must not mutate inputs | harness clones inputs and re-seeds | pass | `clone_value` in `run_forward` |
| fp32 accumulate then bf16 cast | accumulate fp32 before final bf16 cast | kernel uses `tl.float32` accumulate then `tl.bfloat16` store | pass | candidate lines 58-68 |
| device/stream | preserve caller-selected device & current stream | fast-path allocates `out` on `x.device`, no stream change | pass | candidate line 111-113 |

Correctness command (exit 0):
```bash
cd /root/kernelswift-mhc && source /root/.profile && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mhc_post_layer_mix/base.py --v1_file kernels/track1-triton/mhc_post_layer_mix/maca/triton_mhc_001.py --warmup 5 --repeat 10 --full-traceback
```

## Screening Evidence

Not run: correct candidate proceeds directly to authoritative timing (per verifier contract, screening is only required when the candidate may be `screened-out`; here the correctness gate passed and a full 3-pair authoritative timing was executed).

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `sequential complete accepted-reference block, then complete candidate block` (harness default)
- reference_raw_samples_ms: `[7.635280, 7.633116, 7.633507]`
- candidate_raw_samples_ms: `[0.242602, 0.240464, 0.241083]`
- reference_median_ms: `7.633507`
- candidate_median_ms: `0.241083`
- improvement_pct: `96.841779`

Each sample is the harness median over 100 repeats (harness `time_forward` returns `statistics.median(samples)`). Three independent sequential runs were executed to respect the shared C500 (no parallel contamination; all samples < 10ms, none discarded).

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
= (7.633507 - 0.241083) / 7.633507 * 100 = 96.841779
```

speedup = 31.663x. The unrounded improvement (96.84%) far exceeds the 5.0% adoption threshold.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| candidate_kernel_count_per_call | decrease to 1.0 | 1.0 (baseline 6.0) | pass | candidate scope `kernel_count_per_call = 1.0` |
| tf32_gemm_us_per_call | decrease to 0 | 0 (baseline 6071 us/call); `mcblas__Mck_tf32gemm...` absent from candidate scope | pass | candidate scope kernels: only `_mhc_post_layer_mix_fused_kernel` |
| fused_triton_kernel_count_per_call | equal 1.0 | 1.0 | pass | candidate scope: 50 kernels / 50 iterations = 1.0 |
| candidate_device_us_per_call | decrease from 7559 | 168.560713 (baseline 7560.89) | pass | candidate scope `device_us_per_call = 168.560712890625` |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-001`
- intervention: `fuse the K=4 einsum contraction and the x*post_layer_mix + term2 elementwise tail into a single hand-written Triton kernel that does 4 explicit fp32 multiply-accumulates instead of a 64x64x128-tiled tf32 GEMM`
- expected_causal_chain:
  1. the tf32 GEMM disappears → **confirmed** (0 tf32 GEMM kernels in candidate scope)
  2. elementwise mul/add/bf16-cast folded into one kernel, 6 kernels collapse to 1 → **confirmed** (kernel_count_per_call = 1.0)
  3. device time per call drops sharply from 7559 us → **confirmed** (168.56 us/call)
  4. wall time drops accordingly → **confirmed** (7.633507 ms → 0.241083 ms)
- primary_metric: `wall_time`
- Hypothesis verdict: `confirmed`

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `available`
- iterations: `50` forward calls
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- backend_runtime_fields: `not-applicable` (device kernel time available)

Known C500 issue handled identically to report_000: the raw trace contains duplicate nested scope markers (host `user_annotation` + device `gpu_user_annotation` with the same name). The raw trace is preserved verbatim at `log/round_001_forward_50iter.pt.trace.json`; a filtered copy `log/round_001_forward_50iter.filtered.pt.trace.json` drops the 2 host-side `user_annotation` scope markers (keeping device-side `gpu_user_annotation` intervals), and both scopes are summarized from the filtered copy. No kernel data is altered.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| baseline_base (v0) | 378044.667480 | 7560.893350 | 300 | 6.0 | 7.633507 | 0.990488 |
| candidate_triton_mhc_001 (v1) | 8428.035645 | 168.560713 | 50 | 1.0 | 0.241083 | 0.699181 |

```text
device_ratio = device_us_per_call / (median_ms * 1000)
```

### Accepted Reference Top Kernels (baseline_base)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `mcblas__Mck_tf32gemm_nt_64x64x128_4m4n1k_256t_fp32_fp32_tf32_sb_0_0` | 50 | 1.0 | 303559.165039 | 6071.183301 |
| `...elementwise_kernel_3_2<128,2,float,...MulFunctor<float>...>` (mul) | 50 | 1.0 | 37053.440918 | 741.068818 |
| `...vectorized_elementwise_kernel<4,...CUDAFunctor_add<float>...>` (add) | 50 | 1.0 | 16943.358887 | 338.867177 |
| `...unrolled_elementwise_kernel_copy_cast<...float,c10::BFloat16,4>` (bf16 cast) | 100 | 2.0 | 11570.686523 | 231.413731 |
| `...unrolled_elementwise_kernel_copy_cast<...c10::BFloat16,float,4>` (bf16→fp32) | 50 | 1.0 | 8918.016113 | 178.360322 |

### Candidate Top Kernels (candidate_triton_mhc_001)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `_mhc_post_layer_mix_fused_kernel` | 50 | 1.0 | 8428.035645 | 168.560713 |

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | `e54e5b2e553449134eb3b6679d6ed759e30fd2dd42499f9a21716ae57216b944` | `e54e5b2e553449134eb3b6679d6ed759e30fd2dd42499f9a21716ae57216b944` | accepted |

## evidence_for_next_round

- The fused K=4 Triton kernel (`_mhc_post_layer_mix_fused_kernel`) collapses 6 kernels to 1 and reduces device time from 7560.89 us/call to 168.56 us/call (97.8% device-time reduction), with wall improving from 7.633507 ms to 0.241083 ms (96.84%).
- The tf32 GEMM (`mcblas__Mck_tf32gemm_nt_64x64x128_4m4n1k_256t_fp32_fp32_tf32`) is fully eliminated (0 occurrences in the candidate scope).
- candidate device_ratio is now 0.699 (vs ~0.99 for the baseline): the candidate is no longer purely device-bound; ~30% of candidate wall (0.241 ms) is host-side launch/overhead, meaning a fraction of the remaining wall is now launch-bound rather than kernel-bound.
- The kernel uses `num_warps=1` and `BLOCK=1024` with a single program axis of `grid = ceil(2*4096*4*1280/1024) = 40960` programs; there is no `tl.dot`. `tl.zeros([BLOCK])` is used (target profile lists it "Unknown" but it compiled and ran correctly on this C500 runtime).
- Remaining device time (168.56 us/call) is the single fused kernel; further gains would target kernel-level launch/tail effects or vectorization, but the dominant win is already realized.

## Stop Recommendation

- recommendation: `continue`
- evidence: Round 001 accepted with 96.84% wall improvement; the campaign may continue with further rounds if desired, but the primary bottleneck (tf32 GEMM) is eliminated.

## Exact Reproduction Commands

Correctness:
```bash
cd /root/kernelswift-mhc && source /root/.profile && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mhc_post_layer_mix/base.py --v1_file kernels/track1-triton/mhc_post_layer_mix/maca/triton_mhc_001.py --warmup 5 --repeat 10 --full-traceback
```

Authoritative wall timing (run sequentially):
```bash
cd /root/kernelswift-mhc && source /root/.profile && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mhc_post_layer_mix/base.py --v1_file kernels/track1-triton/mhc_post_layer_mix/maca/triton_mhc_001.py --warmup 50 --repeat 100
```

Profiler (forward):
```bash
cd /root/kernelswift-mhc && source /root/.profile && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mhc_post_layer_mix/base.py --v1_file kernels/track1-triton/mhc_post_layer_mix/maca/triton_mhc_001.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/mhc_post_layer_mix/maca/log/round_001_forward_50iter.pt.trace.json
```

Trace summary (filtered copy):
```bash
cd /root/kernelswift-mhc && python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/mhc_post_layer_mix/maca/log/round_001_forward_50iter.filtered.pt.trace.json --iterations 50 --scope baseline_base --wall-ms 7.633507
cd /root/kernelswift-mhc && python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/mhc_post_layer_mix/maca/log/round_001_forward_50iter.filtered.pt.trace.json --iterations 50 --scope candidate_triton_mhc_001 --wall-ms 0.241083
```

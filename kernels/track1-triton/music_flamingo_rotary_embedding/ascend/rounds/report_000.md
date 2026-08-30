# Round 000 Report — Phase 0 Baseline Verification

## Decision

- classification: `baseline-established` (Phase 0 baseline verification, not an adoption decision)
- candidate: `kernels/track1-triton/music_flamingo_rotary_embedding/ascend/baseline_adapter.py` (Model renamed `ModelNew`)
- accepted reference: `kernels/track1-triton/music_flamingo_rotary_embedding/base.py`
- source (reference) sha256: `98be7d25ad94930fd9497e6ab33e64970c509c6c3569c8c3df40ae9d22b9a341`
- candidate sha256: `40c1a8bfbd9a0e957f21ae8ac686aa4c378a28299fd1f053d1e35b5fa8c443e0`
- harness sha256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`

## Correctness

- command: `python3 auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/ascend/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback`
- result: `PASS accuracy; v0=0.601575 ms, v1=0.608120 ms, speedup=0.989x` (1 passed, 0 failed)
- exit status: 0

### Tuple output comparison (how the harness handles `(cos, sin)`)

The harness `compare_values` (in `auto_bench.py`) is a recursive structural comparator. For a
tuple output it branches at the tuple case:

- Both sides must be `tuple` (else "output type mismatch").
- `len(v0) == len(v1)` must hold (else "tuple length mismatch").
- It then recurses element-wise: `compare_values(item0, item1, f"{path}[{i}]", atol, rtol)`.

So the `(cos, sin)` tuple is compared as `output[0]` (cos) then `output[1]` (sin). Each element
is a `Tensor[4,32,128]` fp32 and is compared by the tensor branch:

- shape must match exactly (`v0.shape != v1.shape` -> "tensor shape mismatch").
- for floating-point tensors it uses
  `torch.allclose(lhs, rhs, atol=1e-2, rtol=1e-2, equal_nan=True)`.
- on mismatch it reports `max_abs_diff` and `mean_abs_diff`.

Net: the harness compares the tuple element-wise across `(cos, sin)`, treating each tensor as an
independent fp32 `allclose` check with `atol=1e-2`, `rtol=1e-2`. Tuple outputs are handled
correctly; no flattening/concatenation is performed.

## Authoritative timing (baseline benchmark)

- command: `python3 auto_bench.py --v0_file .../base.py --v1_file .../baseline_adapter.py --warmup 50 --repeat 100`
- result: `PASS accuracy; v0=0.581820 ms, v1=0.581270 ms, speedup=1.001x`

| scope | unrounded median wall_time_ms |
|---|---:|
| reference (`base.py`) | 0.581820 |
| candidate (`baseline_adapter.py`) | 0.581270 |

Both are byte-identical modulo the `Model` -> `ModelNew` class rename; the 1.001x speedup is
within measurement noise, confirming the baseline adapter is a faithful reproduction of the
reference.

## Profiler evidence (CANN msprof, device kernels)

- command:
  `python3 auto_bench.py --v0_file .../base.py --v1_file .../baseline_adapter.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/music_flamingo_rotary_embedding/ascend/log/music_rotary_baseline_forward_50iter.pt.trace.json`
- iterations per scope: 50
- harness emitted per-scope CANN directories:
  - `log/profiling_data/baseline_base/profiling_data`
  - `log/profiling_data/candidate_baseline_adapter/profiling_data`
- summary tool: `skills/kernel-opt-loop/scripts/summarize_cann_trace.py` (device_time_available=true)

| scope | device_total_us | device_us_per_call | kernel_count_total | kernel_count_per_call | device_ratio |
|---|---:|---:|---:|---:|---:|
| baseline_base | 2389.18 | 47.7836 | 700 | 14.0 | 0.0821 |
| candidate_baseline_adapter | 2382.76 | 47.6552 | 700 | 14.0 | 0.0820 |

`device_ratio` = device_us_per_call / (wall_ms * 1000), using wall_ms 0.581820 (reference) and
0.581270 (candidate).

Top kernels (baseline_base, us/call):

| kernel | count/call | us/call |
|---|---:|---:|
| aclnnRepeatInterleaveIntWithDim_RepeatInterleaveV2_RepeatInterleaveV2 | 1 | 9.6756 |
| aclnnMul_MulAiCore_Mul | 2 | 8.7412 |
| aclnnCat_BroadcastToAiCore_BroadcastTo | 2 | 7.8648 |
| aclnnCat_ConcatD_ConcatD | 1 | 4.8380 |
| aclnnCos_CosAiCore_Cos | 1 | 3.9124 |
| aclnnSin_SinAiCore_Sin | 1 | 3.8640 |
| aclnnArange_ArangeAiCore_Range | 1 | 3.8080 |
| aclnnMuls_MulAiCore_Mul | 2 | 2.0320 |
| aclnnRepeatInterleaveIntWithDim_BroadcastToAiCore_BroadcastTo | 1 | 1.1000 |
| aclnnDivs_RealDivAiCore_RealDiv | 1 | 1.0128 |
| aclnnNeg_NegAiCore_Neg | 1 | 0.9348 |

The candidate scope is essentially identical (47.66 us/call, same 14 kernels), confirming the
baseline adapter reproduces the reference kernel pattern exactly.

### Trace scope note (harness behavior)

The harness `_export_profile_npu` writes the chrome trace with `prof.export_chrome_trace(str(args.profile_output))`
inside the per-scope loop using the SAME output path, so the second scope's export overwrites the
first. The resulting `.pt.trace.json` therefore contains only the last scope's `record_function`
(`candidate_baseline_adapter`); the `baseline_base` `record_function` span is absent. This does NOT
affect device-kernel attribution because each scope's CANN `ai_core_op_summary.db` is captured under
its own `ASCEND_WORK_PATH` subdirectory (700 tasks each = 50 iters × 14 kernels), so `summarize_cann_trace.py`
was run per-scope directory WITHOUT `--trace/--scope` isolation. The trace's `ts` clock (~1.78e15)
and the CANN `start_time` clock are unrelated, so time-range isolation via the overwritten trace
would not have worked anyway.

## Measurement fingerprint inputs

| field | value |
|---|---|
| harness_sha256 | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` |
| base_sha256 | `98be7d25ad94930fd9497e6ab33e64970c509c6c3569c8c3df40ae9d22b9a341` |
| baseline_adapter_sha256 | `40c1a8bfbd9a0e957f21ae8ac686aa4c378a28299fd1f053d1e35b5fa8c443e0` |
| shape | `timestamps=[4,32] fp32; seq_len=32; dim=64; output (cos,sin) each [4,32,128] fp32` |
| dtype | fp32 |
| device | npu:0 |
| warmup | 50 |
| repeat | 100 |
| profiler_iterations | 50 |
| seed | 42 |
| atol/rtol | 1e-2 / 1e-2 |

## Evaluation Contract mirror

The project has no per-mechanism target defined (`target_mode: null`). Phase 0 establishes the
baseline. Observable mechanisms by name:

| mechanism | expectation | observation | verdict |
|---|---|---|---|
| tuple output `(cos, sin)` compared correctly | harness recurses element-wise over tuple | confirmed in `compare_values` tuple branch | confirmed |
| many small elementwise/reduction kernels | forward does arange/repeat_interleave/broadcast/cat/cos/sin | 14 device kernels/call | confirmed |
| device-time bottleneck | device ratio low (host/launch-dominated) | device_ratio ≈ 0.082 | confirmed |

Overall hypothesis verdict (kernel fragmentation as optimization target): `confirmed`.

## evidence_for_next_round

- Baseline wall time (reference median): **0.581820 ms**; candidate reproduces it (0.581270 ms).
- Baseline device time: **47.78 us/call** across **14 kernels/call** (700 total over 50 iters).
- Device utilization is only ~8.2% of wall time — the operator is host/launch-overhead dominated.
- The 14 kernels are dominated by `RepeatInterleaveV2` (9.7us), `Mul` (8.7us), `BroadcastTo` (7.9us),
  `ConcatD` (4.8us), `Cos`/`Sin` (3.9us each), `Arange` (3.8us) — a clear fragmentation signature.
- Likely optimization: fuse the small elementwise/reduction chain into fewer kernels (e.g. a single
  Triton kernel or fused precompute of `position_angles`/`inv_freq`), eliminating per-op launch overhead.

## Reproduction commands

```bash
cd /workspace/kernelswift/.worktrees/music-rotary-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/ascend/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback
python3 auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/ascend/baseline_adapter.py --warmup 50 --repeat 100
python3 auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/ascend/baseline_adapter.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/music_flamingo_rotary_embedding/ascend/log/music_rotary_baseline_forward_50iter.pt.trace.json
python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py kernels/track1-triton/music_flamingo_rotary_embedding/ascend/log/profiling_data/baseline_base/profiling_data --iterations 50 --wall-ms 0.581820
python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py kernels/track1-triton/music_flamingo_rotary_embedding/ascend/log/profiling_data/candidate_baseline_adapter/profiling_data --iterations 50 --wall-ms 0.581270
```

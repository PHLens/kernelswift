# Round Status 000

- Round: 000
- Phase: Phase 0 (baseline establishment)
- Result: baseline

## Verification Start

- started_at: 2026-08-18T00:00:00Z (project epoch)
- phase: verification start
- completed_commands: []
- artifact_hashes: pending
- raw_samples: none
- next_safe_action: run correctness + benchmark reproduction command

## After Correctness

- phase: correctness complete
- completed_commands:
  - `python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/ascend/baseline_adapter.py --warmup 50 --repeat 100`
- result: `PASS accuracy; v0=0.320635 ms, v1=0.319655 ms, speedup=1.003x`
- artifact_hashes:
  - base.py: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2`
  - baseline_adapter.py: `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e`
  - auto_bench.py: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- next_safe_action: run profiler command

## After Benchmark

- phase: benchmark complete
- completed_commands:
  - benchmark (correctness command also yields median wall times)
- raw_samples: single comparison (Phase 0 baseline, not interleaved authoritative timing)
- reference_median_ms: 0.320635
- candidate_median_ms: 0.319655
- next_safe_action: run profiler command

## After Profiler

- phase: profiler complete
- completed_commands:
  - profiler command (forward mode, warmup 20, iterations 50)
  - summarize_cann_trace.py per-scope (reference + candidate)
- profiler_output: `kernels/track1-triton/mm_encoder_attention/ascend/log/round_000_forward_50iter.pt.trace.json`
- cann_dirs:
  - reference: `log/profiling_data/reference_baseline_adapter/profiling_data`
  - candidate: `log/profiling_data/candidate_baseline_adapter/profiling_data`
- device evidence:
  - reference_baseline_adapter: device_total_us=6007.62, device_us_per_call=120.1524, kernel_count_total=343, kernel_count_per_call=6.86, device_ratio=0.3472
  - candidate_baseline_adapter: device_total_us=5387.72, device_us_per_call=107.7544, kernel_count_total=336, kernel_count_per_call=6.72, device_ratio=0.3237
- next_safe_action: write report_000.md

## Verification End

- phase: verification complete
- result: baseline
- report: `rounds/report_000.md`
- next_safe_action: none (await Orchestrator)

# Round Status 001

- Round: 001
- Phase: verifying (authoritative timing)
- Result: no-improvement

## Verification Start

- started_at: 2026-08-18
- phase: verification start
- candidate: `triton_attn_001.py`
- candidate_sha256: `61eeb3367619684e6f61ea3a908c1fc78a575834b4a84c032748277d0e76be74`
- accepted_reference: `baseline_adapter.py`
- accepted_reference_sha256: `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e`
- completed_commands: []
- next_safe_action: run correctness + authoritative timing (3 interleaved pairs)

## After Correctness

- phase: correctness complete
- completed_commands:
  - `auto_bench.py --v0_file .../base.py --v1_file .../triton_attn_001.py --warmup 50 --repeat 100`
- result: `PASS accuracy` (all 3 pairs)
- next_safe_action: run profiler command

## After Authoritative Timing (3 interleaved pairs)

- phase: authoritative timing complete
- completed_commands: three `auto_bench.py` runs (base.py vs triton_attn_001.py, warmup 50 / repeat 100)
- reference_raw_samples_ms: `[0.348605, 0.346175, 0.357905]`
- candidate_raw_samples_ms: `[0.339685, 0.336795, 0.356130]`
- reference_median_ms: 0.348605
- candidate_median_ms: 0.339685
- improvement_pct: 2.5588 (< 5.0 → no-improvement)
- next_safe_action: run profiler command

## After Profiler (targeted Level 2)

- phase: profiler complete
- completed_commands:
  - profiler command (forward mode, warmup 20, iterations 50)
  - summarize_cann_trace.py per-scope (reference_baseline_adapter + candidate_triton_attn_001)
- profiler_output: `log/round_001_forward_50iter.pt.trace.json`
- reference_baseline_adapter: device_us_per_call=118.9404, kernel_count_per_call=6.78, device_ratio=0.3287
- candidate_triton_attn_001: device_us_per_call=104.1496, kernel_count_per_call=1.0, device_ratio=0.2998
- mechanism: kernel_count 6.78→1.0 (pass); transpose_wrappers→0 (pass); device_us_per_call −14.79 us only (partial, not −62 us)
- next_safe_action: write report_001.md

## Verification End

- phase: verification complete
- result: no-improvement
- report: `rounds/report_001.md`
- next_safe_action: none (await Orchestrator)

# Round Status 001

Round 001 verification for `fused_moe` (BI150 backend).

## Status

- phase: `verifying` (complete)
- result: `accepted`
- started_at: `2026-08-19T20:10:00Z` (approx)
- completed_at: `2026-08-19T20:20:00Z` (approx)

## Inputs

- candidate: `triton_fused_moe_001.py`, SHA `8424c7a01bc1d293c2b0ef509dd895950112cfb71dedd145053b4ac3f7eb9ad6`
- accepted reference: `baseline_adapter.py`, SHA `8e5c70232e541a02d83343216376ece9127a1c3e6ea6af77dc77a2723783facf`
- decision: `rounds/decision_001.md`, SHA `0745c37ddc4a5e27811d9ad20845d8b168017033b3b61f59f253d2129d9f7681`
- baseline (report_000): wall `3.258671 ms`, device `968.162 us/call`, kernel_count `123.9/call`, device_ratio `0.297`

## Completed Commands

| Step | Command | Return code | Evidence |
|---|---|---|---|
| correctness (base vs candidate) | `auto_bench.py --v0_file base.py --v1_file triton_fused_moe_001.py --warmup 50 --repeat 100 --full-traceback` | `0` | `PASS accuracy; v0=3.214486 ms, v1=2.474534 ms, speedup=1.299x` |
| independent numeric probe | base Model vs ModelNew (tie-constructed router) | `0` | allclose True, max_abs_diff 7.63e-06, shape [83,128] fp16 |
| wrapper generation | `sed 's/^class ModelNew/class Model/' baseline_adapter.py` | `0` | `/tmp/fm_baseline_model_001.py`, SHA `dd7cb62d13f2637522fdaa5975a5a7818745efaa05d7ca3be2a2718089c3ecb3` |
| timing pair 1 | wrapper vs candidate | `0` | ref 3.239021, cand 2.525259 |
| timing pair 2 | wrapper vs candidate | `0` | ref 3.167858, cand 2.450700 |
| timing pair 3 | wrapper vs candidate | `0` | ref 3.158865, cand 2.488731 |
| targeted profiling | `--profile --profile-reference-file baseline_adapter.py` | `0` | `log/round_001_forward_50iter.pt.trace.json` |
| summarize reference | `summarize_trace.py --scope reference_baseline_adapter` | `0` | 968.677 us/call, 123.9 kernels/call |
| candidate scope summarize | `summarize_trace.py --scope candidate_...` | `2` (overlap) | fallback: time-interval separation |

## Raw Samples

- reference (v0) raw ms: `[3.239021, 3.167858, 3.158865]` → median `3.167858`
- candidate (v1) raw ms: `[2.525259, 2.450700, 2.488731]` → median `2.488731`
- improvement_pct: `(3.167858 - 2.488731) / 3.167858 * 100 = 21.44%`

## Profiler (time-interval separation fallback)

- reference: 6195 kernels / 50 = 123.9/call, 968.677 us/call
- candidate: 2705 kernels / 50 = 54.1/call, 504.312 us/call

## Next Safe Action

Verification complete; report written. Orchestrator owns terminal transition.

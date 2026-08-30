# Round Status 002

Round 002 verification for `fused_moe` (BI150 backend).

## Status

- phase: `verifying` (complete)
- result: `accepted`

## Inputs

- candidate: `triton_fused_moe_002.py`, SHA `6ac1f44b111285f5bf746110c51f6486868b12beb2deae3390663d74233f8ae5`
- canonical reference: `triton_fused_moe_001.py`, SHA `8424c7a01bc1d293c2b0ef509dd895950112cfb71dedd145053b4ac3f7eb9ad6`
- decision: `rounds/decision_002.md`, SHA `2d44dd2c808bf27c20cdd4d6ca0aa0ecba422080394462f6d176ccc2c5a146a6`

## Completed Commands

| Step | Command | Return code | Evidence |
|---|---|---|---|
| correctness (base vs 002) | `auto_bench.py --v0_file base.py --v1_file triton_fused_moe_002.py` | `0` | `PASS accuracy; v0=3.223496 ms, v1=0.506451 ms, speedup=6.365x` |
| independent numeric probe | base vs ModelNew (tie-constructed) | `0` | allclose True, max_abs_diff 1.53e-05 |
| wrapper generation | `sed 's/^class ModelNew/class Model/' triton_fused_moe_001.py` | `0` | `/tmp/fm_canonical_model_002.py`, SHA `f3cb21a574bb59f44815acdeeb45cc916b49337cdde82fdd8500fb44f40f3f46` |
| timing pair 1 | wrapper vs 002 | `0` | ref 2.443674, cand 0.493893 |
| timing pair 2 | wrapper vs 002 | `0` | ref 2.464602, cand 0.492143 |
| timing pair 3 | wrapper vs 002 | `0` | ref 2.474194, cand 0.493474 |
| targeted profiling | `--profile --profile-reference-file triton_fused_moe_001.py` | `0` | `log/round_002_forward_50iter.pt.trace.json` |

## Raw Samples

- reference (v0, canonical 001) raw ms: `[2.443674, 2.464602, 2.474194]` → median `2.464602`
- candidate (v1, 002) raw ms: `[0.493893, 0.492143, 0.493474]` → median `0.493474`
- improvement_pct: `(2.464602 - 0.493474) / 2.464602 * 100 = 79.98%`

## Profiler (time-interval separation; both scopes have overlapping Triton record_function)

- reference (001): 2700 kernels / 50 = 54.0/call, 500.652 us/call
- candidate (002): 491 kernels / 50 = 9.82/call, 140.840 us/call

## Next Safe Action

Verification complete; report written. Orchestrator owns terminal transition.

# Round 000 Status

Phase: Phase 0 (baseline establishment) — COMPLETE

## Progress

- [x] Read role contract, base.py, baseline_adapter.py, project.md, team-state.md, harness, summarize_cann_trace.py, role-context-template, report-template, verifier_context.md
- [x] Verified artifact SHA256 hashes match project.md (base, baseline_adapter, harness)
- [x] Confirmed runtime: torch 2.7.1+cpu, torch_npu 2.7.1.post4, npu available
- [x] Correctness + benchmark (warmup 50 / repeat 100): PASS
- [x] CANN profiler (2 scopes): device_us_per_call=41.2us, kernel_count_per_call=10.0, device_ratio~0.095
- [x] Write report_000.md + update verifier_context.md

## Correctness

`PASS accuracy; v0=0.456720 ms, v1=0.433775 ms, speedup=1.053x` (warmup 50, repeat 100)

## Benchmark (unrounded median wall ms)

- base.py (reference): 0.456720 ms
- baseline_adapter.py (candidate): 0.433775 ms

## Profiler (50 iterations, separately scoped CANN captures)

| Scope | Device us/call | Kernel count/call | Device ratio |
|---|---:|---:|---:|
| base.py | 41.162 | 10.0 | 0.0901 |
| baseline_adapter.py | 41.183 | 10.0 | 0.0949 |

## Result

baseline — host-bound (device_ratio ~9.5%), 10 kernels/call, reductions dominate device time (~22.5 us/call).

## Artifact Hashes

| Artifact | SHA-256 |
|---|---|
| `auto_bench.py` | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` |
| `../base.py` | `28d4d213638e5adc7f6bc52928c8f7596cd8446e3e0201beab364b5c3c7988fc` |
| `baseline_adapter.py` | `98cf1e1e5f493ae4ae08c9391f00db83b3cd783a1b194d9c0d63851091133a5d` |

## Next Safe Action

None — Phase 0 complete. Await Orchestrator state transition.

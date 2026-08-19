# Round Status 000

- phase: complete (Phase 0 baseline verification)
- last update: verification complete
- current_round: 000
- terminal_result: baseline

## Completed commands

| Step | Command | Status | Result |
|---|---|---|---|
| Environment probe | import torch, torch_npu | done | torch 2.7.1+cpu, torch_npu 2.7.1.post4, npu avail True, Ascend910B4 |
| Hash verify | sha256sum | done | all three hashes match reported |
| Correctness | --warmup 5 --repeat 10 --full-traceback | done | PASS accuracy; v0=7.697280 ms, v1=7.290885 ms, speedup=1.056x |
| Baseline benchmark | --warmup 50 --repeat 100 | done | PASS accuracy; v0=7.158795 ms, v1=7.159420 ms, speedup=1.000x |
| Profiler | --profile forward 50iter | done | two CANN scopes written + summarized |

## Raw samples

- reference_median_ms (base.py): `7.158795`
- candidate_median_ms (baseline_adapter.py): `7.159420`
- improvement_pct: `-0.008731`

## Profiler summary (iterations=50)

| Scope | Device us/call | Kernel count/call | Device ratio |
|---|---:|---:|---:|
| reference (base.py) | 743.948 | 126.0 | 0.103921 |
| candidate (baseline_adapter.py) | 743.517 | 126.0 | 0.103852 |

## Artifact hashes

| Artifact | SHA-256 |
|---|---|
| base.py | a0269ac15833098c11c63045318e829e404b09ca49ae8dce22b244e0d2894d2b |
| baseline_adapter.py | a7fc0001db3ee9e636241954d2c071b62acee518b23f4c59c19efee886203a02 |
| auto_bench.py | 71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29 |

## Measurement fingerprint

`47e60b0db91c4c67e55f92cf79f5dddf591925620ec4db38704dfb42f0f185dd`

## Next safe action

Await Orchestrator. Phase 0 complete; terminal result `baseline`.

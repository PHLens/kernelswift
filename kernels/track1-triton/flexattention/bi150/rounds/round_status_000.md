# Round Status 000

- phase: `verifying` (Phase 0 baseline establishment)
- round: `000`
- verification_tier: `baseline`
- result: `baseline`

## Artifact Hashes

| Artifact | SHA-256 |
|---|---|
| `kernels/track1-triton/flexattention/base.py` | `dd1359ad88a5d5aae48a115023e5ae9016e4d5b5e6a12781273703e8d5c6a6d0` |
| `kernels/track1-triton/flexattention/bi150/baseline_adapter.py` | `b8ec3458bf810e6f3e81f759a716810b2046e5b32e22fc1339ec526e319445d1` |
| `auto_bench.py` (actual workspace HEAD) | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` |
| `auto_bench.py` (project.md recorded) | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` |
| profiler trace | `f30acbc4127b15bd45427395b65833dd62770b87013dd45f7d4afa5ef85aeae8` |

**Harness hash mismatch**: `project.md` records `3d4fa4ee...` but the actual
workspace `auto_bench.py` (git HEAD `f154ddd`) is `71fb3ad0...`. The `base.py`
and `baseline_adapter.py` hashes match project.md exactly. This mismatch is a
frozen-metadata drift in project.md (the harness gained Ascend910B profiling
support in commit `f154ddd`), NOT a device/import/environment fault. Reported to
Orchestrator for a project.md fingerprint refresh; measurement proceeded on the
actual harness bytes `71fb3ad0...`.

## Completed Commands

| Step | Command | Exit code | Result |
|---|---|---:|---|
| frozen-file SHA256 | `sha256sum ...` | `0` | base/adapter match; harness differs (see above) |
| correctness | `auto_bench.py --warmup 50 --repeat 100 --full-traceback` | `0` | `PASS accuracy; v0=0.149244 ms, v1=0.148740 ms, speedup=1.003x` |
| wall sample 1 | `--warmup 50 --repeat 100` | `0` | v0=0.150274, v1=0.150771 |
| wall sample 2 | `--warmup 50 --repeat 100` | `0` | v0=0.150427, v1=0.149958 |
| wall sample 3 | `--warmup 50 --repeat 100` | `0` | v0=0.210191 (outlier), v1=0.150345 |
| wall sample 4 | `--warmup 50 --repeat 100` | `0` | v0=0.147672, v1=0.147532 |
| wall sample 5 | `--warmup 50 --repeat 100` | `0` | v0=0.150070, v1=0.149600 |
| wall sample 6 | `--warmup 50 --repeat 100` | `0` | v0=0.150458, v1=0.150080 |
| profiler | `--profile --profile-mode forward --profile-warmup 20 --profile-iterations 50` | `0` | trace written to `log/` |
| summarize `baseline_base` | `summarize_trace.py --iterations 50 --scope baseline_base` | `0` | device 12.8803125 us/call |
| summarize `candidate_baseline_adapter` | `summarize_trace.py --iterations 50 --scope candidate_baseline_adapter` | `0` | device 15.60587890625 us/call |

## Raw Samples

- reference (v0) raw samples ms: `[0.150274, 0.150427, 0.210191*, 0.147672, 0.150070, 0.150458]` (`*` = cold-start outlier, excluded from median)
- candidate (v1) raw samples ms: `[0.150771, 0.149958, 0.150345, 0.147532, 0.149600, 0.150080]`
- reference median ms (3 stable contiguous samples `0.147672, 0.150070, 0.150458`): `0.150070`
- candidate median ms: `0.149600`

## Next Safe Action

Report `Result=baseline` to Orchestrator. Do not modify `base.py`,
`baseline_adapter.py`, `project.md`, or `team-state.md`. Do not advance
`last_accepted_kernel`.

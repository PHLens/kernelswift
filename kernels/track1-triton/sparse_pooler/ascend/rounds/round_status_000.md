# Round 000 Status

- phase: verifying (Phase 0 baseline)
- started_at: 2026-08-18T14:01:03Z
- completed_at: 2026-08-18T14:05:00Z
- round: 000
- verification_tier: baseline
- terminal_result: baseline

## Progress

| Step | State | Artifact | SHA256 |
|---|---|---|---|
| correctness | done | PASS accuracy | - |
| baseline benchmark | done | v0=0.935560 ms, v1=0.939350 ms | - |
| profiler evidence | done | device 374.81/378.35 us/call, 14 kernels/call | - |
| report_000.md | done | rounds/report_000.md | (written) |

## Completed Commands

1. correctness: `auto_bench.py --v0_file base.py --v1_file baseline_adapter.py --warmup 5 --repeat 10 --full-traceback` → `PASS accuracy; v0=1.001665 ms, v1=1.013420 ms`
2. baseline benchmark: `auto_bench.py --v0_file base.py --v1_file baseline_adapter.py --warmup 50 --repeat 100` → `PASS accuracy; v0=0.935560 ms, v1=0.939350 ms`
3. profiler: `auto_bench.py ... --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output .../sparse_pooler_baseline_forward_50iter.pt.trace.json` → 2 CANN scopes (`baseline_base`, `candidate_baseline_adapter`)
4. summarize CANN per scope (device_us_per_call: reference 374.8104, candidate 378.3516; kernel_count_per_call 14.0 each)

## Artifacts

- report_000.md: baseline result, wall medians, profiler summary, fingerprints.
- measurement_fingerprint: `f4305d20c3f39dba64e252050fcc6cb437a1ba7a24fb0480530287bcd4e7a6e1`

## Next Safe Action

Await Orchestrator: validate artifacts, compute terminal result, set Phase 0 canonical pointers and commit.

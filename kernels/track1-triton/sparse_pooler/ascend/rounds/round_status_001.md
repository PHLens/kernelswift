# Round 001 Status

- phase: verifying
- started_at: 2026-08-18T14:10:04Z
- completed_at: 2026-08-18T14:14:00Z
- round: 001
- verification_tier: authoritative
- terminal_result: accepted

## Progress

| Step | State | Detail |
|---|---|---|
| correctness | done | PASS accuracy (v0=0.910860, v1=0.627850) |
| interleaved wall timing (3 pairs) | done | ref [0.934505, 0.884145, 0.978735], cand [0.618775, 0.637685, 0.594790] |
| profiler evidence | done | ref 379.47 us/call 14 k/call; cand 202.86 us/call 5 k/call |
| report_001.md | done | result=accepted, improvement=33.78% |

## Completed Commands

1. correctness: `auto_bench.py --v0_file base.py --v1_file triton_sparse_pooler_001.py --warmup 5 --repeat 10 --full-traceback` → PASS
2. authoritative timing (3 pairs, warmup 50 repeat 100):
   - ref 0.934505 / cand 0.618775
   - ref 0.884145 / cand 0.637685
   - ref 0.978735 / cand 0.594790
3. profiler: `--profile --profile-reference-file baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 50` → 2 CANN scopes
4. summarize CANN: reference (379.4664 us/call, 14.0 kernels), candidate (202.8556 us/call, 5.0 kernels)

## Result

- reference_median_ms = 0.934505
- candidate_median_ms = 0.618775
- improvement_pct = 33.784 % (≥ 5.0 threshold)
- hypothesis H-001: confirmed
- terminal result: accepted

## Next Safe Action

Await Orchestrator: validate artifacts, apply state transition (accepted), advance canonical pointer to triton_sparse_pooler_001.py, commit.

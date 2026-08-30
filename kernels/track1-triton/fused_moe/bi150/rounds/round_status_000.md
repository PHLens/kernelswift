# Round Status 000

Phase 0 baseline verification for `fused_moe` (BI150 backend).

## Status

- phase: `verifying`
- result: `baseline`
- started_at: `2026-08-19T20:00:00Z` (approx)
- completed_at: `2026-08-19T20:05:00Z` (approx)

## Completed Commands

| Step | Command | Return code | Evidence |
|---|---|---|---|
| runtime fingerprint check | `python3 -c "import torch,triton; ..."` | `0` | torch 2.7.1, triton 3.1.0, Iluvatar BI-V150 (7,1) |
| frozen-file SHA256 | `sha256sum base.py baseline_adapter.py auto_bench.py` | `0` | all match project.md frozen values |
| correctness 50/100 | `auto_bench.py --v0_file ... --v1_file ... --warmup 50 --repeat 100 --full-traceback` | `0` | `PASS accuracy; v0=3.250513 ms, v1=3.262031 ms, speedup=0.996x` |
| wall sample 1, 50/100 | `auto_bench.py ... --warmup 50 --repeat 100` | `0` | v0=3.281101 ms |
| wall sample 2, 50/100 | `auto_bench.py ... --warmup 50 --repeat 100` | `0` | v0=3.254634 ms |
| wall sample 3, 50/100 | `auto_bench.py ... --warmup 50 --repeat 100` | `0` | v0=3.258671 ms |
| forward profiler 20/50 | `auto_bench.py ... --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50` | `0` | `log/round_000_forward_50iter.pt.trace.json` |
| summarize `baseline_base` | `summarize_trace.py ... --scope baseline_base` | `0` | report Profiler Evidence |
| summarize `candidate_baseline_adapter` | `summarize_trace.py ... --scope candidate_baseline_adapter` | `0` | report Profiler Evidence |

## Artifact Hashes

- base_sha256: `a0269ac15833098c11c63045318e829e404b09ca49ae8dce22b244e0d2894d2b`
- baseline_adapter_sha256: `8e5c70232e541a02d83343216376ece9127a1c3e6ea6af77dc77a2723783facf`
- harness_sha256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- trace_sha256: `4a19895f74c2c1e4c3c3b867e782c47e00e71915282557c6dd866ca3a5d9540d`

## Raw Samples (v0 wall ms)

- `[3.281101, 3.254634, 3.258671]`
- unrounded median: `3.258671`

## Next Safe Action

Phase 0 baseline establishment is complete. Report written to `rounds/report_000.md`.
Orchestrator owns the terminal transition and canonical pointer updates.

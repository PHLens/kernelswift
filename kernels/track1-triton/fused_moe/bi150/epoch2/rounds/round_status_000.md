# Round Status 000

- phase: `initializing -> verifying (Phase 0 baseline verification)`
- measurement_exclusive: `true`
- verifier_owns_machine: `true`
- started_at: `2026-08-28T14:50Z`
- last_updated: `2026-08-28T14:56Z`
- result: `baseline`

## START

Preflight completed before any measurement:

| Check | Observation | Verdict |
|---|---|---|
| base SHA256 | `21e758535af58c21684d6e657ae44f91a0248d246eb424b6923a3c180bc32a3d` (3598 bytes) | matches `project.md` |
| harness SHA256 | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (29428 bytes) | matches `project.md` |
| candidate SHA256 | `752a25033b7629459c6eb128c60a4bdc3ab77b9c7cc97f5d3592bdff4cd45a47` | recorded |
| runtime fingerprint | `triton 3.1.0 / torch 2.7.1 / nvcc V10.2.89 / Iluvatar BI-V150 sm_71 / 16 SM / 17179869184 B` | matches `project.md` |
| measurement fingerprint (live recompute) | `fe73bc58146d8c16f524be2a00fe99b31e1b9678bca6b3702f4284a3ac0a5bef` | matches `project.md` |
| positive-control sibling fingerprints | flexattention `6dc07009…`, mm_encoder `0c4c7d66…` | both reproduced |
| CoreX bootstrap | `export COREX_VERSION=4.4.0; . /usr/local/corex/enable` applied to every command | applied |
| `log/` materialized | yes, gitignored by `epoch2/.gitignore:1:log/` | ok |

Next safe action: run the correctness gate.

## CORRECTNESS

- command: `auto_bench.py --v0_file ../../base.py --v1_file baseline_adapter.py --seed 42 --atol 1e-2 --rtol 1e-2 --warmup 0 --repeat 1`
- output: `PASS accuracy; v0=3.779870 ms, v1=3.379537 ms, speedup=1.118x` / `Summary: 1 passed, 0 failed, 1 total.`
- exit status: `0`
- output guardrail: `Tensor(shape=(83,128), dtype=float16, device=cuda:0)`, all-finite
- raw log: `log/round_000_correctness.txt`

Next safe action: run three ordered interleaved timing pairs.

## PAIRS

Three ordered interleaved pairs, warmup 50 / repeat 100, default stream, byte-identical flags:

| Pair | v0 (base.py) ms | v1 (baseline_adapter.py) ms | speedup |
|---:|---:|---:|---:|
| 1 | 3.253012 | 3.245672 | 1.002x |
| 2 | 3.271220 | 3.280101 | 0.997x |
| 3 | 3.255288 | 3.278401 | 0.993x |

- v0 median (unrounded, of 3 pair medians): `3.255288` ms
- v1 median (unrounded, of 3 pair medians): `3.278401` ms
- improvement_pct: `-0.709971`
- raw log: `log/round_000_wall_pairs.txt`

Note: per-pair w0/v1 numbers are each an internal median over 100 samples; the pair table
above is the ordered pair sequence. Identity-to-within-noise (~1.00x) as expected for
adapter-of-base.

Next safe action: collect separately scoped dual-scope profiler evidence.

## PROFILER

- mode: `forward` (kernel-mode fallback; `ModelNew.run_out` absent in the Phase-0 adapter)
- command: `auto_bench.py … --profile --profile-reference-file baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 100`
- trace: `log/round_000_forward_100iter.pt.trace.json` (80907079 bytes, SHA256 `8be1df73b934679ac723aaaf16acc337530b39cbbd8547cacff6193824093f11`)
- scopes summarized independently: `reference_baseline_adapter`, `candidate_baseline_adapter`
- reference: `device_us_per_call=967.852`, `kernel_count_per_call=123.95`, `device_ratio=0.297317`
- candidate: `device_us_per_call=968.612`, `kernel_count_per_call=124.00`, `device_ratio=0.295453`
- raw summaries: `log/round_000_summary_reference.json`, `log/round_000_summary_candidate.json`

Launch census (static-vs-data-dependent): measured **data-dependent**. Per-call CUDA op
counts are constant across repeated calls on fixed inputs (148, ×12) but scale with the
number of *active* experts: 8 active → 148, 7 active → 134, 2 active → 64. The
`mask.any()` early-exit therefore varies the launch count by ~14 launches per active
expert. Under the seed-42 benchmark inputs all 8 experts are active, so the benchmark
regime is stable at 148 ops/call, but a graph capture over this base path would be
data-shape-fragile.

Next safe action: write `rounds/report_000.md` and `state/verifier_context.md`.

## END

- terminal classification: `baseline`
- result: `baseline` (Phase 0; not `accepted` / `no-improvement`)
- artifacts written: `rounds/report_000.md`, `state/verifier_context.md`, this file
- deviations: one — kernel-mode profiling unavailable (`ModelNew.run_out` absent),
  forward-mode dual-scope fallback applied (sibling precedent).
- incidents: none
- next safe action: Orchestrator sets `phase: ready`, `last_completed_round: "000"`,
  `last_accepted_round: "000"`, `last_accepted_kernel: baseline_adapter.py`,
  `last_accepted_report: rounds/report_000.md`, `last_result: baseline`, and clears
  `measurement_exclusive`.

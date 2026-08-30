# Round Status 004

Updated at: verification start.

## State

- phase: `verifying`
- round: `004`
- classification: `not-yet-determined`
- measurement_exclusive: `true` (Verifier owns local measurement; Designer and Coder idle)
- capability gate: `lifecycle.fast-launcher` — probe outcome `proven` (Coder), so
  only the `accepted` / `no-improvement` rows of the decision's table are live.
  Verifier does not re-adjudicate the probe; it re-verifies its invariants.

## Identity

| Artifact | SHA-256 |
|---|---|
| decision_004.md | `30758ad4dd30ccb0087534e47f61ea0443bdeead40ba64d41c28dd052c397088` |
| candidate `triton_mm_encoder_attention_e2_004.py` | `f5aa1d709e4deeb1562757d795dd4da41217238dd15c53d33c2c338da1938020` |
| accepted reference `triton_mm_encoder_attention_e2_003.py` | `c39142c1df7d719e9ef7680b4712b226e293f683d934228f930fa966324c6bfe` |
| `base.py` (paired `--v0_file`) | `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` |
| `auto_bench.py` | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` |
| sketch_004.json | `d3e52f6af032014381908e03e87a6b1c3f5694090686df2af3bfe3a6d9474dbf` |
| coder_result_004.md | `9c8c46ef1b58233e464a30022fd2b0dedf2fce7b95410a501d95e2e24ac59e0e` |

## Known margin context (from Coder, to be independently checked)

- Coder's in-process forward-level lever: `223.505` -> `205.035 us` = `-18.470 us`
- Adoption threshold from report_003: `0.05 * 297.410 = 14.871 us`
- Margin on paper: `3.599 us`. Machine drifted ~5-7% within a single turn in
  round 003, so the decision is expected to be near the noise floor.

## Completed Commands

| # | Command | Status |
|---:|---|---|
| - | none yet | pending |

## Raw Samples

| Pair | Reference median ms | Candidate median ms | Correctness |
|---:|---|---|---|
| 1 | pending | pending | pending |
| 2 | pending | pending | pending |
| 3 | pending | pending | pending |

## Next Safe Action

Run the correctness gate, then the three interleaved paired measurements at
`--warmup 50 --repeat 100` (`base.py` versus the candidate) in one Verifier turn.
Because the margin is thin, follow with an interleaved control comparison of
e2_003 against e2_004, the dual-scope profiler, and an independent Level 2
forward-level lever measurement.

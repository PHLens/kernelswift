# Round Status 001

- phase: `verifying`
- measurement_exclusive: `true`
- verifier_owns_machine: `true`
- round: `001`
- started_at: `2026-08-28T15:10Z`
- last_updated: `2026-08-28T15:40Z`
- result: `accepted`

## START

Preflight — all immutable inputs re-verified before any measurement:

| Artifact | SHA-256 | Expected | Verdict |
|---|---|---|---|
| `triton_fused_moe_e2_001.py` (candidate) | `da623fa92819185a1e20a8a7cbaca40acd9bfb4a3147f8e1e7b1e757c6b24cb7` | matches team-lead | OK |
| `rounds/decision_001.md` | `62820af457c7b0b84232dc28bffd07009b5bc1ee482059728da06761381fd1d5` | `62820af4…` | OK |
| `rounds/sketch_001.json` | `6a46d4fd67b0cbce7a34ce41eac0c2b4cc19f00dd6e6098cf91a60e879634cb4` | `6a46d4fd…` | OK |
| `../../base.py` | `21e758535af58c21684d6e657ae44f91a0248d246eb424b6923a3c180bc32a3d` (3598 B) | unchanged | OK |
| `auto_bench.py` | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` | unchanged | OK |
| `profile_snapshot/triton_cuda.yaml` | `dc8fa4c0c73caecc78c6a886e5ffb18cb97371f3add7cb382dbe30887743b7ae` | unchanged | OK |

- anchor: `rounds/report_000.md`, v0 canon **3.255288 ms**
- measurement fingerprint: `fe73bc58…` (unchanged, no re-baseline needed)
- CoreX bootstrap applied to every command

Next safe action: correctness gate before timing.

## CORRECTNESS

Harness comparator (seed 42, atol=rtol=1e-2): PASS on all three pairs.

Independent Verifier suite (`log/round_001_correctness_suite.py`, 12 suites) —
**all PASS** after one verifier-side generator correction:

| Suite | max_abs | verdict |
|---|---:|---|
| seed42 | 1.5259e-05 | PASS |
| fp16_extreme | 3.9062e-03 | PASS |
| activation / 8 active | 1.5259e-05 | PASS |
| activation / 7 active (expert 7 excluded) | 1.5259e-05 | PASS |
| activation / 2 active | 1.5259e-05 | PASS |
| activation / all-tie zeros | 1.5259e-05 | PASS |
| activation / all rows to expert 0 | 9.5367e-06 | PASS |
| run_out call 1 vs base | 1.5259e-05 | PASS |
| run_out call 2 vs base | 1.5259e-05 | PASS |
| run_out no stale carry-over | (differs, correct) | PASS |
| non-target T=128 E=16 (tier-3 eager) | 1.5259e-05 | PASS |
| determinism 20 calls | bitwise-identical | PASS |

Correction recorded: the first fp16-extreme generator capped `hidden_states` at
1024, which drives `silu(gate)*up` to ~1.5e5 and overflows fp16 — **in base
itself** (`base_finite: False`, NaN vs NaN). Re-capped to 32 so the pipeline
stays in fp16 range while still spanning 5 magnitude tiers. A `comparison_valid`
flag was added so a non-finite base can never again read as a candidate FAIL.
This was a defect in Verifier instrumentation, not in the candidate.

Next safe action: three ordered interleaved wall pairs.

## PAIRS

Three ordered interleaved pairs, warmup 50 / repeat 100, default stream,
byte-identical flags:

| Pair | v0 (base.py) ms | v1 (candidate) ms | speedup |
|---:|---:|---:|---:|
| 1 | 3.193262 | 0.218936 | 14.585x |
| 2 | 3.219342 | 0.219792 | 14.647x |
| 3 | 3.154682 | 0.229606 | 13.740x |

- v0 median: `3.193262` ms
- v1 median: `0.219792` ms
- improvement vs round-000 canon (3.255288): **93.247%**
- raw log: `log/round_001_wall_pairs.txt`

The number matches the Coder smoke (0.219 ms) — no discrepancy to investigate.

Next safe action: branch-B census duties.

## CENSUS

- active tier: **tier-1 direct-address** (graph_direct bound, no failure flags)
- python launcher executions in 100 timed calls: **0** (8 during warmup/capture)
- cudaGraphLaunch: **1.00/call**; memcpy (DtoD copy-out): **1.00/call**
- kernel events inside the replay interior: **0** (confirms kineto blindness)
- recaptures inside the timed segment: **0** (budget 4 → 4)
- replay wall 0.204 ms vs forced-eager 0.627 ms → graph mechanism saves 423 µs

Harness dual-scope trace (`log/round_001_forward_100iter.pt.trace.json`):
- reference scope: 123.95 kernels/call, 968.869 µs/call — reproduces report_000
- candidate scope: **0.05 kernels/call, 0.549 µs/call** — the declared
  UNAVAILABLE-not-zero artifact; not usable as device evidence

Non-replay device control obtained by forcing the eager tier on the SAME target
shape (measurement-only override, source untouched): 282.5 µs/call.

Cross-check against the epoch-1 starting point (`triton_fused_moe_002.py`):
233.345 µs/call eager, its `_fused_moe_expert_kernel` at 55.954 µs/call.

Next safe action: write the report, verdict, and ledger.

## END

- terminal classification: **accepted**
- improvement_pct: **93.247** (≥ 5.0 gate cleared by a wide margin)
- falsification rules: FR-1 pass, FR-2 **fail** (see report), FR-3 pass,
  FR-4 pass-as-recorded, FR-5 pass
- artifacts: `rounds/report_001.md`, `rounds/verdict_001.json`,
  `state/verifier_context.md`, this file
- no repair requested; candidate hash frozen and unchanged throughout
- next safe action: Orchestrator advances `last_accepted_kernel` to
  `triton_fused_moe_e2_001.py` and `last_accepted_report` to
  `rounds/report_001.md`, sets `last_result: accepted`, resets both streaks,
  and clears `measurement_exclusive`

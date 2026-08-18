# Verifier Context State

- role_contract_sha256: `6bc3277527523950cab2c9295b07e6a5dc66030618c3b7e648c69a095a48f0a2`
- context_epoch: `1`
- last_completed_round: `000` (Verifier evidence complete; Orchestrator transition pending)
- accepted_kernel: `null` (team-state pointer remains Orchestrator-owned until the Phase 0 artifact gate)
- accepted_report: `null` (team-state pointer remains Orchestrator-owned until the Phase 0 artifact gate)
- recent_three_round_evidence: `Round 000 baseline; report rounds/report_000.md`
- open_hypotheses: `No Phase 0 optimization hypothesis; baseline bottleneck evidence is ready for Designer use.`
- artifact_read_hashes: `project/team state, source, harness, role contract, and summarizer ledger below`

## Current Bottleneck

- `baseline_adapter.py` wall median is `0.231739 ms`; its separately scoped profile reports `147.7526708984375 us/device-call` and `15.0 kernels/call`. The gatherTopK and bitonic-sort kernel families together account for `89.6741943359375 us/call` (`~60.69%`) and `4.0` launches/call.

## Recent Three-round Evidence

- Round `000`, result `baseline`, evidence `rounds/report_000.md`, change family `Phase 0 mechanical Model -> ModelNew adapter`; correctness passed and normalized ASTs match after the required rename.

## Open Hypotheses or Checks

- No hypothesis is selected by Verifier. The next design may use the measured wall baseline, 15-kernel call graph, and hot-kernel distribution as evidence.
- Preserve awareness that this C500 profiler trace duplicates named record-function intervals as CPU and GPU annotations; raw trace and the exact audit filter are retained in `log/`.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `project.md` | `bd527a7bbcdc2b9da539e815cc4ca5910826284920feb27ccd7a173e0b4138d3` | `000` |
| `team-state.md` | `26b6752463853667ec615296e46b300affff2fce23d464caacbf9e37970ea4d2` | `000` |
| `base.py` | `49ec0cf7a48679c23c5187eb7ba546fd41025019a9d6e9b59e1273c48a31dfbb` | `000` |
| `baseline_adapter.py` | `d92a1be10bf9ad036287ed0f769b195262132893fa78d88fa30f992fbe757827` | `000` |
| `../../auto_bench.py` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` | `000` |
| `skills/kernel-opt-loop/prompts/verifier.md` | `6bc3277527523950cab2c9295b07e6a5dc66030618c3b7e648c69a095a48f0a2` | `000` |
| `skills/kernel-opt-loop/scripts/summarize_trace.py` | `f625276c05a4539f86f272f68d78dd84ba0c94bb4ce153613dc4437df432148c` | `000` |

## Durable Result

- result: `baseline`
- report: `rounds/report_000.md`
- round status: `rounds/round_status_000.md`
- measurement fingerprint: `3fe7d50260b3670756d26f003427faccb76e1e79c204b0a500a62d2eb481c809`
- next safe action: Orchestrator validates the Phase 0 artifact gate, updates canonical pointers/counters/state, and releases measurement exclusivity.

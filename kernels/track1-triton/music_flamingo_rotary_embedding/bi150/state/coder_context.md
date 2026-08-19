# Coder Context State

- role_contract_sha256: `(coder.md)`
- context_epoch: `1`
- last_completed_round: `001`
- accepted_kernel: `null`
- accepted_report: `null`
- recent_three_round_evidence: `001 kernel-fusion candidate-ready (d91a112c...) smoke PASS 1.979x`
- open_hypotheses: `verify fusion collapsed ~13 launches to 1; confirm 5% adoption threshold on unrounded median wall`
- artifact_read_hashes: `see table`

## Current Bottleneck

- host-bound: device_ratio 0.194, ~80% wall is launch overhead (13 elementwise kernels)

## Recent Three-round Evidence

- `001`, `candidate-ready`, `rounds/coder_result_001.md`, change_family `kernel-fusion`

## Open Hypotheses or Checks

- Verifier to confirm kernel_count_per_call and device_us_per_call decrease under targeted profiling.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `skills/kernel-opt-loop/prompts/coder.md` | - | 001 |
| `skills/kernel-opt-loop/prompts/coder_targets/triton_cuda.md` | - | 001 |
| `skills/kernel-opt-loop/references/invariants.md` | - | 001 |
| `skills/kernel-opt-loop/references/anti-patterns.md` | - | 001 |
| `base.py` | `98be7d25ad94930fd9497e6ab33e64970c509c6c3569c8c3df40ae9d22b9a341` | 001 |
| `baseline_adapter.py` | `433569bbac3bab158ff211a6de7ecb40ec7236d74a3eb7ab7c2b487e1b41772a` | 001 |
| `rounds/decision_001.md` | `28a716e6bafa46e0bd9c39350317e42173694b9406eb3c620c361b55db0bb383` | 001 |
| `auto_bench.py` | - | 001 |
| `project.md` | - | 001 |

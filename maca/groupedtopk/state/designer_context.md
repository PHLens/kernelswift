# Designer Context State

- role_contract_sha256: `d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef`
- context_epoch: `2`
- last_completed_round: `000`
- accepted_kernel: `baseline_adapter.py`
- accepted_report: `rounds/report_000.md`
- recent_three_round_evidence: `Round 000 baseline: 0.231739 ms wall, 147.7526708984375 us/device-call, 15.0 kernels/call; no optimization hypothesis was tested.`
- open_hypotheses: `Three ranked items; H-001 kernel-fusion is selected in rounds/decision_001.md.`
- artifact_read_hashes: `Eleven canonical, contract, evidence, and decision artifacts are recorded below.`

## Current Bottleneck

- Verifier classifies the canonical Round 000 baseline as mixed evidence:
  `baseline_adapter.py` has `0.231739 ms` median wall time,
  `147.7526708984375 us/device-call`, device ratio about `63.76%`, and
  `15.0 kernels/call`. Two gatherTopK plus two bitonic-sort launches consume
  `89.6741943359375 us/call` (`~60.69%` of device time). Evidence:
  `rounds/report_000.md#profiler-evidence`.

## Recent Three-round Evidence

- Round 000 - `baseline`, canonical `baseline_adapter.py`; valid correctness,
  three 200/500 wall samples, and separately scoped forward profile under
  measurement fingerprint `3fe7d50260b3670756d26f003427faccb76e1e79c204b0a500a62d2eb481c809`.
  No candidate/change family exists in this baseline round. Evidence:
  `rounds/report_000.md`.

## Open Hypotheses or Checks

- Rank 1, selected - `kernel-fusion`: replace the fixed softmax/group-top4/
  expert-top8/renorm chain with one per-token Triton-MACA program and cancel the
  exact common softmax denominator. Expected wall gain `15%`; risk `high`
  because 256-lane argmax and tie-ID parity are unproven; evidence
  `rounds/report_000.md#profiler-evidence`; validation cost `high` (actual
  harness correctness, targeted capability/tie probes, wall samples, and scoped
  profiler). Decision: `rounds/decision_001.md`.
- Rank 2, deferred - `selection-algorithm`: if a future accepted fused kernel
  remains selection-bound, test a target-proven partial-selection dataflow
  instead of repeated wide argmax. Conditional expected wall gain `5-10%`;
  risk `high` because no MACA lowering evidence exists and the MLU winner-tree,
  full-sort, gather, and cumsum variants regressed; evidence
  `rounds/report_000.md` plus `references/anti-patterns.md`; validation cost
  `high` (matched primitive/lowering probe before a normative decision).
- Rank 3, deferred - `allocation-reuse`: only if Level 2 host decomposition
  attributes at least the 5% wall gate (`11.58695 us/call`) to compatible
  output allocations, consider per-instance reuse with full cache/concurrency/
  stream rules. Conditional expected wall gain `5-8%`; risk `medium`;
  evidence `rounds/report_000.md` shows a mixed device ratio but does not yet
  attribute host time; validation cost `medium` (targeted host decomposition
  followed by lifecycle and concurrency guardrails).

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `maca/groupedtopk/team-state.md` | `ef2472da1b183faf222f7ed488cf003723e0fea3886e1aa27514a65595e8efb1` | `001` |
| `maca/groupedtopk/project.md` | `6721db4a009b0a539ab70040ab86151ed0cea8990d6d88236ef07abeca0506d3` | `001` |
| `maca/groupedtopk/baseline_adapter.py` | `d92a1be10bf9ad036287ed0f769b195262132893fa78d88fa30f992fbe757827` | `001` |
| `maca/groupedtopk/rounds/report_000.md` | `9b8374ee96d72fa8eed02415440eb778867d9ee0f3d0e8914608695a0c299f00` | `001` |
| `maca/groupedtopk/rounds/decision_001.md` | `c71c970e3bcf6d7984272611627d711ce64b6f3c18d1a057b2aab440c50c173f` | `001` |
| `skills/kernel-opt-loop/prompts/designer.md` | `d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef` | `001` |
| `skills/kernel-opt-loop/prompts/coder_targets/triton_maca.md` | `2cfa08c2664f01e70bb43eec7bb998be836a6a719b17535268a8d6ca18c85540` | `001` |
| `skills/kernel-opt-loop/references/bottleneck-judgment.md` | `664d1e622333559a08419bb39b0b19b04054507a8adb58e3e347ab308c69eae7` | `001` |
| `skills/kernel-opt-loop/references/invariants.md` | `22b53f5f900c8062c445f35be52414b4abba99f8e4893a4dfab996eb1cd8d29c` | `001` |
| `skills/kernel-opt-loop/references/anti-patterns.md` | `aebcdee623024594ad6a19905d626dd7c7ba099d68eba203315229608a40d0c4` | `001` |
| `skills/kernel-opt-loop/references/decision-template.md` | `e25ac46fedb7af63457acdabb92104d6ff2512b9734c309c321dc2a0e1979c50` | `001` |

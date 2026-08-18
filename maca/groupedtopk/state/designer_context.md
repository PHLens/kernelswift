# Designer Context State

- role_contract_sha256: `d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef`
- context_epoch: `3`
- last_completed_round: `001`
- accepted_kernel: `triton_grouped_topk_001.py`
- accepted_report: `rounds/report_001.md`
- recent_three_round_evidence: `Round 000 baseline; Round 001 accepted kernel-fusion with 68.280 us wall, 10.7442822265625 us/device-call, and 1.0 kernel/call.`
- open_hypotheses: `Three ranked items; H-002 fresh-allocation-coalescing is selected in rounds/decision_002.md.`
- artifact_read_hashes: `Thirteen Round 002 canonical, evidence, contract, reference, and decision artifacts are recorded below.`

## Current Bottleneck

- Verifier-backed Round 001 evidence is host-bound: canonical wall is
  `68.280 us/call`, device work is `10.7442822265625 us/call`
  (`15.7356%` device ratio), and the CPU scope is `41.58952 us/call`.
  Within that inclusive CPU scope, `aten::empty` occurs `2.0/call` for
  `10.03988 us/call`; one launch is `4.88562 us/call`. These inclusive
  events may nest/overlap and are not additive. Evidence:
  `rounds/report_001.md` and `state/verifier_context.md`.

## Recent Three-round Evidence

- Round 000 - `baseline`, canonical adapter `0.231739 ms`,
  `147.7526708984375 us/device-call`, `15.0 kernels/call`; evidence
  `rounds/report_000.md`.
- Round 001 - `accepted`, change family `kernel-fusion`; canonical
  `triton_grouped_topk_001.py`, `0.068280 ms`, `69.59021613749428%`
  formal wall improvement, `10.7442822265625 us/device-call`, and
  `1.0 kernel/call`; exact tie parity passed. Evidence:
  `rounds/report_001.md`.

## Open Hypotheses or Checks

- Rank 1, selected - `fresh-allocation-coalescing`: replace two fixed-fast-path
  `torch.empty` calls with one fresh per-call backing and two disjoint typed
  views, without reuse or kernel change. Expected wall gain `6%`; risk
  `medium-high` because MACA CUDA dtype-view support and shared-storage
  lifetime/aliasing require proof; evidence `state/verifier_context.md`;
  validation cost `medium` (storage/lifetime probes, CPU attribution, frozen
  kernel gate, correctness, wall, and scoped profile). Decision:
  `rounds/decision_002.md`.
- Rank 2, deferred - `launcher-overhead`: only if a target-supported direct
  launcher reduction is discovered, attack the observed one launch and
  `4.88562 us/call` inclusive launch event. Conditional expected wall gain
  `5-7%`; risk `high` because `fast_libentry` is unsupported and current
  direct launch is the only proven path; evidence `state/verifier_context.md`
  and `prompts/coder_targets/triton_maca.md`; validation cost `high`.
- Rank 3, deferred - `selection-algorithm`: only if later profiling proves the
  accepted fused kernel itself limits wall time, test a target-proven partial
  selection dataflow. Conditional expected wall gain `5-10%`; risk `high`
  because current device work is only `10.7442822265625 us/call` and no MACA
  alternative lowering evidence exists; evidence `rounds/report_001.md` and
  `references/anti-patterns.md`; validation cost `high`.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `maca/groupedtopk/team-state.md` | `b709e261d98a1558256865808a7aeb2c67cdb4e1735985b43439ea60b8db4e72` | `002` |
| `maca/groupedtopk/project.md` | `503ca8ca604c4cacb2cc9c52c9efedefdf36bad4d59df190da0e453da42ae310` | `002` |
| `maca/groupedtopk/triton_grouped_topk_001.py` | `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384` | `002` |
| `maca/groupedtopk/reference_triton_grouped_topk_001.py` | `70258939973f728858383b832e37069ec6b3d4681200cdb4e70daac42229b2f9` | `002` |
| `maca/groupedtopk/rounds/report_001.md` | `f2866692d8d1c4519e9a2028c7b1d707fcb4f9f945fd856f78139f2dbe2aec4a` | `002` |
| `maca/groupedtopk/state/verifier_context.md` | `283453f35aa90e6ab70f0781fd79c8fb848064ae978354e1b54a19b350abf3aa` | `002` |
| `maca/groupedtopk/rounds/decision_002.md` | `96b175002ab35ebbdeab2e647e1f0acfb150d08ca30792db1c6657a3afea7c55` | `002` |
| `skills/kernel-opt-loop/prompts/designer.md` | `d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef` | `002` |
| `skills/kernel-opt-loop/prompts/coder_targets/triton_maca.md` | `2cfa08c2664f01e70bb43eec7bb998be836a6a719b17535268a8d6ca18c85540` | `002` |
| `skills/kernel-opt-loop/references/bottleneck-judgment.md` | `664d1e622333559a08419bb39b0b19b04054507a8adb58e3e347ab308c69eae7` | `002` |
| `skills/kernel-opt-loop/references/invariants.md` | `22b53f5f900c8062c445f35be52414b4abba99f8e4893a4dfab996eb1cd8d29c` | `002` |
| `skills/kernel-opt-loop/references/anti-patterns.md` | `aebcdee623024594ad6a19905d626dd7c7ba099d68eba203315229608a40d0c4` | `002` |
| `skills/kernel-opt-loop/references/decision-template.md` | `e25ac46fedb7af63457acdabb92104d6ff2512b9734c309c321dc2a0e1979c50` | `002` |

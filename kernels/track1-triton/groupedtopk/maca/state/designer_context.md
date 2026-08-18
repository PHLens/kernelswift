# Designer Context State

- role_contract_sha256: d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef
- context_epoch: 5
- current_round: 004
- last_completed_round: 003
- accepted_kernel: triton_grouped_topk_001.py
- accepted_report: rounds/report_001.md
- current_decision: rounds/decision_004.md
- current_decision_sha256: 5aab9a8df7ab5664cf6a2374d945b06a3a76d60a5fff9e4134a229a75bf0f587
- recent_three_round_evidence: Round 001 accepted kernel fusion; Round 002 fresh-allocation coalescing regressed 13.711567434852972%; Round 003 combined value/index reduction improved only 0.04903708987159917%.
- open_hypotheses: Round 004 tests only fast-path-dispatch-specialization; a third no-improvement reaches the configured stop.
- artifact_read_hashes: Latest Round 004 ledger is recorded below.

## Current Bottleneck

- Canonical Round 001 remains host-bound: authoritative wall is 0.068280 ms,
  device time is 10.7442822265625 us/call in one kernel, and the durable
  inclusive forward CPU scope is 41.58952 us/call.
- A 5% adoption gain requires 3.414 us/call. The canonical fixed predicate
  executes source-visible redundant host work on every benchmark call: two
  tuple(shape) materializations, four hidden metadata eligibility queries, and
  five tensor-device property reads across its guard and output allocations.
- Round 003 removed eight full-width expert value-recovery reductions but
  improved formal wall only 0.04903708987159917%. No Round 003 profiler exists,
  so there is no durable device-time attribution or causal explanation.

## Recent Three-round Evidence

- Round 001, accepted, rounds/report_001.md, kernel-fusion: 0.068280 ms wall,
  10.7442822265625 us/device-call, one kernel/call, and exact targeted ties.
- Round 002, no-improvement, rounds/report_002.md,
  fresh-allocation-coalescing: capability and lifecycle passed, but candidate
  0.081513 ms versus reference 0.071684 ms was -13.711567434852972%;
  canonical remained Round 001 and no profile ran.
- Round 003, no-improvement, rounds/report_003.md,
  value-index-reduction-fusion: source reduction, runtime capability,
  correctness, and ties passed, but candidate 0.067263 ms versus reference
  0.067296 ms was only +0.04903708987159917%; no profile ran.

## Ranked Backlog

| Rank | Status / normalized change_family | Bottleneck and intervention | Expected wall gain | Risk | Evidence pointer | Validation cost |
| ---: | --- | --- | ---: | --- | --- | --- |
| 1 | selected: fast-path-dispatch-specialization | Remove only semantically unnecessary hidden metadata checks and tuple conversions; reuse one invocation-local gating device object while freezing kernel, allocations, launch, config/grad checks, and fallback. | about 6%; adoption requires at least 5% | medium-high: source-count savings may be below 3.414 us/call | rounds/decision_004.md; rounds/report_001.md; canonical source | medium: source equivalence, admitted/fallback semantic probes, wall, targeted CPU/device profile |
| 2 | conditional stop audit: measurement-bound-classification | If Round 004 misses, attribute remaining host time as harness-fixed versus unresolved without changing code. | 0% implementation gain | low semantic risk; deep attribution cost | rounds/report_001.md; rounds/report_003.md; bottleneck-judgment.md | high: Level 2/3 host decomposition |
| 3 | blocked: launcher-overhead-reduction | No supported alternative launcher is available; direct launch is the only proven path. | 0% under current profile | capability blocker: fast_libentry unsupported | triton_maca target profile; state/verifier_context.md | high and not authorized this round |
| 4 | closed: allocation-or-combined-reduction-retry | Do not retry Round 002 allocation coalescing or Round 003 combined reduction without new causal evidence. | below 5% based on formal results | measured no-improvement | rounds/report_002.md; rounds/report_003.md | not applicable |

## Round 004 Design Boundary

- Decision H-004 is host-only. It removes hidden_states width, dtype,
  contiguity, and device-equality eligibility checks after preserving the
  leading-token assertion, compares gating_output.shape directly, and reuses
  one local gating_output.device value.
- Kernel source, grid, T, BLOCK_E, num_warps, direct launch, two independent
  fresh torch.empty allocations, mutable constructor checks, grad predicate,
  canonical fallback, device/current stream, output lifetime, aliasing,
  concurrency, and public behavior remain frozen.
- Required mechanism evidence: shape tuple count 2 to 0; hidden metadata
  eligibility count 4 to 0; tensor device reads 5 to 1; comparable inclusive
  CPU scope falls at least 4.1 us/call; authoritative wall improves at least
  5%. Inclusive CPU events remain nonadditive diagnostics.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
| --- | --- | ---: |
| skills/kernel-opt-loop/prompts/designer.md | d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef | 004 |
| skills/kernel-opt-loop/adapters/codex.md | b77b99e78bbe9cb379ce71deda1b0879bb6c9bd5bc27233e1d53fdd9e74ff151 | 004 |
| skills/kernel-opt-loop/prompts/coder_targets/triton_maca.md | 2cfa08c2664f01e70bb43eec7bb998be836a6a719b17535268a8d6ca18c85540 | 004 |
| skills/kernel-opt-loop/references/bottleneck-judgment.md | 664d1e622333559a08419bb39b0b19b04054507a8adb58e3e347ab308c69eae7 | 004 |
| skills/kernel-opt-loop/references/invariants.md | 22b53f5f900c8062c445f35be52414b4abba99f8e4893a4dfab996eb1cd8d29c | 004 |
| skills/kernel-opt-loop/references/anti-patterns.md | aebcdee623024594ad6a19905d626dd7c7ba099d68eba203315229608a40d0c4 | 004 |
| project.md | 5b97cdfd38c52600dee404fc1319befdc6790973324200345c7e16382af24651 | 004 |
| team-state.md | d015d48eea5e3cd5eac3993f43bb3a04b41337d46f731a31685354bd0324ba85 | 004 |
| triton_grouped_topk_001.py | 9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384 | 004 |
| reference_triton_grouped_topk_001.py | 70258939973f728858383b832e37069ec6b3d4681200cdb4e70daac42229b2f9 | 004 |
| rounds/report_001.md | f2866692d8d1c4519e9a2028c7b1d707fcb4f9f945fd856f78139f2dbe2aec4a | 004 |
| rounds/report_002.md | a5ad9cfe8ead4e1e3cf06ef990ea0817537af4c088219f1eed9a551055426365 | 004 |
| rounds/report_003.md | 6c645acf858745585d4f668546609dc9d3dbc3f7c1b8110a013193f6c89c2fdd | 004 |
| state/coder_context.md | 2cc6de9de80567bbcd2546cf66c660921d4a219bc767c5f591ca6ec0783c26c7 | 004 |
| state/verifier_context.md | 0ed07c489c6f1445cac17bb876bc60c119324c8d8d741dd7fc3bc9ed87ddd545 | 004 |
| rounds/decision_004.md | 5aab9a8df7ab5664cf6a2374d945b06a3a76d60a5fff9e4134a229a75bf0f587 | 004 |

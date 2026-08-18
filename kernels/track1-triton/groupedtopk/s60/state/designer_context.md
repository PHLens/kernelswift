# Designer Context State

- role_contract_sha256: `d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef`
- context_epoch: `2`
- last_completed_round: `005`
- current_design_round: `006`
- accepted_kernel: `triton_grouped_topk_003.py`
- accepted_report: `rounds/report_003.md`
- recent_three_round_evidence: `Round 005 aborted without a candidate because no justified five-percent path existed under the unchanged GCU evidence. Round 004 was valid no-improvement: reference median 0.277370 ms -> candidate median 0.271659 ms, 2.058982586436897%; one candidate-owned stream snapshot per forward and all cache/lifecycle guardrails passed, while one remaining current-stream lookup came from Triton-GCU launch internals outside candidate code. Round 003 accepted: reference median 0.292588 ms -> candidate median 0.273673 ms, 6.464721724746064%; exact-key metadata caching and all guardrails passed. GCU device duration is unavailable in all recorded traces.`
- selected_hypothesis: `H-006 abort: no distinct change family has new Verifier-backed evidence sufficient to justify at least 5% wall improvement; launcher-context-specialization cannot repeat after Round 004's valid no-improvement.`
- evidence_boundary: `No Verifier report, runtime/profile change, GCU device-duration exporter, same-runtime microbenchmark, or accepted candidate changed between the Round 005 abort and Round 006. GCU traces still provide one runtime launch/call and diagnostic runtime-launch duration but no cat=kernel device-duration events. The residual backend stream lookup remains out of candidate scope.`

## Current Bottleneck

- Verifier-backed facts: canonical Round 003 has `1.0` direct GCU runtime launch/call, exact-key metadata cache behavior, output-pool lifecycle safety, and current device/stream guardrails. Round 004 retained these guardrails while changing the candidate-owned stream-query path, then measured only `2.058982586436897%` unrounded median wall improvement.
- Source-backed fact: canonical `triton_grouped_topk_003.py` has one direct `_grouped_topk_kernel[metadata["grid"]](...)` launch per forward with `num_warps=1`; it owns output and metadata cache paths but cannot control Triton-GCU backend-internal launch behavior.
- Classification: measurement-bound for candidate selection under existing evidence. This is not a claim that all remaining wall time is fixed, only that no candidate-owned five-percent intervention is attributable and justified today.

## Recent Three-round Evidence

- Round 005, aborted, `rounds/decision_005.md`, change family `no-change`: no Coder or Verifier artifact; the canonical pointers remained Round 003 because no stable five-percent candidate path was justified.
- Round 004, no-improvement, `rounds/report_004.md`, change family `launcher-context-specialization`: valid wall improvement `2.058982586436897%`, below adoption threshold; candidate-owned stream snapshot count was one, cache/lifecycle/device/stream guardrails passed, one residual stream lookup was backend-internal, runtime launches remained `1.0/call`, and device duration was unavailable.
- Round 003, accepted, `rounds/report_003.md`, change family `host-metadata-specialization`: wall improvement `6.464721724746064%`; metadata exact-key hit/miss/invalidation, output lifetime, concurrency, selected device, and current stream guardrails passed; runtime launches remained `1.0/call`; device duration unavailable.

## Ranked Backlog

| Rank | Hypothesis | Verifier-backed bottleneck or check | Expected wall gain | Risk | Evidence pointer | Validation cost | change_family |
|---:|---|---|---:|---|---|---|---|
| 1 | Abort until new matched evidence identifies a candidate-owned, compressible component with a five-percent path. | No new Verifier evidence since Round 005; Round 004's only candidate-owned stream specialization delivered `2.058982586436897%`. | No qualifying gain | Low: preserves canonical evidence and avoids unsupported changes. | `rounds/decision_005.md#rationale-and-evidence`; `rounds/report_004.md#interleaved-wall-timing` | Decision validation only. | `no-change` |
| 2 | Reconsider expert-selection dataflow only after a matched GCU device-duration exporter or same-runtime microbenchmark establishes an attributable bottleneck and supported lowering. | GCU device duration is unavailable; profile keeps selection alternatives constrained or unknown beyond the accepted implementation. | Unquantified, not currently eligible | High: semantics, ties, reductions, and lowering remain unproven. | `rounds/report_004.md#profiler-evidence`; `prompts/coder_targets/triton_gcu.md` | Matched exporter or microbenchmark, then a new kernel-only decision and full guardrails. | `kernel-selection-dataflow` |
| 3 | Reconsider backend launch/context work only if new Verifier evidence identifies a candidate-owned component distinct from the failed snapshot specialization. | Current residual stream lookup is inside backend internals, outside candidate scope. | Unquantified, not currently eligible | High: candidate cannot safely bypass direct launcher semantics. | `rounds/report_004.md#correctness-and-guardrails`; `prompts/coder_targets/triton_gcu.md` | Targeted same-process decomposition plus a new decision. | `launcher-context-specialization` |

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `skills/kernel-opt-loop/prompts/designer.md` | `d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef` | 006 |
| `skills/kernel-opt-loop/prompts/coder_targets/triton_gcu.md` | `cbc4e4706dfecbab807aaa857dedb374c71629943bbdb549487286cbb6b6eb38` | 006 |
| `skills/kernel-opt-loop/references/decision-template.md` | `e25ac46fedb7af63457acdabb92104d6ff2512b9734c309c321dc2a0e1979c50` | 006 |
| `skills/kernel-opt-loop/references/invariants.md` | `22b53f5f900c8062c445f35be52414b4abba99f8e4893a4dfab996eb1cd8d29c` | 006 |
| `skills/kernel-opt-loop/references/bottleneck-judgment.md` | `664d1e622333559a08419bb39b0b19b04054507a8adb58e3e347ab308c69eae7` | 006 |
| `skills/kernel-opt-loop/references/anti-patterns.md` | `aebcdee623024594ad6a19905d626dd7c7ba099d68eba203315229608a40d0c4` | 006 |
| `s60/groupedtopk/team-state.md` | `bd9e4212cac5660b88a09718ad72fb312efa367440a2e6d3e0f526f267d15b21` | 006 |
| `s60/groupedtopk/project.md` | `b1a4fc2c77e69b5d5fc5da1d8ec1673f75c620cce51323efb9a17c7c49d6e8e1` | 006 |
| `s60/groupedtopk/rounds/decision_005.md` | `cefc9ea3e5a8facd0ace2d9072925918a632c3965a0c3aac68bf6b6553875e59` | 006 |
| `s60/groupedtopk/rounds/report_004.md` | `c515d6f02e6d04bdec5ed34f0c9e0d28de1d152cbfb7b4b7a5ba7b9af21f2032` | 006 |
| `s60/groupedtopk/triton_grouped_topk_003.py` | `3aad6be6422ff08aeb0c6e6be0c92d0645588c7b93429809c525f55c6a6b3e37` | 006 |

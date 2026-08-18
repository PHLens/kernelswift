# Designer Context State

- role_contract_sha256: `d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef`
- context_epoch: `2`
- last_completed_round: `004`
- current_design_round: `005`
- accepted_kernel: `triton_grouped_topk_003.py`
- accepted_report: `rounds/report_003.md`
- recent_three_round_evidence: `Round 004 was valid no-improvement: reference median 0.277370 ms -> candidate median 0.271659 ms, 2.058982586436897%; one candidate-owned stream snapshot per forward and all cache/lifecycle guardrails passed, while one remaining current-stream lookup came from Triton-GCU launch internals outside candidate code. Round 003 accepted: reference median 0.292588 ms -> candidate median 0.273673 ms, 6.464721724746064%; exact-key metadata caching and all guardrails passed. Round 002 accepted: reference median 0.301983 ms -> candidate median 0.274740 ms, 9.02136875254568%; output-pool lifecycle guardrails passed. GCU device duration is unavailable in all recorded traces.`
- selected_hypothesis: `H-005 abort: no stable, Verifier-backed intervention can justify at least 5% wall improvement without repeating the failed launcher-context family or introducing an unproven GCU kernel-dataflow change.`
- evidence_boundary: `GCU traces provide one runtime launch/call and diagnostic runtime-launch duration, but no cat=kernel device-duration events. Round 004 eliminated the remaining candidate-owned duplicate current-stream query and achieved only 2.058982586436897%. The residual backend stream lookup is out of candidate scope. No matched GCU microbenchmark or device-duration exporter supports selection-dataflow work.`

## Current Bottleneck

- Verifier-backed facts: canonical Round 003 has `1.0` direct GCU runtime launch/call, exact-key metadata cache behavior, output-pool lifecycle safety, and current device/stream guardrails. Round 004 retained these guardrails while changing the candidate-owned stream-query path, then measured only `2.058982586436897%` unrounded median wall improvement.
- Source-backed fact: canonical `triton_grouped_topk_003.py` has one direct `_grouped_topk_kernel[metadata["grid"]](...)` launch per forward with `num_warps=1`; it owns output and metadata cache paths but cannot control Triton-GCU backend-internal launch behavior.
- Classification: measurement-bound for new candidate selection. A stop claim is bounded to the absence of a justified candidate intervention under the current evidence, not a claim that all remaining wall time is globally fixed.

## Recent Three-round Evidence

- Round 004, no-improvement, `rounds/report_004.md`, change family `launcher-context-specialization`: valid wall improvement `2.058982586436897%`, below adoption threshold; candidate-owned stream snapshot count was one, cache/lifecycle/device/stream guardrails passed, one residual stream lookup was backend-internal, runtime launches remained `1.0/call`, and device duration was unavailable.
- Round 003, accepted, `rounds/report_003.md`, change family `host-metadata-specialization`: wall improvement `6.464721724746064%`; metadata exact-key hit/miss/invalidation, output lifetime, concurrency, selected device, and current stream guardrails passed; runtime launches remained `1.0/call`; device duration unavailable.
- Round 002, accepted, `rounds/report_002.md`, change family `allocation-reuse`: wall improvement `9.02136875254568%`; sequential compatible forwards reused output storage; retained-output, alias, concurrent-forward, correctness, and stream/device guardrails passed; runtime launches remained `1.0/call`; device duration unavailable.

## Ranked Backlog

| Rank | Hypothesis | Verifier-backed bottleneck or check | Expected wall gain | Risk | Evidence pointer | Validation cost | change_family |
|---:|---|---|---:|---|---|---|---|
| 1 | Abort current campaign direction: no candidate-owned host mechanism remains with a justified five-percent path. | Round 004 removed one candidate-owned current-stream lookup yet measured only `2.058982586436897%`; remaining lookup is backend-internal. | No qualifying gain | Low: preserves canonical evidence and avoids unsupported changes. | `rounds/report_004.md#evidence_for_next_round`; `rounds/report_004.md#interleaved-wall-timing` | Decision validation only. | `no-change` |
| 2 | Reconsider expert-selection dataflow only after a matched GCU device-duration exporter or same-runtime microbenchmark establishes an attributable bottleneck and supported lowering. | GCU device duration is unavailable; profile keeps selection alternatives constrained or unknown beyond the accepted implementation. | Unquantified, not currently eligible | High: semantics, ties, reductions, and lowering remain unproven. | `rounds/report_004.md#profiler-evidence`; `prompts/coder_targets/triton_gcu.md` | Matched exporter or microbenchmark, then a new kernel-only decision and full guardrails. | `kernel-selection-dataflow` |
| 3 | Reconsider backend launch/context work only if new Verifier evidence identifies a candidate-owned component distinct from the failed snapshot specialization. | Current residual stream lookup is inside backend internals, outside candidate scope. | Unquantified, not currently eligible | High: candidate cannot safely bypass direct launcher semantics. | `rounds/report_004.md#correctness-and-guardrails`; `prompts/coder_targets/triton_gcu.md` | Targeted same-process decomposition plus a new decision. | `launcher-context-specialization` |

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `skills/kernel-opt-loop/prompts/designer.md` | `d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef` | 005 |
| `skills/kernel-opt-loop/prompts/coder_targets/triton_gcu.md` | `cbc4e4706dfecbab807aaa857dedb374c71629943bbdb549487286cbb6b6eb38` | 005 |
| `skills/kernel-opt-loop/references/decision-template.md` | `e25ac46fedb7af63457acdabb92104d6ff2512b9734c309c321dc2a0e1979c50` | 005 |
| `skills/kernel-opt-loop/references/invariants.md` | `22b53f5f900c8062c445f35be52414b4abba99f8e4893a4dfab996eb1cd8d29c` | 005 |
| `skills/kernel-opt-loop/references/bottleneck-judgment.md` | `664d1e622333559a08419bb39b0b19b04054507a8adb58e3e347ab308c69eae7` | 005 |
| `skills/kernel-opt-loop/references/anti-patterns.md` | `aebcdee623024594ad6a19905d626dd7c7ba099d68eba203315229608a40d0c4` | 005 |
| `s60/groupedtopk/team-state.md` | `4f497718a1d0a7751da989c5595c4cc10ed6f8f13dc6f12005ad12eb7c8d7a43` | 005 |
| `s60/groupedtopk/project.md` | `e864ea9860a23a3ba6b6ad33285b66d68f092d6f85c79f33529ab9e868e2dd9a` | 005 |
| `s60/groupedtopk/rounds/report_003.md` | `74e8f3623d14535e0699fafb7fe2d920f542d0654ff3c06f25a3e96e18d1a70b` | 005 |
| `s60/groupedtopk/rounds/report_004.md` | `c515d6f02e6d04bdec5ed34f0c9e0d28de1d152cbfb7b4b7a5ba7b9af21f2032` | 005 |
| `s60/groupedtopk/triton_grouped_topk_003.py` | `3aad6be6422ff08aeb0c6e6be0c92d0645588c7b93429809c525f55c6a6b3e37` | 005 |

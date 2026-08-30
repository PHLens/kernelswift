# Designer Context State

- role_contract_sha256: `d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef`
- context_epoch: `2`
- last_completed_round: `007`
- current_design_round: `008`
- accepted_kernel: `triton_grouped_topk_003.py`
- accepted_report: `rounds/report_003.md`
- recent_three_round_evidence: `Round 008 matched GCU probe completed without a candidate result: correctness passed for reference/canonical at 0.282114/0.282032 ms; both profile scopes retained 1.0 runtime launch/call and no cat=kernel device-duration events. Round 007 aborted after SSH execution access was restored but no candidate-owned >=5% path was justified. Round 006 aborted because no distinct change family had new evidence for a five-percent path.`
- selected_hypothesis: `H-008 abort: matched S60 evidence confirms execution and one launch/call but identifies no candidate-owned compressible host component or attributable GCU device bottleneck; no defensible >=5% intervention exists.`
- evidence_boundary: `Round 008 trace SHA-256 1c04a827a50cbb065c1c9943e7c0f5ddf961aeca7f27c06aa2e912f5d2b1a7ec exposes runtime launch events only. Reference runtime-launch is 12.036826171875 us/call and canonical is 11.39935546875 us/call; these are diagnostic and not device time. The single wall pair is measurement-only, not an optimization result.`

## Current Bottleneck

- Verifier-backed facts: canonical Round 003 has one direct GCU runtime launch/call, exact-key metadata cache behavior, output-pool lifecycle safety, and current device/stream guardrails. Round 004's distinct launcher-context candidate reached only `2.058982586436897%`, below threshold.
- Round 008 matched S60 probe facts: correctness PASS; reference adapter `0.282114 ms`, canonical `0.282032 ms`; profile scopes each have `1.0` `topsModuleLaunchKernel` per call; GCU device duration unavailable.
- Classification: measurement-bound for candidate selection. The probe confirms the execution path but does not establish a candidate-owned five-percent mechanism.

## Recent Three-round Evidence

- Round 008, named measurement probe, `rounds/round_status_008.md`: correctness PASS; reference/canonical raw pair `0.282114/0.282032 ms`; runtime launch `1.0/call`; device duration unavailable; no terminal candidate result.
- Round 007, aborted, `rounds/decision_007.md`: SSH execution availability was environment evidence only; no candidate-owned >=5% path.
- Round 006, aborted, `rounds/decision_006.md`: no distinct change family had new Verifier-backed evidence for a five-percent path.

## Ranked Backlog

| Rank | Hypothesis | Verifier-backed bottleneck or check | Expected wall gain | Risk | Evidence pointer | Validation cost | change_family |
|---:|---|---|---:|---|---|---|---|
| 1 | Abort current campaign direction until a candidate-owned compressible mechanism is identified. | Round 008 has one launch/call, no device duration, and no distinct host attribution; Round 004's only candidate-owned context specialization reached 2.058982586436897%. | No qualifying gain | Low: preserves canonical evidence. | `rounds/round_status_008.md`; `rounds/report_004.md#interleaved-wall-timing` | Decision validation only. | `no-change` |
| 2 | Reconsider expert-selection dataflow only after a matched GCU device-duration exporter or a same-runtime microbenchmark establishes an attributable bottleneck and supported lowering/tie behavior. | GCU device duration remains unavailable; alternate reductions and `tl.argmax` tie behavior are constrained or unknown. | Unquantified, not currently eligible | High: exact top-k order and semantics. | `s60/groupedtopk/state/verifier_context.md`; `prompts/coder_targets/triton_gcu.md` | New matched evidence, then candidate and full guardrails. | `kernel-selection-dataflow` |
| 3 | Reconsider backend launcher/context work only if a new probe identifies a candidate-owned component distinct from Round 004. | Current trace shows one launch/call; residual backend lookup is outside candidate scope. | Unquantified, not currently eligible | High: stream ownership and direct-launch semantics. | `rounds/report_004.md`; `rounds/round_status_008.md` | New targeted decomposition, then candidate and full guardrails. | `launcher-context-specialization` |

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `skills/kernel-opt-loop/prompts/designer.md` | `d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef` | 008 |
| `skills/kernel-opt-loop/prompts/coder_targets/triton_gcu.md` | `cbc4e4706dfecbab807aaa857dedb374c71629943bbdb549487286cbb6b6eb38` | 008 |
| `skills/kernel-opt-loop/references/decision-template.md` | `e25ac46fedb7af63457acdabb92104d6ff2512b9734c309c321dc2a0e1979c50` | 008 |
| `skills/kernel-opt-loop/references/invariants.md` | `22b53f5f900c8062c445f35be52414b4abba99f8e4893a4dfab996eb1cd8d29c` | 008 |
| `skills/kernel-opt-loop/references/bottleneck-judgment.md` | `664d1e622333559a08419bb39b0b19b04054507a8adb58e3e347ab308c69eae7` | 008 |
| `skills/kernel-opt-loop/references/anti-patterns.md` | `aebcdee623024594ad6a19905d626dd7c7ba099d68eba203315229608a40d0c4` | 008 |
| `s60/groupedtopk/rounds/round_status_008.md` | `1c04a827a50cbb065c1c9943e7c0f5ddf961aeca7f27c06aa2e912f5d2b1a7ec` | 008 |
| `s60/groupedtopk/state/verifier_context.md` | `pending-final-read` | 008 |
| `s60/groupedtopk/rounds/decision_008.md` | `pending-final-read` | 008 |
| `s60/groupedtopk/triton_grouped_topk_003.py` | `3aad6be6422ff08aeb0c6e6be0c92d0645588c7b93429809c525f55c6a6b3e37` | 008 |

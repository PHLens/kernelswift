# Designer Context State

- role_contract_sha256: `d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef`
- context_epoch: `2`
- last_completed_round: `006`
- current_design_round: `007`
- accepted_kernel: `triton_grouped_topk_003.py`
- accepted_report: `rounds/report_003.md`
- recent_three_round_evidence: `Round 006 aborted without a candidate because no distinct change family had new Verifier-backed evidence for a five-percent path. Round 005 aborted for the same unchanged evidence boundary. Round 004 was valid no-improvement: reference median 0.277370 ms -> candidate median 0.271659 ms, 2.058982586436897%; one candidate-owned stream snapshot per forward and all cache/lifecycle guardrails passed, while one remaining current-stream lookup came from Triton-GCU launch internals outside candidate code. Round 003 accepted: reference median 0.292588 ms -> candidate median 0.273673 ms, 6.464721724746064%; exact-key metadata caching and all guardrails passed. GCU device duration is unavailable in all recorded traces.`
- selected_hypothesis: `H-007 abort: SSH access now permits future Verifier probes but supplies no performance or lowering evidence; no distinct intervention currently justifies >=5% expected wall improvement.`
- evidence_boundary: `New SSH fact is execution availability only. Local and remote source/harness hashes match for the canonical and Round 004 files, but no new Verifier report, profiler evidence, device-duration exporter, same-runtime microbenchmark, or runtime fingerprint change is present. GCU traces still expose runtime-launch events only, and runtime-launch duration is not device time.`

## Current Bottleneck

- Verifier-backed facts: canonical Round 003 has `1.0` direct GCU runtime launch/call, exact-key metadata cache behavior, output-pool lifecycle safety, and current device/stream guardrails. Round 004 retained these guardrails while changing the candidate-owned stream-query path, then measured only `2.058982586436897%` unrounded median wall improvement.
- Source-backed fact: canonical `triton_grouped_topk_003.py` has one direct `_grouped_topk_kernel[metadata["grid"]](...)` launch per forward with `num_warps=1`; it owns output and metadata cache paths but cannot control Triton-GCU backend-internal launch behavior.
- Environment fact: password-based SSH to the recorded S60 host is now working, and remote execution copies hash-match local `base.py`, `auto_bench.py`, canonical Round 003, and Round 004 reference/candidate. This lowers future validation cost only.
- Classification: measurement-bound for candidate selection under current evidence. This does not claim all remaining wall time is fixed; it records that no candidate-owned five-percent intervention is attributable and justified before the new probe runs.

## Recent Three-round Evidence

- Round 006, aborted, `rounds/decision_006.md`, change family `no-change`: no Coder or Verifier artifact; canonical pointers remained Round 003 because no distinct five-percent candidate path was justified.
- Round 005, aborted, `rounds/decision_005.md`, change family `no-change`: no Coder or Verifier artifact; the canonical pointers remained Round 003 because no stable five-percent candidate path was justified.
- Round 004, no-improvement, `rounds/report_004.md`, change family `launcher-context-specialization`: valid wall improvement `2.058982586436897%`, below adoption threshold; candidate-owned stream snapshot count was one, cache/lifecycle/device/stream guardrails passed, one residual stream lookup was backend-internal, runtime launches remained `1.0/call`, and device duration was unavailable.

## Ranked Backlog

| Rank | Hypothesis | Verifier-backed bottleneck or check | Expected wall gain | Risk | Evidence pointer | Validation cost | change_family |
|---:|---|---|---:|---|---|---|---|
| 1 | Run a matched GCU measurement probe first; only then consider a proceeding kernel-selection or other distinct decision. | SSH and hash-match copies are now available, but no performance, lowering, or device-duration observation exists yet. | Unknown until probed; not a Round 007 candidate | Low design risk; no source change. | Orchestrator-provided SSH fact; `project.md#measurement-fingerprint` | Future Verifier remote harness/profiler or targeted microbenchmark, preserving measurement exclusivity. | `measurement-boundary-probe` |
| 2 | Reconsider expert-selection dataflow only after the matched GCU probe identifies an attributable candidate kernel bottleneck and establishes supported lowering/tie behavior. | Current profile has no GCU device duration; `tl.argmax` tie behavior and alternate reduction/dataflow primitives remain constrained or unknown. | Unquantified, not currently eligible | High: exact top-k IDs/order, numerical semantics, compiler resource use, and GCU support are unproven. | `prompts/coder_targets/triton_gcu.md`; `rounds/report_004.md#profiler-evidence` | Matched GCU evidence, then candidate implementation and full correctness/profile/paired timing gates. | `kernel-selection-dataflow` |
| 3 | Reconsider backend launch/context work only if new Verifier evidence identifies a candidate-owned component distinct from the failed snapshot specialization. | Round 004 removed one candidate-owned lookup and reached only `2.058982586436897%`; residual lookup is backend-internal. | Unquantified, not currently eligible | High: direct launcher semantics and stream ownership must remain intact. | `rounds/report_004.md#correctness-and-guardrails` | New targeted decomposition plus a distinct causal mechanism. | `launcher-context-specialization` |

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `skills/kernel-opt-loop/prompts/designer.md` | `d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef` | 007 |
| `skills/kernel-opt-loop/prompts/coder_targets/triton_gcu.md` | `cbc4e4706dfecbab807aaa857dedb374c71629943bbdb549487286cbb6b6eb38` | 007 |
| `skills/kernel-opt-loop/references/decision-template.md` | `e25ac46fedb7af63457acdabb92104d6ff2512b9734c309c321dc2a0e1979c50` | 007 |
| `skills/kernel-opt-loop/references/invariants.md` | `22b53f5f900c8062c445f35be52414b4abba99f8e4893a4dfab996eb1cd8d29c` | 007 |
| `skills/kernel-opt-loop/references/bottleneck-judgment.md` | `664d1e622333559a08419bb39b0b19b04054507a8adb58e3e347ab308c69eae7` | 007 |
| `skills/kernel-opt-loop/references/anti-patterns.md` | `aebcdee623024594ad6a19905d626dd7c7ba099d68eba203315229608a40d0c4` | 007 |
| `s60/groupedtopk/team-state.md` | `a23a5d5971da09f0fa6a45ad319fa9f37b384816eaadf0d705992848f28711a6` | 007 |
| `s60/groupedtopk/project.md` | `b1a4fc2c77e69b5d5fc5da1d8ec1673f75c620cce51323efb9a17c7c49d6e8e1` | 007 |
| `s60/groupedtopk/rounds/decision_006.md` | `f97a104a36a423db2191593c77b61203083310bc8709013a57f51a44414f647d` | 007 |
| `s60/groupedtopk/rounds/report_004.md` | `c515d6f02e6d04bdec5ed34f0c9e0d28de1d152cbfb7b4b7a5ba7b9af21f2032` | 007 |
| `s60/groupedtopk/triton_grouped_topk_003.py` | `3aad6be6422ff08aeb0c6e6be0c92d0645588c7b93429809c525f55c6a6b3e37` | 007 |

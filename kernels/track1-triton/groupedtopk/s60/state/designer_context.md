# Designer Context State

- role_contract_sha256: `d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef`
- context_epoch: `2`
- last_completed_round: `003`
- current_design_round: `004`
- accepted_kernel: `triton_grouped_topk_003.py`
- accepted_report: `rounds/report_003.md`
- recent_three_round_evidence: `Round 003 accepted; reference median 0.292588 ms -> candidate median 0.273673 ms; wall improvement 6.464721724746064%; metadata hit, miss, invalidation, output lifetime, and concurrency guardrails passed; runtime launches remained 1.0/call; GCU device duration unavailable. Round 002 accepted; reference median 0.301983 ms -> candidate median 0.274740 ms; wall improvement 9.02136875254568%; output-pool lifecycle guardrails passed; runtime launches remained 1.0/call; GCU device duration unavailable. Round 001 accepted; wall improvement 39.08693002628853%; runtime launches fell from 12.0 to 1.0/call; GCU device duration unavailable.`
- selected_hypothesis: `H-004 launcher-context-specialization; capture the caller GCU device/current-stream identity once per forward, pass the exact snapshot through both output-pool and metadata-cache lookup, and reuse the existing immutable direct-launch setup without changing the kernel, grid, constexprs, num_warps=1, output pool, metadata cache, or device/stream semantics.`
- evidence_boundary: `Round 003 verifies one runtime launch/call and cache/lifecycle guardrails but has no GCU device-duration event or direct host-time attribution. H-004 is a falsifiable host-path hypothesis only; it requires targeted same-process launcher/context decomposition, authoritative paired wall timing, and all existing guardrails. Runtime-launch duration is diagnostic only and cannot be used as device time or device_ratio.`

## Current Bottleneck

- Verifier-backed facts: Round 003 improved the accepted-reference median from `0.292588 ms` to `0.273673 ms` (`6.464721724746064%`) with `1.0` GCU runtime launch per call in both scopes. Metadata exact-key hit/miss/invalidation, instance ownership, output lifetime, concurrency, selected device, current stream, and one direct launch with `num_warps=1` passed. GCU device duration remains unavailable.
- Source-backed fact: canonical `triton_grouped_topk_003.py` calls `torch.gcu.current_stream(device)` in `_launch_metadata` at lines 139-176 and again in `_output_key` at lines 120-137; `forward` invokes both before packaging the unchanged direct launch at lines 227-255.
- Round 004 classification: launcher/context specialization is a host-bound hypothesis, not a measured host-time claim. Adoption still requires at least `5%` unrounded median wall improvement, correctness, targeted host evidence, unchanged one-launch conformance, and every cache/lifecycle/device/stream guardrail.

## Recent Three-round Evidence

- Round 003, accepted, `rounds/report_003.md`, change family `host-metadata-specialization`: wall improvement `6.464721724746064%`; metadata exact-key hit/miss/invalidation, output lifetime, concurrency, selected device, and current stream guardrails passed; runtime launches remained `1.0/call`; device duration unavailable.
- Round 002, accepted, `rounds/report_002.md`, change family `allocation-reuse`: wall improvement `9.02136875254568%`; sequential compatible forwards reused output storage; retained-output, alias, concurrent-forward, correctness, and stream/device guardrails passed; runtime launches remained `1.0/call`; device duration unavailable.
- Round 001, accepted, `rounds/report_001.md`, change family `kernel-fusion`: wall improvement `39.08693002628853%`; runtime launches fell from `12.0` to `1.0` per call; device duration unavailable.

## Ranked Backlog

| Rank | Hypothesis | Verifier-backed bottleneck or check | Expected wall gain | Risk | Evidence pointer | Validation cost | change_family |
|---:|---|---|---:|---|---|---|---|
| 1 | Snapshot current GCU stream/device identity once per forward and use that exact key material for both existing cache paths while reusing immutable direct-launch setup. Selected for Round 004. | Canonical source independently obtains current-stream metadata in `_launch_metadata` and `_output_key`; Round 003 preserves a one-launch path but has no host-time attribution. | 5% hypothesis only | Medium: exact stream/device behavior, cache invalidation, pool lifetime, and concurrent lookup must remain correct. | `triton_grouped_topk_003.py:120-176,227-255`; `rounds/report_003.md#evidence_for_next_round` | Targeted stream-query/setup decomposition, metadata/output cache hit/miss/invalidation, retained-output/alias/concurrency, device/stream, correctness, profile, and paired wall tests. | `launcher-context-specialization` |
| 2 | Classify the remaining host path as harness-fixed only after targeted same-process decomposition fails to find a compressible launcher/context component. | Runtime launch count is already `1.0/call`; device duration is unavailable, so neither trace proves host time fixed. | Stop hypothesis only | High: an early stop would be unsupported without Level 2 evidence. | `rounds/report_003.md#profiler-evidence`; `references/bottleneck-judgment.md#compressible-versus-fixed-host-time` | Matched decomposition with unchanged call counts and a documented fixed-versus-compressible conclusion. | `measurement-bound-classification` |
| 3 | Revisit expert-selection dataflow only after a matched GCU device-duration exporter or a same-runtime microbenchmark establishes attributable device evidence. | The target profile marks GCU device duration unavailable; MLU selection anti-patterns are not transferable proof. | 5% hypothesis only | High: current evidence does not justify kernel change. | `rounds/report_003.md#evidence_for_next_round`; `skills/kernel-opt-loop/prompts/coder_targets/triton_gcu.md` | Matched exporter or isolated GCU microbenchmark, then a new kernel-only decision and full guardrails. | `kernel-selection-dataflow` |

## Round 004 Host Constraints

- Capture `device.type`, `device.index`, and `current_stream(...).stream_id` once per forward. Pass that immutable snapshot to the output-pool and metadata-cache paths; do not perform a second current-stream query in either path.
- Preserve existing exact cache compatibility. Metadata keying covers input shape/dtype, routing configuration, selected device, current stream, grid, constexpr values, and `num_warps=1`; output-pool keying retains its exact compatible output dimensions/dtypes/device/stream requirements.
- If stream metadata cannot be obtained, do not insert or reuse any cache entry that depends on it. Preserve the existing uncached direct-launch-compatible behavior instead of using an unkeyed fallback.
- Keep `_grouped_topk_kernel`, direct `kernel[(grid,)](...)` launch, `grid=(tokens,)`, all constexpr meanings, and `num_warps=1` unchanged. Do not use `fast_libentry`, device/stream/context changes, waits, or synchronization.
- Do not change output-pool allocation, lease, storage-lifetime, alias, or ownership behavior. Do not alter metadata-cache ownership/lifetime except reusing existing immutable launch setup from an exact hit.
- Stack-local forward context is never shared. Existing ModelNew locks continue to protect metadata initialization and output-pool leases; separate ModelNew instances never share state.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `skills/kernel-opt-loop/prompts/designer.md` | `d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef` | 004 |
| `skills/kernel-opt-loop/prompts/coder_targets/triton_gcu.md` | `cbc4e4706dfecbab807aaa857dedb374c71629943bbdb549487286cbb6b6eb38` | 004 |
| `skills/kernel-opt-loop/references/decision-template.md` | `e25ac46fedb7af63457acdabb92104d6ff2512b9734c309c321dc2a0e1979c50` | 004 |
| `skills/kernel-opt-loop/references/invariants.md` | `22b53f5f900c8062c445f35be52414b4abba99f8e4893a4dfab996eb1cd8d29c` | 004 |
| `skills/kernel-opt-loop/references/bottleneck-judgment.md` | `664d1e622333559a08419bb39b0b19b04054507a8adb58e3e347ab308c69eae7` | 004 |
| `skills/kernel-opt-loop/references/anti-patterns.md` | `aebcdee623024594ad6a19905d626dd7c7ba099d68eba203315229608a40d0c4` | 004 |
| `s60/groupedtopk/team-state.md` | `1a8e26986cbb51626b4a62f6fa754f146b1a86545feb0aab151cc8529a3bc100` | 004 |
| `s60/groupedtopk/project.md` | `e864ea9860a23a3ba6b6ad33285b66d68f092d6f85c79f33529ab9e868e2dd9a` | 004 |
| `s60/groupedtopk/rounds/report_003.md` | `74e8f3623d14535e0699fafb7fe2d920f542d0654ff3c06f25a3e96e18d1a70b` | 004 |
| `s60/groupedtopk/state/designer_context.md` before Round 004 update | `8619382ef059f320d40bc0b623fe8327ac75525b6dc783a74692c022e32d2710` | 004 |
| `s60/groupedtopk/triton_grouped_topk_003.py` | `3aad6be6422ff08aeb0c6e6be0c92d0645588c7b93429809c525f55c6a6b3e37` | 004 |

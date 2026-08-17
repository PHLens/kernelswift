# Designer Context State

- role_contract_sha256: `d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef`
- context_epoch: `2`
- last_completed_round: `002`
- current_design_round: `003`
- accepted_kernel: `triton_grouped_topk_002.py`
- accepted_report: `rounds/report_002.md`
- recent_three_round_evidence: `Round 002 accepted; reference median 0.301983 ms -> candidate median 0.274740 ms; wall improvement 9.02136875254568%; runtime launches remained 1.0/call; GCU device duration unavailable. Round 001 accepted; wall improvement 39.08693002628853%; runtime launches 12.0 -> 1.0/call; GCU device duration unavailable.`
- selected_hypothesis: `H-003 host-metadata-specialization; cache exact-shape/device/stream-compatible block_e, epg, and launch configuration in ModelNew host metadata state without changing the Triton kernel body, grid, constexprs, num_warps, public contract, or device/stream semantics.`
- evidence_boundary: `Round 002 report and accepted source identify repeated host computation of triton.next_power_of_2(experts), expert/group arithmetic, and launch argument construction; no host-time attribution exists. The intervention remains a falsifiable host-bound hypothesis and requires targeted same-process decomposition plus authoritative paired wall timing.`
- reference_adapter: `reference_triton_grouped_topk_001.py`, SHA-256 `800ec0080e66589f6dfcf3a71ee79f08e01be68f145b4cb3c6c6b50dd7c03027`; this is the Round 002 accepted-reference adapter, not the Round 003 canonical source. Round 003 reference is `triton_grouped_topk_002.py`.

## Current Bottleneck

- Verifier-backed facts: the accepted Round 002 candidate has a `0.274740 ms` median wall time against a `0.301983 ms` accepted-reference median, with `9.02136875254568%` improvement; both scopes emit `1.0` GCU runtime launch per call and device duration is unavailable.
- Source-backed fact: `triton_grouped_topk_002.py:192-209` computes `block_e`, derives `epg`, and constructs the launch argument bundle on every forward. The output pool is already instance-owned and lifecycle-guarded; it is outside this round's change boundary.
- Round 003 classification: host metadata specialization is a host-bound hypothesis, not a measured host-time claim. Adoption still requires at least 5% unrounded median wall improvement, correctness, targeted metadata evidence, and every cache/lifecycle/device/stream guardrail.

## Recent Three-round Evidence

- Round 002, accepted, `rounds/report_002.md`, change family `allocation-reuse`: wall improvement `9.02136875254568%`; sequential compatible forwards reused output storage; retained-output, alias, concurrent-forward, correctness, and stream/device guardrails passed; runtime launch count stayed `1.0/call`; device duration unavailable.
- Round 001, accepted, `rounds/report_001.md`, change family `kernel-fusion`: wall improvement `39.08693002628853%`; runtime launches fell from `12.0` to `1.0` per call; device duration unavailable; output allocation was explicitly left untested.

## Ranked Backlog

| Rank | Hypothesis | Verifier-backed bottleneck or check | Expected wall gain | Risk | Evidence pointer | Validation cost | change_family |
|---:|---|---|---:|---|---|---|---|
| 1 | Cache exact-shape/device/stream-compatible `block_e`, `epg`, and launch metadata in private ModelNew state. Selected for Round 003. | Accepted source recomputes shape-derived metadata and launch arguments each forward; no host-time attribution yet. | 5% hypothesis only | Medium: exact invalidation, stream identity, concurrency, and attribution must be proven. | `triton_grouped_topk_002.py:192-209`; `rounds/report_002.md#evidence_for_next_round` | Targeted metadata hit/miss and host decomposition, correctness, stream/device, concurrency, and paired wall tests. | `host-metadata-specialization` |
| 2 | Specialize launcher/context handling only if targeted decomposition identifies a compressible component separate from metadata setup. | Runtime launch count is already `1.0/call`; remaining host time is not attributed. | 5% hypothesis only | High: may be harness-fixed or runtime-dependent; no context semantics may change. | `rounds/report_002.md#profiler-evidence`; `project.md#measurement-regime` | Same-process host decomposition before selection, then full guardrails. | `launcher-context-specialization` |
| 3 | Revisit expert-selection dataflow only after attributable GCU device evidence or a matched same-runtime microbenchmark. | GCU device duration is unavailable; MLU selection anti-patterns are not transferable proof for GCU. | 5% hypothesis only | High: no current device bottleneck attribution and kernel changes are forbidden for Round 003. | `rounds/report_002.md#profiler-evidence`; `references/anti-patterns.md` | Matched exporter or isolated GCU microbenchmark, then full verification. | `kernel-selection-dataflow` |

## Round 003 Host Constraints

- State is private to one `ModelNew` instance and contains immutable metadata only; no output or device-buffer ownership is added.
- Cache compatibility includes exact gating shape, relevant dtype and routing configuration, selected GCU device, current stream identity, `block_e`, `epg`, grid, constexpr values, and `num_warps=1`.
- A key miss creates a separate metadata entry; it never mutates or reuses an incompatible entry. Existing output-pool lease and storage-lifetime behavior remains unchanged.
- Lookup and insertion are synchronized per model instance. Separate instances never share metadata state; concurrent forwards cannot race entry initialization.
- The kernel body, grid, constexprs, `num_warps`, public contract, output semantics, device placement, and stream ownership are unchanged.
- No synchronize, stream switch, device switch, cross-stream wait, or altered device-context operation is allowed. An unprovable compatibility or concurrency property is a capability miss.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `skills/kernel-opt-loop/prompts/designer.md` | `d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef` | 003 |
| `skills/kernel-opt-loop/prompts/coder_targets/triton_gcu.md` | `cbc4e4706dfecbab807aaa857dedb374c71629943bbdb549487286cbb6b6eb38` | 003 |
| `skills/kernel-opt-loop/references/decision-template.md` | `e25ac46fedb7af63457acdabb92104d6ff2512b9734c309c321dc2a0e1979c50` | 003 |
| `skills/kernel-opt-loop/references/invariants.md` | `22b53f5f900c8062c445f35be52414b4abba99f8e4893a4dfab996eb1cd8d29c` | 003 |
| `skills/kernel-opt-loop/references/bottleneck-judgment.md` | `664d1e622333559a08419bb39b0b19b04054507a8adb58e3e347ab308c69eae7` | 003 |
| `skills/kernel-opt-loop/references/anti-patterns.md` | `aebcdee623024594ad6a19905d626dd7c7ba099d68eba203315229608a40d0c4` | 003 |
| `skills/kernel-opt-loop/scripts/validate_decision.py` | `not part of requested role context` | 003 |
| `s60/groupedtopk/team-state.md` | `6834142d17a8f163151daac9cd7d315c7bd76cf5e846e3fb8e09e55084eb5f40` | 003 |
| `s60/groupedtopk/project.md` | `ba83cd2d48bb460b193a6d14ebccdd29623bf45692d0438909e77cbf68d4a5a8` | 003 |
| `s60/groupedtopk/rounds/report_002.md` | `0e07b45c93b470b344b93463474708dbbc51c4f6bcd16d75b00b68182a30cbd1` | 003 |
| `s60/groupedtopk/triton_grouped_topk_002.py` | `90d7b09569d1d155c8e44e1626f2c0f3b3f41e0919a8a9e5b76719e874b17ce3` | 003 |
| `s60/groupedtopk/rounds/decision_003.md` | `2f90569b0cbf786f217cd45fac38c51990d7b5c041dc1f9a5ac6e5ac38129594` | 003 |

# Designer Context State

- role_contract_sha256: `d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef`
- context_epoch: `2`
- last_completed_round: `001`
- current_design_round: `002`
- accepted_kernel: `triton_grouped_topk_001.py`
- accepted_report: `rounds/report_001.md`
- recent_three_round_evidence: `Round 001 accepted; wall median 0.449626 ms -> 0.273881 ms; runtime launches 12.0 -> 1.0 per call; GCU device duration unavailable.`
- selected_hypothesis: `H-002 allocation-reuse; replace two per-forward output allocations with a concurrency-safe, device-and-stream-keyed per-instance lease pool.`
- evidence_boundary: `The source and accepted report prove two allocations remain; they do not prove a host/device ratio or measured allocation cost.`
- reference_adapter: `reference_triton_grouped_topk_001.py`, SHA-256 `800ec0080e66589f6dfcf3a71ee79f08e01be68f145b4cb3c6c6b50dd7c03027`; verified identical to canonical except `ModelNew` renamed to `Model` for the harness v0 entry.

## Current Bottleneck

- Verifier-backed facts: the accepted fused candidate has one GCU runtime launch
  per call, a `0.273881 ms` wall median, and no available GCU device duration.
- Source-backed fact: `ModelNew.forward` executes two `torch.empty` output
  allocations per call.
- Round 002 classification: output allocation is a host-bound hypothesis, not a
  measured host-time claim. Adoption still requires at least 5% unrounded median
  wall improvement and every lifecycle guardrail.

## Recent Three-round Evidence

- Round 001, accepted, `rounds/report_001.md`, change family `kernel-fusion`:
  wall improvement `39.08693002628853%`; runtime launches `12.0 -> 1.0` per call;
  GCU device duration unavailable; output allocation explicitly left untested.

## Ranked Backlog

| Rank | Hypothesis | Verifier-backed bottleneck or check | Expected wall gain | Risk | Evidence pointer | Validation cost | change_family |
|---:|---|---|---:|---|---|---|---|
| 1 | Lease and reuse compatible output pairs without reusing live storage. Selected for Round 002. | Two output allocations remain in canonical source; Round 001 names lifecycle as untested. | 6% | High: lifetime, alias, stream, and concurrent-forward safety must be proven. | `triton_grouped_topk_001.py:120-125`; `rounds/report_001.md#evidence_for_next_round` | Targeted allocation count plus retained-output, alias, concurrent-forward, correctness, and paired wall tests. | `allocation-reuse` |
| 2 | Cache invariant host launch metadata such as block size and validated static configuration per model/key. | Canonical source recomputes shape-derived host metadata; no Verifier timing attribution exists. | 5% hypothesis only | Medium: likely too small and must not hide incompatible shapes. | `triton_grouped_topk_001.py:118-139` | Host decomposition and paired wall tests before selection. | `host-metadata-specialization` |
| 3 | Revisit expert-selection kernel dataflow only after attributable GCU device evidence or a same-runtime microbenchmark. | Device duration is unavailable; MLU selection anti-patterns are not transferable proof for GCU. | 5% hypothesis only | High: no current device bottleneck attribution and several MLU shapes regressed. | `rounds/report_001.md#profiler-evidence`; `references/anti-patterns.md` | New matched profiler exporter or isolated same-runtime microbench, then full verification. | `kernel-selection-dataflow` |

## Round 002 Host Constraints

- State is private to one `ModelNew` instance and lives no longer than that model.
- Cache compatibility includes output shape, both output dtypes, contiguous layout,
  selected GCU device, and current stream identity.
- One distinct lease is required per live or in-flight forward. A miss allocates;
  it does not serialize kernels or overwrite storage still observable by callers.
- Separate model instances never share state. No global cache is allowed.
- Current device and stream are preserved; no explicit synchronize, stream switch,
  device-context switch, cross-stream wait, or new device operation is allowed.
- Inability to prove storage/alias lifetime or stream identity is a
  `capability-miss`, never permission for unsafe single-buffer reuse.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `skills/kernel-opt-loop/prompts/designer.md` | `d32060e9953982eca29c19d6ed7469c2fb5c06ea686385be5da10219981addef` | 002 |
| `skills/kernel-opt-loop/prompts/coder_targets/triton_gcu.md` | `cbc4e4706dfecbab807aaa857dedb374c71629943bbdb549487286cbb6b6eb38` | 002 |
| `skills/kernel-opt-loop/references/decision-template.md` | `e25ac46fedb7af63457acdabb92104d6ff2512b9734c309c321dc2a0e1979c50` | 002 |
| `skills/kernel-opt-loop/references/invariants.md` | `22b53f5f900c8062c445f35be52414b4abba99f8e4893a4dfab996eb1cd8d29c` | 002 |
| `skills/kernel-opt-loop/references/bottleneck-judgment.md` | `664d1e622333559a08419bb39b0b19b04054507a8adb58e3e347ab308c69eae7` | 002 |
| `skills/kernel-opt-loop/references/anti-patterns.md` | `aebcdee623024594ad6a19905d626dd7c7ba099d68eba203315229608a40d0c4` | 002 |
| `team-state.md` | `339642c282b3bca5cef4bce8adeb5ff28dfc232b1f009fd3b4c1f3e426a331e8` | 002 |
| `project.md` | `c56e64d02ac49c8833687f49d0a311c567d99a22b91e7f939aa4b485afb57dab` | 002 |
| `triton_grouped_topk_001.py` | `f42ff6b47b28996199bbe9b8df0a181db2834be99473453f3eea35df51df693e` | 002 |
| `reference_triton_grouped_topk_001.py` | `800ec0080e66589f6dfcf3a71ee79f08e01be68f145b4cb3c6c6b50dd7c03027` | 002 |
| `rounds/report_001.md` | `efe1670ec48f5920b593c06267081c565a156bd88796969361a2c68243a6a610` | 002 |

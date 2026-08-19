# Decision 003

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"003","reference_implementation":"triton_fused_moe_002.py","reference_report":"rounds/report_002.md","language":"triton","backend":"cuda","target_profile":"triton_cuda","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-change"}
```

## Optimization Intent

```json
{"bottleneck_class":"none","intervention":"no stable intervention clears the 5% adoption threshold: the operator is strongly host/launch-bound (device_ratio 0.2854) with the remaining device time dominated by the untouchable torch.topk (~39.4 us/call) and the already-fused single tl.dot expert kernel (55.8 us/call), and the residual small overheads (w1/w2 fp16 cast, renormalize sum/div, out zero-init) are each below 5% wall and their device-time savings do not translate to wall time under the launch-bound regime","allowed_changes":[],"invariants":["ModelNew public contract (num_experts=8, top_k=2, hidden_size=128, intermediate_size=64, renormalize=True)","forward signature (hidden_states,router_logits)->out[83,128] fp16","torch.topk(scores,2,dim=-1) descending-value / ascending-index tie order preserved bit-exactly","routing (fp32 softmax), GEMM contraction dims (128/64), SiLU, and weighted-sum reduction semantics","benchmark semantics (harness seed/synchronization and measurement fingerprint)"],"expected_wall_improvement_pct":0.0}
```

## Unified Sketch

N/A: aborted

## Host Plan

```json
{"applicability":"not-applicable","reason":"aborted"}
```

## Evaluation Contract

```json
{"applicability":"not-applicable","reason":"aborted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`. No remaining entry describes a valid ≥5% path for this operator: the catalog's selection-network failures (winner-tree Entry 011, full sort-32/64 Entry 012, dynamic `tl.gather` Entry 013, cumsum compaction Entry 016) concern reimplementing top-k selection, which is explicitly forbidden here by the tie-semantics invariant (`torch.topk` is preserved bit-exactly and must not be reimplemented). No catalog entry covers the residual host-bound micro-overheads (weight cast, renormalize, zero-init) that remain.
- Target-profile consultation (`prompts/coder_targets/triton_cuda.md`): the only remaining device-time reductions would require either (a) reimplementing `torch.topk` in Triton, which is prohibited by the tie-semantics invariant and is the exact grouped-topk failure mode the catalog warns against, or (b) a host-side weight pre-cast / output-cache change, which is a lifecycle deviation with correctness risk for a sub-5% gain. Neither clears the 5% falsifiable threshold.
- The `tl.dot` capability is now proven on this profile (fp16 contraction 128/64, M>=16), so there is no remaining capability-blocked primitive to unlock; every remaining primitive is either already used optimally or prohibited by an invariant.

## Rationale and Evidence

Round 002 (`triton_fused_moe_002.py`) was accepted at +79.98% wall (2.464602 → 0.493474 ms) by fusing the per-expert GEMM loop, the argsort bucketing, and the weighted reduction into a single `_fused_moe_expert_kernel` (grid=8) using `tl.dot`, collapsing kernel count 54.0 → 9.82/call and device time 500.65 → 140.84 us/call. The `tl.dot` fp16 contraction-128/64 capability risk resolved cleanly.

The round-002 report (`report_002.md`) establishes that the operator is now **strongly host/launch-bound** with no remaining ≥5% falsifiable intervention:

1. **device_ratio fell to 0.2854** (wall 0.493 ms vs device 140.84 us — ~71% is host/launch/other). The dominant remaining time is host-side launch overhead and the harness-fixed `set_seed` / `sync_devices` boundary, which are outside the candidate's change boundary (they are part of the measurement fingerprint and cannot be altered without a new comparable baseline).

2. **The remaining device time is already near-irreducible.** Of the 140.84 us/call, the untouchable `torch.topk` (`gatherTopK` 21.59 + `bitonicSortKVInPlace` 17.85 = 39.44 us/call) is 28% and is preserved bit-exactly per the tie-semantics invariant (reimplementing it is the exact grouped-topk failure the anti-patterns catalog warns against). The single `_fused_moe_expert_kernel` (55.80 us/call) is 40% and is already the optimal single-kernel form (the `BLOCK_M=256` masking over `T*K=166` valid rows is required by `tl.dot`'s `M>=16` power-of-two constraint). Together these two are 68% of device time and are both out of scope for improvement.

3. **The residual small overheads are each below 5% and do not translate to wall.** The `w1`/`w2` fp16 cast (`float16_copy_kernel`, 15.56 us/call = 3.2% wall), the renormalize sum+div (13.74 + 6.71 = 20.45 us/call = 4.1% wall), and the `out` zero-init (`FillFunctor`, 4.60 us/call = 0.9% wall) are all device-time kernels in a launch-bound operator. Even their theoretical combined elimination (~40 us device) would at most approach ~8% wall only if device time mapped 1:1 to wall, which it does not at device_ratio 0.2854 — the harness seed/synchronization floor and the irreducible ~9.82 kernel launches absorb the difference. A host-side weight pre-cast or output-cache change to remove the cast/zero-init is a lifecycle deviation (the reference re-casts weights every forward, and a cached output introduces state ownership/concurrency semantics) whose expected gain is sub-5% and carries correctness risk.

4. **No capability remains blocked.** `tl.dot` (the last unproven primitive) is now confirmed working for these GEMM shapes, so there is no "unlock a primitive" hypothesis left; every remaining primitive is either used optimally (`tl.dot`, `tl.atomic_add`, `tl.load`/`tl.store`) or prohibited by an invariant (`torch.topk` tie semantics).

This matches the established pattern from tasks 5/7/8/9: after a large multi-round kernel-fusion win collapses the kernel count and drives the operator to a launch-bound floor, the terminal round is a measurement-bound abort because the residual time is harness-fixed seed/synchronization plus irreducible launches, with no single falsifiable intervention that clears the 5% unrounded-median threshold. The canonical implementation `triton_fused_moe_002.py` (wall 0.493474 ms, a cumulative ~85% improvement over the 3.258671 ms Phase 0 baseline) remains the accepted kernel.

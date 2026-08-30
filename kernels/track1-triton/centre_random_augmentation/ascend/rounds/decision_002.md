# Decision 002

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"002","reference_implementation":"triton_centre_random_aug_001.py","reference_report":"rounds/report_001.md","language":"triton","backend":"ascend","target_profile":"triton_ascend","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-change"}
```

## Optimization Intent

```json
{"bottleneck_class":"host-bound","intervention":"no falsifiable intervention clears the 5% adoption threshold; wall time remains host-launch-bound (device_ratio ~0.107) and the only remaining launch-count reduction target (~63 torch launches) is blocked by the hard bitwise-RNG-order constraint that keeps 3x torch.rand(4) + 1x torch.randn(4,3) and the quaternion Sin/Cos/Sqrt path in torch","allowed_changes":[],"invariants":["ModelNew public contract (n_sample=4, s_trans=1.0, centre_only=False)","output dtype fp32 and shape [4,256,3]","seeded RNG stream: 3x torch.rand(4) then 1x torch.randn(4,3), identical draw order and shapes as base.py (R/T bitwise identical)","center == mean(dim=-2) matches base.py","benchmark and measurement-fingerprint semantics unchanged"],"expected_wall_improvement_pct":0.0}
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

- Consulted `references/bottleneck-judgment.md`. The candidate is host-bound: device_ratio = 216.063 / (2.023920 * 1000) = 0.1067 (< 20%). The Primary Ratio and the fused_moe worked example rounds 3-5 both record this terminal state: once wall is host-dominated and the remaining host cost is fixed backend launch/dispatch plus harness synchronization, no device-side or host-side falsifiable intervention clears 5%.
- Consulted `references/invariants.md`. The remaining host cost is fixed Triton launch/dispatch plus harness `sync_devices()`/seed setup, not candidate-compressible. `base.py` is immutable and the harness may not be altered to manufacture a speedup. The one remaining compressible lever — fusing the ~63 torch launches that remain — is prohibited by the correctness invariant: the RNG draws (3x `torch.rand(4)`, 1x `torch.randn(4,3)`) and the quaternion-to-matrix Sin/Cos/Sqrt stack must stay in torch to preserve the seeded RNG stream bitwise, and reordering/relocating them is a guaranteed `allclose(1e-2,1e-2)` failure.
- Consulted `references/anti-patterns.md`. The four recorded failures (winner-tree selection, sort-32/64 network, dynamic `tl.gather` compaction, cumsum compaction) are grouped-topk selection networks on MLU590-H8 and are shape/runtime-inapplicable to this fp32 elementwise matvec over [4,256,3]. None names a host-side launcher reduction or a device-side path that moves wall without the host penalty.
- Consulted sibling Ascend campaign terminations: groupedtopk (decision_003), flexattention (decision_004), fused_moe (decision_004), sparse_pooler, music_rotary, mm_encoder_attention (decision_002), mhc_post_layer_mix (decision_003), and mhc_head_compute_mix all aborted at this exact terminal state after exhausting kernel fusion and allocation reuse, measuring a fixed Triton launch/dispatch overhead with `fast_libentry`/stream/context all Unknown on Ascend. This operator converges on the same host-bound floor.

## Rationale and Evidence

Round 001 captured the single dominant win available to this operator: fusing the deterministic linear tail (centering mean + x-center, 3x3 rotation matvec, translation add, mask multiply) into one Triton kernel over [4,256,3] collapsed ~46 deterministic elementwise/stride/broadcast/reduce kernels into a single launch, cut device_us_per_call from 294.97 to 216.06 us (-26.8%), dropped kernel_count_per_call from 110 to 64, and improved wall time 17.84% (2.463270 -> 2.023920 ms, accepted).

The terminal state is now firm and Verifier-backed:

1. **Wall remains host-launch-bound.** device_ratio is 0.1067 (< 20%), so ~89% of wall is host-side launch/dispatch/synchronization. Even the fused `_centre_aug_linear_kernel` — the single largest device consumer at ~122 us/call (56% of device) — is only ~6% of wall (122 / 2024 us). Eliminating it entirely would fall short of the 5% threshold once host overhead is unchanged, and no device-side dataflow change (grid/warp tuning, `tl.dot` Cube routing) can convert 10.7% device fraction into a >=5% wall gain.

2. **The only remaining launch-count target is correctness-blocked.** kernel_count_per_call = 64, dominated by the torch R/T + quaternion Sin/Cos/Sqrt path plus `contiguous()`/`empty`/host-transfer launches. The hard correctness constraint — preserving the seeded RNG stream bitwise (3x `torch.rand(4)` then 1x `torch.randn(4,3)`, then the quaternion-to-matrix transcendentals) — forces these ~63 launches to remain in torch. Relocating or reordering any draw changes R/T by O(1) and FAILS the `allclose(1e-2,1e-2)` gate regardless of mathematical equivalence. This was the explicit reason report_001 marked H-001 `partially-confirmed` (kernel_count 64, not <=25).

3. **No host-side lever is candidate-compressible.** Allocation reuse was already folded into the single `torch.empty` output (one allocation, already minimal). `fast_libentry`/stream/context are Unknown on Ascend (per sibling campaigns on this identical runtime, which measured a fixed Triton launch/dispatch overhead before aborting). The residual host time is fixed backend launch/dispatch plus the harness's per-iteration synchronization and seed setup, which `bottleneck-judgment.md` classifies as "fixed for the regime".

This reproduces the campaign-wide `host-bound-floor` convergence: wall time is dominated by fixed Triton launch/harness overhead, the device fraction is below the point where further tuning moves wall, and the remaining launch reduction is blocked by a hard RNG-bitwise correctness invariant. No stable, falsifiable intervention with expected >=5% unrounded wall improvement remains. The correct decision is to halt.

Stop reason: `host-bound-floor` — wall time is host-launch-bound (device_ratio ~0.107), the fused kernel already captured the single dominant fusion win, the only remaining launch reduction is blocked by the bitwise-RNG-order correctness invariant, and further device tuning is sub-threshold. Final cumulative result: centre_random_augmentation reaches +17.84% wall improvement over baseline (2.463270 -> 2.023920 ms) via `triton_centre_random_aug_001.py`.

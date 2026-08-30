# Decision 003

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"003","reference_implementation":"triton_sparse_pooler_001.py","reference_report":"rounds/report_001.md","language":"triton","backend":"ascend","target_profile":"triton_ascend","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-change"}
```

## Optimization Intent

```json
{"bottleneck_class":"mixed","intervention":"no falsifiable intervention is expected to clear the 5% adoption threshold; the remaining device time is dominated by MLM-head library matmuls (aclnnAddmm) that Triton cannot beat, and the remaining host time is fixed backend launch/dispatch plus harness synchronization, with allocation reuse already falsified and fast_libentry Unknown on Ascend","allowed_changes":[],"invariants":["ModelNew public constructor and forward signature","output is a Python list of num_seq tensors each [vocab_size] fp32 npu:0","numerical semantics log(1+relu(decoder(LayerNorm(GELU(Dense(hidden)))))) max-pooled per sequence within atol=1e-2 rtol=1e-2 equal_nan=True","benchmark wall-time measurement semantics","dense GELU LayerNorm decoder matmul remain PyTorch library ops"],"expected_wall_improvement_pct":0.0}
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

- Consulted `references/bottleneck-judgment.md`. The accepted candidate is mixed-to-host-bound (device_ratio ~0.30 after Round 1, ~0.30 after Round 2's no-improvement). The remaining host time (~70% of wall) is NOT a compressible per-forward allocation (Round 2 proved the NPU caching allocator already amortizes `torch.empty`) — it is fixed backend launch/dispatch plus the harness's `sync_devices()`/seed setup, which are "Fixed for the regime" and outside the candidate change boundary. The remaining device time is 76% `aclnnAddmm` library matmuls, which the `tl.dot` path provably cannot beat. The fused_moe worked example rounds 3-5 and all three sibling Ascend campaigns document this exact terminal state.

- Consulted `references/invariants.md`. The remaining host cost is backend-fixed launch/dispatch plus harness synchronization, not candidate-compressible. Altering `base.py` or the harness to manufacture a speedup is forbidden. No candidate-side mechanism preserves the public contract (list-of-tensors output, library MLM head) while compressing the fixed launch overhead or beating the vendor matmul.

- Consulted `references/anti-patterns.md`. The catalog's recorded device-selection regressions (winner-tree, sort networks, dynamic `tl.gather`, cumsum compaction) do not apply to this operator shape, but neither does any entry name a host-side launcher reduction or a device-side matmul improvement that beats the vendor `aclnnAddmm`. The catalog confirms the known device levers regress; it offers no untried lever here.

- Consulted `prompts/coder_targets/triton_ascend.md`. `fast_libentry`, stream/context, and async-copy semantics are `Unknown` on Ascend (no probe establishes them); direct launch is already the proven path. `tl.dot` is only probed at `(16,16)@(16,16)`; a `[83,768]@[768,30522]` matmul via `tl.dot` is a large, unprobed shape with a strong prior of regression.

- Consulted the three sibling Ascend campaign terminations: groupedtopk-ascend `decision_003` and flexattention-ascend `decision_004` and fused_moe-ascend `decision_004` all aborted at this exact terminal state — after exhausting kernel fusion + allocation reuse, measuring ~107 us of fixed Triton launch/dispatch overhead, and confirming `fast_libentry`/stream/context are Unknown on Ascend. flexattention-ascend additionally measured that the device-side `tl.dot` Cube path regressed wall -8.34% via a net-negative +55 us/call host penalty. The same evidence bounds this campaign.

## Rationale and Evidence

Round 1 (`triton_sparse_pooler_001.py`, accepted, +33.78%) fused relu + log1p + per-sequence max pooling (+ cast) into a single Triton kernel with an on-device prefix scan, eliminating 9 device kernels and the `seq_lens.tolist()` D2H sync, taking wall from 0.935560 ms to 0.618775 ms and device from 374.81 to ~194-202 us/call (5 kernels/call). This is the single large, correctly-attributable win available to this operator, and it is already captured.

Round 2 (`triton_sparse_pooler_002.py`, no-improvement, +2.75%) tested host-side output-buffer reuse and was falsified: the NPU caching allocator already makes the per-call `torch.empty((4, 30522))` output allocation near-free on this runtime (report_002 `evidence_for_next_round`). The mechanism observables were satisfied (cache present, device/kernel count unchanged), but the primary metric (wall ≥5%) was not met. The remaining host time is therefore not allocation — it is fixed backend launch/dispatch plus harness synchronization.

Every remaining lever is now exhausted or carries negative evidence:

1. **MLM-head matmul fusion (`tl.dot`)**: the two `aclnnAddmm` matmuls (dense 768→768 and decoder 768→30522) are ~135-147 us/call (~74-76% of device time) but are vendor library ops. `tl.dot` fusion has **double negative evidence**: the MLU sibling Round 3 proved `tl.dot` matmul fusion regressed device time for small-M matmuls (the Triton matmul was slower than the vendor `aclnnAddmm`/`MLUFusedMatMulGepm`), and flexattention-ascend Round 3 proved the `tl.dot` Cube path regressed wall -8.34% via a +55 us/call host penalty on this exact Ascend runtime. `tl.dot` is only probed at `(16,16)@(16,16)`; a `[83,768]@[768,30522]` Triton matmul is a large, unprobed shape with a strong prior of regression. This lever is explicitly closed.

2. **Tile-tuning (BLOCK_V 1024→2048)**: the MLU sibling Round 2 falsified this — enlarging BLOCK_V regressed the fused kernel (99.71 → 102.42 us/call). The current BLOCK_V=1024 is already the best-known value.

3. **`fast_libentry` (launcher reduction)**: `Unknown` on Ascend (no probe); making it normative would be a capability-miss. Direct launch is already the proven path and in use.

4. **Output allocation reuse**: already falsified in Round 2 (NPU caching allocator).

5. **Intermediate `logits` `[83,30522]` tensor reuse**: allocated inside the `self.decoder(...)` library matmul, not a candidate-owned `torch.empty`; caching it requires rewriting the MLM-head library pipeline (a larger change boundary that risks the library-op regression and does not target a compressible candidate-owned allocation).

6. **Remaining host (~70% of wall)**: launch/dispatch + harness-fixed `sync_devices()`/seed setup, which are "Fixed for the regime" per bottleneck-judgment and not candidate-controllable.

The honest assessment is that no falsifiable intervention with expected ≥5% wall improvement remains. The device-side lever (matmul) is library-optimal and Triton-proven-slower; the host-side lever (allocation) is already amortized by the allocator; the launcher lever (`fast_libentry`) is Unknown; and the residual host time is harness/backend-fixed.

This maps precisely to the terminal state that terminated all three sibling Ascend campaigns (groupedtopk, flexattention, fused_moe), and to the MLU sibling's narrow total (1.60x, with its R2 and R3 both failing). Notably, this Ascend campaign's Round 1 capture (+33.78%) already exceeds the MLU sibling's Round 1 (+33.39%), confirming the one real win has been banked.

Accordingly this round recommends halting: the decision is `abort` (halt) with no further candidate dispatched.

**Final cumulative result**: sparse_pooler-ascend reaches **+33.78% wall improvement** over baseline via Round 1 kernel fusion (wall 0.935560 ms → 0.618775 ms). The accepted canonical `triton_sparse_pooler_001.py` stands as the campaign result. Round 2 (allocation reuse) was a sub-threshold no-improvement (+2.75%) and is not adopted; the canonical pointer remains `triton_sparse_pooler_001.py`.

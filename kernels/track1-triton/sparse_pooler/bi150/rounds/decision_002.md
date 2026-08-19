# Decision 002

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"002","reference_implementation":"triton_sparse_pooler_001.py","reference_report":"rounds/report_001.md","language":"triton","backend":"cuda","target_profile":"triton_cuda","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-change"}
```

## Optimization Intent

```json
{"bottleneck_class":"device-bound","intervention":"no stable intervention clears the 5% adoption threshold: the remaining 92.5% of device time is two GEMMs already running on the vendor tensor-core (TCU) hardware, and a fp32 large-N tl.dot rewrite has no falsifiable >=5% expectation because it is unproven on this profile and cannot credibly beat the vendor-tuned TCU in a compute-bound regime with no launch redundancy to remove","allowed_changes":[],"invariants":["ModelNew public contract (hidden_size=768, vocab_size=30522, pooling=max)","forward signature (hidden_states, seq_lens) -> list[4 x [30522]] fp32","output list structure, per-element shape [30522], dtype fp32","log1p(relu(x)) and per-sequence max-pool semantics","input tensors not mutated","caller-selected device and current stream preserved"],"expected_wall_improvement_pct":0.0}
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

- Consulted `references/anti-patterns.md`. No catalog entry (winner-tree, sort-32/64, dynamic gather, cumsum compaction) matches a fp32 large-N GEMM rewrite, so there is no directly applicable prior failure. However, the catalog's core lesson is directly relevant by analogy: on this class of accelerator, a Triton reimplementation of a vendor-tuned primitive does not automatically beat the vendor path, and the winning cases (fused_moe) won on launch-count reduction in a launch-bound regime, not on out-computing the TCU in a compute-bound regime.
- Target-profile consultation (`prompts/coder_targets/triton_cuda.md`): `tl.dot` is recorded Supported only for `(32,32)@(32,32)` (fp32 exact, bf16→fp32 accumulate). The dense GEMM is `(83,768)@(768,768)` and the decoder GEMM is `(83,768)@(768,30522)` — a fp32 contraction dim 768 with output N=30522, entirely outside the proven probe. The only larger-`tl.dot` evidence is fused_moe's **fp16** contraction-128/64 with small M, which is a different precision and shape regime and does not transfer to fp32 large-N.
- `gemm_tcu_h` is the Iluvatar vendor tensor-core kernel (template `64u,64u,64u,16u,16u,2u`, B-transposed, `float` in/accumulate/out). The Triton `tl.dot` lowering on this CoreX CUDA backend has no recorded evidence that it maps to the same vendor TCU hardware for fp32 large-N, and if it lowers to a generic CUDA tensor-core path it is unlikely to beat the vendor-tuned TCU. This is the central capability/magnitude risk that no decision can finesse without a matched local probe — and a probe whose most likely outcome is "no gain" does not constitute a falsifiable 5% hypothesis.
- `num_warps`/`num_stages` remain Unknown and would have to stay non-normative for any GEMM rewrite; this further reduces the credible tuning lever against a vendor kernel that is already hardware-tuned.

## Rationale and Evidence

Round 001 accepted `triton_sparse_pooler_001.py` at +16.99% wall (1.060573 → 0.880377 ms) by fusing the post-decoder SPLADE activation chain + per-sequence max-pooling into a single kernel and removing the per-call `seq_lens.tolist()` device-to-host sync. The round-001 report's candidate profile shows the operator is now dominated by the two GEMMs: `gemm_tcu_h` at 482.53 us/call plus `GEMM_Epilogue` bias-add at 81.29 us/call, together ~563.8 us/call ≈ 92.5% of the remaining 609.40 us/call device time, with device_ratio ≈ 0.692.

The remaining optimization target is therefore exclusively the GEMM pair — the `dense` 768×768 projection and the `decoder` 768×30522 projection — both already executing on the Iluvatar **TCU** (`gemm_tcu_h`, a vendor tensor-core kernel), not on generic CUDA cores.

I judge that no proceeding intervention has a falsifiable ≥5% wall-time expectation, for three independent reasons:

1. **The GEMM is already on vendor-optimal hardware.** `gemm_tcu_h` is the Iluvatar vendor tensor-core path. A Triton `tl.dot` rewrite would have to out-compute the vendor-tuned TCU on fp32 large-N (contraction 768, N up to 30522). The profile records `tl.dot` only for `(32,32)@(32,32)`; the one larger success (fused_moe) was **fp16 with small M**, a different precision/regime. There is no evidence that Triton `tl.dot` on this CoreX CUDA backend maps to the same vendor TCU for fp32 large-N, and if it lowers to a generic tensor-core path it is unlikely to beat the vendor kernel.

2. **This is a compute-bound regime, not launch-bound.** fused_moe's `tl.dot` win (+79.98%) came from collapsing ~44 per-expert launch-bound kernels into one, removing host/launch overhead in a device_ratio-0.20 regime. sparse_pooler is the opposite: device_ratio ≈ 0.69, with only 2 GEMM launches to begin with. There is no launch redundancy to remove, so the only way a rewrite wins is by out-computing the TCU — the very thing that is unproven and improbable here.

3. **The remaining fusion opportunities are all coupled to the GEMM rewrite.** The `GEMM_Epilogue` bias-add (81.29 us), and the inter-GEMM GELU/LayerNorm (~17 us), can only be fused by rewriting the GEMM itself. They are not independently removable, so they do not constitute a separate ≥5% direction.

The only direction with a plausible large win is an **fp32→fp16 conversion** of the GEMMs, but that is a semantic change: it alters the reference dtype (base is fp32 end-to-end) and the numerical results. It lies outside the kernel-optimization contract, which requires output dtype/shape to remain compatible with the reference unless the user explicitly changes the project contract. Adopting it without user authorization would violate the semantic invariant, so it is not a decision I can issue.

I therefore record `abort`: the remaining device-bound bottleneck is the two vendor-TCU GEMMs, and no stable kernel intervention has a falsifiable ≥5% expectation. This is not a `measurement-bound` claim (device_ratio remains ~0.69 and device work is real), but a `vendor-optimal-bound` claim: the dominant compute is already on the vendor's tuned tensor-core path with no credible Triton-superior alternative evidenced on this profile. If a future matched local probe of fp32 large-N `tl.dot` on this CoreX backend (or a user-authorized fp16 semantic change) materializes, it should be reconsidered as a new decision under an updated fingerprint.

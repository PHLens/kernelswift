# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"cuda","target_profile":"triton_cuda","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"activation-pooling-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"device-bound","intervention":"fuse the post-decoder SPLADE activation chain (ReLU via clamp + log1p) and the per-sequence max-pooling into a single fused Triton kernel, eliminating the intermediate [83,30522] activations (one clamp_scalar kernel, one log1p kernel, and the four per-sequence max-reduce kernels) and their full-tensor read/write traffic, while leaving the dense and decoder GEMMs on the vendor TCU","allowed_changes":["ModelNew.forward dataflow","a fused Triton kernel over the log1p(relu(x)) activation and the per-sequence max(dim=0) pooling stages","replacement of torch log1p/relu/clamp and chunk.max(dim=0) with the fused kernel"],"invariants":["ModelNew public contract (hidden_size=768, vocab_size=30522, pooling=max)","forward signature (hidden_states, seq_lens) -> list[4 x [30522]] fp32","output structure: a list of exactly 4 fp32 tensors each [30522], in seq_lens order","SPLADE activation log1p(relu(x)) and per-sequence max-pool semantics preserved","dense/decoder GEMM, GELU, and LayerNorm semantics unchanged (TCU path preserved)","input tensors not mutated","caller-selected device and current stream preserved"],"expected_wall_improvement_pct":8.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor x shape=[83,30522] dtype=fp32 layout=contiguous memory=global
tensor seq_lens shape=[4] dtype=int32 layout=contiguous memory=global
tensor out0 shape=[30522] dtype=fp32 layout=contiguous memory=global
tensor out1 shape=[30522] dtype=fp32 layout=contiguous memory=global
tensor out2 shape=[30522] dtype=fp32 layout=contiguous memory=global
tensor out3 shape=[30522] dtype=fp32 layout=contiguous memory=global

# O Operations
load seq_len_i <- seq_lens[i]
load chunk <- x[offset:offset+seq_len_i, vocab] for each of 4 sequences
compute act <- log1p(max(relu(chunk), 0)) per column
compute pooled <- max over sequence axis of act
store out_i[vocab] <- pooled

# C Control
parallel vocab over 30522
guard vocab < 30522
for seq in range(4)
end

# H Target Hints
target=triton_cuda
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only change; the fused Triton kernel replaces the torch log1p/relu and chunk.max(dim=0) tail with a single kernel launch, introducing no new host-side state, buffer reuse, or allocation caching; hidden_states and seq_lens are read-only inputs, x is the intermediate decoder output read once, and the four output tensors are each written once"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-001","intervention":"fuse the post-decoder SPLADE activation chain (ReLU via clamp + log1p) and the per-sequence max-pooling into a single fused Triton kernel, eliminating the intermediate [83,30522] activations (one clamp_scalar kernel, one log1p kernel, and the four per-sequence max-reduce kernels) and their full-tensor read/write traffic, while leaving the dense and decoder GEMMs on the vendor TCU","expected_causal_chain":["the clamp_scalar (21.3 us/call) and log1p (33.8 us/call) elementwise kernels disappear, and their [83,30522] intermediate tensors (~10 MB each) are no longer written and re-read","the four per-sequence max-reduce kernels (88.8 us/call total) collapse into the fused kernel's in-kernel column-wise max reduction","per-call kernel count decreases (from 11.92 toward ~6), removing per-launch host overhead","device_us_per_call decreases","wall time decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"kernel_count_per_call","expectation":"decrease"},{"name":"device_us_per_call","expectation":"decrease"}],"guardrails":["correctness:pass","output list structure, per-element shape [30522], dtype fp32 unchanged","log1p(relu(x)) and per-sequence max-pool semantics preserved","dense/decoder GEMM and GELU/LayerNorm unchanged","input not mutated"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`. The catalog entries (winner-tree Entry 011, full sort Entry 012, dynamic `tl.gather` Entry 013, cumsum compaction Entry 016) all concern grouped-topk selection networks on MLU590-H8 / Triton 3.2.0 and do not match this intervention's preconditions: this round performs no index selection, no tie ordering, no on-chip gather, and no hierarchical reduction network. It is a plain elementwise activation plus a column-wise max reduction over a contiguous `[83, 30522]` fp32 tensor, using only `tl.load`, `tl.max`, and elementwise math already marked Supported on the profile.
- Target-profile consultation (`prompts/coder_targets/triton_cuda.md`): `tl.load`/`tl.store` (contiguous fp32), `tl.arange`, `tl.max` (axis-0 and axis-1 reduction), `tl.static_range`, and `tl.where` are all recorded Supported. The fused kernel needs a column-wise max over the sequence axis (equivalent to the proven axis-0 reduction), which is within the proven primitive set. The per-sequence loop uses a `tl.static_range` of 4 iterations (proven). `log1p`/`relu` lower to elementwise math; `relu` is `max(x, 0)` (expressible via `tl.where`/`tl.max`) and `log1p` maps to the same libdevice intrinsic family as the proven transcendentals (`tl.exp`/`tl.sqrt`). The primary residual risk is whether the `log1p` libdevice lowering matches `torch.log1p` bit-for-bit on the CoreX backend — the correctness guardrail at `atol=1e-2, rtol=1e-2` is loose enough to absorb small numeric divergence, and the round is conditional on a matched local correctness pass.
- `num_warps`/`num_stages` remain Unknown on this profile and must stay non-normative; `tl.make_block_ptr`, `vectorize`, and `async_copy` are Unknown and must not be required.
- The GEMMs (dense 768×768 and decoder 768×30522) are deliberately NOT rewritten this round: they already run on the vendor TCU (`gemm_tcu_h`, ~498 us/call, ~67% of device time) and a fp32 `tl.dot` rewrite of an N=30522 GEMM is unproven on this profile (the recorded `tl.dot` probe covers only `(32,32)@(32,32)`; the fused_moe win was fp16 with small M). Rewriting them here would carry a high capability-miss risk with no proven fp32 large-N `tl.dot` evidence; that remains a separately-probed future hypothesis, not this round's change.
- The harness AST loader (`auto_bench.py` `_filter_module_ast`) retains `Import`/`ImportFrom`/`ClassDef`/`FunctionDef` and literal assignments, so a top-level `@triton.jit` function is preserved; the candidate must still expose `ModelNew`, `get_init_inputs`, and `get_inputs`, and `forward` must return a Python `list` of 4 tensors (the harness compares lists recursively).

## Rationale and Evidence

The Phase 0 report (`rounds/report_000.md`) establishes `baseline_adapter.py` at wall median `1.070492 ms`, device `743.064 us/call`, `11.92` kernels/call, device_ratio `0.694` — this operator is **compute-bound**, unlike launch-bound operators such as fused_moe (device_ratio ~0.20).

The dominant device cost is the two GEMMs (`gemm_tcu_h` at `498.5 us/call`) plus their bias-add epilogues (`GEMM_Epilogue` at `83 us/call`), together ~`581 us/call` ≈ 78% of device time. These already execute on the Iluvatar TCU (`gemm_tcu_h`, fp32, tile `64u,64u,64u,16u,16u,2u`), i.e. vendor tensor-core hardware.

The second-largest, safely-removable device cost is the post-decoder tail: the SPLADE activation chain (`clamp_scalar` ReLU `21.3 us` + `log1p` `33.8 us`) and the four per-sequence max-pool reductions (`reduce_kernel<MaxOps>` `88.8 us/call`), together ~`144 us/call` ≈ 13.5% of device time. These six kernels operate on the full `[83, 30522]` tensor; each of the two elementwise kernels writes and the following kernel re-reads a ~10 MB intermediate, so the tail also carries substantial full-tensor memory traffic beyond its raw compute time.

I select the **activation + pooling fusion** as the Round 001 intervention because it is the only direction with a falsifiable ≥5% wall gain that does not depend on an unproven primitive. Specifically:

1. **The GEMM is the real bottleneck but is not safely touchable this round.** A fp32 `tl.dot` rewrite of `dense` (768×768) and especially `decoder` (768×30522, N=30522) is unproven on the `triton_cuda` profile — the recorded `tl.dot` evidence is only `(32,32)@(32,32)`, and the fused_moe `tl.dot` success was fp16 with a small M (~20). The TCU `gemm_tcu_h` is vendor-tuned tensor-core silicon; a Triton `tl.dot` competing with it on a large-N fp32 GEMM is a high capability-miss / regression risk, not a stable 5% bet. This remains a candidate for a later round only after a matched local `tl.dot` fp32 large-N probe.

2. **fp16 conversion is a semantic change**, not a kernel optimization: it changes the reference dtype (base is fp32 end-to-end) and alters numerical results. Although `atol=1e-2` is loose, adopting a dtype change as the first optimization step without user authorization violates the semantic invariant that the output dtype/shape remain compatible with the reference. It is out of scope for a kernel-only Round 001.

3. **The activation + pooling fusion is proven-primitive and falsifiable.** Removing the `clamp_scalar` + `log1p` kernels and the four `reduce_kernel<MaxOps>` launches, and folding the column-wise max into the activation pass, eliminates ~`144 us` of device time and ~`6` kernel launches (11.92 → ~6/call), plus the two ~10 MB intermediate-tensor write/read cycles. Even with imperfect wall-to-device transmission (compute-bound, device_ratio 0.694), removing ~13.5% of device time and ~half the kernel launches conservatively supports `expected_wall_improvement_pct = 8.0`, above the 5% adoption threshold.

The change is kernel-only: `ModelNew.forward` keeps `dense` → `GELU` → `LayerNorm` → `decoder` exactly as-is (all on the TCU), then replaces `log1p(relu(x))` + the four `chunk.max(dim=0)` calls with a single fused Triton kernel that returns the same `list` of 4 `[30522]` fp32 tensors. The public contract, output structure, and numerical semantics (log1p(relu), per-sequence max) are preserved as invariants. The canonical comparison source is `baseline_adapter.py`; the reference report is `rounds/report_000.md`.

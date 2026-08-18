# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"ascend","target_profile":"triton_ascend","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"mixed","change_family":"kernel-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"mixed","intervention":"fuse relu, log1p, and per-sequence max pooling (plus the redundant max-pool output cast) into a single Triton kernel launched once per forward, eliminating the Python for-loop over seq_lens.tolist() and replacing the six device kernels (relu, log1p, 4x aclnnMaxDim_Max2AiCore_ArgMaxWithValue, 4x aclnnMaxDim_CastAiCore_Cast) with one fused reduction kernel that computes per-segment offsets on-device","allowed_changes":["new Triton kernel performing elementwise relu+log1p followed by per-segment max reduction over the sequence axis and writing a [num_seq, vocab_size] fp32 output tensor","ModelNew.forward dispatch path: replace the Python for-loop over seq_lens.tolist() and the four per-chunk torch.max calls with a single fused kernel launch returning [out[i] for i in range(num_seq)]","device-side seq_offset prefix scan from the seq_lens tensor inside the kernel to avoid the D2H synchronization that seq_lens.tolist() triggers"],"invariants":["ModelNew public constructor and forward signature unchanged","output is a Python list of num_seq tensors each of shape [vocab_size] dtype fp32 device npu:0","numerical semantics: log(1+relu(decoder(LayerNorm(GELU(Dense(hidden)))))) max-pooled per sequence within atol=1e-2 rtol=1e-2 equal_nan=True","caller-selected device and current stream preserved; no torch.npu.device() context introduced","dense, GELU, LayerNorm, and decoder matmul remain PyTorch library ops unchanged","load_state_dict compatibility maintained","pooling == \"sum\" branch preserves the original reference behavior"],"expected_wall_improvement_pct":15.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor logits shape=[total_seq, vocab_size] dtype=fp32 layout=row_major memory=global
tensor seq_lens shape=[num_seq] dtype=int32 layout=contiguous memory=global
tensor out shape=[num_seq, vocab_size] dtype=fp32 layout=row_major memory=global
tile vocab_tile shape=[BLOCK_V] dtype=fp32 memory=register
tile acc shape=[BLOCK_V] dtype=fp32 memory=register
scalar pid dtype=int32 memory=register
scalar seq_offset dtype=int32 memory=register
scalar seq_len dtype=int32 memory=register

# O Operations
load pid <- program_id(0)
load seq_len <- seq_lens[pid]
compute seq_offset <- sum(seq_lens[0:pid])
alloc acc <- full([BLOCK_V], -inf, dtype=fp32)
load vocab_tile <- logits[seq_offset + row, v_start:v_end]
compute vocab_tile <- log(1.0 + relu(vocab_tile))
compute acc <- maximum(acc, vocab_tile)
store out[pid, v_start:v_end] <- acc

# C Control
parallel pid over num_seq
parallel v_start over vocab_size stride BLOCK_V
guard pid < num_seq
guard v_start < vocab_size
for row in 0:seq_len
guard row < seq_len
end

# H Target Hints
target=triton_ascend
num_warps=1
```

## Host Plan

```json
{"applicability":"required","affected_scope":["ModelNew.forward dispatch path","Triton kernel launch","output tensor allocation"],"state_owner":"ModelNew instance","lifetime":"model lifetime; kernel compiled once and cached by Triton JIT for the process","allocation_reuse":"output [num_seq, vocab_size] fp32 tensor allocated per forward call with torch.empty; no cross-forward buffer caching in this round","cache_key":["seq_lens shape","num_seq","vocab_size","dtype","device"],"invalidation":"no persistent cache in this round; kernel autotune/JIT cache managed by the Triton runtime","concurrency":"one ModelNew instance is not shared across concurrent forwards; the output tensor is per-call","device_stream_behavior":"caller-selected device and current stream are preserved; no explicit torch.npu.device() context is introduced; seq_lens offsets are computed on-device inside the kernel to avoid the seq_lens.tolist() D2H synchronization","unchanged_behavior":["returned Python list of num_seq tensors","each output tensor shape [vocab_size]","each output tensor dtype fp32","each output tensor device npu:0","numerical semantics log(1+relu(logits)) max-pooled per sequence","dense GELU LayerNorm decoder matmul pipeline unchanged","load_state_dict compatibility","pooling == \"sum\" fallback behavior"]}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-001","intervention":"fuse relu, log1p, and per-sequence max pooling (plus the redundant max-pool output cast) into a single Triton kernel launched once per forward, eliminating the Python for-loop over seq_lens.tolist() and replacing the six device kernels (relu, log1p, 4x aclnnMaxDim_Max2AiCore_ArgMaxWithValue, 4x aclnnMaxDim_CastAiCore_Cast) with one fused reduction kernel that computes per-segment offsets on-device","expected_causal_chain":["the Python for-loop over seq_lens.tolist() and the four per-chunk torch.max dispatches are replaced by one Triton kernel launch","host-side D2H synchronization from seq_lens.tolist() is eliminated because per-segment offsets are computed on-device","device kernel count per call drops from 14 to 6 (relu, log1p, 4x max-pool, 4x cast removed; one fused kernel added)","host dispatch overhead and device kernel launch count decrease","wall time decreases by at least 5%"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"kernel_count_per_call","expectation":"decrease from 14 to 6"},{"name":"device_us_per_call","expectation":"decrease from 374.81 by roughly the fused device block (relu 9.57 + log1p 22.41 + 4x max-pool 247.09 + 4x cast 48.29 = ~327 us/call) minus the fused kernel cost"},{"name":"host_sync_count_per_call","expectation":"decrease because the seq_lens.tolist() D2H sync is eliminated"}],"guardrails":["correctness:pass","output is a Python list of num_seq tensors each [vocab_size] fp32 npu:0","numerical semantics log(1+relu(decoder_logits)) max-pooled per sequence preserved within atol=1e-2 rtol=1e-2 equal_nan=True","caller-selected device and current stream preserved","dense GELU LayerNorm decoder matmul pipeline unchanged","ModelNew public constructor and forward signature unchanged","load_state_dict compatibility maintained","pooling == \"sum\" branch preserves the original reference behavior"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`. No entry matches this intervention. The recorded failures concern winner-tree expert selection, sort networks, dynamic `tl.gather` compaction, and cumsum compaction in grouped top-k — none apply to a straightforward elementwise-plus-segment-max-reduction fusion over a small ragged batch. The reduction here is a flat per-segment max (no index/value pair carried, no hierarchical selection, no dynamic gather), which is structurally different from every recorded failure.

- Consulted `references/bottleneck-judgment.md`. Baseline `device_ratio` is 0.4006 (device 374.81 us/call / wall 935.56 us/call) — squarely in the `mixed` band (20%–80%). The intervention targets both host and device mechanisms: the host-side Python loop + D2H sync (compressible launcher/dispatch overhead) and the device-side kernel count (compressible fusion). The two pieces are inseparable — the fused kernel is precisely what eliminates the Python loop and the `seq_lens.tolist()` sync — and are separately observable via `kernel_count_per_call`, `device_us_per_call`, and `host_sync_count_per_call`, satisfying the mixed-change attribution requirement.

- Consulted `prompts/coder_targets/triton_ascend.md`. The primitives this kernel needs are all either Supported or Constrained on the recorded Ascend runtime: `tl.load`/`tl.store` (masked contiguous), `tl.arange`, `tl.program_id`, `tl.full`, `tl.where`, `tl.max`, `tl.static_range` (compile-time `range` loops), and `tl.program_id(1)` for the vocab tile axis. `num_warps=1` is a proven value (the probe verified `num_warps=1/2/4` all compile and run). `tl.log` is not in the probe table but is a fundamental elementwise math op used here as `tl.log(1.0 + x)` on non-negative relu output (`1 + x >= 1`, numerically stable, matching the MLU sibling's proven expression); Coder must verify `tl.log` compiles on this runtime and, if it does not, fall back to a numerically equivalent form while preserving the Evaluation Contract. `fast_libentry` is **Unknown** on Ascend (no probe establishes it) — this decision does **not** make it normative; the direct Triton launch `kernel[(grid,)](...)` is the proven launcher path. No `import triton_ascend` (metadata-only, not importable); the backend is reached through `import triton` + `import torch_npu`.

- The vocabulary dimension 30522 is tiled with `BLOCK_V`. Coder must pick a power-of-two `BLOCK_V` (e.g. 1024) and apply a mask on the last tile (`v_offs < vocab_size`), matching the MLU sibling's proven tiling. The grid is `(num_seq, cdiv(vocab_size, BLOCK_V))`. With `num_seq=4` the on-device prefix scan `seq_offset = sum(seq_lens[0:pid])` is at most 3 extra `tl.load`s per program — negligible — and is exactly what removes the `seq_lens.tolist()` D2H sync.

- The max reduction uses `acc = tl.maximum(acc, x)` (or, if `tl.maximum` is unavailable, the comparison-based `tl.where(acc < x, x, acc)`). Per the anti-patterns catalog there is no flat per-segment max failure; the `tl.where` fallback is semantically identical and preserves observables. The relu is `tl.where(x > 0.0, x, 0.0)` and log1p is `tl.log(1.0 + x)`. Max-pool tie behavior: the harness tolerance (atol/rtol 1e-2, equal_nan) is wide enough that any deterministic tie-breaking of a flat max reduction stays within tolerance (the reference itself is a library `torch.max` whose exact tie behavior need not be bit-matched).

- Per `references/invariants.md`, this is a mixed kernel+host change. The Host Plan above declares affected scope, state owner, lifetime, allocation reuse (per-call, no caching), cache key, invalidation (none), concurrency (per-instance, no sharing), device/stream behavior (caller-preserved, no device context), and unchanged behavior. The output buffer is allocated per forward with `torch.empty`, so no cross-forward aliasing or cache-invalidation complexity is introduced. `pooling == "sum"` keeps the original reference fallback path (the fused kernel handles only the `"max"` default selected by `get_init_inputs`).

- The harness AST loader (`auto_bench._filter_module_ast`) strips module-level non-literal assignments. The Triton kernel `@triton.jit` decorated function is a top-level `FunctionDef` node and is retained; `ModelNew` is a `ClassDef` and is retained. Candidate `get_inputs` should use `device="npu"` (or derive device from an input tensor) rather than `"cuda"`, though the harness rewrites `"cuda"`→`"npu"` anyway. Coder must not rely on module-level side effects beyond safe literals.

## Rationale and Evidence

The accepted baseline report (`rounds/report_000.md`) records:

- Benchmark wall time median: `0.935560 ms` (935.56 us/call); device time `374.81 us/call`; `device_ratio = 0.4006` (mixed); 14 kernels/call.
- Dominant device kernels (reference scope): `aclnnAddmm_MatMulCommon_MatMulV2` at `252.10 us/call` x 2 (the MLM-head dense 768→768 and decoder 768→30522 matmuls, ~67% of device time), `aclnnMaxDim_Max2AiCore_ArgMaxWithValue` at `61.77 us/call` x 4 (per-sequence max pool), `aclnnLog1p` at `22.41 us/call`, `aclnnMaxDim_CastAiCore_Cast` at `12.07 us/call` x 4, `aclnnLayerNorm` at `11.38 us/call`, `aclnnRelu` at `9.57 us/call`, `aclnnGelu` at `5.50 us/call`.
- The fusion target (relu + log1p + 4x max-pool + 4x cast) sums to ~`327 us/call` of device time (87% of device time excluding the two matmuls), launched from a host Python `for L in seq_lens.tolist()` loop that forces a D2H sync and 4 sequential `chunk.max(dim=0)` dispatches.
- Host-side cost is significant: device_ratio ≈ 0.40 means ~60% of wall time (~560 us/call) is host-side (Python loop, dispatch, D2H sync, allocation, harness-fixed).

The intervention fuses relu + log1p + per-sequence max pooling into one Triton kernel and eliminates the Python loop and its D2H sync. This is the same change family that, on the MLU sibling (`mlu/project.md` Round 001), delivered a 33.39% wall improvement (909.97 → 606.76 us/call) by replacing the identical six-kernel block + host loop with one fused reduction kernel using an on-device prefix scan. The Ascend fuse-able device block (~327 us/call) is roughly double the MLU equivalent (~158 us/call), and the Ascend baseline has the same `seq_lens.tolist()` D2H sync, so the same mechanism is expected to clear the 5% threshold by a comfortable margin. The 15% expected wall improvement is conservative relative to the MLU 33% result and reflects the fact that the Ascend matmuls (504 us/call) are deliberately left untouched and will continue to dominate device time after fusion.

The decoder/dense matmuls are left as PyTorch library ops (`aclnnAddmm`) and are explicitly outside the change boundary. The MLU sibling's Round 003 evidence showed that fusing the decoder matmul into Triton via `tl.dot` with small M **regressed** device time (the Triton matmul was slower than the vendor `aclnnAddmm`/`MLUFusedMatMulGepm`), and `tl.dot` on Ascend is only probed at `(16,16)@(16,16)` — a `[83,768]@[768,30522]` Triton matmul would be a much larger, higher-risk change with a strong prior of regression. The MLU final accepted candidate also kept the MLM head as library ops and fused only pooling+relu+log1p, confirming this is the correct boundary. Therefore the matmul is not fused in this round; if the fused pooling kernel succeeds and the matmul becomes the next dominant bottleneck, matmul fusion is a candidate for a later round with its own evidence.

The expected causal chain is direct: removing 9 device kernel launches and 1 D2H sync reduces both host dispatch overhead and device kernel count. Even if the fused Triton kernel is no faster per-element than the library kernels, the launch consolidation (~327 us/call of device work collapsed into one kernel) and host-loop/sync elimination are expected to clear the 5% adoption threshold. This is a falsifiable intervention with three named mechanism observables (`kernel_count_per_call`, `device_us_per_call`, `host_sync_count_per_call`); if the fused kernel is unexpectedly slow or the host savings are below 5%, the hypothesis is falsified and the round terminates as no-improvement.

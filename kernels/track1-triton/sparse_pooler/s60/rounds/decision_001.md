# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"gcu","target_profile":"triton_gcu","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"mixed","change_family":"kernel-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"mixed","intervention":"fuse relu, log1p, and per-sequence max pooling into a single Triton kernel launched once per forward, eliminating the Python for-loop over seq_lens.tolist() and the D2H synchronization it triggers, and replacing six device kernels (relu, log1p, and 4x per-chunk max) with one fused reduction kernel","allowed_changes":["new Triton kernel performing elementwise relu+log1p followed by per-segment max reduction over the vocabulary dimension","ModelNew.forward dispatch path: replace the Python for-loop over seq_lens.tolist() and the 4x per-chunk torch.max calls with a single fused kernel launch over a contiguous [num_seq, vocab_size] output buffer","device-side per-sequence offset computation from the int32 seq_lens tensor inside the kernel to avoid D2H synchronization"],"invariants":["ModelNew public constructor and forward signature","output is a Python list of num_seq tensors each of shape [vocab_size] dtype fp32 on the caller-selected device","numerical semantics: log(1+relu(decoder_logits)) pooled per sequence with max reduction","tolerance atol=1e-2 rtol=1e-2 equal_nan=True","caller-selected device and current stream preserved","dense, GELU, LayerNorm, and decoder matmul remain PyTorch library ops unchanged","state_dict keys unchanged with nested submodule names dense/layer_norm/decoder"],"expected_wall_improvement_pct":8.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor logits shape=[total_seq, vocab_size] dtype=fp32 layout=row_major memory=global
tensor seq_lens shape=[num_seq] dtype=int32 layout=contiguous memory=global
tensor out shape=[num_seq, vocab_size] dtype=fp32 layout=row_major memory=global
tile vocab_tile shape=[BLOCK_V] dtype=fp32 memory=register
tile acc shape=[BLOCK_V] dtype=fp32 memory=register
scalar pid_s dtype=int32 memory=register
scalar pid_v dtype=int32 memory=register
scalar seq_len dtype=int32 memory=register
scalar seq_offset dtype=int32 memory=register

# O Operations
load pid_s <- program_id(0)
load pid_v <- program_id(1)
load seq_len <- seq_lens[pid_s]
compute seq_offset <- sum(seq_lens[0:pid_s])
alloc acc <- full([BLOCK_V], -inf, dtype=fp32)
load vocab_tile <- logits[seq_offset + row, pid_v * BLOCK_V : (pid_v + 1) * BLOCK_V]
compute vocab_tile <- log(1.0 + relu(vocab_tile))
compute acc <- maximum(acc, vocab_tile)
store out[pid_s, pid_v * BLOCK_V : (pid_v + 1) * BLOCK_V] <- acc

# C Control
parallel pid_s over num_seq
parallel pid_v over cdiv(vocab_size, BLOCK_V)
guard pid_s < num_seq
guard pid_v < cdiv(vocab_size, BLOCK_V)
for row in 0:seq_len
guard row < seq_len
end

# H Target Hints
target=triton_gcu
num_warps=1
```

## Host Plan

```json
{"applicability":"required","affected_scope":["ModelNew.forward dispatch path","Triton kernel launch","output buffer allocation"],"state_owner":"ModelNew instance","lifetime":"model lifetime; the Triton kernel is JIT-compiled once and cached by the Triton runtime","allocation_reuse":"a single contiguous [num_seq, vocab_size] fp32 output buffer is allocated per forward call and sliced into the returned Python list; no cross-forward buffer caching is introduced in this round","cache_key":["num_seq","vocab_size","dtype","device"],"invalidation":"no persistent output cache in this round; each forward allocates a fresh output buffer","concurrency":"one ModelNew instance is not shared across concurrent forwards; output tensors are per-call","device_stream_behavior":"caller-selected device and current stream are preserved; no explicit device context is introduced; per-sequence offsets are computed on-device inside the kernel to avoid the seq_lens.tolist() D2H synchronization","unchanged_behavior":["returned Python list of num_seq tensors","each output tensor shape [vocab_size]","each output tensor dtype fp32","each output tensor on the caller-selected device","numerical semantics log(1+relu(logits)) max-pooled per sequence","dense GELU LayerNorm decoder matmul pipeline unchanged","load_state_dict compatibility"]}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-001","intervention":"fuse relu, log1p, and per-sequence max pooling into a single Triton kernel launched once per forward, eliminating the Python for-loop over seq_lens.tolist() and the D2H synchronization it triggers, and replacing six device kernels (relu, log1p, and 4x per-chunk max) with one fused reduction kernel","expected_causal_chain":["the Python for-loop over seq_lens.tolist() and the 4x per-chunk torch.max dispatches are replaced by one Triton kernel launch","host-side D2H synchronization from seq_lens.tolist() is eliminated because per-sequence offsets are computed on-device","runtime launch count per call drops from 11 toward approximately 6 (relu, log1p, and 4x max removed; one fused kernel added; the MLM head library kernels remain)","host dispatch and synchronization overhead decrease and launch count decreases","wall time decreases by at least 5%"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"runtime_launch_count_per_call","expectation":"decrease from 11 toward approximately 6"},{"name":"host_sync_count_per_call","expectation":"decrease because the seq_lens.tolist() D2H sync is eliminated"}],"guardrails":["correctness:pass","output is a Python list of num_seq tensors each [vocab_size] fp32","numerical semantics log(1+relu(decoder_logits)) max-pooled per sequence preserved within atol=1e-2 rtol=1e-2","caller-selected device and current stream preserved","dense GELU LayerNorm decoder matmul pipeline unchanged","ModelNew public constructor and forward signature unchanged","load_state_dict compatibility maintained"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`. No entry matches this intervention: the recorded failures (winner-tree expert selection, sort-32/sort-64 networks, dynamic `tl.gather` compaction, cumsum compaction) all concern grouped top-k selection networks — none apply to a straightforward elementwise-plus-segment-max-reduction fusion. No anti-pattern invalidates this path.
- Consulted `references/bottleneck-judgment.md`. GCU profiler export provides no `cat=kernel` device-duration events, so `device_ratio` is unavailable; the classification is `mixed` from the available runtime-launch evidence (11 `topsLaunchKernel` launches/call, ~111 us launch ≈ 12.9% wall) plus the host-side D2H sync loop, both of which are compressible. The fused kernel eliminates the host loop and the D2H sync, which are inseparable from the kernel change.
- `fast_libentry` is unavailable on the recorded GCU runtime (both import paths failed). This round uses direct Triton launch `kernel[(grid,)](...)` — the proven launcher path per `triton_gcu` profile. No launcher-reduction mechanism is claimed.
- `tl.dot` is Unknown on GCU and is not used: the fusion only covers elementwise relu/log1p and the per-segment max reduction, leaving the dense/decoder GEMMs as library ops. This avoids the MLU round 003 regression (fusing the decoder matmul into `tl.dot` was 33% slower).
- `num_warps=1` is the only proven value on this architecture; the sketch pins `num_warps=1`. `BLOCK_V` must be chosen conservatively: vocab_size=30522 is large and the last tile requires a bounds mask. Coder must confirm `tl.arange(0, BLOCK_V)` with the chosen BLOCK_V compiles on GCU (the recorded probe proved extents 16 and 4 only); a smaller BLOCK_V (e.g. 256 or 512) with more vocab tiles is the safe fallback if a large BLOCK_V fails to compile or regresses.
- The kernel computes `seq_offset = sum(seq_lens[0:pid_s])` on-device from the int32 `seq_lens` tensor. With num_seq=4 this is at most 3 extra `tl.load` calls per program — negligible — and avoids the D2H synchronization that `seq_lens.tolist()` triggers. `seq_lens` is int32; offsets must be computed in int32.
- GELU stays as the `nn.GELU()` library op per the project invariant (GCU may approximate to tanh; hand-writing erf GELU risks an erf/tanh mismatch against base on the same device).
- `tl.log(1.0 + x)` is numerically stable here because the relu output is non-negative, so `1 + x >= 1`; no `tl.log1p` primitive is required.
- `tl.where`, `tl.maximum`, and `tl.full` are not individually listed in the `triton_gcu` Supported table but are fundamental elementwise/register primitives; `tl.where` and `tl.max` are already proven. Coder must verify the max-reduction lowering with a local compile-and-run probe. If `tl.maximum` fails, the fallback is `tl.where(acc < x, x, acc)`; both preserve the Evaluation Contract observables and guardrails.
- The output must remain a Python `list` of 4 `[30522]` fp32 tensors; a stacked `[4, 30522]` tensor fails the harness `compare_values` type/shape check. The kernel may write a contiguous `[num_seq, vocab_size]` buffer that forward then slices into the returned list.
- The harness AST loader strips module-level non-literal assignments, so the `@triton.jit` kernel definition and any initialization must live in a ClassDef body or a loader-retained location. No `fast_libentry` wrapping is introduced.
- Per `references/invariants.md`, output tensors are allocated per forward; no cross-forward buffer caching is introduced this round, avoiding concurrency/invalidation complexity.

## Rationale and Evidence

The accepted baseline report (`rounds/report_000.md`) records: benchmark wall median 0.862541 ms; GCU device duration unavailable (no `cat=kernel` events); 11 runtime launches per forward call (all `topsLaunchKernel`), ~111 us launch ≈ 12.9% of wall. The 11 launches decompose into the MLM head library kernels (dense GEMM, GELU, LayerNorm, decoder GEMM), elementwise relu/log1p, 4x per-sequence `chunk.max(dim=0)`, and the `seq_lens.tolist()` host-side D2H sync loop.

This round's intervention is the structural fusion target already validated on MLU (round 001, +33.39%): fuse relu + log1p + per-sequence max pooling into one Triton kernel, and compute per-sequence offsets on-device to eliminate the `seq_lens.tolist()` D2H sync and the Python for-loop. The expected causal chain is direct — 5 device launches (relu, log1p, 4x max) plus one D2H sync and the host dispatch loop collapse into a single fused launch. The MLM head GEMMs are deliberately left as library ops: the dense/decoder matmuls are already fused library kernels, and fusing the decoder matmul into `tl.dot` was proven a 33% regression on MLU (round 003) and remains Unknown on GCU.

Honest GCU feasibility assessment: the device-side prefix scan (recomputing each sequence's offset from the int32 `seq_lens` via a bounded in-kernel sum over `program_id(0)`) is directly supported by `tl.program_id` (axis 0, proven) plus `tl.load`; num_seq=4 bounds it to 3 extra loads. The fused kernel grid is `(num_seq, num_vocab_tiles)` with `num_warps=1`; the risk is the large `BLOCK_V` tiling over vocab_size=30522, mitigated by conservative `BLOCK_V` and a last-tile mask. `tl.dot` and `fast_libentry` are not required. The expected ~8% wall improvement is conservative relative to MLU's 33%: GCU's launch overhead is a smaller 12.9% of wall, but removing 5 of 11 launches plus the D2H sync and host loop still clears the 5% adoption threshold if the fused kernel is no slower per element than the library elementwise ops.

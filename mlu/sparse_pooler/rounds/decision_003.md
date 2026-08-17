# Decision 003

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"003","reference_implementation":"triton_sparse_pooler_001.py","reference_report":"rounds/report_001.md","language":"triton","backend":"mlu","target_profile":"triton_mlu","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"mixed","change_family":"kernel-matmul-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"device-bound","intervention":"fuse the decoder matmul (via tl.dot with K-dimension tiling), bias addition, relu, log1p, and per-sequence max reduction into a single Triton kernel launched once per forward, eliminating the library MLUFusedMatMulGepm decoder matmul kernel (90.36 us/call) and the existing fused _sparse_pooler_max_kernel (98.73 us/call) and avoiding materialization of the intermediate logits tensor [total_seq, vocab_size] in global memory","allowed_changes":["new Triton kernel that fuses decoder matmul via tl.dot, bias addition, relu, log1p, and per-segment max reduction into one kernel with K-dimension tiling over hidden_size=768","ModelNew.forward dispatch path: replace self.decoder(self.layer_norm(self.act(self.dense(hidden_states)))) with self.layer_norm(self.act(self.dense(hidden_states))) followed by the fused kernel launch, passing decoder weight and bias tensor pointers and strides to the kernel","on-device seq_offset prefix scan and seq_len load preserved from the accepted kernel","K-dimension tiling (BLOCK_K) of the tl.dot accumulation to keep weight and hidden tiles within register limits","optional local probe of BLOCK_V and BLOCK_K values subject to compile and correctness verification, with num_warps=1 as the normative launch configuration"],"invariants":["ModelNew public constructor and forward signature unchanged","output is a Python list of num_seq tensors each of shape [vocab_size] dtype fp32 device mlu:0","numerical semantics: log(1+relu(decoder(LayerNorm(GELU(Dense(hidden))))) max-pooled per sequence within atol=1e-2 rtol=1e-2 equal_nan=True","caller-selected device and current stream preserved","dense, GELU, and LayerNorm remain PyTorch library ops unchanged","decoder weight and bias remain nn.Linear parameters managed by the module; load_state_dict compatibility maintained","num_warps=2 must not be used (known to fail on this runtime per triton_mlu target profile)"],"expected_wall_improvement_pct":8.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor hidden shape=[total_seq, hidden_size] dtype=fp32 layout=row_major memory=global
tensor decoder_weight shape=[vocab_size, hidden_size] dtype=fp32 layout=row_major memory=global
tensor decoder_bias shape=[vocab_size] dtype=fp32 layout=contiguous memory=global
tensor seq_lens shape=[num_seq] dtype=int32 layout=contiguous memory=global
tensor out shape=[num_seq, vocab_size] dtype=fp32 layout=row_major memory=global
tile hidden_tile shape=[BLOCK_M, BLOCK_K] dtype=fp32 memory=register
tile weight_tile shape=[BLOCK_V, BLOCK_K] dtype=fp32 memory=register
tile bias_tile shape=[BLOCK_V] dtype=fp32 memory=register
tile logits_tile shape=[BLOCK_M, BLOCK_V] dtype=fp32 memory=register
tile acc shape=[BLOCK_V] dtype=fp32 memory=register
scalar pid_s dtype=int32 memory=register
scalar pid_v dtype=int32 memory=register
scalar seq_offset dtype=int32 memory=register
scalar seq_len dtype=int32 memory=register

# O Operations
load pid_s <- program_id(0)
load pid_v <- program_id(1)
load seq_len <- seq_lens[pid_s]
compute seq_offset <- sum(seq_lens[0:pid_s])
alloc logits_tile <- zeros([BLOCK_M, BLOCK_V], dtype=fp32)
load hidden_tile <- hidden[seq_offset:seq_offset+seq_len, k:k+BLOCK_K]
load weight_tile <- decoder_weight[pid_v*BLOCK_V:pid_v*BLOCK_V+BLOCK_V, k:k+BLOCK_K]
compute logits_tile <- logits_tile + dot(hidden_tile, weight_tile.T)
load bias_tile <- decoder_bias[pid_v*BLOCK_V:pid_v*BLOCK_V+BLOCK_V]
compute logits_tile <- logits_tile + bias_tile
compute logits_tile <- log(1 + relu(logits_tile))
compute acc <- max(logits_tile, axis=0)
store out[pid_s, pid_v*BLOCK_V:pid_v*BLOCK_V+BLOCK_V] <- acc

# C Control
parallel pid_s over num_seq
parallel pid_v over cdiv(vocab_size, BLOCK_V)
guard pid_s < num_seq
guard pid_v * BLOCK_V < vocab_size
for k in 0:hidden_size stride BLOCK_K
guard k < hidden_size
end

# H Target Hints
target=triton_mlu
num_warps=1
```

## Host Plan

```json
{"applicability":"required","affected_scope":["ModelNew.forward dispatch path","Triton kernel launch","output tensor allocation","decoder weight and bias parameter access"],"state_owner":"ModelNew instance","lifetime":"model lifetime; kernel compiled once and cached by Triton JIT; decoder weight and bias are nn.Linear parameters managed by the module and updated by load_state_dict","allocation_reuse":"output [num_seq, vocab_size] fp32 allocated per forward via torch.empty; intermediate logits tensor [total_seq, vocab_size] is no longer materialized in global memory because the decoder matmul is fused into the Triton kernel; no cross-forward buffer caching is introduced in this round","cache_key":["seq_lens shape","num_seq","vocab_size","hidden_size","dtype","device"],"invalidation":"no persistent output cache in this round; kernel autotune cache managed by Triton runtime; decoder weight and bias parameter updates via load_state_dict are reflected in the next forward because the kernel reads weight and bias pointers from the live nn.Linear parameters","concurrency":"one ModelNew instance is not shared across concurrent forwards; decoder weights and bias are read-only during forward execution","device_stream_behavior":"caller-selected device and current stream are preserved; no explicit torch.mlu.device() context is introduced; decoder weight, bias, hidden_states, seq_lens, and output all reside on the same mlu:0 device; kernel launch inherits the current stream","unchanged_behavior":["returned Python list of num_seq tensors","each output tensor shape [vocab_size]","each output tensor dtype fp32","each output tensor device mlu:0","numerical semantics log(1+relu(decoder(LayerNorm(GELU(Dense(hidden))))) max-pooled per sequence","dense GELU LayerNorm pipeline remains PyTorch library ops unchanged","ModelNew public constructor and forward signature unchanged","load_state_dict compatibility maintained","decoder weight and bias remain nn.Linear parameters"]}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-003","intervention":"fuse the decoder matmul (via tl.dot with K-dimension tiling), bias addition, relu, log1p, and per-sequence max reduction into a single Triton kernel launched once per forward, eliminating the library MLUFusedMatMulGepm decoder matmul kernel (90.36 us/call) and the existing fused _sparse_pooler_max_kernel (98.73 us/call) and avoiding materialization of the intermediate logits tensor [total_seq, vocab_size] in global memory","expected_causal_chain":["the decoder matmul (MLUFusedMatMulGepm, 90.36 us/call) is fused into the Triton kernel via tl.dot, eliminating the library matmul kernel","the intermediate logits tensor [83, 30522] fp32 (10.16 MB) is no longer materialized in global memory, saving the matmul output write and the fused reduction kernel input read","the new fused matmul+bias+relu+log1p+max kernel replaces two device kernels (decoder matmul + existing fused reduction) with one","total device kernel count per call decreases from 5 to 4","device time decreases from 210.12 us/call because the combined cost of the two replaced kernels (189.09 us/call) exceeds the expected cost of the single fused kernel","wall time decreases by at least 5%"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"decoder_matmul_kernel_count_per_call","expectation":"decrease from 1 to 0; the MLUFusedMatMulGepm library kernel is eliminated because the matmul is fused into the Triton kernel via tl.dot"},{"name":"total_kernel_count_per_call","expectation":"decrease from 5 to 4; the decoder matmul and the existing fused reduction kernel are replaced by one fused matmul+relu+log1p+max kernel; dense matmul, LayerNorm, and GELU are unchanged"},{"name":"device_us_per_call","expectation":"decrease from 210.12 us/call; the combined cost of the two replaced kernels is 189.09 us/call (90.36 + 98.73), and the new fused kernel is expected to cost less because it avoids materializing and re-reading the intermediate logits tensor"},{"name":"fused_kernel_us_per_call","expectation":"the new fused matmul+relu+log1p+max kernel costs less than the 189.09 us/call combined cost of the decoder matmul (90.36 us/call) and the existing fused reduction kernel (98.73 us/call) it replaces"}],"guardrails":["correctness:pass","output is a Python list of num_seq tensors each [vocab_size] fp32 mlu:0","numerical semantics log(1+relu(decoder(LayerNorm(GELU(Dense(hidden))))) max-pooled per sequence preserved within atol=1e-2 rtol=1e-2 equal_nan=True","caller-selected device and current stream preserved","dense GELU LayerNorm pipeline unchanged","ModelNew public constructor and forward signature unchanged","load_state_dict compatibility maintained","kernel_count_per_call decreases (not increases)","num_warps=2 is not used (known to fail on this runtime)"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`. No entry matches this intervention. The recorded failures concern winner-tree expert selection, sort networks, dynamic `tl.gather` compaction, and cumsum compaction in grouped top-k — none apply to a matmul+elementwise+reduction fusion via `tl.dot`.

- Consulted `references/bottleneck-judgment.md`. The candidate device ratio is 0.346 (mixed). The decoder matmul (`MLUFusedMatMulGepm`) at 90.36 us/call is 43.0% of candidate device time and is the second-largest device kernel. The existing fused `_sparse_pooler_max_kernel` at 98.73 us/call is 47.0%. Together they account for 90.0% of device time. The intervention fuses these two kernels into one, targeting a single, separately observable device mechanism: the combined matmul+elementwise+reduction cost and the intermediate tensor materialization. The three non-fused kernels (dense matmul 8.42, LayerNorm 7.21, GELU 5.40 us/call) are outside the change boundary and unchanged.

- Consulted `prompts/coder_targets/triton_mlu.md`. `tl.dot` is Supported with the constraint: "Inputs are 2-D with matching inner dimensions; dtype and shape restrictions must be probed for the current runtime." The repository evidence is from `fused_moe/triton_fused_moe_005.py` and `flexattention/triton_flexattention_003.py`. Coder must verify via a local compile-and-run probe that `tl.dot` works with fp32 inputs of shape `[BLOCK_M, BLOCK_K] x [BLOCK_K, BLOCK_V]` where BLOCK_M is small (max seq_len = 25, padded to BLOCK_M = 32) and BLOCK_K tiles the hidden_size = 768. If `tl.dot` does not support these shapes or dtypes on this runtime, the fallback is NOT to keep the decoder as a library op — that would be a `major-deviation` because the intervention IS the matmul fusion. Instead, Coder must report a `capability-miss` so the round terminates cleanly and the next round can pursue a different change family.

- The decoder weight is stored by `nn.Linear` as `[vocab_size, hidden_size] = [30522, 768]` (PyTorch convention: `weight[out_features, in_features]`). The kernel needs `weight_tile = decoder_weight[v_start:v_start+BLOCK_V, k:k+BLOCK_K]` loaded as `[BLOCK_V, BLOCK_K]`, then used transposed in `tl.dot(hidden_tile, weight_tile.T)` to compute `[BLOCK_M, BLOCK_V]`. Coder must handle the stride layout correctly. Pre-transposing the weight at init time is possible but must not break `load_state_dict` (the harness runs `model_new.load_state_dict(model.state_dict())` before timing, which updates `self.decoder.weight` but not a pre-transposed copy). The simplest approach is to load with transposed strides directly in the kernel; Coder should probe which approach compiles and runs correctly on this runtime.

- The decoder bias is `[vocab_size] = [30522]`. Each vocab tile loads `bias[v_start:v_start+BLOCK_V]` as a 1-D tile and broadcasts it across the `[BLOCK_M, BLOCK_V]` logits tile. This is a standard broadcast add.

- K-dimension tiling: the hidden_size = 768 is too large to load the full weight tile `[BLOCK_V, 768]` at once (e.g., with BLOCK_V = 256, that is 768 KB of fp32). Coder must tile the K dimension with BLOCK_K (e.g., 64 or 128) and accumulate the `tl.dot` result in a `[BLOCK_M, BLOCK_V]` accumulator across K tiles. This is the standard tiled-matmul pattern.

- BLOCK_M must be a compile-time constant >= max(seq_len). With `seq_lens = [20, 25, 18, 20]`, `max(seq_len) = 25`. Coder should set `BLOCK_M = 32` (next power of 2) and mask out rows `>= seq_len` in the `tl.dot` and max reduction. The existing on-device `seq_len` load and prefix scan for `seq_offset` are preserved from the accepted kernel.

- `tl.maximum`, `tl.where`, and `tl.log` were proven to work in Round 001. `tl.dot` is new to this project but is Supported per the target profile. `tl.zeros` is Supported for the dot accumulator initialization. No other new primitives are introduced.

- `num_warps=1` is the proven launch configuration. `num_warps=2` failed in the flexattention experiment and must not be used. Coder may probe other `num_warps` values locally but must fall back to `num_warps=1` if a probe fails to compile, produces incorrect output, or does not improve wall time.

- The harness AST loader strips module-level non-literal assignments. The Triton kernel definition is at module level (decorated with `@triton.jit`) and was retained in Rounds 001 and 002. No changes to the module structure are needed. If `fast_libentry` is used for launcher reduction, it must be initialized from a retained location (class body or function scope); however, this round does not require `fast_libentry` — the Host Plan specifies no cross-forward caching or launcher reduction beyond the kernel fusion itself.

- Per `references/invariants.md`, this change does not introduce output buffer caching, cross-forward state, or a device context. The output tensor is still allocated per-forward with `torch.empty`. The Host Plan is required because the change is mixed (kernel + host dispatch), but it specifies no allocation reuse.

- Round 002 evidence shows that `BLOCK_V=1024` remains the best-known vocab tiling for the existing fused kernel on this runtime. Coder should use `BLOCK_V=1024` or smaller (e.g., 256 or 512) as the starting point for the new fused kernel, given the additional register pressure from the `tl.dot` accumulator and weight tiles. The optional probe space is `BLOCK_V in {256, 512, 1024}` and `BLOCK_K in {64, 128, 256}`, subject to compile and correctness verification.

## Rationale and Evidence

The accepted Round 001 report (`rounds/report_001.md`) records:

- Benchmark wall time median: 0.606758 ms (606.76 us/call), a 33.39% improvement over the
  baseline that was accepted in Round 001.
- Device time: 210.12 us/call; device_ratio: 0.346 (mixed).
- Kernel count: 5 per call (down from 10).
- The fused `_sparse_pooler_max_kernel` is the dominant device kernel at 98.73 us/call (47.0%
  of device time). It is slower on the device than the 6 library kernels it replaced (67.87
  us/call combined), a 30.86 us/call device regression.
- The decoder matmul (`MLUFusedMatMulGepm`) is the second-largest device kernel at 90.36
  us/call (43.0% of device time). It is a PyTorch library op that materializes the intermediate
  logits tensor `[83, 30522]` fp32 (10.16 MB) in global memory.
- The remaining host time (~396 us/call, ~65% of wall) is launcher, wrapper, allocation, and
  harness-fixed cost.

The rejected Round 002 report (`rounds/report_002.md`) records:

- BLOCK_V 1024→2048 was falsified: the fused kernel got slower (99.71 → 102.42 us/call),
  confirming the bottleneck in the fused kernel is per-program elementwise compute, not
  launch-dispatch overhead. The `kernel-tile-tuning` change family is exhausted for this
  kernel on this runtime.

The intervention targets the decoder matmul (90.36 us/call) and the existing fused reduction
kernel (98.73 us/call) together. These two kernels account for 189.09 us/call — 90.0% of
candidate device time. The decoder matmul writes the intermediate logits tensor `[83, 30522]`
(10.16 MB fp32) to global memory, and the fused reduction kernel reads it back. Fusing the
matmul into the reduction kernel via `tl.dot` eliminates both the library matmul kernel and
the intermediate tensor materialization, replacing 189.09 us/call of device kernels with one
fused matmul+relu+log1p+max kernel.

The expected wall improvement of 8.0% is justified as follows:

1. The two replaced kernels cost 189.09 us/call combined (90.36 + 98.73). The new fused kernel
   performs the same computation but avoids writing and re-reading the 10.16 MB intermediate
   tensor. Even if the `tl.dot` matmul is no faster than the library `MLUFusedMatMulGepm`, the
   elimination of the intermediate tensor traffic and the merger of two kernel launches into
   one is expected to reduce the combined device cost. A conservative estimate is that the new
   fused kernel takes 130 us/call, saving 59 us/call of device time.

2. Removing the `self.decoder(...)` library op call from the host dispatch path eliminates one
   PyTorch op dispatch and the associated `torch.empty` allocation for the intermediate logits
   tensor, saving an estimated 5-10 us/call of host overhead.

3. Total expected wall savings: ~65 us/call out of 607 us/call = ~10.7%. The 8.0% expectation
   is conservative, below the 10.7% estimate but well above the 5% adoption threshold.

The `change_family` is `kernel-matmul-fusion`, which is different from the last three rounds
(Round 001: `kernel-fusion`, Round 002: `kernel-tile-tuning`, Round 000: baseline). This is
required by the v2 Designer contract after a `no-improvement` result.

The primary risk is that `tl.dot` with small M (BLOCK_M = 32, actual seq_len = 18-25) may be
inefficient on the MLU590-H8 architecture. The `fused_moe` and `flexattention` evidence kernels
use `tl.dot` with larger M dimensions. If the `tl.dot` path is slower than the library matmul
for these shapes, the new fused kernel may not beat 189.09 us/call, and the hypothesis will be
falsified. This risk is accepted because the intervention is falsifiable: the mechanism
observables (`decoder_matmul_kernel_count_per_call`, `device_us_per_call`,
`fused_kernel_us_per_call`) will directly reveal whether the matmul fusion succeeded or failed.

The dense matmul (`MLUFusedMatMulGepdot`, 8.42 us/call), LayerNorm (7.21 us/call), and GELU
(5.40 us/call) are left as PyTorch library ops. Fusing them would require a larger change
boundary (fusing the full MLM head) and is a candidate for a future round if this round
succeeds and these kernels become the new dominant bottleneck.

Host-side launcher reduction and allocation reuse (the remaining ~396 us/call host time)
require Host Plan lifecycle changes and are deferred to a future round. The current round
targets the single largest device-side opportunity: the decoder matmul and the intermediate
tensor materialization.

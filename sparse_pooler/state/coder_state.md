# Coder State

Concise implementation facts and resume context for the current candidate.
No canonical-state claims.

## Current round

- Round: 003
- Phase: coding (complete from Coder's side; awaiting Orchestrator handoff to Verifier)
- Decision: `rounds/decision_003.md` (SHA-256 `8f78d0425148e387ba82fc827012c63440e8d38edcdf19750a0e79825c8505bb`, `proceed`, change_scope=mixed, change_family=kernel-matmul-fusion)
- Source canonical: `triton_sparse_pooler_001.py` (SHA-256 `182f2ebb32b9762f5f19dae00a30e25618bcdb0b8563f7acbc3d9ea9ca348dcd`)
- Candidate: `triton_sparse_pooler_003.py` (SHA-256 `3406f7c9a731e1fd7560ab95bf1d903fd4d6f8287c5880d9801e9d41e5ed7860`)
- Result file: `rounds/coder_result_003.md`
- Classification: `candidate-ready`

## Candidate facts

- New kernel `_sparse_pooler_fused_matmul_max_kernel` replaces `_sparse_pooler_max_kernel`. Fuses decoder matmul (via `tl.dot` with K-dimension tiling), bias addition, relu, log1p, and per-segment max reduction into one kernel.
- `ModelNew.forward` on the max-pooling path now computes `x = self.layer_norm(self.act(self.dense(hidden_states)))` and passes `x`, `self.decoder.weight`, `self.decoder.bias`, `seq_lens`, `out` to the fused kernel. The library `self.decoder(...)` call is removed from the max path.
- `pooling == "sum"` fallback retains the library decoder call (`logits = self.decoder(x)`) and the Python sum loop. Off the measured hot path.
- Grid: `(num_seq, num_vocab_tiles)` = `(4, triton.cdiv(30522, 512))` = `(4, 60)`.
- `BLOCK_M=32` (>= max seq_len 25; rows >= seq_len masked to 0 in the dot and -inf in the max).
- `BLOCK_K=64` (K-dimension tiling over hidden_size=768; 12 K-tiles).
- `BLOCK_V=512` (last vocab tile covers 30208..30719; 314 in-bounds, 198 masked).
- `num_warps=1` (Constrained, proven, normative). `num_warps=2` not used.
- `tl.dot` with `input_precision="ieee"`: local probe established that the default `tf32` precision exceeds the project's 1e-2 tolerance for this shape; `ieee` matches the library fp32 matmul within ~2e-5. `tf32x3` is Unsupported on this runtime.
- Weight layout: `decoder.weight` is `[vocab_size, hidden_size] = [30522, 768]` (nn.Linear convention). Kernel loads `weight_tile = decoder_weight[v_offs[:, None], k_offs[None, :]]` with strides `(weight.stride(0), weight.stride(1))` producing `[BLOCK_V, BLOCK_K]`; `tl.trans(weight_tile)` presents it as `[BLOCK_K, BLOCK_V]` to `tl.dot`. No init-time transpose; `load_state_dict` compatibility preserved.
- Bias add: `bias_tile = tl.load(decoder_bias_ptr + v_offs, mask=v_mask, other=0.0)` is `[BLOCK_V]`, broadcast via `logits + bias_tile[None, :]`.
- Relu + log1p: `tl.where(logits > 0.0, logits, 0.0)` then `tl.log(1.0 + logits)`.
- Per-segment max: `tl.where(m_mask[:, None], logits, -inf)` then `tl.max(logits, axis=0)`.
- On-device `seq_len` load and `seq_offset = sum(seq_lens[0:pid_s])` via bounded `for i in range(pid_s)`; preserved from the accepted Round 001 kernel.
- `dense`, `GELU`, `LayerNorm`, `decoder` remain `nn.Module` attributes; the four parameters are unchanged, so `load_state_dict(model.state_dict())` accepts the reference state dict.
- `get_inputs` and `get_init_inputs` preserved byte-for-byte.
- No explicit `torch.mlu.device()` context; caller-selected device and current stream preserved.
- `kernel_count_per_call` decreases from 5 to 4 by construction: library decoder matmul (`MLUFusedMatMulGepm`) and existing fused reduction kernel are replaced by one fused kernel; dense matmul, LayerNorm, GELU unchanged.
- Intermediate logits tensor `[83, 30522]` fp32 (10.16 MB) no longer materialized in global memory.

## Probe outcomes (resume reference)

- `tl.dot` capability probe: compiles and runs for `[BLOCK_M=32, BLOCK_K] x [BLOCK_K, BLOCK_V]` fp32 on this runtime. `BLOCK_K=256` with `BLOCK_V >= 512` and `BLOCK_K=128` with `BLOCK_V=1024` exceed NRAM limit (524288 bytes) at `num_warps=1`.
- `tl.dot` precision probe: `input_precision="ieee"` required (max_diff ~1.9e-5, passes 1e-2 and 1e-3). `tf32` fails 1e-2 (max_diff ~0.08). `tf32x3` Unsupported (compile error).
- `ast.parse` ok; harness `load_ks_module` exposes ModelNew/get_inputs/get_init_inputs.
- Correctness smoke vs `base.py`: 4/4 outputs `allclose(atol=1e-2, rtol=1e-2, equal_nan=True)`, max_abs_diff in [2.38e-07, 2.98e-07].
- Harness end-to-end smoke (warmup=5, repeat=5): `PASS accuracy; v0=0.891860 ms, v1=0.823640 ms, speedup=1.083x` (smoke only; Verifier owns the 50/100 measurement).
- All throwaway probe and smoke scripts deleted before handoff.

## Repair budget

- Repairs used this round: 1 of 2. Repair #1 was a non-semantic tile-size accommodation: `BLOCK_K=128 -> 64` and `BLOCK_V=1024 -> 512` (within the decision's allowed probe space) to resolve the NRAM out-of-resource compile defect.

## Ownership

- Coder owns: `triton_sparse_pooler_003.py`, `rounds/coder_result_003.md`, `state/coder_state.md`.
- Coder must not edit: decision, team-state, project.md, base.py, harness, target profile, Verifier-owned files.

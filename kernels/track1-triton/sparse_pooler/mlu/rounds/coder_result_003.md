# Coder Result 003

## Metadata

```json
{
  "schema_version": 1,
  "round": "003",
  "result": "candidate-ready",
  "result_reason": "candidate conforms to the immutable decision; fused matmul+bias+relu+log1p+max kernel via tl.dot with K-dimension tiling replaces the library decoder matmul and the existing fused reduction; tl.dot proven locally with input_precision=ieee; correctness smoke and harness end-to-end pass within tolerance",
  "source_canonical_path": "triton_sparse_pooler_001.py",
  "source_canonical_sha256": "182f2ebb32b9762f5f19dae00a30e25618bcdb0b8563f7acbc3d9ea9ca348dcd",
  "decision_path": "rounds/decision_003.md",
  "decision_sha256": "8f78d0425148e387ba82fc827012c63440e8d38edcdf19750a0e79825c8505bb",
  "selected_profile": "triton_mlu",
  "runtime_fingerprint": {
    "triton_version": "3.2.0",
    "backend_target": "BangDriver (mlu)",
    "backend_version": "torch_mlu 1.32.0+torch2.11.0; MLU driver 6.5.49",
    "device_arch": "MLU590-H8 (capability 5.0)"
  },
  "candidate_path": "triton_sparse_pooler_003.py",
  "candidate_sha256": "3406f7c9a731e1fd7560ab95bf1d903fd4d6f8287c5880d9801e9d41e5ed7860"
}
```

## Decision validation

- Command: `python3 /projs/framework/lipenghui/projects/kernelswift/skills/kernel-opt-loop/scripts/validate_decision.py /projs/framework/lipenghui/projects/kernelswift/sparse_pooler/rounds/decision_003.md --expected-profile triton_mlu`
- Exit code: 0
- Output: `valid: true`
- Language (`triton`), backend (`mlu`), and target profile (`triton_mlu`) match the manifest's Identity and Match rules. No environment-blocked condition.

## Runtime fingerprint verification

- `triton_version=3.2.0` (matches `project.md#runtime-fingerprint`)
- `torch_mlu_version=1.32.0+torch2.11.0` (matches)
- `driver_version=6.5.49` (matches)
- `device_name=MLU590-H8` (matches)
- `device_capability=(5, 0)` (matches cap 5.0)

No fingerprint mismatch. The runtime is the one the decision was authored against.

## Primitive and hint conformance

Round 003 introduces one new Triton primitive (`tl.dot`) and one new host-side stride pattern (loading `decoder_weight` with transposed strides). The kernel also reuses primitives proven in Round 001 (`tl.load`, `tl.store`, `tl.arange`, `tl.program_id`, `tl.zeros`, `tl.where`, `tl.log`, `tl.maximum`/`tl.max`). `tl.trans` is used to present the `[BLOCK_V, BLOCK_K]` weight tile as `[BLOCK_K, BLOCK_V]` to `tl.dot` without an init-time weight transpose; this preserves `load_state_dict` compatibility because the `nn.Linear` weight remains in the canonical `[vocab_size, hidden_size]` layout.

| Primitive / hint | Decision status | Profile status | Probe outcome | Conformance |
|---|---|---|---|---|
| `tl.dot` | Required (decoder matmul fusion) | Supported; "dtype and shape restrictions must be probed for the current runtime" | Local probe on this runtime: `tl.dot` compiles and runs for `[BLOCK_M=32, BLOCK_K] x [BLOCK_K, BLOCK_V]` fp32 with `BLOCK_K in {64, 128}` and `BLOCK_V in {256, 512, 1024}`. The default `input_precision="tf32"` produces ~0.08 max_abs_diff against the reference fp32 matmul (output magnitude ~100), which exceeds the project's 1e-2 tolerance. `input_precision="ieee"` produces ~1.9e-5 max_abs_diff and passes both 1e-2 and 1e-3 tolerance. `input_precision="tf32x3"` is Unsupported on this runtime (compile error). The candidate uses `input_precision="ieee"` | Supported with the `input_precision="ieee"` constraint established by the local probe |
| `tl.trans` | Required (present weight tile as `[BLOCK_K, BLOCK_V]` to `tl.dot` without init-time transpose) | Not in Supported table; standard Triton primitive | Local probe: the transpose-stride load pattern (`b_nk` stored as `[N,K]` row-major, loaded with `stride_b_k=b_nk.stride(1), stride_b_n=b_nk.stride(0)`, then `tl.trans(weight_tile)` inside `tl.dot`) compiles and matches the reference `a @ b_nk.T` within tolerance with `input_precision="ieee"` | Supported via local probe; preserves `load_state_dict` compatibility |
| `tl.zeros((BLOCK_M, BLOCK_V), dtype=tl.float32)` | Required (dot accumulator init) | Supported | Local probe: compiled and ran for `BLOCK_M=32, BLOCK_V in {256, 512, 1024}` | Supported |
| `tl.max(logits, axis=0)` | Required (per-segment max reduction) | Not in Supported table; `tl.maximum` was proven in Round 001 | Local correctness smoke: 4/4 outputs match base.py within tolerance | Supported via local probe (semantically equivalent to the Round 001 `tl.maximum`-based running max for the fused path) |
| `tl.load` (2-D, masked) | Required (hidden tile, weight tile, bias tile, seq_lens) | Supported | Local probe + correctness smoke: all loads compile and produce correct output; masks handle rows >= seq_len, k >= hidden_size, and v >= vocab_size | Supported; mask and bounds validated for the new tile shapes |
| `tl.store` | Required (write `acc` to `out[pid_s, v_offs]`) | Supported | Correctness smoke confirms output shape, dtype, and device | Supported |
| `tl.arange`, `tl.program_id` | Required (grid mapping) | Supported | Grid `(num_seq, num_vocab_tiles)` = `(4, cdiv(30522, BLOCK_V))` | Supported; grid mapping preserves the decision's control structure |
| `tl.where`, `tl.log` | Required (relu, log1p, row mask for max) | Proven in Round 001 | Correctness smoke confirms | Supported via Round 001 evidence |
| `num_warps=1` | Required (Constrained, normative) | Constrained: `num_warps=1` proven; `num_warps=2` failed | Used `num_warps=1` exactly as the decision and profile require | Proven value, no fallback needed; `num_warps=2` not used |
| `BLOCK_M=32` | Required (normative target hint; >= max seq_len 25) | Not a primitive; a `tl.constexpr` tiling parameter | Compiled and ran; rows >= seq_len masked to -inf in the max reduction | Conforms; masks handle the padding rows |
| `BLOCK_V=512` | Optional (probe space {256, 512, 1024}) | Not a primitive; a `tl.constexpr` tiling parameter | `BLOCK_V=1024` with `BLOCK_K=128` failed compile (NRAM 815296 > 524288 limit). `BLOCK_V=512` with `BLOCK_K=64` compiled and passed correctness. This is a non-semantic tile-size accommodation within the decision's allowed probe space | Conforms; tile-size accommodation recorded as a conformance note, not a design change |
| `BLOCK_K=64` | Optional (probe space {64, 128, 256}) | Not a primitive; a `tl.constexpr` tiling parameter | `BLOCK_K=128` with `BLOCK_V=1024` failed compile (NRAM). `BLOCK_K=256` with `BLOCK_V >= 512` failed compile (NRAM). `BLOCK_K=64` with `BLOCK_V=512` compiled and passed correctness | Conforms; tile-size accommodation within the allowed probe space |
| `input_precision="ieee"` | Not named in the decision; a precision attribute on `tl.dot` | Not in the profile; a `tl.dot` keyword argument | Local probe established that the default `tf32` precision exceeds the project's 1e-2 tolerance for this shape; `ieee` matches the library fp32 matmul within ~2e-5. This is a precision accommodation that preserves all normative semantics (the project's numerical semantics require `log(1+relu(decoder(LayerNorm(GELU(Dense(hidden)))))` max-pooled per sequence within atol=1e-2 rtol=1e-2; `ieee` is the only `tl.dot` precision that satisfies this) | Conformance note: a target-language accommodation that preserves all normative semantics, not a new design. Without `ieee`, the matmul fusion cannot meet the project's tolerance and the round would have to return `capability-miss` |
| `fast_libentry` | Not required this round (Host Plan specifies no launcher reduction) | Allowed fallback: ordinary Triton launch | Ordinary `@triton.jit` + grid launch used | N/A — Host Plan does not require launcher reduction |

No Unsupported or unprovable Unknown primitive is required. No silent substitution of a normative construct occurred. The `input_precision="ieee"` attribute is a target-language precision accommodation that preserves all normative semantics; it is recorded here as a conformance note, not a new design.

## Implementation summary

The candidate is a copy of `triton_sparse_pooler_001.py` with two changes:

1. The `_sparse_pooler_max_kernel` is replaced by `_sparse_pooler_fused_matmul_max_kernel`, which fuses the decoder matmul (via `tl.dot` with K-dimension tiling), bias addition, relu, log1p, and per-segment max reduction into one kernel.
2. `ModelNew.forward` no longer calls `self.decoder(...)` on the max-pooling path. Instead it passes the LayerNorm output `x`, the decoder weight, and the decoder bias to the fused kernel. The `pooling == "sum"` fallback retains the library decoder call (off the measured hot path).

Kernel details:

- Grid: `(num_seq, num_vocab_tiles)` = `(4, triton.cdiv(30522, 512))` = `(4, 60)`. One program per (sequence, vocab tile).
- `BLOCK_M = 32` (>= max seq_len 25). Rows >= seq_len are masked to 0 in the `tl.dot` load (so they do not contaminate the accumulator) and to `-inf` in the final max reduction (so they never win the max).
- `BLOCK_K = 64`. K-dimension tiling over `hidden_size = 768` (12 K-tiles). The `tl.dot` accumulator is initialized to `tl.zeros((BLOCK_M, BLOCK_V), dtype=tl.float32)` and accumulated across K-tiles via `acc=logits`.
- `BLOCK_V = 512`. The last vocab tile covers offsets 30208..30719 (314 in-bounds, 198 masked). `v_mask = v_offs < vocab_size` handles the partial tile on load (`other=0.0` for the dot input, since masked vocab lanes do not contribute to any output) and on store.
- `input_precision="ieee"` on `tl.dot`. The default `tf32` precision was probed locally and exceeds the project's 1e-2 tolerance for this shape; `ieee` matches the library fp32 matmul within ~2e-5. This is a precision accommodation that preserves all normative semantics.
- Weight layout: `decoder.weight` is `nn.Linear` weight stored as `[vocab_size, hidden_size] = [30522, 768]`. The kernel loads `weight_tile = decoder_weight[v_offs[:, None], k_offs[None, :]]` with strides `(stride_weight_n, stride_weight_k) = (weight.stride(0), weight.stride(1))`, producing a `[BLOCK_V, BLOCK_K]` tile. `tl.trans(weight_tile)` presents it as `[BLOCK_K, BLOCK_V]` to `tl.dot(hidden_tile, ...)` which computes `[BLOCK_M, BLOCK_V]`. No init-time weight transpose; `load_state_dict` compatibility is preserved because the weight parameter remains in the canonical layout.
- Bias add: `bias_tile = tl.load(decoder_bias_ptr + v_offs, mask=v_mask, other=0.0)` is `[BLOCK_V]`, broadcast to `[BLOCK_M, BLOCK_V]` via `logits + bias_tile[None, :]`.
- Relu + log1p: `logits = tl.where(logits > 0.0, logits, 0.0)` then `logits = tl.log(1.0 + logits)`. Stable for x >= 0 (relu output is non-negative).
- Per-segment max: `logits = tl.where(m_mask[:, None], logits, -float("inf"))` then `acc = tl.max(logits, axis=0)`. Rows >= seq_len are forced to `-inf` so they never win the max.
- Store: `tl.store(out_ptr + pid_s * stride_out_row + v_offs, acc, mask=v_mask)`.
- On-device `seq_len` load and `seq_offset = sum(seq_lens[0:pid_s])` via a bounded `for i in range(pid_s)` loop; preserved from the accepted Round 001 kernel.
- `num_warps=1` (Constrained, proven, normative).

Host dispatch details:

- `ModelNew.forward` now computes `x = self.layer_norm(self.act(self.dense(hidden_states)))` (dense, GELU, LayerNorm remain PyTorch library ops). The decoder matmul is NOT called on the max-pooling path.
- `out = torch.empty((num_seq, vocab_size), dtype=torch.float32, device=device)` is allocated per forward (no cross-forward buffer cache; Host Plan specifies no allocation reuse).
- The fused kernel is launched with `x`, `self.decoder.weight`, `self.decoder.bias`, `seq_lens`, `out`, and the stride/shape arguments.
- `return [out[i] for i in range(num_seq)]` preserves the public output structure (list of 4 tensors, each `[30522]` fp32 `mlu:0`).
- `pooling == "sum"` fallback: the decoder is applied as a library op (`logits = self.decoder(x)`) and the Python loop sums per-sequence chunks. This is off the measured hot path (the harness uses `pooling == "max"`, the default).
- `dense`, `GELU`, `LayerNorm`, `decoder` remain `nn.Module` attributes; the four parameters are unchanged, so `load_state_dict(model.state_dict())` accepts the reference state dict.
- No explicit `torch.mlu.device()` context is introduced; the caller-selected device and current stream are preserved.
- `get_inputs` and `get_init_inputs` are preserved byte-for-byte.
- `kernel_count_per_call` decreases from 5 to 4 by construction: the library decoder matmul (`MLUFusedMatMulGepm`) and the existing fused `_sparse_pooler_max_kernel` are replaced by one fused kernel; the dense matmul, LayerNorm, and GELU library ops are unchanged.

## Attempt ledger

| # | Command | Exit code | Defect | Before candidate hash | After candidate hash |
|---|---|---:|---|---|---|
| 0 | `python3 .../validate_decision.py .../rounds/decision_003.md --expected-profile triton_mlu` | 0 | none — `valid: true` | n/a | n/a |
| 1 | Runtime fingerprint verification (`import triton, torch_mlu, torch; print versions/capability`) | 0 | none — triton 3.2.0, torch_mlu 1.32.0+torch2.11.0, MLU590-H8 cap 5.0 all match | n/a | n/a |
| 2 | `tl.dot` local probe (`_probe_tldot_003.py`): plain and transpose-stride shapes, BLOCK_M=32, BLOCK_K in {64,128,256}, BLOCK_V in {256,512,1024}, num_warps=1 | 0 | none — `tl.dot` compiles and runs; default `tf32` precision produces ~0.08 max_abs_diff (exceeds 1e-2 tolerance); probe deleted before handoff | n/a | n/a |
| 3 | `tl.dot` precision probe (`_probe_precision_003.py`): `input_precision in {tf32, ieee, tf32x3}` | 0 | none — `ieee` passes 1e-2 and 1e-3 tolerance (max_diff ~1.9e-5); `tf32` fails 1e-2 (max_diff ~0.08); `tf32x3` is Unsupported (compile error); probe deleted before handoff | n/a | n/a |
| 4 | `cp triton_sparse_pooler_001.py triton_sparse_pooler_003.py` | 0 | none — byte-identical copy | 182f2ebb... | 182f2ebb... |
| 5 | Edit candidate: replace `_sparse_pooler_max_kernel` with `_sparse_pooler_fused_matmul_max_kernel` (fused matmul+bias+relu+log1p+max via `tl.dot` with K-dimension tiling, `input_precision="ieee"`, `tl.trans(weight_tile)`, on-device seq_offset prefix scan preserved) and update `ModelNew.forward` to pass `x = self.layer_norm(self.act(self.dense(hidden_states)))`, `self.decoder.weight`, `self.decoder.bias` to the fused kernel; initial `BLOCK_M=32, BLOCK_K=128, BLOCK_V=1024` | 0 | none — edit applied | 182f2ebb... | 6750dbba... |
| 6 | `python3 -c "import ast; ast.parse(open('triton_sparse_pooler_003.py').read())"` | 0 | none — candidate parses; top-level nodes: 7x Import, FunctionDef (kernel), ClassDef (ModelNew), 2x FunctionDef (get_inputs/get_init_inputs), If (__main__) | 6750dbba... | 6750dbba... |
| 7 | `python3 -c "from auto_bench import load_ks_module; load_ks_module(Path('sparse_pooler/triton_sparse_pooler_003.py'))"` | 0 | none — loader exposes ModelNew/get_inputs/get_init_inputs; init_args=(768, 30522, 'max'); inputs=[Tensor[83,768] fp32 mlu:0, Tensor[4] int32 mlu:0] | 6750dbba... | 6750dbba... |
| 8 | Correctness smoke (`_smoke_003.py`): instantiate ModelNew, `load_state_dict` from base.py Model, forward on documented inputs, compare 4 outputs with `torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True)` | 1 | NRAM out of resource: Required 815296 > Hardware limit 524288. `BLOCK_V=1024, BLOCK_K=128` exceeded NRAM with the `tl.dot` accumulator and weight tiles. This is a compile-resource defect (non-semantic) | 6750dbba... | 6750dbba... |
| 9 | Repair #1: edit candidate `BLOCK_K=128 -> BLOCK_K=64` and `BLOCK_V=1024 -> BLOCK_V=512` (within the decision's allowed probe space {256, 512, 1024} x {64, 128, 256}) | 0 | none — tile-size accommodation within the allowed probe space | 6750dbba... | 3406f7c9... |
| 10 | Correctness smoke (`_smoke_003.py`): same as attempt 8 with `BLOCK_V=512, BLOCK_K=64` | 0 | none — 4/4 outputs match, max_abs_diff=2.98e-07, allclose=True; smoke script deleted before handoff | 3406f7c9... | 3406f7c9... |
| 11 | `python3 auto_bench.py --v0_file sparse_pooler/base.py --v1_file sparse_pooler/triton_sparse_pooler_003.py --warmup 5 --repeat 5` | 0 | none — `PASS accuracy; v0=0.891860 ms, v1=0.823640 ms, speedup=1.083x` (smoke only; Verifier produces the authoritative 50/100 measurement) | 3406f7c9... | 3406f7c9... |

One repair was used (attempt 9) out of the two-repair budget. The repair was a non-semantic tile-size accommodation: `BLOCK_K=128 -> 64` and `BLOCK_V=1024 -> 512`, both within the decision's allowed probe space. No semantic change was required.

## Probe evidence

### tl.dot capability and precision probe

Before writing the candidate, a local probe was run to verify `tl.dot` works on this runtime with the candidate's actual shapes (`[BLOCK_M=32, BLOCK_K] x [BLOCK_K, BLOCK_V]` fp32, K tiled over `hidden_size=768`). The probe covered `BLOCK_K in {64, 128, 256}` and `BLOCK_V in {256, 512, 1024}` in both plain (`B` as `[K,N]` row-major) and transpose-stride (`B` as `[N,K]` row-major, loaded with transposed strides to emulate the `decoder_weight` layout) configurations.

Findings:

- `tl.dot` compiles and runs on this runtime for all probed shapes that fit in NRAM.
- `BLOCK_K=256` with `BLOCK_V >= 512` and `BLOCK_K=128` with `BLOCK_V=1024` exceed the NRAM hardware limit (524288 bytes) at `num_warps=1`.
- The default `input_precision="tf32"` produces ~0.08 max_abs_diff against the reference fp32 matmul (output magnitude ~100), which exceeds the project's 1e-2 tolerance. This would cause the fused kernel to fail correctness.
- `input_precision="ieee"` produces ~1.9e-5 max_abs_diff and passes both 1e-2 and 1e-3 tolerance.
- `input_precision="tf32x3"` is Unsupported on this runtime (compile error).

The probe established two conformance facts:

1. `tl.dot` is Supported on this runtime for the candidate's shapes (capability is NOT a miss).
2. `input_precision="ieee"` is REQUIRED to meet the project's numerical tolerance. Without it, the matmul fusion cannot satisfy the `log(1+relu(decoder(LayerNorm(GELU(Dense(hidden)))))` max-pooled per-sequence semantics within atol=1e-2 rtol=1e-2, and the round would have to return `capability-miss`. The `ieee` attribute is a target-language precision accommodation that preserves all normative semantics; it is recorded as a conformance note, not a new design.

Probe scripts were deleted before handoff.

### Harness loader probe

`load_ks_module(Path('sparse_pooler/triton_sparse_pooler_003.py'))` returned a module exposing `ModelNew`, `get_inputs`, `get_init_inputs`. The top-level `@triton.jit` kernel (a `FunctionDef`) is retained by `_filter_module_ast`; no module-level non-literal assignments are present, so nothing is dropped. `get_init_inputs()` returns `(768, 30522, 'max')` and `get_inputs()` returns two Tensors (`[83, 768] fp32 mlu:0` and `[4] int32 mlu:0`).

### Correctness smoke

Instantiated `ModelNew`, called `model_new.load_state_dict(model_ref.state_dict())` (load_state_dict compatibility confirmed), ran `forward` on the documented inputs (`hidden_states=[83,768] fp32 mlu:0`, `seq_lens=[20,25,18,20] int32 mlu:0`), and compared all four `[30522]` fp32 outputs against `base.py`'s `Model`. All four passed `torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True)` with `max_abs_diff` between 2.38e-07 and 2.98e-07. The output structure is correct: list of 4 tensors, each `[30522]` fp32 `mlu:0`.

### Harness end-to-end smoke

`auto_bench.py --v0_file sparse_pooler/base.py --v1_file sparse_pooler/triton_sparse_pooler_003.py --warmup 5 --repeat 5` printed `PASS accuracy; v0=0.891860 ms, v1=0.823640 ms, speedup=1.083x`. This is a smoke measurement only; Verifier will produce the authoritative 50-warmup/100-repeat median used for adoption.

## Conformance notes

- The candidate replaces `_sparse_pooler_max_kernel` with `_sparse_pooler_fused_matmul_max_kernel` and updates `ModelNew.forward` to pass the LayerNorm output, decoder weight, and decoder bias to the fused kernel. This is the exact change the decision's `allowed_changes[0]` and `allowed_changes[1]` authorize.
- `input_precision="ieee"` on `tl.dot` is a target-language precision accommodation that preserves all normative semantics. The local probe established that the default `tf32` precision exceeds the project's 1e-2 tolerance for this shape. Without `ieee`, the matmul fusion cannot meet the project's tolerance and the round would have to return `capability-miss`. This is a conformance note under `candidate-ready`, not a new design: the numerical semantics (`log(1+relu(decoder(LayerNorm(GELU(Dense(hidden)))))` max-pooled per sequence within atol=1e-2 rtol=1e-2 equal_nan=True) are preserved exactly.
- `BLOCK_V=512` and `BLOCK_K=64` are non-semantic tile-size accommodations within the decision's allowed probe space (`BLOCK_V in {256, 512, 1024}` and `BLOCK_K in {64, 128, 256}`). `BLOCK_V=1024` with `BLOCK_K=128` exceeded the NRAM hardware limit (524288 bytes) at `num_warps=1`. The candidate uses the largest vocab tile that fits in NRAM with the smallest K-tile, subject to correctness. This is the first repair in the attempt ledger.
- `num_warps=1` (Constrained, proven, normative) is used exactly as the decision and profile require. `num_warps=2` (known to fail) is not used. No other `num_warps` values were probed because the normative `num_warps=1` compiled and ran correctly.
- The `pooling == "sum"` path retains the library decoder call and the Python fallback. This is off the measured hot path (the harness uses `pooling == "max"`, the default).
- `kernel_count_per_call` decreases from 5 to 4 by construction: the library decoder matmul (`MLUFusedMatMulGepm`) and the existing fused `_sparse_pooler_max_kernel` are replaced by one fused kernel; the dense matmul, LayerNorm, and GELU library ops are unchanged. This matches the decision's `mechanism_observables[total_kernel_count_per_call]` expectation.
- The intermediate logits tensor `[83, 30522]` fp32 (10.16 MB) is no longer materialized in global memory because the decoder matmul is fused into the Triton kernel. This matches the decision's `optimization_intent` and `expected_causal_chain`.
- The decoder weight remains an `nn.Linear` parameter in the canonical `[vocab_size, hidden_size]` layout; no init-time transpose is introduced. `load_state_dict(model.state_dict())` compatibility is preserved because the kernel reads `self.decoder.weight` and `self.decoder.bias` pointers from the live `nn.Linear` parameters at forward time.
- No explicit `torch.mlu.device()` context is introduced; the caller-selected device and current stream are preserved.
- No `__pycache__` directories were created in the project root. All throwaway probe and smoke scripts were deleted before handoff.

## Handoff

- Candidate: `/projs/framework/lipenghui/projects/kernelswift/sparse_pooler/triton_sparse_pooler_003.py` (SHA-256 `3406f7c9a731e1fd7560ab95bf1d903fd4d6f8287c5880d9801e9d41e5ed7860`)
- Result: `candidate-ready`
- Next owner: Verifier (authoritative runtime correctness, wall time, and profiler evidence)

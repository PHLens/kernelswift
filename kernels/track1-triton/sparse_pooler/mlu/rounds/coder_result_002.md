# Coder Result 002

## Metadata

```json
{
  "schema_version": 1,
  "round": "002",
  "result": "candidate-ready",
  "result_reason": "candidate conforms to the immutable decision; only BLOCK_V changed from 1024 to 2048 in the fused kernel dispatch; correctness smoke and harness end-to-end pass within tolerance",
  "source_canonical_path": "triton_sparse_pooler_001.py",
  "source_canonical_sha256": "182f2ebb32b9762f5f19dae00a30e25618bcdb0b8563f7acbc3d9ea9ca348dcd",
  "decision_path": "rounds/decision_002.md",
  "decision_sha256": "0d39de9e280f6ffa2cc3d1d3322d393fa400eb8f405b7e7ee3ceb3ef845b3dd4",
  "selected_profile": "triton_mlu",
  "runtime_fingerprint": {
    "triton_version": "3.2.0",
    "backend_target": "BangDriver (mlu)",
    "backend_version": "torch_mlu 1.32.0+torch2.11.0; MLU driver 6.5.49",
    "device_arch": "MLU590-H8 (capability 5.0)"
  },
  "candidate_path": "triton_sparse_pooler_002.py",
  "candidate_sha256": "62dc853db5423cb5d99ad53433f3fb35919abe901a64d6e3acb3d815ac678248"
}
```

## Decision validation

- Command: `python3 /home/lipenghui/.claude/skills/kernel-opt-loop/scripts/validate_decision.py /projs/framework/lipenghui/projects/kernelswift/sparse_pooler/rounds/decision_002.md --expected-profile triton_mlu`
- Exit code: 0
- Output: `valid: true`
- Language/backend/target profile match the manifest's Identity and Match rules. No environment-blocked condition.

## Primitive and hint conformance

Round 002 is a kernel-only tiling-parameter change. No new Triton primitives are introduced; the kernel body is unchanged from the Round 001 accepted candidate. The only change is `BLOCK_V` from 1024 to 2048 in the `ModelNew.forward` dispatch path. `BLOCK_V` is a `tl.constexpr`, so the kernel is recompiled automatically with the new value.

| Primitive / hint | Decision status | Profile status | Probe outcome | Conformance |
|---|---|---|---|---|
| `tl.load` | Required (unchanged from Round 001) | Supported | Candidate load `seq_lens`, `logits` tiles with mask `v_offs < vocab_size`, `other=-inf` | Mask and bounds validated for the new BLOCK_V=2048; last vocab tile covers offsets 28672..30719, 1850 in-bounds lanes, 198 masked lanes |
| `tl.store` | Required (unchanged from Round 001) | Supported | Candidate stores `acc` to `out[pid_s, v_offs]` with `mask=v_mask` | Shape, dtype, bounds preserved for BLOCK_V=2048 |
| `tl.arange` | Required (unchanged from Round 001) | Supported | `tl.arange(0, BLOCK_V)` with BLOCK_V=2048 | Extent 2048 and mask shape-specific; compiled and ran |
| `tl.program_id` | Required (unchanged from Round 001) | Supported | `pid_s = program_id(0)`, `pid_v = program_id(1)` | Grid mapping preserves the decision's control structure; grid is now (4, 15) |
| `tl.full` | Required (accumulator init, unchanged from Round 001) | Not in Supported table, but proven in Round 001 | `tl.full((BLOCK_V,), -inf, dtype=tl.float32)` with BLOCK_V=2048 | Compiled and ran; ~8 KB register pressure for the accumulator tile |
| `tl.where` | Required (unchanged from Round 001) | Not in Supported table, but proven in Round 001 | `tl.where(x > 0.0, x, 0.0)` for relu | Compiled and ran |
| `tl.log` | Required (unchanged from Round 001) | Not in Supported table, but proven in Round 001 | `tl.log(1.0 + x)` for log1p | Compiled and ran |
| `tl.maximum` | Required (unchanged from Round 001) | Not in Supported table, but proven in Round 001 | `tl.maximum(acc, x)` for per-segment max | Compiled and ran |
| `num_warps=1` | Required (Constrained, normative) | Constrained: `num_warps=1` proven; `num_warps=2` failed | Used `num_warps=1` exactly as the decision and profile require | Proven value, no fallback needed; `num_warps=2` not used |
| `BLOCK_V=2048` | Required (normative target hint) | Not a primitive; a `tl.constexpr` tiling parameter | Compiled and ran on MLU590-H8; correctness passes | Grid drops from (4,30)=120 to (4,15)=60 programs |
| `tl.dot` | Not required this round | Supported | Not used — decoder matmul left as PyTorch library op per the decision | N/A |
| `fast_libentry` | Not required this round (Host Plan not-applicable) | Allowed fallback: ordinary Triton launch | Ordinary `@triton.jit` + grid launch used | N/A — Host Plan is not-applicable |

No Unsupported or unprovable Unknown primitive is required. No silent substitution of a normative construct occurred. The change is purely a `tl.constexpr` value change in the host-side dispatch path; the kernel body is byte-identical to the Round 001 accepted candidate.

## Implementation summary

The candidate is a byte-identical copy of `triton_sparse_pooler_001.py` with exactly one line changed in `ModelNew.forward`:

```diff
-            BLOCK_V = 1024
+            BLOCK_V = 2048
```

This is the only change. The diff between the candidate and the last accepted kernel is exactly one line.

- Kernel: `_sparse_pooler_max_kernel` at module top level (`@triton.jit`), byte-identical to Round 001. `BLOCK_V` is a `tl.constexpr` parameter, so the kernel is recompiled automatically with the new value 2048; no kernel-body change is needed.
- Grid: `(num_seq, num_vocab_tiles)` = `(4, triton.cdiv(30522, 2048))` = `(4, 15)`. This halves the vocab tile count from 30 to 15 and the total grid programs from 120 to 60, matching the decision's Optimization Intent and the Unified Sketch's `parallel pid_v over cdiv(vocab_size, BLOCK_V)`.
- `BLOCK_V = 2048`. 30522 / 2048 = 14.9 tiles; the 15th tile covers offsets 28672..30719 with 1850 in-bounds lanes and 198 masked lanes. The existing `v_mask = v_offs < vocab_size` mask handles the partial tile correctly on both `tl.load` (with `other=-inf`) and `tl.store`.
- On-device offset computation unchanged: `seq_len = tl.load(seq_lens_ptr + pid_s)` and `seq_offset = sum(seq_lens[0:pid_s])` via a bounded `for i in range(pid_s)` loop.
- Accumulator unchanged: `acc = tl.full((BLOCK_V,), -inf, dtype=fp32)`; per-row `tl.where(x>0, x, 0)` (relu) + `tl.log(1+x)` (log1p) + `tl.maximum(acc, x)`.
- `num_warps=1` (Constrained, proven, normative) — unchanged.
- `ModelNew.forward` dispatch logic unchanged except for the `BLOCK_V` value.
- `dense`, `GELU`, `LayerNorm`, `decoder` matmul remain PyTorch library ops unchanged.
- Public constructor `ModelNew(hidden_size=768, vocab_size=30522, pooling="max")` and `forward(hidden_states, seq_lens) -> list[Tensor]` preserved. `get_inputs` and `get_init_inputs` preserved byte-for-byte. The four `nn.Module` attributes (`dense`, `act`, `layer_norm`, `decoder`) are unchanged, so `load_state_dict(model.state_dict())` accepts the reference state dict.
- Output: list of 4 tensors, each `[30522]` fp32 `mlu:0`, allocated per forward (no cross-forward buffer cache; Host Plan is not-applicable).
- No explicit `torch.mlu.device()` context is introduced; the caller-selected device and current stream are preserved.
- `kernel_count_per_call` remains 5 by construction: no kernels added or removed, only the fused kernel's tiling parameter changes.

## Attempt ledger

| # | Command | Exit code | Defect | Before candidate hash | After candidate hash |
|---|---|---:|---|---|---|
| 0 | `python3 /home/lipenghui/.claude/skills/kernel-opt-loop/scripts/validate_decision.py .../rounds/decision_002.md --expected-profile triton_mlu` | 0 | none — `valid: true` | n/a | n/a |
| 1 | `cp triton_sparse_pooler_001.py triton_sparse_pooler_002.py` | 0 | none — byte-identical copy | 182f2ebb... | 182f2ebb... |
| 2 | Edit `BLOCK_V = 1024` -> `BLOCK_V = 2048` in `triton_sparse_pooler_002.py` (line 85) | 0 | none — single-line change, diff confirms exactly one line changed | 182f2ebb... | 62dc853d... |
| 3 | `python3 -c "import ast; ast.parse(open('triton_sparse_pooler_002.py').read())"` | 0 | none — candidate parses; top-level nodes: 7x Import, FunctionDef (kernel), ClassDef (ModelNew), 2x FunctionDef (get_inputs/get_init_inputs), If (__main__) | 62dc853d... | 62dc853d... |
| 4 | `python3 -c "from auto_bench import load_ks_module; load_ks_module(Path('triton_sparse_pooler_002.py'))"` | 0 | none — loader exposes ModelNew/get_inputs/get_init_inputs; init_args=(768, 30522, 'max'); inputs=[Tensor, Tensor] | 62dc853d... | 62dc853d... |
| 5 | Correctness smoke: `ModelNew.load_state_dict(ref.state_dict())` + forward + allclose vs base.py (atol=1e-2, rtol=1e-2, equal_nan=True) | 0 | none — 4/4 outputs match, max_abs_diff=1.79e-07, allclose=True | 62dc853d... | 62dc853d... |
| 6 | `python3 auto_bench.py --v0_file sparse_pooler/base.py --v1_file sparse_pooler/triton_sparse_pooler_002.py --warmup 5 --repeat 5` | 0 | none — `PASS accuracy; v0=0.905974 ms, v1=0.615262 ms, speedup=1.473x` (smoke only; Verifier produces the authoritative 50/100 measurement) | 62dc853d... | 62dc853d... |

No repair attempts were needed. The candidate parsed, loaded, and passed correctness on the first write. The two pre-handoff gates (`ast.parse` and the actual harness loader) both succeeded. The optional correctness smoke and the optional harness end-to-end smoke also succeeded.

No fallback probes were needed. The decision's normative value `BLOCK_V=2048` compiled and ran correctly on the first attempt, so the optional `BLOCK_V=4096` fallback and the optional `num_warps` probes were not exercised. The proven `num_warps=1` is used as required.

## Probe evidence

### Harness loader probe

`load_ks_module(Path('triton_sparse_pooler_002.py'))` returned a module exposing `ModelNew`, `get_inputs`, `get_init_inputs`. The top-level `@triton.jit` kernel (a `FunctionDef`) is retained by `_filter_module_ast`; no module-level non-literal assignments are present, so nothing is dropped. `get_init_inputs()` returns `(768, 30522, 'max')` and `get_inputs()` returns two Tensors.

### Correctness smoke

Instantiated `ModelNew`, called `model_new.load_state_dict(model_ref.state_dict())` (load_state_dict compatibility confirmed), ran `forward` on the documented inputs (`hidden_states=[83,768] fp32 mlu:0`, `seq_lens=[20,25,18,20] int32 mlu:0`), and compared all four `[30522]` fp32 outputs against `base.py`'s `Model`. All four passed `torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True)` with `max_abs_diff=1.79e-07`. The output structure is correct: list of 4 tensors, each `[30522]` fp32 `mlu:0`.

### Harness end-to-end

`auto_bench.py --v0_file sparse_pooler/base.py --v1_file sparse_pooler/triton_sparse_pooler_002.py --warmup 5 --repeat 5` printed `PASS accuracy; v0=0.905974 ms, v1=0.615262 ms, speedup=1.473x`. This is a smoke measurement only; Verifier will produce the authoritative 50-warmup/100-repeat median used for adoption.

## Conformance notes

- The candidate is a byte-identical copy of the Round 001 accepted kernel with exactly one line changed: `BLOCK_V = 1024` -> `BLOCK_V = 2048` in the `ModelNew.forward` dispatch path. The kernel body (`_sparse_pooler_max_kernel`) is unchanged. `BLOCK_V` is a `tl.constexpr`, so the kernel is recompiled automatically with the new value; no kernel-body edit is needed. This is the exact change the decision's `allowed_changes[0]` authorizes.
- The host-side grid computation `num_vocab_tiles = triton.cdiv(vocab_size, BLOCK_V)` is recomputed automatically with the new `BLOCK_V` (30522/2048 = 15 tiles). This is the exact change the decision's `allowed_changes[1]` authorizes. It is a direct consequence of the constexpr change and introduces no allocation reuse, output caching, lifecycle, or concurrency changes, consistent with the decision's Host Plan `not-applicable`.
- `num_warps=1` (Constrained, proven, normative) is used exactly as the decision and profile require. `num_warps=2` (known to fail) is not used. No other `num_warps` values were probed because the normative `BLOCK_V=2048` compiled and ran correctly on the first attempt; the optional fallback probes were not needed.
- The optional `BLOCK_V=4096` fallback probe was not exercised because the normative `BLOCK_V=2048` compiled and ran correctly on the first attempt.
- The `pooling == "sum"` path retains the Python fallback (unchanged from Round 001). This is off the measured hot path (the harness uses `pooling == "max"`, the default).
- `kernel_count_per_call` remains 5 by construction: no kernels added or removed, only the fused kernel's tiling parameter changes. The four non-fused kernels (decoder matmul, dense matmul, LayerNorm, GELU) are unchanged.
- Register pressure at `BLOCK_V=2048`: the `acc` tile and the `vocab_tile` each hold 2048 fp32 values, so approximately 16 KB of register space for the two tiles combined (vs 8 KB at BLOCK_V=1024). The kernel compiled and ran correctly on MLU590-H8, so no register spill blocked the change.
- No `__pycache__` directories were created in the project root.

## Handoff

- Candidate: `/projs/framework/lipenghui/projects/kernelswift/sparse_pooler/triton_sparse_pooler_002.py` (SHA-256 `62dc853db5423cb5d99ad53433f3fb35919abe901a64d6e3acb3d815ac678248`)
- Result: `candidate-ready`
- Next owner: Verifier (authoritative runtime correctness, wall time, and profiler evidence)

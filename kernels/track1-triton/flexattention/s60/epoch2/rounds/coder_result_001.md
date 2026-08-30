# Coder Result 001

Result: candidate-ready

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md` @`8a2bb5a7a6bcd2ccb8ecb704c30c5edbb540fb5c52fc4cae34f2afeef57c5d86` (hash re-verified from file; family "triton-attention-dispatch-collapse"; decision "proceed"; expected_wall_improvement_pct 0.0 declared honestly; deliverable-grade round per project.md DELIVERABLE RULE)
- Sketch: `rounds/sketch_001.json` @`aad322a8b806d9f97bc9c5056c8ae1ea62c5bd8ecc8bb502fb6fc72399a61247` (hash re-verified; matches decision `sketch_sha256` `aad322a8...`)
- Candidate: `triton_flexattention_e2_001.py` @`6a62042904bd774006154ba75d8bbcc8212449438d2cd8b4aaa02a5415eed0e9` (project root: `kernels/track1-triton/flexattention/s60/epoch2/`)
- Base (immutable reference): `../../base.py` @`dd1359ad88a5d5aae48a115023e5ae9016e4d5b5e6a12781273703e8d5c6a6d0` (re-verified unchanged; causal flexattention via `F.scaled_dot_product_attention(is_causal=True)`)
- Harness: `/root/CodeBuddy/20260828202827/kernelswift/auto_bench.py` @`71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (untouched; AST loader)
- Runtime fingerprint: `triton 3.6.0 / triton_gcu 3.6.0 / torch 2.10.0 / GCU device` — matches `project.md#runtime-fingerprint` (device `gcu`)

## Implementation Summary (decision-001 exact)

- ONE stateless `@triton.jit` kernel `_flex_attn_fwd`; TWO launch sites (`forward` + `run_out`) both direct-launching `_flex_attn_fwd[(self.num_heads,)]` with `num_warps=1`, `num_stages` unset (count 0). Grid = H = 8 programs (one program per head); each program owns one head `h` and computes full causal attention over the S=83 tokens padded to TP=128.
- `TP=128` (power-of-2, sketch hint `pad_to_power_of_2_128` required), `D=64` (head_size), `S/H/D/TP` all passed as constexpr (frozen literals at call site: S=num_tokens, H=num_heads, D=head_size, TP=128).
- **Direct strided addressing, ZERO layout-copy calls**: inputs are fp16 `[T,H,D]` token-major tensors addressed directly at offset `t*(H*D) + h*D + d` (head stride = D) with strides folded from constexpr (H,D); Q/K/V loaded directly (masked to token<S, other=0.0). The word `contiguous` does not appear anywhere in the source (grep count 0).
- **QK^T fp16 direct dot (NO widening)**: `s = tl.dot(qh, tl.trans(kh)) * scale` — q/k stay fp16 DIRECTLY into `tl.dot` (fp16 x fp16 -> fp32 accumulator, the tensor-core MMA path). There is NO `.to(tl.float32)` on q or k. scale=0.125 (exact power of two).
- **CAUSAL mask merged with out-of-range mask**: `causal = offs_m[:,None] >= offs_n[None,:]` ANDed with `mask_n[None,:]` (`offs_n < S`) into `-inf` via `tl.where` — upper-triangle keys AND out-of-range (S=83..127) keys both masked to -inf (exp(-inf)=0 exact).
- **Softmax without keepdim**: `tl.max(s, axis=1)` + `tl.exp(s - m[:,None])` + `tl.sum(p, axis=1)`, then `attn = p / l[:,None]` — NO keepdim (keepdim count 0), broadcast via `[:, None]`.
- **PV fp32**: `out = tl.dot(attn, vh)` where `attn` is fp32 (softmax output) and `vh` is widened fp16->fp32 on load (`.to(tl.float32)` at line 23 — the ONLY widening cast in the kernel) → fp32 x fp32 -> fp32 accumulator.
- Output store: `tl.where(mask_m[:,None], out, 0.0)` then `.to(tl.float16)` direct store into the final `[T,H,D]` token-major layout; `forward` returns `o.reshape(num_tokens, H*D)` → `[83,512]`.
- `forward` = ONE python-visible `torch.empty` + ONE kernel launch + ONE reshape; `run_out(query,key,value,out)` 4-arg surface (ONE launch into the caller buffer, zero allocations, returns None, bitwise-equal to forward).
- STATELESS: instance attrs exactly 4 constructor-config attrs (`num_heads`, `head_size`, `scale`, `num_kv_heads`); no caches, no workspace, no cross-call state; Triton JIT compile cache is framework-owned, one-time, absorbed by harness warmup 50.

## Sketch Primitive and Hint Conformance

- Required sketch hints bound:
  - `pad_to_power_of_2_128` (modality required): TP=128 (power-of-2) — honored; `tl.arange(0, TP)` extents 128 and `tl.arange(0, D)` extent 64, both power-of-2.
  - `tl_dot_same_dtype` (modality required): QK^T = fp16 x fp16 (q/k both fp16); PV = fp32 x fp32 (attn fp32, v widened to fp32 on load). Both dots same-dtype.
  - `tl_max_sum_no_keepdim` (modality required): `tl.max`/`tl.sum` both `axis=1` WITHOUT keepdim, broadcast via `[:, None]`.
  - `causal_mask` (modality required): `offs_m[:,None] >= offs_n[None,:]` merged with out-of-range mask into -inf.
  - `num_warps_1` (modality preferred): `num_warps=1` at both launch sites.
  - `qk_dot_fp16` / `pv_dot_fp32` (modality required): QK^T fp16 direct, PV fp32.
- Primitives used, mapped to frozen profile `profile_snapshot/triton_gcu.yaml`:
  - `tl.dot` → `matrix.dot.fp16-fp16-fp32` (status **constrained**): power-of-2 tiles (128x64 @ 64x128 → 128x128; 128x128 @ 128x64 → 128x64). S60 probe-backed constraint is **power-of-2** (96=16x6 FAILS; only 16/32/64/128 pass) — TP=128 and D=64 are both power-of-2, legal.
  - `tl.arange` → `index.range.one-dimensional` (status **supported**): extents 128, 128, 64 — all power-of-2.
  - `tl.load` / `tl.store` → `memory.load.contiguous` / `memory.store.contiguous` (status **supported**): masked loads (mask=token<S, other=0.0) and masked store.
  - `tl.program_id` → `parallel.program-id.axis0` (status **supported**): axis-0 only.
  - `num_warps` → `resource.num-warps` (status **constrained**, legal set {1,2,4,8}): value 1, inside legal set.
  - `tl.max` / `tl.sum` → reduction family: axis-1 reduction WITHOUT `keepdim`.
  - `tl.trans`, `tl.exp`, `tl.where` → elementwise/transpose family: used as fp16/fp32 elementwise ops.
- No atomics, no fp64, no `tl.make_block_ptr`, no `tl.async_copy`. No `tl.argmax` (tie-free by construction).
- DANGER tokens (compile/capture/graph/contiguous/torch.compile/TORCHINDUCTOR/reduce-overhead/copy_): all-zero (see binding audit).

## Binding Statement

- **Dot-shape audit**: 2 `tl.dot` call sites — (1) `s = tl.dot(qh, tl.trans(kh))` → (128,64)@(64,128)→(128,128) **fp16 x fp16 -> fp32** (q/k fp16, NO widening cast on q/k); (2) `out = tl.dot(attn, vh)` → (128,128)@(128,64)→(128,64) **fp32 x fp32 -> fp32** (attn fp32, v widened fp16->fp32 on load). Both at power-of-2 tiles (TP=128, D=64), both same-dtype. 0 non-power-of-2 dot shapes.
- **num_warps**: exactly two launch sites (`forward` + `run_out`), both value `1`; `num_stages` absent (count 0). Kernel count: 1 (`@triton.jit` count 1).
- **Addressing audit**: direct strided addressing of `[T,H,D]` inputs (token stride H*D, head stride D — all constexpr-folded); `tl.trans(kh)` used only for the QK^T operand (trans of the already-loaded fp16 tile, not a host copy); direct fp16 final-layout stores. `contiguous` count 0 over the whole source.
- **Stateless audit**: 4 instance-attr writes, all in `__init__` (`num_heads` L41, `head_size` L42, `scale` L43, `num_kv_heads` L44) — constructor parity with baseline_adapter; no call-time instance writes; all `self.` in `forward`/`run_out` are READS. Module level = imports + `@triton.jit` kernel + ClassDef + `get_inputs`/`get_init_inputs` helper FunctionDefs.
- **AST-loader composition**: 3 imports + `@triton.jit` FunctionDef + ClassDef + 2 helper FunctionDefs; no unsafe module-level statements. `ast.parse` gate PASS (`AST_PARSE_OK`).

## Deviations

- None. The candidate is a faithful rendering of `sketch_001.json` and `decision_001.md` with no deviation from the decision's `allowed_changes`, `invariants`, or the S60 capability envelope.

## Evidence for Verifier

- Candidate: `triton_flexattention_e2_001.py` @`6a62042904bd774006154ba75d8bbcc8212449438d2cd8b4aaa02a5415eed0e9`.
- Canonical measurement route (Verifier-owned): unchanged harness `auto_bench.py`, device gcu, seed 42, warmup 50 / repeat 100 interleaved pairs.
- Coder correctness smoke (non-authoritative, PASS/FAIL only, timing NOT relied upon): `PASS accuracy; v0=0.332924 ms, v1=0.333505 ms, speedup=0.998x` → correctness **PASS** under the unchanged comparator.
- Coder ran NO timing/benchmark/profiler measurements for the verdict; no verdict is claimed here. Classification is `candidate-ready`.

### Artifact hash ledger

```text
6a62042904bd774006154ba75d8bbcc8212449438d2cd8b4aaa02a5415eed0e9  triton_flexattention_e2_001.py
8a2bb5a7a6bcd2ccb8ecb704c30c5edbb540fb5c52fc4cae34f2afeef57c5d86  rounds/decision_001.md
aad322a8b806d9f97bc9c5056c8ae1ea62c5bd8ecc8bb502fb6fc72399a61247  rounds/sketch_001.json
dd1359ad88a5d5aae48a115023e5ae9016e4d5b5e6a12781273703e8d5c6a6d0  ../../base.py
71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29  /root/CodeBuddy/20260828202827/kernelswift/auto_bench.py
```

### Binding audit table (source counts)

| Token / construct | Count | Verdict |
|---|---|---|
| `compile` / `capture` / `graph` / `contiguous` | 0 / 0 / 0 / 0 | DANGER-free |
| `torch.compile` / `TORCHINDUCTOR` / `reduce-overhead` / `copy_` | 0 / 0 / 0 / 0 | DANGER-free |
| `tl.dot` | 2 | QK^T fp16 x fp16 -> fp32; PV fp32 x fp32 -> fp32; power-of-2 tiles |
| `num_warps` | 2 (both value 1) | legal (two launch sites, both nw=1) |
| `@triton.jit` | 1 | single kernel |
| `tl.arange` | 3 (extents 128, 128, 64) | power-of-2 |
| `tl.max` / `tl.sum` | 1 / 1 | no keepdim |
| `keepdim` | 0 | no-keepdim satisfied |
| `.to(tl.float32)` | 1 | v load only (PV path); q/k NOT widened |
| `.to(tl.float16)` | 1 | store narrow only |
| `self.<attr>` writes | 4 | all in `__init__` (stateless) |

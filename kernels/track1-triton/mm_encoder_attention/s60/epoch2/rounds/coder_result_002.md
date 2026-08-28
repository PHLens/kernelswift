# Coder Result 002

Result: candidate-ready

## Identity

- Round: `002`
- Decision: `rounds/decision_002.md` @`<decision_hash>` (family "triton-attention-dot-dtype"; expected_wall_improvement_pct 0.0 declared honestly; deliverable-grade round per project.md DELIVERABLE RULE)
- Sketch: `rounds/sketch_002.json` @`c3c585d1f95337f25ac1c9ff5dc3c3591637b1e9a7c906174fb60d0da97695dd` (matches decision `sketch_sha256`)
- Candidate: `triton_mm_encoder_attention_e2_002.py` @`7b411daf3903c88ebcaa9426a628f6fe76638fd7be635c0563ee4f63fc1be818` (project root: `kernels/track1-triton/mm_encoder_attention/s60/epoch2/`)
- Round-1 candidate (implementation start, only dot dtype + num_warps changed): `triton_mm_encoder_attention_e2_001.py` @`f2f8b9b6c6f6a16cfbf162cf3f9b115461fc7a5716601eb8e3723961a8536ead`
- Canonical start (last_accepted_kernel): `baseline_adapter.py` @`1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e` (2331 bytes) — semantics derived from it.
- Base (immutable reference): `../../base.py` @`86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` (2284 bytes; unchanged; equals project.md declaration)
- Harness: `/root/CodeBuddy/20260828202827/kernelswift/auto_bench.py` @`71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (29428 bytes; untouched; AST loader)
- Runtime fingerprint: `triton 3.6.0 / triton_gcu 3.6.0+1.0.20260722 / torch 2.10.0+cpu / torch_gcu 2.10.0+3.8.0.2 / Enflame GCU major=3 minor=0 multi_processor_count=2 total_memory=43878764544`, interpreter `/usr/bin/python3`, device `gcu` — matches `project.md#runtime-fingerprint`

## Implementation Summary (decision-002 exact)

- Kernel structure IDENTICAL to round-001 (one stateless `@triton.jit` kernel `_mm_encoder_attn_fwd`; single shared launch site `ModelNew._launch`; grid = B*H = 2*8 = **16 programs**; single-tile TP=128 = S=83 padded to power-of-2; direct strided `[B,S,H*D]` addressing; -inf masking on tile-padding columns; bidirectional no-causal softmax; direct `[2,83,512]` fp16 stores; two-op forward host path; 4-arg run_out). The ONLY kernel differences vs round-001 are:
  1. **QK^T dot operand dtype → fp16**: q/k loaded and kept **fp16** (the `.to(tl.float32)` widening cast is REMOVED); `s = tl.dot(q, tl.trans(k)) * scale` feeds fp16 x fp16 → fp32 accumulator directly (the tensor-core MMA path).
  2. **PV dot operand dtype → fp32 primary**: v loaded and widened `.to(tl.float32)`; `out = tl.dot(attn, v)` is fp32 x fp32 → fp32 accumulator (attn is the fp32 softmax result, NOT cast back to fp16). The fp16 PV fallback is declared in the decision but NOT shipped (primary path compiled correctly).
  3. **num_warps → 1** (from round-001's 2).
- Softmax stays fp32: `tl.where(mask_t[None,:], s, -inf)` → `tl.max(s, axis=1)` (no keepdim) → `tl.exp(s - m[:,None])` → `tl.sum(p, axis=1)` → `attn = p / l[:,None]`.
- `forward` = TWO python-visible ops (`torch.empty` + ONE kernel launch); `run_out(query,key,value,out)` 4-arg surface (ONE launch, zero allocations, returns None).
- STATELESS: instance attrs exactly 4 constructor-config attrs (`num_heads`, `head_size`, `num_kv_heads`, `scale`); no caches, no workspace, no cross-call state.

## Sketch Primitive and Hint Conformance

- Required sketch hints bound:
  - `qk_dot_fp16_no_widen` (modality required): q/k stay fp16, no `.to(tl.float32)` before QK^T dot — honored (source has exactly 1 `.to(tl.float32)` on v, 0 on q/k).
  - `tl_dot_same_dtype` (modality required): QK^T = fp16 x fp16; PV = fp32 x fp32 — both same-dtype.
  - `pad_to_power_of_2_128` (modality required): TP=128, D=64 (both power-of-2).
  - `pv_dot_fp32_primary` (modality preferred): v widened to fp32, attn fp32 — honored.
  - `num_warps_1` (modality preferred): `num_warps=1` — honored.
- Primitives mapped to frozen profile `profile_snapshot/triton_gcu.yaml`:
  - `tl.dot` → `matrix.dot.fp16-fp16-fp32.mult-of-16-tiles` (status **constrained**): QK^T (128x64)@(64x128) and PV (128x128)@(128x64), all power-of-2 tiles, fp32 accumulator. The S60 probe-backed correction (from r001 verdict): `tl.dot` AND `tl.arange` require **power-of-2** (NOT mult-of-16 — 96=16x6 FAILS; only 16/32/64/128 pass). TP=128 and D=64 are power-of-2, so legal under both readings.
  - `tl.arange` → `index.range.one-dimensional` (supported): extents 128 and 64, power-of-2.
  - `tl.load` / `tl.store` → memory load/store (supported): masked (token<S, other=0.0).
  - `tl.program_id` → `parallel.program-id.axis0` (supported).
  - `num_warps` → `resource.num-warps` (constrained, legal set {1,2,4,8}): value 1.
  - `tl.max` / `tl.sum` → reduction family: axis-1, WITHOUT keepdim (keepdim count 0), broadcast via `[:, None]`.
  - `tl.trans`, `tl.exp`, `tl.where` → elementwise/transpose family (fp32).

## Binding Statement

- **Dot-shape / dot-dtype audit**: 2 `tl.dot` call sites. (1) QK^T `s = tl.dot(q, tl.trans(k))` → (128,64)@(64,128)→(128,128), operands **fp16 x fp16** (zero widening cast on q/k), fp32 accumulator. (2) PV `out = tl.dot(attn, v)` → (128,128)@(128,64)→(128,64), operands **fp32 x fp32** (attn fp32 softmax, v widened on load). Both power-of-2 tiles (TP=128, D=64); 0 non-power-of-2 dot shapes.
- **num_warps**: exactly one launch site (shared `_launch`), value `1`; `num_stages` absent (count 0). Kernel count: 1 (`@triton.jit` count 1).
- **Addressing audit**: direct strided addressing of `[B,S,H*D]` inputs (batch stride S*H*D, token stride H*D, head stride D — all constexpr-folded); `tl.trans(k)` only for the QK^T operand; direct fp16 final-layout stores; `contiguous` count 0 over the whole source.
- **Stateless audit**: 4 instance-attr writes, all in `__init__` (`num_heads`, `head_size`, `num_kv_heads`, `scale`); no call-time instance writes; module level = imports + kernel + ClassDef + `get_inputs`/`get_init_inputs` helpers.
- **AST-loader composition**: 4 imports + `@triton.jit` FunctionDef + ClassDef + 2 helper FunctionDefs — all retained node types; `ast.parse` gate PASS (`AST_PARSE_OK`).
- **Shipped configuration**: fixed module-level literals — QK^T fp16 x fp16, PV fp32 x fp32 (primary), num_warps=1. No runtime switching; the fp16 PV fallback variant is NOT shipped (primary path compiled correctly under the smoke).

## Deviations

None. The shipped configuration is exactly decision-002's primary path (fp16 QK^T + fp32 PV + num_warps=1). The fp16 PV fallback was declared-but-not-shipped per the decision's selection rule (primary path compiled and correctness-smoked without miscompile, so the fallback is not exercised).

## Evidence for Verifier

- Candidate: `triton_mm_encoder_attention_e2_002.py` @`7b411daf3903c88ebcaa9426a628f6fe76638fd7be635c0563ee4f63fc1be818`.
- Canonical measurement route (Verifier-owned): unchanged harness `auto_bench.py`, device gcu, seed 42, warmup 50 / repeat 100 / 3 interleaved pairs.
- Coder ran NO authoritative timing/benchmark/profiler. The smoke below is recorded for PASS/FAIL only; its timing figures are NOT adopted as claims.

### Gates run (self-check)

| Gate | Command | Result |
|---|---|---|
| AST | `python3 -c "import ast; ast.parse(...); print('AST_PARSE_OK')"` | `AST_PARSE_OK` |
| Correctness smoke | `auto_bench.py --v0_file base.py --v1_file candidate --warmup 50 --repeat 100` | `PASS accuracy` (exit 0) |

Smoke printed `PASS accuracy; v0=0.324812 ms, v1=0.334159 ms, speedup=0.972x` — recorded as a harness-printed observation only (NOT a Coder measurement claim; timing belongs to Verifier).

### Artifact hash ledger

```text
7b411daf3903c88ebcaa9426a628f6fe76638fd7be635c0563ee4f63fc1be818  triton_mm_encoder_attention_e2_002.py
04f6dc0b6a92429ba7538d2dfa3d6c4e10471a05d80188a716d5770e2f031e2f  rounds/decision_002.md
c3c585d1f95337f25ac1c9ff5dc3c3591637b1e9a7c906174fb60d0da97695dd  rounds/sketch_002.json
1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e  baseline_adapter.py
86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2  ../../base.py
71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29  /root/CodeBuddy/20260828202827/kernelswift/auto_bench.py
```

### Binding audit table (source counts)

| Token / construct | Count | Verdict |
|---|---|---|
| `compile` / `capture` / `graph` / `contiguous` | 0 / 0 / 0 / 0 | DANGER-free |
| `torch.compile` / `TORCHINDUCTOR` / `reduce-overhead` / `copy_` | 0 / 0 / 0 / 0 | DANGER-free |
| `tl.dot` | 2 | QK^T fp16x fp16, PV fp32x fp32 |
| `num_warps` | 1 (value 1) | legal |
| `@triton.jit` | 1 | single kernel |
| `tl.arange` | 2 (extents 128, 64) | power-of-2 |
| `tl.max` / `tl.sum` | 1 / 1 | no keepdim |
| `tl.float32` | 1 | on v only (PV widen) |
| `tl.float16` | 1 | on output store |
| `keepdim` | 0 | no-keepdim satisfied |
| `self.<attr>` writes | 4 | all in `__init__` |

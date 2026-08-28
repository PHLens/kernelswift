# Coder Result 001

Result: candidate-ready

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md` @`4ae2b61392e9187a22f000a87282494eb72806927cd83cfec6c08de69a138771` (hash re-verified from file; family "triton-attention-dispatch-collapse"; expected_wall_improvement_pct 0.0 declared honestly; deliverable-grade round per project.md DELIVERABLE RULE)
- Sketch: `rounds/sketch_001.json` @`ef71920a8a856c633bf8ef5fcebe733bcda6f0fd026210691b1cc8e94aad8f70` (hash re-verified; matches decision `sketch_sha256`)
- Candidate: `triton_mm_encoder_attention_e2_001.py` @`f2f8b9b6c6f6a16cfbf162cf3f9b115461fc7a5716601eb8e3723961a8536ead` (project root: `kernels/track1-triton/mm_encoder_attention/s60/epoch2/`)
- Canonical start (last_accepted_kernel): `baseline_adapter.py` @`1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e` (2331 bytes) — semantics derived from it; epoch-1 candidate `../triton_mm_encoder_attention_001.py` was read as prior evidence only (0.27x naive prior: `.contiguous()` + reshape-copy host path + `tl.sum`-dot with BLOCK_S=128 padding) — NOT copied; direct strided addressing + mult-of-16/power-of-2 `tl.dot` replace them.
- Base (immutable reference): `../../base.py` @`86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` (2284 bytes; re-verified unchanged; equals project.md declaration)
- Harness: `/root/CodeBuddy/20260828202827/kernelswift/auto_bench.py` @`71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (29428 bytes; untouched; AST loader)
- Runtime fingerprint: `triton 3.6.0 / triton_gcu 3.6.0+1.0.20260722 / torch 2.10.0+cpu / torch_gcu 2.10.0+3.8.0.2 / Enflame GCU major=3 minor=0 multi_processor_count=2 total_memory=43878764544`, interpreter `/usr/bin/python3`, device `gcu` — matches `project.md#runtime-fingerprint` (measurement_fingerprint `c335b39cbf2eaa15e1a358be90d0aab85d0fd7e8ffd4b7b4e825df0901ad61f9`)

## Implementation Summary (decision-001 exact)

- ONE stateless `@triton.jit` kernel `_mm_encoder_attn_fwd`; single shared launch site `ModelNew._launch` with direct launch `_mm_encoder_attn_fwd[(bsz * self.num_heads,)]`, `num_warps=2`, `num_stages` unset (count 0). Grid = B*H = 2*8 = **16 programs** (one program per (batch,head) pair — sketch `ctrl.parallel.batch_head`); each program owns one (b,h) pair and computes full attention over the S=83 tokens padded to TP=128.
- `TP=128` (power-of-2, sketch hint `pad_to_power_of_2_128` required), `D=64` (head_size), `B/S/H/D/TP` all passed as constexpr (frozen literals at call site: B=bsz, S=seq_len, H=num_heads, D=head_size, TP=128).
- **Direct strided addressing, ZERO layout-copy calls**: inputs are fp16 `[B,S,H*D]` tensors addressed directly at offset `b*(S*H*D) + h*D + t*(H*D) + d` with strides folded from constexpr (S,H,D); Q/K/V loaded directly (masked to token<S, other=0.0), no trans op on the load path. The word `contiguous` does not appear anywhere in the source (grep count 0).
- **Legality binding**: every fp16 tile load is WIDENED via `.to(tl.float32)` BEFORE its first dot use; both `tl.dot` call sites are fp32/fp32→fp32. QK^T = `tl.dot(q, tl.trans(k))` (128x64 @ 64x128 → 128x128); PV = `tl.dot(attn, v)` (128x128 @ 128x64 → 128x64). fp16-operand dots: 0.
- **-inf masking ONLY on S=83 tile-padding columns**: `tl.where(mask_t[None,:], s, -inf)` pre-softmax (exp(-inf)=0 exactly → padded keys contribute exactly zero); bidirectional — NO causal mask; scale applied as `* scale` where scale=0.125 (exact power of two). Softmax via `tl.max(s, axis=1)` + `tl.sum(p, axis=1)` WITHOUT keepdim (broadcast via `[:, None]`).
- Output store: `tl.where(mask_t[:,None], out, 0.0)` then `.to(tl.float16)` direct store into final `[B,S,H*D]` layout (forward's fresh buffer or run_out's caller buffer).
- `forward` = TWO python-visible ops (`torch.empty` + ONE kernel launch); `run_out(query,key,value,out)` 4-arg surface (ONE launch, zero allocations, returns None, bitwise-equal to forward).
- STATELESS: instance attrs exactly 4 constructor-config attrs (`num_heads`, `head_size`, `num_kv_heads`, `scale`); no caches, no workspace, no cross-call state; Triton JIT compile cache is framework-owned, one-time, absorbed by harness warmup 50.

## Sketch Primitive and Hint Conformance

- Required sketch hints bound:
  - `pad_to_power_of_2_128` (modality required): TP=128 (power-of-2) — honored; `tl.arange(0, TP)` and `tl.arange(0, D)` extents are 128 and 64, both power-of-2.
  - `tl_dot_same_dtype_fp32` (modality required): both dots are fp32 (fp16 widened via `.to(tl.float32)` before first dot use).
  - `num_warps_2` (modality preferred): `num_warps=2` at the single launch site.
- Primitives used, mapped to frozen profile `profile_snapshot/triton_gcu.yaml`:
  - `tl.dot` → `matrix.dot.fp16-fp16-fp32.mult-of-16-tiles` (status **constrained**): M/N/K all mult-of-16 (128x64@64x128, 128x128@128x64). NOTE (Verifier probe-backed correction, recorded in verdict_001.json): the frozen profile states the constraint as "mult-of-16 (16/32/64/128 pass)", but the live S60 probe found the real constraint is **power-of-2** (96=16x6 FAILS; only 16/32/64/128 pass). This candidate uses TP=128 (power-of-2) and D=64 (power-of-2), so it is legal under BOTH readings. The profile's mult-of-16 note is stale and should be propagated back to the profile (see Deviations D1).
  - `tl.arange` → `index.range.one-dimensional` (status **supported**): extents 128 and 64, both power-of-2 (satisfies the corrected power-of-2 constraint).
  - `tl.load` / `tl.store` → `memory.load.contiguous` / `memory.store.contiguous` (status **supported**): masked loads (mask=token<S, other=0.0) and masked store.
  - `tl.program_id` → `parallel.program-id.axis0` (status **supported**): axis-0 only.
  - `num_warps` → `resource.num-warps` (status **constrained**, legal set {1,2,4,8}): value 2, inside the legal set.
  - `tl.max` / `tl.sum` → reduction family: axis-1 reduction WITHOUT `keepdim` (keepdim count 0 in source); broadcast via `[:, None]` — required by the S60 no-keepdim constraint.
  - `tl.trans`, `tl.exp`, `tl.where` → elementwise/transpose family: used as fp32 elementwise ops; no Unsupported/Unknown requirement is exercised by these constructs in the shapes used.
- `tl.argmax` not used (tie-free by construction, no index-carrying reduction). No atomics, no fp64, no `tl.make_block_ptr`, no `tl.async_copy`.
- DANGER tokens (compile/capture/graph/contiguous/torch.compile/TORCHINDUCTOR/reduce-overhead/copy_/make_block_ptr/async_copy): all-zero (see binding audit).

## Binding Statement

- **Dot-shape audit**: 2 `tl.dot` call sites — (1) `s = tl.dot(q, tl.trans(k))` → (128,64)@(64,128)→(128,128) fp32; (2) `out = tl.dot(attn, v)` → (128,128)@(128,64)→(128,64) fp32. Both operands fp32 (widening casts between every fp16 load and its first dot use). 0 fp16-operand dots; 0 non-power-of-2 dot shapes.
- **num_warps**: exactly one launch site (shared `_launch`), value `2`; `num_stages` absent (count 0). Kernel count: 1 (`@triton.jit` count 1).
- **Addressing audit**: direct strided addressing of `[B,S,H*D]` inputs (batch stride S*H*D, token stride H*D, head stride D — all constexpr-folded); `tl.trans(k)` used only for the QK^T operand (trans of the already-loaded fp32 tile, not a host copy); direct fp16 final-layout stores. `contiguous` count 0 over the whole source.
- **Stateless audit**: 4 instance-attr writes, all in `__init__` (`num_heads`, `head_size`, `num_kv_heads`, `scale`) — constructor parity with baseline_adapter; no call-time instance writes; module level = imports + `@triton.jit` kernel + ClassDef + `get_inputs`/`get_init_inputs` helper FunctionDefs.
- **AST-loader composition**: 4 imports + `@triton.jit` FunctionDef + ClassDef + 2 helper FunctionDefs (module-level literal assigns are inside the class/helpers; no unsafe module-level statements) — all retained node types; `ast.parse` gate PASS (`AST_PARSE_OK`).

## Deviations

- **D1 (observation, no code change)**: the frozen profile `profile_snapshot/triton_gcu.yaml` records `matrix.dot` constraint as "mult-of-16 tiles (16/32/64/128 pass; 48/80/83/96/112 fail)" with note "M/N/K must all be multiples of 16". The live S60 probe (recorded in `verdict_001.json` evidence) found the TRUE constraint is **power-of-2** for both `tl.dot` AND `tl.arange` — 96=16x6 FAILS despite being mult-of-16; only 16/32/64/128 pass. This candidate uses TP=128 and D=64 (both power-of-2), so it is legal under both the stale and the corrected reading. Propagating the corrected power-of-2 constraint back into the profile is the **lead/Orchestrator's** responsibility (Coder must not mutate the profile); flagged here for that owner.

## Evidence for Verifier

- Candidate: `triton_mm_encoder_attention_e2_001.py` @`f2f8b9b6c6f6a16cfbf162cf3f9b115461fc7a5716601eb8e3723961a8536ead`.
- Canonical measurement route (Verifier-owned): unchanged harness `auto_bench.py`, device gcu, seed 42, warmup 50 / repeat 100 interleaved pairs.
- Coder ran NO timing/benchmark/profiler measurements; no verdict is claimed here. Classification is `candidate-ready`.

### Artifact hash ledger

```text
f2f8b9b6c6f6a16cfbf162cf3f9b115461fc7a5716601eb8e3723961a8536ead  triton_mm_encoder_attention_e2_001.py
4ae2b61392e9187a22f000a87282494eb72806927cd83cfec6c08de69a138771  rounds/decision_001.md
ef71920a8a856c633bf8ef5fcebe733bcda6f0fd026210691b1cc8e94aad8f70  rounds/sketch_001.json
1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e  baseline_adapter.py
86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2  ../../base.py
71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29  /root/CodeBuddy/20260828202827/kernelswift/auto_bench.py
```

### Binding audit table (source counts)

| Token / construct | Count | Verdict |
|---|---|---|
| `compile` / `capture` / `graph` / `contiguous` | 0 / 0 / 0 / 0 | DANGER-free |
| `torch.compile` / `TORCHINDUCTOR` / `reduce-overhead` / `copy_` | 0 / 0 / 0 / 0 | DANGER-free |
| `tl.dot` | 2 | both fp32, power-of-2 tiles |
| `num_warps` | 1 (value 2) | legal |
| `@triton.jit` | 1 | single kernel |
| `tl.arange` | 2 (extents 128, 64) | power-of-2 |
| `tl.max` / `tl.sum` | 1 / 1 | no keepdim |
| `tl.load` / `tl.store` | 3 / 1 | masked |
| `keepdim` | 0 | no-keepdim satisfied |
| `self.<attr>` writes | 4 | all in `__init__` |

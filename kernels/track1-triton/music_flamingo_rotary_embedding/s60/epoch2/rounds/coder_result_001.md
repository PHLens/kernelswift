# Coder Result 001

Result: candidate-ready

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md` @`378478c5cf21bbd28c6e7e7df413c5a9e270ce7e58a34f8d4abf2bc196f1278b` (hash re-verified from file; family "triton-launch-fusion"; change_scope "mixed"; expected_wall_improvement_pct 49.0 declared honestly)
- Sketch: `rounds/sketch_001.json` @`15c2055ed921227a35490a3d010e2ba730f4254bd76918ab50564908f6336827` (hash re-verified; matches decision `sketch_sha256`)
- Candidate: `triton_music_flamingo_rotary_embedding_e2_001.py` @`d47620a7777116f6cba97be6b37064be01adafff339706c3824cf44783d8e153` (project root: `kernels/track1-triton/music_flamingo_rotary_embedding/s60/epoch2/`)
- Canonical start (last_accepted_kernel): `baseline_adapter.py` @`9fc87abbe0e6268f06c969e94f5400abea51cdf315276a4ac5cef5bd0ad8a26f` — semantics derived from it; epoch-1 candidate `../triton_rotary_001.py` was read as prior evidence only (full-fusion `tl.cos`/`tl.sin` → -13%) — NOT copied; the kernel boundary deliberately stops at freqs and keeps cos/sin vendor.
- Base (immutable reference): `../../base.py` @`99829754f4cdc4bfd2808e051de549f0791414241e7fdbad7a1b8294a15be475` (re-verified unchanged; semantic authority for `__init__` inv_freq/position_angles construction)
- Harness: `/root/CodeBuddy/20260828202827/kernelswift/auto_bench.py` @`71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (untouched; AST loader; device-string rewriter maps 'cuda'→'gcu')

## Implementation Summary (decision-001 exact)

- ONE stateless `@triton.jit` kernel `_rotary_freqs_kernel`; single direct launch site in `ModelNew.forward` with grid `(B, seq_len)` = `(4, 32)` = **128 programs** (one per (b,t) pair), `num_warps=1`, `num_stages` unset.
- `HALF=32` (power-of-2 `tl.arange` extent, satisfies the S60 corrected power-of-2 constraint), `D=64` (per-branch freq width), `SEQ=seq_len`, `MAX_SEQ_LEN=max_seq_len` all passed as constexpr (frozen at call site).
- Kernel body per (b,t): `i = tl.arange(0, HALF)`; `bpos = b.to(tl.float32) / MAX_SEQ_LEN`; `bf = bpos * tl.load(inv_freq_ptr + i)`; `tf = tl.load(position_angles_ptr + t*(2*HALF) + i*2)` (even column read — position_angles is already repeat_interleave(2), adjacent columns duplicate); `angle = -tl.load(timestamps_ptr + b*SEQ + t) * 6.283185307179586`; `f_bf = bf*angle`, `f_tf = tf*angle`; store `freqs[b,t,2i]=f_bf`, `freqs[b,t,2i+1]=f_bf`, `freqs[b,t,64+2i]=f_tf`, `freqs[b,t,64+2i+1]=f_tf` (repeat_interleave emulated by writing f to BOTH adjacent columns).
- **Vendor cos/sin retention**: `forward` returns `(freqs.cos(), freqs.sin())` — cos/sin are vendor torch methods on the host against the kernel-written freqs buffer. The kernel contains **ZERO `tl.cos`/`tl.sin`** (the epoch-1 -13% root cause: GCU math-dialect trig ~44% slower than vendor).
- `__init__` constructs `inv_freq` [32] and `position_angles` [256,64] host-side, byte-for-byte identical to base, registered via `register_buffer` (state_dict keys `{inv_freq, position_angles}` unchanged; NOT fused).
- `forward` = THREE python-visible submissions (one `torch.empty` allocation + one Triton launch + two vendor torch.cos/torch.sin), collapsing base's ~13 eager elementwise launches.
- STATELESS: instance attrs exactly 2 constructor-config attrs (`max_seq_len`, `dim`); no caches, no workspace, no call-time instance writes; Triton JIT compile cache is framework-owned, one-time, absorbed by harness warmup 50.

## Sketch Primitive and Hint Conformance

- Required sketch hints bound:
  - `num_warps_1` (modality required): `num_warps=1` at the single launch site.
  - `tl_arange_power_of_2_32` (modality required): `tl.arange(0, HALF)` with HALF=32 (power-of-2).
  - `vendor_cos_sin_retention` (modality required): cos/sin realized exclusively by host `freqs.cos()` / `freqs.sin()`; kernel has no device trig.
- Primitives used, mapped to frozen profile:
  - `tl.program_id` → `parallel.program-id` (axis-0 and axis-1): 2-D program-id decomposition; each program owns exactly one (b,t).
  - `tl.arange` → `index.range.one-dimensional` (status supported): extent 32, power-of-2.
  - `tl.load` / `tl.store` → `memory.load` / `memory.store` (status supported): unmasked contiguous/strided loads/stores (all indices in-bounds by construction; no masking required).
  - `num_warps` → `resource.num-warps` (status constrained, legal set {1,2,4,8}): value 1, inside the legal set.
  - elementwise mul/div only — no `tl.dot` (pure elementwise, no GEMM), no reduction, no `tl.cos`/`tl.sin`, no atomics, no fp64, no `tl.make_block_ptr`, no `tl.async_copy`.
- DANGER tokens (compile/capture/graph/contiguous/torch.compile/TORCHINDUCTOR/reduce-overhead/copy_): all-zero (see binding audit).

## Binding Statement

- **Device-trig audit**: 0 `tl.cos` / 0 `tl.sin` call sites in the entire source (grep-verified); cos/sin realized exclusively by host `freqs.cos()` / `freqs.sin()` (vendor torch methods, count 1 each). This is the structural guarantee separating this round from epoch-1's -13% full fusion.
- **num_warps**: exactly one launch site, value `1`; `num_stages` absent (count 0). Kernel count: 1 (`@triton.jit` count 1).
- **tl.arange audit**: exactly 1 `tl.arange(0, HALF)` with HALF=32 (power-of-2); no non-power-of-2 extent.
- **Addressing audit**: direct strided addressing — `timestamps[b,t]` at `b*SEQ + t`; `inv_freq[i]` at `i`; `position_angles[t,2i]` at `t*(2*HALF) + i*2` (row stride 2*HALF=D=64); `freqs[b,t,d]` at `b*(SEQ*2*D) + t*(2*D) + d`. All strides constexpr-folded. `contiguous` count 0; `copy_` count 0.
- **Stateless audit**: 2 instance-attr writes, all in `__init__` (`self.max_seq_len`, `self.dim`); `inv_freq`/`position_angles` registered via `register_buffer` (buffer registration, not plain instance-attr assignment). No call-time instance writes.
- **AST-loader composition**: 4 imports + `@triton.jit` FunctionDef + ClassDef + 2 helper FunctionDefs (`get_inputs`, `get_init_inputs`) + `if __name__ == "__main__"` guard — all retained node types; `ast.parse` gate PASS (`AST_PARSE_OK`).

## Deviations

- None. All decision-001 intervention points, invariants, and sketch declarations are satisfied as specified.

## Evidence for Verifier

- Candidate: `triton_music_flamingo_rotary_embedding_e2_001.py` @`d47620a7777116f6cba97be6b37064be01adafff339706c3824cf44783d8e153`.
- Canonical measurement route (Verifier-owned): unchanged harness `auto_bench.py`, device gcu, warmup 50 / repeat 100 interleaved pairs.
- Coder ran NO timing/benchmark/profiler measurements and claims NO verdict on speedup. Classification is `candidate-ready`. The correctness smoke reported `PASS` (accuracy), which is the only signal Coder reports.

### Artifact hash ledger

```text
d47620a7777116f6cba97be6b37064be01adafff339706c3824cf44783d8e153  triton_music_flamingo_rotary_embedding_e2_001.py
378478c5cf21bbd28c6e7e7df413c5a9e270ce7e58a34f8d4abf2bc196f1278b  rounds/decision_001.md
15c2055ed921227a35490a3d010e2ba730f4254bd76918ab50564908f6336827  rounds/sketch_001.json
9fc87abbe0e6268f06c969e94f5400abea51cdf315276a4ac5cef5bd0ad8a26f  baseline_adapter.py
99829754f4cdc4bfd2808e051de549f0791414241e7fdbad7a1b8294a15be475  ../../base.py
71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29  /root/CodeBuddy/20260828202827/kernelswift/auto_bench.py
```

### Binding audit table (source counts)

| Token / construct | Count | Verdict |
|---|---|---|
| `compile` / `capture` / `graph` / `contiguous` | 0 / 0 / 0 / 0 | DANGER-free |
| `torch.compile` / `TORCHINDUCTOR` / `reduce-overhead` / `copy_` | 0 / 0 / 0 / 0 | DANGER-free |
| `tl.cos` / `tl.sin` | 0 / 0 | vendor trig retained (epoch-1 lesson) |
| `freqs.cos()` / `freqs.sin()` (vendor) | 1 / 1 | host torch.cos/torch.sin |
| `tl.dot` / reduction | 0 / 0 | pure elementwise |
| `num_warps` | 1 (value 1) | legal |
| `@triton.jit` | 1 | single kernel |
| `tl.arange` | 1 (extent 32) | power-of-2 |
| `tl.load` / `tl.store` | 3 / 4 | unmasked, in-bounds |
| `self.<attr>` writes | 2 | all in `__init__` |

### Self-check gate results

- AST: `AST_PARSE_OK`
- DANGER tokens: all 0
- `tl.cos`/`tl.sin`: 0
- `@triton.jit` kernel count: 1
- `num_warps=1`: 1
- correctness smoke: `PASS` (accuracy)

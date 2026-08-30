# Coder Result 001

## Result

`candidate-ready`

The candidate conforms to the immutable design in `decision_001.md`. The fused
elementwise Triton kernel compiles and runs on the GCU runtime with exact
numerical agreement against `base.py` (max abs err 0.0 at the compared inputs).

## Identity

- Round: `001`
- Decision: `decision_001.md`
- Decision SHA256: (decision_001.md, prose — not part of the machine ledger)
- Selected target profile: `triton_gcu`
- Runtime fingerprint: `project.md#runtime-fingerprint` (matched; no mismatch)
- Source canonical: `baseline_adapter.py`
- Source canonical SHA256: `9fc87abbe0e6268f06c969e94f5400abea51cdf315276a4ac5cef5bd0ad8a26f`
- Candidate: `triton_rotary_001.py`
- Candidate SHA256: `74a960f54b4519f43948d4c2e374d9d93edbf76645ab7adc0d63595f5a4802b5`

## Implementation Decision

The candidate fuses the entire forward elementwise/view chain into a single
Triton kernel `_rotary_embedding_kernel` that writes both `cos` and `sin`
output buffers in one launch. The grid is one flattened program over the
`[B=4, T=seq_len=32, D=128]` = 16384-element output, with a `guard offs < 16384`
mask (matching the decision sketch's `guard index < 16384`).

Per output element `(b, t, d)`:

1. Load `ts = timestamps[b, t]`, compute `angle = -ts * 2π`.
2. Branch on `is_time = d >= D` (D=64):
   - batch branch (`d < 64`): `batch_freq = (b / max_seq_len) * inv_freq[d // 2]`
     — this reproduces the eager `arange(B)/max_seq_len @ inv_freq` followed by
     `repeat_interleave(2)`, since `repeat_interleave(2)` maps column `d` to
     source column `d // 2`.
   - time branch (`d >= 64`): `time_freq = position_angles[t, d - 64]`, i.e. a
     direct load from the precomputed `position_angles` buffer (whose
     `repeat_interleave(2)` is already baked host-side).
3. `theta = freq * angle`.
4. `c = tl.cos(theta)`, `s = tl.sin(theta)`, stored to `cos_out` / `sin_out`.

`inv_freq` [32] and `position_angles` [256,64] remain host-side `register_buffer`
tensors computed identically to `baseline_adapter.py.__init__` and loaded by the
kernel; they are NOT fused (per the decision's fusion scope). `seq_len` and
`max_seq_len` are Python ints passed as runtime scalars to the kernel.

`forward` returns a Python tuple `(cos_out, sin_out)`, both `[4,32,128]` fp32.

## Key Design Notes

- **repeat_interleave semantics**: the eager `batch_freqs` is `[4,64]` produced by
  `(arange(B)/max_seq_len)[:,None] * inv_freq[None,:]` (shape `[4,32]`) then
  `repeat_interleave(2, dim=-1)` -> `[4,64]`. Column `d` therefore reads
  `inv_freq[d // 2]`, which the kernel computes directly as `inv_idx = (d // 2) % D`.
  The `% D` is a safety no-op for `d < 64` but keeps the index in `[0,32)`.
- **position_angles branch**: the time branch loads `position_angles[t, d - D]`
  directly; no `d//2` recomputation is needed because the buffer already carries
  the repeat_interleave expansion (columns are `[k0,k0,k1,k1,...]`).
- **branch select**: `tl.where(is_time, pa, batch_freq)` implements the decision
  sketch's `freq = select(d < 64, batch_freq, pa)`.
- **flattened indexing**: `b = offs // (seq_len*2D)`, `rem = offs % (seq_len*2D)`,
  `t = rem // (2D)`, `d = rem % (2D)`. All int32 arithmetic; `b` is cast to fp32
  only where multiplied into `batch_freq`.

## GCU Adaptation Points

- `import torch_gcu` / `import triton_gcu` (no `torch_mlu`).
- `device` literal `"cuda"` in `get_inputs` — the harness `_rewrite_device_for_backend`
  rewrites `"cuda"` -> `"gcu"` for the GCU target. This matches the project
  invariant "device literal 'cuda' is rewritten to 'gcu'".
- Direct Triton launch `_rotary_embedding_kernel[grid](...)` with `num_warps=1`
  (the only proven launch configuration on this architecture). No `fast_libentry`.
- `grid = (triton.cdiv(total, block),)` with `block = next_power_of_2(16384) = 16384`,
  so a single one-program launch over the full output.

## Capability Verification: tl.cos / tl.sin

The single largest capability risk for this round (per the decision's Pitfalls
section) was whether `tl.cos` and `tl.sin` compile and lower on the GCU runtime —
they are absent from the `triton_gcu` profile's Supported/Constrained/Unknown
tables.

A minimal compile+run probe (`/tmp/probe_cos_sin.py`) was executed on the S60
GCU runtime:

- Kernel loads an fp32 vector, applies `tl.cos` / `tl.sin`, stores two buffers.
- `num_warps=1`, direct launch, `torch.gcu.synchronize()`.
- Result: **compiled and ran successfully**. max abs err vs `torch.cos`/`torch.sin`:
  `cos = 1.19e-7`, `sin = 5.96e-8` — within fp32 rounding.

Conclusion: `tl.cos` and `tl.sin` are **available** on this runtime. No
capability-miss, no eager fallback. The full candidate smoke test then matched
`base.py` with max abs err `0.0`.

## Attempt Ledger

| # | Command | Exit | Defect | Before hash | After hash |
|---|---|---|---|---|---|
| 1 | `python3 -m py_compile triton_rotary_001.py` | 0 | - | - | (first write) |
| 2 | smoke test `/tmp/probe_rotary_smoke.py` | 1 | `CompilationError` — kernel accessed module global `TWO_PI`; Triton forbids non-constexpr globals in `@triton.jit` | (first write) | (pre-fix) |
| 3 | replaced `TWO_PI` global with literal `6.283185307179586` inside the kernel | 0 | - | (pre-fix) | `74a960f5...` |
| 4 | smoke test `/tmp/probe_rotary_smoke.py` | 0 | - | `74a960f5...` | `74a960f5...` |

Smoke outcome: `cos allclose=True`, `sin allclose=True`, max abs err `0.0`,
shapes `[4,32,128]`, output is a `tuple`.

## Gate Results

- `ast.parse`: pass (implicitly via `py_compile`).
- Real harness loader (`auto_bench.load_ks_module`): pass — `ModelNew`,
  `get_inputs`, `get_init_inputs` all resolve.
- `py_compile`: pass.
- Compile + warm-up smoke execution on GCU: pass (see smoke test above).

## Conformance Notes (candidate-ready, non-semantic)

- Removed a module-level `TWO_PI` constant and inlined the literal `2π` into the
  kernel, because Triton `@triton.jit` functions cannot read non-constexpr module
  globals. This is purely a target-language accommodation; the numeric value is
  identical (`6.283185307179586` = `2 * math.pi`), preserving all normative
  semantics.
- Used `b.to(tl.float32)` for the batch index before the multiply; int32 indexing
  is retained for all load/store address computation.

## Deviations from Decision

None. The implementation matches the Unified Sketch (branch select on `d < 64`,
single elementwise map, one launch writing two buffers) and the Host Plan
(`not-applicable`: no host-state/allocation/lifecycle change beyond the two
output buffer allocations inside `forward`, which are local to the call and not
part of module state).

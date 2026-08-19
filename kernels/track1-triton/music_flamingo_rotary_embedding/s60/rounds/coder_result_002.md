# Coder Result 002

## Result

`candidate-ready`

The candidate conforms to the immutable design in `decision_002.md` (grid
parallelism). The fused elementwise kernel body is unchanged from Round 1; only
the launch grid and tile size changed, partitioning the 16384-element flat map
across 128 programs instead of one serial program. The candidate compiles and
runs on the GCU runtime with exact numerical agreement against `base.py`
(max abs err 0.0).

## Identity

- Round: `002`
- Decision: `decision_002.md`
- Selected target profile: `triton_gcu`
- Runtime fingerprint: `project.md#runtime-fingerprint` (matched; no mismatch)
- Source canonical: `baseline_adapter.py`
- Source canonical SHA256: `9fc87abbe0e6268f06c969e94f5400abea51cdf315276a4ac5cef5bd0ad8a26f`
- Round 1 candidate (kernel-body source): `triton_rotary_001.py`
- Candidate: `triton_rotary_002.py`
- Candidate SHA256: `a1c5c38a4ecd0a038ebbcd9e6f04b0b5a18e437aea8acabd4104a4e5d579ad9d`

## Implementation Decision

This round is a launch-grid repair, not a new algorithm. The kernel body is
byte-for-byte identical to `triton_rotary_001.py` in its computational content
(the fused elementwise map, branch select `is_time = d >= D`, `tl.cos`/`tl.sin`,
single launch writing both buffers). The only change is in `forward`:

- Round 1 (defect): `block = triton.next_power_of_2(16384) = 16384` →
  `grid = (1,)` → a single program with `num_warps=1` serially processed all
  16384 elements, exposing zero device parallelism (device `multi_processor_count=2`).
- Round 2 (fix): `BLOCK = 128`, `grid = (total // BLOCK,) = (128,)` → 128 programs
  each process 128 consecutive flattened elements concurrently.

The kernel keeps the same flattened-index decomposition:
`b = offs // (seq_len*2D)`, `rem = offs % (seq_len*2D)`,
`t = rem // (2D)`, `d = rem % (2D)`, with `guard offs < 16384`. It uses a 1-D
grid (only `tl.program_id(0)`, the proven axis) and recovers `(b, t, d)` in-kernel
rather than a 3-D grid. `num_warps=1` retained; `num_stages` not asserted.

## Capability Verification: tl.arange extent 128

The new capability risk (decision_002 Pitfalls) was that `tl.arange` is only
proven at extent 16 and extent 4 in the `triton_gcu` profile; a `BLOCK=128` tile
is an unproven extent.

A minimal compile+run probe (`/tmp/probe_arange128.py`) was executed on the S60
GCU runtime:

- `tl.arange(0, 128)` inside a kernel, `grid = (n // 128,)`, `num_warps=1`,
  `torch.gcu.synchronize()`.
- Result: **compiled and ran successfully**. `BLOCK=128 grid=(128,)`,
  `max_abs_err=0.0`.

Conclusion: `tl.arange` extent 128 is **available** on this runtime. No
capability-miss, no fallback to `BLOCK=16`/`grid=1024` was needed. (Note: the
16384 elements divide evenly by 128, so every program has exactly 128 in-range
elements and the `mask` is a no-op; the guard is retained for correctness.)

`tl.cos`/`tl.sin` remain PROVEN available (Round 1 coder probe).

## Key Design Notes

- The full candidate smoke test matched `base.py` with max abs err `0.0` for both
  cos and sin, shapes `[4,32,128]`, output a Python tuple.
- `state_dict` contract unchanged: `inv_freq` [32] and `position_angles` [256,64]
  remain host-side `register_buffer` tensors (not parameters), computed in
  `__init__` identically to `baseline_adapter.py`.
- `get_inputs` returns `[timestamps ("cuda" literal, harness-rewritten to "gcu"),
  seq_len int]`; `get_init_inputs` returns `[64, 256, 10000.0]` — unchanged.

## Attempt Ledger

| # | Command | Exit | Defect | Before hash | After hash |
|---|---|---|---|---|---|
| 1 | `python3 -m py_compile triton_rotary_002.py` | 0 | - | - | `a1c5c38a...` |
| 2 | `/tmp/probe_arange128.py` (extent 128 compile probe) | 0 | - | - | - |
| 3 | smoke test `/tmp/probe_rotary002_smoke.py` | 0 | - | `a1c5c38a...` | `a1c5c38a...` |
| 4 | harness loader `load_ks_module` | 0 | - | `a1c5c38a...` | `a1c5c38a...` |

## Gate Results

- `ast.parse`: pass (implicitly via `py_compile`).
- Real harness loader (`auto_bench.load_ks_module`): pass — `ModelNew`,
  `get_inputs`, `get_init_inputs` all resolve.
- `py_compile`: pass.
- Compile + warm-up smoke execution on GCU: pass (cos/sin allclose, max abs err 0.0).

## Conformance Notes (candidate-ready, non-semantic)

- None beyond the launch-grid change itself, which is the decision's normative
  intervention.

## Deviations from Decision

None. BLOCK=128, grid=(128,), num_warps=1, 1-D grid with in-kernel index
decomposition, kernel body unchanged, single launch writing both buffers —
exactly per the Unified Sketch and Host Plan (`not-applicable`).

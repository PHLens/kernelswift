# Coder Result 001

Result: `candidate-ready`

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md`
- Decision SHA256: `dc0a4837cc8a5aeb867e9d71f8c1e4bc1930ee57d431a279f761329271e5371a`
- Selected target profile: `triton_cuda`
- Runtime fingerprint: `project.md#runtime-fingerprint` (torch 2.7.1, triton 3.1.0, Iluvatar BI-V150, capability (7,1))
- Source canonical: `baseline_adapter.py`
- Source canonical SHA256: `98cf1e1e5f493ae4ae08c9391f00db83b3cd783a1b194d9c0d63851091133a5d`
- Base SHA256: `28d4d213638e5adc7f6bc52928c8f7596cd8446e3e0201beab364b5c3c7988fc`
- Candidate: `triton_mhc_head_compute_mix_backward_001.py`
- Candidate SHA256: `5d419f5d2e920abf3cf583a22f155e76047f9e5bc3a5cc36baca5477fae94349`

## Implementation

The candidate fuses the full sigmoid-backward chain — the affine modulation
`z = input_mix * mhc_scale + mhc_base`, `torch.sigmoid(z)`, the sigmoid backward
`grad_z = grad_out * σ(z) * (1 - σ(z))`, `grad_input_mix = grad_z * mhc_scale`,
and the two sum reductions — into a **single Triton kernel** (one launch per
forward call), matching the Optimization Intent `change_family: kernel-fusion`.

Kernel structure (`_mhc_head_compute_mix_backward_kernel`):

- The `[2, 1024, 4]` fp32 inputs are viewed as `[R=2048, 4]`
  (`R = batch0 * batch1`); the last dim `mhc_mult = 4` is fixed and small.
- Grid is `(ceil(R / BLOCK),)` with `BLOCK = 128` rows per program (16 row
  blocks total). One-dimensional grid + `tl.program_id(0)` matches the recorded
  BI150 launch convention.
- Each program loads a `[BLOCK, 4]` tile of `input_mix` and `grad_out`, loads the
  scalar `mhc_scale[0]` and the `[4]` `mhc_base` vector, then computes in
  registers: `z -> sigmoid -> grad_z -> grad_input_mix`, storing
  `grad_input_mix` with a row mask.
- On-chip reductions:
  - `grad_mhc_base = sum(grad_z, dim=(0,1))` is realized as `tl.sum(gz, axis=0)`
    over the `[BLOCK, 4]` tile, producing a per-column `[4]` partial that is
    accumulated into the global `[4]` output with `tl.atomic_add`.
  - `grad_mhc_scale = sum(grad_z * input_mix)` (full 3-D reduction) is realized
    as `tl.sum(gz * im)` (scalar full reduction over the tile), accumulated into
    the global `[1]` output with `tl.atomic_add`.
- The two reduction contracts are preserved exactly:
  - `grad_mhc_base` sums `grad_z` over the batch dims `(0,1)` while keeping the
    last dim (`mhc_mult=4`) — the axis-0 reduce keeps the column structure.
  - `grad_mhc_scale` is the full sum of `grad_z * input_mix` (not `grad_z`
    alone), matching `(grad_z * input_mix).sum(dim=(0,1,2))`.

Reduction path decision: the decision authorized a two-program fallback if the
multi-dim reduce failed to lower. A local file-backed probe (see Gate Evidence)
confirmed on the BI150 runtime that `tl.sigmoid`, `tl.sum(axis=0)` over
`(BLOCK, 4)`, full scalar `tl.sum`, and `tl.atomic_add` all lower correctly and
reproduce the reference within `1e-5` abs error. Therefore the **single-kernel
atomic-accumulate path** was used — strictly stronger than the authorized
fallback and still within the `kernel-fusion` change family.

Numerical semantics: all computation in fp32; `tl.sigmoid` matches
`torch.sigmoid`; no intermediate tensors (`z`, `sigmoid`, `grad_z`) are
materialized. Outputs are re-viewed to the exact reference shapes
`grad_input_mix [2,1024,4]`, `grad_mhc_scale [1]`, `grad_mhc_base [4]`, all
fp32, in the exact return order `(grad_input_mix, grad_mhc_scale, grad_mhc_base)`.

## Gate Evidence

| Gate | Command | Observation | Verdict |
|---|---|---|---|
| decision validation | `python3 skills/kernel-opt-loop/scripts/validate_decision.py .../decision_001.md --expected-profile triton_cuda` | `"valid": true`, `target_profile: triton_cuda` | pass |
| local primitive probe | `/tmp/bi150_probe_mhcbwd.py` (file-backed) | `err_gim=7.15e-7`, `err_base=7.15e-6`, `err_scale=9.54e-7`; `tl.sigmoid`/`tl.sum(axis=0)`/`tl.sum`/`tl.atomic_add` all lower and match reference | pass |
| AST syntax | `python3 -m py_compile .../triton_mhc_head_compute_mix_backward_001.py` | return code 0 | pass |
| harness smoke | `python3 auto_bench.py --v0_file .../base.py --v1_file .../triton_mhc_head_compute_mix_backward_001.py --warmup 50 --repeat 100 --full-traceback` | `PASS accuracy; v0=0.346594 ms, v1=0.197791 ms, speedup=1.752x`; return code 0 | pass |

The harness AST loader retains the `@triton.jit` top-level function (a
`FunctionDef`) and the top-level `ModelNew` class; the module exposes
`ModelNew`/`get_inputs`/`get_init_inputs` as required.

## Conformance

- `ModelNew()` constructor takes no arguments; `get_init_inputs()` returns `[]`.
- `forward(input_mix, mhc_scale, mhc_base, grad_out)` preserves the reference
  argument and return signature `(grad_input_mix, grad_mhc_scale, grad_mhc_base)`.
- Output structure, shapes, and dtypes match the reference exactly.
- Sigmoid-backward numerical semantics preserved (`grad_z = grad_out * σ * (1-σ)`).
- Both reduction contracts preserved (`sum(dim=(0,1))` keeping last dim for
  `grad_mhc_base`; full `sum(grad_z * input_mix)` for `grad_mhc_scale`).
- No edit to `base.py`, `baseline_adapter.py`, `decision_001.md`, `team-state.md`,
  `project.md`, or the harness.

No normative semantic deviation; no `capability-miss`. The single-kernel
atomic-accumulate reduce is a conformance note (a stronger realization of the
authorized kernel-fusion path), not a new design.

## Attempt Ledger

| Attempt | Command | Exit | Defect | Before SHA | After SHA |
|---|---|---|---|---|---|
| 1 | local probe (first draft, merged scale in/out buffer) | 1 | scale input/output buffer aliasing produced wrong values in the throwaway probe only | n/a (probe) | n/a (probe) |
| 2 | local probe (separate scale_in / accumulators) | 0 | none | n/a (probe) | n/a (probe) |
| 3 | write candidate + `py_compile` + harness smoke | 0 | none | - | `5d419f5d...` |

The only defect was in a throwaway `/tmp` probe (buffer aliasing), not in the
candidate. The candidate was written once and passed all gates on the first
attempt.

## Handoff

Candidate is ready for Verifier measurement. Candidate path:

`kernels/track1-triton/mhc_head_compute_mix_backward/bi150/triton_mhc_head_compute_mix_backward_001.py`

Smoke timing is indicative only (v0=0.346594 ms, v1=0.197791 ms, speedup=1.752x);
Verifier owns authoritative wall timing and profiler evidence.

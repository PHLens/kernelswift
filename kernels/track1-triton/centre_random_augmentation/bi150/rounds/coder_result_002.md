# Coder Result 002

Result: candidate-ready

## Identity

- Round: `002`
- Decision: `kernels/track1-triton/centre_random_augmentation/bi150/rounds/decision_002.md`
- Decision SHA256: `2290e37b81072b794ca5735dddba52ed19805c943a8e7109b598e5fd1f65af8e`
- Candidate: `kernels/track1-triton/centre_random_augmentation/bi150/triton_centre_random_augmentation_002.py`
- Candidate SHA256: `efac6ee782e859701bb14aca04b7f56516a575a5f74507958e1930a95005a530`
- Canonical reference (last_accepted_kernel): `triton_centre_random_augmentation_001.py`
- Canonical reference SHA256: `4e33276ec28f3695aa08462aa6cb796a160aca47dad889168a7cdd8aa8e16036`
- Selected target profile: `triton_cuda`
- Runtime fingerprint: `project.md#runtime-fingerprint` (triton 3.1.0 / torch 2.7.1 / Iluvatar BI-V150, capability (7,1))
- Measurement fingerprint: `a5f980780c4dcde731df913710ad9dfded4f07a66b90e334fea0a6f2aa1fd5fa`
- validate_decision: `valid=true` (with `--expected-profile triton_cuda`)

## Implementation

### Kernel structure

`_centre_aug_kernel` (grid `(4,)`, one program per sample, `BLOCK_N=256`) now
absorbs the **entire** deterministic transform, in reference order:

1. Load scalar `u1s/u2s/u3s` for the sample.
2. Quaternion construction (deterministic, from identical `u` values):
   `q1 = sqrt(1-u1)*sin(2π u2)`, `q2 = sqrt(1-u1)*cos(2π u2)`,
   `q3 = sqrt(u1)*sin(2π u3)`, `q4 = sqrt(u1)*cos(2π u3)`.
3. Quaternion -> 3x3 rotation matrix (9-entry arithmetic: `xx/yy/zz/xy/xz/yz/wx/wy/wz`
   and the 9 matrix entries).
4. Centering (`tl.sum` reductions, `eps=1e-12`).
5. `rot_vec_mul` (3x3-by-3 vector product) + `+ T` translation + `* mask` multiply.
6. Store three output columns.

### Randomness boundary (critical)

- `u1 = torch.rand(n_sample)`, `u2 = torch.rand(n_sample)`,
  `u3 = torch.rand(n_sample)`, then `T = s_trans * torch.randn(n_sample, 3)` —
  all host-side, inside `forward`, in the exact reference order (3x `torch.rand`
  then 1x `torch.randn`). The fused kernel only reads `u1/u2/u3/T` and performs
  deterministic math; it never draws random numbers.
- This preserves the harness's per-call `set_seed(42)` contract, so v0 and v1
  consume the RNG identically and produce bit-identical `u1/u2/u3/T`.

### Transcendental lowering (the key capability question)

A file-backed local probe confirmed that `tl.sqrt`, `tl.sin`, and `tl.cos`
**lower successfully** on the CoreX Triton 3.1.0 BI150 backend and produce
**bit-identical** results to `torch.sqrt/sin/cos` (max abs diff `0.0` for the
four quaternion entries across 4 samples). Therefore the quaternion construction
is **not** a capability-miss; no unproven approximation was substituted.

`2 * math.pi` is inlined as the literal `6.283185307179586` inside the kernel,
because Triton JIT cannot resolve module-level Python globals (and the AST
loader would strip a non-literal `Assign`). This is a conformance note (literal
vs. `math.pi` reference), not a design change.

### Fallback paths

- `centre_only=True`: reproduces the reference early-return (no RNG consumed).
- `mask is None`: reproduces the reference mask-None dataflow via a host-side
  `_random_rotation_matrices_host` helper + `rot_vec_mul` (no fusion). Not
  exercised by the harness (always supplies a non-None all-ones mask).

## Gate Evidence

| Gate | Command | Result | Evidence |
|---|---|---|---|
| Decision validation | `validate_decision.py decision_002.md --expected-profile triton_cuda` | pass | `valid=true` |
| AST parse | `python3 -m py_compile .../triton_centre_random_augmentation_002.py` | pass | exit `0` |
| Harness loader | AST loader loaded `ModelNew/get_init_inputs/get_inputs` + `@triton.jit` top-level function | pass | smoke run without load/constructor error |
| Accuracy smoke | `auto_bench.py --v0_file base.py --v1_file candidate_002 --warmup 50 --repeat 100 --full-traceback` | pass | `PASS accuracy; v0=1.013514 ms, v1=0.240272 ms, speedup=4.218x` |

### Primitive conformance

| Primitive | Profile status | Used? | Note |
|---|---|---|---|
| `tl.load` | Supported (contiguous fp32) | yes | `[256]` vector + scalar loads |
| `tl.store` | Supported | yes | three `[256]` column stores |
| `tl.arange` | Supported (extent 256) | yes | `tl.arange(0, BLOCK_N)` |
| `tl.program_id` | Supported (axis 0) | yes | `grid=(4,)` |
| `tl.sum` | Supported (axis-0 over `(256,)`) | yes | centering |
| `tl.sqrt` | Unknown on profile; locally proven | yes | probe: max abs diff 0.0 vs torch |
| `tl.sin` | Unknown on profile; locally proven | yes | probe: max abs diff 0.0 vs torch |
| `tl.cos` | Unknown on profile; locally proven | yes | probe: max abs diff 0.0 vs torch |

The `tl.sqrt/sin/cos` primitives were `Unknown` on the profile; a file-backed
local probe (not stdin) proved their lowering and numerical equivalence before
use. No `num_warps`/`num_stages`/block pointers/mixed precision are required.

## Conformance

- Public contract preserved: `ModelNew(n_sample=4, s_trans=1.0, centre_only=False)`,
  `forward(x_input_coords, mask) -> out[4,256,3] fp32`.
- `get_init_inputs()` returns `[4, 1.0, False]`; `get_inputs()` returns
  `[x_input_coords, mask]` seeded by `torch.manual_seed(42)`.
- RNG consumption order (3x `torch.rand` + 1x `torch.randn`) preserved exactly.
- Centering formula `sum / (sum + eps)` with `eps=1e-12` preserved.
- Quaternion-to-matrix construction numerically compatible (bit-identical on
  identical `u`, verified by probe and by accuracy PASS).
- Output dtype/shape/device unchanged; `forward` does not mutate inputs and
  preserves the caller-selected device/stream.
- No new host-side state, cache, buffer reuse, or lifecycle semantics
  (Host Plan: `not-applicable`).

## Attempt Ledger

| Attempt | Command | Exit | Defect | Candidate before | Candidate after |
|---|---|---|---|---|---|
| 1 | `py_compile` + smoke 50/100 | non-zero | `NameError: _PI2 is not defined` (Triton JIT cannot resolve module global `_PI2`) | `(initial)` | `efac6ee7...` |
| 2 | `py_compile` + smoke 50/100 | 0 | - | `efac6ee7...` | `efac6ee7...` (unchanged) |

Defect repair was a non-semantic fix (inlined the `2*math.pi` literal into the
kernel and removed the module-level `_PI2` global). No semantic change was made.

## Handoff

- Candidate is `candidate-ready`: accuracy PASS (speedup 4.218x on smoke timing;
  authoritative wall timing is Verifier's), no semantic deviation from the
  immutable decision, and the transcendental lowering was locally proven (not a
  capability-miss).
- Verifier must perform authoritative runtime verification; Coder does not
  return `accepted`.

## Exact Reproduction Commands

```bash
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
cd /root/CodeBuddy/20260818191200/kernelswift
python3 skills/kernel-opt-loop/scripts/validate_decision.py kernels/track1-triton/centre_random_augmentation/bi150/rounds/decision_002.md --expected-profile triton_cuda
python3 -m py_compile kernels/track1-triton/centre_random_augmentation/bi150/triton_centre_random_augmentation_002.py
python3 auto_bench.py --v0_file kernels/track1-triton/centre_random_augmentation/base.py --v1_file kernels/track1-triton/centre_random_augmentation/bi150/triton_centre_random_augmentation_002.py --warmup 50 --repeat 100 --full-traceback
```

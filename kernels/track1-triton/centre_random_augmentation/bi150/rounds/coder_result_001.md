# Coder Result 001

Result: candidate-ready

## Identity

- Round: `001`
- Decision: `kernels/track1-triton/centre_random_augmentation/bi150/rounds/decision_001.md`
- Decision SHA256: `ad2f891ebb8929b7c8b290388081573f25dbb78dc39ab04585cf258e99a1156b`
- Candidate: `kernels/track1-triton/centre_random_augmentation/bi150/triton_centre_random_augmentation_001.py`
- Candidate SHA256: `4e33276ec28f3695aa08462aa6cb796a160aca47dad889168a7cdd8aa8e16036`
- Canonical reference (last_accepted_kernel): `baseline_adapter.py`
- Canonical reference SHA256: `012754740961f6ec10d515563e51cd07eeaf35caefe33731d5c1e9a88387fe9b`
- Selected target profile: `triton_cuda`
- Runtime fingerprint: `project.md#runtime-fingerprint` (triton 3.1.0 / torch 2.7.1 / Iluvatar BI-V150, capability (7,1))
- Measurement fingerprint: `a5f980780c4dcde731df913710ad9dfded4f07a66b90e334fea0a6f2aa1fd5fa`
- validate_decision: `valid=true` (with `--expected-profile triton_cuda`)

## Implementation

### Kernel structure

A single fused Triton kernel `_centre_aug_kernel` with `grid=(n_sample,)`
(`= (4,)`), one program per sample; `BLOCK_N = next_power_of_2(N_ATOM) = 256`.
Each program:

1. Loads the `[256]` mask and the three coordinate columns `x0/x1/x2`
   (`[256]` each) via contiguous `tl.load`.
2. Computes the centering via `tl.sum` reductions (masked, `eps=1e-12`):
   `center_j = sum_i(xc[i,j] * m[i]) / (sum_i(m[i]) + eps)` for `j = 0,1,2`.
3. Loads the 9 scalar entries of `R[s]` and 3 scalar entries of `T[s]`
   (flattened `[n_sample,9]` / `[n_sample,3]`, row-major).
4. Applies `rot_vec_mul` (`out_i = sum_j R[i,j] * x_centered_j`), the
   `+ T` translation, and the `* mask` multiply in one elementwise chain.
5. Stores the three output columns back to `out[s, :, j]`.

The centering reduction is recomputed per program (sample-independent, tiny
data), trading a trivial amount of redundant reduction work for a simpler
single-level grid with no cross-program communication.

### Randomness boundary (critical)

- `random_rotation_matrices` is kept **byte-identical** to the reference: three
  `torch.rand(n)` draws (`u1, u2, u3`) followed by the quaternion-to-rotation
  construction (`sqrt`/`sin`/`cos`/`mul`/`stack`/`reshape`), all on the host.
- `T = s_trans * torch.randn(n_sample, 3)` follows it, exactly one normal draw.
- Both are called **inside `forward`**, in the reference order (3x `torch.rand`
  then 1x `torch.randn`). No random draw is moved into the Triton kernel, and
  none is reordered/added/removed.

This preserves the harness's per-call `set_seed(42)` contract: v0 and v1 consume
the RNG identically, producing bit-comparable `R`/`T`.

### Fusion scope (conformance note)

The deterministic chain fused into the kernel is: centering subtraction
(`reduce sum` + subtract), the 3x3-by-3 vector product `rot_vec_mul`, the `+ T`
translation, and the `* mask` multiply. The quaternion-to-rotation-matrix
elementwise conversion (`sqrt`/`sin`/`cos`/`stack`) is left on the host inside
`random_rotation_matrices`. The decision explicitly allows this conversion to be
done either inside the kernel or on the host ("由你判断"); keeping it on the
host preserves bit-identical `R` (avoiding any `tl.sqrt`/`tl.sin`/`tl.cos` vs
`torch` libdevice numerical divergence) while still collapsing the dominant
`rot_vec_mul`/`add`/`mul`/`mask` kernel mass. This is a conformance note, not a
design change.

### Centre-only and mask-None fallbacks

- `centre_only=True`: reproduces the reference early-return (no RNG consumed).
- `mask is None`: reproduces the reference mask-None dataflow with the original
  `rot_vec_mul` host function (no fusion), preserving semantics. The harness
  always supplies a non-None all-ones mask, so this path is not exercised.

## Gate Evidence

| Gate | Command | Result | Evidence |
|---|---|---|---|
| Decision validation | `python3 skills/kernel-opt-loop/scripts/validate_decision.py .../decision_001.md --expected-profile triton_cuda` | pass | `valid=true` |
| AST parse | `python3 -m py_compile .../triton_centre_random_augmentation_001.py` | pass | exit `0` |
| Harness loader | `auto_bench.py` AST loader loaded `ModelNew/get_init_inputs/get_inputs` and the `@triton.jit` top-level function | pass | smoke run completed without load/constructor error |
| Accuracy smoke | `auto_bench.py --v0_file base.py --v1_file candidate --warmup 50 --repeat 100 --full-traceback` | pass | `PASS accuracy; v0=1.024789 ms, v1=0.719950 ms, speedup=1.423x` |

### Primitive conformance

| Primitive | Profile status | Used? | Note |
|---|---|---|---|
| `tl.load` | Supported (contiguous fp32) | yes | `[256]` vector loads + 9/3 scalar loads |
| `tl.store` | Supported (contiguous fp32) | yes | three `[256]` column stores |
| `tl.arange` | Supported (extent 256) | yes | `tl.arange(0, BLOCK_N)`, `BLOCK_N=256` |
| `tl.program_id` | Supported (axis 0, 1-D launch) | yes | `grid=(4,)` |
| `tl.sum` | Supported (axis-0 over `(256,)` fp32) | yes | centering reductions |
| scalar `tl.load` | not explicitly listed | yes | 9+3 scalar entries; exercised and passed in smoke |
| `tl.sin`/`tl.cos`/`tl.sqrt` | not used in kernel | no | kept on host (conformance note above) |

No `num_warps`, `num_stages`, block pointers, or mixed precision are used, so no
Unknown/Unsupported primitive is required.

## Conformance

- Public contract preserved: `ModelNew(n_sample=4, s_trans=1.0, centre_only=False)`,
  `forward(x_input_coords, mask) -> out[4,256,3] fp32`.
- `get_init_inputs()` returns `[4, 1.0, False]`; `get_inputs()` returns
  `[x_input_coords, mask]` seeded by `torch.manual_seed(42)`.
- RNG consumption order (3x `torch.rand` + 1x `torch.randn`) preserved exactly.
- Centering formula `sum / (sum + eps)` with `eps=1e-12` preserved.
- Output dtype/shape/device unchanged; `forward` does not mutate inputs and
  preserves the caller-selected device/stream.
- No new host-side state, cache, buffer reuse, or lifecycle semantics
  introduced (Host Plan: `not-applicable`).

## Attempt Ledger

| Attempt | Command | Exit | Defect | Candidate before | Candidate after |
|---|---|---|---|---|---|
| 1 | `py_compile` | 0 | - | - | `4e33276e...` |
| 2 | accuracy smoke 50/100 | 0 | - | `4e33276e...` | `4e33276e...` (unchanged) |

No repair was required; the candidate compiled and passed accuracy on the first
attempt.

## Handoff

- Candidate is `candidate-ready`: accuracy PASS (speedup 1.423x on smoke timing;
  authoritative wall timing is Verifier's), no semantic deviation from the
  immutable decision.
- The candidate must be benchmarked/verified by Verifier; Coder does not return
  `accepted`.

## Exact Reproduction Commands

```bash
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
cd /root/CodeBuddy/20260818191200/kernelswift
python3 skills/kernel-opt-loop/scripts/validate_decision.py kernels/track1-triton/centre_random_augmentation/bi150/rounds/decision_001.md --expected-profile triton_cuda
python3 -m py_compile kernels/track1-triton/centre_random_augmentation/bi150/triton_centre_random_augmentation_001.py
python3 auto_bench.py --v0_file kernels/track1-triton/centre_random_augmentation/base.py --v1_file kernels/track1-triton/centre_random_augmentation/bi150/triton_centre_random_augmentation_001.py --warmup 50 --repeat 100 --full-traceback
```

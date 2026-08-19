# Centre Random Augmentation Optimization Project

## Project Identity

- schema_version: 1
- skill_version: 2.0.0
- project_root: `/root/CodeBuddy/20260818191200/kernelswift`
- base: `../base.py`
- baseline_adapter: `baseline_adapter.py`
- harness: `/root/CodeBuddy/20260818191200/kernelswift/auto_bench.py`
- interpreter: `/usr/local/bin/python3`
- device: `cuda:0 (Iluvatar BI-V150)`
- implementation_language: `triton`
- implementation_backend: `cuda`
- target_profile: `triton_cuda`

## Semantics

- operator: `centre_random_augmentation` — a stochastic rigid-body transform used in
  diffusion sampling: center the input coordinates, then apply a random rotation
  (drawn from random unit quaternions) and a random Gaussian translation to
  produce `n_sample` augmented copies.

- inputs: two tensors, both float32 on the caller-selected accelerator:
  - `x_input_coords`: `[256, 3]` float32. `N_atom = 256` atom coordinate columns,
    each row `[x, y, z]`. Produced in `get_inputs()` as `torch.randn(N_ATOM, 3, device='cuda')`.
  - `mask`: `[256]` float32. In `get_inputs()` it is `torch.ones(N_ATOM, ...)`
    (all ones), so the `mask is not None` branch is always taken in the reference
    path. `forward` treats `mask` as optional and has a `mask is None` fallback,
    but the harness always supplies a non-None all-ones mask.
  `get_inputs()` returns the list `[x_input_coords, mask]`. `get_inputs()` calls
  `torch.manual_seed(42)` before generating the inputs. No integer or non-tensor
  inputs are present in `get_inputs()`; `get_init_inputs()` returns three scalars.

- outputs: a single float32 tensor `out: [4, 256, 3]` (`n_sample=4`, `N_atom=256`,
  3 spatial dims). `forward` returns exactly this one tensor (not a tuple).

- mathematical_behavior (reference, with `mask` non-None, `eps=1e-12`):
  - `device = x_input_coords.device`, `dtype = x_input_coords.dtype` (fp32).
  - Center (mask branch): `m = mask.to(dtype).unsqueeze(-1)` → `[256,1]`;
    `center = (x_input_coords * m).sum(dim=-2, keepdim=True) / (m.sum(dim=-2, keepdim=True) + eps)`
    → `[1,3]`. (The `mask is None` fallback uses `x_input_coords.mean(dim=-2, keepdim=True)`
    instead; not exercised by the harness.)
  - `x = x_input_coords - center` → `[256,3]`.
  - `x = x.unsqueeze(0).expand(n_sample, -1, -1).contiguous()` → `[4,256,3]`.
  - If `centre_only` were True, return `x` here. With `centre_only=False`
    (default harness value), continue.
  - `R = random_rotation_matrices(n_sample, device, dtype)` → `[4,3,3]`.
    `random_rotation_matrices` draws three uniform vectors `u1,u2,u3 = torch.rand(n)`
    and builds a unit quaternion via:
    `q1 = sqrt(1-u1)*sin(2*pi*u2)`, `q2 = sqrt(1-u1)*cos(2*pi*u2)`,
    `q3 = sqrt(u1)*sin(2*pi*u3)`, `q4 = sqrt(u1)*cos(2*pi*u3)`
    (quaternion `(x,y,z,w) = (q1,q2,q3,q4)`). This is the standard "Shoemake
    uniform-random quaternion" construction: `sqrt(1-u1)` / `sqrt(u1)` distribute
    the `w`-axis, and `u2`, `u3` are uniform azimuthal angles. The quaternion is
    then converted to a 3×3 rotation matrix via the standard formula, flattened
    row-major as 9 stacked elements and `reshape(n,3,3)`:
    row0 = `[1-2*(yy+zz), 2*(xy-wz), 2*(xz+wy)]`,
    row1 = `[2*(xy+wz), 1-2*(xx+zz), 2*(yz-wx)]`,
    row2 = `[2*(xz-wy), 2*(yz+wx), 1-2*(xx+yy)]`,
    where `xx=x*x`, `yy=y*y`, `zz=z*z`, `xy=x*y`, `xz=x*z`, `yz=y*z`,
    `wx=w*x`, `wy=w*y`, `wz=w*z`.
  - `T = s_trans * torch.randn(n_sample, 3)` → `[4,3]` (with `s_trans=1.0`).
  - `x = rot_vec_mul(R[:, None, :, :].expand(-1, x.shape[1], -1, -1), x) + T[:, None, :]`.
    `rot_vec_mul` applies the 3×3 rotation to each 3-vector by unbinding the last
    dim into `x,y,z` and summing `r[..., i, j] * t[..., j]` over `j`
    (i.e. `out[..., i] = sum_j R[..., i, j] * t[..., j]`). `R[:, None, :, :]`
    broadcasts `[4,3,3]` → `[4,1,3,3]`, expanded to `[4,256,3,3]`; result is
    `[4,256,3]`. Translation `T[:, None, :]` broadcasts `[4,3]` → `[4,1,3]` and
    is added to every atom row.
  - Masking (mask branch, always taken): `x = x * mask.to(dtype)[None, :, None]`
    (broadcast `[256]` → `[1,256,1]`). With the all-ones harness mask this is a
    no-op numerically.
  - Return `x` → `[4,256,3]`.

- dtype: all computation and the single output are float32. `mask` is cast to the
  input dtype (`mask.to(dtype=dtype)`) inside `forward`.

- randomness (critical): `random_rotation_matrices` draws `torch.rand` three times
  (`u1, u2, u3`) and the translation draws `torch.randn` once, all *inside*
  `forward` (i.e. at call time, not at init time). See the tolerance section for
  how the harness makes this comparable.

- tolerance_and_tie_rules: the harness (`auto_bench.py` `compare_values`) compares
  the candidate output against the reference with
  `torch.allclose(..., atol=1e-2, rtol=1e-2, equal_nan=True)` (harness defaults
  `--atol 1e-2 --rtol 1e-2`). The single output is a float tensor, so the
  `allclose` floating-point path applies; there are no integer outputs and no
  tie-break rule. Output shape, dtype, and device must match exactly.

  Randomness comparison contract (the key correctness constraint for a random
  operator): the harness does **not** fix a single shared random stream across
  v0/v1. Instead `run_forward` (harness) calls `set_seed(seed)` immediately before
  *each* forward call, then runs `model.forward(*cloned_inputs)` under
  `torch.no_grad()`. `set_seed` does `torch.manual_seed(seed)` plus
  `manual_seed_all(seed)` for every available accelerator. Consequently, v0
  (`Model`) and v1 (`ModelNew`) are each re-seeded to the same `seed` (default 42)
  right before their forward, so they will draw **identical** random quaternion
  uniforms (`u1,u2,u3`) and identical translation normals, producing identical
  `R` and `T` **provided the candidate consumes the RNG in the exact same order
  and count** (three `torch.rand` for the quaternion, one `torch.randn` for the
  translation). This is the critical constraint: a candidate must not reorder,
  add, remove, or change the distribution of random draws, otherwise its `R`/`T`
  diverge and the `allclose` comparison fails even with correct math. The
  reproducibility relies on identical RNG consumption, not on caching or freezing
  the rotation/translation values.

- public_contract: the candidate module must expose `ModelNew`, `get_init_inputs`,
  and `get_inputs`.
  `ModelNew.__init__(n_sample: int = 1, s_trans: float = 1.0, centre_only: bool = False)`
  and `forward(x_input_coords: torch.Tensor, mask: Optional[torch.Tensor] = None)
  -> torch.Tensor` must remain compatible. `forward` does not mutate its inputs
  and preserves the caller-selected device/current stream. With the harness's
  `get_init_inputs()` the effective config is `n_sample=4`, `s_trans=1.0`,
  `centre_only=False`. `get_init_inputs()` returns `[4, 1.0, False]` (n_sample,
  s_trans, centre_only). `get_inputs()` returns `[x_input_coords, mask]` with
  `x_input_coords = torch.randn(256, 3, dtype=torch.float32, device='cuda')` and
  `mask = torch.ones(256, dtype=torch.float32, device='cuda')`, both seeded by
  `torch.manual_seed(42)`.

Unknown user-owned semantics must be resolved with the user. Do not infer them
from a candidate implementation.

## Invariants

- Semantic invariant: `base.py` is user-owned and immutable; no role edits it.
  The public constructor (`ModelNew(n_sample=4, s_trans=1.0, centre_only=False)`),
  forward signature `(x_input_coords, mask) -> out`, output shape `[4,256,3]`,
  all-fp32 dtype, the centering formula (`sum / (sum + eps)` with `eps=1e-12`),
  the uniform-random-quaternion → 3×3 rotation-matrix construction, the Gaussian
  translation `s_trans * randn`, and the `rot_vec_mul` 3×3-by-3 vector product
  must remain numerically compatible with the reference. The stochastic behavior
  (three `torch.rand` draws for the quaternion plus one `torch.randn` draw for the
  translation, consumed in that exact order inside `forward`) must be preserved so
  the harness's identical per-call re-seeding yields bit-comparable `R`/`T`.
- Environment invariant: on the BI150 host a fresh shell must set
  `export COREX_VERSION=4.4.0` and source `/usr/local/corex/enable` before
  importing `torch` or `triton`; without that bootstrap, imports and `ixsmi` fail.
  The Triton active compiler backend is `cuda` on the CoreX environment.
- Lifecycle invariant: candidate execution preserves caller-selected device and
  current stream; `forward` does not mutate inputs; any output-buffer reuse must
  have explicit per-instance ownership, compatibility keys including
  shape/dtype/device, invalidation, aliasing, and concurrency semantics.
- Measurement invariant: the harness re-seeds both sides identically (default
  `--seed 42`) immediately before each forward call, clones inputs, replaces
  candidate inputs with a clone of the reference inputs, runs under
  `torch.no_grad()`, and compares the candidate output against the reference with
  `atol=1e-2, rtol=1e-2, equal_nan=True`. Benchmark wall time (unrounded median)
  controls adoption. A change to the measurement fingerprint requires a new
  comparable baseline.

The complete workflow-level rules are in `references/invariants.md`.

## Runtime Fingerprint

```yaml
triton_distribution: corex
triton_version: 3.1.0
backend_target: cuda
backend_version: 2.7.1
device_arch: cuda:0 (Iluvatar BI-V150), capability (7,1), 16 SM, 16 GiB
```

- target_profile_match: `pass`
- discovery_commands: `export COREX_VERSION=4.4.0; . /usr/local/corex/enable; python3 -c "import torch,triton; print(torch.__version__, triton.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"`
- discovered_at: `2026-08-19T12:30:00Z`

These values are observed in Phase 0. They are not assumed from the profile.

## Measurement Regime

- harness_path: `/root/CodeBuddy/20260818191200/kernelswift/auto_bench.py`
- harness_sha256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- shape: `x_input_coords[256,3] fp32; mask[256] fp32; output [4,256,3] fp32`
- dtype: `fp32 (x_input_coords, mask, out)`
- device: `cuda:0 (Iluvatar BI-V150)`
- warmup: `50`
- repeat: `100`
- timing_order: `interleaved accepted-reference/candidate`
- primary_metric: `unrounded median wall_time_ms`
- profiler_iterations: `50`
- profiler_scopes: `accepted_reference,candidate`
- correctness_command: `python3 auto_bench.py --v0_file kernels/track1-triton/centre_random_augmentation/base.py --v1_file kernels/track1-triton/centre_random_augmentation/bi150/baseline_adapter.py --warmup 50 --repeat 100`
- benchmark_command: `python3 auto_bench.py --v0_file kernels/track1-triton/centre_random_augmentation/base.py --v1_file kernels/track1-triton/centre_random_augmentation/bi150/baseline_adapter.py --warmup 50 --repeat 100`
- profiler_command: `python3 auto_bench.py --v0_file kernels/track1-triton/centre_random_augmentation/base.py --v1_file kernels/track1-triton/centre_random_augmentation/bi150/baseline_adapter.py --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50`

Benchmark wall time controls adoption. Profiler data is attributable diagnostic
evidence and is normalized per forward call.

## Measurement Fingerprint

- measurement_fingerprint: `a5f980780c4dcde731df913710ad9dfded4f07a66b90e334fea0a6f2aa1fd5fa`
- base_sha256: `02e7020fef34db401cfc2cf8031700262493ab96a89449d4e70d333978e78553`
- baseline_adapter_sha256: `012754740961f6ec10d515563e51cd07eeaf35caefe33731d5c1e9a88387fe9b`
- fingerprint_command: `sha256sum(base.py); sha256sum(baseline_adapter.py); python3 -c "import hashlib,json; print(hashlib.sha256(open("base.py","rb").read()+b" "+open("auto_bench.py","rb").read()+b" "+json.dumps(settings,sort_keys=True,separators=(",",":")).encode()).hexdigest())"`

A fingerprint change requires a new comparable baseline before optimization can
continue.

## Optional Target

- target_mode: `null`
- target_value: `null`
- target_metric: `wall_time_ms`
- source: user
- target_measurement_fingerprint: `null`

An optional target is comparable only when it uses `wall_time_ms` under the
baseline measurement fingerprint. It is not an estimated bound, device-time
goal, or inferred target. Orchestrator records later target amendments in the
append-only team-state policy-revision table at a safe terminal boundary.

## Git Run Identity

- base_branch: `dev`
- base_commit: `e8533192f65ed4610a4b59859f1969ea83955f87`
- run_branch: `kernel-opt/centre_random_augmentation-bi150-20260818`

These fields mirror `team-state.md` and identify the dedicated optimization
branch. The run branch is never `main`, `master`, or `dev`.

## Round Overview

| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 000 | Phase 0 | `baseline_adapter.py` | baseline | `base.py` | 1.073250 | 420.684 | - | not-applicable | `baseline_adapter.py` |
| 001 | `rounds/decision_001.md` | `triton_centre_random_augmentation_001.py` | accepted | `baseline_adapter.py` | 0.712600 | 237.95 | 30.35390886976103 | confirmed | `triton_centre_random_augmentation_001.py` |
| 002 | `rounds/decision_002.md` | `triton_centre_random_augmentation_002.py` | accepted | `triton_centre_random_augmentation_001.py` | 0.239284 | 29.24 | 66.37353809388355 | confirmed | `triton_centre_random_augmentation_002.py` |
| 003 | `rounds/decision_003.md` | - | aborted | - | - | - | - | not-applicable | `triton_centre_random_augmentation_002.py` |

Orchestrator appends one row only after a terminal round transition is validated
and committed. Rejected candidates remain listed but never become the comparison
source.

## Reproduction

```bash
<baseline correctness and benchmark command>
```

```bash
<separately scoped accepted-reference/candidate profiler command>
```

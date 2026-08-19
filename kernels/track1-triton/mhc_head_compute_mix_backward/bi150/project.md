# MHC Head Compute Mix Backward Optimization Project

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

- operator: `mhc_head_compute_mix_backward` — the manual backward pass of
  `mhc_head_compute_mix`, restricted to the sigmoid-gated affine modulation
  `z = input_mix * mhc_scale + mhc_base` followed by `torch.sigmoid(z)`. Given
  the upstream gradient `grad_out`, it returns the gradients with respect to
  `input_mix`, `mhc_scale`, and `mhc_base`. It does NOT contain the Sinkhorn
  normalization or the `pre`/`post`/`comb` head structure; it is a pure
  sigmoid-backward over a single affine modulation.

- inputs: four tensors, all float32 on the caller-selected accelerator:
  - `input_mix`: `[2, 1024, 4]` float32 (`batch0=2`, `batch1=1024`,
    `mhc_mult=4`).
  - `mhc_scale`: `[1]` float32.
  - `mhc_base`: `[4]` float32 (`mhc_mult=4`).
  - `grad_out`: `[2, 1024, 4]` float32 (same shape as `input_mix`).
  `get_inputs()` returns the list `[input_mix, mhc_scale, mhc_base, grad_out]`
  with each tensor created via `torch.randn(..., dtype=torch.float32,
  device="cuda")`. No integer or non-tensor inputs are present.

- outputs: a 3-tuple of float32 tensors returned by `forward`:
  - `grad_input_mix`: `[2, 1024, 4]` float32.
  - `grad_mhc_scale`: `[1]` float32.
  - `grad_mhc_base`: `[4]` float32.

- mathematical_behavior: `batch0=2`, `batch1=1024`, `mhc_mult=4`. `forward`
  computes:
  - `z = input_mix * mhc_scale + mhc_base` — `mhc_scale` (`[1]`) and
    `mhc_base` (`[4]`) broadcast against `input_mix`'s last dim to shape
    `[2, 1024, 4]`.
  - `sigmoid = torch.sigmoid(z)`.
  - `grad_z = grad_out * sigmoid * (1 - sigmoid)` — sigmoid backward:
    `dσ/dz = σ(1-σ)`, so `grad_z = grad_out * dσ/dz`.
  - `grad_input_mix = grad_z * mhc_scale` — `mhc_scale` (`[1]`) broadcasts to
    `[2, 1024, 4]`.
  - `grad_mhc_base = grad_z.sum(dim=(0, 1), keepdim=True).view(-1)` → `[4]`.
    Sums over the first two dims (`batch0` and `batch1`), keeping the last dim
    (`mhc_mult`), then flattens to `[4]`. Because `mhc_base` broadcasts over the
    batch dims, its gradient is the batch-sum of `grad_z`.
  - `grad_mhc_scale = (grad_z * input_mix).sum(dim=(0, 1, 2), keepdim=True).view(1)`
    → `[1]`. Full reduction over all three dims, then flattened to `[1]`.
    Because `mhc_scale` is a scalar multiplier of `input_mix`, its gradient is
    `sum(grad_z * input_mix)`.
  - Return order: `(grad_input_mix, grad_mhc_scale, grad_mhc_base)`.

- dtype: all computation and all outputs are float32. Inputs are created as
  `torch.float32` in `get_inputs()`; no cast is performed inside `forward`.

- tolerance_and_tie_rules: the harness compares candidate output against the
  reference recursively (tuple → tensors) with
  `torch.allclose(..., atol=1e-2, rtol=1e-2, equal_nan=True)` (harness defaults
  `--atol 1e-2 --rtol 1e-2`). All three outputs are floating point, so the
  allclose path applies; there are no integer outputs and no tie-break rule.
  Output structure, shapes, and dtype must match exactly.

- public_contract: the candidate module must expose `ModelNew`,
  `get_init_inputs`, and `get_inputs`. `ModelNew.__init__()` takes no
  arguments; `get_init_inputs()` returns `[]`.
  `forward(input_mix, mhc_scale, mhc_base, grad_out) -> (grad_input_mix,
  grad_mhc_scale, grad_mhc_base)` must remain compatible. `forward` does not
  mutate its inputs and preserves the caller-selected device/current stream.
  `get_inputs()` returns `[input_mix, mhc_scale, mhc_base, grad_out]` with
  `input_mix = torch.randn(2, 1024, 4, dtype=torch.float32, device="cuda")`,
  `mhc_scale = torch.randn(1, dtype=torch.float32, device="cuda")`,
  `mhc_base = torch.randn(4, dtype=torch.float32, device="cuda")`,
  `grad_out = torch.randn(2, 1024, 4, dtype=torch.float32, device="cuda")`.

Unknown user-owned semantics must be resolved with the user. Do not infer them
from a candidate implementation.

## Invariants

- Semantic invariant: `base.py` is user-owned and immutable; no role edits it.
  The public constructor (`ModelNew()` with no arguments), forward signature
  `(input_mix, mhc_scale, mhc_base, grad_out) -> (grad_input_mix,
  grad_mhc_scale, grad_mhc_base)`, output shapes `grad_input_mix [2,1024,4]`,
  `grad_mhc_scale [1]`, `grad_mhc_base [4]`, all-fp32 dtype, the sigmoid
  backward `grad_z = grad_out * σ(z) * (1 - σ(z))`, and the two reduction
  contracts (`grad_mhc_base = grad_z.sum(dim=(0,1))`, `grad_mhc_scale =
  (grad_z * input_mix).sum(dim=(0,1,2))`) must remain numerically compatible
  with the reference.
- Environment invariant: on the BI150 host a fresh shell must set
  `export COREX_VERSION=4.4.0` and source `/usr/local/corex/enable` before
  importing `torch` or `triton`; without that bootstrap, imports and `ixsmi`
  fail. The Triton active compiler backend is `cuda` on the CoreX environment.
- Lifecycle invariant: candidate execution preserves caller-selected device and
  current stream; `forward` does not mutate inputs; any output-buffer reuse must
  have explicit per-instance ownership, compatibility keys including
  shape/dtype/device, invalidation, aliasing, and concurrency semantics.
- Measurement invariant: the harness seeds each side identically, clones
  inputs, replaces candidate inputs with a clone of the reference inputs, runs
  under `torch.no_grad()`, and compares candidate outputs recursively against
  the reference with `atol=1e-2, rtol=1e-2, equal_nan=True`. Benchmark wall time
  (unrounded median) controls adoption. A change to the measurement fingerprint
  requires a new comparable baseline.

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
- discovered_at: `2026-08-19T15:45:00Z`

These values are observed in Phase 0. They are not assumed from the profile.

## Measurement Regime

- harness_path: `<absolute-harness-path>`
- harness_sha256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- shape: `input_mix[2,1024,4] fp32; mhc_scale[1] fp32; mhc_base[4] fp32; grad_out[2,1024,4] fp32; output (grad_input_mix[2,1024,4], grad_mhc_scale[1], grad_mhc_base[4]) fp32`
- dtype: `fp32 (all)`
- device: `cuda:0 (Iluvatar BI-V150)`
- warmup: `50`
- repeat: `100`
- timing_order: `interleaved accepted-reference/candidate`
- primary_metric: `unrounded median wall_time_ms`
- profiler_iterations: `50`
- profiler_scopes: `accepted_reference,candidate`
- correctness_command: `python3 auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix_backward/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix_backward/bi150/baseline_adapter.py --warmup 50 --repeat 100`
- benchmark_command: `python3 auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix_backward/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix_backward/bi150/baseline_adapter.py --warmup 50 --repeat 100`
- profiler_command: `python3 auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix_backward/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix_backward/bi150/baseline_adapter.py --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50`

Benchmark wall time controls adoption. Profiler data is attributable diagnostic
evidence and is normalized per forward call.

## Measurement Fingerprint

- measurement_fingerprint: `a03823074048c8cb5e8199b593c8c19aa3b259180969321015e5a1679461b71a`
- base_sha256: `28d4d213638e5adc7f6bc52928c8f7596cd8446e3e0201beab364b5c3c7988fc`
- baseline_adapter_sha256: `98cf1e1e5f493ae4ae08c9391f00db83b3cd783a1b194d9c0d63851091133a5d`
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
- run_branch: `kernel-opt/mhc_head_compute_mix_backward-bi150-20260818`

These fields mirror `team-state.md` and identify the dedicated optimization
branch. The run branch is never `main`, `master`, or `dev`.

## Round Overview

| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 000 | Phase 0 | `baseline_adapter.py` | baseline | `base.py` | 0.351449 | 185.599 | - | not-applicable | `baseline_adapter.py` |
| 001 | `rounds/decision_001.md` | `triton_mhc_head_compute_mix_backward_001.py` | accepted | `baseline_adapter.py` | 0.198597 | 14.692 | 43.11572468785218 | confirmed | `triton_mhc_head_compute_mix_backward_001.py` |
| 002 | `rounds/decision_002.md` | - | aborted | - | - | - | - | not-applicable | `triton_mhc_head_compute_mix_backward_001.py` |

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

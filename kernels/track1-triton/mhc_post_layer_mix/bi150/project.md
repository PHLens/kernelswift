# mhc_post_layer_mix Optimization Project

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

- operator: `mhc_post_layer_mix` — a multi-head-combined post-layer mix. It
  combines a per-head gated input `x` with a combiner-matrix-mixed residual
  `residual` to produce a mixed head output. The core is a `torch.einsum` matrix
  multiply that contracts over the head-count multiplier dimension.

- inputs: Four tensors, all contiguous on the caller-selected accelerator
  (`device="cuda"`), produced by `generate_mhc_post_test_data` and returned in
  order by `get_inputs()`:
  - `x`: `[2, 4096, 1280]` `torch.bfloat16`
  - `residual`: `[2, 4096, 4, 1280]` `torch.bfloat16`
  - `post_layer_mix`: `[2, 4096, 4, 1]` `torch.float32`
  - `comb_res_mix`: `[2, 4096, 4, 4]` `torch.float32`
  `get_inputs()` returns `[x, residual, post_layer_mix, comb_res_mix]` (the
  internal `o_grad` tensor is generated but not returned).

- outputs: A single tensor `out` of shape `[2, 4096, 4, 1280]`,
  `torch.bfloat16`, contiguous, on the input accelerator.

- mathematical_behavior:
  1. `residual` and `x` are promoted from `bfloat16` to `float32` via
     `.float()`; `post_layer_mix` and `comb_res_mix` are already `float32`.
  2. `term2 = torch.einsum('abmn,abmc->abnc', comb_res_mix, residual.float())`.
     With `comb_res_mix` of shape `[2, 4096, 4, 4]` (indices `a=2, b=4096,
     m=4, n=4`) and `residual.float()` of shape `[2, 4096, 4, 1280]` (indices
     `a=2, b=4096, m=4, c=1280`), the output `'abnc'` is `[2, 4096, 4, 1280]`
     (`a=2, b=4096, n=4, c=1280`). The `m` index (size 4) is contracted: for
     each `(a, b)` the `[4, 4]` matrix `comb_res_mix[a,b]` left-multiplies the
     `[4, 1280]` matrix `residual.float()[a,b]`, i.e.
     `term2[a,b,n,c] = sum_m comb_res_mix[a,b,n,m] * residual.float()[a,b,m,c]`.
  3. `out_f32 = x.float().unsqueeze(-2) * post_layer_mix + term2`. Here
     `x.float().unsqueeze(-2)` is `[2, 4096, 1, 1280]` (a new head axis of
     size 1), broadcasting against `post_layer_mix` `[2, 4096, 4, 1]` to
     `[2, 4096, 4, 1280]`, then added to `term2` `[2, 4096, 4, 1280]`.
  4. `out = out_f32.bfloat16()` casts the fp32 result back to `torch.bfloat16`.
  All arithmetic (einsum contraction and the elementwise multiply-add) is
  performed in float32.

- tolerance_and_tie_rules: The harness compares the single output tensor
  recursively with `torch.allclose(..., atol=1e-2, rtol=1e-2, equal_nan=True)`.
  The output is floating point (`bfloat16`); there are no integer outputs and
  no tie-break rule.

- public_contract: The candidate module must expose `ModelNew`,
  `get_init_inputs`, and `get_inputs`. `ModelNew.__init__()` takes no
  arguments (`get_init_inputs()` returns `[]`). `forward(x, residual,
  post_layer_mix, comb_res_mix) -> out` must remain compatible, where the four
  positional arguments are the inputs listed above and `out` is the single
  `[2, 4096, 4, 1280]` `bfloat16` tensor. `forward` does not mutate its inputs
  and preserves the caller-selected device/current stream. `get_inputs()`
  returns `[x, residual, post_layer_mix, comb_res_mix]`.

Unknown user-owned semantics must be resolved with the user. Do not infer them
from a candidate implementation.

## Invariants

- `base.py` and `auto_bench.py` are immutable after Phase 0 begins. The
  `baseline_adapter.py` bytes must not overwrite the `base.py` content; adapter
  generation renames the single top-level `Model` class to `ModelNew`.
- Candidate code uses the CUDA-compatible PyTorch surface (the Triton active
  compiler backend is `cuda` on the BI150 / CoreX environment).
- The harness AST loader (`_filter_module_ast`) retains imports,
  class/function definitions, and literal top-level assignments while
  discarding other top-level statements; loaded candidate code must still
  expose `ModelNew`, `get_inputs`, and `get_init_inputs`. The loader rewrites
  the `'npu'` device string to the detected accelerator (no-op on the `cuda`
  BI150 path).
- The environment invariant: on the BI150 host a fresh shell must set
  `export COREX_VERSION=4.4.0` and source `/usr/local/corex/enable` before
  importing `torch` or `triton`; without that bootstrap, imports fail.
- The lifecycle invariant: candidate execution preserves caller-selected device
  and current stream; `forward` does not mutate inputs; any output-buffer reuse
  must have explicit per-instance ownership, compatibility keys including
  shape/dtype/device, invalidation, aliasing, and concurrency semantics.
- The measurement invariant: the harness seeds each side identically, clones
  inputs, replaces candidate inputs with a clone of the reference inputs, runs
  under `torch.no_grad()`, and compares candidate outputs recursively against
  the reference with `atol=1e-2, rtol=1e-2, equal_nan=True`. Benchmark wall
  time (unrounded median) controls adoption.

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
- discovery_commands: `export COREX_VERSION=4.4.0; . /usr/local/corex/enable; python3 -c 'import torch,triton; print(torch.__version__, triton.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))'`
- discovered_at: `2026-08-18T17:20:00Z`

These values are observed in Phase 0. They are not assumed from the profile.

## Measurement Regime

- harness_path: `/root/CodeBuddy/20260818191200/kernelswift/auto_bench.py`
- harness_sha256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- shape: `x [2,4096,1280] bf16; residual [2,4096,4,1280] bf16; post_layer_mix [2,4096,4,1] fp32; comb_res_mix [2,4096,4,4] fp32; output [2,4096,4,1280] bf16`
- dtype: `bf16 (x,residual,out); fp32 (post_layer_mix,comb_res_mix)`
- device: `cuda:0 (Iluvatar BI-V150)`
- warmup: `50`
- repeat: `100`
- timing_order: `interleaved accepted-reference/candidate`
- primary_metric: `unrounded median wall_time_ms`
- profiler_iterations: `50`
- profiler_scopes: `accepted_reference,candidate`
- correctness_command: `python3 auto_bench.py --v0_file kernels/track1-triton/mhc_post_layer_mix/base.py --v1_file kernels/track1-triton/mhc_post_layer_mix/bi150/baseline_adapter.py --warmup 50 --repeat 100`
- benchmark_command: `python3 auto_bench.py --v0_file kernels/track1-triton/mhc_post_layer_mix/base.py --v1_file kernels/track1-triton/mhc_post_layer_mix/bi150/baseline_adapter.py --warmup 50 --repeat 100`
- profiler_command: `python3 auto_bench.py --v0_file kernels/track1-triton/mhc_post_layer_mix/base.py --v1_file kernels/track1-triton/mhc_post_layer_mix/bi150/baseline_adapter.py --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50`

Benchmark wall time controls adoption. Profiler data is attributable diagnostic
evidence and is normalized per forward call.

## Measurement Fingerprint

- measurement_fingerprint: `c17c7f45d44b1bec047c7c3a315275d33b21af3226dd34e884d483899ef039b6`
- base_sha256: `e392799f72edcf7210373ac8f2522bcf3d4f740c8f9dd0d1f117bc627939ebf3`
- baseline_adapter_sha256: `66a3a2c31863d18c725a52ab57fd1b9f89fe655dd7bab7cb4da158b8130b5d07`
- fingerprint_command: `sha256sum(base.py); sha256sum(baseline_adapter.py); python3 -c "import hashlib,json; print(hashlib.sha256(open('base.py','rb').read()+b'\0'+open('auto_bench.py','rb').read()+b'\0'+json.dumps(settings,sort_keys=True,separators=(',',':')).encode()).hexdigest())"`

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
- run_branch: `kernel-opt/mhc_post_layer_mix-bi150-20260818`

These fields mirror `team-state.md` and identify the dedicated optimization
branch. The run branch is never `main`, `master`, or `dev`.

## Round Overview

| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 000 | Phase 0 | `baseline_adapter.py` | baseline | `base.py` | 8.189047 | 7323.8471875 | - | not-applicable | `baseline_adapter.py` |
| 001 | `rounds/decision_001.md` | `triton_mhc_post_layer_mix_001.py` | accepted | `baseline_adapter.py` | 6.427432 | 6122.542 | 20.09 | confirmed | `triton_mhc_post_layer_mix_001.py` |
| 002 | `rounds/decision_002.md` | - | aborted | - | - | - | - | not-applicable | `triton_mhc_post_layer_mix_001.py` |

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

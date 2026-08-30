# MhcHeadComputeMix Optimization Project

## Project Identity

- schema_version: 1
- skill_version: 2.0.0
- project_root: `/workspace/kernelswift/.worktrees/mhc-head-compute-mix-ascend/kernels/track1-triton/mhc_head_compute_mix/ascend`
- base: `../base.py` (shared reference; no torch_mlu dependency)
- baseline_adapter: `baseline_adapter.py`
- harness: `/workspace/kernelswift/.worktrees/mhc-head-compute-mix-ascend/auto_bench.py`
- interpreter: `/usr/local/python3.11.15/bin/python3`
- device: `npu:0`
- implementation_language: `triton`
- implementation_backend: `ascend`
- target_profile: `triton_ascend`

## Semantics

- operator: `MhcHeadComputeMix` — a batched head-mixing operator that (a) applies two sigmoid gates, and (b) builds a 4x4 combination matrix and iteratively normalizes it to a doubly-stochastic matrix via Sinkhorn row/column scaling.
- inputs (all fp32, all on `npu:0` via the harness device rewrite):
  - `mixes`: `Tensor[2,8,24]` — batch `b=2`, sequence `s=8`, flattened feature dim `mix_hc=(2+hc)*hc=24` with `hc=4`.
  - `hc_scale`: `Tensor[3]` — three scalar gate/scale factors `(s0, s1, s2)`.
  - `hc_base`: `Tensor[24]` — bias vector, consumed in three contiguous segments of length `hc`, `hc`, and `hc*hc`.
- outputs: a tuple `(pre, post, comb)`, all fp32, all on the caller device:
  - `pre`: `Tensor[2,8,4]`
  - `post`: `Tensor[2,8,4]`
  - `comb`: `Tensor[2,8,4,4]`
- mathematical_behavior (exact, in forward order):
  - Flatten `x = mixes.reshape(-1, 24)` (so `x` is `[16, 24]`, one row per `(b,s)`).
  - `pre = sigmoid(x[:, :4] * s0 + hc_base[:4]) + eps` — note the additive `+ eps` is applied only to `pre`.
  - `post = 2 * sigmoid(x[:, 4:8] * s1 + hc_base[4:8])` — note `post` has **no** `+ eps`, and has the factor 2.
  - `raw = x[:, 8:24]` reshaped to `[16,4,4]`; `comb = raw * s2 + hc_base[8:24].view(1,4,4)`.
  - Row-stabilized softmax over the last dim: `row_max = comb.amax(-1, keepdim=True)`; `comb = exp(comb - row_max)`; `comb = comb / comb.sum(-1, keepdim=True) + eps`.
  - First column normalize: `comb = comb / (comb.sum(-2, keepdim=True) + eps)`.
  - Sinkhorn loop for `sinkhorn_iters - 1 = 19` iterations, each doing row then column normalization: `comb = comb / (comb.sum(-1, keepdim=True) + eps)`; `comb = comb / (comb.sum(-2, keepdim=True) + eps)`. Together with the first column normalize this yields 20 row and 20 column normalizations total.
  - Reshape outputs back: `pre.view(2,8,4)`, `post.view(2,8,4)`, `comb.view(2,8,4,4)`.
- numerical_details:
  - `eps = 1e-6` is added to `pre` (output value) and to Sinkhorn denominators; it is **not** added to `post` and **not** inside the `exp` argument.
  - The softmax subtracts `row_max` per row for numerical stability before `exp`.
  - All computation is fp32; `mixes` and `hc_base` are explicitly cast `.to(torch.float32)` inside forward.
- validation_behavior: forward raises `ValueError` if `mixes.shape[-1] != (2 + hc) * hc` where `hc = hc_mult`.
- tolerance_and_tie_rules: base.py declares no tolerance; harness compares outputs with `atol=1e-2`, `rtol=1e-2` (harness defaults), `equal_nan=True`.
- public_contract:
  - constructor `ModelNew(hc_mult=4, sinkhorn_iters=20, eps=1e-6)`;
  - `forward(mixes, hc_scale, hc_base) -> (pre, post, comb)`.
  - Harness also requires module-level `get_init_inputs()` returning `[4, 20, 1e-6]` and `get_inputs()` returning `[mixes, hc_scale, hc_base]`.

## Invariants

- `base.py` (shared `../base.py`) is the immutable reference; its bytes must remain unchanged after `baseline_adapter.py` generation and throughout the run.
- Public constructor and forward signature, output structure, dtype (fp32), shapes, and the exact numerical semantics above are preserved by every candidate.
- Candidate output is a tuple `(pre[2,8,4], post[2,8,4], comb[2,8,4,4])`, all fp32, on the caller-selected device.
- The `+eps` asymmetry (`pre` gets it, `post` does not) and the factor 2 on `post` are semantic, not incidental — candidates must reproduce them.
- Harness AST loader (`auto_bench.py`) rewrites bare device string literals `"cuda"` -> detected accelerator (`npu`), so `get_inputs()` may write `device="cuda"` and still run on `npu:0`. The loader also filters the module to top-level Import/ImportFrom/ClassDef/FunctionDef/AsyncFunctionDef and literal assignments, so candidates must define `ModelNew`, `get_inputs`, and `get_init_inputs` as top-level definitions only.
- Benchmark wall time is measured by the unchanged harness with `warmup=50`, `repeat=100` (median), interleaved reference/candidate ordering; profiler uses separate reference/candidate scopes.
- Measurement fingerprint binds base bytes, harness bytes, shape, dtype, device, warmup/repeat, and profiler settings; any change requires a new comparable baseline.
- Lifecycle/device/stream invariants: candidates preserve the caller-selected device and current stream; no implicit global caches or shared cross-instance state; buffer reuse (if introduced in a later round) must key on shape/dtype/device.

## Runtime Fingerprint

```yaml
triton_distribution: triton
triton_version: 3.2.0
torch_version: 2.7.1+cpu
torch_npu_version: 2.7.1.post4
backend_target: triton_ascend
device_name: Ascend910B4
cube_core_num: 20
vector_core_num: 40
total_memory_bytes: 31662800896
L2_cache_size: 100663296
```

- target_profile_match: `pass`
- host: `ascend910b4`

## Measurement Regime

- harness_path: `/workspace/kernelswift/.worktrees/mhc-head-compute-mix-ascend/auto_bench.py`
- harness_sha256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- shape: `mixes=[2,8,24] fp32; hc_scale=[3] fp32; hc_base=[24] fp32`
- dtype: `fp32`
- device: `npu:0`
- warmup: `50`
- repeat: `100`
- primary_metric: `unrounded median wall_time_ms`
- profiler_iterations: `50`

## Measurement Fingerprint

- measurement_fingerprint: `52025b1bb12ac09c6a26db2a94fd681e9ac9b325db572734a4af3689a43c38ed`
- base_sha256: `4c5167f6cfe9099786f204fad222f3b780d3ebb814494b9582112422abf84ac5`
- baseline_adapter_sha256: `c3cc90deb93c69f092e876ddc0057ddf6c2d7ac28020f1b7fd1ed260bca72fee`

## Optional Target

- target_mode: `null`
- target_value: `null`
- target_metric: `wall_time_ms`
- source: user

## Git Run Identity

- base_branch: `dev`
- base_commit: `3337f08`
- run_branch: `kernel-opt/mhc-head-compute-mix-ascend`

## Round Overview

| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 000 | Phase 0 | `baseline_adapter.py` | baseline | `base.py` | 3.396440 | 280.60 | - | not-applicable | `baseline_adapter.py` |
| 001 | sinkhorn-loop-fusion | `candidate_001.py` | accepted | `baseline_adapter.py` | 3.526815 | 8.784 | +88.88% | confirmed | `candidate_001.py` |
| 002 | no-change (abort) | - | aborted | `candidate_001.py` | - | - | - | - | `candidate_001.py` |

## Reproduction

```bash
cd /workspace/kernelswift/.worktrees/mhc-head-compute-mix-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix/ascend/baseline_adapter.py --warmup 50 --repeat 100
```

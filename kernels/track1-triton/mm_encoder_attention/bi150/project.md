# MM Encoder Attention Optimization Project

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

- operator: `mm_encoder_attention` — a standard scaled dot-product (self)
  attention over a single query/key/value triple, implemented as
  `F.scaled_dot_product_attention` with no mask, no dropout, and no causal
  flag. It is a pure encoder-style attention block: a single input tensor's
  head-split `q`, `k`, `v` are attended together and the concatenated head
  outputs are returned.
- inputs: `query`, `key`, `value` are three contiguous `[2, 83, 512]` float16
  tensors on the caller-selected accelerator, produced by
  `torch.randn(2, 83, 512, dtype=torch.float16, device="cuda")` in
  `get_inputs()`. `512 = num_heads * head_size = 8 * 64`, so each tensor holds
  the concatenated 8 heads of size 64. `get_inputs()` returns the list
  `[query, key, value]`. No integer or non-tensor inputs are present.
- outputs: A single tensor `out` of shape `[2, 83, 512]`, float16, contiguous,
  on the input accelerator. `forward` returns this tensor directly (not a
  tuple/list): the attended result is transposed from `[B, H, S, D]` back to
  `[B, S, H, D]` and reshaped to `[B, S, H*D] = [2, 83, 512]`.
- mathematical_behavior: Given `B=2`, `S=83` (both `q_len` and `kv_len`),
  `num_heads=8`, `head_size=D=64`, `num_kv_heads=8`, and
  `hidden = num_heads * head_size = 512`, `forward` reshapes each input to
  `[B, S, num_heads, head_size]` and transposes to `[B, num_heads, S, D]`
  (`q`, `k`, `v`). It then computes
  `out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)` with
  `self.scale = 1.0 / head_size**0.5 = 1/sqrt(64) = 0.125`. There is no
  `attn_mask`, no `dropout_p`, and no `is_causal` argument, so full (unmasked)
  attention over the full 83-length sequence is computed. Because
  `num_heads == num_kv_heads == 8`, this is plain multi-head self-attention
  with no grouped-query attention (GQA) and no multi-query reduction.
  `q_len == kv_len == 83` makes it self-attention. The result is transposed
  back to `[B, S, H, D]` and reshaped to `[B, S, 512]`.
- tolerance_and_tie_rules: The harness requires identical output structure
  (a single tensor, not a tuple), identical shape `[2, 83, 512]`, and
  identical float16 dtype. The output is floating point and is compared with
  `torch.allclose(..., atol=1e-2, rtol=1e-2, equal_nan=True)` (the harness
  defaults `--atol 1e-2 --rtol 1e-2`). There are no integer outputs and no
  tie-break rule.
- public_contract: The candidate module must expose `ModelNew`,
  `get_init_inputs`, and `get_inputs`.
  `ModelNew.__init__(num_heads: int = 8, head_size: int = 64,
  num_kv_heads: int = 8)` and
  `forward(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) ->
  torch.Tensor` must remain compatible. `forward` does not mutate its inputs
  and preserves the caller-selected device/current stream.
  `get_init_inputs()` returns `[8, 64, 8]` (num_heads, head_size, num_kv_heads).
  `get_inputs()` returns `[query, key, value]`, each `[2, 83, 512]` fp16 on
  `device="cuda"`.

Unknown user-owned semantics must be resolved with the user. Do not infer them
from a candidate implementation.

## Invariants

- `base.py` and `auto_bench.py` are immutable after Phase 0 begins. The
  `baseline_adapter.py` bytes must not be overwritten by the `base.py` content;
  its generation must leave `base.py` unchanged.
- Candidate code uses the CUDA-compatible PyTorch surface (the Triton active
  compiler backend is `cuda` on the BI150 / CoreX environment).
- The harness AST loader (`auto_bench.py` `_filter_module_ast`) retains
  `Import`/`ImportFrom`, class/function definitions, and literal top-level
  assignments while discarding other top-level statements; loaded candidate
  code must still expose `ModelNew`, `get_inputs`, and `get_init_inputs`.
- The environment invariant: on the BI150 host a fresh shell must set
  `export COREX_VERSION=4.4.0` and source `/usr/local/corex/enable` before
  importing `torch` or `triton`; without that bootstrap, imports and `ixsmi`
  fail.
- The lifecycle invariant: candidate execution preserves caller-selected device
  and current stream; `forward` does not mutate inputs; any output-buffer reuse
  must have explicit per-instance ownership, compatibility keys including
  shape/dtype/device, invalidation, aliasing, and concurrency semantics.
- The measurement invariant: the harness seeds each side identically, clones
  inputs, replaces candidate inputs with a clone of the reference inputs, runs
  under `torch.no_grad()`, and compares candidate outputs recursively against
  the reference with `atol=1e-2, rtol=1e-2, equal_nan=True`. Benchmark wall time
  (unrounded median) controls adoption.

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
- discovered_at: `2026-08-18T15:55:00Z`

These values are observed in Phase 0. They are not assumed from the profile.

## Measurement Regime

- harness_path: `/root/CodeBuddy/20260818191200/kernelswift/auto_bench.py`
- harness_sha256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- shape: `query/key/value [2,83,512] fp16; output [2,83,512] fp16`
- dtype: `fp16 (query, key, value, out)`
- device: `cuda:0 (Iluvatar BI-V150)`
- warmup: `50`
- repeat: `100`
- timing_order: `interleaved accepted-reference/candidate`
- primary_metric: `unrounded median wall_time_ms`
- profiler_iterations: `50`
- profiler_scopes: `accepted_reference,candidate`
- correctness_command: `python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/bi150/baseline_adapter.py --warmup 50 --repeat 100`
- benchmark_command: `python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/bi150/baseline_adapter.py --warmup 50 --repeat 100`
- profiler_command: `python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/bi150/baseline_adapter.py --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50`

Benchmark wall time controls adoption. Profiler data is attributable diagnostic
evidence and is normalized per forward call.

## Measurement Fingerprint

- measurement_fingerprint: `b8029499f0964a738f50b09164e419511d0bc89df5e260573e607bb7345afc2e`
- base_sha256: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2`
- baseline_adapter_sha256: `c3980a2c4ae79171c73c7cf85c29c62f6ac98b9e9fb643c249fc9a6e0a921c9f`
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
- run_branch: `kernel-opt/mm_encoder_attention-bi150-20260818`

These fields mirror `team-state.md` and identify the dedicated optimization
branch. The run branch is never `main`, `master`, or `dev`.

## Round Overview

| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 000 | Phase 0 | `baseline_adapter.py` | baseline | `base.py` | 0.151139 | 14.9492578125 | - | not-applicable | `baseline_adapter.py` |
| 001 | `rounds/decision_001.md` | - | aborted | - | - | - | - | not-applicable | `baseline_adapter.py` |
| 002 | `rounds/decision_002.md` | - | aborted | - | - | - | - | not-applicable | `baseline_adapter.py` |

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

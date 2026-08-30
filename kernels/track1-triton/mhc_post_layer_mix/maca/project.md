# MHCPostLayerMix C500 Optimization Project

## Project Identity

- schema_version: 1
- skill_version: 2.0.0
- project_root: `/root/kernelswift-mhc/kernels/track1-triton/mhc_post_layer_mix/maca`
- base: `../base.py`
- baseline_adapter: `baseline_adapter.py`
- harness: `/root/kernelswift-mhc/auto_bench.py`
- interpreter: `/opt/conda/bin/python`
- device: `cuda:0` (MetaX C500 through the MACA compatibility surface)
- implementation_language: `triton`
- implementation_backend: `maca`
- target_profile: `triton_maca`

## Semantics

- operator: MHCPostLayerMix — a multi-head combination (MHC) post-layer mixing
  step: a small per-position `comb_res_mix` matmul applied to the residual, then
  a gated sum with `x` scaled by `post_layer_mix`.
- inputs: `x` is `(n0, n1, h)` bf16 (benchmark `(2, 4096, 1280)`); `residual` is
  `(n0, n1, mhc_mult, h)` bf16 (benchmark `(2, 4096, 4, 1280)`);
  `post_layer_mix` is `(n0, n1, mhc_mult, 1)` fp32 (benchmark `(2, 4096, 4, 1)`);
  `comb_res_mix` is `(n0, n1, mhc_mult, mhc_mult)` fp32 (benchmark
  `(2, 4096, 4, 4)`). All contiguous, on the input device.
- outputs: A single tensor `(n0, n1, mhc_mult, h)` bf16 (benchmark
  `(2, 4096, 4, 1280)`), contiguous, on the input device.
- mathematical_behavior:
  `term2 = einsum('abmn,abmc->abnc', comb_res_mix, residual.float())` — a
  batched matmul: for each `(a,b)` position, contract `m` over the `mhc_mult=4`
  comb coefficients against the residual's `m` index, yielding `(4, 1280)`.
  Then `out = (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()` —
  broadcast `x` (adding the `mhc_mult=4` dim) scaled per-head by
  `post_layer_mix`, add `term2`, and cast to bf16. Constructor takes no args
  (`get_init_inputs()` returns `[]`).
- tolerance_and_tie_rules: Single bf16 tensor output compared with
  `torch.allclose(..., atol=1e-2, rtol=1e-2, equal_nan=True)`. fp32 accumulate
  then bf16 cast; no tie-break rule applies.
- public_contract: The candidate module must expose `ModelNew`,
  `get_init_inputs`, and `get_inputs`. `ModelNew()` (no constructor args) and
  `forward(x, residual, post_layer_mix, comb_res_mix) -> Tensor` must remain
  compatible; forward must not mutate inputs and must preserve caller-selected
  device/current stream.

Unknown user-owned semantics must be resolved with the user. Do not infer them
from a candidate implementation.

## Invariants

- `base.py` and `auto_bench.py` are immutable after Phase 0 begins.
- Candidate code uses the CUDA-compatible PyTorch surface while Triton's active compiler backend remains MACA.
- The actual harness AST loader, measurement regime, device, and stream behavior remain unchanged.
- `mhc_mult = 4`, `h = 1280`, `n0 = 2`, `n1 = 4096`; the `einsum` contracts the
  `m` index (size 4) of `comb_res_mix [4,4]` against `residual [4,1280]` per
  `(a,b)` position, and the elementwise path uses fp32 accumulate before the
  final bf16 cast.
- Output shape/dtype/device contract and non-mutation of inputs remain
  compatible with `base.py`.
- The harness seeds each side identically, clones inputs, replaces candidate
  inputs with a clone of the reference inputs, runs under `torch.no_grad()`,
  and compares candidate outputs recursively against the reference.
- The AST loader retains imports, class/function definitions, and literal
  top-level assignments while discarding other top-level statements; loaded
  candidate code must still expose all required entry points.
- Candidate execution preserves caller-selected device and current stream. Any
  output-buffer reuse must have explicit per-instance ownership, compatibility
  keys including shape/dtype/device, invalidation, aliasing, and concurrency
  semantics.

The complete workflow-level rules are in `references/invariants.md`.

## Runtime Fingerprint

```yaml
python_version: 3.12.11
torch_version: 2.8.0+metax3.5.3.9
triton_distribution: triton 3.0.0+metax3.5.3.9
triton_version: 3.0.0
maca_version: 3.5.3.26
backend_target: GPUTarget(backend='maca', arch=80, warp_size=64)
backend_version: 3.5.3.9
device_name: MetaX C500
device_arch: capability=8.0, triton_arch=80, warp_size=64
device_memory: 65536 MiB
```

- target_profile_match: `pass`
- discovery_commands: `/opt/conda/bin/python -c 'import torch, triton; ...'`
- discovered_at: `2026-08-18T21:20:00Z`
- remote_host: `localhost` (C500 host is the current machine)
- environment_requirement: `MACA_PATH=/opt/maca` must be set before importing Triton

## Measurement Regime

- harness_path: `/root/kernelswift-mhc/auto_bench.py`
- harness_sha256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- shape: `n0=2,n1=4096,h=1280,mhc_mult=4; x [2,4096,1280] bf16, residual [2,4096,4,1280] bf16, post_layer_mix [2,4096,4,1] fp32, comb_res_mix [2,4096,4,4] fp32; output [2,4096,4,1280] bf16`
- dtype: `x/residual bf16, post_layer_mix/comb_res_mix fp32, output bf16`
- device: `cuda:0`
- seed: `42`
- atol: `1e-2`
- rtol: `1e-2`
- warmup: `50`
- repeat: `100`
- timing_order: `sequential complete accepted-reference block, then complete candidate block`
- primary_metric: `unrounded median wall_time_ms`
- profile_mode: `forward`
- profiler_warmup: `20`
- profiler_iterations: `50`
- profiler_scopes: `accepted_reference,candidate`
- correctness_command: `cd /root/kernelswift-mhc && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mhc_post_layer_mix/base.py --v1_file kernels/track1-triton/mhc_post_layer_mix/maca/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback`
- benchmark_command: `cd /root/kernelswift-mhc && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mhc_post_layer_mix/base.py --v1_file kernels/track1-triton/mhc_post_layer_mix/maca/baseline_adapter.py --warmup 50 --repeat 100`

Benchmark wall time controls adoption. Profiler data is attributable diagnostic
evidence and is normalized per forward call.

## Measurement Fingerprint

- measurement_fingerprint: `17bf289997ea6c7a2961ba2640125464ed046471dbff9261a8dcba7fbfccc17e`
- base_sha256: `e392799f72edcf7210373ac8f2522bcf3d4f740c8f9dd0d1f117bc627939ebf3`
- base_bytes: `1342`
- harness_sha256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- harness_bytes: `26142`
- baseline_adapter_sha256: `2c0c121e2e4504e791fee3675ac1cd54d1322059fc537555578a91cabd2a24e6`
- fingerprint_command: `SHA-256(base bytes || NUL || harness bytes || NUL || canonical JSON settings with sort_keys=True and separators=(',', ':'))`

## Optional Target

- target_mode: `null`
- target_value: `null`
- target_metric: `wall_time_ms`
- source: user
- target_measurement_fingerprint: `null`

## Git Run Identity

- base_branch: `dev`
- base_commit: `8c1ebcd04afe4da31357bf426bc3e523129e411c`
- run_branch: `kernel-opt/mhc-post-layer-mix-c500-20260818`

## Round Overview

| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 000 | Phase 0 | `baseline_adapter.py` | baseline | `base.py` | 7.635598 | 7559.1986 | - | not-applicable | `baseline_adapter.py` |
| 001 | H-001 tiny-k-gemm-fusion | `triton_mhc_001.py` | accepted | `baseline_adapter.py` | 0.241083 | 168.5607 | +96.84% | confirmed | `triton_mhc_001.py` |
| 002 | abort (memory-bound) | - | aborted | `triton_mhc_001.py` | 0.241083 | 168.5607 | - | not-applicable | `triton_mhc_001.py` |

## Reproduction

```bash
cd /root/kernelswift-mhc && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mhc_post_layer_mix/base.py --v1_file kernels/track1-triton/mhc_post_layer_mix/maca/baseline_adapter.py --warmup 50 --repeat 100
```

# MHCHeadComputeMix C500 Optimization Project

## Project Identity

- schema_version: 1
- skill_version: 2.0.0
- project_root: `/root/kernelswift-mhcc/kernels/track1-triton/mhc_head_compute_mix/maca`
- base: `../base.py`
- baseline_adapter: `baseline_adapter.py`
- harness: `/root/kernelswift-mhcc/auto_bench.py`
- interpreter: `/opt/conda/bin/python`
- device: `cuda:0` (MetaX C500 through the MACA compatibility surface)
- implementation_language: `triton`
- implementation_backend: `maca`
- target_profile: `triton_maca`

## Semantics

- operator: MHCHeadComputeMix — head-combination (HC) mixing with sigmoid gates
  and a Sinkhorn-normalized `hc x hc` combination matrix, iterated `sinkhorn_iters`
  times.
- inputs: `mixes` is `(b, s, mix_hc)` fp32 (benchmark `(2, 8, 24)` with
  `mix_hc = (2 + hc) * hc = 24`); `hc_scale` is `(3,)` fp32 (benchmark
  `[0.5, 0.25, 1.0]`); `hc_base` is `(mix_hc,)` fp32 (benchmark `[24]`,
  `randn * 0.1`). All on `cuda:0`.
- outputs: A three-tensor tuple `(pre, post, comb)`:
  `pre` `(b, s, hc)` fp32, `post` `(b, s, hc)` fp32, `comb` `(b, s, hc, hc)`
  fp32 (benchmark `(2, 8, 4)` and `(2, 8, 4, 4)`).
- mathematical_behavior: Reshape `mixes` to `(-1, mix_hc)`, cast all to fp32.
  `pre = sigmoid(x[:, :hc] * s0 + base[:hc]) + eps`.
  `post = 2 * sigmoid(x[:, hc:2hc] * s1 + base[hc:2hc])`.
  `raw = x[:, 2hc : 2hc + hc*hc]`, `comb = raw.view(-1, hc, hc) * s2 + base[2hc:...].view(1, hc, hc)`.
  Then: `row_max = comb.amax(-1, keepdim=True)`; `comb = exp(comb - row_max)`;
  `comb = comb / comb.sum(-1, keepdim=True) + eps`;
  `comb = comb / (comb.sum(-2, keepdim=True) + eps)`;
  then `for _ in range(sinkhorn_iters - 1)`: `comb = comb / (comb.sum(-1, keepdim=True) + eps)` and
  `comb = comb / (comb.sum(-2, keepdim=True) + eps)`.
  Return the three reshaped views. Constructor benchmark
  `(hc_mult=4, sinkhorn_iters=20, eps=1e-6)`.
- tolerance_and_tie_rules: Three fp32 tensors in a tuple, each compared with
  `torch.allclose(..., atol=1e-2, rtol=1e-2, equal_nan=True)`. Sinkhorn is a
  deterministic alternating normalization; no tie-break rule applies.
- public_contract: The candidate module must expose `ModelNew`,
  `get_init_inputs`, and `get_inputs`.
  `ModelNew.__init__(hc_mult=4, sinkhorn_iters=20, eps=1e-6)` and
  `forward(mixes, hc_scale, hc_base) -> tuple[Tensor, Tensor, Tensor]` must
  remain compatible; forward must not mutate inputs and must preserve
  caller-selected device/current stream.

Unknown user-owned semantics must be resolved with the user. Do not infer them
from a candidate implementation.

## Invariants

- `base.py` and `auto_bench.py` are immutable after Phase 0 begins.
- Candidate code uses the CUDA-compatible PyTorch surface while Triton's active compiler backend remains MACA.
- The actual harness AST loader, measurement regime, device, and stream behavior remain unchanged.
- `hc = hc_mult = 4`, `mix_hc = (2 + hc) * hc = 24`, `sinkhorn_iters = 20`,
  `eps = 1e-6`; the Sinkhorn loop runs exactly `sinkhorn_iters - 1 = 19`
  additional iterations after the first normalization pair.
- The sigmoid gates, `exp(comb - row_max)` numerical stabilization, the `+eps`
  additions, and the alternating row/column normalizations preserve the exact
  fp32 semantics of `base.py`.
- Output tuple/shape/dtype/device contract and non-mutation of inputs remain
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
- discovered_at: `2026-08-19T00:10:00Z`
- remote_host: `localhost` (C500 host is the current machine)
- environment_requirement: `MACA_PATH=/opt/maca` must be set before importing Triton

## Measurement Regime

- harness_path: `/root/kernelswift-mhcc/auto_bench.py`
- harness_sha256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- shape: `b=2,s=8,hc=4,mix_hc=24; mixes [2,8,24] fp32, hc_scale [3] fp32, hc_base [24] fp32; output pre/post [2,8,4], comb [2,8,4,4]`
- dtype: `all fp32`
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
- correctness_command: `cd /root/kernelswift-mhcc && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix/maca/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback`
- benchmark_command: `cd /root/kernelswift-mhcc && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix/maca/baseline_adapter.py --warmup 50 --repeat 100`

Benchmark wall time controls adoption. Profiler data is attributable diagnostic
evidence and is normalized per forward call.

## Measurement Fingerprint

- measurement_fingerprint: `d8f4b63bfbf09ce8a32f3bdcd4d85553f34abce7384e495ba5f66baf49bf795e`
- base_sha256: `4c5167f6cfe9099786f204fad222f3b780d3ebb814494b9582112422abf84ac5`
- base_bytes: `2194`
- harness_sha256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- harness_bytes: `26142`
- baseline_adapter_sha256: `c3cc90deb93c69f092e876ddc0057ddf6c2d7ac28020f1b7fd1ed260bca72fee`
- fingerprint_command: `SHA-256(base bytes || NUL || harness bytes || NUL || canonical JSON settings with sort_keys=True and separators=(',', ':'))`

## Optional Target

- target_mode: `null`
- target_value: `null`
- target_metric: `wall_time_ms`
- source: user
- target_measurement_fingerprint: `null`

## Git Run Identity

- base_branch: `dev`
- base_commit: `138f0caf13784399abdc29507a6ac1f29e0fd947`
- run_branch: `kernel-opt/mhc-head-compute-mix-c500-20260819`

## Round Overview

| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 000 | Phase 0 | `baseline_adapter.py` | baseline | `base.py` | 1.515187 | 534.685 | - | not-applicable | `baseline_adapter.py` |
| 001 | H-001 sinkhorn-loop-fusion | `triton_mhcc_001.py` | accepted | `baseline_adapter.py` | 0.118357 | 43.791 | +92.89% | confirmed | `triton_mhcc_001.py` |
| 002 | abort (latency-floor) | - | aborted | `triton_mhcc_001.py` | 0.118357 | 43.791 | - | not-applicable | `triton_mhcc_001.py` |

## Reproduction

```bash
cd /root/kernelswift-mhcc && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix/maca/baseline_adapter.py --warmup 50 --repeat 100
```

# MMEncoderAttention C500 Optimization Project

## Project Identity

- schema_version: 1
- skill_version: 2.0.0
- project_root: `/root/kernelswift-mma/kernels/track1-triton/mm_encoder_attention/maca`
- base: `../base.py`
- baseline_adapter: `baseline_adapter.py`
- harness: `/root/kernelswift-mma/auto_bench.py`
- interpreter: `/opt/conda/bin/python`
- device: `cuda:0` (MetaX C500 through the MACA compatibility surface)
- implementation_language: `triton`
- implementation_backend: `maca`
- target_profile: `triton_maca`

## Semantics

- operator: MMEncoderAttention — a standard multi-head attention (MHA) forward
  via `torch.nn.functional.scaled_dot_product_attention`.
- inputs: `query`, `key`, `value` are each contiguous `(bsz, seq_len, hidden)`
  tensors (benchmark `(2, 83, 512)`, `float16`, `cuda:0`) with
  `hidden = num_heads * head_size`; `num_kv_heads == num_heads == 8` (MHA, not
  GQA/MQA).
- outputs: A single tensor `(bsz, seq_len, hidden)` (benchmark `(2, 83, 512)`),
  `float16`, contiguous, on the input device.
- mathematical_behavior: Reshape each input to `(bsz, heads, seq, head_size)`
  via `view(...).transpose(1, 2)`; compute
  `out = F.scaled_dot_product_attention(q, k, v, scale=1/sqrt(head_size))`
  (no attention mask, `cu_seqlens=None`); reshape back to
  `(bsz, seq_len, hidden)` via `transpose(1, 2).reshape(...)`.
  Constructor benchmark `(num_heads=8, head_size=64, num_kv_heads=8)`.
- tolerance_and_tie_rules: Single fp16 tensor output compared with
  `torch.allclose(..., atol=1e-2, rtol=1e-2, equal_nan=True)`. SDPA is a
  floating-point reduction over `head_size=64` with fp16 accumulation; the
  harness tolerance is loose enough to absorb backend-specific SDPA algorithm
  differences. No tie-break rule applies.
- public_contract: The candidate module must expose `ModelNew`,
  `get_init_inputs`, and `get_inputs`.
  `ModelNew.__init__(num_heads=8, head_size=64, num_kv_heads=8)` and
  `forward(query, key, value) -> Tensor` must remain compatible; forward must
  not mutate inputs and must preserve caller-selected device/current stream.

Unknown user-owned semantics must be resolved with the user. Do not infer them
from a candidate implementation.

## Invariants

- `base.py` and `auto_bench.py` are immutable after Phase 0 begins.
- Candidate code uses the CUDA-compatible PyTorch surface while Triton's active compiler backend remains MACA.
- The actual harness AST loader, measurement regime, device, and stream behavior remain unchanged.
- `num_heads == num_kv_heads == 8` (MHA), `head_size == 64`, `hidden == 512`;
  scale is `1/sqrt(head_size)`.
- The reshape/transpose to `(bsz, heads, seq, head_size)`, the SDPA call with no
  mask and `cu_seqlens=None`, and the transpose/reshape back to
  `(bsz, seq_len, hidden)` preserve the exact output layout.
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
- discovered_at: `2026-08-18T20:45:00Z`
- remote_host: `localhost` (C500 host is the current machine)
- environment_requirement: `MACA_PATH=/opt/maca` must be set before importing Triton

## Measurement Regime

- harness_path: `/root/kernelswift-mma/auto_bench.py`
- harness_sha256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- shape: `bsz=2,seq_len=83,num_heads=8,head_size=64,num_kv_heads=8,hidden=512`
- dtype: `query/key/value=fp16, output=fp16`
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
- correctness_command: `cd /root/kernelswift-mma && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/maca/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback`
- benchmark_command: `cd /root/kernelswift-mma && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/maca/baseline_adapter.py --warmup 50 --repeat 100`

Benchmark wall time controls adoption. Profiler data is attributable diagnostic
evidence and is normalized per forward call.

## Measurement Fingerprint

- measurement_fingerprint: `29ecde127206fc1808c2d7f28951e44ee55a257aadfda78517e64d3493ce1862`
- base_sha256: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2`
- base_bytes: `2284`
- harness_sha256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- harness_bytes: `26142`
- baseline_adapter_sha256: `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e`
- fingerprint_command: `SHA-256(base bytes || NUL || harness bytes || NUL || canonical JSON settings with sort_keys=True and separators=(',', ':'))`

## Optional Target

- target_mode: `null`
- target_value: `null`
- target_metric: `wall_time_ms`
- source: user
- target_measurement_fingerprint: `null`

## Git Run Identity

- base_branch: `dev`
- base_commit: `99cd9f4ee002f83e21c7c639c891ebcc2d5ba689`
- run_branch: `kernel-opt/mm-encoder-attention-c500-20260818`

## Round Overview

| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 000 | Phase 0 | `baseline_adapter.py` | baseline | `base.py` | 0.115761 | 14.9812 | - | not-applicable | `baseline_adapter.py` |
| 001 | H-001 fused-mha-kernel | `triton_mha_001.py` | accepted | `baseline_adapter.py` | 0.164166 | 79.6977 | -48.16% | partially-confirmed | `triton_mha_001.py` |
| 002 | H-002 remove-transpose-copy | `triton_mha_002.py` | accepted | `triton_mha_001.py` | 0.127777 | 67.7273 | +23.54% | confirmed | `triton_mha_002.py` |

## Reproduction

```bash
cd /root/kernelswift-mma && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/maca/baseline_adapter.py --warmup 50 --repeat 100
```

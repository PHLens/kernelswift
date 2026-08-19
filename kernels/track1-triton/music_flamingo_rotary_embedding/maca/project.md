# MusicFlamingoRotaryEmbedding C500 Optimization Project

## Project Identity

- schema_version: 1
- skill_version: 2.0.0
- project_root: `/root/kernelswift-rotary/kernels/track1-triton/music_flamingo_rotary_embedding/maca`
- base: `../base.py`
- baseline_adapter: `baseline_adapter.py`
- harness: `/root/kernelswift-rotary/auto_bench.py`
- interpreter: `/opt/conda/bin/python`
- device: `cuda:0` (MetaX C500 through the MACA compatibility surface)
- implementation_language: `triton`
- implementation_backend: `maca`
- target_profile: `triton_maca`

## Semantics

- operator: MusicFlamingoRotaryEmbedding — a batch (song) + time positional
  embedding that returns `(cos, sin)` where each combines batch and time
  frequencies.
- inputs: `timestamps` is a contiguous `(B, SEQ)` tensor (benchmark `(4, 32)`,
  `float32`, `cuda:0`) of normalized per-time-step song timestamps; `seq_len` is
  a Python `int` (benchmark `32`) that slices the precomputed time-position
  angle table. Constructor buffers `inv_freq` (shape `dim//2`, benchmark
  `dim=64`) and `position_angles` (shape `(max_seq_len, dim)`, benchmark
  `(256, 64)`) are registered and moved to the input device by the harness.
- outputs: A two-tensor tuple `(cos, sin)`, each with shape `(B, SEQ, 2*dim)`
  (benchmark `(4, 32, 128)`), `float32`, contiguous, on the input device.
- mathematical_behavior: Build `inv_freq = 1 / (base ** (arange(0,dim,2)/dim))`.
  Build `position_angles = (arange(max_seq_len)/max_seq_len * 2π) ⊗ inv_freq`,
  `repeat_interleave(2, dim=-1)`. In forward: batch frequencies
  `batch_freqs = (arange(B)/max_seq_len) ⊗ inv_freq`, `repeat_interleave(2)`;
  broadcast `batch_freqs[:, None, :]` against
  `position_angles[:seq_len][None, :, :]`, `cat` along the last dim to
  `freqs` of shape `(B, SEQ, 2*dim)`; `angle = (-timestamps * 2π).to(freqs)`;
  `freqs = freqs * angle.unsqueeze(-1)`; return `(freqs.cos(), freqs.sin())`.
  Constructor benchmark `(dim=64, max_seq_len=256, base=10000.0)`.
- tolerance_and_tie_rules: The harness requires identical tuple structure,
  tensor shapes, and dtypes; `cos`/`sin` use
  `torch.allclose(..., atol=1e-2, rtol=1e-2, equal_nan=True)`. No tie-break
  rule applies (pure floating elementwise transform).
- public_contract: The candidate module must expose `ModelNew`,
  `get_init_inputs`, and `get_inputs`.
  `ModelNew.__init__(dim: int = 64, max_seq_len: int = 256, base: float = 10000.0)`
  and `forward(timestamps: torch.Tensor, seq_len: int) -> tuple[Tensor, Tensor]`
  must remain compatible; forward must not mutate `timestamps` and must preserve
  the caller-selected device/current stream.

Unknown user-owned semantics must be resolved with the user. Do not infer them
from a candidate implementation.

## Invariants

- `base.py` and `auto_bench.py` are immutable after Phase 0 begins.
- Candidate code uses the CUDA-compatible PyTorch surface while Triton's active compiler backend remains MACA.
- The actual harness AST loader, measurement regime, device, and stream behavior remain unchanged.
- `inv_freq` (dim//2) and `position_angles` (max_seq_len × dim) buffers are
  built once in the constructor from `dim`, `max_seq_len`, and `base`; the
  benchmark constructor is `(64, 256, 10000.0)`.
- Forward uses only the first `seq_len` rows of `position_angles`, the batch
  frequency uses `arange(B)/max_seq_len` (not `/seq_len`), the angle uses
  `-timestamps * 2π`, and the final output is `(cos, sin)` of the combined
  broadcast-then-concatenated frequency scaled by the angle.
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
- discovered_at: `2026-08-18T19:50:00Z`
- remote_host: `localhost` (C500 host is the current machine)
- environment_requirement: `MACA_PATH=/opt/maca` must be set before importing Triton

## Measurement Regime

- harness_path: `/root/kernelswift-rotary/auto_bench.py`
- harness_sha256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- shape: `B=4,SEQ=32,dim=64,max_seq_len=256,output=cos/sin (4,32,128)`
- dtype: `timestamps=fp32, inv_freq=fp32, position_angles=fp32, output=fp32`
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
- correctness_command: `cd /root/kernelswift-rotary && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/maca/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback`
- benchmark_command: `cd /root/kernelswift-rotary && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/maca/baseline_adapter.py --warmup 50 --repeat 100`

Benchmark wall time controls adoption. Profiler data is attributable diagnostic
evidence and is normalized per forward call.

## Measurement Fingerprint

- measurement_fingerprint: `486242286573efe11bdd7b852247cb0ed4d63113e0e41c7c432ab65e654a6518`
- base_sha256: `98be7d25ad94930fd9497e6ab33e64970c509c6c3569c8c3df40ae9d22b9a341`
- base_bytes: `2138`
- harness_sha256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- harness_bytes: `26142`
- baseline_adapter_sha256: `40c1a8bfbd9a0e957f21ae8ac686aa4c378a28299fd1f053d1e35b5fa8c443e0`
- fingerprint_command: `SHA-256(base bytes || NUL || harness bytes || NUL || canonical JSON settings with sort_keys=True and separators=(',', ':'))`

## Optional Target

- target_mode: `null`
- target_value: `null`
- target_metric: `wall_time_ms`
- source: user
- target_measurement_fingerprint: `null`

## Git Run Identity

- base_branch: `dev`
- base_commit: `e8533192f65ed4610a4b59859f1969ea83955f87`
- run_branch: `kernel-opt/music-flamingo-rotary-c500-20260818`

## Round Overview

| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 000 | Phase 0 | `baseline_adapter.py` | baseline | `base.py` | 0.190557 | 50.9489 | - | not-applicable | `baseline_adapter.py` |
| 001 | H-001 kernel-fusion | `triton_rotary_001.py` | accepted | `baseline_adapter.py` | 0.080036 | 16.9011 | +55.57% | confirmed | `triton_rotary_001.py` |
| 002 | abort (host-bound) | - | aborted | `triton_rotary_001.py` | 0.080036 | 16.9011 | - | not-applicable | `triton_rotary_001.py` |

## Reproduction

```bash
cd /root/kernelswift-rotary && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/maca/baseline_adapter.py --warmup 50 --repeat 100
```

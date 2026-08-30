# MusicFlamingoRotaryEmbedding Optimization Project

## Project Identity

- schema_version: 1
- skill_version: 2.0.0
- project_root: `/workspace/kernelswift/.worktrees/music-rotary-ascend/kernels/track1-triton/music_flamingo_rotary_embedding/ascend`
- base: `../base.py` (shared reference; no torch_mlu dependency)
- baseline_adapter: `baseline_adapter.py`
- harness: `/workspace/kernelswift/.worktrees/music-rotary-ascend/auto_bench.py`
- interpreter: `/usr/local/python3.11.15/bin/python3`
- device: `npu:0`
- implementation_language: `triton`
- implementation_backend: `ascend`
- target_profile: `triton_ascend`

## Semantics

- operator: MusicFlamingoRotaryEmbedding — batch (song) + time positional embedding, returning (cos, sin).
- inputs:
  - `timestamps`: `Tensor[4,32]` fp32 (batch_size=4, seq_len=32)
  - `seq_len`: int = 32
- outputs: tuple `(cos, sin)`, each `Tensor[4,32,128]` fp32
- mathematical_behavior:
  - `inv_freq = 1/(base^(arange(0,dim,2)/dim))`, dim=64, base=10000.0 (register_buffer)
  - `position_angles`: `[max_seq_len, dim]` from `positions/max_seq_len*2π * inv_freq` then `repeat_interleave(2)` (register_buffer, max_seq_len=256)
  - `batch_freqs = batch_positions/max_seq_len * inv_freq` then `repeat_interleave(2)` -> `[B, dim]`
  - `time_freqs = position_angles[:seq_len]` -> `[seq_len, dim]`
  - broadcast to `[B, seq_len, dim]`, cat -> `[B, seq_len, 2*dim]`
  - `angle = (-timestamps * 2π)` -> `[B, seq_len]`; `freqs = freqs * angle.unsqueeze(-1)`
  - return `(freqs.cos(), freqs.sin())`
- tolerance_and_tie_rules: no explicit tolerance is declared in `base.py`; outputs are fp32 `cos`/`sin` of the same `freqs` tensor, so exact bitwise match is not guaranteed across backends — the harness performs numeric comparison and a candidate must match the reference within the harness's default fp32 comparison bound.
- public_contract: `ModelNew(dim=64, max_seq_len=256, base=10000.0)`, `forward(timestamps, seq_len) -> (cos, sin)`

## Invariants

- `base.py` is the immutable reference; its bytes are unchanged after baseline adapter generation.
- Candidate output is a tuple `(cos, sin)` of two `Tensor[4,32,128]` fp32.
- The harness AST loader rewrites device strings ("cuda" -> "npu") and filters module AST.
- Wall time is measured by the unchanged harness; reference and candidate profiler scopes are separate.

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

- harness_path: `/workspace/kernelswift/.worktrees/music-rotary-ascend/auto_bench.py`
- harness_sha256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- shape: `timestamps=[4,32] fp32; seq_len=32; dim=64; output (cos,sin) each [4,32,128] fp32`
- dtype: `fp32`
- device: `npu:0`
- warmup: `50`
- repeat: `100`
- primary_metric: `unrounded median wall_time_ms`
- profiler_iterations: `50`

## Measurement Fingerprint

- measurement_fingerprint: `pending`
- base_sha256: `98be7d25ad94930fd9497e6ab33e64970c509c6c3569c8c3df40ae9d22b9a341`
- baseline_adapter_sha256: `40c1a8bfbd9a0e957f21ae8ac686aa4c378a28299fd1f053d1e35b5fa8c443e0`

## Optional Target

- target_mode: `null`
- target_value: `null`
- target_metric: `wall_time_ms`
- source: user

## Git Run Identity

- base_branch: `dev`
- base_commit: `a1dce831e30275a6411c643c08080867774904a5`
- run_branch: `kernel-opt/music-rotary-ascend`

## Round Overview

| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|

## Reproduction

```bash
cd /workspace/kernelswift/.worktrees/music-rotary-ascend
python3 auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/ascend/baseline_adapter.py --warmup 50 --repeat 100
```

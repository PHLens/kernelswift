# MusicFlamingo Rotary Embedding Optimization Project (S60 / GCU)

## Project Identity

- schema_version: 1
- skill_version: 2.0.0
- project_root: `/root/kernelswift/.worktrees/rotary-embedding-s60/kernels/track1-triton/music_flamingo_rotary_embedding/s60`
- base: `../base.py`
- baseline_adapter: `baseline_adapter.py`
- harness: `/root/kernelswift/.worktrees/rotary-embedding-s60/auto_bench.py`
- interpreter: `/usr/bin/python3` on the S60 host
- device: `gcu:0`
- implementation_language: `triton`
- implementation_backend: `gcu`
- target_profile: `triton_gcu`

## Semantics

- operator: `MusicFlamingoRotaryEmbedding` — batch (song) + time positional frequency embedding; returns `(cos, sin)`.
- inputs:
    - `timestamps`: `Tensor[4, 32]`, fp32, on `gcu:0` (normalized song timestamps)
    - `seq_len`: Python `int` = 32
- outputs:
    - `tuple(Tensor[4, 32, 128], Tensor[4, 32, 128])`, both fp32, on `gcu:0`
- mathematical_behavior:
    - `inv_freq = 1 / (base ** (arange(0, dim, 2) / dim))`, dim=64 -> shape [32]; registered buffer
    - `position_angles` buffer: `(arange(max_seq_len)/max_seq_len * 2pi) @ inv_freq`, repeat_interleave(2) -> [256, 64]; registered buffer
    - forward: `batch_freqs = (arange(B)/max_seq_len) @ inv_freq`, repeat_interleave(2) -> [4, 64]
    - `time_freqs = position_angles[:seq_len]` -> [32, 64]
    - broadcast both to `[4, 32, 64]`, cat along last dim -> `[4, 32, 128]`
    - `angle = (-timestamps * 2pi)`; `freqs = freqs * angle.unsqueeze(-1)`
    - return `(freqs.cos(), freqs.sin())`
- tolerance_and_tie_rules: `torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True)`; output fp32
- public_contract:
    - `ModelNew(dim=64, max_seq_len=256, base=10000.0)`
    - `ModelNew.forward(timestamps, seq_len) -> (cos, sin)`
    - `get_init_inputs() -> [64, 256, 10000.0]`
    - `get_inputs() -> [timestamps, seq_len]` (2 args; seq_len is Python int)

## Invariants

- state_dict keys must be exactly `{inv_freq, position_angles}` (registered buffers, not parameters), shapes `[32]` and `[256, 64]` fp32. `compare_case` calls `load_state_dict` in a silent try/except, so mismatched keys silently skip sync.
- candidate `get_inputs()` returns exactly 2 arguments: `timestamps` (fp32 tensor) and `seq_len` (Python int). `clone_value` passes the int through unchanged.
- output must be a Python `tuple` of exactly 2 tensors `[4,32,128]` fp32 (harness `compare_values` recurses tuples; a list or stacked tensor fails the type check).
- device literal `'cuda'` is rewritten to `'gcu'` by `_rewrite_device_for_backend`.
- module-level non-literal assignments are stripped by `_filter_module_ast`.
- The computation is pure elementwise/view (arange, mul/div, repeat_interleave, broadcast, cat, cos, sin); no GEMM, no reduction across data (only over the synthetic `dim` axis during buffer construction, which is host-side in `__init__`).

## Runtime Fingerprint

```yaml
triton_distribution: triton
triton_version: 3.6.0
triton_gcu_version: 3.6.0+1.0.20260722
torch_version: 2.10.0+cpu
torch_gcu_version: 2.10.0+3.8.0.2
backend_target: triton_gcu
backend_version: 3.6.0+1.0.20260722
device_name: GCU
device_arch: major=3, minor=0
multi_processor_count: 2
total_memory: 43878764544
```

- target_profile_match: `pass`
- discovered_at: `2026-08-18T22:35:00Z`
- host: `5d02974bab32`

## Measurement Regime

- harness_path: `/root/kernelswift/.worktrees/rotary-embedding-s60/auto_bench.py`
- harness_sha256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- shape: `timestamps=[4,32] fp32; seq_len=32 int; output tuple (cos,sin) each [4,32,128] fp32`
- dtype: `fp32`
- device: `gcu:0`
- warmup: `50`
- repeat: `100`
- timing_order: `interleaved accepted-reference/candidate`
- primary_metric: `unrounded median wall_time_ms`
- profiler_iterations: `50`
- profiler_scopes: `baseline_base,candidate_triton_rotary_001`
- profiler_device_time: `unavailable on recorded GCU exporter; runtime_launch_* fields are retained`
- correctness_command: `cd /root/kernelswift/.worktrees/rotary-embedding-s60 && python3 auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/s60/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback`
- benchmark_command: `cd /root/kernelswift/.worktrees/rotary-embedding-s60 && python3 auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/s60/triton_rotary_001.py --warmup 50 --repeat 100`
- profiler_command: `cd /root/kernelswift/.worktrees/rotary-embedding-s60 && python3 auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/s60/triton_rotary_001.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/music_flamingo_rotary_embedding/s60/log/rotary_round_001_forward_50iter.pt.trace.json`

## Measurement Fingerprint

- measurement_fingerprint: `a1ee09ca54ab2210943bd030a6649c57d96b09d4c1beed863f4a98681ae425f2`
- base_sha256: `99829754f4cdc4bfd2808e051de549f0791414241e7fdbad7a1b8294a15be475`
- baseline_adapter_sha256: `9fc87abbe0e6268f06c969e94f5400abea51cdf315276a4ac5cef5bd0ad8a26f`
- fingerprint_command: `sha256(base.py || NUL || auto_bench.py || NUL || canonical JSON settings with sort_keys=True and separators=(',', ':'))`

## Optional Target

- target_mode: `null`
- target_value: `null`
- target_metric: `wall_time_ms`
- source: user
- target_measurement_fingerprint: `null`

## Git Run Identity

- base_branch: `dev`
- base_commit: `e853319`
- run_branch: `kernel-opt/rotary-embedding-s60`

## Round Overview

| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 000 | Phase 0 | `baseline_adapter.py` | baseline | `base.py` | 0.465926 | unavailable: GCU runtime-launch-only | - | not-applicable | `baseline_adapter.py` |
| 001 | `rounds/decision_001.md` | `triton_rotary_001.py` | no-improvement | `baseline_adapter.py` | 5.162427 | unavailable: GCU runtime-launch-only | -1010.99% (0.09x) | falsified (launch 13→1 but single-program device serial) | `baseline_adapter.py` |
| 002 | `rounds/decision_002.md` | `triton_rotary_002.py` | no-improvement | `baseline_adapter.py` | 0.525050 | unavailable: GCU runtime-launch-only | -13.00% (0.91x) | partially-confirmed (grid fixed, device 5.15→0.53ms, still +13% vs eager) | `baseline_adapter.py` |
| 003 | `rounds/decision_003.md` | not-created | aborted | `baseline_adapter.py` | - | - | - | not-applicable (measurement-bound) | `baseline_adapter.py` |

## Reproduction

```bash
cd /root/kernelswift/.worktrees/rotary-embedding-s60
python3 auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/s60/baseline_adapter.py --warmup 50 --repeat 100
```

```bash
cd /root/kernelswift/.worktrees/rotary-embedding-s60
python3 auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/s60/baseline_adapter.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/music_flamingo_rotary_embedding/s60/log/rotary_baseline_forward_50iter.pt.trace.json
```

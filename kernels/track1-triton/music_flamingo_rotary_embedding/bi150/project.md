# Music Flamingo Rotary Embedding Optimization Project

## Project Identity

- schema_version: 1
- skill_version: 2.0.0
- project_root: `/root/CodeBuddy/20260818191200/kernelswift`
- base: `kernels/track1-triton/music_flamingo_rotary_embedding/base.py`
- baseline_adapter: `baseline_adapter.py`
- harness: `/root/CodeBuddy/20260818191200/kernelswift/auto_bench.py`
- interpreter: `/usr/local/bin/python3`
- device: `cuda:0 (Iluvatar BI-V150)`
- implementation_language: `triton`
- implementation_backend: `cuda`
- target_profile: `triton_cuda`

## Semantics

- operator: `music_flamingo_rotary_embedding` — a batch(song) + time dual-frequency
  rotary positional embedding. Given per-timestep song timestamps, it returns
  `(cos, sin)` where each combines batch-frequency and time-frequency angle terms.
- inputs: `timestamps` is a contiguous `(4, 32)` float32 tensor on the
  caller-selected accelerator (`torch.rand(4, 32, device="cuda")`); only the
  batch and time dimensions participate in the reference computation.
  `seq_len` is a Python `int` (benchmark value `32`), not a tensor.
  `get_inputs()` returns the list `[timestamps, seq_len]`.
- outputs: A two-tensor tuple `(cos, sin)`, each with shape `(4, 32, 128)`,
  float32, contiguous, on the input accelerator. The inner dim `128` comes from
  `dim=64` with `repeat_interleave(2, dim=-1)` doubling each frequency to a
  `cos`/`sin` pair, then concatenating `batch_freqs` `(4, 1, 64) ->
  (4, 32, 64)` and `time_freqs` `(1, 32, 64) -> (4, 32, 64)` along `dim=-1` to
  yield `(4, 32, 128)`.
- mathematical_behavior: `inv_freq = 1.0 / (base ** (arange(0, dim, 2) / dim))`
  with `dim=64`, `base=10000.0`; `arange(0, 64, 2)` yields length `32`, so
  `inv_freq` has length `32` and `repeat_interleave(2, dim=-1)` doubles each
  entry to length `64`. `position_angles` precomputes
  `(arange(max_seq_len) / max_seq_len * 2*pi) * inv_freq`, then
  `repeat_interleave(2)` to length `64`. In `forward`: `batch_positions =
  arange(timestamps.shape[0]) / max_seq_len`; `batch_freqs = batch_positions[...,None]
  * inv_freq` then `repeat_interleave(2)`, reshaped to `(B, 1, 64)`.
  `time_freqs = position_angles[:seq_len][None]` is `(1, S, 64)`; both broadcast
  to `(B, S, 64)`. `freqs = cat((batch_freqs, time_freqs), dim=-1)` -> `(B, S, 128)`;
  then `angle = (-timestamps * 2*pi)` cast to `freqs` dtype and unsqueezed to
  `(B, S, 1)`, `freqs = freqs * angle`, and the result returns
  `(freqs.cos(), freqs.sin())`, each of shape `(B, S, 128)`.
- tolerance_and_tie_rules: The harness requires the same tuple structure, tensor
  shapes, and dtypes. All outputs are floating point and are compared with
  `torch.allclose(..., atol=1e-2, rtol=1e-2, equal_nan=True)`. There are no
  integer outputs and no tie-break rule.
- public_contract: The candidate module must expose `ModelNew`,
  `get_init_inputs`, and `get_inputs`.
  `ModelNew.__init__(dim: int = 64, max_seq_len: int = 256, base: float = 10000.0)`
  and `forward(timestamps: torch.Tensor, seq_len: int) ->
  tuple[torch.Tensor, torch.Tensor]` must remain compatible. `forward` does not
  mutate the input and preserves the caller-selected device/current stream.
  `get_init_inputs()` returns `[64, 256, 10000.0]`.

Unknown user-owned semantics must be resolved with the user. Do not infer them
from a candidate implementation.

## Invariants

- `base.py` and `auto_bench.py` are immutable after Phase 0 begins. The
  `baseline_adapter.py` bytes must not be overwritten by the `base.py` content.
- Candidate code uses the CUDA-compatible PyTorch surface (the Triton active
  compiler backend is `cuda` on the BI150 / CoreX environment).
- The harness AST loader retains imports, class/function definitions, and
  literal top-level assignments while discarding other top-level statements;
  loaded candidate code must still expose `ModelNew`, `get_inputs`, and
  `get_init_inputs`.
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
- discovered_at: `2026-08-18T11:40:00Z`

These values are observed in Phase 0. They are not assumed from the profile.

## Measurement Regime

- harness_path: `/root/CodeBuddy/20260818191200/kernelswift/auto_bench.py`
- harness_sha256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- shape: `timestamps [4,32] fp32; seq_len 32 int; outputs (cos,sin) each [4,32,128] fp32`
- dtype: `fp32 (timestamps, cos, sin)`
- device: `cuda:0 (Iluvatar BI-V150)`
- warmup: `50`
- repeat: `100`
- timing_order: `interleaved accepted-reference/candidate`
- primary_metric: `unrounded median wall_time_ms`
- profiler_iterations: `50`
- profiler_scopes: `accepted_reference,candidate`
- correctness_command: `python3 auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/bi150/baseline_adapter.py --warmup 50 --repeat 100`
- benchmark_command: `python3 auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/bi150/baseline_adapter.py --warmup 50 --repeat 100`
- profiler_command: `python3 auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/bi150/baseline_adapter.py --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50`

Benchmark wall time controls adoption. Profiler data is attributable diagnostic
evidence and is normalized per forward call.

## Measurement Fingerprint

- measurement_fingerprint: `896adb91dbe5f84f9de83644e058462173cd5423a61bdf1ebcb2a15ca783c0be`
- base_sha256: `98be7d25ad94930fd9497e6ab33e64970c509c6c3569c8c3df40ae9d22b9a341`
- baseline_adapter_sha256: `433569bbac3bab158ff211a6de7ecb40ec7236d74a3eb7ab7c2b487e1b41772a`
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
- run_branch: `kernel-opt/music_flamingo_rotary_embedding-bi150-20260818`

These fields mirror `team-state.md` and identify the dedicated optimization
branch. The run branch is never `main`, `master`, or `dev`.

## Round Overview

| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 000 | Phase 0 | `baseline_adapter.py` | baseline | `base.py` | 0.353447 | 68.63642578125 | - | not-applicable | `baseline_adapter.py` |
| 001 | `rounds/decision_001.md` | `triton_music_flamingo_rotary_embedding_001.py` | accepted | `baseline_adapter.py` | 0.176121 | 30.829 | 48.63869398610698 | confirmed | `triton_music_flamingo_rotary_embedding_001.py` |
| 002 | `rounds/decision_002.md` | - | aborted | - | - | - | - | not-applicable | `triton_music_flamingo_rotary_embedding_001.py` |

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

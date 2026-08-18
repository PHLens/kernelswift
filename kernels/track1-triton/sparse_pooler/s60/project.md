# SPLADE Sparse Pooler Optimization Project (S60 / GCU)

## Project Identity

- schema_version: 1
- skill_version: 2.0.0
- project_root: `/root/kernelswift/.worktrees/sparse-pooler-s60/kernels/track1-triton/sparse_pooler/s60`
- base: `../base.py`
- baseline_adapter: `baseline_adapter.py`
- harness: `/root/kernelswift/.worktrees/sparse-pooler-s60/auto_bench.py`
- interpreter: `/usr/bin/python3` on the S60 host
- device: `gcu:0`
- implementation_language: `triton`
- implementation_backend: `gcu`
- target_profile: `triton_gcu`

## Semantics

- operator: `SPLADESparsePooler` — MLM head followed by SPLADE activation `log(1+relu(x))` and per-sequence pooling (max or sum) over the sequence axis.
- inputs:
    - `hidden_states`: `Tensor[83, 768]`, fp32, contiguous, on `gcu:0`
    - `seq_lens`: `Tensor[4]`, int32, on `gcu:0`; values `[20, 25, 18, 20]`, sum `83`
- outputs:
    - `list[Tensor[30522]]` of length 4, each fp32, on `gcu:0` — one pooled vector per sequence
- mathematical_behavior:
    - `x = decoder(LayerNorm(GELU(Dense(hidden_states))))` — MLM head: `dense` Linear(768→768), GELU, LayerNorm(eps=1e-12), `decoder` Linear(768→30522, bias=True)
    - `x = log1p(relu(x))` elementwise — SPLADE activation
    - For each sequence `i` with length `L_i`, pool `x[offset:offset+L_i]` over the sequence axis: `max(dim=0).values` when `pooling == "max"` (default), `sum(dim=0)` when `pooling == "sum"`; `offset` accumulates `L_i`
- tolerance_and_tie_rules: `torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True)`; output fp32
- public_contract:
    - `ModelNew(hidden_size=768, vocab_size=30522, pooling="max")`
    - `ModelNew.forward(hidden_states, seq_lens) -> list[Tensor]`
    - `get_init_inputs() -> [768, 30522, "max"]`
    - `get_inputs() -> [hidden_states, seq_lens]` (2 args)

## Invariants

- state_dict keys must be exactly `{dense.weight, dense.bias, layer_norm.weight, layer_norm.bias, decoder.weight, decoder.bias}` with shapes `[768,768]`/`[768]`/`[768]`/`[768]`/`[30522,768]`/`[30522]` respectively (nested submodule names `dense`/`act`/`layer_norm`/`decoder`). `compare_case` calls `load_state_dict` inside a silent try/except, so mismatched keys silently skip weight sync and cause a numeric FAIL with no direct hint.
- candidate `get_inputs()` must return exactly 2 arguments (used only for arg-count check); actual forward inputs come from `v1_inputs = clone_value(v0_inputs)`.
- output must be a Python `list` of 4 `Tensor[30522]` fp32 (harness `compare_values` recurses lists; a stacked tensor would fail the type/shape check).
- device literal `'cuda'` in source is rewritten to `'gcu'` by `_rewrite_device_for_backend` (target == gcu branch).
- module-level non-literal assignments are stripped by `_filter_module_ast`; any `fast_libentry()`-style wrapping must live inside a ClassDef body or forward, not at module top level.
- GELU must stay as the `nn.GELU()` library op (GCU may approximate to tanh; matching the library avoids erf/tanh mismatch against base on the same device).

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
- discovered_at: `2026-08-18T22:05:00Z`
- host: `5d02974bab32`

## Measurement Regime

- harness_path: `/root/kernelswift/.worktrees/sparse-pooler-s60/auto_bench.py`
- harness_sha256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- shape: `hidden_states=[83,768] fp32; seq_lens=[20,25,18,20] int32; output list 4x[30522] fp32`
- dtype: `fp32`
- device: `gcu:0`
- warmup: `50`
- repeat: `100`
- timing_order: `interleaved accepted-reference/candidate`
- primary_metric: `unrounded median wall_time_ms`
- profiler_iterations: `50`
- profiler_scopes: `baseline_base,candidate_triton_sparse_pooler_001`
- profiler_device_time: `unavailable on recorded GCU exporter; runtime_launch_* fields are retained`
- correctness_command: `cd /root/kernelswift/.worktrees/sparse-pooler-s60 && python3 auto_bench.py --v0_file kernels/track1-triton/sparse_pooler/base.py --v1_file kernels/track1-triton/sparse_pooler/s60/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback`
- benchmark_command: `cd /root/kernelswift/.worktrees/sparse-pooler-s60 && python3 auto_bench.py --v0_file kernels/track1-triton/sparse_pooler/base.py --v1_file kernels/track1-triton/sparse_pooler/s60/triton_sparse_pooler_001.py --warmup 50 --repeat 100`
- profiler_command: `cd /root/kernelswift/.worktrees/sparse-pooler-s60 && python3 auto_bench.py --v0_file kernels/track1-triton/sparse_pooler/base.py --v1_file kernels/track1-triton/sparse_pooler/s60/triton_sparse_pooler_001.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/sparse_pooler/s60/log/sparse_pooler_round_001_forward_50iter.pt.trace.json`

## Measurement Fingerprint

- measurement_fingerprint: `15ffdaf1e8fcc0a9b8b5af2a429e4ddad7c4e3ac67b345a9600d6cb8aa6bd226`
- base_sha256: `46106baa46b9f1952b32388d99880c967438ea99becdc874e3f68ba81a727d58`
- baseline_adapter_sha256: `359f4c808a0cf210416116322e4cc01f74ee42961b68c1fd365672af2a59bde8`
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
- run_branch: `kernel-opt/sparse-pooler-s60`

## Round Overview

| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 000 | Phase 0 | `baseline_adapter.py` | baseline | `base.py` | 0.862541 | unavailable: GCU runtime-launch-only | - | not-applicable | `baseline_adapter.py` |
| 001 | `rounds/decision_001.md` | `triton_sparse_pooler_001.py` | no-improvement | `baseline_adapter.py` | 1.092186 | unavailable: GCU runtime-launch-only | -26.79% (0.79x) | falsified (launch 11→6 but device slower) | `baseline_adapter.py` |
| 002 | `rounds/decision_002.md` | not-created | aborted | `baseline_adapter.py` | - | - | - | not-applicable (measurement-bound) | `baseline_adapter.py` |

## Reproduction

```bash
cd /root/kernelswift/.worktrees/sparse-pooler-s60
python3 auto_bench.py --v0_file kernels/track1-triton/sparse_pooler/base.py --v1_file kernels/track1-triton/sparse_pooler/s60/baseline_adapter.py --warmup 50 --repeat 100
```

```bash
cd /root/kernelswift/.worktrees/sparse-pooler-s60
python3 auto_bench.py --v0_file kernels/track1-triton/sparse_pooler/base.py --v1_file kernels/track1-triton/sparse_pooler/s60/baseline_adapter.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/sparse_pooler/s60/log/sparse_pooler_baseline_forward_50iter.pt.trace.json
```

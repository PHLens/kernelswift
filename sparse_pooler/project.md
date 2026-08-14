# SPLADE Sparse Pooler Optimization Project

## Project Identity

- schema_version: 1
- skill_version: 2.0.0
- project_root: `/projs/framework/lipenghui/projects/kernelswift/sparse_pooler`
- base: `base.py`
- baseline_adapter: `baseline_adapter.py`
- harness: `/projs/framework/lipenghui/projects/kernelswift/auto_bench.py`
- interpreter: `/projs/framework/lipenghui/venv/pytorch_main/bin/python3`
- device: `mlu:0 (MLU590-H8)`
- implementation_language: `triton`
- implementation_backend: `mlu`
- target_profile: `triton_mlu`

## Semantics

- operator: `SPLADESparsePooler` — MLM head followed by SPLADE activation
  `log(1+relu(x))` and per-sequence pooling (max or sum) over the vocabulary
  dimension.
- inputs:
  - `hidden_states`: `Tensor[total_seq, hidden_size]` with `hidden_size=768`,
    `dtype=fp32`, `layout=contiguous`, `device=mlu:0`. `total_seq` is the sum of
    all sequence lengths in the batch (ragged batch concatenation).
  - `seq_lens`: `Tensor[num_seq]` with `num_seq=4`, `dtype=int32`,
    `layout=contiguous`, `device=mlu:0`. Sum of `seq_lens` equals `total_seq`.
    Default `seq_lens = [20, 25, 18, 20]`, so `total_seq = 83`.
- outputs: `list[Tensor[vocab_size]]` of length `num_seq`, each
  `Tensor[30522]`, `dtype=fp32`, `device=mlu:0`. One pooled vector per input
  sequence.
- mathematical_behavior:
  - `x = decoder(LayerNorm(GELU(Dense(hidden_states))))` — MLM head: a linear
    `dense` (768→768), GELU activation, LayerNorm with `eps=1e-12`, then a
    linear `decoder` (768→30522, bias=True).
  - `x = log(1 + relu(x))` elementwise — SPLADE activation.
  - For each sequence `i` with length `L_i`, pool `x[offset:offset+L_i]` over
    the sequence axis: `max(dim=0).values` when `pooling == "max"` (default)
    or `sum(dim=0)` when `pooling == "sum"`. `offset` accumulates `L_i`.
- tolerance_and_tie_rules: harness `--atol 1e-2 --rtol 1e-2` (defaults);
  `torch.allclose(..., equal_nan=True)`. Max pooling tie behavior follows
    PyTorch's `max` (first occurrence wins on ties is not guaranteed; the
    candidate must match the reference value within tolerance).
- public_contract:
  - `ModelNew(hidden_size: int = 768, vocab_size: int = 30522, pooling: str = "max")`
  - `ModelNew.forward(hidden_states: Tensor, seq_lens: Tensor) -> list[Tensor]`
  - `get_init_inputs() -> [768, 30522, "max"]`
  - `get_inputs() -> [hidden_states(Tensor[83,768],fp32,mlu:0), seq_lens(Tensor[4],int32,mlu:0)]`

Unknown user-owned semantics must be resolved with the user. Do not infer them
from a candidate implementation.

## Invariants

- `base.py` is user-owned and immutable; no role edits, formats, replaces, or
  commits a generated version over it.
- The harness AST loader (`auto_bench.py:load_ks_module`) rewrites `'npu'`
  device strings to the detected accelerator and filters the module AST to keep
  only imports, class/function definitions, and safe-literal top-level
  assignments. Module-level non-literal assignments are dropped. Candidate
  modules must expose `ModelNew`, `get_inputs`, and `get_init_inputs` and must
  not rely on module-level side effects beyond safe literals.
- The harness compares v0 (`Model`) against v1 (`ModelNew`) and runs
  `model_new.load_state_dict(model.state_dict())` before timing; the candidate
  must accept the reference state dict or safely ignore mismatches.
- The harness detects the device from the model/inputs and moves everything to
  one device; all tensors and model parameters must land on the same `mlu:0`.
- Correctness is checked with `torch.allclose(atol=1e-2, rtol=1e-2,
  equal_nan=True)`; the candidate output is a Python `list` of `Tensor[30522]`.
- Benchmark wall time controls adoption. Profiler data is diagnostic and
  normalized per forward call. Reference and candidate profiler scopes are
  never combined.

The complete workflow-level rules are in `references/invariants.md`.

## Runtime Fingerprint

```yaml
triton_distribution: triton 3.2.0 (/projs/framework/lipenghui/venv/pytorch_main/lib/python3.10/site-packages/triton)
triton_version: 3.2.0
backend_target: BangDriver (mlu)
backend_version: torch_mlu 1.32.0+torch2.11.0; MLU driver 6.5.49
device_arch: MLU590-H8 (capability 5.0)
```

- target_profile_match: `pass`
- discovery_commands: `python3 -c "import triton; print(triton.__version__)"; python3 -c "import torch_mlu; print(torch_mlu.__version__); print(torch_mlu.get_driver_version())"; python3 -c "import torch,torch_mlu; print(torch.mlu.get_device_name(0)); print(torch.mlu.get_device_capability(0))"`
- discovered_at: `2026-08-14T13:30:00Z`

These values are observed in Phase 0. They are not assumed from the profile.

## Measurement Regime

- harness_path: `/projs/framework/lipenghui/projects/kernelswift/auto_bench.py`
- harness_sha256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- shape: `hidden_states=[83,768] fp32; seq_lens=[20,25,18,20] int32; output list of 4 × [30522] fp32`
- dtype: `fp32 (hidden_states, output), int32 (seq_lens)`
- device: `mlu:0`
- warmup: `50`
- repeat: `100`
- timing_order: `interleaved accepted-reference/candidate`
- primary_metric: `unrounded median wall_time_ms`
- profiler_iterations: `50`
- profiler_scopes: `accepted_reference,candidate`
- correctness_command: `python3 /projs/framework/lipenghui/projects/kernelswift/auto_bench.py --v0_file sparse_pooler/base.py --v1_file sparse_pooler/baseline_adapter.py --warmup 50 --repeat 100`
- benchmark_command: `python3 /projs/framework/lipenghui/projects/kernelswift/auto_bench.py --v0_file sparse_pooler/base.py --v1_file sparse_pooler/baseline_adapter.py --warmup 50 --repeat 100`
- profiler_command: `python3 /projs/framework/lipenghui/projects/kernelswift/auto_bench.py --v0_file sparse_pooler/base.py --v1_file sparse_pooler/baseline_adapter.py --profile --profile-reference-file sparse_pooler/baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output sparse_pooler/log/round_000_forward_50iter.pt.trace.json`

Benchmark wall time controls adoption. Profiler data is attributable diagnostic
evidence and is normalized per forward call.

## Measurement Fingerprint

- measurement_fingerprint: `a0208c7da7e371d45c88f82ebddd3850d01669aa5d912f31db9234a7a56ebab7`
- base_sha256: `ccccbbefadf1d697341451b542f17392acc8a2b9e4a3a41e50b2f9d58dbf61de`
- baseline_adapter_sha256: `d7e69ed4b66a4193a4475cc307fc0929cef807f875785652df6cc36fb2c487e5`
- fingerprint_command: `python3 -c "import hashlib,json; base=open('base.py','rb').read(); h=open('/projs/framework/lipenghui/projects/kernelswift/auto_bench.py','rb').read(); s=json.dumps({'shape':'hidden_states=[83,768] fp32; seq_lens=[20,25,18,20] int32; output list of 4 × [30522] fp32','dtype':'fp32 (hidden_states, output), int32 (seq_lens)','device':'mlu:0','warmup':50,'repeat':100,'profile_mode':'forward','profile_warmup':20,'profile_iterations':50},sort_keys=True,separators=(',',':')); print(hashlib.sha256(base+b'\x00'+h+b'\x00'+s.encode()).hexdigest())"`

A fingerprint change requires a new comparable baseline before optimization can
continue.

## Optional Target

- target_mode: `null`
- target_value: `null`
- target_metric: `wall_time_ms`
- source: `null`
- target_measurement_fingerprint: `null`

An optional target is comparable only when it uses `wall_time_ms` under the
baseline measurement fingerprint. It is not an estimated bound, device-time
goal, or inferred target. Orchestrator records later target amendments in the
append-only team-state policy-revision table at a safe terminal boundary.

## Git Run Identity

- base_branch: `dev`
- base_commit: `92c8f7f`
- run_branch: `kernel-opt/sparse_pooler-2`

These fields mirror `team-state.md` and identify the dedicated optimization
branch. The run branch is never `main`, `master`, or `dev`.

## Upbound

- kind: `estimated`
- source: `semantic analysis — a single fused Triton kernel that performs the MLM head matmul, GELU, LayerNorm, ReLU/log1p, and per-sequence max pooling in one device pass would reduce the current host-side Python loop (4 sequential max reductions over vocabulary) and the multiple PyTorch library kernels (dense matmul, GELU, LayerNorm, decoder matmul, relu, log1p, per-chunk max) to one launch. A 30–50% wall improvement is plausible but unverified; the upbound is not used as the adoption threshold.`
- regime_match: `same operator, same shapes, same device, same harness`
- wall_time_ms: `null`
- device_us_per_call: `null`
- limitations: `This is a semantic upper bound only, not a measured upbound. The 5% adoption threshold and stop criteria use measured wall time, not this estimate.`

## Round Overview

| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 000 | Phase 0 | `baseline_adapter.py` | baseline | `base.py` | 0.909974 | 180.05 | - | not-applicable | `baseline_adapter.py` |
| 001 | `rounds/decision_001.md` | `triton_sparse_pooler_001.py` | accepted | `baseline_adapter.py` | 0.606758 | 210.12 | 33.39 | partially-confirmed | `triton_sparse_pooler_001.py` |
| 002 | `rounds/decision_002.md` | `triton_sparse_pooler_002.py` | no-improvement | `triton_sparse_pooler_001.py` | 0.621848 | 213.02 | 0.65 | falsified | `triton_sparse_pooler_001.py` |

Orchestrator appends one row only after a terminal round transition is validated
and committed. Rejected candidates remain listed but never become the comparison
source.

## Reproduction

```bash
python3 /projs/framework/lipenghui/projects/kernelswift/auto_bench.py \
  --v0_file sparse_pooler/base.py \
  --v1_file sparse_pooler/baseline_adapter.py \
  --warmup 50 --repeat 100
```

```bash
python3 /projs/framework/lipenghui/projects/kernelswift/auto_bench.py \
  --v0_file sparse_pooler/base.py \
  --v1_file sparse_pooler/baseline_adapter.py \
  --profile --profile-reference-file sparse_pooler/baseline_adapter.py \
  --profile-mode forward --profile-warmup 20 --profile-iterations 50 \
  --profile-output sparse_pooler/log/round_000_forward_50iter.pt.trace.json
```

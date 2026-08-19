# Fused MoE Optimization Project

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

- operator: `fused_moe` — Mixture of Experts: softmax router over expert logits,
  top-k gating, per-expert gate/up + down GEMM with SiLU activation, and a
  weighted reduction across the selected experts.

- inputs (2 tensors, produced by `get_inputs()`):
  - `hidden_states`: `[83, 128]` `float16`, contiguous, on the caller-selected
    accelerator (`cuda:0`). `num_tokens=83`, `hidden_size=128`.
  - `router_logits`: `[83, 8]` `float32`, on the caller-selected accelerator.
    `num_experts=8`.

- parameters (module `nn.Parameter`s, normal init `std=0.02`, `.to(dtype)` to
  fp16 at forward time):
  - `w1`: `[8, 128, 128]` = `[E, 2*intermediate_size, hidden_size]`
    (`2*intermediate=128`), the fused gate+up projection.
  - `w2`: `[8, 128, 64]` = `[E, hidden_size, intermediate_size]`, the down
    projection.

- outputs: a single tensor `out` `[83, 128]` `float16`, on the same device as
  the inputs.

- constructor parameters (`get_init_inputs()` returns `[8, 2, 128, 64]`):
  `num_experts=8`, `top_k=2`, `hidden_size=128`, `intermediate_size=64`,
  `renormalize=True`.

- mathematical_behavior:
  1. routing: `scores = torch.softmax(router_logits.float(), dim=-1)` →
     `[83, 8]` (computed in fp32).
  2. `topk_weights, topk_ids = torch.topk(scores, 2, dim=-1)` → each `[83, 2]`.
     `topk_weights` is sorted descending by value; `topk_ids` holds the
     corresponding expert indices. **Tie semantics are sensitive**: for equal
     score values, `torch.topk` selects the smaller index first (ties broken by
     ascending index order). This is a correctness-critical detail inherited
     from the grouped-topk lesson.
  3. renormalize: `topk_weights = topk_weights / topk_weights.sum(-1,
     keepdim=True)` (because `renormalize=True`) → `[83, 2]`, then cast
     `.to(dtype)` to fp16.
  4. flatten/dispatch: `flat_ids = topk_ids.view(-1)` → `[166]`;
     `flat_w = topk_weights.view(-1)` → `[166]`;
     `x_rep = hidden_states.unsqueeze(1).expand(-1, 2, -1).reshape(-1, 128)` →
     `[166, 128]` (each token row duplicated `top_k=2` times; `166 = 83*2`).
  5. per-expert loop over `e in range(8)`: `mask = flat_ids == e` selects
     `n_e` rows; for each expert `e`:
     - gate/up GEMM: `gate_up = x_e[n_e,128] @ w1[e].T[128,128]` → `[n_e, 128]`
       (contraction dim 128 = hidden).
     - `gate, up = gate_up.chunk(2, dim=-1)` → each `[n_e, 64]`.
     - `act = F.silu(gate) * up` → `[n_e, 64]` (SiLU activation, `x*sigmoid(x)`).
     - down GEMM: `expert_out[mask] = act[n_e,64] @ w2[e].T[64,128]` →
       `[n_e, 128]` (contraction dim 64 = intermediate).
     Experts with no assigned token are skipped (`if not mask.any(): continue`).
  6. weighted reduction: `expert_out = expert_out[166,128] *
     flat_w.unsqueeze(-1)[166,1]`, then
     `out = expert_out.view(83, 2, 128).sum(dim=1)` → `[83, 128]` (sum of the
     top-2 weighted expert outputs).

- tolerance_and_tie_rules: the harness `compare_values` compares the single
  fp16 output with `torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True)`
  (harness defaults `--atol 1e-2 --rtol 1e-2`). Output structure (single tensor),
  shape `[83, 128]`, and dtype `float16` must match exactly. The routing is done
  in fp32 while the weights and final output are fp16; the top-k tie order is
  `torch.topk` default (largest values, ties broken by ascending index).

- public_contract:
  - `ModelNew(num_experts, top_k, hidden_size, intermediate_size,
    renormalize=True)`.
  - `forward(hidden_states, router_logits) -> out[83, 128]`.
  - module-level `get_init_inputs() -> [8, 2, 128, 64]`.
  - module-level `get_inputs() -> [hidden_states, router_logits]`.

Unknown user-owned semantics must be resolved with the user. Do not infer them
from a candidate implementation.

## Invariants

- Semantic invariant: `base.py` is user-owned and immutable; no role edits,
  formats, or replaces it. The public constructor (`ModelNew(num_experts=8,
  top_k=2, hidden_size=128, intermediate_size=64, renormalize=True)`), the
  forward signature `(hidden_states, router_logits) -> out`, the output shape
  `[83, 128]` fp16, and the routing/GEMM/activation/reduction semantics above
  must remain numerically compatible with the reference.
- Tie-rule invariant: `torch.topk(scores, 2, dim=-1)` tie ordering (descending
  value, ties broken by ascending index) is a correctness-critical semantic. A
  Triton reimplementation must reproduce the same selected expert indices for
  equal score values; this mirrors the grouped-topk lesson.
- Environment invariant: on the BI150 host a fresh shell must set
  `export COREX_VERSION=4.4.0` and source `/usr/local/corex/enable` before
  importing `torch` or `triton`; without that bootstrap, imports and `ixsmi`
  fail. The Triton active compiler backend is `cuda` on the CoreX environment.
- Lifecycle invariant: candidate execution preserves caller-selected device and
  current stream; `forward` does not mutate its inputs; any output-buffer reuse
  must have explicit per-instance ownership, compatibility keys including
  shape/dtype/device, invalidation, aliasing, and concurrency semantics.
- Measurement invariant: the harness seeds each side identically, clones inputs,
  replaces candidate inputs with a clone of the reference inputs, runs under
  `torch.no_grad()`, and compares candidate output against the reference with
  `atol=1e-2, rtol=1e-2, equal_nan=True`. Benchmark wall time (unrounded median)
  controls adoption. A change to the measurement fingerprint requires a new
  comparable baseline.

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
- discovery_commands: `export COREX_VERSION=4.4.0; . /usr/local/corex/enable; python3 -c "import torch,triton; print(torch.__version__, triton.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"`
- discovered_at: `2026-08-19T19:45:00Z`

These values are observed in Phase 0. They are not assumed from the profile.

## Measurement Regime

- harness_path: `/root/CodeBuddy/20260818191200/kernelswift/auto_bench.py`
- harness_sha256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- shape: `hidden_states[83,128] fp16; router_logits[83,8] fp32; output [83,128] fp16`
- dtype: `fp16 (hidden_states, out); fp32 (router_logits)`
- device: `cuda:0 (Iluvatar BI-V150)`
- warmup: `50`
- repeat: `100`
- timing_order: `interleaved accepted-reference/candidate`
- primary_metric: `unrounded median wall_time_ms`
- profiler_iterations: `50`
- profiler_scopes: `accepted_reference,candidate`
- correctness_command: `python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/bi150/baseline_adapter.py --warmup 50 --repeat 100`
- benchmark_command: `python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/bi150/baseline_adapter.py --warmup 50 --repeat 100`
- profiler_command: `python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/bi150/baseline_adapter.py --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50`

Benchmark wall time controls adoption. Profiler data is attributable diagnostic
evidence and is normalized per forward call.

## Measurement Fingerprint

- measurement_fingerprint: `5c2a51ab3f3ebaab1123b9fa534d4e4b940f3334f80fac00252df780d3900150`
- base_sha256: `a0269ac15833098c11c63045318e829e404b09ca49ae8dce22b244e0d2894d2b`
- baseline_adapter_sha256: `8e5c70232e541a02d83343216376ece9127a1c3e6ea6af77dc77a2723783facf`
- fingerprint_command: `sha256sum(base.py); sha256sum(baseline_adapter.py); python3 -c "import hashlib,json; print(hashlib.sha256(open("base.py","rb").read()+b" "+open("auto_bench.py","rb").read()+b" "+json.dumps(settings,sort_keys=True,separators=(",",":")).encode()).hexdigest())"`

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
- base_commit: `453590c`
- run_branch: `kernel-opt/fused_moe-bi150-<run-epoch-or-timestamp>`

These fields mirror `team-state.md` and identify the dedicated optimization
branch. The run branch is never `main`, `master`, or `dev`.

## Round Overview

| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 000 | Phase 0 | `baseline_adapter.py` | baseline | `base.py` | 3.258671 | 968.1624609375 | - | not-applicable | `baseline_adapter.py` |
| 001 | `rounds/decision_001.md` | `triton_fused_moe_001.py` | accepted | `baseline_adapter.py` | 2.488731 | 504.312 | 21.44 | confirmed | `triton_fused_moe_001.py` |
| 002 | `rounds/decision_002.md` | `triton_fused_moe_002.py` | accepted | `triton_fused_moe_001.py` | 0.493474 | 140.84 | 79.98 | confirmed | `triton_fused_moe_002.py` |
| 003 | `rounds/decision_003.md` | - | aborted | - | - | - | - | not-applicable | `triton_fused_moe_002.py` |

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

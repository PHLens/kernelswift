# MHC Head Compute Mix Optimization Project

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

- operator: `mhc_head_compute_mix` — a multi-head-combined (MHC) head compute
  mix that produces a gating `pre`, a modulated `post`, and a Sinkhorn-normalized
  doubly-stochastic `comb` matrix from a single combined `mixes` input, two
  affine scales, and a base offset vector. The core is a Sinkhorn iteration
  (row/column alternating normalization) applied to an exponentiated
  (log-)softmax-style coupling matrix.

- inputs: three tensors, all float32 on the caller-selected accelerator.
  - `mixes`: `[2, 8, 24]` float32. `b=2`, `s=8`,
    `mix_hc=24=(2+hc)*hc=(2+4)*4`, so the last dim holds the concatenation of
    the `pre` slice `[0:4]`, the `post` slice `[4:8]`, and the `comb` raw slice
    `[8:24]` (viewed as `[4,4]`).
  - `hc_scale`: `[3]` float32, values `s0=0.5`, `s1=0.25`, `s2=1.0`.
  - `hc_base`: `[24]` float32 (the same `mix_hc=24` length as `mixes`'s last
    dim), produced in `get_inputs()` as `torch.randn(24) * 0.1`.
  `get_inputs()` returns the list `[mixes, hc_scale, hc_base]`. No integer or
  non-tensor inputs are present.

- outputs: a 3-tuple of float32 tensors returned by `forward`:
  - `pre`: `[2, 8, 4]` float32 (sigmoid-gated, `+ eps`).
  - `post`: `[2, 8, 4]` float32 (`2 * sigmoid`).
  - `comb`: `[2, 8, 4, 4]` float32 (Sinkhorn-normalized doubly-stochastic
    matrix per (b, s) position).

- mathematical_behavior: `hc = self.hc_mult = 4`; `eps = self.eps = 1e-6`;
  `expected = (2 + hc) * hc = 24`. `forward` validates `mix_hc == expected`
  (raises `ValueError` otherwise). Then `x = mixes.reshape(-1, 24).float()`
  (`N = b*s = 16`), `base = hc_base.float()`, and
  `s0, s1, s2 = hc_scale[0], hc_scale[1], hc_scale[2]`.
  - `pre = torch.sigmoid(x[:, :4] * s0 + base[:4].unsqueeze(0)) + eps` → `[N,4]`.
    `eps` is added to the sigmoid result (not to the denominator).
  - `post = 2 * torch.sigmoid(x[:, 4:8] * s1 + base[4:8].unsqueeze(0))` → `[N,4]`.
  - `raw = x[:, 8:24]`; `comb = raw.view(-1,4,4) * s2 + base[8:24].view(1,4,4)`
    → `[N,4,4]`.
  - `row_max = comb.amax(dim=-1, keepdim=True)`;
    `comb = torch.exp(comb - row_max)` (numerically stable softmax).
  - First explicit normalization pair:
    `comb = comb / comb.sum(dim=-1, keepdim=True) + eps` (row-normalize, then
    add `eps` to the normalized matrix, NOT to the denominator), then
    `comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)` (column-normalize,
    with `eps` added to the column-sum denominator).
  - Loop `for _ in range(self.sinkhorn_iters - 1)` (i.e. 19 iterations), each
    iteration performing exactly the same pair:
    `comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)` (row-normalize) then
    `comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)` (column-normalize).
    NOTE: inside the loop the row-normalize denominator also adds `eps`, unlike
    the first explicit row-normalize which adds `eps` to the numerator (result)
    instead.
  - Total Sinkhorn normalization is `sinkhorn_iters = 20` rounds: 1 explicit
    row+column pair followed by 19 looped row+column pairs, each round
    alternating row then column normalization. `eps=1e-6` is added either to
    the normalized matrix or to the sum denominator (see the exact placement
    above), acting as a floor to avoid division by zero / numerical instability.
  - Final reshape: `return pre.view(b,s,hc), post.view(b,s,hc), comb.view(b,s,hc,hc)`.

- dtype: all computation and all outputs are float32. `mixes` and `hc_base`
  are explicitly cast `.to(torch.float32)` in `forward` (they already are fp32
  in `get_inputs()`).

- tolerance_and_tie_rules: the harness compares candidate output against the
  reference recursively (tuple → tensors) with
  `torch.allclose(..., atol=1e-2, rtol=1e-2, equal_nan=True)` (harness defaults
  `--atol 1e-2 --rtol 1e-2`). All three outputs are floating point, so the
  allclose path applies; there are no integer outputs and no tie-break rule.
  Output structure, shapes, and dtype must match exactly.

- public_contract: the candidate module must expose `ModelNew`,
  `get_init_inputs`, and `get_inputs`.
  `ModelNew.__init__(hc_mult: int = 4, sinkhorn_iters: int = 20, eps: float = 1e-6)`
  and `forward(mixes: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor)
  -> tuple[Tensor, Tensor, Tensor]` must remain compatible. `forward` does not
  mutate its inputs and preserves the caller-selected device/current stream.
  `get_init_inputs()` returns `[4, 20, 1e-6]` (hc_mult, sinkhorn_iters, eps).
  `get_inputs()` returns `[mixes, hc_scale, hc_base]` with `mixes` a
  `torch.randn(2, 8, 24, dtype=torch.float32, device="cuda")`,
  `hc_scale = torch.tensor([0.5, 0.25, 1.0], dtype=torch.float32, device="cuda")`,
  and `hc_base = torch.randn(24, dtype=torch.float32, device="cuda") * 0.1`,
  all seeded by `torch.manual_seed(0)`.

Unknown user-owned semantics must be resolved with the user. Do not infer them
from a candidate implementation.

## Invariants

- Semantic invariant: `base.py` is user-owned and immutable; no role edits it.
  The public constructor (`ModelNew(hc_mult=4, sinkhorn_iters=20, eps=1e-6)`),
  forward signature `(mixes, hc_scale, hc_base) -> (pre, post, comb)`, output
  shapes `pre/post [2,8,4]`, `comb [2,8,4,4]`, all-fp32 dtype, and the
  Sinkhorn normalization count (`sinkhorn_iters=20` rounds of alternating
  row/column normalization with `eps=1e-6` floor) must remain numerically
  compatible with the reference.
- Environment invariant: on the BI150 host a fresh shell must set
  `export COREX_VERSION=4.4.0` and source `/usr/local/corex/enable` before
  importing `torch` or `triton`; without that bootstrap, imports and `ixsmi`
  fail. The Triton active compiler backend is `cuda` on the CoreX environment.
- Lifecycle invariant: candidate execution preserves caller-selected device and
  current stream; `forward` does not mutate inputs; any output-buffer reuse must
  have explicit per-instance ownership, compatibility keys including
  shape/dtype/device, invalidation, aliasing, and concurrency semantics.
- Measurement invariant: the harness seeds each side identically, clones
  inputs, replaces candidate inputs with a clone of the reference inputs, runs
  under `torch.no_grad()`, and compares candidate outputs recursively against
  the reference with `atol=1e-2, rtol=1e-2, equal_nan=True`. Benchmark wall time
  (unrounded median) controls adoption. A change to the measurement fingerprint
  requires a new comparable baseline.

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
- discovered_at: `2026-08-19T10:00:00Z`

These values are observed in Phase 0. They are not assumed from the profile.

## Measurement Regime

- harness_path: `/root/CodeBuddy/20260818191200/kernelswift/auto_bench.py`
- harness_sha256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- shape: `mixes[2,8,24] fp32; hc_scale[3] fp32; hc_base[24] fp32; output (pre[2,8,4], post[2,8,4], comb[2,8,4,4]) fp32`
- dtype: `fp32 (mixes, hc_scale, hc_base, pre, post, comb)`
- device: `cuda:0 (Iluvatar BI-V150)`
- warmup: `50`
- repeat: `100`
- timing_order: `interleaved accepted-reference/candidate`
- primary_metric: `unrounded median wall_time_ms`
- profiler_iterations: `50`
- profiler_scopes: `accepted_reference,candidate`
- correctness_command: `python3 auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix/bi150/baseline_adapter.py --warmup 50 --repeat 100`
- benchmark_command: `python3 auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix/bi150/baseline_adapter.py --warmup 50 --repeat 100`
- profiler_command: `python3 auto_bench.py --v0_file kernels/track1-triton/mhc_head_compute_mix/base.py --v1_file kernels/track1-triton/mhc_head_compute_mix/bi150/baseline_adapter.py --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50`

Benchmark wall time controls adoption. Profiler data is attributable diagnostic
evidence and is normalized per forward call.

## Measurement Fingerprint

- measurement_fingerprint: `4e4f0575e2251810e9be7667c98eb4923c6910787d4412abc2c4a976c2b26a8e`
- base_sha256: `4c5167f6cfe9099786f204fad222f3b780d3ebb814494b9582112422abf84ac5`
- baseline_adapter_sha256: `ceebdc6185de4c980156a7833073678a0964fb7ccb5edf74b42be6156652eaed`
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
- run_branch: `kernel-opt/mhc_head_compute_mix-bi150-20260818`

These fields mirror `team-state.md` and identify the dedicated optimization
branch. The run branch is never `main`, `master`, or `dev`.

## Round Overview

| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 000 | Phase 0 | `baseline_adapter.py` | baseline | `base.py` | 1.517299 | 926.3953515625 | - | not-applicable | `baseline_adapter.py` |
| 001 | `rounds/decision_001.md` | `triton_mhc_head_compute_mix_001.py` | accepted | `baseline_adapter.py` | 0.183889 | 12.996 | 87.16869672492618 | confirmed | `triton_mhc_head_compute_mix_001.py` |
| 002 | `rounds/decision_002.md` | - | aborted | - | - | - | - | not-applicable | `triton_mhc_head_compute_mix_001.py` |

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

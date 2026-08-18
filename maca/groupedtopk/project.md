# Grouped TopK C500 Optimization Project

## Project Identity

- schema_version: 1
- skill_version: 2.0.0
- project_root: `/home/phlens/kernelswift/.worktrees/grouptopk-c500-20260818/maca/groupedtopk`
- remote_project_root: `/data/kernelswift-c500/maca/groupedtopk`
- base: `base.py`
- baseline_adapter: `baseline_adapter.py`
- harness: `/home/phlens/kernelswift/.worktrees/grouptopk-c500-20260818/auto_bench.py`
- remote_harness: `/data/kernelswift-c500/auto_bench.py`
- interpreter: `/opt/conda/bin/python` on the C500 host
- remote_runner: `/data/kernelswift-c500/c500_run.sh`
- device: `cuda:0` (MetaX C500 through the MACA compatibility surface)
- implementation_language: `triton`
- implementation_backend: `maca`
- target_profile: `triton_maca`

## Semantics

- operator: Grouped Top-K expert routing: score experts, select the
  highest-scoring expert groups, then select experts only from those groups.
- inputs: `hidden_states` is a contiguous `(T, H)` tensor on the
  caller-selected accelerator (benchmark: `(83, 7168)`, `float16`,
  `cuda:0`); only its leading dimension participates in the reference
  computation. `gating_output` is a contiguous `(T, E)` tensor on that
  accelerator (benchmark: `(83, 256)`, `float32`, `cuda:0`). `forward`
  asserts equal token counts; the group reshape requires `E` to be divisible
  by `num_expert_group` (benchmark: 32 experts per group).
- outputs: A two-tensor tuple `(topk_weights, topk_ids)`, each with shape
  `(T, topk)` (benchmark: `(83, 8)`), contiguous and on the input
  accelerator. `topk_weights` is `float32`; `topk_ids` is `int32` and
  contains global expert indices.
- mathematical_behavior: Apply `softmax(gating_output, dim=-1)` when
  `scoring_func == "softmax"`, or elementwise sigmoid when it is `"sigmoid"`;
  reject any other value. Partition each token's experts into
  `num_expert_group` contiguous expert-index groups, score each group by its
  maximum expert score, retain the `topk_group` highest-scoring groups, mask
  all experts outside them to `-inf`, and take the `topk` highest remaining
  expert scores and global indices. If `renormalize` is true, divide selected
  weights by their per-token sum; then multiply by `routed_scaling_factor`
  when it differs from 1.0. The benchmark constructor is
  `(topk=8, renormalize=True, num_expert_group=8, topk_group=4,
  scoring_func="softmax", routed_scaling_factor=1.0)`.
- tolerance_and_tie_rules: The harness requires the same tuple structure,
  tensor shapes, and dtypes. Floating outputs use
  `torch.allclose(..., atol=1e-2, rtol=1e-2, equal_nan=True)`; integer IDs use
  exact `torch.equal`. `torch.topk` supplies descending selected-value order,
  but `base.py` defines no extra equal-value tie-break rule and PyTorch does
  not promise stable tied indices; candidates must match the reference IDs for
  evaluated inputs rather than assume lowest- or highest-index ties.
- public_contract: The candidate module must expose `ModelNew`,
  `get_init_inputs`, and `get_inputs`.
  `ModelNew.__init__(topk: int, renormalize: bool, num_expert_group: int,
  topk_group: int, scoring_func: str = "softmax",
  routed_scaling_factor: float = 1.0)` and
  `forward(hidden_states: torch.Tensor, gating_output: torch.Tensor) ->
  tuple[torch.Tensor, torch.Tensor]` must remain compatible; forward does not
  mutate either input and preserves the caller-selected device/current stream.

Unknown user-owned semantics must be resolved with the user. Do not infer them
from a candidate implementation.

## Invariants

- `base.py` and `auto_bench.py` are immutable after Phase 0 begins.
- Candidate code uses the CUDA-compatible PyTorch surface while Triton's active compiler backend remains MACA.
- The actual harness AST loader, measurement regime, device, and stream behavior remain unchanged.
- Expert groups remain contiguous expert-index blocks, group selection precedes
  expert selection, renormalization occurs before routed scaling, and the fixed
  benchmark uses 8 selected experts from 4 of 8 groups.
- The token-count assertion, supported scoring-function behavior, output
  tuple/shape/dtype/device contract, and non-mutation of inputs remain compatible
  with `base.py`.
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
mx_driver_version: 3.8.30
backend_target: GPUTarget(backend='maca', arch=80, warp_size=64)
backend_version: 3.5.3.9
device_name: MetaX C500
device_arch: capability=8.0, triton_arch=80, warp_size=64
device_memory: 65536 MiB
```

- target_profile_match: `pass`
- discovery_commands: `/opt/conda/bin/python -c 'import torch, triton; ...'`; `/opt/conda/bin/python /data/maca_runtime_probe.py`; `mx-smi`
- discovered_at: `2026-08-18T05:01:57Z`
- remote_host: `d46a1e74eb76`
- environment_requirement: `MACA_PATH=/opt/maca` must be set before importing Triton; the recorded runner sources `/root/.profile`

## Measurement Regime

- harness_path: `/home/phlens/kernelswift/.worktrees/grouptopk-c500-20260818/auto_bench.py`
- remote_harness_path: `/data/kernelswift-c500/auto_bench.py`
- harness_sha256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- shape: `T=83,E=256,hidden=7168,topk=8,num_expert_group=8,topk_group=4`
- dtype: `hidden_states=fp16,gating_output=fp32,weights=fp32,ids=int32`
- device: `cuda:0`
- seed: `42`
- atol: `1e-2`
- rtol: `1e-2`
- warmup: `200`
- repeat: `500`
- timing_order: `sequential complete accepted-reference block, then complete candidate block`
- primary_metric: `unrounded median wall_time_ms`
- profile_mode: `forward`
- profiler_warmup: `20`
- profiler_iterations: `100`
- profiler_scopes: `accepted_reference,candidate`
- correctness_command: `/data/kernelswift-c500/c500_run.sh /opt/conda/bin/python auto_bench.py --v0_file maca/groupedtopk/base.py --v1_file maca/groupedtopk/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback`
- benchmark_command: `/data/kernelswift-c500/c500_run.sh /opt/conda/bin/python auto_bench.py --v0_file maca/groupedtopk/base.py --v1_file maca/groupedtopk/baseline_adapter.py --warmup 200 --repeat 500`
- profiler_command: `/data/kernelswift-c500/c500_run.sh /opt/conda/bin/python auto_bench.py --v0_file maca/groupedtopk/base.py --v1_file maca/groupedtopk/baseline_adapter.py --warmup 200 --repeat 500 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 100 --profile-output maca/groupedtopk/log/round_000_forward_100iter.pt.trace.json`

Benchmark wall time controls adoption. Profiler data is attributable diagnostic
evidence and is normalized per forward call.

## Measurement Fingerprint

- measurement_fingerprint: `3fe7d50260b3670756d26f003427faccb76e1e79c204b0a500a62d2eb481c809`
- base_sha256: `49ec0cf7a48679c23c5187eb7ba546fd41025019a9d6e9b59e1273c48a31dfbb`
- base_bytes: `2446`
- harness_sha256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- harness_bytes: `26142`
- baseline_adapter_sha256: `d92a1be10bf9ad036287ed0f769b195262132893fa78d88fa30f992fbe757827`
- fingerprint_command: `SHA-256(base bytes || NUL || harness bytes || NUL || canonical JSON settings with sort_keys=True and separators=(',', ':'))`

## Optional Target

- target_mode: `null`
- target_value: `null`
- target_metric: `wall_time_ms`
- source: user
- target_measurement_fingerprint: `null`

## Git Run Identity

- base_branch: `dev`
- base_commit: `6a970c921dfb0c031b885190122ce1335d8d4cd7`
- run_branch: `kernel-opt/grouptopk-c500-20260818`

## Round Overview

| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |
|---:|---|---|---|---|---:|---:|---:|---|---|
| 000 | Phase 0 | `baseline_adapter.py` | baseline | `base.py` | `0.231739` | `147.7526708984375` | - | not-applicable | `baseline_adapter.py` |
| 001 | `rounds/decision_001.md` | `triton_grouped_topk_001.py` | accepted | `baseline_adapter.py` | `0.068280` | `10.7442822265625` | `69.59021613749428%` | confirmed | `triton_grouped_topk_001.py` |
| 002 | `rounds/decision_002.md` | `triton_grouped_topk_002.py` | no-improvement | `triton_grouped_topk_001.py` | `0.081513` | not-run | `-13.711567434852972%` | partially-confirmed; wall-time claim falsified | `triton_grouped_topk_001.py` |

## Reproduction

```bash
/data/kernelswift-c500/c500_run.sh /opt/conda/bin/python auto_bench.py --v0_file maca/groupedtopk/base.py --v1_file maca/groupedtopk/baseline_adapter.py --warmup 200 --repeat 500
```

```bash
/data/kernelswift-c500/c500_run.sh /opt/conda/bin/python auto_bench.py --v0_file maca/groupedtopk/base.py --v1_file maca/groupedtopk/baseline_adapter.py --warmup 200 --repeat 500 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 100 --profile-output maca/groupedtopk/log/round_000_forward_100iter.pt.trace.json
```

The remote executable copy is `/data/kernelswift-c500`; the local project and
harness above are the source-controlled copies.

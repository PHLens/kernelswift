# FlexAttention Optimization Project (Epoch 2 of lineage; epoch-1 archive preserved intact at ../ [v2, measurement-bound abort])

## Project Identity

- schema_version: 1
- skill_version: 3.0.0
- contract_version: 3
- semantic_contract: typed-sketch-v1
- attribution_contract: verdict-v1
- project_root: `/root/CodeBuddy/20260818191200/kernelswift/kernels/track1-triton/flexattention/bi150/epoch2`
- base: `../../base.py`
- base_sha256: `dd1359ad88a5d5aae48a115023e5ae9016e4d5b5e6a12781273703e8d5c6a6d0` (2479 bytes)
- harness: `/root/CodeBuddy/20260818191200/kernelswift/auto_bench.py`
- harness_sha256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (29428 bytes)
- interpreter: `/usr/local/bin/python3`
- device: `cuda:0 (Iluvatar BI-V150)`
- implementation_language: `triton`
- implementation_backend: `cuda`
- target_id: `bi150`
- target_profile: `triton_cuda`
- target_profile_snapshot_ref: `profile_snapshot/triton_cuda.yaml`
- target_profile_snapshot_sha256: `dc8fa4c0c73caecc78c6a886e5ffb18cb97371f3add7cb382dbe30887743b7ae`
- project_capability_claim_ref: `profile_snapshot/capability_claim.json`
- project_capability_claim_sha256: `07aa5d489acb9c21717032087812d264dd5170fe79e7ea2326edb04cab657c1d`
- prior_lineage: `epoch-1 naive causal deliverable, no local artifacts (matrix: 0.61x); sibling campaign ../groupedtopk/bi150-round2 landed manual-cuda-graph architecture + triton_cuda profile promotion`
- base_branch: `kernel-opt/round2-bi150-20260827 @fa095a4` (stacked lineage carries triton_cuda profile v1)
- run_branch: `kernel-opt/flexattention-e2-20260828`

## Semantics

- operator: causal scaled-dot-product attention (MHA, num_kv_heads == num_heads == 8)
- inputs: `query/key/value [83, 8, 64] fp16` contiguous on cuda:0
- outputs: `[83, 512] fp16` reshaped concatenation of per-head outputs
- mathematical_behavior: scale=1/8 → QK^T causal-masked softmax → ·V, per head
- tolerance_and_tie_rules: allclose(atol=1e-2, rtol=1e-2) on fp16 out; NO integer-id
  tie surface here (dense attention, selection-free) — numerics dominated by exp/accum order
- public_contract: `ModelNew(num_heads, head_size, scale=None, num_kv_heads=8).
  forward(query, key, value)` and a `run_out(query,key,value,out)` preallocated-output
  surface MUST be provided for kernel-mode profiling (binding requirement inherited).

## Invariants

- `../../base.py` bytes unchanged after adapter generation.
- Harness loaded through AST loader only.
- CoreX bootstrap before any torch/triton use:
  `export COREX_VERSION=4.4.0; . /usr/local/corex/enable`.
- Candidate shape/dtype/device/semantics/public contract remain base-compatible.
- Wall time measured by unchanged harness incl. seed + CUDA sync.
- tl.dot constrained statuses apply verbatim from frozen snapshot: proven (32,32)@
  (32,32) fp32-acc; LARGER TILE SHAPES REQUIRED BY ATTENTION ARE UNPROVEN — any
  candidate relying on them needs Decision-scoped capability probes BEFORE candidate-ready.

## Runtime Fingerprint

```yaml
triton 3.1.0 (/usr/local/corex-4.4.0/lib64/python3/dist-packages/triton) / torch 2.7.1 / CoreX 4.4.0 nvcc V10.2.89 / Iluvatar BI-V150 capability major=7 minor=1 multi_processor_count=16 total_memory=17179869184
```

- discovered_at: `2026-08-28T04:00:00Z` (host identical to groupedtopk e2 campaign box)
- target_profile_match: `pass`

## Measurement Regime

- measurement_settings_canonical_json: `{"device":"cuda:0","dtype":"fp16-all","profile_iterations":100,"profile_mode":"kernel","profile_warmup":20,"repeat":100,"shape":{"key":[83,8,64],"query":[83,8,64],"value":[83,8,64]},"warmup":50}`
- measurement_fingerprint: `6dc07009177b649f7c2cad8f7be5e9aad74235bd9f50abfebc88bdb273e32af4`
- fingerprint_definition: `sha256(base ‖ NUL ‖ harness ‖ NUL ‖ canonical_json_settings)`

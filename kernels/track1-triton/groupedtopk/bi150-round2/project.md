# Grouped TopK Optimization Project (Epoch 2)

## Project Identity

- schema_version: 1
- skill_version: 3.0.0
- contract_version: 3
- semantic_contract: typed-sketch-v1
- attribution_contract: verdict-v1
- project_root: `/root/CodeBuddy/20260818191200/kernelswift/kernels/track1-triton/groupedtopk/bi150-round2`
- base: `../base.py` (shared device-neutral reference, bytes immutable)
- base_sha256: `12f3324896d6b72bd4eac556839db53fb7045b2965b7f8caf2734551daad0f58` (3541 bytes)
- harness: `/root/CodeBuddy/20260818191200/kernelswift/auto_bench.py` (AST loader; direct import insufficient)
- harness_sha256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (29428 bytes)
- interpreter: `/usr/local/bin/python3`
- device: `cuda:0 (Iluvatar BI-V150)`
- implementation_language: `triton`
- implementation_backend: `cuda`
- target_id: `bi150`
- target_profile: `triton_cuda`
- target_profile_snapshot_ref: `profile_snapshot/triton_cuda.md`
- target_profile_snapshot_sha256: `8b9cb9836c4abf97141081288d9eb68af7a571309057181e5ec1914827249a2f`
- project_capability_claim_ref: `profile_snapshot/capability_claim.json`
- project_capability_claim_sha256: `bc50f7f974f025e6be49d611e2546b6db6426d0761b794001898482f80f91371`
- prior_lineage: `../bi150 (v2 campaign, contract_version 2, read-only history)`
- base_branch: `dev @389053e`
- run_branch: `kernel-opt/round2-bi150-20260827`

## Semantics

- operator: grouped top-k router selection
- inputs: `hidden_states[83,7168] fp16` (batch-size assertion only) and
  `gating_output[83,256] fp32`, both on `cuda:0`
- outputs: `topk_weights[83,8] fp32`, `topk_ids[83,8] int32`
- mathematical_behavior: softmax over experts → per-group max → top-4 groups →
  mask others → top-8 experts → renormalize → routed scaling
- tolerance_and_tie_rules: `allclose(atol=1e-2, rtol=1e-2)` fp outputs; exact
  int equality; PyTorch top-k ordering/tie behavior preserved exactly
- public_contract: `ModelNew(topk, renormalize, num_expert_group, topk_group,
  scoring_func="softmax", routed_scaling_factor=1.0).forward(hidden_states, gating_output)`

## Invariants

- `../base.py` bytes are unchanged after adapter generation (verified below).
- The harness is loaded through its AST loader.
- CoreX bootstrap before any import/use:
  `export COREX_VERSION=4.4.0; . /usr/local/corex/enable`.
- Candidate shape/dtype/device/semantics/public contract remain base-compatible.
- Wall time measured by unchanged harness incl. seed + CUDA sync.
- Profiler scopes separate; BI150 trace exposes `cat=kernel` device durations.

## Runtime Fingerprint

```yaml
triton 3.1.0 (/usr/local/corex-4.4.0/lib64/python3/dist-packages/triton) / torch 2.7.1 / CoreX 4.4.0 nvcc V10.2.89 / Iluvatar BI-V150 capability major=7 minor=1 multi_processor_count=16 total_memory=17179869184
```

- discovered_at: `2026-08-27T12:31:22Z` (live ixsmi + torch.cuda probes this host)
- discovery_commands: `ixsmi`; `python3 -c 'import torch,triton; …'`;
  bootstrap `export COREX_VERSION=4.4.0; . /usr/local/corex/enable`
- target_profile_match: `pass`

## Measurement Regime

- measurement_settings_canonical_json: `{"device":"cuda:0","dtype":"mixed(fp16-hidden_states,fp32-gating_output)","profile_iterations":100,"profile_mode":"kernel","profile_warmup":20,"repeat":100,"shape":{"gating_output":[83,256],"hidden_states":[83,7168]},"warmup":50}`
- measurement_fingerprint: `8deb1b012de31b18887562e736c7b9e120b9d9f9500230e237ee003c5fa5a431`
- fingerprint_definition: `sha256(base_bytes ‖ NUL ‖ harness_bytes ‖ NUL ‖ canonical_json_settings)`

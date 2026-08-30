# FusedMoE Optimization Project (Epoch 2)

## Project Identity

- schema_version: 1
- skill_version: 3.0.0
- contract_version: 3
- semantic_contract: typed-sketch-v1
- attribution_contract: verdict-v1
- project_root: `/root/CodeBuddy/20260818191200/kernelswift/kernels/track1-triton/fused_moe/bi150/epoch2`
- base: `../../base.py`
- base_sha256: `21e758535af58c21684d6e657ae44f91a0248d246eb424b6923a3c180bc32a3d` (3598 bytes)
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
- project_capability_claim_sha256: `fcba080f084be2791c43bbe45baaaff695cb2b4a72cc4053a3e070ae6912cff5`
- prior_lineage: `epoch-1 6.60x preserved intact at ../ (final_summary.md, triton_fused_moe_00(1, 2).py)`
- base_branch: `kernel-opt/mmenc-attn-e2-20260828 @bb66016` (stacked; carries triton_cuda profile v1)
- run_branch: `kernel-opt/fusedmoe-e2-20260828`

## Semantics

- operator: MoE gated FFN — softmax routing → top-2 → per-expert dispatch →
  gate/up projection → SiLU(gate)*up → down projection → weighted reduction
- inputs: `hidden_states [83,128] fp16`, `router_logits [83,8] fp32`;
  params `w1 [8,128,128]`, `w2 [8,128,64]` (2*intermediate=128)
- outputs: `[83,128] fp16`
- public_contract: `ModelNew(num_experts, top_k, hidden_size, intermediate_size, renormalize=True).forward(hidden_states, router_logits)` + `run_out(hidden_states, router_logits, out)`.
- DELIVERABLE RULE (binding): submission is ALWAYS the best correctness-PASS Triton
  candidate even if it does not beat base; canonical anchor ≠ submission.

## Invariants

- base.py bytes unchanged; AST-loader only; CoreX bootstrap before torch/triton.
- tl.dot constrained: proven (32,32)@(32,32) fp32-acc envelope; larger tiles and
  fp16-operand dots require Decision-scoped probes (before-fallback).
- Cross-campaign measured facts usable as priors (labeled): Triton python launcher
  tax ~85 µs/call; manual-graph overhead ≈ frontend 46 + replay sync 66 µs;
  graph replay pays only when several launches are compressible; nw=2 beat nw=1 by
  31% device on the sibling attention kernel; fp16-operand dot exactness-negative.

## Runtime Fingerprint

```yaml
triton 3.1.0 (/usr/local/corex-4.4.0/lib64/python3/dist-packages/triton) / torch 2.7.1 / CoreX 4.4.0 nvcc V10.2.89 / Iluvatar BI-V150 capability major=7 minor=1 multi_processor_count=16 total_memory=17179869184
```

- discovered_at: `2026-08-28T23:30:00Z`
- target_profile_match: `pass`

## Measurement Regime

- measurement_settings_canonical_json: `{"device":"cuda:0","dtype":"mixed(fp16-hidden/w1,w2; fp32-router)","profile_iterations":100,"profile_mode":"kernel","profile_warmup":20,"repeat":100,"shape":{"hidden_states":[83,128],"router_logits":[83,8],"w1":[8,128,128],"w2":[8,128,64]},"warmup":50}`
- measurement_fingerprint: `fe73bc58146d8c16f524be2a00fe99b31e1b9678bca6b3702f4284a3ac0a5bef`

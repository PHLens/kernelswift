# MM Encoder Attention Optimization Project (Epoch 2)

## Project Identity

- schema_version: 1
- skill_version: 3.0.0
- contract_version: 3
- semantic_contract: typed-sketch-v1
- attribution_contract: verdict-v1
- project_root: `/root/CodeBuddy/20260818191200/kernelswift/kernels/track1-triton/mm_encoder_attention/bi150/epoch2`
- base: `../../base.py`
- base_sha256: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` (2284 bytes)
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
- project_capability_claim_sha256: `aeba3a87f0494c2bb349b92fe668370c70d77fdebea29eac52824c3556b0d4d8`
- prior_lineage: `epoch-1 naive deliverable 0.55x preserved intact at ../ (final_summary.md, triton_mm_encoder_attention_001.py)`
- base_branch: `kernel-opt/flexattention-e2-20260828 @fb7af57` (stacked lineage carries triton_cuda profile v1)
- run_branch: `kernel-opt/mmenc-attn-e2-20260828`

## Semantics

- operator: full (bidirectional) MHA encoder attention, bsz=2, NO causal mask
- inputs: `query/key/value [2, 83, 512] fp16` (= [bsz, seq, num_heads*head_size])
- outputs: `[2, 83, 512] fp16`
- mathematical_behavior: scale=1/8 → QK^T (full, unmasked) softmax → ·V per (batch, head)
- tolerance_and_tie_rules: allclose(atol=1e-2, rtol=1e-2) fp16 out; dense attention, selection-free
- public_contract: `ModelNew(num_heads, head_size, num_kv_heads=8).forward(query,key,value)`
  plus a `run_out(query,key,value,out)` preallocated-output surface (binding requirement).
- DELIVERABLE RULE (binding, corrected precedent): competition submission is ALWAYS the
  best correctness-PASS Triton candidate, even when it does not beat base; canonical
  manifest anchor and submission deliverable are distinct concepts.

## Invariants

- `../../base.py` bytes unchanged after adapter generation.
- Harness loaded through AST loader only.
- CoreX bootstrap before any torch/triton use:
  `export COREX_VERSION=4.4.0; . /usr/local/corex/enable`.
- tl.dot constrained statuses apply verbatim: proven (32,32)@(32,32) fp32-acc only;
  larger tiles require Decision-scoped capability probes before candidate-ready.

## Runtime Fingerprint

```yaml
triton 3.1.0 (/usr/local/corex-4.4.0/lib64/python3/dist-packages/triton) / torch 2.7.1 / CoreX 4.4.0 nvcc V10.2.89 / Iluvatar BI-V150 capability major=7 minor=1 multi_processor_count=16 total_memory=17179869184
```

- discovered_at: `2026-08-28T13:00:00Z`
- target_profile_match: `pass`

## Measurement Regime

- measurement_settings_canonical_json: `{"device":"cuda:0","dtype":"fp16-all","profile_iterations":100,"profile_mode":"kernel","profile_warmup":20,"repeat":100,"shape":{"key":[2,83,512],"query":[2,83,512],"value":[2,83,512]},"warmup":50}`
- measurement_fingerprint: `0c4c7d664c85e65d0580091ca5e3a77ff769a0d28f7e679f5bdf78fe5d0d966e`
- fingerprint_definition: `sha256(base ‖ NUL ‖ harness ‖ NUL ‖ canonical_json_settings)`

# MM Encoder Attention Optimization Project (S60 Epoch 2)

## Project Identity

- schema_version: 1
- skill_version: 3.0.0
- contract_version: 3
- semantic_contract: typed-sketch-v1
- attribution_contract: verdict-v1
- project_root: `/root/CodeBuddy/20260828202827/kernelswift/kernels/track1-triton/mm_encoder_attention/s60/epoch2`
- base: `../../base.py`
- base_sha256: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` (2284 bytes)
- harness: `/root/CodeBuddy/20260828202827/kernelswift/auto_bench.py`
- harness_sha256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (29428 bytes)
- interpreter: `/usr/bin/python3`
- device: `gcu (Enflame GCU)`
- implementation_language: `triton`
- implementation_backend: `gcu`
- target_id: `s60`
- target_profile: `triton_gcu`
- target_profile_snapshot_ref: `profile_snapshot/triton_gcu.yaml`
- target_profile_snapshot_sha256: `8dfabd0af59b8f6640b47179fee19bca2f5fe35b18535a3db24f60c842e42b70`
- project_capability_claim_ref: `profile_snapshot/capability_claim.json`
- project_capability_claim_sha256: `a175f2727b9198a92da978aca9e8f87834a74884372746699412931890d9748e`
- prior_lineage: `epoch-1 naive deliverable 0.27x preserved intact at ../ (decision.md, triton_mm_encoder_attention_001.py)`
- base_branch: `dev @91a1a89`
- run_branch: `kernel-opt/mm-encoder-attention-s60-e2`

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
- `torch_gcu`/`triton_gcu` bootstrap before any torch/triton use.
- tl.dot constrained statuses apply verbatim: M/N/K must all be multiples of 16;
  seq_len 83 must be padded to 96 (or 128) before tl.dot. num_warps 1/2/4/8 all legal.

## Runtime Fingerprint

```yaml
triton 3.6.0 / triton_gcu 3.6.0+1.0.20260722 / torch 2.10.0+cpu / torch_gcu 2.10.0+3.8.0.2 / Enflame GCU major=3 minor=0 multi_processor_count=2 total_memory=43878764544
```

- discovered_at: `2026-08-28T23:20:00Z`
- target_profile_match: `pass`

## Measurement Regime

- measurement_settings_canonical_json: `{"device":"gcu","dtype":"fp16-all","profile_iterations":100,"profile_mode":"forward","profile_warmup":20,"repeat":100,"shape":{"key":[2,83,512],"query":[2,83,512],"value":[2,83,512]},"warmup":50}`
- measurement_fingerprint: `c335b39cbf2eaa15e1a358be90d0aab85d0fd7e8ffd4b7b4e825df0901ad61f9`
- fingerprint_definition: `sha256(base ‖ NUL ‖ harness ‖ NUL ‖ canonical_json_settings)`

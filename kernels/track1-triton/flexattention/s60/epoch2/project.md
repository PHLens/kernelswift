# FlexAttention Optimization Project (S60 Epoch 2)

## Project Identity

- schema_version: 1
- skill_version: 3.0.0
- contract_version: 3
- semantic_contract: typed-sketch-v1
- attribution_contract: verdict-v1
- project_root: `/root/CodeBuddy/20260828202827/kernelswift/kernels/track1-triton/flexattention/s60/epoch2`
- base: `../../base.py`
- base_sha256: `dd1359ad88a5d5aae48a115023e5ae9016e4d5b5e6a12781273703e8d5c6a6d0`
- harness: `/root/CodeBuddy/20260828202827/kernelswift/auto_bench.py`
- harness_sha256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- interpreter: `/usr/bin/python3`
- device: `gcu (Enflame GCU)`
- implementation_language: `triton`
- implementation_backend: `gcu`
- target_id: `s60`
- target_profile: `triton_gcu`
- target_profile_snapshot_ref: `profile_snapshot/triton_gcu.yaml`
- project_capability_claim_ref: `profile_snapshot/capability_claim.json`
- prior_lineage: `epoch-1 deliverable 0.42x preserved at ../ (final_summary.md, rounds/)`
- base_branch: `dev @91a1a89`
- run_branch: `kernel-opt/mm-encoder-attention-s60-e2` (shared epoch-2 branch)

## Semantics

- operator: CAUSAL scaled-dot-product attention (flexattention), num_tokens=83, 8 heads, head_size 64, fp16
- inputs: `query/key/value [83, 8, 64]` fp16 (= [num_tokens, num_heads, head_size])
- outputs: `[83, 512]` fp16
- mathematical_behavior: scale=1/8, QK^T with CAUSAL mask (upper triangle masked to -inf), softmax, ·V
- tolerance: allclose(atol=1e-2, rtol=1e-2) fp16 out

## Runtime Fingerprint

```yaml
triton 3.6.0 / triton_gcu 3.6.0+1.0.20260722 / torch 2.10.0+cpu / torch_gcu 2.10.0+3.8.0.2 / Enflame GCU major=3 minor=0 multi_processor_count=2
```

## Measurement Regime

- warmup 50 / repeat 100 / seed 42 / interleaved pairs / median wall
- measurement_fingerprint: computed as sha256(base ‖ NUL ‖ harness ‖ NUL ‖ canonical_json_settings)

## Key Prior (from mm_encoder_attention s60 e2, same backend)

- fp16 QK^T tl.dot + fp32 PV, single-tile TP=128, num_warps=1 is the S60-optimal attention recipe (0.27x → 0.92x for non-causal)
- tl.dot/tl.arange require power-of-2 (S=83 → TP=128, 58% FLOP waste); tl.max/tl.sum no-keepdim; tl.dot same-dtype
- S60 device-bound; hand-written attention ~0.9x ceiling vs vendor flash-attention library

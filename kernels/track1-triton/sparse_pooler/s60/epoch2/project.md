# SPLADE Sparse Pooler Optimization Project (S60 Epoch 2)

## Project Identity

- schema_version: 1
- skill_version: 3.0.0
- contract_version: 3
- semantic_contract: typed-sketch-v1
- attribution_contract: verdict-v1
- project_root: `/root/CodeBuddy/20260828202827/kernelswift/kernels/track1-triton/sparse_pooler/s60/epoch2`
- base: `../../base.py`
- base_sha256: `46106baa46b9f1952b32388d99880c967438ea99becdc874e3f68ba81a727d58`
- harness: `/root/CodeBuddy/20260828202827/kernelswift/auto_bench.py`
- harness_sha256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- interpreter: `/usr/bin/python3`
- device: `gcu (Enflame GCU)`
- implementation_language: `triton`
- implementation_backend: `gcu`
- target_id: `s60`
- target_profile: `triton_gcu`
- target_profile_snapshot_ref: `profile_snapshot/triton_gcu.yaml`
- prior_lineage: `epoch-1 0.79x preserved at ../ (final_summary.md, rounds/)`
- base_branch: `dev @91a1a89`
- run_branch: `kernel-opt/mm-encoder-attention-s60-e2` (shared epoch-2 branch)

## Semantics

- operator: SPLADESparsePooler — MLM head logits -> ReLU log1p -> sequence max pooling
- inputs: `hidden_states [83,768] fp32`, `seq_lens [4] int32 = [20,25,18,20]`
- outputs: `list of 4 x [30522] fp32`
- mathematical_behavior: decoder(LayerNorm(GELU(dense(h)))), then log1p(relu), then
  per-sequence max pooling (segments of lengths [20,25,18,20]).
- tolerance: allclose fp32

## Runtime Fingerprint

```yaml
triton 3.6.0 / triton_gcu 3.6.0+1.0.20260722 / torch 2.10.0+cpu / torch_gcu 2.10.0+3.8.0.2 / Enflame GCU major=3 minor=0 multi_processor_count=2
```

## Key Prior (preflight, orchestrator-scoped, all directions falsified)

Time decomposition (per forward call):
- dense GEMM [83,768]@[768,768] + GELU + LayerNorm: ~165us
- decoder GEMM [83,768]@[768,30522]: ~316us
- log1p(relu) elementwise [83,30522]: ~110us
- max pooling (4 segments) + D2H sync: ~183us (of which D2H `seq_lens.tolist()` = **125us**, 16%!)

Explored directions, all falsified:
1. epoch-1 fused relu/log1p/max + device prefix-scan: -26.79% (device penalty ~270us > host save ~49us)
2. scatter_reduce segment max (avoid tolist): **5153us (7x slower, catastrophic)** — scatter_reduce unoptimized on GCU
3. D2H sync elimination requires hand-written segment reduction (~150us penalty), net negative

Conclusion: GEMM (481us, 61%) is vendor-bound and untouchable; the 125us D2H sync
cannot be eliminated without a hand-written segment reduction that costs more. This
operator is measurement-bound.

# MusicFlamingo Rotary Embedding Optimization Project (S60 Epoch 2)

## Project Identity

- schema_version: 1
- skill_version: 3.0.0
- contract_version: 3
- semantic_contract: typed-sketch-v1
- attribution_contract: verdict-v1
- project_root: `/root/CodeBuddy/20260828202827/kernelswift/kernels/track1-triton/music_flamingo_rotary_embedding/s60/epoch2`
- base: `../../base.py`
- base_sha256: `99829754f4cdc4bfd2808e051de549f0791414241e7fdbad7a1b8294a15be475`
- harness: `/root/CodeBuddy/20260828202827/kernelswift/auto_bench.py`
- harness_sha256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- interpreter: `/usr/bin/python3`
- device: `gcu (Enflame GCU)`
- implementation_language: `triton`
- implementation_backend: `gcu`
- target_id: `s60`
- target_profile: `triton_gcu`
- target_profile_snapshot_ref: `profile_snapshot/triton_gcu.yaml`
- prior_lineage: `epoch-1 deliverable 0.9x preserved at ../ (final_summary.md, rounds/)`
- base_branch: `dev @91a1a89`
- run_branch: `kernel-opt/mm-encoder-attention-s60-e2` (shared epoch-2 branch)

## Semantics

- operator: MusicFlamingoRotaryEmbedding — batch+time positional embedding
- inputs: `timestamps [4,32] fp32`, `seq_len=32`
- outputs: `(cos, sin)` each `[4,32,128] fp32`
- mathematical_behavior: combine batch frequencies (inv_freq × batch_position,
  repeat_interleave 2) and time frequencies (position_angles, precomputed buffer)
  via broadcast+cat, multiply by -timestamps*2π, return cos/sin.
- tolerance: exact-match (pure deterministic elementwise + trig)

## Runtime Fingerprint

```yaml
triton 3.6.0 / triton_gcu 3.6.0+1.0.20260722 / torch 2.10.0+cpu / torch_gcu 2.10.0+3.8.0.2 / Enflame GCU major=3 minor=0 multi_processor_count=2
```

## Key Prior (preflight, orchestrator-scoped)

- base has 13 launches (elementwise chain: div/mul/repeat_interleave/broadcast/cat/mul
  + vendor cos/sin), wall ~367us.
- epoch-1 FULL fusion (cos/sin via tl.cos/tl.sin) was -13%: GCU's math-dialect
  tl.cos/tl.sin is ~44% slower than the vendor trig library kernel.
- **preflight NEW direction (epoch-1 did NOT try)**: PARTIAL fusion — fuse only the
  freqs elementwise chain (div/mul/repeat_interleave/broadcast/cat/mul-angle) into a
  single kernel, keep cos/sin as vendor torch.cos/torch.sin. Reached **1.49x**
  (367us → 246us), correctness exact-match (diff=0.0).
- This splits the difference: kernel avoids the slow tl.cos/tl.sin (the epoch-1
  device penalty) while still collapsing ~10 elementwise launches.

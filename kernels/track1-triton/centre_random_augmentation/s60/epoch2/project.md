# Centre Random Augmentation Optimization Project (S60 Epoch 2)

## Project Identity

- schema_version: 1
- skill_version: 3.0.0
- contract_version: 3
- semantic_contract: typed-sketch-v1
- attribution_contract: verdict-v1
- project_root: `/root/CodeBuddy/20260828202827/kernelswift/kernels/track1-triton/centre_random_augmentation/s60/epoch2`
- base: `../../base.py`
- base_sha256: `02e7020fef34db401cfc2cf8031700262493ab96a89449d4e70d333978e78553`
- harness: `/root/CodeBuddy/20260828202827/kernelswift/auto_bench.py`
- harness_sha256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- interpreter: `/usr/bin/python3`
- device: `gcu (Enflame GCU)`
- implementation_language: `triton`
- implementation_backend: `gcu`
- target_id: `s60`
- target_profile: `triton_gcu`
- target_profile_snapshot_ref: `profile_snapshot/triton_gcu.yaml`
- prior_lineage: `epoch-1 deliverable 0.95x preserved at ../ (decision.md, triton_centre_random_augmentation_001.py)`
- base_branch: `dev @91a1a89`
- run_branch: `kernel-opt/mm-encoder-attention-s60-e2` (shared epoch-2 branch)

## Semantics

- operator: centre_random_augmentation (diffusion rigid-body random augmentation)
- inputs: `x_input_coords [256,3] fp32`, `mask [256] fp32` (all ones in get_inputs)
- outputs: `[4, 256, 3] fp32` (n_sample=4)
- mathematical_behavior: masked centering → random quaternion rotation + random
  translation → optional masking. Random numbers (u1,u2,u3,T) generated on host.
- tolerance: exact-match expected (random sequences identical under seed 42)

## Runtime Fingerprint

```yaml
triton 3.6.0 / triton_gcu 3.6.0+1.0.20260722 / torch 2.10.0+cpu / torch_gcu 2.10.0+3.8.0.2 / Enflame GCU major=3 minor=0 multi_processor_count=2
```

## Key Prior (preflight, orchestrator-scoped)

- base is LAUNCH-BOUND: 78 `topsLaunchKernel`/call (census: rand×3 + sqrt/sin/cos
  quaternion chain + stack/reshape rotation matrix + unbind/stack rot_vec_mul +
  expand/contiguous + mul/add/sub + mask), tiny tensors (n_sample=4, N_atom=256).
- epoch-1 candidate fused ONLY rot_vec_mul (saved 1 launch) → 0.95x (no real gain).
- preflight probe: fusing quaternion→R + rot_vec_mul + translation + mask into a
  single kernel (grid=(n_sample,), host generates only u1/u2/u3/T) reached ~1.59x
  (correctness max_abs_diff 4.77e-7, exact-match within fp32).
- This is the fused_moe class (launch-bound, many tiny ops) → fusion wins.
- Random numbers MUST stay host-generated (GCU kernel has no torch.rand), and the
  sequence/order must match base exactly for correctness.

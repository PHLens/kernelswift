# Coder Context

Compact, ownership-safe state for the Coder role. Authoritative measurement
claims and the round result live in `rounds/coder_result_002.md`, not here.

## Contract

- role: Coder
- workflow: kernel-opt-loop (task 9 `centre_random_augmentation`, BI150)

## Last Completed Round

- round: `002`
- result: `candidate-ready`
- candidate: `kernels/track1-triton/centre_random_augmentation/bi150/triton_centre_random_augmentation_002.py`
- candidate_sha256: `efac6ee782e859701bb14aca04b7f56516a575a5f74507958e1930a95005a530`

## Selected Profile / Fingerprint Facts

- target_profile: `triton_cuda`
- backend: `cuda`
- runtime fingerprint: triton 3.1.0 / torch 2.7.1 / Iluvatar BI-V150, capability (7,1)
- measurement_fingerprint: `a5f980780c4dcde731df913710ad9dfded4f07a66b90e334fea0a6f2aa1fd5fa`
- canonical reference: `triton_centre_random_augmentation_001.py`

## Artifact Read Hashes

- base.py: `02e7020fef34db401cfc2cf8031700262493ab96a89449d4e70d333978e78553`
- triton_centre_random_augmentation_001.py: `4e33276ec28f3695aa08462aa6cb796a160aca47dad889168a7cdd8aa8e16036`
- decision_002.md: `2290e37b81072b794ca5735dddba52ed19805c943a8e7109b598e5fd1f65af8e`
- harness (auto_bench.py): `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`

## Open Local Checks

None. All Coder gates (validate_decision, py_compile, harness loader, accuracy
smoke) passed.

## Key Implementation Facts

- Fused the quaternion-to-rotation-matrix construction (sqrt/sin/cos + 9-entry
  arithmetic) into the single `_centre_aug_kernel` (grid `(4,)`), on top of Round
  001's centering + rot_vec_mul + translation + mask fusion.
- RNG boundary: 3x `torch.rand` (u1/u2/u3) + 1x `torch.randn` (T) kept as
  host-side calls inside `forward`, order preserved; kernel never draws random.
- `tl.sqrt`/`tl.sin`/`tl.cos` locally proven to lower on BI150 and match torch
  bit-for-bit (probe max abs diff 0.0) — not a capability-miss.
- `2*math.pi` inlined as literal `6.283185307179586` in the kernel (Triton JIT
  cannot resolve module globals; AST loader strips non-literal Assign).
- Smoke accuracy PASS with speedup 4.218x (non-authoritative timing).

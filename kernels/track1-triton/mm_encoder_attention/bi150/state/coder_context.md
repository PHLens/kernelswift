# Coder Context — mm_encoder_attention (BI150, task 6)

Ownership-safe local state only. Not a replacement for `coder_result_deliverable.md`.

## Task

- 参赛交付物（正确性优先，非优化），task `mm_encoder_attention`，BI150 后端。

## Profile / Fingerprint Facts

- language: triton; backend: cuda; target_profile: triton_cuda
- runtime: CoreX 4.4.0 / Triton 3.1.0 / Torch 2.7.1
- device: `cuda:0 (Iluvatar BI-V150)`, capability (7,1)
- measurement_fingerprint: `b8029499f0964a738f50b09164e419511d0bc89df5e260573e607bb7345afc2e`

## Candidate

- path: `kernels/track1-triton/mm_encoder_attention/bi150/triton_mm_encoder_attention_001.py`
- sha256: `88ade697da35a51362c2a8643e054a61362a68ff3e9e2e60110bd3e45285e87e`

## Open Local Checks

- py_compile: PASS
- harness correctness: PASS accuracy (v0=0.196228 ms, v1=0.358623 ms, speedup=0.547x)

## Artifact Read Hashes

- base.py: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2`

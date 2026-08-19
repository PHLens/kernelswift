# Coder Context

## Contract

- role: Coder
- contract hash (coder.md): tracked by workflow; round 001

## Last Completed Round

- round: `001`
- result: `candidate-ready`
- decision SHA-256: `335389df2498f37fb9f2c5c7ebc10986ab4edf555d939525413900e0e885ecfc`
- candidate path: `kernels/track1-triton/mhc_post_layer_mix/bi150/triton_mhc_post_layer_mix_001.py`
- candidate SHA-256: `08a9d59f17ffa80224943b19bdcce390d908ca8ba15bf2e06ae469f45787d9fb`

## Selected Profile / Fingerprint Facts

- target_profile: `triton_cuda`
- language: `triton`; backend: `cuda`
- triton: `3.1.0` (corex); torch: `2.7.1`
- device: `cuda:0 (Iluvatar BI-V150)`, capability `(7,1)`, 16 SM
- runtime fingerprint: match `pass`

## Artifact Read Hashes (as read this round)

- base.py: `e392799f72edcf7210373ac8f2522bcf3d4f740c8f9dd0d1f117bc627939ebf3`
- baseline_adapter.py: `66a3a2c31863d18c725a52ab57fd1b9f89fe655dd7bab7cb4da158b8130b5d07`
- auto_bench.py: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`

## Open Local Checks

- none (harness smoke passed accuracy; candidate handed to Verifier).

## Notes

- Intervention: elementwise-fusion (single Triton kernel for the post-GEMM
  tail). GEMM `term2` (`torch.einsum`) left unchanged.
- Smoke: `v0=8.023760 ms, v1=6.446620 ms, speedup=1.245x`.
- Conformance notes: `num_warps`/`num_stages` left unset (Unknown/non-normative);
  direct Triton launch syntax used (not `fast_libentry`).

# Coder Context

## Contract

- role: Coder
- contract hash (coder.md): tracked by workflow; round 002

## Last Completed Round

- round: `002`
- result: `candidate-ready`
- decision SHA-256: `2d44dd2c808bf27c20cdd4d6ca0aa0ecba422080394462f6d176ccc2c5a146a6`
- candidate path: `kernels/track1-triton/fused_moe/bi150/triton_fused_moe_002.py`
- candidate SHA-256: `6ac1f44b111285f5bf746110c51f6486868b12beb2deae3390663d74233f8ae5`

## Selected Profile / Fingerprint Facts

- target_profile: `triton_cuda`
- language: `triton`; backend: `cuda`
- triton: `3.1.0` (corex); torch: `2.7.1`
- device: `cuda:0 (Iluvatar BI-V150)`, capability `(7,1)`, 16 SM
- runtime fingerprint: match `pass`
- measurement fingerprint: `5c2a51ab3f3ebaab1123b9fa534d4e4b940f3334f80fac00252df780d3900150`

## Artifact Read Hashes (as read this round)

- base.py: `a0269ac15833098c11c63045318e829e404b09ca49ae8dce22b244e0d2894d2b`
- baseline_adapter.py: `8e5c70232e541a02d83343216376ece9127a1c3e6ea6af77dc77a2723783facf`
- triton_fused_moe_001.py (canonical): `8424c7a01bc1d293c2b0ef509dd895950112cfb71dedd145053b4ac3f7eb9ad6`
- auto_bench.py: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`

## Open Local Checks

- none (harness smoke passed accuracy; candidate handed to Verifier).

## Notes

- Intervention: gemm-fusion. A single `_fused_moe_expert_kernel` (grid=(8,),
  one program per expert) fuses the per-expert GEMM loop (gate/up + down via
  `tl.dot`), chunk/SiLU/mul, in-kernel expert dispatch (static `tl.arange`
  compare, no sort), and the weighted reduction (`tl.atomic_add`). This removes
  the `torch.argsort` chain (107+20+12.5 us/call) and the ~40 per-expert kernels
  entirely.
- Matched `tl.dot` probe (file-backed): fp16 `(M,128)@(128,128)` and
  `(M,64)@(64,128)` correct for M>=16 (max_rel_err ~2e-4); M=1/2/4 do NOT lower
  (CompilationError). This forced a per-expert batched layout (`BLOCK_M=256`).
- Preserved bit-exactly: `torch.softmax`, `torch.topk` (tie order), renormalize.
  GEMM contraction dims (128/64), SiLU, weighted-sum semantics preserved;
  max_abs_diff 1.53e-5 vs base.
- No on-chip `tl.gather` (avoids Entry 013); dispatch via static mask/`tl.where`.
- Smoke (canonical 001 vs 002): speedup ~5.0x (2.488 ms -> 0.492 ms), all PASS.
- Conformance notes: `tl.dot`/`tl.trans`/`tl.sigmoid`/`tl.atomic_add` exercised
  and verified (outside original profile); `num_warps`/`num_stages` left unset.

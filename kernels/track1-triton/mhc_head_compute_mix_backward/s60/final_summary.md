# Final Summary — mhc_head_compute_mix_backward (S60 Epoch 1 + Epoch 2 assessment)

## Result (accepted, epoch-1; no epoch-2 space)

- status: **accepted (epoch-1 1.23x)**; epoch-2 assessment confirms no further space
- **deliverable**: `triton_mhc_head_compute_mix_backward_001.py` (correctness-PASS)
- **speedup vs base**: **1.23x** (base 0.40ms → candidate 0.32ms, 3-pair stable 1.22-1.23x)

## Epoch-1 intervention (accepted)

Elementwise sigmoid-backward (z → sigmoid → grad_z → grad_input_mix) fused into a
single Triton kernel; the two small reductions (grad_mhc_base [4], grad_mhc_scale [1])
kept as host `torch.sum` (vendor reduction, exact). 8192-element operator, launch-bound.

## Epoch-2 assessment (no space — atomic unavailable)

Attempted to fuse the two reductions into the kernel via `tl.atomic_add` to eliminate
the 2 host reduction launches. **Falsified: `tl.atomic_add` is UNAVAILABLE on triton_gcu**
(minimal 1-block atomic example fails to compile with `Pipeline run failed:
PassManager execution failed`). Cross-program reduction therefore cannot be fused,
and the two-host-reduction structure of epoch-1 is optimal.

## Census (dual-scope forward profile)

| metric | base | candidate |
|---|---:|---:|
| topsLaunchKernel/call | ~6.5 | (1 Triton kernel + 2 vendor reductions) |
| aten mul/call | 6.0 | 0 (fused) |
| aten sum/call | 4.0 | 2 (vendor reductions) |

## Terminal Classification

- terminal_result: accepted (epoch-1 1.23x); epoch-2 no-space (atomic unavailable)
- stop_reason: atomic_add unavailable on GCU → cross-program reduction cannot be fused;
  epoch-1 elementwise fusion + host reduction is the ceiling.

## Key capability finding (written back to triton_gcu profile)

- `tl.atomic_add` is **unsupported** on triton_gcu 3.6.0 (compile failure). Any operator
  requiring cross-program reduction (segment-max/sum, scatter) MUST keep host-side
  `torch.sum`/`torch.max` reductions — this is the structural reason sparse_pooler's
  segment-max and this operator's two reductions cannot be fused on S60.

# Final Summary — centre_random_augmentation (S60 Epoch 2)

## Delivery Result (SUCCESS — accepted, beats base)

- status: **complete — accepted**
- **deliverable**: `triton_centre_random_augmentation_e2_001.py` (correctness-PASS)
- **speedup vs base**: **1.90x (+47.60%)** — S60's FIRST operator to beat base
- **speedup vs epoch-1**: **~2.0x (0.95x → 1.90x)**

## Intervention

Single direct-launched Triton kernel (grid=(n_sample,)=4, num_warps=1) that fuses
quaternion→rotation-matrix (tl.sqrt/tl.sin/tl.cos) + 3×3 matvec (statically-unrolled
3 dot products) + translation + masking into ONE kernel. Host keeps only the random
sources (u1,u2,u3=torch.rand ×3, then T=s_trans*torch.randn) and the masked-mean
center (torch, matching base exactly).

## Why it wins (launch-bound, the fused_moe class)

| metric | base | candidate |
|---|---:|---:|
| topsLaunchKernel/call | 96 (@921.87us) | 10 (@96.43us) |
| candidate Triton kernel | — | 1 (topsModuleLaunchKernel @11.06us) |
| aten+GCU cpu_ops/call | 534 | 62 |
| launch-API time | ~922us | ~118us |

Base is launch-bound (96 tiny-launch ops: rand/sqrt/sin/cos quaternion chain,
stack/reshape rotation matrix, unbind/stack rot_vec_mul, expand/contiguous, mul/add/sub,
mask on tiny tensors n_sample=4, N_atom=256). Net launch saving ~814us ≈ the wall delta.
This is the opposite of the device-bound attention/GEMM operators on S60.

## Correctness

4/4 PASS (3 timing pairs + profile run). Random sequences (3×torch.rand + 1×torch.randn)
generated on host in exactly the same order/shapes as base → bit-identical under seed 42
(exact-match). Max_abs_diff from preflight 4.77e-7.

## Terminal Classification

- terminal_result: accepted (+47.60%, decisively above +5% bar)
- stop_reason: accepted; remaining center-reduction fusion is marginal (~6%) and needs
  cross-program reduction, not worth an additional round.

## Note

- No tl.dot (3×3 matvec statically unrolled; primary contract math.elementwise).
- Deviation D1 (centre_only=True returns expand view without .contiguous()) is dead code
  for this instantiation (CENTRE_ONLY=False), flagged but non-blocking.

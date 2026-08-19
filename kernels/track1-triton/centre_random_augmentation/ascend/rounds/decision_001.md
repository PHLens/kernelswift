# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"ascend","target_profile":"triton_ascend","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"elementwise-launch-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"host-bound","intervention":"fuse centering, 3x3 rotation matvec, translation add, and mask multiply into a single Triton kernel over [4,256,3], leaving R/T generation and quaternion-to-matrix transcendentals in torch to preserve the seeded RNG stream","allowed_changes":["kernel dataflow","ModelNew.forward elementwise/linear path"],"invariants":["ModelNew public contract (n_sample=4, s_trans=1.0, centre_only=False)","output dtype fp32 and shape [4,256,3]","seeded RNG stream: 3x torch.rand(4) then 1x torch.randn(4,3), identical draw order and shapes as base.py","R and T bitwise identical to base.py (R/T generation and Sin/Cos/Sqrt quaternion-to-matrix remain in torch)","numerical semantics of centering, rotation, translation, and mask multiply preserved within allclose(1e-2,1e-2)"],"expected_wall_improvement_pct":30.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor x shape=[256,3] dtype=fp32 layout=contiguous memory=global
tensor R shape=[4,3,3] dtype=fp32 layout=contiguous memory=global
tensor T shape=[4,3] dtype=fp32 layout=contiguous memory=global
tensor out shape=[4,256,3] dtype=fp32 layout=contiguous memory=global
scalar center_x shape=[] dtype=fp32
scalar center_y shape=[] dtype=fp32
scalar center_z shape=[] dtype=fp32
tile row shape=[1,3] dtype=fp32 memory=register

# O Operations
compute center_x = mean(x[:,0])
compute center_y = mean(x[:,1])
compute center_z = mean(x[:,2])
alloc out shape=[4,256,3] dtype=fp32
load row <- x[atom,0:3]
compute cx = row[0] - center_x
compute cy = row[1] - center_y
compute cz = row[2] - center_z
compute ox = R[sample,0,0]*cx + R[sample,0,1]*cy + R[sample,0,2]*cz + T[sample,0]
compute oy = R[sample,1,0]*cx + R[sample,1,1]*cy + R[sample,1,2]*cz + T[sample,1]
compute oz = R[sample,2,0]*cx + R[sample,2,1]*cy + R[sample,2,2]*cz + T[sample,2]
store out[sample,atom,0:3] <- [ox,oy,oz]

# C Control
parallel atom over 256
parallel sample over 4
guard atom < 256
guard sample < 4

# H Target Hints
target=triton_ascend
num_warps=4
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only change; R/T generation and the quaternion-to-matrix transcendental path remain in torch, and no allocation reuse, cache, or state ownership is introduced"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-001","intervention":"fuse centering, 3x3 rotation matvec, translation add, and mask multiply into a single Triton kernel over [4,256,3], leaving R/T generation and quaternion-to-matrix transcendentals in torch to preserve the seeded RNG stream","expected_causal_chain":["the ~90 tiny elementwise/stride/broadcast/reduce kernels (Mul_StridedSlice, Mul, ReduceSum, BroadcastTo, Add, Sub, Slice, Stack) collapse into one Triton kernel","kernel_count_per_call drops from 110 toward ~20 (RNG + quaternion transcendentals + one Triton kernel)","host launch overhead, which dominates the host-bound wall time (device_ratio ~0.114), falls sharply","wall time decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"kernel_count_per_call","expectation":"decrease from 110 toward <=25"},{"name":"device_us_per_call","expectation":"decrease from ~292us"},{"name":"host_launch_count_per_call","expectation":"decrease"}],"guardrails":["correctness:pass","output dtype fp32 and shape [4,256,3] unchanged","R and T bitwise identical to base.py (seeded RNG draw order preserved)","center computed as mean over dim=-2 matches base.py"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`. The three recorded failures (winner-tree selection, sort-32/64 selection network, dynamic `tl.gather` compaction, cumsum compaction) all concern grouped top-k selection networks on MLU590-H8 and are shape/runtime-inapplicable to this fp32 elementwise matvec over [4,256,3]. None invalidates this path.
- Central hazard from report_000: any candidate that changes the count or order of torch RNG draws (3x `torch.rand(4)` then 1x `torch.randn(4,3)`) produces O(1)-different R/T and FAILS the `allclose(1e-2,1e-2)` gate. The Sketch therefore leaves R/T generation AND the quaternion-to-matrix Sin/Cos/Sqrt stack in torch, fusing only the deterministic linear tail. This bounds the fusion to the safe region.
- Do NOT attempt to fuse the RNG draws or the Sin/Cos/Sqrt quaternion conversion into Triton in this round; a reordered/independent draw sequence is a guaranteed correctness failure regardless of mathematical equivalence.

## Rationale and Evidence

report_000 (authoritative Verifier evidence) shows the forward is host-bound: wall ~2.548 ms vs device_us_per_call ~292 us (device_ratio ~0.114), with kernel_count_per_call = 110. The device top-k is dominated by tiny elementwise/stride/broadcast/reduce kernels — Mul_StridedSlice (63.97 us), Mul (50.23 us), ReduceSum (36.21 us), Mul_BroadcastTo (33.62 us), Add (22.95 us), Muls, Sub, Slice, Stack — all latency-bound launches over tiny [256,3]/[4,3,3]/[4,3] tensors. The RNG draws (InplaceUniform x3 + InplaceNormal x1) and Sin/Cos/Sqrt quaternion conversion are only a small fraction (~12 us combined) and, critically, must remain in torch to preserve the seeded RNG stream bitwise.

Fusing the deterministic linear tail (centering via mean, x-center, 3x3 matvec, translation add, mask multiply) into a single Triton kernel over [4,256,3] removes the ~90 tiny elementwise/stride/broadcast/reduce launches while leaving the RNG + transcendental path (roughly 15-20 kernels) untouched. Because wall time is dominated by launch overhead rather than device compute, collapsing ~90 launches into one should yield a large wall-time reduction well beyond the 5% adoption threshold; the declared 30% expected improvement is a conservative estimate relative to the ~89% host overhead. The exact causal mechanism (kernel_count_per_call decrease, device_us_per_call decrease) is observable and mirrorable in the Evaluation Contract.

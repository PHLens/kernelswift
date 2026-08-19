# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"cuda","target_profile":"triton_cuda","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"kernel-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"device-bound","intervention":"fuse the full sigmoid-backward chain (elementwise multiply/add/sigmoid plus the two sum reductions) into a single Triton kernel","allowed_changes":["kernel dataflow"],"invariants":["ModelNew public contract","output dtype and shape","sigmoid-backward numerical semantics","two reduction contracts grad_mhc_base=sum(dim=(0,1)) and grad_mhc_scale=sum(dim=(0,1,2))"],"expected_wall_improvement_pct":20.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor input_mix shape=[2,1024,4] dtype=fp32 layout=contiguous memory=global
tensor mhc_scale shape=[1] dtype=fp32 layout=contiguous memory=global
tensor mhc_base shape=[4] dtype=fp32 layout=contiguous memory=global
tensor grad_out shape=[2,1024,4] dtype=fp32 layout=contiguous memory=global
tensor grad_input_mix shape=[2,1024,4] dtype=fp32 layout=contiguous memory=global
tensor grad_mhc_base shape=[4] dtype=fp32 layout=contiguous memory=global
tensor grad_mhc_scale shape=[1] dtype=fp32 layout=contiguous memory=global
tile tile shape=[BLOCK] dtype=fp32 memory=register

# O Operations
load im <- input_mix[flat:flat+BLOCK]
load go <- grad_out[flat:flat+BLOCK]
compute z = im * mhc_scale + mhc_base
compute sig = sigmoid(z)
compute gz = go * sig * (1 - sig)
compute gim = gz * mhc_scale
store grad_input_mix[flat:flat+BLOCK] <- gim
compute gb_partial = sum(gz) over dim=(0,1) keeping dim=2
compute gs_partial = sum(gz * im) over dim=(0,1,2)
store grad_mhc_base <- gb_partial
store grad_mhc_scale <- gs_partial

# C Control
parallel flat over N=8192
guard flat < N

# H Target Hints
target=triton_cuda
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only change: replace the eager elementwise-plus-reduce chain with one Triton kernel while keeping ModelNew.forward's argument/return signature unchanged"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-001","intervention":"fuse the full sigmoid-backward chain (elementwise multiply/add/sigmoid plus the two sum reductions) into a single Triton kernel","expected_causal_chain":["intermediate z/sigmoid/grad_z tensors stop being materialized","kernel count per call drops from ~9.74 toward 1-2","the two dominant sum reductions (147.98 us/call, ~80% of device time) are absorbed into the fused kernel","device time per call decreases","wall time decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"kernel_count_per_call","expectation":"decrease toward 1-2"},{"name":"device_us_per_call","expectation":"decrease"},{"name":"reduce_sum_kernel_us_per_call","expectation":"decrease toward zero as a standalone kernel"}],"guardrails":["correctness:pass","output structure, shape, and dtype unchanged","numerical semantics match atol=1e-2, rtol=1e-2"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`; the four recorded failures (winner tree, sort network, dynamic `tl.gather`, cumsum compaction) all concern grouped-topk selection lowering on MLU590, not single-kernel sigmoid-backward fusion on BI150. None matches this decision's shape (`[2,1024,4]` fp32) or intervention (elementwise + reduction fusion).
- Target-profile constraint noted: `tl.sum` is proven on BI150 only for axis-0 and axis-1 reductions over `(256,)` and `(8,32)` fp32; the two reductions here (`sum(dim=(0,1))` keeping dim=2, and full `sum(dim=(0,1,2))`) are reduction-over-reduced-dim patterns not yet in the evidence ledger. The decision does not require an unsupported primitive, but Coder must validate the exact reduction lowering on BI150; if the multi-dim reduce does not lower, a two-program decomposition (one elementwise program plus one reduction program) remains within the kernel-fusion change family and is not a major deviation.
- `num_warps` and `num_stages` are unproven on this profile revision; the sketch leaves them unspecified so Coder does not assert an unproven hint.

## Rationale and Evidence

Baseline report `rounds/report_000.md` shows the operator runs ~9.74 kernels per forward call over an 8192-element tensor, with device time `185.599 us/call` and `device_ratio ≈ 0.528`. The top-kernel breakdown names two `sum_functor` reductions (`reduce_kernel<1024,1,...>`, ~1.92/call, `147.977 us/call`) as ~80% of device time, plus a sigmoid kernel (~0.98/call) and a mul/add elementwise chain (~6.8/call). The entire computation materializes three intermediate tensors (`z`, `sigmoid`, `grad_z`), each `[2,1024,4]` fp32 = 32 KB, which are written and re-read across distinct kernels.

The intervention fuses this chain into a single Triton kernel: load `input_mix`/`grad_out`/`mhc_scale`/`mhc_base`, compute `z -> sigmoid -> grad_z -> grad_input_mix` in registers, and reduce `grad_z` and `grad_z * input_mix` on-chip to produce `grad_mhc_base [4]` and `grad_mhc_scale [1]`. This eliminates intermediate materialization and collapses ~10 launches toward 1-2, removing both the dominant reduction kernel and the launch overhead that makes up part of the ~47% non-device wall time.

Expected wall improvement: device time is ~80% reduction-bound and ~53% of wall is device, so removing the reduction kernel and the intermediate materialization targets the largest compressible component; the remaining elementwise work is tiny and the launch-count reduction also attacks the host fraction. A conservative 20% expected improvement is justified (the two reductions alone are ~42% of wall time: `147.98 us / 351.449 us ≈ 0.42`, and launch elimination contributes further).

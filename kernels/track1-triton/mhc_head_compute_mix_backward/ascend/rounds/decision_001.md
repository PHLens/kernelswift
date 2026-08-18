# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"ascend","target_profile":"triton_ascend","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"kernel-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"host-bound","intervention":"fuse the sigmoid elementwise chain and both reductions into one Triton kernel, collapsing 10 unfused library kernels to 1","allowed_changes":["kernel dataflow","kernel launch count","forward reduction computation"],"invariants":["ModelNew public contract (forward signature and output tuple)","output shapes and dtypes (grad_input_mix[2,1024,4], grad_mhc_scale[1], grad_mhc_base[4], all fp32)","numerical semantics (sigmoid chain plus both reductions, equivalent to keepdim then view)","caller-selected device and current stream preserved"],"expected_wall_improvement_pct":15.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor input_mix shape=[2048,4] dtype=fp32 layout=row_major memory=global
tensor grad_out shape=[2048,4] dtype=fp32 layout=row_major memory=global
tensor mhc_base shape=[4] dtype=fp32 layout=contiguous memory=global
scalar mhc_scale dtype=fp32 memory=global
tensor grad_input_mix shape=[2048,4] dtype=fp32 layout=row_major memory=global
tensor grad_mhc_base shape=[4] dtype=fp32 layout=contiguous memory=global
tensor grad_mhc_scale shape=[1] dtype=fp32 layout=contiguous memory=global
tile row shape=[BLOCK_R,4] dtype=fp32 memory=register
tile row_grad shape=[BLOCK_R,4] dtype=fp32 memory=register
tile base_partial shape=[4] dtype=fp32 memory=register
scalar scale_partial dtype=fp32 memory=register

# O Operations
load row <- input_mix[row0:row0+BLOCK_R, 0:4]
load row_grad <- grad_out[row0:row0+BLOCK_R, 0:4]
compute z = row * mhc_scale + mhc_base
compute sig = sigmoid(z)
compute grad_z = row_grad * sig * (1 - sig)
compute grad_input_row = grad_z * mhc_scale
store grad_input_mix[row0:row0+BLOCK_R, 0:4] <- grad_input_row
compute base_partial = sum(grad_z, axis=0)
compute scale_partial = sum(grad_z * row)
store grad_mhc_base[0:4] <- atomic_add(grad_mhc_base[0:4], base_partial)
store grad_mhc_scale[0:1] <- atomic_add(grad_mhc_scale[0:1], scale_partial)

# C Control
parallel row0 over 2048 step BLOCK_R
guard row0 < 2048

# H Target Hints
target=triton_ascend
num_warps=4
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only change; no host-side state, cache, or lifecycle modification beyond normal per-call output allocation"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-001","intervention":"fuse the sigmoid elementwise chain and both reductions into one Triton kernel, collapsing 10 unfused library kernels to 1","expected_causal_chain":["the 10 unfused library kernels (2 ReduceSum, 5 Mul, 1 Add, 1 Rsubs, 1 Sigmoid) collapse into one fused Triton kernel","kernel launch count per call decreases from 10 to 1","host launch overhead per call decreases","wall time decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"kernel_count_per_call","expectation":"decrease from 10 to 1"},{"name":"device_us_per_call","expectation":"decrease"}],"guardrails":["correctness:pass","output shapes and dtypes unchanged","numerical semantics preserved"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`. The catalog entries (winner tree, sort-32/64 selection, dynamic `tl.gather`, cumsum compaction) all concern MLU590-H8 hierarchical top-k selection networks with heavy value/ID state and dynamic gather/prefix control flow. None of their preconditions match this operator: MhcHeadComputeMixBackward is a pure elementwise sigmoid chain plus two static full-tensor reductions over 32 KB on Ascend910B4, with no dynamic indexing, no sort/select networks, and no on-chip gather. No catalog entry invalidates this fusion path.
- Reduction-accumulator correctness: the two reductions (`sum((0,1))` and `sum((0,1,2))`) must match `keepdim=True` then `.view(-1)`/`.view(1)` exactly. The fused kernel accumulates block partials via `atomic_add` into `grad_mhc_base[4]` and `grad_mhc_scale[1]`; those two outputs MUST be zero-initialized (e.g. `torch.zeros`) before the kernel launches each forward, or the atomic accumulation is incorrect.
- Shape equivalence: the reference `grad_z.sum((0,1), keepdim=True).view(-1)` yields `[4]` and `grad_z.sum((0,1,2), keepdim=True).view(1)` yields `[1]`. Treating the tensor as `[2048,4]` (since `2*1024=2048`), the first reduction is a per-column sum over axis 0 (producing `[4]`) and the second is a full sum over all 8192 elements (producing `[1]`). Both must be returned as `[4]` and `[1]` fp32, exactly.
- Broadcast note: `mhc_base[4]` broadcasts along the trailing (column) axis only; `mhc_scale[1]` is scalar-like and broadcasts everywhere. The fused `z = input_mix * mhc_scale + mhc_base` must preserve this alignment.
- A single fused kernel is selected (rather than a two-kernel split) because total data is only 32 KB (8192 elements), so one kernel with a small row grid and a handful of atomic contributions is both correct and minimal-launch. If Ascend Triton lowers `tl.atomic_add` on the tiny `[4]`/`[1]` accumulators with unacceptable overhead, the Coder may fall back to a two-kernel split (elementwise kernel writing `grad_input_mix` plus a `grad_z` staging buffer, then a single reduction kernel producing both `[4]` and `[1]`), which still collapses 10 kernels to 2 and remains inside this decision's allowed change boundary.

## Rationale and Evidence

Verifier report `rounds/report_000.md` establishes the baseline as host-bound: wall ≈ 434 us/call vs device ≈ 41 us/call (`device_ratio ≈ 0.095`), so ~91% of wall time is host-side. The forward decomposes into 10 unfused library kernels per call (2 `aclnnReduceSum`, 5 `aclnnMul`, 1 `aclnnAdd`, 1 `aclnnRsubs`, 1 `aclnnSigmoid`). The two `ReduceSum` kernels alone account for ~22.5 us/call (~55% of all device time) and the remaining elementwise kernels are tiny. Because wall time is dominated by kernel-launch and host dispatch overhead rather than device compute, collapsing 10 kernels into a single fused Triton kernel directly removes 9 launch/host overheads and folds the two reduction kernels into the elementwise pass, which is the single attributable mechanism expected to move wall time. This mirrors the music_rotary 14-to-1 fusion precedent (+46%) and is a pure kernel dataflow change: no host-side caching, allocation reuse, or lifecycle state is introduced, so the Host Plan is not applicable.

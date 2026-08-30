# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"cuda","target_profile":"triton_cuda","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"kernel-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"mixed","intervention":"fuse the deterministic rigid-transform chain (centering subtraction, quaternion-to-rotation-matrix construction, 3x3-by-3 vector product, translation, and mask multiply) into a small number of Triton kernels while leaving the random number draws (torch.rand/randn) unchanged on the host so the RNG consumption order stays identical to the reference","allowed_changes":["kernel dataflow","kernel count"],"invariants":["ModelNew public contract","output dtype and shape","random quaternion and translation construction","RNG consumption order (3x torch.rand + 1x torch.randn inside forward)","centering formula with eps=1e-12"],"expected_wall_improvement_pct":15.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor x_input_coords shape=[256,3] dtype=fp32 layout=row_major memory=global
tensor mask shape=[256] dtype=fp32 layout=contiguous memory=global
tensor R shape=[4,3,3] dtype=fp32 layout=row_major memory=global
tensor T shape=[4,3] dtype=fp32 layout=row_major memory=global
tensor out shape=[4,256,3] dtype=fp32 layout=row_major memory=global
scalar eps value=1e-12

# O Operations
load xc <- x_input_coords[0:256,0:3]
load m <- mask[0:256]
compute center = sum(xc * m[:, None]) / (sum(m) + eps)
compute x_centered = xc - center
load r <- R[sample,0:3,0:3]
load t <- T[sample,0:3]
compute rotated = r @ x_centered  (3x3 times 3-vector, per atom)
compute translated = rotated + t
compute masked = translated * m[:, None]
store out[sample,0:256,0:3] <- masked

# C Control
parallel sample over 4
parallel atom over 256
guard atom < 256

# H Target Hints
target=triton_cuda
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only change; the random number draws torch.rand/torch.randn remain ordinary host-dispatched torch calls inside forward, with no new host-side state, cache, allocation reuse, or lifecycle semantics introduced"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-001","intervention":"fuse the deterministic rigid-transform chain (centering subtraction, quaternion-to-rotation-matrix construction, 3x3-by-3 vector product, translation, and mask multiply) into a small number of Triton kernels while leaving the random number draws (torch.rand/randn) unchanged on the host so the RNG consumption order stays identical to the reference","expected_causal_chain":["the ~70 deterministic elementwise/reduce/transcendental/cat/copy kernels collapse into a few fused Triton kernels","kernel count per call drops from ~79 toward single digits","device launch overhead and device kernel time both decrease","wall time decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"kernel_count_per_call","expectation":"decrease"},{"name":"device_us_per_call","expectation":"decrease"}],"guardrails":["correctness:pass","output dtype and shape unchanged","RNG consumption order preserved","random quaternion and translation construction numerically compatible"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`. The catalog entries (winner tree, sort networks, dynamic gather, cumsum compaction) all concern grouped-topk expert selection on MLU590-H8 and do not match this operator's shape, dtype, or runtime fingerprint, so no listed failure invalidates this path.
- The highest-risk element of this intervention is not the math but the randomness boundary. The anti-pattern that does apply here is a "silent RNG-order change": if Coder reorders, adds, removes, or changes the distribution of the `torch.rand`/`torch.randn` draws inside `forward`, the harness's identical per-call `set_seed(42)` will no longer produce bit-comparable `R`/`T`, and `allclose` will fail even with correct math. This decision therefore explicitly keeps the four random draws (3 uniform for the quaternion + 1 normal for the translation) as the exact same host-side `torch` calls in the same order, and fuses only the downstream deterministic transform.
- The `target_profile` `triton_cuda` profile records that `tl.load`, `tl.store`, `tl.arange`, elementwise `tl.sqrt`/`tl.exp`-class math, and reductions over `(256,)` are supported, but `num_warps`, `num_stages`, block pointers, and mixed precision remain `Unknown` on this profile revision. The sketch therefore uses only proven primitives and leaves launch hints non-normative; `sin`/`cos` and `sqrt` transcendental support is carried over from the observed `sin_kernel_cuda`/`cos_kernel_cuda`/`sqrt_kernel_cuda` reference kernels, but Coder should treat any unsupported transcendental as a capability-miss, not assume NVIDIA behavior.

## Rationale and Evidence

The Phase 0 baseline (`rounds/report_000.md`) measured `baseline_adapter.py` at wall median `1.073250 ms` with `420.684 us/call` device time and `78.8 kernels/call`, giving `device_ratio ≈ 0.392`. This classifies the operator as **mixed** (leaning host-bound): only ~39% of wall time is device kernel execution, and ~61% is host/launch overhead, which is the expected signature of a forward that produces only `[4,256,3] = 3072` output elements yet launches ~79 tiny kernels.

The Level 1 kernel breakdown shows the device time is dominated by many small elementwise kernels: binary/unary `mul` (the quaternion-to-matrix entries and `rot_vec_mul` products), `add` (matrix-entry sums and the `+ T` translation), `reduce sum` (the masked centering `sum(dim=-2)`), `sqrt`/`sin`/`cos` (the quaternion construction), `cat`/`copy` (the `torch.stack` and `.contiguous()`), and the random distribution kernels (3 uniform + 1 normal). The reference dataflow is a fixed sequence of ~79 elementwise/reduce/transcendental/copy kernels that can be collapsed into a small constant number of fused Triton kernels without changing any numerical result.

The randomness boundary is the decisive constraint. The four random draws are tiny (`torch.rand(4)` × 3 and `torch.randn(4,3)` × 1), and their host-side distribution kernels are only ~15.58 + 5.40 ≈ 21 us/call of device time — a small fraction of the 420 us. They must, however, stay byte-for-byte order-identical to the reference because the harness re-seeds to `42` before every forward and the comparison requires identical `R`/`T`. Leaving these draws on the host (unchanged `torch` calls) preserves correctness with zero risk, while fusing the deterministic transform (centering, quaternion→matrix, `rot_vec_mul`, translation, mask) targets the ~70 remaining kernels and the bulk of the launch overhead.

The expected wall gain is damped but still well above the 5% adoption threshold. Fusing ~79 kernels into roughly 4-8 kernels removes ~70 host-side launch dispatches per forward (the dominant cost in a host-bound operator) and collapses ~400 us of device elementwise time into a few tens of microseconds. Even conservatively assuming host time is only partially compressible and device time drops by a large fraction, a wall reduction in the 15% range is a defensible expectation; the Evaluation Contract's `primary_metric.expected_improvement_pct` remains fixed at `5.0` as the adoption threshold, which this intervention is expected to clear comfortably. The canonical comparison source is `baseline_adapter.py` under measurement fingerprint `a5f980780c4dcde731df913710ad9dfded4f07a66b90e334fea0a6f2aa1fd5fa`.

# Decision 002

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"002","reference_implementation":"triton_centre_random_augmentation_001.py","reference_report":"rounds/report_001.md","language":"triton","backend":"cuda","target_profile":"triton_cuda","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"kernel-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"mixed","intervention":"extend the fused Triton kernel to absorb the quaternion-to-rotation-matrix construction (sqrt/sin/cos and the 9-entry matrix arithmetic) from u1/u2/u3, so the ~48 host-side transcendental/elementwise/stack kernels collapse into the single existing _centre_aug_kernel; the random draws (3x torch.rand for u1/u2/u3, 1x torch.randn for T) remain unchanged host-side calls so the RNG consumption order stays bit-identical to the reference","allowed_changes":["kernel dataflow","kernel count"],"invariants":["ModelNew public contract","output dtype and shape","RNG consumption order (3x torch.rand + 1x torch.randn inside forward)","centering formula with eps=1e-12","quaternion-to-rotation-matrix construction numerically compatible within atol=1e-2"],"expected_wall_improvement_pct":12.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor x_input_coords shape=[256,3] dtype=fp32 layout=row_major memory=global
tensor mask shape=[256] dtype=fp32 layout=contiguous memory=global
tensor u1 shape=[4] dtype=fp32 layout=contiguous memory=global
tensor u2 shape=[4] dtype=fp32 layout=contiguous memory=global
tensor u3 shape=[4] dtype=fp32 layout=contiguous memory=global
tensor T shape=[4,3] dtype=fp32 layout=row_major memory=global
tensor out shape=[4,256,3] dtype=fp32 layout=row_major memory=global
scalar pi value=3.141592653589793
scalar eps value=1e-12

# O Operations
load u1s <- u1[sample]
load u2s <- u2[sample]
load u3s <- u3[sample]
compute q1 = sqrt(1 - u1s) * sin(2 * pi * u2s)
compute q2 = sqrt(1 - u1s) * cos(2 * pi * u2s)
compute q3 = sqrt(u1s) * sin(2 * pi * u3s)
compute q4 = sqrt(u1s) * cos(2 * pi * u3s)
compute r = quaternion_to_matrix(q1, q2, q3, q4)
load xc <- x_input_coords[0:256,0:3]
load m <- mask[0:256]
compute center = sum(xc * m[:, None]) / (sum(m) + eps)
compute x_centered = xc - center
compute rotated = r @ x_centered
compute translated = rotated + T[sample, :]
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
{"applicability":"not-applicable","reason":"kernel-only change; the random draws torch.rand (u1/u2/u3) and torch.randn (T) remain ordinary host-dispatched torch calls in the exact reference order inside forward, with no new host-side state, cache, allocation reuse, or lifecycle semantics introduced"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-002","intervention":"extend the fused Triton kernel to absorb the quaternion-to-rotation-matrix construction (sqrt/sin/cos and the 9-entry matrix arithmetic) from u1/u2/u3, so the ~48 host-side transcendental/elementwise/stack kernels collapse into the single existing _centre_aug_kernel; the random draws (3x torch.rand for u1/u2/u3, 1x torch.randn for T) remain unchanged host-side calls so the RNG consumption order stays bit-identical to the reference","expected_causal_chain":["the ~48 host-side transcendental/elementwise/stack/copy kernels for the quaternion-to-matrix conversion collapse into the single fused kernel","kernel count per call drops from ~55 toward single digits","device launch overhead and device kernel time both decrease","wall time decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"kernel_count_per_call","expectation":"decrease"},{"name":"device_us_per_call","expectation":"decrease"}],"guardrails":["correctness:pass","output dtype and shape unchanged","RNG consumption order preserved","quaternion-to-rotation-matrix construction numerically compatible within atol=1e-2"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`. The catalog entries (winner tree, sort networks, dynamic gather, cumsum compaction) concern grouped-topk expert selection on MLU590-H8 and do not match this operator's shape, dtype, or runtime fingerprint, so no listed failure invalidates this path.
- The dominant risk here is **capability, not correctness**. `tl.sqrt`, `tl.sin`, and `tl.cos` are absent from the `triton_cuda` profile's Supported list (only `tl.exp` is proven); the profile marks transcendental coverage beyond `tl.exp` as unproven on the CoreX Triton 3.1.0 BI150 backend. The reference `base.py` does use `torch.sqrt/sin/cos` (which lower to `sqrt_kernel_cuda`/`sin_kernel_cuda`/`cos_kernel_cuda`, all observed in the profiler), so the hardware/runtime supports these functions at the torch level, but the Triton lowering of `tl.sqrt/sin/cos` is not yet locally proven. If they fail to lower, the outcome is a `capability-miss` (a rejected round, not a correctness violation); this is an acceptable risk given the clear ≥5% benefit. The Coder must treat any transcendental that does not lower as a capability-miss and report it rather than substituting an unproven approximation.
- The numerical divergence between `tl.sqrt/sin/cos` and torch is expected to be ~1 ulp (both lower to the same libdevice `__fsqrt_rn`/`__sinf`/`__cosf` intrinsics on the CUDA backend), and the propagation through the bounded quaternion→matrix→`rot_vec_mul` chain is ~1e-7, far inside `atol=1e-2`. No amplification is expected because `q ∈ [-1,1]`, matrix entries ∈ `[-1,1]`, and `rot_vec_mul` is a bounded 3-term sum. If the Coder observes a larger-than-expected divergence in a local probe, it must be reported as a `major-deviation` rather than silently accepted.
- The RNG boundary remains the hard invariant: `u1,u2,u3` must still be drawn by three `torch.rand` calls and `T` by one `torch.randn` call, in that exact order, before the fused kernel runs. The fused kernel only reads `u1/u2/u3/T` and performs deterministic math; it must not itself draw random numbers.

## Rationale and Evidence

Round 001 (`accepted`, wall `1.023173 → 0.712600 ms`, +30.35%) fused the centering + `rot_vec_mul` + translation + mask chain into `_centre_aug_kernel` at only `6.56 us/call`, but the quaternion→rotation-matrix construction (`sqrt`/`sin`/`cos`/`mul`/`add`/`stack`/`cat`) was deliberately left on the host to preserve bit-identical `R`. The result is that the candidate still launches `54.8 kernels/call` with `237.95 us/call` device time and `device_ratio ≈ 0.334` — the operator remains host/launch-bound.

The candidate-scope top kernels show the remaining cost is concentrated in the host-side quaternion→matrix conversion: `mul` unary (54.74 us), `mul` binary (52.42 us), `add` (36.34 us), `add` other (19.71 us), `sqrt` (16.05 us), `sin` (11.30 us), `cos` (11.27 us), `cat` batched copy (8.43 us) — roughly `~210 us/call` of device time across ~48 kernels. All of these operate on tiny tensors (`u1/u2/u3 ∈ [4]`, quaternion `[4]`, `R ∈ [4,3,3]` = 36 elements), so the ~210 us is almost entirely per-kernel launch overhead, not computation. The random-distribution kernels (`rand` 15.37 us, `randn` 5.17 us) must remain host-side and are not part of this fusion target.

Fusing the quaternion→matrix construction into the single Triton kernel removes ~48 host kernel launches, collapsing both the ~210 us of device launch overhead and the associated ~48 host-side dispatch costs. Round 001 established an empirical multiplier: it removed ~24 kernels and ~173 us of device time to save ~310 us of wall time (wall savings ≈ 1.79× device savings). Applying a conservative version of that relationship to the ~48 remaining removable kernels yields a defensible `expected_wall_improvement_pct` of `12.0`, comfortably above the `5.0` adoption threshold (which remains fixed in the Evaluation Contract's `primary_metric`).

The numerical risk is bounded: the random `u1/u2/u3/T` values stay bit-identical (drawn by the same host-side `torch.rand`/`torch.randn` in the same order), so only the deterministic `sqrt`/`sin`/`cos`/`mul`/`add` on top of identical `u` values moves into Triton. On the CUDA backend these map to the same libdevice intrinsics as torch, so divergence is ~1 ulp (~1e-7), and the bounded quaternion→matrix→`rot_vec_mul` chain does not amplify it — a ~1e-7 output perturbation against `atol=1e-2` leaves a ~5-order-of-magnitude safety margin. The genuine uncertainty is whether `tl.sqrt`/`tl.sin`/`tl.cos` lower on the CoreX Triton 3.1.0 BI150 backend (they are unproven on this profile); a failure there is a `capability-miss` (rejected round), not a correctness violation, and is a tolerable risk given the clear ≥5% benefit. The canonical comparison source is `triton_centre_random_augmentation_001.py` under measurement fingerprint `a5f980780c4dcde731df913710ad9dfded4f07a66b90e334fea0a6f2aa1fd5fa`.

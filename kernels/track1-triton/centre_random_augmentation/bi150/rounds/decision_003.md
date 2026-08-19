# Decision 003

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"003","reference_implementation":"triton_centre_random_augmentation_002.py","reference_report":"rounds/report_002.md","language":"triton","backend":"cuda","target_profile":"triton_cuda","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-change"}
```

## Optimization Intent

```json
{"bottleneck_class":"measurement-bound","intervention":"no stable intervention clears the 5% adoption threshold: the deterministic transform is already fused into a single Triton kernel, and the remaining wall time is dominated by the irreducible host-side RNG draws (3x torch.rand + 1x torch.randn, a hard RNG-order invariant) plus the single kernel launch and harness-fixed overhead","allowed_changes":[],"invariants":["ModelNew public contract","output dtype and shape","RNG consumption order (3x torch.rand + 1x torch.randn inside forward)","centering formula with eps=1e-12","quaternion-to-rotation-matrix construction numerically compatible within atol=1e-2"],"expected_wall_improvement_pct":0.0}
```

## Unified Sketch

N/A: aborted

## Host Plan

```json
{"applicability":"not-applicable","reason":"aborted"}
```

## Evaluation Contract

```json
{"applicability":"not-applicable","reason":"aborted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`; no matching entry invalidates this stop. The catalog concerns grouped-topk selection on MLU590-H8 and does not apply.
- The one remaining "extra" kernel is the `s_trans * torch.randn(...)` scalar multiply (candidate 002 line 209). With `s_trans = 1.0` it is a numerical no-op but still a host launch (~0.92 kernels/call, ~3.6 us/call device). Folding it into the fused kernel (passing `s_trans` as a scalar and multiplying `T` inside) would remove exactly one host launch, worth well under 1.5% of the 0.239 ms wall — far below the 5% threshold. It is not a falsifiable ≥5% intervention.
- The remaining device work is the fused `_centre_aug_kernel` (6.81 us/call) plus the four mandatory RNG distribution kernels (`rand` ~13.84 us + `randn` ~4.82 us). The RNG draws cannot be reduced, reordered, or moved into the kernel because the harness's per-call `set_seed(42)` comparison requires the candidate to consume the RNG in the exact reference order (3× `torch.rand` + 1× `torch.randn`) — this is a hard invariant recorded in `project.md` and every prior decision.

## Rationale and Evidence

Round 002 (`accepted`, wall `0.711623 → 0.239284 ms`, +66.37%) fused the entire deterministic transform — quaternion construction (`sqrt`/`sin`/`cos`), the 9-entry rotation-matrix arithmetic, centering, `rot_vec_mul`, translation, and mask multiply — into a single `_centre_aug_kernel`. The profiler confirms the remaining candidate-scope kernels are only the irreducible RNG draws and one stray launch:

- `rand` uniform (2.74/call, 13.84 us): the three mandatory `torch.rand` draws for `u1/u2/u3`.
- `randn` normal (0.92/call, 4.82 us): the single mandatory `torch.randn` draw for `T`.
- FUSED `_centre_aug_kernel` (0.92/call, 6.81 us): the entire deterministic transform.
- `elementwise mul` unary (0.92/call, 3.60 us): the `s_trans * torch.randn(...)` scalar multiply (a no-op at `s_trans=1.0`).
- one stray `cat` (0.02/call): a scope-boundary artifact.

`kernel_count_per_call` fell `54.92 → 5.52`, `device_us_per_call` `238.19 → 29.24`, and `device_ratio` dropped to `0.122`, so the operator is now strongly host-bound at a floor set by the four mandatory RNG draws and the single fused-kernel launch. The ~210 us of host time that remains is dominated by the host-side dispatch of those four RNG draws (a hard RNG-order invariant), the single Triton launch, and harness-fixed `set_seed`/`sync_devices`/`clone_value` overhead that no role may change.

The only conceivable further kernel change — folding the numerical no-op `s_trans` multiply into the kernel — removes a single host launch and is worth well under 1.5% of wall time, not the 5% required for adoption. No host-side allocation or launcher reduction remains available without violating the RNG-order invariant. Per `references/bottleneck-judgment.md`, the normalized evidence shows remaining device work (29.24 us/call) is near the irreducible single-kernel floor and the residual host time is harness-fixed, which is the definition of a measurement-bound stop. The canonical comparison source is `triton_centre_random_augmentation_002.py` under measurement fingerprint `a5f980780c4dcde731df913710ad9dfded4f07a66b90e334fea0a6f2aa1fd5fa`.

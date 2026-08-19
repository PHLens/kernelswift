# Decision 002

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"002","reference_implementation":"triton_mhc_post_layer_mix_001.py","reference_report":"rounds/report_001.md","language":"triton","backend":"cuda","target_profile":"triton_cuda","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-change"}
```

## Optimization Intent

```json
{"bottleneck_class":"none","intervention":"no stable intervention clears the 5% adoption threshold: the remaining bottleneck is a memory-bound narrow batched GEMM (M=4, K=4, N=1280) that tl.dot cannot beat, and the fused tail plus residual cast are already near-optimal and separated by a cublasLt library call","allowed_changes":[],"invariants":["ModelNew public contract","output dtype and shape","fp32 intermediate precision","einsum term2 result unchanged"],"expected_wall_improvement_pct":0.0}
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

- Consulted `references/anti-patterns.md`; no catalog entry directly matches, but
  the relevant preconditions are evaluated and reject the only remaining
  candidate. The remaining bottleneck is a narrow `[4,4]@[4,1280]` batched GEMM
  (contraction dim 4). A `tl.dot` rewrite would require padding M=4 and K=4 up to
  the proven 16x16/32x32 tile, wasting roughly 16x compute on a GEMM that is
  already memory-bound, without reducing any of the ~250 MB of A/B read and C
  write traffic. The post-GEMM elementwise tail and the `residual.float()` input
  cast cannot be fused further: they are separated by the `cublasLt` GEMM, a
  library call outside any Triton kernel's boundary. No anti-pattern authorizes
  forcing a proceed when the only named mechanism is expected to regress.

## Rationale and Evidence

report_001.md confirms Round 001 accepted with 20.09% wall improvement, leaving a
device ratio of ~0.95 and three kernels: the unchanged TCU batched GEMM
(`gemm_tcu_h`, `5183.49 us/call`, ~85% of remaining device time), the fused
Triton tail (`_fused_tail_kernel`, `496.18 us/call`), and the `residual.float()`
input cast (`direct_copy_kernel_cuda`, `442.86 us/call`).

The dominant cost is the `torch.einsum('abmn,abmc->abnc',
comb_res_mix, residual.float())` GEMM, which lowers to an Iluvatar TCU batched
GEMM over 8192 batches of `[4,4]@[4,1280]`. This GEMM is memory-bound, not
compute-bound: per-batch FLOPs are only `2*4*1280*4 = 40960`, while the A/B/C
memory traffic totals roughly 250 MB across the batch. The `tl.dot` primitive is
now Supported (per `scripts/bi150_tl_dot_probe2.py` and
`scripts/bi150_tl_dot_probe_bf16.py`, profile updated), but only for `(32,32)@(32,32)`
(and `(16,16)`-tile) shapes; the GEMM's M=4 and K=4 are far below that tile, so a
`tl.dot` rewrite must pad M and K to the tile size, wasting ~16x compute while
adding mask and padding overhead, and it does not reduce the memory traffic that
dominates the 5183 us. It is expected to regress, not improve.

The remaining ~15% (fused tail + residual cast) cannot be fused further: the
`residual.float()` cast feeds the GEMM and the fused tail consumes the GEMM
output, with the `cublasLt` library call sitting between them outside any Triton
kernel's boundary. Eliminating the `residual.float()` cast by feeding bf16
directly would depend on the unverified assumption that cublasLt's internal
bf16-to-fp32 promotion is numerically identical to the explicit `.float()` and
would change the explicit reference semantics; at best it saves ~6.9% of wall
time, which is not a robust 5% falsifiable win given that uncertainty. The device
ratio is already ~0.95, leaving no host-side headroom.

Therefore no falsifiable intervention is expected to clear the 5% unrounded
median wall-time threshold against `triton_mhc_post_layer_mix_001.py`. The honest
decision is to abort.

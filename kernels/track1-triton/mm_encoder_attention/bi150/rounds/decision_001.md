# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"cuda","target_profile":"triton_cuda","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-change"}
```

## Optimization Intent

```json
{"bottleneck_class":"measurement-bound","intervention":"no stable intervention clears the adoption threshold; device time is a single vendor-tuned fused flash-attention kernel and the remaining host time is harness-fixed","allowed_changes":[],"invariants":["ModelNew public contract","output dtype and shape","benchmark semantics"],"expected_wall_improvement_pct":0.0}
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

- Consulted `references/anti-patterns.md`. No single entry is a direct precondition
  match (the catalog records MLU590-H8 grouped-topk selection-network failures),
  but the same structural lesson governs this round: a hand-written Triton kernel
  that re-implements a vendor library kernel's dataflow cannot be justified to
  beat that library kernel without a matching primitive probe. Here `tl.dot`
  (the matrix-multiply primitive required to express flash-attention QK^T and
  PV) is explicitly marked `Unknown` in `prompts/coder_targets/triton_cuda.md`,
  with no qualifying BI150 probe. Writing a Triton flash-attention to outrun the
  vendor's `FlashAttnFwdF16Ixmma` tensor-core kernel therefore lacks even the
  foundational primitive evidence, mirroring the groupedtopk `torch.topk`
  library-kernel dead end.

## Rationale and Evidence

The baseline (report_000.md) establishes that 100% of device time is a single
fused `FlashAttnFwdF16Ixmma<128,128,16,64,64,Causal=0,Alibi=0>` flash-attention
kernel at `14.949 us/call` with `0.86 kernels/call` (sub-1.0 is a scope-boundary
sampling artifact). There is no `bmm`/`softmax` decomposition and no routing or
wrapper kernel inside the candidate boundary to fuse or remove — the attention
is already maximally fused into one vendor-tuned Ixmma tensor-core kernel. The
device side offers no compressible structure.

`device_ratio ≈ 0.099` classifies the workload as host-bound by the
`references/bottleneck-judgment.md` heuristic (<20%), but the ~90% of wall time
outside the kernel is harness-fixed rather than candidate-owned: the harness
`time_forward` path runs `set_seed(seed)` (which iterates all accelerators via
`manual_seed_all`) plus `sync_devices()` (a full `torch.cuda.synchronize()`) on
every timed call, before and after `one_call`, and clones inputs. These are
outside `ModelNew.forward` and outside any candidate change boundary; the
benchmark regime requires reference and candidate to share the identical harness
and synchronization boundary. The `forward` body itself contributes only
zero-copy `view`/`transpose`/`reshape` view operations and a single SDPA
dispatch, so there is no meaningful candidate-owned host work to compress.

A `torch.compile`-driven dispatch-reduction hypothesis is not falsifiable at the
5% threshold in this regime: the dominant host cost is fixed seed plus device
synchronization (~136 us of a ~151 us wall), not per-op dispatch, and the
triton_cuda profile records `torch.compile` only for trivial file-backed CUDA
add-one functions, with attention-graph coverage and the reduce-overhead cache
behavior under this harness explicitly unproven. The expected host savings
cannot be shown to clear the 5% adoption threshold, so the intervention is not
defensible.

Per `references/bottleneck-judgment.md`, a stop is measurement-bound when the
device work is below the stated bound and the remaining host time is
harness-fixed. Both hold: device work is a single non-decomposable kernel, and
the remaining host time is harness seed/synchronization that the candidate must
not alter. No falsifiable intervention is expected to improve unrounded median
wall time by at least 5% against `baseline_adapter.py`, so this round aborts
rather than manufacturing an unprovable proceeding decision.

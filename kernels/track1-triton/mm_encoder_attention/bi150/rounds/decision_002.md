# Decision 002

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"002","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"cuda","target_profile":"triton_cuda","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-change"}
```

## Optimization Intent

```json
{"bottleneck_class":"host-bound","intervention":"no Triton flash-attention rewrite clears the 5% adoption threshold: tl.dot correctness is now proven but its BI150 tensor-core performance is unverified, device time is a single vendor-tuned Ixmma kernel, and ~90% of wall is harness-fixed host overhead outside the candidate boundary","allowed_changes":[],"invariants":["ModelNew public contract","output dtype and shape","benchmark semantics"],"expected_wall_improvement_pct":0.0}
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

- Consulted `references/anti-patterns.md`. The catalog records MLU590-H8
  grouped-topk failures where a hand-written Triton kernel that re-implements a
  vendor library kernel's dataflow regressed against that library kernel. The
  structural lesson applies directly here: re-implementing the vendor's
  `FlashAttnFwdF16Ixmma` flash-attention dataflow in Triton cannot be justified
  to beat it without a matched performance probe of the matrix-multiply
  primitive on the actual backend. The Round 001 blocking precondition (`tl.dot`
  Unknown) is now removed — `tl.dot` is proven *correct* at a `(32,32) @ (32,32)`
  tile — but correctness at one tile shape is not a performance claim, and no
  probe yet shows that `tl.dot` lowers to the Ixmma tensor-core MMA or that a
  tiled online-softmax attention reaches the vendor kernel's device time. The
  anti-patterns requirement to demonstrate a resource-bounded primitive win
  before committing remains unmet.

## Rationale and Evidence

The Round 001 abort precondition is partially overturned: Orchestrator now
records `tl.dot` as Supported in `prompts/coder_targets/triton_cuda.md`, with
`(32,32) @ (32,32)` fp32 matmul at exact `0.0` max abs err and bf16-input fp32
accumulate at `9.5e-7` max abs err. A Triton flash attention is therefore no
longer a capability-miss. However, this does not open a falsifiable ≥5% wall
improvement, for three evidence-backed reasons.

First, the workload is host-bound with harness-fixed overhead. From
`report_000.md`, wall median is `0.151139 ms` (= `151.139 us`) while device time
is `14.9492578125 us/call` at `0.86 kernels/call`, giving `device_ratio ≈
0.099`. The remaining ~136 us of wall is outside `ModelNew.forward`: the harness
`time_forward` path (read directly from `auto_bench.py`) runs `set_seed(seed)`
(which iterates `manual_seed_all` over every accelerator) plus `sync_devices()`
(a full `torch.cuda.synchronize()`) on every timed call, and clones inputs. These
are harness-fixed and outside any candidate change boundary; the benchmark
regime requires reference and candidate to share the identical harness and
synchronization boundary. The upper bound on any device-only optimization is
therefore the device time itself: even a hypothetical zero-cost Triton attention
could only remove 14.949 us of 151.139 us, and the candidate must not touch the
~136 us host side.

Second, the device time is already a single vendor-tuned tensor-core kernel. The
entire QK^T + softmax + PV computation is fused into one
`FlashAttnFwdF16Ixmma<128,128,16,64,64,Causal=0,Alibi=0>` kernel — an Iluvatar
Ixmma (vendor matrix-multiply hardware) flash-attention kernel with fp16
elements. There is no bmm/softmax decomposition, no routing kernel, and no
compressible structure inside the candidate boundary. Direction A (a naive
decomposed QK^T + softmax + PV Triton attention) would replace one fused kernel
with at least three kernels and materialize intermediate scores, strictly
increasing device time and kernel count. Direction B (a tiled online-softmax
Triton flash attention) must beat an Ixmma tensor-core kernel whose 14.949 us
device time is already vendor-optimized.

Third, the primitive evidence is correctness-only, not performance. The
`tl.dot` probe establishes numerical correctness at one square tile shape only;
it does not establish that `tl.dot` on the BI150 lowers to the Ixmma tensor-core
MMA, nor what device time a `(83,83,64)` attention tile achieves, nor whether
`num_warps`/`num_stages`/block pointers (all still Unknown in the profile) are
available to shape a competitive kernel. The profile explicitly states: "Do not
assume ... vendor-specific trace fields ... until a local scoped export proves
them," and `num_warps`/`num_stages` remain Unknown. Writing a Triton flash
attention to outrun the vendor Ixmma kernel therefore lacks the foundational
performance evidence the anti-patterns catalog requires.

The interaction of these three facts is decisive: the only compressible region
(device, 14.949 us) is a vendor tensor-core kernel that an unproven-performance
Triton rewrite has no evidenced path to beat, while the dominant ~90% of wall is
harness-fixed host overhead the candidate must not alter. Even a perfect device
win cannot reliably clear the 5% adoption threshold (`151.139 us` → would need
`≤ 143.58 us`, requiring the entire 14.949 us device time to be eliminated with
no added host launch cost — an impossibility for a Triton kernel that introduces
its own JIT/launch overhead on first call and per-call). The deliverable value of
a correct Triton attention is real but is not the adoption criterion, which is
strictly a ≥5% unrounded median wall improvement against `baseline_adapter.py`.

Per `references/bottleneck-judgment.md`, a stop is justified when normalized
evidence shows remaining device work below the stated bound and targeted host
evidence shows the remainder is harness-fixed. Both hold and are unchanged from
Round 001; the `tl.dot` correction removes only the capability-miss, not the
host-bound bottleneck. No falsifiable intervention is expected to improve
unrounded median wall time by at least 5% against `baseline_adapter.py`, so this
round aborts rather than manufacturing an unprovable proceeding decision.

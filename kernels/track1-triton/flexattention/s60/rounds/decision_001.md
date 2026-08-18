# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"gcu","target_profile":"triton_gcu","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-change"}
```

## Optimization Intent

```json
{"bottleneck_class":"none","intervention":"no stable intervention clears the 5% adoption threshold: the eager SDPA path is already a single fused CNNL kernel and a hand-written Triton kernel cannot beat it on device time","allowed_changes":[],"invariants":["ModelNew public contract","output dtype and shape","numerical semantics","benchmark semantics (unchanged harness, seed setup and GCU synchronization preserved)"],"expected_wall_improvement_pct":0.0}
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

- Consulted `references/bottleneck-judgment.md`; the Phase 0 trace shows exactly 1 GCU runtime launch (`topsLaunchKernel`) per forward call, so the kernel-fusion change family has no remaining kernel-count headroom (unlike the groupedtopk case, which had 12 launches to fuse).
- Consulted `references/anti-patterns.md`; no matching failure invalidates this stop, but the prior groupedtopk host-only round (output cache / device-context removal) gained only ~2.06%, below the 5% adoption threshold, which preconditions against a host-side intervention here.
- Consulted `prompts/coder_targets/triton_gcu.md`; `tl.dot` was verified locally as 2D-fp32-capable, but the eager path already lowers to a fused CNNL flash-attention kernel that a naive hand-written Triton SDPA cannot match on device time.

## Rationale and Evidence

The accepted report (`rounds/report_000.md`) records that the unchanged eager `F.scaled_dot_product_attention` reference already lowers to a single fused GCU kernel: 1 `topsLaunchKernel` per forward call in both `baseline_base` and `candidate_baseline_adapter` scopes. The forward's transpose/unsqueeze/repeat_interleave/reshape operations are zero-copy views that issue no additional launches, so there is no separate routing-kernel chain to fuse.

A local same-regime probe (Orchestrator, outside the measurement regime) established the decisive fact: a hand-written Triton causal-SDPA kernel (one program per `(token, head)`, `tl.dot` for QK^T and AV) is correct (max_abs_diff 1.95e-3 < atol 1e-2) but its device execution is ~100x slower than the eager fused CNNL path (`forward+sync` ~22 ms vs ~0.15 ms), because 664 tiny programs of `[1,64]x[64,128]` dot work cannot match a fused flash-attention library kernel on this architecture. Reaching flash-attention-level device time would require online-softmax tiling whose fp16 `tl.dot` performance on GCU is unproven and whose launch count is still 1 -> 1.

The wall median is ~0.269 ms while the runtime launch is only ~10.5 us/call and GCU device duration is unavailable (`device_time_available=false`). The overwhelming remainder of wall time is harness-fixed host cost (seed setup and `torch.gcu.synchronize()`, ~85 us bare sync plus harness overhead), which is outside the allowed change boundary (base.py and the harness are immutable). A hand-written Triton kernel cannot reduce the launch count below 1 and cannot beat the fused CNNL kernel on device time.

Therefore no falsifiable intervention with an expected wall improvement of at least 5% can be justified; the correct Round 001 outcome is a measurement-bound abort.

# Decision 002

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"002","reference_implementation":"triton_fused_moe_001.py","reference_report":"rounds/report_001.md","language":"triton","backend":"gcu","target_profile":"triton_gcu","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"routing-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"host-bound","intervention":"fuse softmax, top-2, renormalize, and fp16 weight cast of routing into the per-token fused-MoE kernel so the kernel computes topk from raw router_logits in-place","allowed_changes":["kernel dataflow","kernel routing computation"],"invariants":["ModelNew public contract","output dtype and shape","routing numerical semantics (softmax, top-2, renormalize, fp16 cast)","int32 expert indexing"],"expected_wall_improvement_pct":20.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor hidden_states shape=[T,H] dtype=fp16 layout=row_major memory=global
tensor router_logits shape=[T,E] dtype=fp32 layout=row_major memory=global
tensor w1 shape=[E,2I,H] dtype=fp16 layout=row_major memory=global
tensor w2 shape=[E,H,I] dtype=fp16 layout=row_major memory=global
tensor out shape=[T,H] dtype=fp16 layout=row_major memory=global
tile token_logits shape=[E] dtype=fp32 memory=register
tile topk_vals shape=[K] dtype=fp32 memory=register
tile topk_ids shape=[K] dtype=int32 memory=register

# O Operations
load token_logits <- router_logits[token,0:E]
compute scores = softmax(token_logits)
compute topk_vals,topk_ids = repeated_argmax(scores,K)
compute topk_weights = renormalize(topk_vals)
load x <- hidden_states[token,0:H]
compute out_acc = weighted_expert_ffn(x,topk_ids,topk_weights,w1,w2)
store out[token,0:H] <- out_acc

# C Control
parallel token over T
guard token < T
for k over K

# H Target Hints
target=triton_gcu
num_warps=1
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only fusion; routing moves from eager torch ops into the existing per-token kernel, no host state, allocation, or lifecycle change"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-002","intervention":"fuse softmax, top-2, renormalize, and fp16 weight cast of routing into the per-token fused-MoE kernel so the kernel computes topk from raw router_logits in-place","expected_causal_chain":["eager softmax, topk, renormalize, and routing cast kernels disappear","runtime_launch_count_per_call decreases below 8","host overhead per call decreases","benchmark wall time decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"runtime_launch_count_per_call","expectation":"decrease from 8 to fewer launches"},{"name":"runtime_launch_us_per_call","expectation":"decrease in the GCU runtime-launch diagnostic"}],"guardrails":["correctness:pass","output dtype and shape unchanged","top-2 semantics and renormalization match base","int32 expert indexing preserved","selected GCU device and current stream are preserved"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `prompts/coder_targets/triton_gcu.md`; the top-2 argmax is implemented as `tl.max` + `tl.where(is_best, e_idx, 0)` + `tl.sum`, using only primitives marked Supported (`tl.max`, `tl.where`, `tl.sum`, `tl.exp`), avoiding `tl.argmax` entirely, whose axis-1 reduction is not proven on GCU.
- Consulted `references/anti-patterns.md` and report_001 retry history; the GCU slice-index compile failure (`gate_up[:I]`) is avoided by keeping the two independent `[I]` gate/up GEMMs from the accepted v1 kernel. No `tl.dot`, no `tl.argmax`, no `tl.int64`.
- Consulted `references/invariants.md`; the kernel-only change preserves the immutable harness, public contract, and int32 indexing invariant; no output cache or stream change is introduced.
- Consulted `references/bottleneck-judgment.md`; GCU exposes only runtime-launch diagnostics, so launch-count and launch-us are the mechanism observables, not relabeled device time.

## Rationale and Evidence

Round 1 confirmed the per-token fusion mechanism (147 to 8 launches, 10.55x wall). report_001 `evidence_for_next_round` identifies the remaining 8 launches/call as still-eager routing (softmax/topk/renorm/cast) plus weight fp16 casts. The MLU v2 reference (`triton_fused_moe_002.py`) proves this exact routing fusion (softmax + repeated-argmax top-2 + renorm, kernel reads raw `router_logits`) compiles and runs, cutting wall from 0.5638 to 0.2178 ms (~2.6x). The GCU profile supports every primitive that MLU v2 uses for the routing stage, so the same fusion is portable modulo the two GCU-specific constraints (int32 offsets instead of `tl.int64`, and two independent gate/up GEMMs instead of a `gate_up[:I]` slice). Removing the remaining eager routing launches is expected to reduce host overhead and wall time by well over the 5% threshold.

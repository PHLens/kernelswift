# Decision 001

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"maca","target_profile":"triton_maca","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"kernel-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"mixed","intervention":"replace the fixed benchmark softmax, group-max/group-top4, masked expert-top8, and renormalization chain with one direct-launch Triton-MACA program per token, ranking raw logits and normalizing only the selected eight logits after exact softmax-denominator cancellation","allowed_changes":["candidate-only top-level Triton kernel/helpers and ModelNew.forward fixed-contract dispatch","one program per token with BLOCK_E=256, direct launch, and num_warps=1","per-forward fp32 weight and int32 ID output allocation on gating_output.device","unchanged canonical PyTorch fallback copied from baseline_adapter.py for every call outside the exact fast-path guard"],"invariants":["base.py, auto_bench.py, project.md, team-state.md, and baseline_adapter.py are not modified","ModelNew constructor/forward signatures and get_inputs/get_init_inputs remain compatible with the AST loader","assert hidden_states.size(0) == gating_output.size(0) before dispatch","fast path only when T=83, H=7168, E=256, hidden_states is contiguous fp16, gating_output is contiguous fp32, both tensors are on the same CUDA-compatible device, topk=8, num_expert_group=8, topk_group=4, scoring_func=softmax, renormalize=true, routed_scaling_factor=1.0, and autograd is not required; otherwise execute the unchanged reference path","softmax is strictly monotone so raw-logit group and expert ordering is exactly the reference ordering, and after renormalize=true the common full-softmax denominator cancels exactly","final weights are computed as exp(selected_logit - max_selected_logit) divided by the sum over the eight selected exponentials, with the selected maximum shift preventing overflow","output is a tuple of contiguous [83,8] tensors on gating_output.device with weights fp32 and IDs int32; inputs are not mutated","expert groups are contiguous 32-expert blocks; group selection precedes expert selection; selected values/IDs are emitted in descending top-k order","exact reference ID behavior is required for group-boundary and expert-value ties; no epsilon, ID perturbation, or approximate tie key is allowed","caller-selected device and current stream are preserved; no cache, global mutable state, launcher replacement, device-context change, or output reuse is introduced"],"expected_wall_improvement_pct":15.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor gating_output shape=[T,256] dtype=fp32 layout=row_major memory=global
tensor output_weights shape=[T,8] dtype=fp32 layout=row_major memory=global
tensor output_ids shape=[T,8] dtype=int32 layout=row_major memory=global
tile raw_logits shape=[256] dtype=fp32 memory=register
tile grouped_logits shape=[8,32] dtype=fp32 memory=register
tile group_maxima shape=[8] dtype=fp32 memory=register
tile selected_group_mask shape=[8] dtype=int1 memory=register
tile eligible_logits shape=[256] dtype=fp32 memory=register
tile selected_logits shape=[8] dtype=fp32 memory=register
tile selected_ids shape=[8] dtype=int32 memory=register
scalar max_selected dtype=fp32
scalar selected_exp_sum dtype=fp32

# O Operations
load raw_logits <- gating_output[token,0:256]
compute grouped_logits = reshape(raw_logits,[8,32])
compute group_maxima = max(grouped_logits,axis=experts)
compute selected_group_mask = four_unrolled_argmax_and_mask_steps(group_maxima)
compute eligible_logits = where(broadcast(selected_group_mask,[8,32]),grouped_logits,-inf)
compute selected_logits,selected_ids = eight_unrolled_argmax_extract_and_mask_steps(reshape(eligible_logits,[256]))
compute max_selected = selected_logits[0]
compute selected_exp = exp(selected_logits-max_selected)
compute selected_exp_sum = sum_eight_scalars(selected_exp)
compute normalized_weights = selected_exp/selected_exp_sum
store output_weights[token,0:8] <- normalized_weights
store output_ids[token,0:8] <- selected_ids

# C Control
parallel token over T
guard token < T
guard exact_fixed_fast_path_contract
for group_rank over 4 unrolled
end
for expert_rank over 8 unrolled
end

# H Target Hints
target=triton_maca
num_warps=1
block_e=256
programs_per_token=1
direct_launch=true
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only fusion; the fixed-contract dispatch and per-forward output allocations preserve the existing lifecycle, and no cache, allocation reuse, launcher substitution, device-context change, or stream change is allowed"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-001","intervention":"replace the fixed benchmark softmax, group-max/group-top4, masked expert-top8, and renormalization chain with one direct-launch Triton-MACA program per token, ranking raw logits and normalizing only the selected eight logits after exact softmax-denominator cancellation","expected_causal_chain":["the fixed fast path replaces 15 baseline device kernels per forward, including four gatherTopK/bitonicSort launches, with one direct Triton-MACA launch","full-softmax denominator work and materialized score/group/mask intermediates disappear while selection remains exactly ordered by raw logits","candidate kernel_count_per_call, gatherTopK+bitonicSort device time, and total device_us_per_call decrease","unrounded median benchmark wall_time decreases by at least 5% against baseline_adapter.py"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"candidate_kernel_count_per_call","expectation":"decrease from 15.0 to 1.0 on the fixed fast path"},{"name":"gatherTopK_plus_bitonicSort_us_per_call","expectation":"decrease from 89.6741943359375 to 0 because those baseline library kernels disappear"},{"name":"candidate_device_us_per_call","expectation":"decrease from the canonical 147.7526708984375 us/call"},{"name":"fused_triton_kernel_count_per_call","expectation":"equal 1.0 in the separately scoped candidate profile"},{"name":"wide_argmax_capability","expectation":"the unproven 256-lane repeated argmax/extract path compiles and executes through the actual harness; otherwise classify capability-miss rather than assume support"},{"name":"tie_id_parity","expectation":"exact IDs match base.py for the fixed seeded input and targeted equal-logit cases at both the group cutoff and expert cutoff"}],"guardrails":["correctness:pass","unrounded median wall improvement is at least 5% against baseline_adapter.py under measurement fingerprint 3fe7d50260b3670756d26f003427faccb76e1e79c204b0a500a62d2eb481c809","floating weights pass atol=1e-2, rtol=1e-2, equal_nan=true and integer IDs match exactly","output tuple, [83,8] shapes, fp32/int32 dtypes, contiguous layout, and gating-output device are unchanged","fallback preserves the full public constructor and forward behavior for every call outside the exact fixed fast-path contract, including sigmoid, renormalize=false, other scaling factors/shapes/dtypes/layouts, and autograd-required calls","hidden_states token assertion and non-mutation of both inputs are preserved","caller-selected CUDA-compatible device and current stream are preserved","base.py, auto_bench.py, project.md, team-state.md, baseline_adapter.py, warmup/repeat, profiler settings, and measurement fingerprint remain unchanged"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `prompts/coder_targets/triton_maca.md`. Direct launch and
  `num_warps=1` are the only proven launcher/warp choices on this matched
  C500; `fast_libentry` is unsupported, and other warp counts, `num_stages`,
  block pointers, async copy, vectorization directives, and sorting primitives
  are not allowed by this decision.
- The 8-lane group argmax shape is probed, but a 256-lane expert argmax is not.
  This decision requires the latter as an explicit capability gate, not as a
  claimed supported fact. Failure to compile or execute that fixed shape is
  `capability-miss`; Coder must not silently switch algorithms.
- Equal-value ordering is not established for Triton-MACA and PyTorch
  `torch.topk` does not promise a stable public tie rule. Exact ID comparison
  on the harness plus targeted group-cutoff and expert-cutoff tie cases is
  mandatory. A mismatch may be repaired only inside the same raw-logit,
  single-program selection dataflow; adding epsilon/ID perturbations is
  forbidden, and inability to achieve parity is a capability miss.
- Full softmax is not approximated: strict monotonicity makes raw-logit ranking
  equivalent, and renormalization cancels the shared softmax denominator.
  The final selected-eight exponential uses a selected-logit maximum shift.
- Consulted `references/anti-patterns.md`. The MLU-specific winner tree,
  full sort-32/sort-64 network, dynamic `tl.gather` compaction, and cumsum
  compaction failures do not prove MACA behavior, but this round avoids all
  four mechanisms. The four/eight selection steps are explicitly unrolled;
  an unproven long `tl.static_range` is not required.
- Consulted `references/invariants.md` and
  `references/bottleneck-judgment.md`. The change is kernel-only, uses no
  reusable buffer or global state, preserves device/stream ownership, and
  requires separately scoped Level 1 evidence; profiler time is diagnostic,
  while unrounded wall median controls adoption.
- The AST loader removes nonliteral top-level assignments. Kernel definitions,
  imports, helpers, and any constants must be expressed in retained forms.

## Rationale and Evidence

Round 000 establishes a mixed bottleneck: the canonical
`baseline_adapter.py` median is `0.231739 ms`, its separately scoped device
time is `147.7526708984375 us/call` (device ratio about `63.76%`), and it
launches `15.0` kernels per call. The two `gatherTopK` plus two bitonic-sort
launches alone consume `89.6741943359375 us/call`, about `60.69%` of device
time. Replacing the whole routing chain with one launch attacks both the
dominant device work and launch multiplicity. A conservative 15% wall
expectation is above the 5% gate and needs only about `34.76 us/call` of wall
reduction, materially less than the observed top-k device-time opportunity.

The GCU Round 001 reference was consulted only for decision structure and the
general fusion idea. Its runtime-launch-only evidence and backend assumptions
are not transferred: this MACA decision uses the C500 report's attributable
kernel durations, the matched warp size 64, the proven direct launcher, and the
explicit capability/tie gates above.


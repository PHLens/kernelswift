# Decision 003

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"003","reference_implementation":"triton_grouped_topk_001.py","reference_report":"rounds/report_001.md","language":"triton","backend":"maca","target_profile":"triton_maca","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"value-index-reduction-fusion"}
```

## Optimization Intent

```json
{"bottleneck_class":"kernel-latency-constrained-host-bound","intervention":"in each of the eight expert-selection ranks, replace the accepted kernel's separate full-width tl.argmax plus tl.sum(tl.where(...)) value extraction with one standard Triton tl.max over the same masked 256 lanes that returns both the selected value and its left-tie-breaking index","allowed_changes":["candidate-only expert rank selection/value extraction inside _grouped_topk_fixed_kernel","for each unrolled expert rank, use tl.max(expert_remaining_i, axis=0, return_indices=True, return_indices_tie_break_left=True) or the exact standard-Triton spelling accepted by the pinned frontend","retain the same selected value for renormalization and selected index for masking/output IDs"],"invariants":["triton_grouped_topk_001.py, reference_triton_grouped_topk_001.py, base.py, auto_bench.py, project.md, and team-state.md are not modified","group-score computation, four-group selection, expert masks, eight-rank unrolling, selected-max subtraction before exp, renormalization, output assembly, and stores remain semantically and structurally unchanged except for the value/index reduction primitive","the launch grid (83,), argument order, T=83, BLOCK_E=256, and num_warps=1 remain exactly Round-001-equivalent","ModelNew constructor, forward signature, exact fixed-fast-path guard, canonical PyTorch fallback, get_inputs, and get_init_inputs remain unchanged","the fixed fast path keeps two independent fresh torch.empty outputs exactly as Round 001; the falsified Round-002 shared backing/view intervention is not carried forward","output tuple order, [83,8] shapes, fp32/int32 dtypes, contiguity, device, values, exact tie IDs, input non-mutation, per-call ownership, retained-output safety, concurrency, and cross-instance isolation remain unchanged","caller-selected CUDA-compatible device and current stream are preserved; direct launch remains the only supported launcher","256-lane value-plus-index reduction with explicit left tie break on the pinned MACA Triton frontend is a compile/runtime capability gate; unsupported spelling/lowering is capability-miss and must not be hidden by algorithm changes"],"expected_wall_improvement_pct":6.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor logits shape=[83,256] dtype=fp32 memory=global
tile expert_remaining shape=[256] dtype=fp32 memory=register
scalar selected_value dtype=fp32 memory=register
scalar selected_id dtype=int32 memory=register

# O Operations
compute Round-001-equivalent group scoring, group selection, expert mask, output assembly
compute expert_remaining = where(selected_group_mask_for_each_expert, logits, -inf)
compute selected_value,selected_id = tl.max(expert_remaining, axis=0, return_indices=True, return_indices_tie_break_left=True)
compute expert_remaining = where(expert_offsets == selected_id, -inf, expert_remaining)
compute Round-001-equivalent exp(selected_value - max_selected_value) and renormalization
store Round-001-equivalent weights and IDs

# C Control
parallel one program per token over grid=(83,)
for rank in static_range(8) apply the combined value-index reduction
guard only with the unchanged Round-001 host fast-path predicate

# H Target Hints
target=triton_maca
BLOCK_E=256
num_warps=1
direct_launch=true
combined_256_lane_value_index_reduction=capability_gate
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only change; candidate host wrapper, two independent allocations, guard, direct launch, device/current-stream behavior, canonical fallback, ownership, lifetime, aliasing, non-overlap, concurrency, and cross-instance safety must remain exactly Round 001"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-003","intervention":"in each of the eight expert-selection ranks, replace the accepted kernel's separate full-width tl.argmax plus tl.sum(tl.where(...)) value extraction with one standard Triton tl.max over the same masked 256 lanes that returns both the selected value and its left-tie-breaking index","expected_causal_chain":["eight expert ranks perform eight combined value/index reductions instead of eight argmax reductions plus eight sum reductions","removing eight full-width selected-value reductions lowers the sole device kernel time by at least 35% without adding launches or changing host work","unrounded median benchmark wall_time decreases by at least 5% against accepted Round 001"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"expert_full_width_reductions_per_token","expectation":"decrease from 16 expert-selection reductions (8 argmax plus 8 sum) to exactly 8 combined max-with-index reductions"},{"name":"expert_tl_sum_reductions_per_token","expectation":"decrease from exactly 8 to exactly 0; the final normalization sum over eight selected values remains unchanged and is not counted here"},{"name":"max_with_indices_capability","expectation":"the pinned frontend/backend compiles and correctly executes the standard 256-lane value-plus-index reduction with explicit left tie break; otherwise capability-miss"},{"name":"tie_id_parity","expectation":"exact IDs and permitted values match the frozen harness on fixed-seed inputs plus targeted group-cutoff and expert-cutoff tie probes"},{"name":"candidate_kernel_count_per_call","expectation":"remain exactly 1.0"},{"name":"candidate_device_us_per_call","expectation":"at least 35% lower than the durable accepted 10.7442822265625 us/call, i.e. no more than 6.983783447265625 us/call under a comparable targeted profile"},{"name":"launch_grid_num_warps_equivalence","expectation":"one direct launch with grid (83,), T=83, BLOCK_E=256, and num_warps=1, with no unsupported launcher or scheduling knob"},{"name":"host_allocation_source_equivalence","expectation":"retain Round 001's two independent torch.empty outputs; no shared backing, dtype view, cache, pool, or reuse from Round 002"},{"name":"fixed_host_guard_fallback_equivalence","expectation":"constructor, forward signature, exact fixed guard, and canonical PyTorch fallback are unchanged"},{"name":"reference_adapter_class_rename_only","expectation":"reference_triton_grouped_topk_001.py SHA256 is 70258939973f728858383b832e37069ec6b3d4681200cdb4e70daac42229b2f9 and its only byte diff from canonical is ModelNew renamed to Model"}],"guardrails":["correctness:pass","unrounded median wall improvement is at least 5% against triton_grouped_topk_001.py through the verified class-rename reference under measurement fingerprint 3fe7d50260b3670756d26f003427faccb76e1e79c204b0a500a62d2eb481c809","output values pass frozen tolerance and output IDs are exact, including targeted ties at group and expert cutoffs","selected values are the same logits selected by the exact IDs; normalization keeps exp(selected_value - max_selected_value) to avoid overflow","one kernel and one direct launch per call remain; no host allocation, guard, fallback, grid, T, BLOCK_E, or num_warps change","Round 002's fresh-allocation-coalescing is excluded and both independent output allocations remain canonical","caller-selected device/current stream, per-call ownership, retained-output safety, concurrency, cross-instance isolation, contiguity, shapes, dtypes, and input non-mutation are preserved","if combined max-with-index/left-tie support is absent or incorrect, report capability-miss rather than changing the algorithm","CPU/profile event durations remain inclusive diagnostics and are never added, subtracted, or used to reconstruct authoritative wall time","base.py, auto_bench.py, project.md, team-state.md, canonical/reference sources, profiler settings, and measurement fingerprint remain unchanged"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- The target profile proves `num_warps=1`, warp size 64, the accepted
  256-lane separate argmax/sum path, and direct launch. It does not prove a
  256-lane `tl.max` returning indices. Compilation and exact tie probes are
  mandatory capability evidence; generic CUDA or Triton availability is not
  evidence for this pinned MACA runtime.
- Explicit left tie breaking is required. Any combined reduction that returns a
  different equal-logit lane changes expert IDs and fails the public contract,
  even if weights are numerically equal. Group-selection ordering remains
  untouched.
- Do not introduce winner trees, full sorts, dynamic gather compaction, or
  cumulative-sum compaction. Those families are documented anti-patterns on
  another backend and are unnecessary here; this round changes one reduction
  primitive only and makes no unverified cross-backend performance claim.
- Do not carry forward Round 002's one-backing/two-view code. Its mechanism and
  safety checks passed, but its formal wall result regressed by
  `13.711567434852972%` and no Round-002 profile exists to justify a causal
  reinterpretation.
- The canonical host scope is `41.58952 us/call`, two inclusive
  `aten::empty` events total `10.03988 us/call`, and one inclusive launch
  event is `4.88562 us/call`. These nested/overlapping events cannot be summed
  or subtracted. This kernel-only hypothesis is adopted only by benchmark wall.
- Keep the stable launch envelope: `grid=(83,)`, `T=83`,
  `BLOCK_E=256`, and `num_warps=1`. No `num_stages`, block pointers,
  async operations, forced vectorization, alternate launcher, or other
  unproven target feature is authorized.

## Rationale and Evidence

Round 001 is the accepted canonical because fusing the fixed benchmark into one
token program reduced wall to `68.280 us/call` and device execution to one
`10.7442822265625 us/call` kernel. The accepted source now exposes a narrow,
countable kernel redundancy: each of eight expert ranks first reduces 256 lanes
to an ID with `tl.argmax`, then performs another 256-lane
`tl.sum(tl.where(...))` solely to recover the value at that ID. A combined
value/index reduction expresses the same selection once and removes eight
full-width expert reductions while leaving the full algorithm and launch
envelope intact.

Round 002 tested a host allocation family. Although the requested backing/view
mechanism, capability, storage, lifetime, alias, concurrency, kernel, fallback,
and correctness checks passed, its formal median was `0.081513 ms` versus
`0.071684 ms` for the concurrent accepted reference, a
`-13.711567434852972%` improvement. Since profiling was correctly skipped
after that wall miss, there is no durable attribution for the regression.
Round 003 therefore changes family to `value-index-reduction-fusion` and does
not reuse the falsified coalescing intervention.

For this kernel-only change to clear the 5% wall gate, it must save at least
`3.414 us/call` at the canonical `68.280 us/call` wall, about 31.8% of the
durable device time if host work is otherwise unchanged. The contract asks for
at least 35% device reduction and expects about 6% wall improvement. That is a
single, explicit, falsifiable hypothesis rather than a guarantee; the unrounded
benchmark median is authoritative.

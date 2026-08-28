# Decision 002

## Metadata

```json
{
  "schema_version": 2,
  "decision": "proceed",
  "decision_kind": "optimization",
  "round": "002",
  "reference_implementation": "triton_fused_moe_e2_001.py",
  "reference_report": "rounds/report_001.md",
  "language": "triton",
  "backend": "cuda",
  "runtime_fingerprint_ref": "project.md#runtime-fingerprint",
  "change_scope": "host",
  "change_family": "manual-graph-replay-fused",
  "sketch_ref": "rounds/sketch_002.json",
  "sketch_sha256": "015da3456f18582ad6114d3f5a0bfd14c5122a365bfbdd8031b1e543ecfe7ebe",
  "implementation_profile_snapshot_ref": "profile_snapshot/triton_cuda.yaml",
  "implementation_profile_snapshot_sha256": "dc8fa4c0c73caecc78c6a886e5ffb18cb97371f3add7cb382dbe30887743b7ae",
  "project_capability_claim_ref": "profile_snapshot/capability_claim.json",
  "project_capability_claim_sha256": "fcba080f084be2791c43bbe45baaaff695cb2b4a72cc4053a3e070ae6912cff5"
}
```

## Optimization Intent

```json
{
  "bottleneck_class": "host_bound",
  "intervention": "Reuse one persistent non-graph-pool output tensor as the fixed copy-out destination on the tier-1 replay path, removing the per-call aten::empty_strided + aten::empty_like allocation while keeping the copy_ so the tensor returned by forward() never aliases across calls.",
  "allowed_changes": [
    "allocate a persistent out_dest [83,128] fp16 tensor once, outside the capture region, and reuse it as the copy_ target on every tier-1 replay call",
    "keep the copy_ from out_ws into out_dest so the returned tensor stays fresh and non-aliased; the persistent buffer is a destination only and is never returned",
    "sweep BLOCK_M over {16,32} and num_stages over {1,2} as an in-round pre-adoption configuration sweep",
    "leave the two Triton kernel bodies, the grid shapes, the counting sort, the routing prelude, and the tier machinery semantically unchanged"
  ],
  "invariants": [
    "public contract unchanged: ModelNew(num_experts, top_k, hidden_size, intermediate_size, renormalize=True) and forward(hidden_states, router_logits)",
    "forward() returns a tensor that does NOT alias across calls; out_dest is a copy target only, never a returned alias",
    "out_ws remains graph-pool memory and is never returned",
    "run_out keeps its 3-arg contract and must not write into a buffer the caller retains as a reference output",
    "torch.topk tie semantics preserved verbatim",
    "output shape [83,128], dtype fp16, all finite, within atol=rtol=1e-2 of base.py under seed 42",
    "base.py untouched (sha256 21e758535af58c21684d6e657ae44f91a0248d246eb424b6923a3c180bc32a3d, 3598 bytes)",
    "static grid, no host data-dependent branch, no .item(), no D2H read inside the captured region",
    "num_warps stays at 1"
  ],
  "expected_wall_improvement_pct": 7.4
}
```

## Unified Sketch

```json
{
  "artifact": "rounds/sketch_002.json",
  "sha256": "015da3456f18582ad6114d3f5a0bfd14c5122a365bfbdd8031b1e543ecfe7ebe"
}
```

## Host Plan

```json
{
  "applicability": "required",
  "affected_scope": [
    "one new persistent module attribute out_dest, [83,128] fp16, allocated lazily on the first tier-1 serve and reused thereafter",
    "the tier-1 copy-out destination: out_dest replaces the per-call torch.empty_like(out_ws)",
    "the returned tensor from forward(): still produced per call and never aliased to out_dest",
    "unchanged: out_ws, x_in, rl_in, sorted_rows, sorted_w, expert_counts, expert_offsets, both graph handles, the anchor set, the bound-set history, the recapture counter, the two monotone tier flags"
  ],
  "state_owner": "Model owns out_dest alongside out_ws; both are created on cuda:0 and live for the module lifetime",
  "lifetime": "out_dest is allocated on the first call that reaches a replay tier after construction and persists; it is reallocated only if the served shape or dtype changes, which cannot happen inside the fixed-shape target regime",
  "allocation_reuse": "out_dest is reused as the copy_ target on every replay call; the per-call torch.empty_like and its underlying aten::empty_strided are eliminated entirely. out_ws is still the graph's write target and is still zero-initialized inside the capture region. The tensor returned by forward() is produced per call and carries no reference to out_dest.",
  "cache_key": [
    "hidden_states.data_ptr()",
    "router_logits.data_ptr()",
    "self.w1.data_ptr()",
    "self.w2.data_ptr()",
    "hidden_states.shape",
    "hidden_states.dtype",
    "router_logits.shape",
    "router_logits.dtype"
  ],
  "invalidation": "any cache_key mismatch, capture failure, or non-target regime falls through to tier-2 or tier-3, which allocate their own destinations and never touch out_dest; correctness never depends on the reuse path being taken",
  "concurrency": "single-device, single-stream, no cross-thread sharing; out_dest is not safe for concurrent multi-stream use, exactly as the rest of the round-001 state is",
  "device_stream_behavior": "unchanged from round 001: capture and replay run on the current torch cuda stream, the copy-out runs outside the replay boundary on the same stream, and no stream is created, switched, or synchronized beyond what the harness already performs",
  "unchanged_behavior": [
    "the two Triton kernel bodies and their grids",
    "the counting sort and the grouped expert GEMM arithmetic",
    "the routing prelude: torch.softmax, torch.topk, the renormalize divide, and the fp16 casts all remain aten and unchanged",
    "output values: bitwise-equal to round 001 on identical input bits",
    "output shape, dtype, and finiteness",
    "the three-tier fallback ladder and both monotone failure flags",
    "run_out semantics and arity"
  ]
}
```

## Evaluation Contract

```json
{
  "hypothesis_id": "H-002",
  "intervention": "Reuse one persistent non-graph-pool output tensor as the fixed copy-out destination on the tier-1 replay path, removing the per-call aten::empty_strided + aten::empty_like allocation while keeping the copy_ so the tensor returned by forward() never aliases across calls.",
  "expected_causal_chain": [
    "out_dest is allocated once outside the capture region",
    "the per-call torch.empty_like(out_ws) and its aten::empty_strided disappear from the timed path",
    "empty_strided plus empty_like falls from 16.219 us/call to about 0",
    "the copy-out itself is unchanged at 22.497 us/call, which is the price of returning a non-aliased tensor",
    "wall time falls by about 16.2 us/call",
    "the retained reference output in compare_case stays valid because the returned tensor never aliases out_dest"
  ],
  "primary_metric": {
    "name": "wall_time",
    "expected_improvement_pct": 5.0
  },
  "causal_graph": {
    "nodes": [
      "out_dest_allocated_once",
      "per_call_alloc_removed",
      "retained_output_unchanged",
      "wall_time_falls_16us",
      "adoption_gate_cleared"
    ],
    "edges": [
      ["out_dest_allocated_once", "per_call_alloc_removed"],
      ["per_call_alloc_removed", "wall_time_falls_16us"],
      ["out_dest_allocated_once", "retained_output_unchanged"],
      ["retained_output_unchanged", "adoption_gate_cleared"],
      ["wall_time_falls_16us", "adoption_gate_cleared"]
    ]
  },
  "mechanism_observables": [
    {
      "name": "host_output_alloc_us_per_call",
      "expectation": "decrease: aten::empty_strided plus aten::empty_like falls from 16.219 us/call to below 2.0 us/call on the timed replay path"
    },
    {
      "name": "host_submission_count_per_call",
      "expectation": "hold at 2.0: one cudaGraphLaunch plus one copy-out memcpy; the change must not add or remove a submission"
    },
    {
      "name": "host_triton_launcher_executions_per_call",
      "expectation": "hold at 0: the python Triton launcher must still never execute during timed calls"
    },
    {
      "name": "retained_output_unchanged",
      "expectation": "pass: a tensor returned by forward and retained across 50 further forward calls is byte-identical before and after"
    },
    {
      "name": "device_us_per_call_nonreplay_tier",
      "expectation": "hold: device time is unchanged because this round touches only the host copy-out boundary; a change here signals an unintended kernel edit"
    },
    {
      "name": "best_block_m",
      "expectation": "directional: the BLOCK_M sweep over {16,32} reports the argmin under a bitwise-identical-output tie-break"
    },
    {
      "name": "best_num_stages",
      "expectation": "exploratory: the num_stages probe over {1,2} reports the argmin under a bitwise-identical-output tie-break; adopt only with a margin outside the 0.5 us tie band"
    },
    {
      "name": "graph_capture_stability",
      "expectation": "hold at round-001 quality: zero recaptures inside the timed segment, zero capture errors, zero diverged outputs"
    },
    {
      "name": "device_us_per_call",
      "expectation": "directional cross-check only: UNAVAILABLE-not-zero on the replay scope, never a falsification trigger"
    }
  ],
  "guardrails": [
    "correctness:pass",
    "retained_output_unchanged: pass across 50 further forward calls",
    "run_out poisoned x2: no stale carry-over, call1 != call2, both match base",
    "output shape [83,128]",
    "output dtype fp16",
    "all finite",
    "base.py sha256 unchanged at 21e758535af58c21684d6e657ae44f91a0248d246eb424b6923a3c180bc32a3d",
    "manifest.json untouched",
    "seed 42 on every command",
    "determinism: 20 consecutive forward calls byte-identical",
    "num_warps remains 1"
  ],
  "profiling_level": "targeted",
  "falsification_rules": [
    {
      "id": "FR-1",
      "observable": "host_output_alloc_us_per_call",
      "threshold": "did not fall below 2.0 us/call",
      "reads": "the per-call allocation was not actually removed"
    },
    {
      "id": "FR-2",
      "observable": "retained_output_unchanged",
      "threshold": "the retained tensor changed at all",
      "reads": "aliasing was introduced; the change is unsafe regardless of its speed"
    },
    {
      "id": "FR-3",
      "observable": "host_submission_count_per_call",
      "threshold": "does not hold at 2.0",
      "reads": "the replay boundary was restructured, not just the allocation"
    },
    {
      "id": "FR-4",
      "observable": "device_us_per_call_nonreplay_tier",
      "threshold": "moves by more than 15 us in either direction",
      "reads": "an unintended kernel or pipeline edit slipped in"
    },
    {
      "id": "FR-5",
      "observable": "mean_wall_ms",
      "threshold": "did not improve by at least 5 percent",
      "reads": "global conservative guard"
    }
  ],
  "observability_note": "The replay scope remains kineto-blind: kernel_count_per_call and kernel_count-derived device_us_per_call are UNAVAILABLE, not zero, and no falsification rule references them. Device observables are collected with KS_E2_REPLAY=0 and the eager control must PRE-BIND the workspaces first, per the round-001 methodology finding, or the aten::fill_ allocation churn will contaminate the reading. The host API census in log/diagnostic_scope_census_002.json is the Level-2 substitute for launch-side attribution."
}
```

## Pitfalls and Anti-pattern Consultation

**Entry 018 — Triton/CUDA-graph manual replay (direct-address tier).** This round refines the boundary of the already-accepted tier-1 path rather than rebuilding it. The per-call sequence stays: guard predicate, one replay, one copy-out. Only the destination of that copy-out changes from a freshly allocated tensor to a reused one. Capture is still amortized at first call and the guard-failure ladder is untouched.

**Entry 011 — launch-side bookkeeping.** The census must report *submissions*, not raw kernel launches. This round is expected to leave the submission count at exactly 2.0, and FR-3 exists specifically to catch a change that quietly restructured the boundary (for example by folding the copy into the graph) while still reporting a speedup.

**Entry 016 — cumsum-based compaction.** Not touched. The counting sort is carried over verbatim from round 001; this round must not disturb it, and FR-4 exists to catch any accidental device-side drift.

**The aliasing trap this round is designed to avoid.** `compare_case` computes `v1_output = run_forward(model_new, ...)` at `auto_bench.py:735`, compares it, then **retains that very tensor** into `export_profile` at line 743 as `(f"candidate_{v1_path.stem}", model_new, v1_inputs, v1_output)`. Between those two points, `time_forward` runs `warmup 50 + repeat 100` forwards. Any design where `forward()` returns a persistent buffer would let those 150 calls overwrite the retained reference output, corrupting the profile reference while still passing `compare_values` and still reporting a large wall improvement. That is a correctness-of-evidence defect, which is why the Orchestrator denied option (i). The retained-output test is therefore not a nicety — it is the guard on the exact failure the harness makes possible.

**Further consultations.**
- *Do not return `out_dest`.* It is a copy target, nothing else. Returning it, or a view of it, reintroduces the aliasing the ruling forbids even though the wall number would look better.
- *Do not fold the copy-out into the graph* to try to reclaim the remaining 22.497 us. That is option (i) in disguise: it would make the graph write directly into a buffer the caller retains.
- *Do not re-sweep `num_warps`.* It is settled at 1 on a 24.4% margin (FR-4 at round 001). Re-testing it costs a round and the prior is now measured, not imported.
- *Do not expect the device lever to move.* Round 001 established that arithmetic reduction does not convert to device time on this rig (isolated 58.231 vs 55.954 us, device-neutral). The `BLOCK_M` and `num_stages` sweeps are free riders and FR-4 is deliberately tight at 15 us so a real regression is caught.
- *Pre-bind workspaces before any eager control.* Round 001's forced-eager control was contaminated by ~49 us of `aten::fill_` churn because disabling the tier guards also bypassed `_alloc_workspace`. FR-4's threshold assumes a clean control; a contaminated one will produce a spurious firing.

## Rationale and Evidence

### Accepted state

`report_001.md`: wall **0.219792 ms**, `+93.248%` against the round-000 canon 3.255288 ms, speedup 14.81x. Round-000's canon is **superseded**; the comparison anchor for round 002 is **0.219792 ms**, so the adoption gate is **10.99 us/call** (5%).

### The bottleneck inverted

Round 001 exhausted the launch lever: **0 python Triton launcher executions per timed call**, **2.0 submissions/call** (1 `cudaGraphLaunch` + 1 copy-out memcpy), 0 recaptures in 100 calls. There is no launch-side overhead left to compress, so round 002 cannot win the way round 001 did.

Decomposing the 219.792 us/call from `log/diagnostic_scope_census_001.json` and the `p02b` CUDA-event isolation:

| component | us/call | addressable |
|---|---:|---|
| `cudaDeviceSynchronize` (CPU) | ~122.1 | **no** — harness `sync_devices()` inside the timed region |
| `_grouped_expert_kernel` | 30.192 | yes (small) |
| `_counting_sort_kernel` | 28.038 | yes (small) |
| routing prelude | 33.704 | yes, but topk frozen inside |
| `aten::empty_strided` + `aten::empty_like` | **16.219** | **yes — this round** |
| `aten::copy_` (CPU) | 13.936 | no (price of non-aliasing) |
| `cudaGraphLaunch` | 7.281 | no |
| `cudaMemcpyAsync` | 6.160 | no (price of non-aliasing) |
| `Memcpy DtoD` (device) | 2.401 | no (price of non-aliasing) |
| out zero-init | 1.871 | no |

Ours is ~139.8 us/call; the harness sync is ~122 us/call and is not addressable.

### Why G1 (the boundary) and not G2 (the routing prelude)

The boundary's eliminable portion (16.219 us) is **7.4% of wall**, which clears the 10.99 us gate on its own. The routing prelude (33.7-41.6 us) is larger in total but its addressable part is much smaller: **topk alone is ~41.6 us and is frozen by the tie-semantics invariant**, so folding softmax+sum+div+cast into Triton can reclaim at best ~20 us, and `reduction.sum` is waiver-gated. G1 is also pure host work requiring no new capability, while G2 touches a waiver and the frozen tie semantics.

### The corrected priors this decision rests on

1. **The graph is re-priced at the measured 423 us**, not the modeled `N_triton x 85 = 170 us`. The old identity is retired. Round 001 showed collapsing the 9.82 aten launches removes far more than the two Triton launcher taxes.
2. **The `BLOCK_M 256 -> 16` arithmetic argument does not carry to device time.** Isolated device is 58.231 us/call (sort 28.038 + expert 30.192) against epoch-1's single kernel at 55.954 us/call — device-neutral, not the predicted 2.4x. No device claim in this decision is derived from an arithmetic-reduction argument.
3. **`num_warps` is settled at 1** with a 24.4% margin (92.855 vs 122.253 us). FR-4 fired at round 001; the sibling nw2 prior does not transfer. Not re-swept.
4. **fp16-dot exactness is POSITIVE here** (p01, 2.441e-04 against a 1e-2 tolerance) unlike the mm_encoder negative — supporting the sweeps, though any new tile still needs in-probe re-qualification.
5. **A forced-eager control must pre-bind the workspaces**, or ~49 us of `aten::fill_` churn contaminates the device reading.

### The Orchestrator's ruling and why option (ii)

The Orchestrator checked the harness directly and found that `time_forward` (459-475) discards the returned value, so aliasing is harmless *for the wall path* — but `compare_case` retains `v1_output` at line 743 and hands it to `export_profile` as the profile reference output, after 150 forwards have already run. **Option (i) is DENIED** (it saves all 38.7 us but corrupts a tensor the harness demonstrably retains); **option (ii) is APPROVED**.

Option (ii) removes the 16.219 us per-call allocation while keeping the copy, so:

```
saving  = 16.219 us/call  = 7.4% of 219.792   (clears the 10.99 us gate alone)
residual = 13.936 + 6.160 + 2.401 = 22.497 us/call  (the price of non-aliasing)
```

The residual is deliberately retained. It is not an inefficiency to be optimized later within this contract — it is what keeps the returned tensor non-aliased and the evidence chain intact.

### Family pool

| # | family | verdict |
|---|---|---|
| **G1** | graph-boundary elimination, option (ii) | **selected** — 7.4%, clears the gate, no new capability |
| G2 | routing prelude to Triton | deferred — ~20 us best case, waiver-gated, topk frozen |
| G3 | `BLOCK_M` {16,32} and `num_stages` {1,2} | **folded in** as a free-rider sweep, zero extra rounds |
| G4 | deeper launch fusion | **excluded** — 0 launcher executions, 2.0 submissions |
| G5 | weight pre-cast | **excluded** — inside the graph a launch is nearly free, so -2 launches buys ~0 |
| G6 | arithmetic-reduction device argument | **excluded** — falsified at round 001 |
| G1-i | return a persistent buffer | **excluded by ruling** — corrupts the retained `v1_output` |

### Budget and convergence

Round 002 of 20; `performance_miss_streak` 0, `failed_attempt_streak` 0, `budget_state` normal. With the ~122 us harness sync non-addressable, the practical wall floor is ~214 us unless the routing prelude also moves, so **G1 and G2 are likely the last two meaningful items**. Per the Orchestrator's instruction: if G1 lands and G2 subsequently looks sub-gate, the honest recommendation is convergence rather than a marginal round. G3 is folded in here specifically so that no round is spent on a G3-sized move later.

### Self-validation

- `schema_v2`: `valid`
- `self-validation`: `green`
- `validator`: `skills/kernel-opt-loop/scripts/validate_decision.py --expected-implementation-profile triton_cuda`
- `sketch_validator`: `skills/kernel-opt-loop/scripts/validate_sketch.py`
- `profile_validator`: `skills/kernel-opt-loop/scripts/validate_profile.py`

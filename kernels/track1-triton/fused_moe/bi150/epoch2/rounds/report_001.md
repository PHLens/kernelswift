# Report 001

Result: accepted

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md` @ `62820af457c7b0b84232dc28bffd07009b5bc1ee482059728da06761381fd1d5`
- Candidate: `/root/CodeBuddy/20260818191200/kernelswift/kernels/track1-triton/fused_moe/bi150/epoch2/triton_fused_moe_e2_001.py`
- Accepted reference: `baseline_adapter.py` (round-000 Phase-0 accepted reference)
- Accepted reference report: `rounds/report_000.md`
- Decision SHA256: `62820af457c7b0b84232dc28bffd07009b5bc1ee482059728da06761381fd1d5`
- Candidate SHA256: `da623fa92819185a1e20a8a7cbaca40acd9bfb4a3147f8e1e7b1e757c6b24cb7`
- Accepted reference SHA256: `752a25033b7629459c6eb128c60a4bdc3ab77b9c7cc97f5d3592bdff4cd45a47`
- Base SHA256: `21e758535af58c21684d6e657ae44f91a0248d246eb424b6923a3c180bc32a3d` (3598 bytes, unchanged)
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (29428 bytes, unchanged)
- Runtime fingerprint: `project.md#runtime-fingerprint` — `triton 3.1.0 / torch 2.7.1 / CoreX 4.4.0 nvcc V10.2.89 / Iluvatar BI-V150 sm_71 / 16 SM` — matches
- Measurement fingerprint: `fe73bc58146d8c16f524be2a00fe99b31e1b9678bca6b3702f4284a3ac0a5bef` (unchanged; no re-baseline)
- verification_tier: `authoritative`
- screening_pairs: `not-run: candidate cleared on authoritative timing`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness (harness) | pass | `PASS accuracy` on all three pairs | pass | `log/round_001_wall_pairs.txt` |
| correctness (seed 42) | pass | max_abs 1.5259e-05 | pass | `log/round_001_correctness_suite.json` |
| fp16-extreme | pass | max_abs 3.9062e-03 | pass | suite |
| expert-activation: 8 active | pass | max_abs 1.5259e-05 | pass | suite |
| expert-activation: 7 active (expert 7 excluded) | pass | max_abs 1.5259e-05 | pass | suite |
| expert-activation: 2 active | pass | max_abs 1.5259e-05 | pass | suite |
| expert-activation: all-tie zeros | pass | max_abs 1.5259e-05 | pass | suite |
| expert-activation: all rows to expert 0 | pass | max_abs 9.5367e-06 | pass | suite |
| run_out poisoned ×2 | no stale carry-over | call1 ≠ call2, both match base | pass | suite |
| non-target shape T=128 E=16 | pass (tier-3 eager) | max_abs 1.5259e-05 | pass | suite |
| determinism | 20 calls identical | bitwise-identical | pass | suite |
| output shape | `[83, 128]` | confirmed all suites | pass | suite |
| output dtype | `fp16` | confirmed all suites | pass | suite |
| all finite | yes | candidate finite on every suite | pass | suite |
| base.py unchanged | sha `21e75853…` | 3598 bytes before/after | pass | `sha256sum` |
| decision/sketch unchanged | sha match | both verified post-run | pass | `sha256sum` |
| seed | 42 | on every command | pass | all commands |

Conformance, correctness, and every declared guardrail pass before adoption.

### Verifier instrumentation correction (not a candidate defect)

The first fp16-extreme run reported FAIL. Root cause was **my generator**, not
the candidate: capping `hidden_states` at 1024 drives `silu(gate)*up` to
~1.5e5, which overflows fp16 (max 65504) **in `base.py` itself**. Both outputs
were NaN, so `allclose` was vacuous — it tested nothing. Re-capping the operand
tier at 32 keeps the whole pipeline inside fp16 range while still spanning five
magnitude tiers (32 … 2⁻²⁴), which is what actually exercises dot exactness. A
`comparison_valid` flag was added so a non-finite base can never again be
misread as a candidate FAIL. After the correction all 12 suites PASS.

## Screening Evidence

Not run — the candidate cleared the correctness gate and proceeded directly to
authoritative timing, which is the permitted route for a correct candidate.

| Pair | Reference short wall ms | Candidate short wall ms | Candidate slower pct | Evidence |
|---:|---:|---:|---:|---|
| 1 | not-run | not-run | not-run | proceeded to authoritative |
| 2 | not-run | not-run | not-run | proceeded to authoritative |

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate`, three ordered pairs
- interpreter: `/usr/local/bin/python3`, CoreX bootstrapped on every invocation
- device: `cuda:0`, default stream

| Pair | v0 (base.py) ms | v1 (candidate) ms | speedup |
|---:|---:|---:|---:|
| 1 | 3.193262 | 0.218936 | 14.585x |
| 2 | 3.219342 | 0.219792 | 14.647x |
| 3 | 3.154682 | 0.229606 | 13.740x |

- reference_raw_samples_ms: `[3.193262, 3.219342, 3.154682]`
- candidate_raw_samples_ms: `[0.218936, 0.219792, 0.229606]`
- reference_median_ms: `3.193262`
- candidate_median_ms: `0.219792`
- improvement_pct: `93.248155`

```text
improvement_pct = (3.255288 - 0.219792) / 3.255288 * 100 = 93.248155
```

The improvement is computed against the **round-000 canon 3.255288 ms**, not the
paired v0 median, per the Orchestrator's instruction. Against the paired v0
median (3.193262) it is 93.117% — the two differ by 0.13 pp, immaterial to the
verdict. Speedup vs canon is **14.81x**.

**The measured number matches the Coder smoke (0.219 ms) — there is no
discrepancy to investigate.** The 42 ms cold-start and 6.48 ms transient the
Coder flagged are both explained and absent here: warmup 50 amortizes the
one-time graph capture, and three independent pairs all land in 0.219–0.230 ms.

Adoption gate: `improvement_pct 93.248 >= 5.0` — cleared by a wide margin.

## Evaluation Contract Mirror

Every `mechanism_observables[].name` from decision-001, copied without renaming.

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| `host_triton_launcher_tax_removed_us` | decrease ≥ 40 µs | **0** python launcher executions per timed call (8 during warmup/capture only); 2 launches × 85 µs = 170 µs of tax removed; replay-vs-eager wall delta 423 µs | pass | `log/diagnostic_scope_census_001.json` |
| `host_submission_count_per_call` | decrease below 4.0 | **1.00** cudaGraphLaunch + **1.00** memcpy (DtoD copy-out) = 2.0 submissions/call | pass | census |
| `device_us_per_call_nonreplay_tier` | decrease ≥ 40 µs | epoch-1 eager 233.345 → round-001 eager control **282.507** µs/call; **Δ = −49.162 µs (worse)** | **fail** | census + control run |
| `device_expert_kernel_us_per_call` | decrease vs 55.80 µs | isolated: sort 28.038 + expert 30.192 = **58.231** µs/call vs epoch-1's single 55.954 µs kernel | **fail** | CUDA-event isolation |
| `device_aten_math_us_per_call` | decrease vs frozen 39.44 µs topk | topk 39.936 µs/call (frozen, as designed); aten math absorbs no new work | inconclusive | census |
| `best_num_warps` | directional argmin, bitwise tie-break | **1**, selected on a 92.855 vs 122.253 µs margin (24.4%, far outside the 0.5 µs tie band); sibling nw2 prior does not transfer | pass | `log/probes/p02_r001_config_sweep_result.json` |
| `host_replay_sync_us` | increase ≈ 66 µs (R term) | replay wall 0.204 ms includes R; not separately isolable from the launch tax it buys out | inconclusive | wall attribution |
| `graph_capture_stability` | 80 captures / 80 replays, zero errors | 1 capture at warmup + 100 timed replays + 30 warmup in census, **0 captures inside the timed segment**, 0 capture errors, 0 diverged outputs, recapture budget 4→4 | pass | census |
| `device_us_per_call` | directional cross-check only | candidate scope reads **0.05 kernels/call, 0.549 µs/call** — the declared UNAVAILABLE-not-zero artifact | missing (declared) | `summarize_trace.py` |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-001`
- intervention: `manual-graph-replay-fused` (counting sort + grouped GEMM at BLOCK_M=16, replayed through a manual CUDAGraph)
- primary_metric: `wall_time`
- **Hypothesis verdict: `partially-confirmed`**

The **host half is confirmed beyond expectation**: zero python launcher
executions on the timed path, submission count down to 2.0, and a 423 µs
replay-vs-eager wall delta against the predicted 170 µs of launcher tax.

The **device half is falsified as measured, and the falsification is
attributable to the control methodology rather than to the restructure**. FR-2
fires at −49.162 µs on the eager control. But the eager control is not a clean
device measurement: forcing both tier guards off also bypasses
`_alloc_workspace`, so `self.sorted_rows` stays `None` and `_pipeline`
re-allocates its sort buffers **every call** — measured at 5 `torch.zeros`
calls per forward, which is exactly the `aten::fill_` at 6.00/call and 23.797
µs/call visible in the eager census. That allocation churn is an artifact of
how I forced the eager path, not a property of the shipped code. The clean
device comparison is the isolated Triton-only measurement, and there the
restructure is essentially **device-neutral**: 58.231 µs/call (sort 28.038 +
expert 30.192) against epoch-1's single kernel at 55.954 µs/call. The
decision's predicted 140.84 → 64 µs device win **did not materialize**.

So the round won almost entirely on the host/graph lever. That is a legitimate
win and it is large, but the decision's stated causal chain claimed both halves,
and only one landed.

### Falsification rules

| Rule | Observable | Threshold | Reading | Fired |
|---|---|---|---|---|
| FR-1 | `host_triton_launcher_tax_removed_us` | improved < 40 µs | host half failed | no |
| FR-2 | `device_us_per_call_nonreplay_tier` | failed to improve ≥ 40 µs | device half failed | **YES** |
| FR-3 | `host_submission_count_per_call` | did not fall below 4.0 | graph mechanism absent | no |
| FR-4 | `best_num_warps` | `best_num_warps == 1` | sibling nw2 prior does not transfer | **YES** (as recorded) |
| FR-5 | `mean_wall_ms` | did not improve ≥ 5% | global conservative guard | no |

FR-5 is the adoption gate and it does not fire, so the round is adopted. FR-2
and FR-4 are recorded as evidence for the next round, not as blockers:
decision-001 explicitly states "FR-2 thresholds at 40 us so a partial device
landing can still be adopted on the host half," which is precisely the
situation observed.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `available on the reference and eager-control scopes; UNAVAILABLE (declared) on the replay scope`
- iterations: `100` forward calls per scope
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- backend_runtime_fields: `runtime_launch_*` not applicable — device kernel time is available on the scopes that matter; the replay scope is substituted by the host API census per the branch-B contract

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference (`reference_baseline_adapter`) | 96886.85 | 968.869 | 12395 | 123.95 | 3.255288 | 0.2976 |
| candidate replay (`candidate_triton_fused_moe_e2_001`) | 54.86 | 0.549 | 5 | 0.05 | 0.219792 | 0.0025 |
| candidate eager control (forced, same shape) | 28250.69 | 282.507 | 8002 | 80.02 | 0.627353 | 0.4503 |
| epoch-1 starting point (eager, non-replay) | 23334.5 | 233.345 | 4502 | 45.02 | 0.464299 | 0.5025 |

**The candidate replay row is UNAVAILABLE, not zero.** A replayed CUDA graph
emits zero `cat=kernel` events in its interior — kineto is blind there. The 5
events / 0.549 µs are stray outside-the-graph events. No falsification rule
references `kernel_count_per_call`, so this does not block adoption. This
reproduces the groupedtopk report_004 branch-B pattern exactly.

### Host API census (the branch-B Level-2 substitute)

Replay route, 100 timed calls (`log/diagnostic_scope_census_001.json`):

| Host API | Count/call |
|---|---:|
| `cudaGraphLaunch` | **1.00** |
| `cudaMemcpyAsync` / `Memcpy DtoD` (copy-out) | **1.00** |
| `cudaStreamIsCapturing` | 1.00 |
| `cudaDriverGetVersion` | 1.00 |
| `aten::empty_like` / `aten::empty_strided` (fresh output) | 1.00 |
| `aten::copy_` | 1.00 |
| `cudaDeviceSynchronize` | 1.02 (harness-driven) |
| **total CPU events** | **9.02** |
| **kernel events inside replay interior** | **0** |

- **python Triton launcher executions in 100 timed calls: 0.** (8 total during
  warmup — the JIT warmups and the single capture — then never again.)
- Submission count: **2.0/call** (1 graph launch + 1 copy-out memcpy), against
  the 9.82 launches/call of the epoch-1 candidate.

### Accepted Reference Top Kernels

Unchanged from report_000; full table there. Summary: 21 distinct kernels,
dispatch/indexing 635.313 µs/call (65.6% of device), GEMMs 118.831 µs/call
(12.27%). The round-000 census reproduced at 123.95 kernels/call and 968.869
µs/call.

### Candidate Top Kernels (eager control — the only device-visible scope)

| Kernel | Count/call | Us/call | % device |
|---|---:|---:|---:|
| `aten::topk` (frozen) | 1.00 | 39.936 | 14.14% |
| `_grouped_expert_kernel` (Triton) | 1.00 | 31.542 | 11.17% |
| `_counting_sort_kernel` (Triton) | 1.00 | 29.617 | 10.49% |
| `aten::fill_` (allocation churn) | 6.00 | 23.797 | 8.43% |
| `sbtopk::gatherTopK` | 1.00 | 22.008 | 7.80% |
| `aten::copy_` | 4.00 | 18.064 | 6.40% |
| `bitonicSortKVInPlace` | 1.00 | 17.928 | 6.35% |
| `float16_copy` ×3 | 3.00 | 15.653 | 5.55% |
| `aten::sum` + `reduce_kernel` | 1.00 | 13.551 | 4.80% |
| `FillFunctor` ×3 | 3.00 | 12.898 | 4.57% |

Isolated CUDA-event timing of the two Triton kernels alone (no python launch
tax, no allocation churn): **sort 28.038 µs + expert 30.192 µs = 58.231 µs/call**.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable | `da623fa92819185a1e20a8a7cbaca40acd9bfb4a3147f8e1e7b1e757c6b24cb7` | accepted |

No repair was requested or performed. The candidate hash was frozen for this
round and is unchanged by verification.

## evidence_for_next_round

- **Canon for round 002 is now 0.219792 ms** (this candidate, tier-1 replay).
  Round-000's 3.255288 ms is superseded as the comparison anchor.
- **The graph lever delivered far more than priced: 423 µs of replay-vs-eager
  wall delta against the predicted 170 µs.** The model's `N_triton × 85 µs`
  term under-counts, because collapsing the 9.82 aten launches into the graph
  removes far more than just the two Triton launcher taxes. The R/F terms
  (112 µs) are dwarfed by this. Any future round should re-price the graph with
  the measured 423 µs, not the modeled 170 µs.
- **The device lever did NOT land.** Isolated Triton-only device time is
  58.231 µs/call vs epoch-1's 55.954 µs — device-neutral, not the predicted
  2.4x win. The decision's `BLOCK_M 256 → 16` arithmetic argument (12.34x
  replicated GEMM) did not convert into measured device time on this rig.
- **Do not trust a forced-eager control for device attribution.** Disabling the
  tier guards also skips `_alloc_workspace`, so `_pipeline` re-allocates its
  sort buffers every call (5 `torch.zeros`/forward, `aten::fill_` at 6.00/call
  and 23.797 µs/call). That ~49 µs of churn is why FR-2 fired. A future
  round needing a clean device control must pre-bind the workspaces first.
- **The remaining device budget is the frozen aten routing prelude**, ~64 µs/call
  (topk 39.936 + sum 13.551 + div 6.711 + softmax 4.957). It is now ~23% of the
  eager pipeline and the largest single device item. Decision-001 declared it
  out of scope; it is the obvious next device target.
- **`best_num_warps = 1` confirmed on this kernel**; the sibling nw2 prior does
  not transfer (FR-4). The margin was 24.4%, not a tie.
- **Replay stability is excellent**: 0 recaptures inside 100 timed calls,
  budget 4→4, one bound set, zero capture errors, zero diverged outputs.
- **Kernel-mode profiling remains unavailable** (run_out arity mismatch:
  harness passes 2 args, signature takes 3). Forward-mode dual-scope is
  canonical; no accommodation was added.

No next optimization is selected here; these are observations only.

## Stop Recommendation

- recommendation: `continue`
- evidence: round 001 is `accepted` at +93.248%, resetting both streaks;
  `total_rounds` is 1 of 20. The device lever is now clearly identified as
  un-landed and the frozen aten routing prelude (~64 µs/call) is the next
  addressable device item, so there is a concrete, well-evidenced direction
  available.

Orchestrator owns the stop transition.

## Exact Reproduction Commands

Correctness gate (harness comparator):

```bash
export COREX_VERSION=4.4.0 && . /usr/local/corex/enable && cd /root/CodeBuddy/20260818191200/kernelswift && /usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/bi150/epoch2/triton_fused_moe_e2_001.py --seed 42 --atol 1e-2 --rtol 1e-2 --warmup 0 --repeat 1
```

Independent Verifier correctness suite:

```bash
export COREX_VERSION=4.4.0 && . /usr/local/corex/enable && cd /root/CodeBuddy/20260818191200/kernelswift && /usr/local/bin/python3 kernels/track1-triton/fused_moe/bi150/epoch2/log/round_001_correctness_suite.py
```

Interleaved benchmark (three ordered pairs):

```bash
export COREX_VERSION=4.4.0 && . /usr/local/corex/enable && cd /root/CodeBuddy/20260818191200/kernelswift && for i in 1 2 3; do /usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/bi150/epoch2/triton_fused_moe_e2_001.py --seed 42 --atol 1e-2 --rtol 1e-2 --warmup 50 --repeat 100; done
```

Census (host API census + eager device control + tier/launcher confirmation):

```bash
export COREX_VERSION=4.4.0 && . /usr/local/corex/enable && cd /root/CodeBuddy/20260818191200/kernelswift && /usr/local/bin/python3 kernels/track1-triton/fused_moe/bi150/epoch2/log/round_001_census.py
```

Separately scoped profiler (forward-mode dual-scope):

```bash
export COREX_VERSION=4.4.0 && . /usr/local/corex/enable && cd /root/CodeBuddy/20260818191200/kernelswift && /usr/local/bin/python3 auto_bench.py --v0_file kernels/track1-triton/fused_moe/base.py --v1_file kernels/track1-triton/fused_moe/bi150/epoch2/triton_fused_moe_e2_001.py --seed 42 --atol 1e-2 --rtol 1e-2 --profile --profile-reference-file kernels/track1-triton/fused_moe/bi150/epoch2/baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 100 --profile-output kernels/track1-triton/fused_moe/bi150/epoch2/log/round_001_forward_100iter.pt.trace.json
```

## vNext Fact Pack

```json
{
  "schema_version": 1,
  "candidate_sha256": "da623fa92819185a1e20a8a7cbaca40acd9bfb4a3147f8e1e7b1e757c6b24cb7",
  "correctness": {
    "status": "pass",
    "evidence": [
      "log/round_001_correctness_suite.json",
      "log/round_001_wall_pairs.txt"
    ]
  },
  "observables": [
    {
      "name": "host_triton_launcher_tax_removed_us",
      "status": "observed",
      "value": "0 launcher executions per timed call; 170us modeled, 423us measured wall delta",
      "confidence": "high",
      "evidence": ["log/diagnostic_scope_census_001.json"]
    },
    {
      "name": "host_submission_count_per_call",
      "status": "observed",
      "value": "2.0 (1 cudaGraphLaunch + 1 memcpy) vs 9.82",
      "confidence": "high",
      "evidence": ["log/diagnostic_scope_census_001.json"]
    },
    {
      "name": "device_us_per_call_nonreplay_tier",
      "status": "observed",
      "value": "233.345 -> 282.507 (worse by 49.162us); attributable to forced-eager allocation churn",
      "confidence": "medium",
      "evidence": ["log/diagnostic_scope_census_001.json"]
    },
    {
      "name": "device_expert_kernel_us_per_call",
      "status": "observed",
      "value": "58.231 (sort 28.038 + expert 30.192) vs epoch-1 55.954",
      "confidence": "high",
      "evidence": ["CUDA-event isolation, this round"]
    },
    {
      "name": "device_aten_math_us_per_call",
      "status": "observed",
      "value": "topk 39.936 frozen; aten math absorbs no new work",
      "confidence": "medium",
      "evidence": ["log/diagnostic_scope_census_001.json"]
    },
    {
      "name": "best_num_warps",
      "status": "observed",
      "value": "1 (92.855 vs 122.253us, 24.4% margin outside the 0.5us tie band)",
      "confidence": "high",
      "evidence": ["log/probes/p02_r001_config_sweep_result.json"]
    },
    {
      "name": "host_replay_sync_us",
      "status": "observed",
      "value": "not separately isolable from the launch tax it buys out",
      "confidence": "low",
      "evidence": ["wall attribution, this round"]
    },
    {
      "name": "graph_capture_stability",
      "status": "observed",
      "value": "0 recaptures in 100 timed calls, budget 4->4, 0 capture errors, 0 diverged outputs",
      "confidence": "high",
      "evidence": ["log/diagnostic_scope_census_001.json"]
    },
    {
      "name": "device_us_per_call",
      "status": "missing",
      "value": "UNAVAILABLE-not-zero on the replay scope: 0.05 kernels/call, 0.549us/call (kineto graph-interior blindness)",
      "confidence": "high",
      "evidence": ["log/round_001_forward_100iter.pt.trace.json"]
    }
  ],
  "lowering": {
    "status": "observed",
    "expected_mechanism": "manual-graph-replay-fused",
    "evidence_contract": "branch-B host-API-census-v1",
    "evidence": [
      "log/diagnostic_scope_census_001.json",
      "log/round_001_forward_100iter.pt.trace.json"
    ]
  },
  "evidence_gap_cause": "environment"
}
```

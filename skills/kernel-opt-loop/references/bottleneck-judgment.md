# Bottleneck Judgment

Use attributable evidence to classify the current bottleneck as device-bound,
host-bound, mixed, or measurement-bound. The classification informs one
falsifiable intervention; it never replaces the round's Evaluation Contract.

## Measurement Rules

1. Benchmark wall time and profiler time are different measurements. Interleaved
   accepted-reference/candidate benchmark wall time is authoritative for
   adoption. Profiler time explains mechanisms.
2. Collect and summarize separate reference and candidate scopes. Never add or
   compare events from a combined scope.
3. Normalize every multi-iteration profiler total per forward call before using
   it. For `N` calls:

   ```text
   device_us_per_call = device_total_us / N
   kernel_count_per_call = kernel_count_total / N
   device_ratio = device_us_per_call / (benchmark_wall_ms_per_call * 1000)
   ```

4. Compare the candidate to `last_accepted_kernel`, not merely the numerically
   previous candidate. Keep shape, dtype, device, harness, warmup/repeat, and
   profiler settings inside one measurement fingerprint.
5. Use unrounded medians for the 5% adoption decision. A profiler improvement
   alone does not adopt a candidate.

## Required Evidence Levels

- **Level 0, every candidate:** conformance, correctness, guardrails, and
  interleaved paired wall samples for the accepted reference and candidate.
- **Level 1, after correctness passes:** separately scoped device time per call,
  kernel count per call, and top-k kernel breakdown for both implementations.
  If the selected target profile explicitly marks device duration unavailable,
  record its normalized backend-specific runtime-launch evidence and preserve
  the unavailable device fields; never substitute launch time for device time.
- **Level 2, intent-driven:** targeted kernel, host, launcher, allocation,
  synchronization, or backend probe requested by the Evaluation Contract.
- **Level 3, deep on demand:** complete trace work only when conflicting results,
  an unattributed regression, decisive noise, or a stop claim requires it.

Detailed host decomposition is Level 2 evidence. Do not run it unconditionally;
request it when the causal hypothesis concerns host, launcher, allocation,
context, stream, or harness-fixed time.

## The Primary Ratio

```text
device_ratio = device_us_per_call / (benchmark_wall_ms_per_call * 1000)
```

`device_us_per_call` is the sum of all kernel durations inside one implementation
scope divided by that scope's forward-call count. It is not one selected kernel,
the complete device span, or an unnormalized 50-call total.

Use these ranges as heuristics, not as adoption rules:

| Device ratio | Class | Likely next evidence or intervention |
|---:|---|---|
| > 80% | device-bound | Dominant kernel dataflow, computation, or fusion |
| 20%-80% | mixed | Choose one separately observable device or host mechanism |
| < 20% | host-bound | Launcher, wrapper, allocation, routing, or context probe |
| < 5% and wall is stuck | measurement-bound candidate | Prove remaining host time is harness-fixed before stopping |

The ratio says where time is observed, not whether that time is compressible.

## Procedure

### 1. Establish Comparable Scopes

Run the unchanged benchmark in interleaved order and retain every raw wall
sample. Profile the accepted reference under a scope such as
`accepted_reference` and the candidate under a separate `candidate` scope. Use
the same positive iteration count for both. Reject a trace whose scopes overlap or
are missing. A target profile may define an explicit runtime-launch-only trace;
in that case require its declared runtime event class and do not call the trace
an attributable device-kernel profile.

For a 50-call candidate scope with 1,000 us of device work and 50 kernel events:

```text
device_us_per_call = 1000 / 50 = 20 us
kernel_count_per_call = 50 / 50 = 1
```

If its unrounded benchmark median is 0.100 ms per call:

```text
device_ratio = 20 / (0.100 * 1000) = 0.20
```

Apply the same normalization independently to the accepted-reference scope.

### 2. Inspect the Level 1 Breakdown

Aggregate kernel names within each scope and report total count, count per call,
total duration, and duration per call. Sort by total duration. A disappearing
library kernel, reduced launch count, or faster target kernel can support the
declared mechanism, but the unrounded wall result still controls adoption.

Do not compare a candidate's one-call number with a reference's 50-call total.
Do not use one implementation's wall median in the other implementation's ratio.

### 3. Request Targeted Level 2 Evidence When Needed

When the Evaluation Contract names host or lifecycle behavior, measure only the
declared components in the same process and regime. A useful decomposition may
include:

1. authoritative harness wall time;
2. kernel call plus the harness's synchronization boundary;
3. the same call with seed setup;
4. the same call with the harness's device synchronization;
5. the residual wrapper or case-construction cost.

Subtract only measurements with matching units and call counts. Treat results as
diagnostic because changing the loop can change the measured regime. Record
allocator, launcher, context, device, and stream assumptions in the Host Plan.

### 4. Select One Intervention

- If one candidate kernel dominates scoped device time, consider its dataflow,
  redundant work, loads, math, or target-supported tuning.
- If separate library kernels dominate, consider fusion only when they are inside
  the decision's allowed change boundary.
- If allocation or launcher work is observed, specify state owner, lifetime,
  cache key, invalidation, concurrency, device, and stream behavior before making
  a host change.
- If only harness-fixed work remains, collect enough targeted evidence for a
  measurement-bound stop recommendation. Do not optimize `base.py` or alter the
  harness to manufacture a speedup.

Before dispatch, confirm that the intervention is expected to improve unrounded
wall time by at least 5%, has a named mechanism observable, preserves all
guardrails, and changes only one attributable cause. A mixed change is acceptable
only when its kernel and host pieces are inseparable and separately observable.

## Worked Example

The historical fused_moe values below illustrate classification only. Each
device value is already normalized per forward call; no raw multi-call total is
used as a per-call value.

| Round | Benchmark wall us/call | Device us/call | Device ratio | Class | Observed target |
|---:|---:|---:|---:|---|---|
| 0 | 6940 | 2700 | 39% | mixed | Remove mask/scatter work |
| 1 | 564 | 21 | 4% | host-bound | Fuse routing kernels |
| 2 | 218 | 23 | 11% | host-bound | Launcher and output allocation |
| 3 | 153 | 23 | 15% | host-bound | Test whether device work can still move wall time |
| 4 | 164 | 21 | 13% | host-bound | Device-context overhead |
| 5 | 138 | 21 | 15% | host-bound | Stop after proving remaining host cost is fixed |

The source project found compressible launcher and context costs before reaching
harness-fixed seed and synchronization costs. Those findings are evidence for
that recorded runtime and Host Plan, not universal instructions to remove device
contexts or cache buffers.

## Compressible Versus Fixed Host Time

Classify a component from current evidence:

| Source | Typical classification | Required check |
|---|---|---|
| Launcher path | Potentially compressible | Target profile and same-regime wall evidence |
| Per-forward allocation | Potentially compressible | Host Plan lifecycle and correctness |
| Wrapper routing operations | Potentially compressible | Scoped kernel count and semantic boundary |
| Device/stream context | Runtime-dependent | Caller ownership and stream preservation |
| Seed setup in user-owned harness | Fixed for the regime | Measurement fingerprint and Level 2 evidence |
| Harness device synchronization | Fixed for the regime | Measurement fingerprint and Level 2 evidence |
| Case construction/load state | Fixed for the regime | Same-process decomposition |

Stop as measurement-bound only when normalized evidence shows remaining device
work is below the stated bound and targeted Level 2 evidence shows the remaining
host time is harness-fixed. Otherwise return the unresolved observation to the
next Designer without inventing a cause.

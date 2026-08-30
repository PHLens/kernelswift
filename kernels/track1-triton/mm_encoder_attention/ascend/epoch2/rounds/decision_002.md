# Decision 002

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"002","reference_implementation":"triton_mm_encoder_attention_e2_001.py","reference_report":"rounds/report_001.md","language":"triton","backend":"ascend","target_profile":"triton_ascend","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-change"}
```

## Optimization Intent

```json
{"bottleneck_class":"host-bound","intervention":"abandon round 002: no kernel-side intervention can clear the five percent wall-time adoption threshold because device time is 13.4064 us/call against a 327.770 us/call wall median, so even driving device time to zero yields 4.09 percent, launch count is already 1.00 and cannot fall further, and the maintainer constraint forbids touching host-side code where the only remaining headroom lives","allowed_changes":[],"invariants":["ModelNew public contract","output shape dtype and device","numerical tolerance atol=1e-2 rtol=1e-2","base.py bytes unchanged","benchmark semantics","triton_mm_encoder_attention_e2_001.py remains the accepted canonical kernel"],"expected_wall_improvement_pct":0.0}
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

- Consulted `references/anti-patterns.md`: entries 011 through 016 are all
  grouped-top-k selection on MLU590-H8 with Triton 3.2.0 and a different
  compiler lowering. None matches this operator, backend, shape, or lowering, so
  no recorded failure invalidates or supports a retry on this target. Critically,
  none of them is a device-bound shape reaching for a wall win, which is the
  situation here.
- Consulted `references/bottleneck-judgment.md`. `device_ratio` is `0.0407`,
  which falls in the `< 5% and wall is stuck` band whose prescribed next step is
  `prove remaining host time is harness-fixed before stopping`. This decision
  deliberately does **not** claim `measurement-bound`: that label requires
  targeted Level 2 host-decomposition evidence that has not been collected, and
  collecting it would not authorize an intervention under the current maintainer
  constraint. The declared class is therefore `host-bound`, which is what the
  measured ratio supports.
- Consulted KernelWiki `patterns/device-win-wall-loss.md`. Round 001 is already
  the successful version of this fight: device time fell 88.7% and wall time fell
  10.3%. Repeating a device-side round now is the exact trap the pattern records
  (the retained flexattention Round 003 improved device time 55.8% while wall
  regressed 8.3%), and here the ceiling is far worse than in that counterexample
  because only `13.4064 us` of device work remains against a `327.770 us` wall.
- Capability legality check against the frozen snapshot: the only lever with more
  than `16.3885 us` of theoretical headroom is the Triton launch path, and
  `lifecycle.fast-launcher` is recorded `Unknown` with no Ascend probe. Declaring
  a fast launcher normative would be a `capability-miss`, not an optimization.
- The frozen profile also records `make_block_ptr`, `async_copy`, and `vectorize`
  as Unknown. None of them can be declared normative, and all three would in any
  case only move device time, which is already below the 5% budget in total.
- Round 001 introduced no host state, so there is no allocation, cache, or stream
  behavior to reclaim. `torch.empty_like` is the only per-call host allocation and
  it is host-side, hence out of scope under the maintainer constraint.

## Rationale and Evidence

All runtime facts below come from `rounds/report_001.md`, the accepted report for
the current canonical kernel `triton_mm_encoder_attention_e2_001.py`. No value
here is a Designer measurement.

### 1. The measured state after round 001

| Quantity | Value |
|---|---:|
| candidate wall median | `0.327770 ms` = `327.770 us/call` |
| candidate device time | `13.4064 us/call` |
| candidate `kernel_count_per_call` | `1.00` |
| candidate `device_ratio` | `0.0407` |
| reference `kernel_count_per_call` | `6.98` |
| reference device time | `118.8920 us/call` |
| round-001 improvement | `+10.2983%` (accepted) |

### 2. The arithmetic bound that closes the kernel-side search

The adoption threshold is `5%` of the candidate wall median:

```text
5% budget = 0.05 * 327.770 = 16.3885 us/call
```

The complete device budget is `13.4064 us/call`. Therefore:

```text
best possible device-only wall improvement = 13.4064 / 327.770 = 4.0902%
deficit against the threshold             = 16.3885 - 13.4064 = 2.9821 us/call
```

This is the decisive fact for round 002. A kernel-side change that removed
**one hundred percent** of device time would still miss the 5% adoption
threshold by `2.98 us/call`. No tiling, fusion, precision, hint, or dataflow
change inside the kernel can escape that bound, because the bound is on the total
device budget, not on any particular inefficiency within it. Any proceeding
kernel decision would therefore be predicting an outcome that is arithmetically
impossible, which is precisely the failure mode `device-win-wall-loss` exists to
prevent.

### 3. The launch-count lever is exhausted

Round 001's host gain came indirectly from collapsing `6.98` launches to `1.00`.
That lever is now at its floor: a correct attention kernel cannot issue fewer
than one launch per call. The report further shows the lever was weak even before
it bottomed out — removing six of seven launches while cutting device time by
`105.4856 us/call` recovered only about `38 us` of wall time, so the dominant
per-call host term does not scale with launch count. There is no second launch to
remove and no first launch to remove either.

### 4. The remaining headroom is host-side and the maintainer forbids touching it

The retained mechanism with real headroom is the per-call non-device residual.
Using the report's own `device_ratio` convention (`device_us_per_call` against
the scoped wall value), the residual is:

```text
candidate residual = 329.365 - 13.4064  = 315.9586 us/call
reference residual = 358.720 - 118.8920 = 239.8280 us/call
residual increase  = 76.1306 us/call
```

These residuals are an arithmetic consequence of two Verifier-reported numbers
under an additive device-plus-non-device assumption. They are **not** a measured
host decomposition and must not be treated as one.

The direction is still informative and is consistent with the report's own
statement that roughly `316 us/call` of host cost remains and did not scale with
launch count: replacing seven `aclnn` launches with one Triton launch left the
candidate carrying *more* per-call non-device time than the reference, not less.
The Triton dispatch path itself, not kernel count, is the plausible location of
the remaining `>16 us` of headroom.

That location is host-side code. The maintainer constraint for this epoch is
explicit: host-side code is not to be touched, and host gain may arrive only
indirectly from launch reduction. Launch reduction is exhausted (section 3), so
the one lever with enough headroom is both off-limits and empty. There is no
third option inside the constraint.

### 5. Why this is not dressed up as a configuration-tuning round

A `final-autotune` round over `num_warps` and `num_stages` is profile-legal, but
it is bounded by the same `4.0902%` ceiling and would therefore be predicted to
fail the adoption test before it ran. It is additionally unavailable here:
`last_completed_binding` is `null`, so the `binding_sha256` anchor that the
final-tuning contract requires cannot be resolved. Neither fact makes a tuning
round attractive; both make it unjustifiable.

### 6. Why `measurement-bound` is not claimed

`bottleneck-judgment.md` reserves `measurement-bound` for the case where targeted
Level 2 evidence shows the remaining host time is harness-fixed. That evidence
does not exist for this epoch, and requesting it would consume a round to produce
a fact we are not authorized to act on. The declared class is `host-bound`, which
the `0.0407` ratio directly supports. If Orchestrator wants the
`measurement-bound` label recorded as the epoch's terminal attribution, that is a
Verifier-owned Level 2 host decomposition, not a Designer decision.

### 7. What would have to change to justify another round

Round 002 is aborted because of a missing authorization, not because of a missing
idea. Any one of the following would reopen the search honestly:

1. **Maintainer authorization to touch host-side code.** This is the necessary
   and sufficient change. With `~316 us/call` of non-device residual, the launch
   path, per-call output allocation, and launcher selection all clear 5% on paper
   individually. A Host Plan round over `launch-path-reduction` or
   `allocation-reuse` would then be the first thing to write.
2. **A matched Ascend probe qualifying `lifecycle.fast-launcher`** (currently
   Unknown), combined with item 1. The frozen profile records no fast launcher
   evidence, so today that path is a `capability-miss` rather than an
   optimization.
3. **A Level 2 host decomposition** showing whether the residual is harness-fixed
   or candidate-owned. This is diagnostic only; it becomes actionable only
   together with item 1. It is the correct next request if Orchestrator wants the
   epoch's stop reason to be `measurement-bound` rather than `host-blocked`.
4. **A change in the measurement regime.** Not permitted: `bottleneck-judgment.md`
   forbids optimizing `base.py` or altering the harness to manufacture a
   speedup, and the measurement fingerprint is fixed for this epoch.

Absent item 1, every remaining mechanism lives on the far side of a constraint
this role cannot lift. Spending a round on a device-side change whose best
possible outcome is `4.09%` would burn budget, risk a regression in the accepted
`+10.2983%` kernel, and produce a `no-improvement` result that was predictable
from `report_001.md` before any code was written.

### 8. Disposition of the campaign

`triton_mm_encoder_attention_e2_001.py` remains the canonical accepted kernel and
satisfies the binding deliverable rule that the submission be a correctness-PASS
Triton implementation. Round 002 records `no-change`. The two known limitations
to carry forward unchanged: the kernel requires `S <= 128` (the campaign shape is
`S=83`, so a row-blocked loop would be needed for longer sequences, and that is a
generality fix, not a wall win), and the second `tl.dot` shape `(128,64,128)`
compiles and is numerically correct but was not among the eleven probed tiles.

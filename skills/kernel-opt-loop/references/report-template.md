# Report <NNN>

Result: baseline | accepted | no-improvement | design-rejected | candidate-failed

## Identity

- Round: `<NNN>`
- Decision: `rounds/decision_NNN.md | not-applicable: Phase 0`
- Candidate: `<candidate path | baseline_adapter.py | not-created>`
- Accepted reference: `<last_accepted_kernel | base.py for Phase 0>`
- Accepted reference report: `<last_accepted_report | not-applicable: Phase 0>`
- Decision SHA256: `<hash | not-applicable: Phase 0>`
- Candidate SHA256: `<hash | not-created>`
- Accepted reference SHA256: `<hash>`
- Base SHA256: `<hash>`
- Harness SHA256: `<hash>`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `<measurement_fingerprint>`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | pass | `<observation>` | `<pass, fail, or not-run>` | `<command/output>` |
| `<guardrail name>` | `<requirement>` | `<observation>` | `<pass, fail, or not-run>` | `<evidence>` |

Conformance, correctness, and every declared guardrail must pass before adoption.

## Interleaved Wall Timing

- warmup: `<count>`
- repeat: `<count>`
- order: `interleaved accepted-reference/candidate`
- reference_raw_samples_ms: `[<unrounded values>]`
- candidate_raw_samples_ms: `[<unrounded values>]`
- reference_median_ms: `<unrounded value>`
- candidate_median_ms: `<unrounded value>`
- improvement_pct: `<unrounded value>`

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
```

The unrounded improvement controls the 5% adoption threshold. Profiler time does
not replace this benchmark result.

## Evaluation Contract Mirror

Copy every `mechanism_observables[].name` from the validated decision without
renaming it. One row is required for every declared observable.

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| `<exact observable name>` | `<declared expectation>` | `<measured observation>` | `<pass, fail, inconclusive, or missing>` | `<artifact or command>` |

- Evaluation Contract applicability: `required | not-applicable: Phase 0`
- hypothesis_id: `<H-NNN | not-applicable: Phase 0>`
- intervention: `<decision intervention | not-applicable: Phase 0>`
- expected_causal_chain: `<declared chain | not-applicable: Phase 0>`
- primary_metric: `wall_time | not-applicable: Phase 0`
- Hypothesis verdict: `confirmed | partially-confirmed | falsified | inconclusive`

A missing required observable yields `measurement-incomplete`; it cannot be
silently converted to `accepted` or `no-improvement`.

## Profiler Evidence

- profiler_level: `summary | targeted | deep-on-demand`
- iterations: `<forward call count>`
- normalized_fields: `device_total_us`, `device_us_per_call`,
  `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`

Reference and candidate scopes are collected and summarized independently. All
totals below are normalized by `iterations` before they are compared.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference | `<total>` | `<per-call>` | `<total>` | `<per-call>` | `<reference median>` | `<ratio>` |
| candidate | `<total>` | `<per-call>` | `<total>` | `<per-call>` | `<candidate median>` | `<ratio>` |

```text
device_ratio = device_us_per_call / (candidate_median_ms * 1000)
```

### Accepted Reference Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `<kernel>` | `<count>` | `<per-call>` | `<total>` | `<per-call>` |

### Candidate Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `<kernel>` | `<count>` | `<per-call>` | `<total>` | `<per-call>` |

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | `<initial verification or local repair>` | `<hash-or-not-applicable>` | `<hash>` | `<outcome>` |

At most one Verifier-to-Coder repair is allowed in the same round.

## Upbound Gap

- upbound_source: `<project.md reference>`
- comparable_metric: `<wall or device metric>`
- absolute_gap: `<value>`
- ratio_to_upbound: `<value>`
- interpretation: `<bounded statement>`

## evidence_for_next_round

- `<observation supported by this report>`
- `<failed or confirmed mechanism>`

Record evidence only; do not select the next optimization.

## Stop Recommendation

- recommendation: `continue | measurement-bound | diminishing-returns | upbound-reached | resource-exhausted`
- evidence: `<specific measurements>`

Orchestrator owns the stop transition.

## Exact Reproduction Commands

```bash
<correctness command>
```

```bash
<interleaved benchmark command>
```

```bash
<separately scoped profiler command>
```

For `Result: baseline`, this report must contain correctness, baseline wall
samples, a Level 1 profiler summary, runtime and measurement fingerprints, and
exact reproduction commands. Its Evaluation Contract mirror is
`not-applicable: Phase 0` because no round decision exists.

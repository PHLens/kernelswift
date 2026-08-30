# Report <NNN>

Result: baseline | accepted | no-improvement | screened-out | design-rejected | candidate-failed | aborted

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
- verification_tier: baseline | screening | authoritative
- screening_pairs: `<two ordered pairs-or-not-run>`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | pass | `<observation>` | `<pass, fail, or not-run>` | `<command/output>` |
| `<guardrail name>` | `<requirement>` | `<observation>` | `<pass, fail, or not-run>` | `<evidence>` |

Conformance, correctness, and every declared guardrail must pass before adoption.

## Screening Evidence

Screening follows correctness and uses exactly two ordered short interleaved
accepted-reference/candidate pairs. A correct candidate is `screened-out` only
when both pairs are at least 10% slower than the accepted reference. Any other
correct candidate proceeds to authoritative timing.

| Pair | Reference short wall ms | Candidate short wall ms | Candidate slower pct | Evidence |
|---:|---:|---:|---:|---|
| 1 | `<unrounded>` | `<unrounded>` | `<unrounded>` | `<command or artifact>` |
| 2 | `<unrounded>` | `<unrounded>` | `<unrounded>` | `<command or artifact>` |

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
not replace this benchmark result. Only authoritative timing can yield `accepted` or `no-improvement`.

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

- profiler_applicability: `required | not-run: screened-out | not-run: not-needed`
- profiler_level: `summary | targeted | deep-on-demand`
- profiler_device_time: `available | unavailable: <target-profile reason>`
- iterations: `<forward call count>`
- normalized_fields: `device_total_us`, `device_us_per_call`,
  `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- backend_runtime_fields: `runtime_launch_total_us`,
  `runtime_launch_us_per_call`, `runtime_launch_count_total`,
  `runtime_launch_count_per_call`, `runtime_launches` when applicable

Reference and candidate scopes are collected and summarized independently. All
totals below are normalized by `iterations` before they are compared.
Profiler evidence is required for baseline and accepted candidates, and is not
run for `screened-out` candidates.

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

## evidence_for_next_round

- `<observation supported by this report>`
- `<failed or confirmed mechanism>`

Record evidence only; do not select the next optimization.

## Stop Recommendation

- recommendation: `continue | target-reached | valid-no-improvement-limit | round-budget-exhausted | user-intervention`
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

## vNext Verifier Fact Pack

A vNext report carries the structured fact pack under `## vNext Fact Pack` with
exactly one fenced JSON object. It is the only input to deterministic verdict
attribution and records no design/code blame:

```json
{
  "schema_version": 1,
  "candidate_sha256": "88c41c1f1d6ee5fb35a55f1f8638f3dd3f4b27c63a4a2d91b54f5b9a6d8c7e31",
  "correctness": {
    "status": "pass",
    "evidence": ["python3 auto_bench.py --check-correctness --v1_file candidate.py"]
  },
  "observables": [
    {
      "name": "external-kernel-count",
      "status": "observed",
      "value": "3 -> 2",
      "confidence": "high",
      "evidence": ["log/profiler_candidate_summary.json"]
    }
  ],
  "lowering": {
    "status": "observed",
    "expected_mechanism": "absent",
    "evidence_contract": "mlu-kernel-summary-v1",
    "evidence": ["log/profiler_candidate_summary.json"]
  },
  "evidence_gap_cause": "none"
}
```

For `decision_kind: final-autotune`, report metadata uses
`artifact_kind: submission-finalization` plus the Decision-matching
`artifact_index` (no campaign `round`), and the same fact pack adds
`final_configuration_tuning` with the canonical `submission_snapshot_id`,
immutable contract hashes, `search_trials`, selected configuration and selector
rule, `selection_outcome: improved|fallback-retained`, `temporary_storage_clean`,
final candidate/binding hashes, and a separate `post_pin_official` block.
Verifier writes the report atomically only after pinning or accepted-fallback
confirmation and final verification; search measurements never authorize a
submission.

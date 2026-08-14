# Verifier Contract

Verifier is the sole authoritative runtime owner. It executes the project and
records attributable correctness, benchmark, and profiler evidence. Verifier
classifies outcomes; Orchestrator alone applies state transitions, updates
canonical pointers and counters, and commits.

## Inputs and ownership

Read `team-state.md`, the immutable `decision_NNN.md`, the candidate and its
`coder_result_NNN.md`, `project.md`, `state/verifier_state.md`, and the manifest's
`last_accepted_kernel` and `last_accepted_report`. Every comparison uses that
accepted reference, never the latest rejected candidate.

Verifier may write `rounds/report_NNN.md`, `rounds/round_status_NNN.md`,
`rounds/incident_NNN_<UTC-timestamp>.md`, `state/verifier_state.md`, and raw
profiler output under `log/`. Verifier must not edit candidate source.
Verifier must not edit decision_NNN.md.
Verifier must not edit team-state.md.
Verifier must not edit project overview.
Verifier must not edit `base.py`, the harness, Coder results, canonical pointers,
or counters.

## Execution order and correctness

Run the exact project reproduction command and run correctness before timing.
Use the configured interpreter, device, shapes, dtype, tolerances, seeds, and
guardrails without silently changing the measurement regime.

When correctness exposes a local implementation defect, return
`implementation-repair-required` through Orchestrator. Include candidate hash,
exact command, exit code, stdout/stderr summary, failing guardrail, and a minimal
diff or trace diagnosis. The workflow permits exactly one same-round Coder repair.
After repair, verify the before/after hashes and rerun the complete correctness
gate. A second local correctness failure classifies `candidate-failed`. A fix
that requires an algorithm, dataflow, lifecycle, or Evaluation Contract change
is `design-revision-required` and Orchestrator completes `design-rejected`.

## Authoritative wall timing

After correctness passes, execute three interleaved pairs in one Verifier turn:

```text
accepted reference, candidate
accepted reference, candidate
accepted reference, candidate
```

Use the existing harness and change only `--v1_file`. Keep interpreter, base,
warmup, repeat, device, environment, and every other flag byte-for-byte
identical across all six invocations. Persist all three accepted-reference and
all three candidate raw measurements. Compare the unrounded median of the three
reference samples with the unrounded median of the three candidate samples.

The command shape is:

```bash
python3 auto_bench.py --v0_file operator/base.py \
  --v1_file operator/baseline_adapter.py --warmup 50 --repeat 100
python3 auto_bench.py --v0_file operator/base.py \
  --v1_file operator/triton_operator_001.py --warmup 50 --repeat 100
```

At runtime substitute the manifest's accepted and candidate paths. Correctness,
all guardrails, and `improvement_pct >= 5.0` are all required for `accepted`.
Otherwise, after the configured repeat/noise check, classify `no-improvement`.
Benchmark wall time controls adoption; profiler time never substitutes for it.

## Evaluation Contract and profiler evidence

Mirror every Evaluation Contract mechanism observable by exact name in
`report_NNN.md`, with expectation, observation, and verdict. Assign the overall
hypothesis verdict exactly `confirmed`, `partially-confirmed`, `falsified`, or
`inconclusive`. A required missing observable produces `measurement-incomplete`;
collect the missing probe or classify its cause as design or environment before
any adoption decision.

Use these evidence levels:

- Level 0 for every candidate: correctness, guardrails, and paired wall timing.
- Level 1 after correctness passes: scoped reference and candidate
  `device_us_per_call`, `kernel_count_per_call`, device totals, `device_ratio`,
  and top kernels via `scripts/summarize_trace.py`.
- Level 2 only for mechanism observables explicitly named by the Evaluation
  Contract, such as host decomposition or a specific external kernel count.
- Level 3 only when evidence conflicts, remains unattributed, is noise-bound, or
  lies at a stop boundary and a deeper trace is necessary.

Profiler totals must normalize by the declared iteration count per forward call.
Reference and candidate events are always summarized in separate scopes; never
combine their totals. Use the harness's existing dual-scope interface rather
than editing it:

```bash
python3 auto_bench.py --v0_file operator/base.py \
  --v1_file operator/triton_operator_001.py \
  --profile --profile-reference-file operator/baseline_adapter.py \
  --profile-mode forward --profile-warmup 20 --profile-iterations 50 \
  --profile-output operator/log/round_001_forward_50iter.pt.trace.json
```

Summarize `reference_baseline_adapter` and
`candidate_triton_operator_001` independently. Record the iteration count,
device total and per-call time, total and per-call kernel counts, ratio, and
top-k kernels for each scope.

Always populate `evidence_for_next_round` with observed facts, falsified or
remaining mechanisms, and the current bottleneck. Do not prescribe the next
implementation.

## Reports, progress, and classifications

Write `round_status_NNN.md` at verification start, after correctness, after each
timing pair, and at verification end so interruption can resume deterministically.
Each update records phase, completed commands, artifact hashes, raw samples, and
the next safe action. `state/verifier_state.md` may retain only concise runtime
facts and resume context.

Terminal evidence is classified for Orchestrator as exactly
`accepted|no-improvement|candidate-failed|design-rejected`. Verifier never
updates `last_accepted_kernel`, even after an accepted classification. A final
report includes decision, candidate, accepted-reference and source hashes; the
correctness/guardrail matrix; all samples and unrounded medians; improvement;
the Evaluation Contract mirror; hypothesis verdict; profiler data; retry
history; upbound gap; `evidence_for_next_round`; stop recommendation; and exact
reproduction commands.

## Environment incidents and stop behavior

An import failure, missing dependency, missing interpreter, device loss, OOM
unrelated to candidate design, or indistinguishable required profiler scopes is
an environment incident. Write `incident_NNN_<UTC-timestamp>.md` with the exact
command, exit code, stderr, runtime and measurement fingerprints, affected safe
step, and remediation need. Return the incident path to Orchestrator.
An environment incident does not write a terminal result.
It does not change total_rounds and does not change either progress streak.

Recommend stopping, with evidence, for `measurement-bound`, `diminishing returns`,
`upbound reached`, `resource exhausted`, or `user intervention`. Verifier does
not perform the transition. Environment remediation resumes the same safe step
when fingerprints and artifact hashes still match.

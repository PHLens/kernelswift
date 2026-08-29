# KernelWiki Knowledge Lift Contract

## Boundary

Knowledge lift is an explicit, offline maintenance workflow. It never runs from the optimization loop and never writes campaign state.

```text
strict current-vNext bundle -> validated proposal only
historical manual manifest -> noncanonical local Source only after Curator review
proposal -> review decision -> manual Source/Card change -> validate/generate -> Git commit
```

## Strict lane

The strict lane accepts one caller-selected terminal bundle, validates it against the checked-in current contract, and may write only an experience proposal. It never falls back to historical parsing when current validation fails.

## Historical lane

The historical lane records available immutable hashes, explicit missing evidence, noncanonical profile authority, and Designer-only scope. It never claims that historical artifacts passed current-vNext validation.

## Review and publication

A proposal has no campaign or corpus authority. A Curator records `include`, `defer`, or `exclude` with the exact proposal hash. Publication is a separate Git-reviewed edit that creates an immutable Source and, when justified, a scoped example on a general Card. Extractors and review validators never publish automatically or promote evidence to Coder visibility.

## Prohibited output

Proposals must not contain `next_candidate`, `recommended_next_change`, or `implementation_instruction`. Knowledge lift does not modify `kernel-opt-loop`, active campaigns, project state, profiles, prompts, Decisions, Sketches, harnesses, Sources, Cards, catalog files, or query views.

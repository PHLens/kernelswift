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

## Review validation

`validate_lift.py` validates closed proposal/review schemas, exact proposal identity and bytes, lane boundaries, artifact hashes, scope, transfer limits, and missing evidence. Reviews record `include`, `defer`, or `exclude` with reviewer identity, UTC time, rationale, and the exact proposal SHA-256.

An `include` review targets either a scoped example on an existing Card or a new general Card with independent teaching value. Operator-specific Card targets and automatic Coder visibility are invalid. A `defer` or `exclude` review has no publication target. Missing reviews are allowed; duplicate reviews and reviews of missing proposals are invalid.

An included proposal still has no campaign or corpus authority. The Curator performs a separate Git-reviewed change:

```text
create an immutable local Source with exact hashes
add a scoped example to an existing general Card by default
preserve target/profile/runtime/shape/dtype/measurement/transfer scope
leave contradictory examples visible
default audiences to designer
run validate.py and generate_indices.py
review Source/Card/generated diffs and commit
```

Validation never creates or edits Sources, Cards, catalog files, or query views, and `include` is never interpreted as automatic publication permission. After an include review, `capture_source.py reviewed-historical` is the sole explicit Source-only materialization path; it verifies the proposal/review ID and SHA, creates no code bundle for metadata-only evidence, and never edits a Card.

## Prohibited output

Proposals must not contain `next_candidate`, `recommended_next_change`, or `implementation_instruction`. Extractors and validators do not modify `kernel-opt-loop`, active campaigns, project state, profiles, prompts, Decisions, Sketches, harnesses, Sources, Cards, catalog files, or query views. The explicit reviewed-historical command may write only the approved immutable Source; subsequent Card and generated-output edits remain manual Git work.

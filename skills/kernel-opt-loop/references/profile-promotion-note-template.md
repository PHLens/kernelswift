# Proposed profile promotion

- Review status: `proposed`
- Implementation profile: `{implementation_profile_id}`
- Probe: `{probe_id}` (definition `{probe_definition_sha256[:12]}…`, result `{result_sha256[:12]}…`)
- Run: `{run_id}`
- Onboarding disposition: `{onboarding_disposition}` (only for eligible demand-selected success)

## Recommendations

- `{capability_id}`: `{current_status}` -> `{recommended_status}`
  - Scope: `{source_scope as sorted JSON}`
  - Rationale: {rationale}

## Unresolved gaps

- {gap}

This note is a rendering of `promotion-candidate.json`, which remains
authoritative. It never edits the canonical profile; promotion requires an
explicit maintainer review commit. The v1 renderer never recommends `supported`.

# KernelWiki v1 Source and Card schema

`data/schemas.yaml` pins Source, Card, catalog, and query-result schema version `1`. YAML is loaded with `yaml.safe_load`; records fail closed on missing or unknown fields. IDs match `[a-z0-9][a-z0-9._-]*`. Lists described as sorted are lexicographically sorted and duplicate-free.

## Closed taxonomy

`data/taxonomy.yaml` is authoritative. The v1 enum sets are:

- Card types: `hardware`, `kernel`, `language`, `measurement`, `migration`, `pattern`, `runtime`, `technique`.
- Source kinds: `github-commit`, `github-pr`, `local-campaign`, `manual-doc`, `official-doc`.
- Audiences: `coder`, `designer`.
- Authority: `advisory`.
- Target match/disposition: `analogy-only`, `backend`, `exact`, `family`, `unknown`.
- Evidence: `experimental`, `inferred`, `local-verifier`, `official-doc-and-upstream-code`, `source-reported`.
- Reproduction: `benchmarked`, `concept`, `pseudocode`, `runnable`, `snippet`.
- Comparability states: `comparable-to-current-baseline`, `historical-local`, `project-reproduced`, `source-reported`.
- License: `approved`, `incompatible`, `metadata-only`, `unknown`.
- Example roles: `capability-gap`, `counterexample`, `positive`.
- Example subtypes: `design-pitfall`, `device-wall-mismatch`, `implementation-pitfall`, `performance`, `profile`, `screening`, `source-example`.
- Profile authority: `current-vnext`, `historical-noncanonical`, `not-applicable`, `source-only`.
- Terminal classification: `aborted`, `accepted`, `no-improvement`, `not-applicable`, `screened-out`, `source-reported`.
- Example comparability: `current-contract`, `historical-local`, `not-comparable`, `source-reported`.
- Metrics: `correctness_pass`, `device_improvement_pct`, `device_time_ms`, `kernel_count_per_call`, `latency_ms`, `throughput_items_per_second`, `wall_improvement_pct`, `wall_time_ms`.
- Statistics: `exact`, `max`, `mean`, `median`, `min`, `p50`, `p95`, `source-reported`.
- Units: `boolean`, `count`, `items-per-second`, `milliseconds`, `percent`, `ratio`.
- Languages: `ascendc`, `cpp`, `python`, `triton`.
- Dtypes: `bf16`, `fp16`, `fp32`, `int32`, `int64`.
- Kernel types: `attention`, `data-preparation`, `moe`, `normalization`, `reduction`, `selection`, `sparse-attention`, `topk`.
- Techniques: `double-buffering`, `kernel-fusion`, `launch-collapse`, `layout-transformation`, `output-reuse`, `software-pipelining`, `tiling`, `work-partitioning`.
- Symptoms: `capability-gap`, `device-win-wall-loss`, `launch-bound`, `materialization-overhead`, `memory-bound`.
- Hardware features: `cube`, `dma`, `execution-pipeline`, `memory-hierarchy`, `vector`.
- Tags are the sorted values checked into `taxonomy.yaml`; unknown tags are rejected.

`data/aliases.yaml` maps known canonical taxonomy values to sorted unique lexical aliases. Aliases never widen taxonomy or evidence authority.

## Source record

A Source is Markdown below `sources/` with YAML frontmatter and a nonempty body.

Required fields:

- `schema_version: 1`
- `id`
- `source_kind`
- `title`
- `url`
- `repository_id`
- `captured_at` as a checked-in string
- `target_disposition`
- sorted `languages`, `kernel_types`, `techniques`, `hardware_features`, and `tags`
- `license_state`

Optional fields:

- `artifact_dir`: a skill-root-relative path; absolute paths and root escapes are rejected
- sorted `implementation_profile_ids`
- sorted `runtime_fingerprints`
- nonempty sorted `audiences`

`repository_id` must be `local` or resolve in `data/source-repositories.yaml` once that registry exists. Absence of profile/runtime/audience metadata means Designer metadata only; it never implies Coder eligibility.

A `local-campaign` Source additionally requires `profile_authority`, boolean `strict_vnext_validated`, sorted `missing_evidence`, and `audiences`. Authority is `current-vnext` or `historical-noncanonical`. Historical evidence must set `strict_vnext_validated: false` and exactly `audiences: [designer]`. These fields are forbidden on other Source kinds.

## Card record

A Card is Markdown below `wiki/` with YAML frontmatter and a nonempty body.

Required fields:

- `schema_version: 1`, `id`, `title`, `type`
- nonempty sorted `audiences`
- `authority: advisory`
- `summary`
- nonempty sorted `targets` and closed `target_match`
- sorted `languages`, `kernel_types`, `techniques`, `hardware_features`, `tags`, and `symptoms`
- sorted ID lists `sources`, `related`, `prerequisites`, and `version_sensitive`
- `observations` and `examples` lists

A `pattern` Card also requires sorted `candidate_techniques`; each ID must resolve to a Card whose type is `technique`. Other Card types may not carry that field.

Technique, pattern, and kernel Cards require these H2 sections: Summary; Problem or symptom; Mechanism; Applicability; Implementation approaches; Expected observables; Risks and counterexamples; Examples; Transfer boundaries; Required local checks; Sources. Kernel Cards additionally require Shape and contract; Implementation structure; Source excerpt or snippet; Measured claims; What transfers; What does not transfer.

Track 2 development and holdout operator names from `evaluation-holdouts.yaml` may not become Card IDs or paths.

## Observation

Every observation has exactly:

`id`, `text`, `source_id`, `locator`, `evidence_level`, `reproduction`, sorted nonempty `targets`, `target_match`, explicit nullable `implementation_profile_id`, explicit nullable `runtime_fingerprint`, sorted `versions`, and sorted nonempty `transfer_boundaries`.

Its `source_id` must exist and must also appear in the containing Card's `sources` list.

## Example

Every example has exactly:

`id`, `role`, `subtype`, `source_id`, `locator`, `evidence_level`, `reproduction`, `target_id`, explicit nullable `implementation_profile_id`, `profile_authority`, explicit nullable `runtime_fingerprint`, `operator_family`, sorted `shape`, `dtype`, `terminal_classification`, `comparability`, explicit nullable `measurement_fingerprint`, `baseline_id`, and `candidate_id`, sorted `observed`, nonempty `transfer_boundary`, and nonempty sorted `reconsider_when`.

Shape values are positive integers or symbolic dimensions matching `[A-Z][A-Z0-9_]*`. Every observed measurement has exactly `metric`, finite numeric-or-boolean `value`, `statistic`, and `unit`; measurements are sorted by metric.

`positive` and `counterexample` require at least one observation. `local-verifier` examples require nonnull profile, runtime, measurement, baseline, and candidate identity. `capability-gap` requires `observed: []`, `capability_id`, `capability_status: unknown|unsupported`, and nonempty `required_probe_or_authority`; those three fields are forbidden for other roles.

An example's Source must exist and be listed in the containing Card's `sources` scope.

## Version claims

`data/version-claims.yaml` has exactly `schema_version: 1` and a sorted `claims` list. Each claim has exactly `id`, sorted `card_ids`, `status: current|stale|unknown`, sorted `supported_versions`, checked-in `last_verified_at: YYYY-MM-DD|null`, and sorted `source_ids`. Card and Source IDs must resolve. Every Card `version_sensitive` ID must resolve to a claim that back-references the Card.

## Validation

Run:

```bash
python3 skills/kernelwiki/scripts/validate.py --root skills/kernelwiki
```

Success emits sorted JSON. Stable validation/input failures exit `2` through `KernelWikiError`; validation performs no network or accelerator access.

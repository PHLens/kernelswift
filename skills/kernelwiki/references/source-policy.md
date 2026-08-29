# KernelWiki Source and Artifact Policy

KernelWiki separates immutable Source metadata from optional retained artifact bundles. Production query and page retrieval are offline and may expose only checked-in bytes that pass this policy.

## Immutable revisions

A captured Source record and its referenced artifact directory are immutable. If evidence, upstream revision, extraction boundaries, license review, or hashes change, create a new Source ID and artifact directory. Never overwrite a published bundle and never provide a force-update path.

## PROVENANCE.yaml

Every retained bundle is rooted under `skills/kernelwiki/artifacts/` and contains one `PROVENANCE.yaml` with schema version 1. The top-level fields are exactly:

- `schema_version: 1`
- `origin_url`
- nullable `upstream_repo` and full 40-character lowercase `upstream_sha`
- `license_state`
- checked-in UTC `retrieved_at`
- `asset_mode`
- sorted `allowed_audiences`
- `coder_access`
- sorted `source_ids`
- a nonempty, local-path-sorted `files` list

Each file declares exactly `local_path`, nullable `upstream_path`, nullable `heading_path`, `role`, `mode`, and the SHA-256 of the retained bytes. Local paths are POSIX-relative, root-confined, non-symlink paths. The manifest must declare every regular file in the bundle and no file outside it.

## Evidence modes

- A top-level `verbatim` bundle may contain only `verbatim` and `upstream-patch` files. Both require an upstream repository, pinned full Git SHA, and upstream path.
- A top-level `extracted` bundle may contain only `extracted` files. Every file requires the upstream identity/path plus a heading locator.
- A top-level `derived` bundle may contain only `derived` files. Derived evidence requires one or more Source IDs and may not claim an upstream path.

The validator enforces this compatibility matrix; derived or extracted material is never relabeled as verbatim upstream text.

## License and audience gates

License states are `approved`, `metadata-only`, `unknown`, and `incompatible`. Any state other than `approved` denies code asset exposure. Code-bearing roles include PR diffs, upstream files, snippets, and historical candidates. Such a Source may keep metadata, but its bundle cannot grant Coder access or retain files with any of those roles.

`coder_access` is one of `denied`, `snippet-only`, or `exact-profile`. Non-denied access requires the `coder` audience and an approved license. `snippet-only` may expose only files explicitly classified as snippets. Full upstream files and historical candidates remain Designer-only unless a later role-aware admission contract explicitly grants exact-profile access.

## Deterministic size limits

`data/size-budget.yaml` fixes repository, bundle, and per-file byte limits. Validation counts checked-in file byte lengths under `artifacts/`, including every `PROVENANCE.yaml`, while ignoring only `.gitkeep`. It does not use filesystem timestamps, compression estimates, or network metadata.

Whole-corpus validation fails closed if a Source references a missing bundle, its Source ID is absent from the bundle's `source_ids`, retained bytes disagree with declared hashes, undeclared files exist, or any size limit is exceeded.

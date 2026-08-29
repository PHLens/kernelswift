# KernelWiki candidate inclusion policy

Repository discovery is a read-only aid for Curator review. It never creates a Source, edits a reviewed candidate ledger, or publishes an artifact. Only an explicit `source_capture.py discover` invocation may construct a GitHub client. Manual lanes are reviewed from stable metadata and fail automated discovery with `adapter-manual`; v1 does not scrape HTML or simulate an AtomGit adapter.

## Decisions

- **include** — the candidate contains transferable device-kernel knowledge, all relevant changed files were reviewed, the target relationship is explicit, and immutable provenance plus an acceptable license state can be captured.
- **defer** — the candidate may be useful, but target applicability, changed-file accounting, license state, version identity, retained assets, or performance/correctness evidence is incomplete.
- **exclude** — the candidate is outside KernelWiki scope or contains no device-kernel knowledge that can be supported by source evidence.

Discovery never assigns `include`. New discoveries enter a proposed merge as `defer` with reason `unreviewed-discovery`. Existing Curator decisions and reasons are preserved. Candidates no longer returned upstream remain in the ledger with `discovery_state: not-returned` so review history is not erased.

## Typical exclusions and deferrals

- **wrapper-only** — host dispatch or registration without a device implementation: exclude unless it establishes a reusable runtime integration mechanism; otherwise defer for supporting device evidence.
- **config-only** — target flags, build configuration, or version pins without kernel behavior: exclude.
- **benchmark-only** — benchmark code or numbers without the implementation and measurement contract: defer; exclude if no kernel mechanism can be located.
- **test-only** — correctness tests without an implementation mechanism: exclude, or defer only when they pin a missing behavioral contract.
- **host-framework-only** — scheduler, routing, launcher, or Python orchestration changes with no device-kernel consequence: exclude.
- **missing provenance** — useful claims whose immutable URL, full commit/PR identity, changed-file ledger, or retained-file hashes cannot be established: defer.
- **missing license authority** — metadata may remain reviewable, but code/assets cannot be retained until the license state is approved: defer or capture as metadata-only in the later capture phase.

## Path classification

Automated GitHub lanes search only configured terms. A PR is returned only after its complete changed-file list is fetched and at least one non-skipped path matches a configured kernel path glob. Documentation, tests, and benchmarks are skipped by default even if their names contain `kernel`; a Curator can still add a manual candidate when those files support a separately identified implementation.

Candidate order is ascending PR number. Search-term duplicates are collapsed by PR number, paths are sorted, and output is canonical JSON. Reviewed YAML ledgers are changed manually only after inspecting that output.

# KernelWiki v1 Design

**Created**: 2026-08-17

**Status**: Approved for specification and planning

**Scope**: Define a standalone, Git-versioned KernelWiki skill that curates,
retrieves, and promotes qualified kernel-optimization knowledge for
`kernel-opt-loop`. This specification adds no kernel implementation, scheduler,
remote service, vector database, or automatic continuation behavior.

## 1. Motivation

`kernel-opt-loop` already preserves per-project decisions, candidates,
measurements, runtime fingerprints, and rejection evidence. Its current reusable
knowledge is intentionally narrow: target profiles record capability facts and
`references/anti-patterns.md` records a small static set of failed mechanisms.
This prevents unsafe transfer, but it leaves valuable local evidence isolated in
project reports and historical logs.

KernelWiki provides a bounded knowledge plane. It must improve the quality of
future Designer hypotheses and Coder constraints without weakening the existing
rules that make a local optimization result credible:

- the current project's base, harness, target profile, and measurement regime
  remain authoritative;
- only Verifier evidence can establish current-project correctness or adoption;
- a successful implementation, tuning value, or compiler lowering on one target
  does not prove it on another target; and
- the optimizer must work normally when KernelWiki has no matching knowledge or
  is unavailable.

The design uses the Karpathy LLM Wiki model: source material is compiled into
small, reviewed, cross-referenced wiki knowledge and generated navigation rather
than retrieved as arbitrary raw chunks at candidate-generation time. The model
is extended with a first-class atomic claim layer because kernel optimization
requires approval, runtime scoping, contradiction handling, and deterministic
snapshotting at finer granularity than a prose wiki page.

## 2. Goals

KernelWiki v1 must:

1. exist as a standalone skill at `skills/kernelwiki/`, including its canonical
   corpus, schemas, tooling, and tests;
2. compile immutable sources into approved atomic claims, human- and LLM-readable
   wiki pages, and generated query indices;
3. express a portable optimization mechanism separately from its target-specific
   MLU, GCU, CUDA, or other backend projections;
4. distinguish target-local facts, cross-target safety rules, local evidence,
   and unverified external analogies;
5. give `kernel-opt-loop` a deterministic, immutable, bounded `KnowledgePacket`
   before a decision is written;
6. require explicit user approval before a proposed claim becomes active;
7. retain provenance, source hashes, runtime scope, version-sensitive claims, and
   conditions for reconsidering positive and negative evidence;
8. support manual curation of a small set of existing MLU/GCU evidence before
   expanding to external repositories; and
9. demonstrate through fixtures and holdout replay that retrieval is safe and
   does not reduce local-loop correctness or reproducibility.

## 3. Non-goals

The following are explicitly outside v1:

- copying an upstream or historical kernel into a current candidate;
- code RAG or automatic injection of upstream snippets, patches, configurations,
  launch parameters, or exact tile sizes into Coder context;
- treating external CUDA, Blackwell, or other accelerator results as proof of
  Triton-MLU or Triton-GCU capability or performance;
- modifying a target profile through Wiki retrieval;
- automatic promotion, automatic trust elevation, automated GitHub crawling, or
  remote synchronization;
- vector-database retrieval, embedding-based ranking, or a persistent query
  service;
- rewriting historical logs or migrating a running optimization campaign;
- relaxing target profile, base, harness, canonical-pointer, correctness,
  timing, or measurement-fingerprint gates; and
- a new role that writes candidates, runs measurements, or transitions the
  kernel-opt-loop state machine.

## 4. System Boundaries and Ownership

KernelWiki is a knowledge-plane skill. `kernel-opt-loop` remains the control
plane.

| Concern | Owner |
|---|---|
| Corpus, source registry, claim lifecycle, wiki pages, queries, schemas, validators | KernelWiki |
| Immutable base/harness, target selection, campaign state, canonical pointers, Git run ledger | kernel-opt-loop Orchestrator |
| Candidate implementation and Coder result | kernel-opt-loop Coder |
| Correctness, timing, trace evidence, and per-project KernelEvidenceRecord | kernel-opt-loop Verifier |
| Writing a current round's snapshot and calling KernelWiki tools | kernel-opt-loop Orchestrator |
| Approving promotion into active claims | User |

The dependency is one-way and optional:

```text
kernel-opt-loop Orchestrator -> KernelWiki query and validation tools
```

KernelWiki must not mutate a project's `team-state.md`, candidate, decision,
report, canonical pointer, counter, or phase. It may read only explicitly passed,
read-only evidence paths; it must not discover project artifacts by scanning or
treat a report as mutable workflow state. A missing skill, failed query, empty
result, or incompatible corpus revision returns a valid empty KnowledgePacket; it
never blocks a local optimization round.

## 5. Corpus Architecture

KernelWiki is self-contained at `skills/kernelwiki/`:

```text
skills/kernelwiki/
  SKILL.md
  README.md
  index.md
  schemas/
    source.schema.json
    claim.schema.json
    wiki-page.schema.json
    knowledge-query.schema.json
    knowledge-packet.schema.json
    kernel-evidence-record.schema.json
  data/
    taxonomy.yaml
    aliases.yaml
    version-claims.yaml
    source-policy.yaml
  sources/
    local/mlu/
    local/gcu/
    external/prs/
    external/docs/
    external/blogs/
    external/contests/
  claims/
    proposed/
    active/
    archived/
  wiki/
    techniques/
    patterns/
    operators/
    runtime/
    semantics/
    measurement/
    target-guides/
  queries/
    by-target-profile.md
    by-operator.md
    by-technique.md
    by-bottleneck.md
    by-symptom.md
    by-semantic-constraint.md
    by-source-repo.md
  candidates/
    sources/
  artifacts/
  references/
    taxonomy-guide.md
    promotion-rubric.md
    source-policy.md
    integration-contract.md
  scripts/
    validate.py
    generate_indices.py
    query.py
    get_page.py
    build_packet.py
    validate_evidence.py
    propose_claim.py
    promote_claim.py
    repo_status.py
  tests/
```

The directory hierarchy represents a compilation pipeline:

```text
sources/ -> claims/ -> wiki/ -> queries/
```

### 5.1 Sources

A source is immutable, provenance-first input. A local source records an
already-committed project artifact; it is not a mutable working-tree log. An
external source records a pinned upstream artifact, official document, contest
result, or carefully scoped community report.

Each source has a unique `source-*` ID and includes, at minimum:

```yaml
id: source-local-mlu-groupedtopk-round-003
kind: local-report | local-evidence | external-pr | external-doc | external-blog | external-contest
origin:
  project_or_repo: ...
  url: ...
  commit: ...
  paths: [...]
  sha256: ...
  retrieved_at: ...
license: local-project | SPDX-id | unknown-pending-review
runtime_scope: {...}
measurement_scope: {...}
tags: [...]
```

Local sources must reference the project commit, source artifact paths, and
SHA-256 digests for the relevant decision, report, candidate, and sidecar
evidence record. External sources must include a stable URL, commit or release
identifier when available, content hash, retrieval date, and license state.

Sources are not directly inserted into Coder prompts. They are for curation,
review, reproducibility, and citation.

### 5.2 Claims

A claim is the smallest active knowledge unit. It is the only unit that can be
approved, selected into a KnowledgePacket, superseded, marked stale, or cited by
a frozen project snapshot. A claim may be positive, negative, safety-oriented,
or measurement-oriented.

Every active claim has:

```yaml
id: claim-<stable-slug>
kind: mechanism | counterexample | semantic-safety | measurement | capability-update-proposal
status: proposed | active | stale | superseded | revoked | archived
transfer_class: target-local | cross-backend-replicated
core:
  abstraction_level: semantic | graph | mechanism
  statement: ...
  preconditions: [...]
  exclusions: [...]
  expected_observables: [...]
  reconsider_when: [...]
regime_tags: [...]
sources: [source-*]
version_claims: [version-*]
projections:
  triton_mlu: {...}
  triton_gcu: {...}
related_claims: [claim-*]
supersedes: [claim-*]
```

A claim body is concise and structured. It contains no historical or upstream
candidate body, exact launch configuration, precise tuning parameter, copied
patch, or portable acceptance decision.

#### 5.2.1 Claim kinds

- `mechanism`: A conditional causal mechanism, such as reducing materialization
  and dispatch through a legal fusion boundary.
- `counterexample`: A qualified failed path, including observed failure,
  preconditions, evidence, and explicit reconsideration conditions.
- `semantic-safety`: A safety requirement independent of performance, such as
  output lifetime, alias, stream, tie, mask, or public-contract preservation.
- `measurement`: A restriction on interpretation or comparison, such as runtime
  launch duration not being device kernel duration.
- `capability-update-proposal`: A proposal to amend a target profile after a
  matched probe. It is not a capability fact until the target-profile process
  separately approves it.

### 5.3 Core Claims and Target Projections

A core claim is a backend-neutral statement at the `semantic`, `graph`, or
`mechanism` abstraction level. It must not name a target DSL primitive, hardware
instruction, schedule, or code shape as if those were portable.

Each target projection is separately scoped:

```yaml
projections:
  triton_mlu:
    status: external-candidate | local-qualified | local-replicated | stale | superseded | revoked | unavailable
    runtime_scope: {...}
    target_recipe: {...}
    required_local_checks: [...]
    evidence: [source-*]
  triton_gcu:
    status: ...
```

A projection is a matrix entry, not a fallback chain. Absence of a projection is
`unavailable`; it does not permit an implementation to borrow an MLU, GCU, CUDA,
or other projection.

A claim may be labeled `cross-backend-replicated` only when two independent local
target profiles each have evidence for the same core mechanism and the same
causal observable. External sources may support source confidence but cannot
replace either local projection.

### 5.4 Wiki Pages

Wiki pages compile and explain multiple claims. They are optimized for LLM and
human navigation, not used as the authoritative unit for promotion or packet
selection. Pages use YAML frontmatter with IDs, tags, source IDs, claim IDs,
version claims, and target relevance.

v1 page types are:

- `techniques`: Conditional optimization mechanisms.
- `patterns`: Symptom-to-candidate-technique diagnosis.
- `operators`: Case studies and operator-family context.
- `runtime`: Loader, launcher, and integration behavior.
- `semantics`: Public-contract, alias, tie, mask, stream, and lifetime rules.
- `measurement`: Timing, profiling, attribution, and comparability rules.
- `target-guides`: Navigation summaries that point to, but do not supersede,
  target profiles.

A page can cite mutually contradictory claims when their regime tags and target
projections differ. For example, a `tl.dot` page may cite a qualified MLU
success in a suitable shape and a qualified MLU counterexample for small M.

### 5.5 Generated Queries

`queries/` contains generated Markdown navigation pages. They are never edited
by hand. The generator derives them from active claim and wiki frontmatter.

v1 also provides JSON-oriented tooling:

- `query.py`: Exploratory query for Curators and human investigation. It may
  search natural-language keywords, tags, aliases, source repositories, page
  types, target profiles, and symptoms.
- `build_packet.py`: Deterministic production query used only by the
  kernel-opt-loop adapter. It accepts structured JSON and emits a bounded JSON
  KnowledgePacket.

No v1 production path uses a vector database or embedding ranking.

## 6. Controlled Vocabulary and Applicability

### 6.1 Exact Evidence and Regime Tags

Exact values belong to source evidence: complete runtime fingerprint, base and
harness hashes, measurement fingerprint, shape, dtype, layout, device, and
commands. Claims use controlled regime tags for bounded conditional
abstraction, such as:

- `token-count-small`
- `matmul-m-small`
- `reduction-width-256`
- `topk-k-small`
- `launch-bound`
- `device-bound`
- `mixed-bound`
- `contiguous-row-major`
- `tie-order-required`
- `retained-output-observable`

The taxonomy is finite and validator-enforced. Claims may not use arbitrary
Python predicates or only prose conditions as their matching contract. New tags
require a reviewed taxonomy update in the same commit as the first claim using
them.

### 6.2 Aliases

`data/aliases.yaml` maps user and project spellings to canonical terms. Aliases
improve exploratory navigation only. They do not widen hard target, runtime,
semantic, or regime matching.

### 6.3 Version-sensitive Claims

`data/version-claims.yaml` centrally records version-sensitive statements. Each
entry defines target profile, applicable runtime/version range, last matched
verification date, source IDs, and the claims/pages that cite it. The validator
must enforce bidirectional consistency between the registry and claim/page
references.

A target profile, compiler distribution, backend version, device architecture,
harness contract, or measurement regime change never silently preserves a
performance projection. The affected projection becomes `stale` until a
matching source or probe requalifies it.

## 7. Evidence Quality and Lifecycle

### 7.1 Evidence dimensions

Every projection records two independent dimensions:

- `source_confidence`: `local-verifier`, `official-doc-and-upstream-code`,
  `source-reported`, `inferred`, or `experimental`.
- `transfer_confidence`: `local-qualified`, `local-replicated`,
  `analogy-only`, `unavailable`, or `stale`.

High source confidence does not imply high transfer confidence. CUDA code,
Blackwell documentation, and an upstream merged PR can be authoritative sources
while remaining only `analogy-only` for MLU/GCU.

### 7.2 Promotion states

Claim lifecycle is:

```text
source evidence -> proposed -> user-approved active
                                   |
                                   +-> stale -> requalified active
                                   +-> superseded
                                   +-> revoked
```

Projection lifecycle is independent:

```text
external-candidate -> local-qualified -> local-replicated
                         |                    |
                         +--------------------+-> stale | superseded | revoked
```

`local-qualified` requires complete current-target evidence for a tightly scoped
claim. `local-replicated` requires two independent local evidence sets, or one
project evidence set plus a separately tracked matched microprobe, with the same
core mechanism and causal observable.

User approval is required for every transition that creates an active claim,
changes a claim statement or scope, adds a target recipe, marks a claim
cross-backend-replicated, or supersedes/revokes an active claim. A source record
or a project proposal can be created before approval but cannot be retrieved by
production packet building.

### 7.3 Conflict handling

Conflicting results are modeled, never averaged away:

- contradictory claims remain separately source-backed;
- each has exact evidence and controlled regime tags;
- packet matching selects the most applicable target projection;
- exact matched counterexamples are included as constrained warnings; and
- Designer may revisit a counterexample only by citing a matching
  `reconsider_when` condition backed by new local evidence.

A counterexample does not become a universal ban because one dimension may have
changed: target profile, runtime, architecture, shape regime, semantic
requirement, lowering, or evidence quality.

## 8. Restricted Recipes and Packet Authority

KernelWiki is not code RAG. A target recipe is a non-executable generation
contract. It may describe change boundary, dataflow requirements, semantic
guardrails, forbidden shortcuts, mandatory local checks, and expected
observables. It must not contain code, copied pseudocode structured to be
mechanically transcribed, exact tile sizes, exact launch settings, absolute
latency, or a portable acceptance conclusion.

Example:

```yaml
target_recipe:
  change_boundary: [kernel-dataflow]
  dataflow_requirements:
    - preserve value/index tie semantics explicitly
    - apply mask before selected reduction
  forbidden_shortcuts:
    - do not infer another backend's launcher support
  required_local_checks:
    - actual-harness-loader
    - compile-smoke
    - tie-case-correctness
  expected_observables:
    - runtime_launch_count_per_call: decrease
```

### 8.1 KnowledgeQuery

The kernel-opt-loop adapter writes a structured query before design. Required
fields are:

```yaml
schema_version: 1
project: <relative project identity>
round: "NNN"
target_profile: triton_mlu | triton_gcu
runtime_fingerprint_ref: <project anchor and hash>
measurement_fingerprint: <sha256>
operator_tags: [...]
regime_tags: [...]
dtype_layout_tags: [...]
bottleneck_tags: [...]
semantic_constraints: [...]
change_scope: kernel | host | mixed
change_family: <slug>
```

The query is a projection from authoritative project evidence. It is not an LLM
summary and may not claim a capability inferred from candidate source.

### 8.2 KnowledgePacket

`build_packet.py` produces a stable JSON packet with corpus revision, query hash,
selected claims, excluded claims, match reasons, and empty-result reason. It has
four authority sections:

1. `hard_constraints`: Explicit identifier references to current target-profile
   or project-invariant constraints, plus approved applicable semantic-safety
   requirements. They cannot introduce or revise L0 capability facts, and Coder
   must obey them.
2. `target_recipes`: Only current-target `local-qualified` or
   `local-replicated` projections. Coder may use them only while preserving the
   immutable decision; it records use or rejection in its result.
3. `designer_hypotheses`: Core mechanisms, cross-backend claims, and external
   analogies. They are available only to Designer and cannot add normative
   implementation requirements by themselves.
4. `counterexamples`: Applicable local counterexamples. Exact matches require a
   Designer consultation entry and a reconsideration justification before reuse.

The packet is bounded: at most three target recipes, at most three Designer
hypotheses, and all exact matched counterexamples. A deterministic tie-breaker
uses transfer confidence, exactness, regime overlap, source confidence,
recency, and stable claim ID.

## 9. kernel-opt-loop Integration

KernelWiki integration is optional and adapter-based.

### 9.1 Designing

Before dispatching Designer, Orchestrator:

1. builds `<project>/state/kernelwiki_query_NNN.json` from canonical project
   evidence and the selected target profile;
2. invokes `build_packet.py` with the known KernelWiki revision;
3. writes `<project>/state/kernelwiki_snapshot_NNN.json`; and
4. records the snapshot path and hash in the immutable decision.

Designer reads the snapshot and adds a `KernelWiki Consultation` section to the
decision. It identifies selected claim IDs, match classes, adopted constraints,
rejected alternatives, and any counterexample reconsideration rationale.

A missing or failed KernelWiki invocation creates an empty snapshot with a
reason code. The round remains valid and follows the current local workflow.

### 9.2 Coding

Coder receives the immutable decision and its frozen snapshot. It may use only
`hard_constraints` and current-target `target_recipes`. It may not open raw
sources, artifacts, external code, a non-current target projection, or a newer
Wiki revision to change the decision.

`coder_result_NNN.md` records consulted claim IDs, recipe constraints followed,
and any target-profile reason a recipe could not be used. A required change to
the decision remains a `major-deviation` or `capability-miss` under the existing
contract.

### 9.3 Verifying and evidence publication

Verifier writes `rounds/kernel_evidence_NNN.json` beside its report. The sidecar
has the report, decision, candidate, accepted-reference, base, harness, runtime,
and measurement hashes; the target profile; exact source facts; declared
mechanism observables; observations; and a bounded candidate claim proposal.

The sidecar is a project artifact, not an active KernelWiki claim. It contains no
next implementation prescription. KernelWiki's `validate_evidence.py` may
validate that artifact after the terminal project commit. `propose_claim.py` can
produce a proposal, but only user-approved `promote_claim.py` creates or changes
an active claim in a separate commit.

## 10. External Source Curation and Artifacts

External sources initially enter a per-repository candidate ledger with
`include`, `defer`, or `exclude` decisions and recorded rationale. No bulk
crawler runs in v1.

For an included source, the source record stores URL, pinned commit/release,
license state, content digest, path, source category, tags, and a concise
non-executable summary. The default is metadata-only.

`artifacts/` may contain a compact evidence bundle only after review. Every
bundle has `PROVENANCE.yaml` containing origin URL, upstream repository, commit,
license, retrieval date, asset mode, file hashes, paths, and size-cap result.
Asset modes are `verbatim`, `extracted`, and `derived`. Derived material must
list its source IDs and must never be represented as upstream source.

Artifacts are Curator-only by default. Production packet building excludes them.
They can be inspected in a user-initiated deep dive, subject to license and
source-policy checks.

## 11. Initial Corpus and Migration

v1 begins with a deliberately small manually reviewed corpus:

- MLU local sources for grouped top-k, fused MoE, flexattention, and sparse
  pooler;
- GCU local sources for grouped top-k direct launch and its profiler limitation;
- the existing target profiles and anti-pattern catalog as reference material;
  and
- a small external candidate ledger without active external-to-Coder claims.

Likely initial claims include launch-collapse as a cross-backend mechanism,
actual-harness-loader validation as runtime safety, output lifetime safety,
measurement limits for runtime launch traces, MLU top-k counterexamples, and
MLU `tl.dot` shape-conditioned evidence. A claim is only added after validating
that its source artifacts and conditions satisfy the schema and promotion rubric.

Existing projects are not migrated automatically. Active projects, including
`s60/groupedtopk`, retain their current contracts. The adapter applies only to
new campaigns or to a project the user explicitly elects to migrate at a safe
terminal boundary.

## 12. Validation and Acceptance Requirements

The KernelWiki implementation is accepted only when tests demonstrate all of the
following without accelerator hardware:

### 12.1 Corpus integrity

1. Every source, claim, and wiki page validates against its schema.
2. Every source ID, claim ID, related link, projection reference, and version
   claim resolves.
3. Active claims reference approved sources and required promotion metadata.
4. Generated query indices are deterministic and contain no hand-maintained
   divergence.
5. Unknown taxonomy tags and broken aliases fail validation.
6. Artifact provenance validates its URL, mode, hash, license state, and size
   policy.

### 12.2 Target and transfer isolation

1. An MLU projection never becomes a GCU target recipe.
2. A GCU projection never becomes an MLU target recipe.
3. An external CUDA/Blackwell source remains a Designer-only analogy without a
   matched local projection.
4. A core claim without a current target projection cannot become a Coder recipe.
5. A cross-backend-replicated label fails validation without two independent
   local target projections that share a core causal observable.
6. A capability-update proposal cannot alter the selected target profile through
   retrieval.

### 12.3 Packet and campaign safety

1. `build_packet.py` produces identical output for identical corpus revision and
   query input.
2. Exact applicable counterexamples are present in the packet.
3. The four authority sections prevent Designer-only hypotheses and raw artifacts
   from reaching Coder.
4. A snapshot remains immutable after later corpus updates.
5. Empty, missing, or failed Wiki queries produce a schema-valid empty packet and
   do not block the local workflow.
6. A decision and Coder result reference only the frozen round snapshot.

### 12.4 Promotion and lifecycle

1. A terminal KernelEvidenceRecord can create only a proposal, not an active
   claim.
2. User approval is required to activate, update, replicate, supersede, or revoke
   a claim.
3. Runtime/profile/fingerprint changes mark affected projections stale according
   to version-claim rules.
4. Contradictory qualified claims remain separately retrievable under their
   distinct regime tags.

### 12.5 Quality evaluation

A leave-one-operator replay uses completed local projects. The Wiki is built
without the held-out operator's sources, then compared with an empty-packet
baseline. The evaluation reports decision validity, compile-smoke success,
correctness pass rate, repeated-counterexample rate, terminal classification,
and rounds to the first accepted candidate. Wiki adoption must not reduce
correctness, reproducibility, or target isolation; performance claims remain
subject to the existing Verifier protocol.

## 13. Implementation Sequencing Constraint

The implementation plan must proceed in this order:

1. standalone skill skeleton, schemas, taxonomy, source policy, and validators;
2. deterministic sources/claims/wiki/query compilation and fixture corpus;
3. manual local MLU/GCU seed sources and approved claims;
4. structured query and packet generation with tests;
5. optional kernel-opt-loop adapter and sidecar evidence integration; and
6. candidate ledgers and limited external artifact handling.

No phase may introduce code RAG, automatic promotion, target-profile mutation,
remote service dependency, vector retrieval, or active-project migration.

## 14. Governance

This document is the architectural source of truth for KernelWiki v1. Any
implementation plan must cite its sections, assign ownership, list the exact
files to change, and include the acceptance fixtures in Section 12. Future
features outside this scope require a separate approved specification revision.

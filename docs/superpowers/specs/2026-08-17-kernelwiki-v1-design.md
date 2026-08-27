# KernelWiki v1 Design — Ascend-First Open Kernel Knowledge Skill

**Created**: 2026-08-17

**Revised**: 2026-08-21

**Status**: Draft for user review

**Supersedes**: the earlier claim-plane and prebuilt-KnowledgePacket design in this file

**Reference model**: [MIT Han Lab KernelWiki](https://github.com/mit-han-lab/KernelWiki) at `b6b4301f15e8ce6955a56776690643ce5db369e6`

**Related control plane review point**: `skills/kernel-opt-loop/` as inspected at `origin/dev@389053e203610a78cb95340520f3b60bb33a58fe`; the normative authority is always the latest reviewed, checked-in vNext contracts rather than this historical review SHA

## 1. Summary

KernelWiki v1 is a standalone, Git-versioned, Ascend-first knowledge skill for
finding and synthesizing reusable kernel-writing experience from open-source
repositories, official documentation, performance pull requests, retained code
artifacts, and locally verified optimization campaigns.

The design adopts the useful product shape of MIT KernelWiki:

```text
sources/ -> wiki/ -> queries/
```

- `sources/` preserves pinned evidence and provenance.
- `wiki/` contains concise, cross-referenced knowledge cards that agents and
  humans read directly.
- `queries/` contains deterministic, generated views by problem, technique,
  hardware feature, kernel type, language, target, source repository, version,
  and evidence level.

In this specification, a **Wiki Card is one Markdown Wiki Page**. There is no
separate authored atomic-claim or Evidence Card layer. Each Card covers one
coherent, reusable topic with structured applicability metadata, source
references, and optional positive or counterexample cases.

KernelWiki is not organized around the current competition operators. A Track 2
operator is a query context: Designer extracts its semantic and performance
features, then queries general hardware, technique, pattern, language,
measurement, and kernel-case Cards. When a local campaign produces reliable
results, those results may be added as scoped examples to a general Card or, when
independently instructive, as a kernel case-study Card.

KernelWiki is a knowledge plane, not a second optimization controller. Current
project semantics, implementation profiles, project capability claims, typed
Sketches, Decisions, binding ledgers, Verifier fact packs, and deterministic
verdicts remain authoritative under `kernel-opt-loop`.

## 2. Motivation

`kernel-opt-loop` now preserves a strong project-local evidence chain:

- immutable base and harness identities;
- runtime and measurement fingerprints;
- implementation-profile snapshots and project capability claims;
- typed Sketches and immutable Decisions;
- candidate-to-Sketch binding ledgers;
- correctness, lowering, profiling, and wall-time evidence;
- deterministic attribution verdicts;
- canonical candidate and report pointers.

That chain makes one campaign credible, but it does not make open-source and
historical kernel engineering knowledge easy to discover. Valuable information
remains scattered across upstream repositories, PR descriptions, changed files,
official documents, blogs, benchmark reports, and completed local campaigns.

KernelWiki supplies the missing research surface. It should help Designer and,
under narrower rules, Coder answer questions such as:

- Which kernel-writing techniques are used for this kernel family?
- Which Ascend-native repositories contain relevant implementations?
- What symptoms suggest fusion, tiling, double buffering, layout changes,
  persistent scheduling, or resource-budget changes?
- What implementation and compiler pitfalls have been observed for this exact
  target profile?
- Which apparently successful device-level changes failed to improve
  synchronized wall time?
- Which capability facts are Unknown and require profile onboarding or a bounded
  probe?
- Which open-source examples are exact-target evidence, family evidence,
  backend-level context, or cross-backend analogy only?

The Wiki must improve research quality without weakening local-loop evidence and
promotion rules.

## 3. Design Principles

### 3.1 Source, synthesis, and generated views are separate

Raw evidence, editorial synthesis, and navigation must not become one mutable
blob. Source records are provenance-first. Wiki Cards are reviewed synthesis.
Query views are disposable deterministic materializations.

### 3.2 Wiki Card equals Wiki Page

A Card is one Markdown page with YAML frontmatter and a concise body. It is both
the human-readable unit and the machine-indexed unit. `compiled/catalog.jsonl`
contains only generated metadata and hashes; it is not a second copy of the
knowledge body.

### 3.3 General knowledge comes first

The primary ontology is hardware features, kernel-writing techniques,
performance symptoms, language/runtime guides, measurement methods, and kernel
families. Competition operators are query fixtures and local evidence sources,
not the default Wiki hierarchy.

### 3.4 Exact and broad evidence classes never collapse

Every target-sensitive Card or example distinguishes:

- exact target/profile/runtime evidence;
- device-family evidence;
- backend-level evidence;
- analogy-only evidence;
- target identity unknown.

A broad or analogy result cannot silently become an exact-target recipe.

### 3.5 Unknown is not Supported or Unsupported

Missing profile evidence remains Unknown. A Wiki page may identify a capability
gap and a required probe, but it cannot promote a capability into an
implementation profile.

### 3.6 Knowledge is advisory until restated in campaign authority

Designer may use Wiki research to select and justify a mechanism, but adopted
requirements become normative only when restated in the validated typed Sketch
and Decision. Coder may use only profile-compatible implementation guidance that
preserves those frozen artifacts.

### 3.7 Positive and negative examples are equally important

Accepted results, no-improvement results, design rejections, lowering gaps, and
stable implementation pitfalls can all be useful. Contradictions are represented
with their scopes and reconsideration conditions; they are not averaged away.

### 3.8 Network access is explicit maintenance work

Production queries are offline. Network access occurs only through explicit
candidate refresh or source-capture commands. No query path silently contacts an
external service.

## 4. Goals

KernelWiki v1 must:

1. exist as a standalone skill under `skills/kernelwiki/`;
2. use a MIT-KernelWiki-style `sources -> wiki -> queries` architecture;
3. curate Ascend-native open-source repositories before broad cross-backend
   expansion;
4. preserve PR, commit, file, artifact, license, architecture, version, and hash
   provenance;
5. provide readable Wiki Cards for hardware, techniques, patterns, languages,
   runtimes, measurement, migrations, and instructive kernel cases;
6. generate deterministic query views and a generated catalog;
7. support active Designer research through search, filters, page retrieval,
   source following, and reviewed artifact inspection;
8. support active Coder research through a narrower exact-profile implementation
   view that cannot change the frozen Sketch or Decision;
9. support structured, deterministic applicability resolution and bounded role
   dossiers without replacing direct page reading;
10. use completed local campaigns as scoped positive or counterexample cases on
    general Cards;
11. turn terminal campaign evidence into reviewable knowledge-lift proposals,
    never automatic Wiki publication;
12. preserve profile, project, Sketch, binding, Verifier, verdict, and canonical
    pointer authority in `kernel-opt-loop`;
13. use Track 2 operators as holdout query fixtures that test whether general
    knowledge is useful without operator-specific Wiki pages; and
14. demonstrate target isolation, Unknown preservation, deterministic output,
    citation integrity, contradiction visibility, and useful retrieval through
    hardware-free tests.

## 5. Non-goals

The following are outside v1:

- a vector database or embedding service;
- a persistent query daemon;
- automatic code generation from retrieved sources;
- automatic copying of a historical or upstream kernel into a candidate;
- automatic Wiki publication from a Verifier report;
- automatic target-profile mutation;
- using source-reported performance as current-project acceptance evidence;
- treating a CUDA, Hopper, Blackwell, MLU, GCU, or other backend implementation
  as an Ascend Coder recipe;
- creating a Wiki Card for every competition operator;
- migrating historical v1/v2 campaigns into vNext;
- changing base, harness, correctness, measurement, canonical-pointer, or verdict
  semantics;
- making KernelWiki availability a prerequisite for a valid local optimization
  campaign;
- implementing an AtomGit or arbitrary-web crawler before stable source adapters
  and provenance rules exist; and
- solving the missing `triton_ascend` or `ascendc` canonical implementation
  profiles inside KernelWiki.

## 6. Ownership and Authority Boundaries

KernelWiki is a research and knowledge-maintenance skill. `kernel-opt-loop` is
the campaign control plane.

| Concern | Authority |
|---|---|
| Open-source candidate ledgers, source records, Wiki Cards, generated views, corpus validation | KernelWiki |
| Base, harness, operator semantics, public contract, correctness tolerances | Current project |
| Target legality and capability state | Frozen implementation-profile snapshot |
| Current-run fallback authorization | Project capability claim |
| Algorithm, dataflow, precision, effects, aliases, host plan, expected mechanism | Typed Sketch and Decision |
| Candidate conformance to the Sketch | Binding ledger |
| Correctness, lowering, profiler facts, wall measurements | Verifier fact pack and report |
| Design/code/evidence attribution and routing | Deterministic verdict and Orchestrator |
| Canonical candidate/report pointers, counters, run state, Git ledger | Orchestrator |
| Publishing or changing a Wiki Card | Curator review and Git commit |

KernelWiki may cite authoritative project and profile facts. It may not replace or
revise them.

## 7. Repository Architecture

```text
skills/kernelwiki/
  SKILL.md
  README.md
  index.md
  requirements.txt

  references/
    primer.md
    schema.md
    examples.md
    source-policy.md
    inclusion-policy.md
    consultation-contract.md
    knowledge-lift-contract.md

  data/
    schemas.yaml
    taxonomy.yaml
    aliases.yaml
    version-claims.yaml
    tool-versions.yaml
    source-repositories.yaml
    inclusion-policy.yaml
    size-budget.yaml

  candidates/
    repos/
      triton-ascend.yaml
      vllm-ascend.yaml
      cann-samples.yaml
      triton-ascend-kernels.yaml
      mskl.yaml
    experience/

  sources/
    prs/<repo>/PR-<number>.md
    commits/<repo>/<sha>.md
    docs/
    blogs/
    contests/
    local/ascend/

  wiki/
    hardware/
    techniques/
    patterns/
    languages/
    runtimes/
    measurement/
    kernels/
    migration/

  artifacts/
    prs/<repo>/PR-<number>/
    commits/<repo>/<sha>/
    docs/<slug>/
    kernels/<slug>/
    local/<campaign-source-id>/

  queries/
    by-problem.md
    by-technique.md
    by-hardware-feature.md
    by-kernel-type.md
    by-language.md
    by-target.md
    by-source-repo.md
    by-version.md
    by-evidence-level.md

  compiled/
    catalog.jsonl

  scripts/
    query.py
    get_page.py
    grep_wiki.py
    build_dossier.py
    validate_consultation.py
    validate.py
    generate_indices.py
    repo_status.py
    refresh_candidate_ledger.py
    capture_source.py
    generate_source_pages.py
    validate_provenance.py
    propose_from_campaign.py
    validate_lift.py

  tests/
```

`queries/` and `compiled/catalog.jsonl` are generated and checked into Git for
human navigation, offline use, and deterministic review. They are never edited
manually.

## 8. Source Repository Registry and Candidate Discovery

### 8.1 Repository registry

`data/source-repositories.yaml` records discoverable repositories and manual
source lanes.

Example:

```yaml
repositories:
  - id: vllm-ascend
    host: github
    repo: vllm-project/vllm-ascend
    lane: ascend-native
    languages: [ascendc, cpp, python]
    target_families: [ascend]
    kernel_path_globs:
      - "**/ops/**"
      - "**/kernels/**"
      - "**/*ascendc*"
    skip_path_globs:
      - "**/docs/**"
      - "**/tests/**"
      - "**/benchmarks/**"
    search_terms:
      - AscendC
      - kernel
      - performance
      - fused
      - topk
      - attention
      - reduction
      - tiling
      - double buffer
```

Supported source adapters in v1 are:

- `github`: candidate discovery and pinned PR/commit capture;
- `manual`: reviewed metadata for official pages, AtomGit, or sources without a
  stable automated adapter.

An `atomgit` adapter is a future extension. It must not be simulated by brittle
HTML scraping in v1.

### 8.2 Initial Ascend-native lanes

Phase A begins with two automated GitHub lanes:

- `Ascend/triton-ascend`;
- `vllm-project/vllm-ascend`.

Manual candidate records initially cover:

- CANN and AscendC official operator samples;
- CANN samples;
- `Ascend/triton-ascend-kernels`;
- `Ascend/mskl`;
- official profiling and architecture documentation.

### 8.3 Cross-backend research lane

Selected CUDA/Hopper/Blackwell sources may be included for Designer research,
including material already curated by MIT KernelWiki. Such sources are always
marked analogy-only for Ascend unless independent Ascend evidence supports the
same general mechanism.

They cannot enter a Coder result for an Ascend profile.

### 8.4 Candidate ledgers

Each repository has a reviewed ledger:

```yaml
repo: vllm-project/vllm-ascend
searched_at: <UTC>
keywords_used: [AscendC, kernel, performance, fusion, topk]
total_candidates: 0
included: 0
deferred: 0
excluded: 0
prs:
  - number: 814
    title: Custom AscendC Kernel of Multi-Step Prepare Input
    date: <date>
    decision: include
    reason: adds a device-kernel implementation with performance evidence
    kernel_types: [data-preparation, fused-kernel]
    techniques: [kernel-fusion, launch-collapse]
    languages: [ascendc]
    changed_paths: []
    files_reviewed_count: 0
    target_evidence: []
```

Candidate decisions are:

- `include`: sufficient kernel relevance and capturable provenance;
- `defer`: potentially useful but target, license, path, or evidence is
  incomplete;
- `exclude`: outside scope or lacking device-kernel knowledge.

Typical exclusions include Python-dispatch-only, configuration-only,
benchmark-only, test-only, host-framework-only, and documentation-only changes
without independent kernel-programming value.

## 9. Source Records

A Source records evidence without making a transferable conclusion.

### 9.1 PR source schema

A PR source includes:

- stable source ID;
- repository, PR number, title, author, date, URL;
- open/merged/closed status and merge SHA when applicable;
- capture date;
- target architecture evidence and disposition;
- languages, kernel types, techniques, hardware features, and tags;
- inclusion and scope decisions;
- complete or explicitly incomplete changed-file accounting;
- upstream description and exact locators;
- hashes for description, excerpt, files, patch, and retained assets;
- artifact directory and provenance reference;
- license state.

### 9.2 Local campaign source schema

A local campaign source pins:

- repository commit;
- project path;
- Decision, Sketch, binding, report, verdict, candidate, base, and harness paths
  when available;
- SHA-256 for every cited artifact;
- target/profile/runtime scope;
- measurement fingerprint;
- terminal result;
- exact observations copied from structured artifacts;
- historical contract version;
- missing evidence declarations.

A local source never claims that the result transfers to another operator,
shape, runtime, language, or profile.

### 9.3 Source immutability

Captured Source records and artifact bundles are immutable. When upstream or
local evidence changes, create a new source revision or source ID. Wiki Cards may
add the new source while retaining historical citations.

## 10. Artifact and Provenance Model

Every retained asset bundle has one `PROVENANCE.yaml`:

```yaml
origin_url: ...
upstream_repo: ...
upstream_sha: ...
license: ...
retrieved_at: ...
asset_mode: verbatim | extracted | derived
allowed_audiences: [designer]
coder_access: denied | snippet-only | exact-profile
source_ids: []
files:
  - local_path: ...
    upstream_path: ...
    heading_path: ...
    role: pr-diff | upstream-file | snippet | historical-candidate | bench-record
    mode: verbatim | extracted | derived | upstream-patch
    sha256: ...
```

Rules:

1. `verbatim` requires an upstream revision and byte hash.
2. `extracted` requires an upstream revision and source locator.
3. `derived` requires source IDs and must not be represented as upstream text.
4. Unknown or incompatible licenses permit metadata-only Source records but block
   code asset exposure.
5. Full upstream kernels and historical candidates are Designer-only by default.
6. Coder may inspect only explicitly approved short snippets or exact-profile
   implementation assets.
7. Production query never fetches an artifact from the network.
8. Repository size is capped and checked by a deterministic size-budget script.

## 11. Wiki Card Model

### 11.1 Card types

v1 Card types are:

- `hardware`: architecture and execution/memory features;
- `technique`: general kernel-writing techniques;
- `pattern`: symptom, likely causes, and candidate techniques;
- `language`: programming-language and DSL guidance;
- `runtime`: launcher, loader, compiler, integration, and lifecycle behavior;
- `measurement`: timing, profiling, attribution, and comparability guidance;
- `kernel`: instructive open-source or local kernel case studies;
- `migration`: bounded translation and transfer lessons between languages or
  architectures.

### 11.2 Topic cohesion

Each Card covers one coherent hardware feature, technique, diagnostic pattern,
language/runtime topic, measurement method, migration lesson, or kernel case.
A Card may synthesize several closely related statements and examples, as MIT
KernelWiki pages do. Split a page only when its audience, target scope, version
scope, evidence class, or practical use differs materially.

A technique Card may contain multiple positive and counterexample cases, but all
examples must remain relevant to the same general technique topic.

### 11.3 Frontmatter

Example technique Card:

```yaml
---
id: technique-kernel-fusion
title: Kernel fusion
type: technique
audiences: [designer]
authority: advisory
summary: >
  A legal producer-consumer fusion boundary can reduce global materialization
  and launch overhead, but resource growth and host overhead determine whether
  synchronized wall time improves.

targets: [backend-neutral, ascend]
target_match: backend
languages: [ascendc, triton]
kernel_types: [attention, topk, moe, reduction]
tags: [kernel-fusion, launch-collapse, materialization]
symptoms: [launch-bound, materialization-overhead]

preconditions:
  - a legal semantic fusion boundary exists
  - required intermediate lifetime can remain inside the kernel boundary
exclusions:
  - fusion violates public output lifetime or alias semantics
expected_observables:
  - kernel_count_per_call decreases
  - intermediate global traffic decreases
risks:
  - register or local-memory pressure increases
  - occupancy or parallelism decreases
  - host or synchronization overhead dominates

confidence:
  evidence_level: source-reported
  replication: multi-source
reproduction:
  level: snippet

sources:
  - source-vllm-ascend-fused-gdn
  - source-triton-ascend-fusion-doc
related:
  - pattern-launch-bound
  - pattern-device-win-wall-loss
prerequisites: []
version_sensitive: []

examples:
  - id: example-ascend-groupedtopk-launch-collapse
    role: positive
    source_id: source-local-ascend-groupedtopk-round-001
    target_id: ascend910b
    implementation_profile_id: triton_ascend
    profile_authority: historical-noncanonical
    operator_family: topk-routing
    shape: "T=83,E=256,K=8"
    dtype: "recorded source regime"
    outcome: accepted
    observed:
      kernel_count_per_call: "19 -> 1"
      wall_improvement_pct: 54.88
    measurement_fingerprint: <sha256>
    transfer_boundary:
      - does not establish T=2600/topk=128 behavior

  - id: example-ascend-flexattention-device-win-wall-loss
    role: counterexample
    source_id: source-local-ascend-flexattention-round-003
    target_id: ascend910b
    implementation_profile_id: triton_ascend
    operator_family: attention
    outcome: no-improvement
    observed:
      device_us_per_call: "54.43 -> 24.05"
      wall_improvement_pct: -8.34
    measurement_fingerprint: <sha256>
    lesson: device-level success did not produce synchronized wall improvement
    reconsider_when:
      - host and synchronization overhead is independently reduced
---
```

### 11.4 Body structure

Technique and pattern bodies use a consistent LLM-readable structure:

```text
Summary
Problem or symptom
Mechanism
Applicability
Implementation approaches
Expected observables
Risks and counterexamples
Examples
Transfer boundaries
Required local checks
Sources
```

Kernel case studies add:

```text
Shape and contract
Implementation structure
Source excerpt or snippet
Measured claims
What transfers
What does not transfer
```

## 12. Confidence, Reproducibility, and Performance

### 12.1 Confidence is Card-level, examples remain individually scoped

Allowed evidence levels are:

- `local-verifier`;
- `official-doc-and-upstream-code`;
- `source-reported`;
- `inferred`;
- `experimental`.

A strong Card-level label does not widen any example's target, runtime, shape, or
measurement scope.

### 12.2 Reproducibility ladder

- `concept`: text only;
- `pseudocode`: language-neutral structure;
- `snippet`: retained code fragment with provenance;
- `runnable`: buildable or executable artifact with pinned environment;
- `benchmarked`: runnable plus structured measurement evidence.

The presence of a code fence alone does not prove runtime reproducibility. A
`runnable` or `benchmarked` label requires validator-enforced artifacts and
commands.

### 12.3 Performance observations

Every performance example records at least:

- target ID;
- implementation profile ID and authority state;
- runtime fingerprint or source reference;
- measurement fingerprint when local;
- operator family;
- shape and dtype;
- baseline and candidate identity when available;
- metric, value, and statistic;
- source ID and exact locator;
- terminal classification;
- comparability state.

Comparability states are:

- `source-reported`;
- `historical-local`;
- `project-reproduced`;
- `comparable-to-current-baseline`.

Only current Verifier evidence under the active measurement fingerprint can
control campaign adoption.

## 13. Generated Catalog and Query Views

### 13.1 Catalog

`compiled/catalog.jsonl` contains one record per Source or Card:

```json
{"id":"technique-kernel-fusion","path":"wiki/techniques/kernel-fusion.md","body_sha256":"...","type":"technique","audiences":["designer"],"targets":["backend-neutral","ascend"],"kernel_types":["attention","topk","moe","reduction"],"tags":["kernel-fusion"],"source_count":2,"positive_example_count":1,"counterexample_count":1}
```

The catalog is generated from source files and frontmatter. It contains no
independently authored statements.

### 13.2 Query views

`generate_indices.py` produces:

1. `by-problem.md`: symptom -> pattern -> candidate techniques;
2. `by-technique.md`: technique, targets, confidence, reproducibility, source and
   example counts;
3. `by-hardware-feature.md`: feature -> hardware, technique, kernel, and source
   pages;
4. `by-kernel-type.md`: kernel family -> sources, case studies, techniques, and
   patterns;
5. `by-language.md`: AscendC, Triton Ascend, and other language guidance;
6. `by-target.md`: exact, family, backend, analogy, and unknown evidence;
7. `by-source-repo.md`: repository -> captured PR/commit pages and derived Cards;
8. `by-version.md`: version-sensitive pages and last verification state;
9. `by-evidence-level.md`: confidence, reproducibility, and local replication.

Generated views are first-class human and agent navigation interfaces, matching
the successful MIT KernelWiki pattern.

## 14. Active Query Interfaces

### 14.1 Unified search

`query.py` supports:

```bash
python3 scripts/query.py "ascend sparse attention masked gather"
python3 scripts/query.py --tag double-buffering --type technique
python3 scripts/query.py --repo vllm-ascend --limit 20
python3 scripts/query.py --language ascendc --kernel-type topk
python3 scripts/query.py --target ascend910b --symptom launch-bound
python3 scripts/query.py --audience coder --profile-snapshot <path>
```

Filters include:

- type;
- tag;
- repository;
- language;
- target;
- architecture disposition;
- symptom;
- kernel type;
- confidence;
- reproducibility;
- audience;
- has-code.

Exploratory ranking is transparent and lexical: title, structured fields,
aliases, bounded body hits, then stable path/ID tie-break. It discovers pages; it
does not determine campaign legality.

### 14.2 Page retrieval

`get_page.py` resolves an ID or path and supports:

- body or frontmatter output;
- `--follow-sources`;
- bounded source excerpts;
- `--include-code` subject to audience and provenance policy;
- canonical JSON metadata output.

### 14.3 Regex search

`grep_wiki.py` supports scoped regex search across Wiki bodies, source pages, or
both. It is an investigation tool, not an applicability gate.

## 15. Role-aware Applicability and Dossiers

### 15.1 Active access model

Designer and Coder actively access Wiki Cards. They are not passive consumers of
one preselected packet.

Future loop flow:

```text
Orchestrator pins one KernelWiki revision
        ↓
Designer actively searches, reads pages, follows sources, inspects approved assets
        ↓
Designer records adopted/rejected research and writes Sketch + Decision
        ↓
Orchestrator validates consultation references and campaign authority
        ↓
Coder actively searches a narrower implementation view
        ↓
Coder records page/snippet use and preserves the frozen Sketch/Decision
```

### 15.2 Designer view

Designer may inspect:

- exact target, family, and backend Cards;
- capability-gap and measurement Cards;
- positive and counterexample cases;
- cross-backend analogy Cards;
- source pages;
- reviewed code artifacts.

Every result states its match class. Analogy-only evidence cannot become a target
capability or Coder recipe.

### 15.3 Coder view

Coder may inspect only Cards that satisfy all of the following:

1. `audiences` includes `coder`;
2. target/profile/runtime match is exact;
3. the referenced profile/project capability is not Unknown or Unsupported;
4. the guidance preserves algorithm, dataflow, precision, effects, aliases, Host
   Plan, and public interface;
5. it binds to one or more frozen Sketch statements;
6. version claims are current;
7. snippets and artifacts permit Coder access;
8. provenance and license checks pass.

A Coder query never falls back to another backend or language. If a useful Card
would require a Sketch or Decision change, Coder reports the existing deviation
or capability route instead of applying it.

### 15.4 Structured context

Designer context contains current project facts but no invented capability:

```json
{
  "schema_version":1,
  "role":"designer",
  "project":"kernels/track2-clike/index_topk/ascendc",
  "target_id":"ascend910b",
  "implementation_profile_id":"ascendc",
  "implementation_profile_status":"missing",
  "operator_tags":["attention","topk","rope","causal-mask"],
  "kernel_types":["sparse-attention","topk","reduction"],
  "shape_signature":{"batch":8,"sequence":2600,"hidden":1024,"topk":128},
  "dtypes":["bf16","fp32","int32"],
  "semantic_features":["causal-mask","rope","topk-ordering"],
  "current_bottlenecks":[]
}
```

Coder context additionally requires frozen Sketch, Decision, implementation
profile, and project capability claim references and hashes.

### 15.5 Applicability resolution

`build_dossier.py` takes explicit page IDs or query results and deterministically
classifies them as:

- `admitted`;
- `conditional`;
- `analogy_only`;
- `counterexamples`;
- `capability_gaps`;
- `excluded`.

Stable exclusion reasons include:

- `audience-mismatch`;
- `target-mismatch`;
- `profile-missing`;
- `profile-version-mismatch`;
- `capability-unknown`;
- `capability-unsupported`;
- `sketch-change-required`;
- `version-stale`;
- `artifact-designer-only`;
- `license-unapproved`;
- `source-broken`.

Ranking occurs only after admissibility and uses a documented tuple over target
specificity, profile/runtime exactness, kernel-type overlap, semantic/regime
overlap, evidence level, reproduction level, freshness, lexical relevance, and
stable ID.

Counterexamples and capability gaps use separate groups so positive techniques
cannot rank them out of the dossier.

### 15.6 Bounded dossiers

Default Designer limits:

```yaml
pages: 12
source_excerpts: 8
snippets: 3
```

Default Coder limits:

```yaml
pages: 6
source_excerpts: 4
snippets: 2
```

An empty or unavailable Wiki returns a schema-valid empty dossier and does not
block the local loop.

## 16. Consultation Records

### 16.1 Designer consultation

A future loop adapter records:

```text
rounds/kernelwiki_consultation_NNN.json
```

with:

- Wiki revision;
- context and dossier hashes;
- queries and result receipts;
- pages read;
- sources followed;
- artifacts inspected;
- pages adopted and rejected;
- counterexamples reviewed;
- capability gaps;
- page-to-Sketch statement bindings.

Designer writes the consultation record with its Decision and Sketch.
Orchestrator validates and hashes it before Coder dispatch.

### 16.2 Coder consultation

Coder records in `coder_result_NNN.md`:

- Wiki revision;
- pages read and used;
- snippets inspected;
- Sketch statement bindings;
- confirmation that no forbidden design field changed;
- any required deviation or capability route.

Coder does not read Designer's research as a replacement for the Decision; it
may use only the frozen campaign artifacts and eligible implementation Cards.

## 17. Track 2 Usage Model

Track 2 operators do not become Wiki Cards by default. Designer reads a base and
forms a structured query.

For `sparse_attn`, an illustrative query context is:

```yaml
target: ascend910b
language: ascendc
kernel_types: [sparse-attention, topk, gather, softmax, reduction]
semantic_features: [invalid-index-mask, denominator-only-sink]
dtypes: [bf16, fp32, int32]
shape_regime:
  batch: 8
  sequence: 2600
  heads: 64
  head_dim: 128
symptoms: [memory-bound, launch-bound]
```

The query should return general technique Cards, Ascend hardware/language guides,
measurement Cards, source PRs, instructive kernel cases, local positive and
counterexample cases, and capability gaps.

It must not return a fabricated AscendC recipe while the canonical AscendC
profile is missing.

The three Track 2 references are evaluation fixtures for:

- sparse attention, masked gather, top-k, and softmax;
- learned index top-k with RoPE and causal masking;
- repeated small Sinkhorn reductions.

They are not initial Wiki ontology roots.

## 18. Local Campaign Knowledge Lift

### 18.1 Authority flow

Verifier never writes Wiki conclusions directly.

```text
Verifier fact pack/report
+ Decision causal graph
+ typed Sketch
+ binding ledger
+ deterministic verdict
+ terminal Git commit
        ↓
Orchestrator invokes proposal extractor
        ↓
project-side knowledge-lift proposal
        ↓
Curator include/defer/exclude review
        ↓
immutable local Source + Wiki Card example update
```

### 18.2 Safe timing

Extraction occurs only after a terminal round commit or final-stop commit. No Wiki
write occurs during measurement exclusivity or before verdict attribution.

### 18.3 Proposal artifact

The extractor writes a project-side proposal such as:

```text
state/kernelwiki_lift_proposal_<run-epoch>.json
```

It pins profile, project claim, runtime, measurement, Sketch, Decision, binding,
Coder result, report/fact pack, verdict, and terminal commit identities.

It records:

- expected intervention and causal observables;
- actual correctness, lowering, profiler, and wall observations;
- deterministic terminal classification;
- exact target/shape/runtime scope;
- suggested general Card and example role;
- transfer boundaries;
- reconsideration conditions.

It contains no instruction for the next candidate.

### 18.4 Outcome mapping

- `accepted`: candidate positive example, scoped implementation evidence, or
  measurement example;
- `no-improvement`: counterexample, residual-bottleneck, or device-win/wall-loss
  example when evidence supports it;
- `screened-out`: slower-result example only, with no unsupported causal claim;
- `design-rejected`: design pitfall or capability-gap proposal according to the
  verdict rule;
- `lowering-unknown`: Unknown/probe proposal, never Unsupported;
- `candidate-failed`: implementation pitfall only when repeated or tied to a
  stable exact-profile diagnostic;
- `environment-blocked`: deferred by default.

### 18.5 Review and publication

Imported proposals live in `candidates/experience/` with:

```yaml
decision: include | defer | exclude
reviewed_by: ...
reviewed_at: ...
rationale: ...
```

An included proposal creates an immutable local Source and either:

- appends a scoped positive/counterexample case to an existing general Card; or
- creates a new general Card when a genuinely new mechanism exists; or
- creates a kernel case-study Card when the implementation has independent
  teaching value.

Git review and commit are the publication boundary. There is no independent
`claims/proposed|active` lifecycle.

### 18.6 Coder eligibility never auto-promotes

New local examples default to Designer visibility. Adding Coder visibility
requires a separate review confirming exact profile/runtime scope, local
reproduction, preserved Sketch semantics, approved snippets, and non-Unknown
profile capability.

## 19. Ascend-First Initial Corpus

The initial corpus is built from open-source evidence, not competition-operator
Cards.

### 19.1 Initial general Cards

Phase B targets approximately 8–12 initial Cards across:

- AscendC language and build model;
- Triton Ascend language/backend model;
- Ascend kernel launch/runtime behavior;
- CANN profiling and device attribution;
- kernel fusion and launch collapse;
- tiling and work partitioning;
- double buffering and software pipelining;
- memory access and layout transformation;
- top-k/selection/reduction strategies;
- launch-bound and materialization patterns;
- device-time improvement without wall improvement;
- version/profile qualification boundaries.

A small number of kernel case-study Cards may be added when captured sources
contain independently instructive implementations.

### 19.2 Initial local examples

Selected historical Ascend campaigns provide scoped examples for general Cards,
including:

- grouped top-k launch collapse;
- grouped top-k output allocation reuse;
- attention fusion;
- device-time improvement with wall regression;
- materialized attention losing to a native fused path;
- reduction fusion with insufficient wall gain;
- conflicting output-reuse outcomes.

Historical `triton_ascend` evidence is explicitly noncanonical until a reviewed
machine-readable implementation profile and probes exist.

### 19.3 Initial Coder corpus

No AscendC Coder-eligible implementation Card is published while the canonical
AscendC profile, build/runner/profiler payloads, source analyzer, and local
qualification evidence are missing.

A Coder query in that state returns an exact-profile empty result and related
capability-gap pages.

## 20. Error Handling and Offline Behavior

- malformed user input: exit `2` with stable `error:` message;
- invalid corpus: fail closed and return no partial production dossier;
- no match: success with a schema-valid empty result;
- unavailable Wiki: valid empty consultation and local workflow continues;
- unresolved ID/link: corpus validation failure;
- stale version claim: Card is conditional or excluded according to role;
- missing artifact: page remains readable when metadata suffices, but code access
  is denied and the missing asset is reported;
- hash mismatch: provenance failure;
- unknown license: metadata-only source, no code exposure;
- unsupported host adapter: manual candidate record required;
- generated-index drift: validation failure;
- size-budget overflow: capture or publication failure;
- revision cannot be resolved: future loop consultation cannot be frozen.

All production query paths are offline and deterministic.

## 21. Validation and Acceptance Requirements

### 21.1 Corpus integrity

1. Every Source and Card validates against its schema.
2. IDs are unique and all source/related/prerequisite/candidate-technique links
   resolve.
3. Unknown vocabulary values fail validation.
4. Every example resolves to a Source and includes required scope fields.
5. Every version pointer resolves bidirectionally.
6. Every retained artifact validates provenance, license, asset mode, paths, and
   hashes.
7. Every Coder-visible Card satisfies exact-profile and code-access policy.
8. Generated catalog and query views are current and byte-stable.

### 21.2 Source policy

Fixtures prove include/defer/exclude behavior for device-kernel changes,
ambiguous sources, wrapper-only changes, config-only changes, benchmark-only
changes, and missing provenance.

### 21.3 Query and target isolation

1. Designer may see exact, family, backend, analogy, counterexample, and gap
   evidence with explicit match classes.
2. Coder sees only exact-profile eligible implementation Cards.
3. Missing AscendC profile produces no Coder recipe.
4. Cross-backend analogy never becomes an Ascend Coder result.
5. Unknown never becomes Supported or Unsupported.
6. Positive pages do not suppress exact counterexamples or capability gaps.
7. Identical corpus revision and inputs produce byte-identical results.

Required adversarial queries cover:

- generic versus dtype/shape-specific `tl.dot` support;
- output-reuse positive and negative examples;
- device-time versus wall-time conclusions;
- grouped-top-k versus index-top-k transfer;
- raw torch profiler versus CANN device evidence.

### 21.4 Knowledge lift

1. Extractor requires an explicit validated artifact chain.
2. Extractor writes proposals only, not Wiki or Source pages.
3. Accepted and no-improvement examples preserve exact scope and measurement
   identity.
4. Screened-out results cannot create unsupported mechanisms.
5. Lowering Unknown remains Unknown.
6. Single code failures and environment blocks default to nonpublication.
7. Contradictory examples remain visible.
8. Coder eligibility requires separate review.

### 21.5 Holdout evaluation

#### Repository holdout

Hold out one Ascend-native repository or PR group. Evaluate whether existing
Cards explain it, or whether the evidence correctly proposes a new Card.

#### Local campaign holdout

Build initial local examples from grouped top-k, flexattention, and an MHC
reduction campaign. Hold out MM encoder attention and sparse pooler to test
materialization and output-reuse counterexamples.

#### Track 2 query holdout

Keep all three Track 2 references outside the Wiki corpus. Use them as structured
queries and compare retrieved general Cards to an expert gold set.

Suggested acceptance targets:

- unsafe Coder admissions: `0`;
- Unknown-to-Supported/Unsupported conversions: `0`;
- broken source citations: `0`;
- cross-target recipe leaks: `0`;
- required capability-gap recall: `100%`;
- top-5 relevant Card recall on the curated Track 2 query set: at least `90%`;
- applicable counterexample surfacing: at least `90%`.

Performance claims remain subject to local Verifier measurement and are not part
of Wiki acceptance.

### 21.6 Nonfunctional acceptance

- no vector database or daemon;
- no network access during production query;
- query and page retrieval target under two seconds for the initial corpus;
- no mandatory accelerator hardware for tests;
- deterministic generated files;
- repository size budget enforced;
- optional bundled YAML compatibility or one explicitly pinned safe dependency.

## 22. Implementation Phases

Phases A–D are standalone KernelWiki work. They may modify
`skills/kernelwiki/` and KernelWiki documentation, but they must not modify
`skills/kernel-opt-loop/`, active campaign artifacts, base files, harnesses,
profiles, or vNext validators. Phase E requires a separate reviewed integration
specification and implementation plan after its compatibility gates are met.

### Phase A: standalone skill and source pipeline

- skill skeleton and documentation;
- schemas, taxonomy, aliases, version and repository registries;
- candidate ledgers;
- GitHub discovery/capture and manual-source path;
- Source and provenance validation;
- query/get-page/grep tools;
- generated catalog and query views.

### Phase B: general Ascend Cards

- capture initial Ascend-native sources;
- synthesize approximately 8–12 general Cards;
- add a small number of kernel case studies;
- validate query views and offline navigation.

### Phase C: role-aware consultation

- Designer and Coder views;
- deterministic admissibility and dossiers;
- consultation validation;
- Track 2 query fixtures and adversarial tests;
- exact-profile empty behavior for missing AscendC capability authority.

### Phase D: local evidence lift

- artifact-chain validator;
- project-side proposal extractor;
- experience candidate ledger;
- local positive and counterexample cases;
- replication, contradiction, and version-staleness updates.

### Phase E: optional `kernel-opt-loop` adapter

Proceed only after:

- a canonical `triton_ascend` or `ascendc` implementation profile exists for the
  intended Coder path;
- vNext contract/schema mismatches are resolved;
- consultation artifact ownership and validator contracts are stable.

The adapter adds:

- one pinned Wiki revision per round;
- active Designer and Coder access;
- consultation records and hashes;
- final-stop knowledge-lift proposal generation;
- no change to Verifier or verdict authority.

## 23. vNext Integration Compatibility Gate

The future Wiki adapter targets the intended vNext authority model, not observed
implementation inconsistencies. Phase E remains blocked until the owning
`kernel-opt-loop` work establishes compatible, reviewed contracts for:

1. binding data needed by the validator for `elided-by` relations;
2. verdict repair and finalization fields consumed by the validator;
3. legal multi-field final-configuration binding validation;
4. evidence-path and hash trust around the Verifier fact pack;
5. canonical machine-readable Ascend implementation profiles and probes.

These are compatibility conditions, not KernelWiki Phase A–D deliverables. This
specification does not authorize changes to `kernel-opt-loop` to satisfy them.
A later Phase E design must pin supported artifact contract versions and consume
only artifacts that have passed their role validators and Orchestrator gates.

## 24. Governance

- Source capture changes, Wiki Card changes, generated-view changes, and local
  example promotions are reviewed in Git.
- Candidate discovery may be automated; inclusion and publication are reviewed.
- Generated views are never hand-edited.
- Source histories remain immutable.
- Performance examples retain exact scope and are never silently rebased onto a
  new runtime or measurement regime.
- Page scope changes, Coder audience additions, version requalification, and
  contradiction resolution require explicit review.
- The old independent claim lifecycle, direct Coder target recipes, and prebuilt
  Orchestrator-only KnowledgePacket architecture are superseded by this design.

## 25. Design Decision Summary

KernelWiki v1 will be an Ascend-first, MIT-style open kernel knowledge skill:

```text
open-source and local evidence
        ↓
pinned Source records and artifacts
        ↓
general, readable Wiki Cards
        ↓
deterministic query views and active agent research
        ↓
Designer/Coder role-aware consultation
        ↓
validated Sketch/Decision and implementation work
        ↓
terminal campaign evidence
        ↓
reviewable scoped examples back into general Cards
```

This structure helps Track 2 by giving Designer and eventually Coder a broad,
source-backed kernel-engineering memory without turning the competition operators
into the Wiki ontology and without weakening current-project authority.

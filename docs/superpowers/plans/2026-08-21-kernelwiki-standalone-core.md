# KernelWiki Standalone Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the standalone Ascend-first `sources -> wiki -> queries` KernelWiki core with pinned source capture, generic Wiki Cards, deterministic generated views, and offline search/page retrieval.

**Architecture:** Implement a self-contained Python skill under `skills/kernelwiki/`. Reviewed Source Markdown and immutable provenance bundles feed generic Markdown Wiki Cards; a deterministic compiler emits `compiled/catalog.jsonl` and nine checked-in query views. Production query commands read only the local corpus and generated catalog, never the network or `kernel-opt-loop` state.

**Tech Stack:** Python 3 standard library, `PyYAML==6.0.2` with `yaml.safe_load`, Markdown with YAML frontmatter, JSONL, `unittest`, Git.

**Spec:** `docs/superpowers/specs/2026-08-17-kernelwiki-v1-design.md`

## Lean Core Revision

This revision supersedes the exhaustive named-test and failure-injection lists later in the original plan.

- The completed standalone core is guarded by roughly **95 focused tests across Core and Phase C**, not one method per field, malformed type, race, symlink, or internal object.
- Keep representative happy paths, one grouped fail-closed boundary per interface, deterministic output checks, offline/no-loop isolation, and production smokes.
- Local Git-reviewed files are not treated as a hostile concurrent security boundary. Ordinary root confinement, hash verification, no-overwrite publication, and normal exception cleanup are sufficient for v1.
- Capture staging is temporary and recoverable by cleanup; no persistent transaction journal or process-crash replay is required.
- Generated catalog/views are reproducible derived files. Write through a temporary generation directory and `os.replace`; an interrupted partial refresh is repaired by rerunning generation rather than a rollback/trash protocol.
- Additional fuzzing, TOCTOU analysis, forged internal object tests, recursive immutability, proxy/seal machinery, or one-test-per-validator-field belongs to optional follow-up work and must not be reintroduced without explicit user approval.

## Global Constraints

- Work only under `skills/kernelwiki/`, `skills/README.md`, and KernelWiki documentation. Do not modify `skills/kernel-opt-loop/`, campaign directories, profiles, harnesses, or vNext validators.
- Preserve the authored knowledge shape `sources/ -> wiki/ -> queries/`. Do not add `claims/`, Evidence Cards, a claim promotion lifecycle, `KnowledgePacket`, or a persisted packet compiler.
- A Wiki Card is one coherent generic Markdown page, not an atomic proposition. Evidence and transfer qualification remain observation- or example-scoped.
- Keep Ascend-native repositories first. Cross-backend material is metadata-tagged `analogy-only` and cannot become an Ascend implementation recipe.
- Network access is allowed only in explicit maintenance commands such as candidate discovery and source capture. `query.py`, `get_page.py`, `grep_wiki.py`, `validate.py`, and `generate_indices.py` remain offline.
- Seal repository and Track 2 holdout judgments before authoring taxonomy, aliases, ranking, or seed Cards. Holdout files may be read only by final evaluation tests.
- Source and provenance records are immutable. A changed upstream revision creates a new Source ID or revision; capture commands never overwrite an existing Source or artifact bundle.
- Unknown or incompatible licenses allow metadata-only Source records but deny code-asset exposure.
- Generated `queries/*.md` and `compiled/catalog.jsonl` are checked in, byte-stable for identical inputs, and never hand-edited.
- Use stable CLI behavior: malformed input or invalid corpus exits `2` with `error[code]: message`; no match exits `0` with a schema-valid empty result.
- Add no vector database, embedding service, daemon, accelerator dependency, or production-time network fallback.
- Performance observations in Cards are advisory historical/source evidence only. They never control current campaign acceptance.
- Phase C role-aware exact-profile admission is implemented by `2026-08-21-kernelwiki-role-aware-query.md`; Phase D campaign extraction is implemented by `2026-08-21-kernelwiki-offline-knowledge-lift.md`; Phase E is not part of this plan.

## Planned File Map

```text
skills/kernelwiki/
  SKILL.md                         # Agent-facing standalone research/curation contract
  README.md                        # Human setup, maintenance, and query guide
  index.md                         # Links to generated views and Wiki sections
  requirements.txt                # Exactly PyYAML==6.0.2

  references/
    primer.md                      # Source/Card/query mental model
    schema.md                      # Versioned Source, Card, catalog contracts
    examples.md                    # Complete valid Source/Card examples
    source-policy.md               # Immutable capture and license rules
    inclusion-policy.md            # include/defer/exclude editorial policy
    evaluation-protocol.md         # Sealed repository/Track 2 holdout rules

  data/
    schemas.yaml                   # Current schema versions
    taxonomy.yaml                  # Closed Card/source vocabulary
    aliases.yaml                   # Deterministic search aliases
    version-claims.yaml            # Version-sensitive Card registry
    source-repositories.yaml       # Ascend-native and manual lanes
    size-budget.yaml               # Repository and per-bundle byte caps
    evaluation-holdouts.yaml       # Sealed repository/Track 2 boundaries

  candidates/repos/
    triton-ascend.yaml
    vllm-ascend.yaml
    cann-samples.yaml
    triton-ascend-kernels.yaml
    mskl.yaml

  sources/{prs,commits,docs,local}/ # Authored immutable Source Markdown
  wiki/{hardware,techniques,patterns,languages,runtimes,measurement,kernels,migration}/
  artifacts/                       # Immutable retained files plus PROVENANCE.yaml
  queries/*.md                     # Generated first-class navigation views
  compiled/catalog.jsonl           # Generated one-record-per-Card catalog

  scripts/
    kernelwiki_common.py            # YAML/frontmatter, hashing, paths, CLI errors
    corpus.py                       # Source/Card loading and structural validation
    provenance.py                   # Artifact/provenance and size-budget validation
    source_capture.py               # Testable GitHub/manual capture library
    catalog.py                      # Catalog records and Markdown view rendering
    search.py                       # Offline lexical filtering/ranking/page retrieval
    validate.py                     # Whole-corpus validator CLI
    validate_provenance.py          # One-bundle or whole-tree provenance CLI
    capture_source.py               # Explicit discovery/network/manual capture CLI
    generate_indices.py             # Deterministic compiler CLI
    query.py                        # Offline search CLI
    get_page.py                     # Offline Card/Source retrieval CLI
    grep_wiki.py                    # Offline scoped regex CLI

  tests/
    fixture_factory.py
    test_common.py
    test_corpus.py
    test_provenance.py
    test_source_capture.py
    test_catalog.py
    test_search.py
    test_contracts.py
    fixtures/valid-corpus/
    fixtures/invalid-corpus/
```

### Stable Core Interfaces

```python
class KernelWikiError(Exception):
    code: str
    message: str
    path: Path | None

@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    path: Path
    source_kind: str
    metadata: Mapping[str, Any]
    body: str

@dataclass(frozen=True)
class WikiCard:
    card_id: str
    path: Path
    card_type: str
    metadata: Mapping[str, Any]
    body: str

@dataclass(frozen=True)
class Corpus:
    root: Path
    sources: Mapping[str, SourceRecord]
    cards: Mapping[str, WikiCard]
    taxonomy: Mapping[str, frozenset[str]]
    aliases: Mapping[str, tuple[str, ...]]

@dataclass(frozen=True)
class QueryRequest:
    text: str
    filters: Mapping[str, tuple[str, ...]]
    scope: str  # cards | sources | both
    limit: int

@dataclass(frozen=True)
class SearchCandidate:
    record_kind: str  # card | source
    record_id: str
    path: str
    title: str
    record_type: str
    structured_fields: Mapping[str, tuple[str, ...]]
    body: str
    record: WikiCard | SourceRecord

@dataclass(frozen=True)
class SearchHit:
    record_kind: str  # card | source
    record_id: str
    path: str
    title: str
    record_type: str
    score: tuple[int, ...]
    matched_fields: tuple[str, ...]
    excerpt: str

@dataclass(frozen=True)
class FollowedSource:
    source_id: str
    path: str
    title: str
    metadata: Mapping[str, Any]
    body: str

@dataclass(frozen=True)
class AssetAccess:
    source_id: str
    artifact_dir: str | None
    metadata_visible: bool
    code_visible: bool
    reason: str

@dataclass(frozen=True)
class PageResult:
    schema_version: int
    record_kind: str
    record_id: str
    path: str
    title: str
    metadata: Mapping[str, Any]
    body: str
    followed_sources: tuple[FollowedSource, ...]
    asset_access: tuple[AssetAccess, ...]

@dataclass(frozen=True)
class GrepMatch:
    record_kind: str
    record_id: str
    path: str
    line_number: int
    excerpt: str
```

Stable function signatures:

```text
build_card_candidate(card: WikiCard, catalog_record: Mapping[str, Any], corpus: Corpus) -> SearchCandidate
build_source_candidate(source: SourceRecord, corpus: Corpus) -> SearchCandidate
collect_unlimited_candidates(corpus: Corpus, request: QueryRequest) -> tuple[SearchCandidate, ...]
search_records(corpus: Corpus, request: QueryRequest) -> tuple[SearchHit, ...]
query_payload(corpus: Corpus, request: QueryRequest) -> Mapping[str, Any]
retrieve_page(corpus: Corpus, record_id: str, *, follow_sources: bool, access: str) -> PageResult
grep_corpus(corpus: Corpus, pattern: str, *, scope: str, max_matches: int, context_chars: int) -> tuple[GrepMatch, ...]
```

`access` is `metadata|approved-assets`; `approved-assets` exposes code only when provenance has approved license, allowed `designer` audience, and non-metadata asset mode. `query_payload` has exactly `schema_version`, `query`, `filters`, `scope`, and `results`; each result serializes all `SearchHit` fields except the internal score tuple, which becomes a JSON list. Page JSON serializes every `PageResult` field. Regex JSON has `schema_version`, `pattern`, `scope`, and sorted `matches` from `GrepMatch`.

Run all standalone-core tests with:

```bash
python3 -m unittest discover -s skills/kernelwiki/tests -p 'test_*.py' -v
```

---

### Task 1: Skill Skeleton, Safe YAML, Frontmatter, and CLI Errors

**Files:**
- Create: `skills/kernelwiki/requirements.txt`
- Create: `skills/kernelwiki/scripts/kernelwiki_common.py`
- Create: `skills/kernelwiki/tests/test_common.py`
- Create: `skills/kernelwiki/SKILL.md`
- Create: `skills/kernelwiki/README.md`
- Create: `skills/kernelwiki/index.md`
- Create: `skills/kernelwiki/data/evaluation-holdouts.yaml`
- Create: `skills/kernelwiki/tests/fixtures/holdout/track2-sinkhorn-gold.yaml`
- Create: `skills/kernelwiki/references/evaluation-protocol.md`
- Modify: `skills/README.md`

**Interfaces:**
- Produces: `KernelWikiError`, `load_yaml_document`, `parse_markdown`, `sha256_file`, `sha256_bytes`, `canonical_json_bytes`, `require_within`, `write_text_atomic`, `run_cli`.
- Consumed by: every later script and test.

- [ ] **Step 1: Add the pinned parser dependency and failing common-library tests**

Create `requirements.txt` with exactly:

```text
PyYAML==6.0.2
```

Create `test_common.py` with tests that import the future module and exercise safe parsing and stable errors:

```python
from pathlib import Path
import sys
import tempfile
import unittest

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

from kernelwiki_common import (  # noqa: E402
    KernelWikiError,
    load_yaml_document,
    parse_markdown,
    require_within,
)


class CommonTests(unittest.TestCase):
    def test_parse_markdown_returns_frontmatter_and_body(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "page.md"
            path.write_text("---\nid: card-one\ntags: [fusion]\n---\n# Body\n", encoding="utf-8")
            metadata, body = parse_markdown(path)
            self.assertEqual("card-one", metadata["id"])
            self.assertEqual(["fusion"], metadata["tags"])
            self.assertEqual("# Body\n", body)

    def test_yaml_unsafe_python_tag_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.yaml"
            path.write_text("!!python/object/apply:os.system ['false']\n", encoding="utf-8")
            with self.assertRaisesRegex(KernelWikiError, "yaml-invalid"):
                load_yaml_document(path)

    def test_require_within_rejects_parent_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(KernelWikiError, "path-escape"):
                require_within(root, root / ".." / "outside")
```

- [ ] **Step 2: Install the pinned dependency and verify the tests fail for the missing module**

Run:

```bash
python3 -m pip install -r skills/kernelwiki/requirements.txt
python3 -m unittest skills/kernelwiki/tests/test_common.py -v
```

Expected: dependency installation succeeds; tests fail with `ModuleNotFoundError: No module named 'kernelwiki_common'`.

- [ ] **Step 3: Implement the minimal shared library**

Create `kernelwiki_common.py` with these exact public behaviors:

```python
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import yaml


class KernelWikiError(Exception):
    def __init__(self, code: str, message: str, path: Path | None = None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.path = path


def load_yaml_document(path: Path) -> Any:
    try:
        return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise KernelWikiError("yaml-invalid", str(error), Path(path)) from error


def parse_markdown(path: Path) -> tuple[dict[str, Any], str]:
    text = Path(path).read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        raise KernelWikiError("frontmatter-missing", "Markdown must begin with ---", Path(path))
    marker = text.find("\n---\n", 4)
    if marker < 0:
        raise KernelWikiError("frontmatter-unclosed", "Markdown frontmatter is not closed", Path(path))
    metadata_text = text[4:marker]
    try:
        metadata = yaml.safe_load(metadata_text)
    except yaml.YAMLError as error:
        raise KernelWikiError("frontmatter-invalid", str(error), Path(path)) from error
    if not isinstance(metadata, dict):
        raise KernelWikiError("frontmatter-object-required", "frontmatter must be an object", Path(path))
    return metadata, text[marker + 5 :]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def require_within(root: Path, path: Path) -> Path:
    root = Path(root).resolve()
    candidate = Path(path).resolve()
    if candidate != root and root not in candidate.parents:
        raise KernelWikiError("path-escape", f"{candidate} escapes {root}", candidate)
    return candidate


def write_text_atomic(path: Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def run_cli(main: Callable[[Sequence[str]], int], argv: Sequence[str] | None = None) -> int:
    try:
        return main(list(sys.argv[1:] if argv is None else argv))
    except KernelWikiError as error:
        location = f" ({error.path})" if error.path else ""
        print(f"error[{error.code}]: {error.message}{location}", file=sys.stderr)
        return 2
```

Add tests for malformed/unclosed frontmatter, missing files, canonical JSON ordering, SHA-256 stability, and atomic replacement that leaves no temporary file behind.

- [ ] **Step 4: Run the common-library tests**

Run:

```bash
python3 -m unittest skills/kernelwiki/tests/test_common.py -v
```

Expected: all common tests pass without network or accelerator access.

- [ ] **Step 5: Seal evaluation boundaries before taxonomy or ranking work**

Create `evaluation-holdouts.yaml` with this reviewed boundary:

```yaml
schema_version: 1
sealed_at: 2026-08-21T00:00:00Z
repository_holdout:
  repository_id: triton-ascend-kernels
  use: final-evaluation-only
track2:
  development_contexts: [sparse_attn, index_topk]
  holdout_contexts: [sinkhorn_normalize]
  gold_fixture_sha256: a7ea16d878f10060aff1cf7f5a2b4d99db7f18b297ef523d9d5fda327f4b2c13
  corpus_policy: never-author-operator-cards
  ranking_policy: do-not-read-holdout-until-final-evaluation
```

Create `track2-sinkhorn-gold.yaml` before taxonomy/alias/ranking implementation with exactly:

```yaml
schema_version: 1
case_id: track2-sinkhorn-normalize-holdout
query_text: "ascend repeated sinkhorn row column normalization reduction"
designer_context:
  schema_version: 1
  role: designer
  contract_version: null
  target_id: ascend910b
  implementation_profile_id: ascendc
  implementation_profile_status: missing
  runtime_fingerprint: null
  languages: [ascendc]
  operator_tags: [sinkhorn, normalization]
  kernel_types: [normalization, reduction]
  dtypes: [fp32]
  semantic_features: [row-normalization, column-normalization, repeated-iteration]
  shape_signature: {batch: 1, groups: 1024, rows: 4, columns: 4, repeat: 10}
  current_bottlenecks: []
  project_root: null
  artifacts: {}
  guidance_bindings: {}
  loop_contract_identity: null
coder_context:
  schema_version: 1
  role: coder
  contract_version: 3
  target_id: ascend910b
  implementation_profile_id: ascendc
  implementation_profile_status: missing
  runtime_fingerprint: null
  languages: [ascendc]
  operator_tags: [sinkhorn, normalization]
  kernel_types: [normalization, reduction]
  dtypes: [fp32]
  semantic_features: [row-normalization, column-normalization, repeated-iteration]
  shape_signature: {batch: 1, groups: 1024, rows: 4, columns: 4, repeat: 10}
  current_bottlenecks: []
  project_root: null
  artifacts: {}
  guidance_bindings: {}
  loop_contract_identity: null
gold:
  relevant_card_ids: [technique-kernel-fusion, technique-tiling-and-work-partitioning, technique-topk-selection-and-reduction, measurement-cann-device-attribution]
  counterexample_card_ids: [pattern-device-win-wall-loss]
  capability_gap_card_ids: [pattern-ascend-capability-gap]
  forbidden_outcomes: [operator-named-card, coder-recipe, cross-target-fallback, unknown-promotion]
metrics:
  top5_relevant_denominator: 4
  counterexample_denominator: 1
  capability_gap_denominator: 1
  unsafe_coder_expected: 0
  cross_target_leak_expected: 0
  unknown_promotion_expected: 0
```

Its SHA-256, including the final newline, is `a7ea16d878f10060aff1cf7f5a2b4d99db7f18b297ef523d9d5fda327f4b2c13`; record that value in `evaluation-holdouts.yaml`. `evaluation-protocol.md` defines top-5 recall as relevant IDs returned in the first five admitted Designer Cards divided by `4`, counterexample/gap recall against their explicit one-ID denominators, and safety counts against the expected zeros. Later tasks may run final evaluation but may not edit the fixture to improve recall.

- [ ] **Step 6: Add the skill-facing documentation**

Create `SKILL.md` with frontmatter:

```yaml
---
name: kernelwiki
description: Curate and query an offline, provenance-pinned, Ascend-first kernel engineering wiki without changing campaign authority.
---
```

Document these commands as the only core entry points: `validate.py`, `capture_source.py`, `generate_indices.py`, `query.py`, `get_page.py`, and `grep_wiki.py`. State explicitly that query commands are offline and that the skill never edits `kernel-opt-loop` or active campaigns.

Create `README.md` with installation and test commands. Create `index.md` with links to `queries/by-problem.md`, `queries/by-technique.md`, `queries/by-target.md`, and each Wiki directory. Add a `KernelWiki` section to `skills/README.md` linking to the new skill.

- [ ] **Step 7: Commit the skeleton and sealed holdouts**

```bash
git add skills/kernelwiki skills/README.md
git diff --cached --name-only
git commit -m "feat(kernelwiki): add standalone skill skeleton"
```

---

### Task 2: Taxonomy, Source/Card Models, and Whole-Corpus Validation

**Files:**
- Create: `skills/kernelwiki/data/schemas.yaml`
- Create: `skills/kernelwiki/data/taxonomy.yaml`
- Create: `skills/kernelwiki/data/aliases.yaml`
- Create: `skills/kernelwiki/data/version-claims.yaml`
- Create: `skills/kernelwiki/scripts/corpus.py`
- Create: `skills/kernelwiki/scripts/validate.py`
- Create: `skills/kernelwiki/tests/fixture_factory.py`
- Create: `skills/kernelwiki/tests/test_corpus.py`
- Create: `skills/kernelwiki/tests/fixtures/valid-corpus/`
- Create: `skills/kernelwiki/tests/fixtures/invalid-corpus/`
- Create: `skills/kernelwiki/references/schema.md`
- Create: `skills/kernelwiki/references/examples.md`

**Interfaces:**
- Consumes: `parse_markdown`, `load_yaml_document`, `KernelWikiError`.
- Produces: `SourceRecord`, `WikiCard`, `Corpus`, `load_corpus(root: Path) -> Corpus`, `validate_corpus(corpus: Corpus) -> None`.

- [ ] **Step 1: Write failing model and corpus-integrity tests**

Create tests for one valid Source and Card plus each fail-closed rule:

```python
class CorpusTests(unittest.TestCase):
    def test_valid_card_resolves_source_and_related_links(self):
        root = make_valid_corpus()
        corpus = load_corpus(root)
        validate_corpus(corpus)
        self.assertIn("technique-kernel-fusion", corpus.cards)

    def test_duplicate_ids_fail(self):
        root = make_valid_corpus(duplicate_card_id=True)
        with self.assertRaisesRegex(KernelWikiError, "id-duplicate"):
            load_corpus(root)

    def test_unknown_taxonomy_value_fails(self):
        root = make_valid_corpus(extra_tag="not-in-taxonomy")
        with self.assertRaisesRegex(KernelWikiError, "taxonomy-unknown"):
            validate_corpus(load_corpus(root))

    def test_example_requires_existing_source_and_scope(self):
        root = make_valid_corpus(example_source="missing-source")
        with self.assertRaisesRegex(KernelWikiError, "example-source-missing"):
            validate_corpus(load_corpus(root))
```

Add separate test methods `test_unresolved_related_fails`, `test_unresolved_prerequisite_fails`, `test_invalid_card_type_fails`, `test_missing_required_heading_fails`, `test_invalid_evidence_level_fails`, `test_invalid_reproduction_level_fails`, `test_invalid_target_disposition_fails`, `test_local_example_without_transfer_boundary_fails`, and `test_track2_operator_named_card_fails` using Card ID `sparse-attn-operator`.

- [ ] **Step 2: Run the corpus tests and verify they fail**

Run:

```bash
python3 -m unittest skills/kernelwiki/tests/test_corpus.py -v
```

Expected: import failure for `corpus` or missing `load_corpus`.

- [ ] **Step 3: Define the closed machine-readable vocabulary**

Create `schemas.yaml` with schema versions `source: 1`, `card: 1`, `catalog: 1`, and `query_result: 1`.

Create `taxonomy.yaml` containing at least:

```yaml
card_types: [hardware, technique, pattern, language, runtime, measurement, kernel, migration]
source_kinds: [github-pr, github-commit, official-doc, manual-doc, local-campaign]
audiences: [designer, coder]
authorities: [advisory]
target_matches: [exact, family, backend, analogy-only, unknown]
evidence_levels: [local-verifier, official-doc-and-upstream-code, source-reported, inferred, experimental]
reproduction_levels: [concept, pseudocode, snippet, runnable, benchmarked]
comparability_states: [source-reported, historical-local, project-reproduced, comparable-to-current-baseline]
license_states: [approved, metadata-only, incompatible, unknown]
example_roles: [positive, counterexample, capability-gap]
example_subtypes: [performance, screening, device-wall-mismatch, design-pitfall, implementation-pitfall, profile, source-example]
profile_authorities: [current-vnext, historical-noncanonical, source-only, not-applicable]
terminal_classifications: [accepted, no-improvement, screened-out, aborted, source-reported, not-applicable]
comparability_classes: [current-contract, historical-local, source-reported, not-comparable]
measurement_metrics: [kernel_count_per_call, wall_improvement_pct, device_improvement_pct, latency_ms, wall_time_ms, device_time_ms, throughput_items_per_second, correctness_pass]
measurement_statistics: [exact, median, mean, p50, p95, min, max, source-reported]
measurement_units: [count, percent, milliseconds, items-per-second, boolean, ratio]
languages: [ascendc, triton, cpp, python]
dtypes: [fp16, bf16, fp32, int32, int64]
kernel_types: [attention, sparse-attention, topk, selection, reduction, normalization, moe, data-preparation]
techniques: [kernel-fusion, launch-collapse, tiling, work-partitioning, double-buffering, software-pipelining, layout-transformation, output-reuse]
symptoms: [launch-bound, materialization-overhead, memory-bound, device-win-wall-loss, capability-gap]
hardware_features: [memory-hierarchy, execution-pipeline, vector, cube, dma]
tags: [ascend, ascendc, triton-ascend, mskl, memory-hierarchy, execution-pipeline, language-model, backend, kernel-authoring, launcher, integration, kernel-fusion, launch-collapse, materialization, tiling, work-partitioning, double-buffering, software-pipelining, layout-transformation, topk, selection, reduction, launch-bound, materialization-overhead, cann, profiling, device-attribution, device-win-wall-loss, host-bound, synchronization, capability-gap]
```

Create `aliases.yaml` as a mapping from canonical terms to sorted unique aliases. Start with `kernel-fusion`, `double-buffering`, `launch-bound`, `materialization-overhead`, `topk`, `attention`, `reduction`, `ascendc`, and `triton-ascend`.

Create `version-claims.yaml` with `schema_version: 1` and a sorted `claims` list. Each claim has exactly `id`, `card_ids`, `status: current|stale|unknown`, `supported_versions`, `last_verified_at` as checked-in `YYYY-MM-DD` or `null`, and `source_ids`. Validate both Card and Source back-references; no generator consults current time.

Create the empty version-claim registry with an explicit schema:

```yaml
schema_version: 1
claims: []
```

Tool/runtime identities remain Source metadata until a concrete version-sensitive Card demonstrates the need for a separate registry.

- [ ] **Step 4: Implement Source/Card loading and validation**

In `corpus.py`, define frozen dataclasses matching the stable interfaces and implement:

```python
def load_corpus(root: Path) -> Corpus:
    root = Path(root).resolve()
    taxonomy = _load_taxonomy(root / "data" / "taxonomy.yaml")
    aliases = _load_aliases(root / "data" / "aliases.yaml")
    sources = _load_sources(root / "sources")
    cards = _load_cards(root / "wiki")
    _reject_cross_kind_duplicate_ids(sources, cards)
    return Corpus(root=root, sources=sources, cards=cards, taxonomy=taxonomy, aliases=aliases)


def validate_corpus(corpus: Corpus) -> None:
    for source in corpus.sources.values():
        _validate_source(source, corpus)
    for card in corpus.cards.values():
        _validate_card(card, corpus)
    _validate_links(corpus)
    _validate_examples(corpus)
    _validate_version_registry(corpus)
```

Require Source frontmatter fields: `schema_version`, `id`, `source_kind`, `title`, `url`, `repository_id`, `captured_at`, `target_disposition`, `languages`, `kernel_types`, `techniques`, `hardware_features`, `tags`, `license_state`, and `artifact_dir` when assets exist. Optional `implementation_profile_ids`, `runtime_fingerprints`, and `audiences` are closed lists used by later exact-profile admission; absence means Designer metadata only, never inferred Coder eligibility. A `source_kind: local-campaign` record additionally requires `profile_authority: current-vnext|historical-noncanonical`, `strict_vnext_validated: true|false`, and sorted `missing_evidence`; historical authority requires `strict_vnext_validated: false` and `audiences: [designer]`. `repository_id` must resolve in `source-repositories.yaml` or equal `local` for local evidence.

Require Card frontmatter fields: `schema_version`, `id`, `title`, `type`, `audiences`, `authority`, `summary`, `targets`, `target_match`, `languages`, `kernel_types`, `techniques`, `hardware_features`, `tags`, `symptoms`, `sources`, `related`, `prerequisites`, `version_sensitive`, `observations`, and `examples`. A `pattern` Card additionally requires `candidate_techniques`, and every ID there resolves to a `technique` Card. Every `version_sensitive` entry resolves to one `version-claims.yaml` ID whose `card_ids` contains the Card ID.

Each `observations` item uses this exact v1 shape:

```yaml
- id: observation-fusion-launch-count
  text: Fusion can reduce separately materialized producer-consumer launches in the cited implementation.
  source_id: source-vllm-ascend-pr-814
  locator: "PR description and changed-file accounting"
  evidence_level: source-reported
  reproduction: concept
  targets: [ascend]
  target_match: backend
  implementation_profile_id: null
  runtime_fingerprint: null
  versions: []
  transfer_boundaries: [requires an independently legal fusion boundary]
```

Every example has exactly `id`, `role`, `subtype`, `source_id`, `locator`, `evidence_level`, `reproduction`, `target_id`, `implementation_profile_id`, `profile_authority`, `runtime_fingerprint`, `operator_family`, `shape`, `dtype`, `terminal_classification`, `comparability`, `measurement_fingerprint`, `baseline_id`, `candidate_id`, `observed`, `transfer_boundary`, and `reconsider_when`, plus the capability-gap-only fields below. `shape` is a sorted mapping whose values are positive integers or symbolic dimension strings matching `[A-Z][A-Z0-9_]*`; `dtype` is one taxonomy language dtype token; nullable strings are explicit `null`, never omitted. `observed` is a list sorted by metric, and each item has `metric`, numeric-or-boolean `value`, `statistic`, and `unit` from the closed taxonomy. `positive|counterexample` require at least one observation; local measured examples require nonnull measurement/baseline/candidate IDs. `capability-gap` requires `observed: []`, `capability_id`, `capability_status: unknown|unsupported`, and nonempty `required_probe_or_authority`.

Create these complete literal fixtures in `tests/fixtures/valid-corpus/`:

```yaml
positive:
  id: example-grouped-topk-fusion
  role: positive
  subtype: performance
  source_id: source-local-ascend-groupedtopk-round-001
  locator: rounds/report_001.md#kernelwiki-fact-pack
  evidence_level: local-verifier
  reproduction: benchmarked
  target_id: ascend910b
  implementation_profile_id: triton_ascend
  profile_authority: historical-noncanonical
  runtime_fingerprint: cann-8.0-ascend910b
  operator_family: grouped-topk
  shape: {experts: 256, tokens: 1024, topk: 8}
  dtype: fp16
  terminal_classification: accepted
  comparability: historical-local
  measurement_fingerprint: measurement-groupedtopk-round-001
  baseline_id: baseline_adapter.py
  candidate_id: candidate_001.py
  observed:
    - {metric: kernel_count_per_call, value: 1, statistic: exact, unit: count}
    - {metric: wall_improvement_pct, value: 54.88, statistic: median, unit: percent}
  transfer_boundary: Same target/profile/runtime and grouped-routing shape regime only.
  reconsider_when: [shape regime changes, runtime fingerprint changes]
counterexample:
  id: example-flexattention-device-wall-loss
  role: counterexample
  subtype: device-wall-mismatch
  source_id: source-local-ascend-flexattention-round-003
  locator: rounds/report_003.md#kernelwiki-fact-pack
  evidence_level: local-verifier
  reproduction: benchmarked
  target_id: ascend910b
  implementation_profile_id: triton_ascend
  profile_authority: historical-noncanonical
  runtime_fingerprint: cann-8.0-ascend910b
  operator_family: attention
  shape: {batch: 1, heads: 32, sequence: 4096}
  dtype: fp16
  terminal_classification: no-improvement
  comparability: historical-local
  measurement_fingerprint: measurement-flexattention-round-003
  baseline_id: baseline_adapter.py
  candidate_id: candidate_003.py
  observed:
    - {metric: device_improvement_pct, value: 8.0, statistic: median, unit: percent}
    - {metric: wall_improvement_pct, value: -3.0, statistic: median, unit: percent}
  transfer_boundary: Device-time movement is not wall-time authority outside this exact measurement contract.
  reconsider_when: [host overhead is removed, measurement policy changes]
capability_gap:
  id: example-ascendc-profile-missing
  role: capability-gap
  subtype: profile
  source_id: source-ascendc-programming-model-cann-900beta1
  locator: What is Ascend C
  evidence_level: source-reported
  reproduction: concept
  target_id: ascend910b
  implementation_profile_id: ascendc
  profile_authority: source-only
  runtime_fingerprint: null
  operator_family: normalization
  shape: {columns: 4, repeat: 10, rows: 4}
  dtype: fp32
  terminal_classification: not-applicable
  comparability: not-comparable
  measurement_fingerprint: null
  baseline_id: null
  candidate_id: null
  observed: []
  transfer_boundary: Documentation does not prove the missing local build/runner/profile capability.
  reconsider_when: [a reviewed canonical AscendC profile and qualification probes exist]
  capability_id: ascendc.local-runner
  capability_status: unknown
  required_probe_or_authority: Reviewed canonical AscendC profile plus local qualification evidence.
```

Write one failing test for every required field, enum, nullability rule, measurement item field, role conditional, sorted-observation rule, and local-measurement identity. Add `test_literal_example_fixtures_use_closed_evidence_and_reproduction_values`, loading all three fixtures and asserting membership in `taxonomy["evidence_levels"]` and `taxonomy["reproduction_levels"]`.

Reject IDs outside `[a-z0-9][a-z0-9._-]*`. Reject absolute paths and root escapes. Require nonempty Markdown bodies and the Card headings documented in the spec for technique/pattern and kernel pages.

- [ ] **Step 5: Add the validator CLI and fixtures**

Implement `validate.py`:

```python
def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    corpus = load_corpus(args.root)
    validate_corpus(corpus)
    print(json.dumps({"schema_version": 1, "valid": True, "sources": len(corpus.sources), "cards": len(corpus.cards)}, sort_keys=True))
    return 0
```

Exit through `run_cli`. Build `fixture_factory.py` so tests create isolated complete corpus trees instead of mutating the real corpus. Put one readable valid fixture and focused invalid fixtures under `tests/fixtures/` for contract-review visibility.

- [ ] **Step 6: Run corpus validation tests**

```bash
python3 -m unittest skills/kernelwiki/tests/test_corpus.py -v
python3 skills/kernelwiki/scripts/validate.py --root skills/kernelwiki/tests/fixtures/valid-corpus
```

Expected: tests pass and the CLI prints `{"cards": 1, "schema_version": 1, "sources": 1, "valid": true}` with sorted keys.

- [ ] **Step 7: Document the exact schemas and commit**

In `references/schema.md`, document every required field, enum, path rule, and example scope rule. In `references/examples.md`, include one complete Source and one complete generic Card matching the test fixture.

```bash
git add skills/kernelwiki/data skills/kernelwiki/scripts/corpus.py skills/kernelwiki/scripts/validate.py skills/kernelwiki/tests skills/kernelwiki/references/schema.md skills/kernelwiki/references/examples.md
git commit -m "feat(kernelwiki): validate sources and generic cards"
```

---

### Task 3: Provenance, License Gates, Immutability, and Size Budgets

**Files:**
- Create: `skills/kernelwiki/data/size-budget.yaml`
- Create: `skills/kernelwiki/scripts/provenance.py`
- Create: `skills/kernelwiki/scripts/validate_provenance.py`
- Create: `skills/kernelwiki/tests/test_provenance.py`
- Create: `skills/kernelwiki/references/source-policy.md`
- Modify: `skills/kernelwiki/scripts/validate.py`

**Interfaces:**
- Produces: `ProvenanceBundle`, `load_provenance(path)`, `validate_provenance(bundle, skill_root)`, `validate_size_budget(skill_root)`.
- Extends: `validate.py` to validate all referenced bundles and the checked-in size budget.

- [ ] **Step 1: Write failing provenance and size-budget tests**

Cover valid verbatim/extracted/derived assets and these failures:

```python
def test_hash_mismatch_fails(self):
    bundle_path, skill_root = make_bundle(mode="verbatim", declared_sha="0" * 64)
    with self.assertRaisesRegex(KernelWikiError, "asset-hash-mismatch"):
        validate_provenance(load_provenance(bundle_path), skill_root)


def test_unknown_license_denies_code_but_keeps_metadata(self):
    bundle_path, skill_root = make_bundle(license_state="unknown", coder_access="exact-profile")
    with self.assertRaisesRegex(KernelWikiError, "license-code-exposure"):
        validate_provenance(load_provenance(bundle_path), skill_root)
```

Add separate test methods `test_verbatim_without_upstream_sha_fails`, `test_extracted_without_locator_fails`, `test_derived_without_source_ids_fails`, `test_provenance_path_escape_fails`, `test_declared_file_missing_fails`, `test_undeclared_file_fails`, `test_bundle_budget_overflow_fails`, and `test_repository_budget_overflow_fails`.

- [ ] **Step 2: Run the provenance tests and verify failure**

```bash
python3 -m unittest skills/kernelwiki/tests/test_provenance.py -v
```

Expected: import failure for `provenance`.

- [ ] **Step 3: Implement provenance loading and validation**

Define:

```python
@dataclass(frozen=True)
class ProvenanceFile:
    local_path: str
    upstream_path: str | None
    heading_path: str | None
    role: str
    mode: str
    sha256: str

@dataclass(frozen=True)
class ProvenanceBundle:
    schema_version: int
    path: Path
    origin_url: str
    upstream_repo: str | None
    upstream_sha: str | None
    license_state: str
    retrieved_at: str
    asset_mode: str
    allowed_audiences: tuple[str, ...]
    coder_access: str
    source_ids: tuple[str, ...]
    files: tuple[ProvenanceFile, ...]
```

`validate_provenance` must enumerate every regular file below the bundle directory except `PROVENANCE.yaml`, compare the exact set with `files`, verify SHA-256, enforce mode-specific fields, and deny any code exposure when `license_state != "approved"`.

- [ ] **Step 4: Add deterministic size-budget enforcement**

Create:

```yaml
schema_version: 1
repository_max_bytes: 52428800
bundle_max_bytes: 5242880
file_max_bytes: 1048576
```

Implement `validate_size_budget` using file byte lengths only. Ignore `.gitkeep`; count all retained artifact files and provenance manifests. Raise `size-budget-file`, `size-budget-bundle`, or `size-budget-repository` with exact measured and allowed values.

- [ ] **Step 5: Wire provenance into the whole-corpus validator**

`validate.py` must resolve every non-null `artifact_dir` from a Source, require `PROVENANCE.yaml`, validate it, ensure the Source ID appears in `source_ids`, then run the size-budget check. It must not return a partially valid result after any failure.

- [ ] **Step 6: Run tests and commit**

```bash
python3 -m unittest skills/kernelwiki/tests/test_provenance.py skills/kernelwiki/tests/test_corpus.py -v
git add skills/kernelwiki/data/size-budget.yaml skills/kernelwiki/scripts/provenance.py skills/kernelwiki/scripts/validate_provenance.py skills/kernelwiki/scripts/validate.py skills/kernelwiki/tests/test_provenance.py skills/kernelwiki/references/source-policy.md
git commit -m "feat(kernelwiki): enforce source provenance and size budgets"
```

---

### Task 4: Ascend Repository Registry and Reviewed Candidate Ledgers

**Files:**
- Create: `skills/kernelwiki/data/source-repositories.yaml`
- Create: `skills/kernelwiki/candidates/repos/triton-ascend.yaml`
- Create: `skills/kernelwiki/candidates/repos/vllm-ascend.yaml`
- Create: `skills/kernelwiki/candidates/repos/cann-samples.yaml`
- Create: `skills/kernelwiki/candidates/repos/triton-ascend-kernels.yaml`
- Create: `skills/kernelwiki/candidates/repos/mskl.yaml`
- Create: `skills/kernelwiki/scripts/source_capture.py`
- Create: `skills/kernelwiki/tests/test_source_capture.py`
- Create: `skills/kernelwiki/references/inclusion-policy.md`

**Interfaces:**
- Produces: `RepositorySpec`, `CandidateLedger`, `GitHubClient`, `discover_candidates`, `merge_discovery`, `render_candidate_ledger`.
- Does not write Sources yet; Task 5 uses the same `GitHubClient` for immutable capture.

- [ ] **Step 1: Write failing registry/discovery tests with a fake GitHub client**

Test that discovery is explicit and review decisions are preserved:

```python
class FakeGitHubClient:
    def __init__(self, responses):
        self.responses = responses
        self.urls = []

    def get_json(self, url):
        self.urls.append(url)
        return self.responses[url]


def test_merge_preserves_reviewed_decision(self):
    existing = CandidateLedger(prs=(Candidate(number=814, decision="include", reason="device kernel"),))
    discovered = [Candidate(number=814, title="changed title"), Candidate(number=900, title="new")]
    merged = merge_discovery(existing, discovered, searched_at="2026-08-21T00:00:00Z")
    self.assertEqual("include", merged.by_number[814].decision)
    self.assertEqual("defer", merged.by_number[900].decision)
    self.assertEqual("unreviewed-discovery", merged.by_number[900].reason)
```

Add methods `test_discovery_sort_is_stable`, `test_path_glob_filters_candidates`, `test_skip_glob_filters_candidates`, `test_github_pagination_is_complete`, `test_rate_limit_error_is_stable`, `test_only_explicit_discovery_cli_uses_http`, and `test_manual_repositories_are_not_sent_to_github`; the last method covers `cann-samples`, `triton-ascend-kernels`, `mskl`, and `huawei-ascend-docs` with error `discovery-manual-lane`.

- [ ] **Step 2: Run discovery tests and verify failure**

```bash
python3 -m unittest skills/kernelwiki/tests/test_source_capture.py -v
```

Expected: missing `source_capture` module.

- [ ] **Step 3: Create the repository registry and initial ledgers**

Register these exact lanes:

```yaml
schema_version: 1
repositories:
  - id: triton-ascend
    host: github
    repo: Ascend/triton-ascend
    lane: ascend-native
    languages: [triton, python, cpp]
    target_families: [ascend]
  - id: vllm-ascend
    host: github
    repo: vllm-project/vllm-ascend
    lane: ascend-native
    languages: [ascendc, cpp, python]
    target_families: [ascend]
  - id: cann-samples
    host: manual
    repo: Ascend/cann-samples
    lane: ascend-native-manual
    languages: [ascendc, cpp]
    target_families: [ascend]
  - id: huawei-ascend-docs
    host: manual
    repo: hiascend.com/CANNCommunityEdition
    lane: ascend-native-manual
    languages: [ascendc, cpp]
    target_families: [ascend]
  - id: triton-ascend-kernels
    host: manual
    repo: Ascend/triton-ascend-kernels
    lane: reviewed-holdout
    languages: [triton, python]
    target_families: [ascend]
  - id: mskl
    host: manual
    repo: Ascend/mskl
    lane: ascend-native-manual
    languages: [python, cpp]
    target_families: [ascend]
```

Add the path/skip globs and search terms from the spec. Each ledger starts with `schema_version: 1`, repository identity, an explicit `searched_at`, counts, and sorted candidates. Seed `vllm-ascend.yaml` with PR `814` as `defer` until capture validation confirms changed-file accounting and license state.

- [ ] **Step 4: Implement explicit GitHub discovery**

`GitHubClient` uses `urllib.request`, sends `Accept: application/vnd.github+json`, and uses `GITHUB_TOKEN` only when present. `discover_candidates` searches the configured repository/terms, then fetches changed files for each PR before path classification. It returns data; it never writes directly.

`merge_discovery` preserves all existing `include|defer|exclude` decisions and reasons. New candidates enter as `defer` with reason `unreviewed-discovery`. Removed/upstream-hidden candidates remain in the ledger with `discovery_state: not-returned`.

- [ ] **Step 5: Keep discovery output read-only**

Add a pure `render_discovery(candidates) -> str` function that prints canonical JSON for curator review. Discovery never edits a reviewed ledger. After reviewing the output, the Curator manually changes the matching ledger and supplies `decision` and `reason`; contract tests reject any library function that opens a path under `candidates/repos/` for writing.

Manual lanes raise `KernelWikiError("adapter-manual", "repository requires reviewed manual candidates")` instead of attempting HTML scraping.

- [ ] **Step 6: Document inclusion policy, run tests, and commit**

Document exact include/defer/exclude criteria and the wrapper-only, config-only, benchmark-only, test-only, host-framework-only, and missing-provenance cases.

```bash
python3 -m unittest skills/kernelwiki/tests/test_source_capture.py -v
git add skills/kernelwiki/data/source-repositories.yaml skills/kernelwiki/candidates/repos skills/kernelwiki/scripts/source_capture.py skills/kernelwiki/tests/test_source_capture.py skills/kernelwiki/references/inclusion-policy.md
git commit -m "feat(kernelwiki): add Ascend source discovery ledgers"
```

---

### Task 5: Immutable GitHub and Manual Source Capture

**Files:**
- Create: `skills/kernelwiki/scripts/capture_source.py`
- Modify: `skills/kernelwiki/scripts/source_capture.py`
- Modify: `skills/kernelwiki/tests/test_source_capture.py`
- Modify: `skills/kernelwiki/scripts/validate.py`

**Interfaces:**
- Produces: `GitHubPRCaptureRequest`, `GitHubCommitCaptureRequest`, `ManualCaptureManifest`, `ManualCaptureRequest`, `CaptureResult`, `capture_github_pr`, `capture_github_commit`, `capture_manual_source`, and lightweight stale-staging cleanup through `recover_capture_transactions`. `CaptureTransaction`/`CaptureRecovery` remain compatibility data types only; v1 does not run a persistent recovery journal.
- Writes: one immutable Source Markdown and, when licensed/selected, one immutable artifact directory with `PROVENANCE.yaml`.

```python
@dataclass(frozen=True)
class SourceCaptureMetadata:
    source_id: str
    title: str
    repository_id: str
    captured_at: str
    target_disposition: str
    languages: tuple[str, ...]
    kernel_types: tuple[str, ...]
    techniques: tuple[str, ...]
    hardware_features: tuple[str, ...]
    tags: tuple[str, ...]
    license_state: str
    audiences: tuple[str, ...]

@dataclass(frozen=True)
class CaptureSelection:
    upstream_path: str
    heading_path: str | None
    role: str
    mode: str

@dataclass(frozen=True)
class GitHubPRCaptureRequest:
    skill_root: Path
    metadata: SourceCaptureMetadata
    repo: str
    number: int
    selections: tuple[CaptureSelection, ...]

@dataclass(frozen=True)
class GitHubCommitCaptureRequest:
    skill_root: Path
    metadata: SourceCaptureMetadata
    repo: str
    sha: str
    selections: tuple[CaptureSelection, ...]

@dataclass(frozen=True)
class ManualFileSelection:
    input_path: Path
    upstream_path: str | None
    heading_path: str | None
    role: str
    mode: str

@dataclass(frozen=True)
class ManualCaptureManifest:
    schema_version: int
    metadata: SourceCaptureMetadata
    source_kind: str
    url: str
    document_revision: str
    files: tuple[ManualFileSelection, ...]

@dataclass(frozen=True)
class ManualCaptureRequest:
    skill_root: Path
    manifest_path: Path
```

The manual YAML has exactly the `ManualCaptureManifest` fields above; `metadata` uses the same keys as the dataclass, and every file item uses `input_path`, `upstream_path`, `heading_path`, `role`, and `mode`. Unknown fields fail `capture-manifest-field`; taxonomy and enum values pass through the core Source validator before publication.

- [ ] **Step 1: Add failing capture tests**

Test an API-backed PR capture with description, file list, patch, and one selected code file. Assert:

```python
result = capture_github_pr(request, fake_client)
self.assertEqual("source-vllm-ascend-pr-814", result.source_id)
self.assertTrue(result.source_path.exists())
self.assertTrue((result.artifact_dir / "PROVENANCE.yaml").exists())
validate_provenance(load_provenance(result.artifact_dir / "PROVENANCE.yaml"), request.skill_root)
```

Keep the capture module to about twelve focused tests total: PR/commit/manual happy paths, pinned identity/hash/provenance, metadata-only license behavior, incomplete accounting, destination collision, one normal rollback case, generated-output freshness, stale staging cleanup, and one CLI smoke. Do not add crash-state, inode-swap, symlink-race, forged-dataclass, or transport-permutation matrices. Historical local-bundle capture belongs only to the Phase D plan.

- [ ] **Step 2: Run the focused tests and verify failure**

```bash
python3 -m unittest skills/kernelwiki/tests/test_source_capture.py -v
```

Expected: capture functions are missing.

- [ ] **Step 3: Implement basic immutable capture functions**

Retain the public `CaptureResult` shape and compatibility transaction data types, but use a simple local publication flow:

```text
preflight final paths absent
-> build Source/artifact under a unique .capture-staging directory
-> validate Source/provenance/corpus bytes
-> publish artifact directory, then Source file, without overwriting
-> regenerate generated outputs
-> on a normal Exception, remove only paths created by this invocation
-> remove staging directory
```

`recover_capture_transactions(skill_root)` only removes stale private staging entries that have no published authority. It does not replay process-crash states or maintain `transaction.json`. Generated Sources and artifacts are Git-reviewed local files; protection against hostile concurrent inode swaps, process termination between syscalls, or forged internal transaction objects is outside v1.

If the final Source path or artifact directory already exists, raise `capture-exists`; never offer `--force`. The first reviewed PR capture uses `PR-814.md`; a later upstream revision requires an explicit new Source ID and revision suffix such as `PR-814-r2.md` with its own artifact directory and hashes.

GitHub PR capture records repository, PR number, title, author, state, dates, head SHA, merge SHA, complete changed-file list, exact API URLs, description hash, selected-file hashes, and patch hash. Commit capture pins the full 40-character SHA and selected repository paths. Manual capture requires an explicit immutable URL or document revision, capture date, license state, target disposition, and optional local files. Historical campaign capture is deliberately absent and belongs to the Phase D plan.

- [ ] **Step 4: Implement the CLI with explicit maintenance subcommands**

Support these exact forms:

```bash
python3 skills/kernelwiki/scripts/capture_source.py discover \
  --repository triton-ascend \
  --limit 100

python3 skills/kernelwiki/scripts/capture_source.py github-pr \
  --metadata /tmp/kernelwiki-vllm-pr-814-source.yaml \
  --repo vllm-project/vllm-ascend \
  --pr 814

python3 skills/kernelwiki/scripts/capture_source.py github-commit \
  --metadata /tmp/kernelwiki-triton-ascend-readme-source.yaml \
  --repo Ascend/triton-ascend \
  --sha 865691e2e9b656bc58008170207b4108d92e8dd1

python3 skills/kernelwiki/scripts/capture_source.py manual \
  --metadata skills/kernelwiki/tests/fixtures/manual-source.yaml
```

The CLI is the only KernelWiki path allowed to contact GitHub. `discover` prints candidates only and never edits ledgers; capture subcommands return sorted JSON containing Source/artifact paths and hashes.

- [ ] **Step 5: Wire capture output through full validation**

Before atomic rename, build an isolated temporary skill root containing the new Source plus required data files, call the same Source/provenance validators used by production, and reject invalid output. Add a regression test proving a failed capture leaves no Source, artifact directory, or partial file.

- [ ] **Step 6: Run tests and commit**

```bash
python3 -m unittest skills/kernelwiki/tests/test_source_capture.py skills/kernelwiki/tests/test_provenance.py -v
git add skills/kernelwiki/scripts/capture_source.py skills/kernelwiki/scripts/source_capture.py skills/kernelwiki/scripts/validate.py skills/kernelwiki/tests/test_source_capture.py
git commit -m "feat(kernelwiki): capture immutable source evidence"
```

---

### Task 6: Deterministic Catalog and Nine First-Class Query Views

**Files:**
- Create: `skills/kernelwiki/scripts/catalog.py`
- Create: `skills/kernelwiki/scripts/generate_indices.py`
- Create: `skills/kernelwiki/tests/test_catalog.py`
- Create: `skills/kernelwiki/tests/fixtures/expected-queries/` with one literal golden Markdown file per view
- Generate: `skills/kernelwiki/compiled/catalog.jsonl`
- Generate: `skills/kernelwiki/queries/by-problem.md`
- Generate: `skills/kernelwiki/queries/by-technique.md`
- Generate: `skills/kernelwiki/queries/by-hardware-feature.md`
- Generate: `skills/kernelwiki/queries/by-kernel-type.md`
- Generate: `skills/kernelwiki/queries/by-language.md`
- Generate: `skills/kernelwiki/queries/by-target.md`
- Generate: `skills/kernelwiki/queries/by-source-repo.md`
- Generate: `skills/kernelwiki/queries/by-version.md`
- Generate: `skills/kernelwiki/queries/by-evidence-level.md`
- Modify: `skills/kernelwiki/scripts/validate.py`

**Interfaces:**
- Produces: `card_to_catalog_record`, `build_catalog`, `render_query_views`, `write_generated_outputs`, `assert_generated_outputs_current`.
- Catalog contains Cards only; Source discovery occurs through citations and `by-source-repo.md`.

- [ ] **Step 1: Write failing deterministic-generation tests**

Build the same fixture corpus in two different temporary absolute directories and require byte-identical outputs:

```python
def test_generation_is_independent_of_absolute_root(self):
    left = generate_fixture_outputs(make_valid_corpus())
    right = generate_fixture_outputs(make_valid_corpus())
    self.assertEqual(left, right)


def test_catalog_is_card_only_and_sorted(self):
    outputs = generate_fixture_outputs(make_valid_corpus(two_cards=True))
    records = [json.loads(line) for line in outputs["compiled/catalog.jsonl"].splitlines()]
    self.assertEqual(["pattern-launch-bound", "technique-kernel-fusion"], [record["id"] for record in records])
    self.assertNotIn("source_kind", records[0])
```

Use one table-driven golden-output test for the catalog plus nine views, one drift test, one stale-managed-file cleanup test, one production-output check, and a small number of catalog-record semantic tests. Keep the module to about eight tests; do not add per-view filesystems, rollback injection, or partial-replacement race matrices.

- [ ] **Step 2: Run catalog tests and verify failure**

```bash
python3 -m unittest skills/kernelwiki/tests/test_catalog.py -v
```

Expected: missing `catalog` module.

- [ ] **Step 3: Implement canonical catalog records**

`card_to_catalog_record` returns exactly:

```python
{
    "schema_version": 1,
    "id": card.card_id,
    "path": card.path.relative_to(corpus.root).as_posix(),
    "body_sha256": sha256_bytes(card.body.encode("utf-8")),
    "type": card.card_type,
    "title": card.metadata["title"],
    "summary": card.metadata["summary"],
    "audiences": sorted(card.metadata["audiences"]),
    "targets": sorted(card.metadata["targets"]),
    "target_match": card.metadata["target_match"],
    "languages": sorted(card.metadata["languages"]),
    "kernel_types": sorted(card.metadata["kernel_types"]),
    "techniques": sorted(card.metadata["techniques"]),
    "hardware_features": sorted(card.metadata["hardware_features"]),
    "candidate_techniques": sorted(card.metadata.get("candidate_techniques", [])),
    "tags": sorted(card.metadata["tags"]),
    "symptoms": sorted(card.metadata.get("symptoms", [])),
    "source_ids": sorted(card.metadata["sources"]),
    "source_repositories": sorted(resolve_source_repositories(card, corpus)),
    "version_claims": sorted(resolve_version_claims(card, corpus), key=lambda claim: claim["id"]),
    "source_count": len(card.metadata["sources"]),
    "evidence_levels": sorted(resolve_evidence_levels(card)),
    "reproduction_levels": sorted(resolve_reproduction_levels(card)),
    "positive_example_count": count_examples(card, "positive"),
    "counterexample_count": count_examples(card, "counterexample"),
    "capability_gap_count": count_examples(card, "capability-gap"),
}
```

`resolve_version_claims` returns objects with exactly `id`, `status`, `last_verified_at`, and `supported_versions`; all values come from checked-in `version-claims.yaml`. Serialize one canonical JSON object per line, sorted by Card ID, with one trailing newline.

- [ ] **Step 4: Implement all nine Markdown renderers**

Each renderer groups by canonical taxonomy value, sorts group headings and stable IDs, and emits relative links from `queries/`. Use these exact normalized rows (`join` sorts values and renders an empty list as `none`):

```python
def card_row(record: Mapping[str, Any]) -> str:
    return (
        f"- [{record['id']}](../{record['path']}) — {one_line(record['summary'])} "
        f"— target `{record['target_match']}:{join(record['targets'])}` "
        f"— evidence `{join(record['evidence_levels'])}` "
        f"— reproduction `{join(record['reproduction_levels'])}` "
        f"— sources `{record['source_count']}`"
    )


def source_row(source: SourceRecord, corpus: Corpus) -> str:
    path = source.path.relative_to(corpus.root).as_posix()
    return (
        f"- [{source.source_id}](../{path}) — {one_line(source.metadata['title'])} "
        f"— captured `{source.metadata['captured_at']}` "
        f"— license `{source.metadata['license_state']}` "
        f"— target `{source.metadata['target_disposition']}`"
    )
```

`one_line(text)` is `" ".join(text.split())`; `join(values)` is `", ".join(sorted(values))` or `none`. Every file uses `generated_header(title) = "<!-- GENERATED by scripts/generate_indices.py; DO NOT EDIT. -->\\n# {title}\\n\\n"`. Empty sections contain exactly `- _None._`. Renderer schemas are exact:

```text
by-problem.md
  # By Problem
  ## <symptom>
  ### Patterns
  <card_row for pattern Cards carrying symptom>
  ### Candidate techniques
  - [<technique-id>](../<technique-path>)

by-technique.md
  # By Technique
  ## <technique taxonomy value>
  <card_row for Cards whose techniques contains value, plus the technique Card itself>

by-hardware-feature.md
  # By Hardware Feature
  ## <hardware feature>
  <card_row for Cards carrying feature>

by-kernel-type.md
  # By Kernel Type
  ## <kernel type>
  <card_row for Cards carrying kernel type>

by-language.md
  # By Language
  ## <language>
  <card_row for Cards carrying language>

by-target.md
  # By Target
  ## <target ID>
  ### exact
  ### family
  ### backend
  ### analogy-only
  ### unknown
  <card rows appear only in their target_match subsection>

by-source-repo.md
  # By Source Repository
  ## <repository_id>
  ### Sources
  <source_row for Sources from repository>
  ### Cards
  <card_row for Cards citing at least one listed Source>

by-version.md
  # By Version
  ## current
  ## stale
  ## unknown
  - <claim-id> — versions `<joined supported_versions>` — verified `<last_verified_at or unknown>` — Cards: <sorted Card links>

by-evidence-level.md
  # By Evidence Level
  ## <evidence level>
  <card_row> — examples `positive=<n>, counterexample=<n>, capability-gap=<n>`
```

`source_count = len(source_ids)`. `resolve_evidence_levels` and `resolve_reproduction_levels` return sorted unique values from Card observations/examples only. Source rows are loaded from Source frontmatter and never inserted into the Card-only catalog. Version rows are loaded from `version-claims.yaml` and joined to catalog Card paths by claim `card_ids`.

The first line of every view is:

```markdown
<!-- GENERATED by scripts/generate_indices.py; DO NOT EDIT. -->
```

- [ ] **Step 5: Implement simple reproducible generation and drift validation**

`generate_indices.py --check` compares generated bytes with checked-in files and exits `2` with `error[generated-drift]` on the first path that differs. Without `--check`, render all outputs into a temporary directory first, then replace each managed output with `os.replace` and remove stale managed files. The generated set is reproducible and repairable by rerunning; v1 does not maintain rollback, trash, descriptor-binding, or crash-recovery protocols for derived files.

Wire `validate.py` to call `assert_generated_outputs_current` after corpus/provenance validation.

- [ ] **Step 6: Run tests and commit**

```bash
python3 -m unittest skills/kernelwiki/tests/test_catalog.py skills/kernelwiki/tests/test_corpus.py -v
python3 skills/kernelwiki/scripts/generate_indices.py --root skills/kernelwiki/tests/fixtures/valid-corpus
python3 skills/kernelwiki/scripts/generate_indices.py --root skills/kernelwiki/tests/fixtures/valid-corpus --check
python3 skills/kernelwiki/scripts/generate_indices.py --root skills/kernelwiki
python3 skills/kernelwiki/scripts/generate_indices.py --root skills/kernelwiki --check
git add skills/kernelwiki/scripts/catalog.py skills/kernelwiki/scripts/generate_indices.py skills/kernelwiki/scripts/validate.py skills/kernelwiki/tests/test_catalog.py skills/kernelwiki/tests/fixtures/expected-queries skills/kernelwiki/queries skills/kernelwiki/compiled/catalog.jsonl
git commit -m "feat(kernelwiki): generate deterministic query views"
```

---

### Task 7: Offline Query, Page Retrieval, and Regex Investigation

**Files:**
- Create: `skills/kernelwiki/scripts/search.py`
- Create: `skills/kernelwiki/scripts/query.py`
- Create: `skills/kernelwiki/scripts/get_page.py`
- Create: `skills/kernelwiki/scripts/grep_wiki.py`
- Create: `skills/kernelwiki/tests/test_search.py`
- Modify: `skills/kernelwiki/README.md`

**Interfaces:**
- Produces the Stable Core Interface functions `build_card_candidate`, `build_source_candidate`, `collect_unlimited_candidates`, `parse_query_request`, `search_records`, `query_payload`, `retrieve_page`, and `grep_corpus`.
- Reads: validated local corpus and catalog only.
- Phase A/B supports Designer/general navigation; exact-profile Coder admission is deliberately added only by the Phase C plan.

- [ ] **Step 1: Write failing search and retrieval tests**

Test transparent ranking and stable empty results:

```python
def test_title_and_structured_matches_rank_before_body_only(self):
    corpus = load_search_fixture()
    result = search_records(corpus, QueryRequest("kernel fusion", {}, "both", 10))
    self.assertEqual("technique-kernel-fusion", result[0].record_id)
    self.assertEqual(("title", "tags"), result[0].matched_fields)


def test_no_match_is_schema_valid_empty_result(self):
    payload = query_payload(load_search_fixture(), QueryRequest("impossible-token", {}, "both", 10))
    self.assertEqual({"schema_version": 1, "query": "impossible-token", "filters": {}, "scope": "both", "results": []}, payload)
```

Add separate methods `test_each_filter_field`, `test_alias_normalization`, `test_stable_id_tie_break`, `test_invalid_regex_exits_two`, `test_excerpts_are_bounded`, `test_follow_sources_uses_source_frontmatter`, `test_metadata_only_asset_is_denied`, `test_query_modules_do_not_import_network_clients`, and `test_malformed_limit_exits_two`.

- [ ] **Step 2: Run search tests and verify failure**

```bash
python3 -m unittest skills/kernelwiki/tests/test_search.py -v
```

Expected: missing `search` module.

- [ ] **Step 3: Implement lexical search and filters**

Normalize text with Unicode case folding and `re.findall(r"[\w.+-]+", text.casefold())`. Expand only checked-in aliases. Build Card candidates from the Card-only catalog and Source candidates directly from validated Source frontmatter; do not insert Sources into `catalog.jsonl`. Respect `scope=cards|sources|both`. Score with this documented tuple, descending except final ID:

```python
(
    title_match_count,
    structured_field_match_count,
    alias_match_count,
    min(body_match_count, 8),
)
```

Repository matches are one component of `structured_field_match_count`; no extra component follows bounded body hits. Sort by the negated numeric tuple, then ascending `candidate.path`, then ascending `candidate.record_id` for both Cards and Sources. Filters are exact OR-within-field and AND-across-fields for: `type`, `tag`, `repository`, `language`, `target`, `target-match`, `symptom`, `kernel-type`, `evidence-level`, `reproduction`, `audience`, and `has-code`.

- [ ] **Step 4: Implement the three offline CLIs**

`query.py` supports the spec examples plus `--scope cards|sources|both`, except `--profile-snapshot`, which exits `2` with `error[phase-c-required]` until the Phase C plan lands. Default output is canonical JSON; `--format markdown` emits links and summaries. Source hits are loaded directly from validated Source frontmatter and never appear in the Card-only catalog.

`get_page.py` resolves exact Card/Source ID or relative path, supports `--frontmatter`, `--follow-sources`, `--source-excerpt-lines 40`, and `--include-code`. It maps `--include-code` to `access="approved-assets"`; otherwise `access="metadata"`. In Phase A/B, approved asset access is Designer/general navigation only and still requires approved license/provenance; there is no Coder fallback.

`grep_wiki.py` accepts `--scope wiki|sources|both`, compiles the supplied Python regex, sorts matches by relative path/line, and returns `GrepMatch` JSON with exactly `record_kind`, `record_id`, `path`, `line_number`, and `excerpt`. It never decides applicability.

- [ ] **Step 5: Prove production query is offline**

Patch `socket.socket`, `urllib.request.urlopen`, and any imported GitHub client to raise if called. Run `query_payload`, `retrieve_page`, and `grep_corpus`; all must pass. Add an AST contract test that query/retrieval modules do not import `urllib`, `http`, `requests`, or `source_capture`.

- [ ] **Step 6: Run tests, document commands, and commit**

```bash
python3 -m unittest skills/kernelwiki/tests/test_search.py -v
git add skills/kernelwiki/scripts/search.py skills/kernelwiki/scripts/query.py skills/kernelwiki/scripts/get_page.py skills/kernelwiki/scripts/grep_wiki.py skills/kernelwiki/tests/test_search.py skills/kernelwiki/README.md
git commit -m "feat(kernelwiki): add offline active search"
```

---

### Task 8: Curate the Minimal Reviewed Ascend Corpus

**Files:**
- Create through capture: `skills/kernelwiki/sources/commits/triton-ascend/865691e2e9b656bc58008170207b4108d92e8dd1.md`
- Create through capture: `skills/kernelwiki/sources/commits/vllm-ascend/7702ccd7d8dea6b4dabdacb0118adb522dedbec7.md`
- Create through manual capture: `skills/kernelwiki/sources/docs/source-mskl-user-guide-f9fbf4d2.md`
- Create through manual capture: `skills/kernelwiki/sources/docs/source-ascendc-programming-model-cann-900beta1.md`
- Create through capture: `skills/kernelwiki/sources/prs/vllm-ascend/PR-814.md`
- Create: `skills/kernelwiki/wiki/hardware/ascend-execution-and-memory.md`
- Create: `skills/kernelwiki/wiki/languages/triton-ascend-backend.md`
- Create: `skills/kernelwiki/wiki/languages/mskl-kernel-authoring.md`
- Create: `skills/kernelwiki/wiki/languages/ascendc-programming-model.md`
- Create: `skills/kernelwiki/wiki/runtimes/ascend-kernel-integration.md`
- Create: `skills/kernelwiki/wiki/techniques/kernel-fusion.md`
- Create: `skills/kernelwiki/wiki/techniques/tiling-and-work-partitioning.md`
- Create: `skills/kernelwiki/wiki/techniques/topk-selection-and-reduction.md`
- Create: `skills/kernelwiki/wiki/patterns/launch-bound-materialization.md`
- Create: `skills/kernelwiki/wiki/measurement/cann-device-attribution.md`
- Create: `skills/kernelwiki/wiki/patterns/device-win-wall-loss.md`
- Create: `skills/kernelwiki/wiki/patterns/ascend-capability-gap.md`
- Generate: `skills/kernelwiki/queries/*.md`
- Generate: `skills/kernelwiki/compiled/catalog.jsonl`
- Modify: `skills/kernelwiki/index.md`

**Interfaces:**
- Supplies the first production corpus consumed by all core commands.
- Does not create an AscendC Coder recipe and does not turn Track 2 operators into Cards.

- [ ] **Step 1: Capture two automated pinned sources, one reviewed manual source, and the reviewed PR**

Run the commit capture command from Task 5 for:

```text
Ascend/triton-ascend@865691e2e9b656bc58008170207b4108d92e8dd1:README.md
vllm-project/vllm-ascend@7702ccd7d8dea6b4dabdacb0118adb522dedbec7:README.md
```

Capture the MSKL user guide through the manual manifest lane, pinning `Ascend/mskl@f9fbf4d21273a4f1fa2793b902872fa7da47d86b` and `docs/en/user_guide/mskl_user_guide.md`; do not enable automated MSKL discovery in v1. Capture the official AscendC "What is Ascend C" CANN 9.0.0-beta.1 page at `https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta1/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html#1` through the same manual lane as metadata/extracted prose only unless its license is explicitly approved.

Capture `vllm-project/vllm-ascend#814` only after the changed-file ledger and license state have been reviewed. If PR 814 is not capturable under the policy, keep it `defer`, do not fabricate the Source, and omit only the Cards that would otherwise depend exclusively on it. The corpus remains valid with fewer Cards because Card count is not an acceptance metric.

- [ ] **Step 2: Write the Card frontmatter before prose**

Every Card must use `schema_version: 1`, `audiences: [designer]`, `authority: advisory`, explicit target/evidence scope, and only captured Source IDs. Use these exact IDs and primary tags:

```text
hardware-ascend-execution-and-memory        tags: [ascend, memory-hierarchy, execution-pipeline]
language-triton-ascend-backend              tags: [triton-ascend, language-model, backend]
language-mskl-kernel-authoring               tags: [mskl, kernel-authoring]
language-ascendc-programming-model            tags: [ascend, ascendc, kernel-authoring, memory-hierarchy]
runtime-ascend-kernel-integration            tags: [ascend, launcher, integration]
technique-kernel-fusion                      tags: [kernel-fusion, launch-collapse, materialization]
technique-tiling-and-work-partitioning       tags: [tiling, work-partitioning, reduction]
technique-topk-selection-and-reduction       tags: [topk, selection, reduction]
pattern-launch-bound-materialization         tags: [launch-bound, materialization-overhead]
measurement-cann-device-attribution          tags: [cann, profiling, device-attribution]
pattern-device-win-wall-loss                 tags: [device-win-wall-loss, host-bound, synchronization]
pattern-ascend-capability-gap                 tags: [ascend, capability-gap, device-attribution]
```

Do not add `coder` to any Card. Do not name Cards after `sparse_attn`, `index_topk`, `sinkhorn_normalize`, or any Track 2 path. Add `test_seed_card_frontmatter_is_closed_taxonomy` that loads every Card path listed by this task and asserts all tags, languages, kernel types, techniques, symptoms, and hardware features resolve through `taxonomy.yaml` before prose is accepted.

- [ ] **Step 3: Write evidence-scoped bodies**

For each Card, use the body headings required by `references/schema.md`. Every factual paragraph ends with a Source link or appears under an example carrying its own `source_id`, `evidence_level`, target/version scope, and transfer boundary. `summary` remains navigation text, not a promoted claim.

For `pattern-device-win-wall-loss`, do not invent local measurements in this task. If no reviewed local Source exists yet, publish a source-backed general pattern without local numeric examples; the Phase D plan adds reviewed local cases later.

- [ ] **Step 4: Validate editorial isolation rules**

Run focused checks:

```bash
python3 skills/kernelwiki/scripts/validate.py
python3 skills/kernelwiki/scripts/query.py "ascend kernel fusion" --format markdown
python3 skills/kernelwiki/scripts/query.py "sparse attention topk" --format markdown
```

The second query must return generic Cards. The third may return generic attention/top-k/measurement material but must not return an operator-named Card or an AscendC implementation recipe.

- [ ] **Step 5: Generate and review first-class views**

```bash
python3 skills/kernelwiki/scripts/generate_indices.py
python3 skills/kernelwiki/scripts/generate_indices.py --check
```

Review all nine query files. Confirm `by-source-repo.md` links every captured source through at least one Card, `by-target.md` separates backend and unknown evidence, and `by-evidence-level.md` preserves mixed evidence levels.

- [ ] **Step 6: Commit the reviewed corpus**

```bash
git add skills/kernelwiki/sources skills/kernelwiki/artifacts skills/kernelwiki/wiki skills/kernelwiki/queries skills/kernelwiki/compiled skills/kernelwiki/index.md skills/kernelwiki/candidates/repos
git commit -m "docs(kernelwiki): add reviewed Ascend seed corpus"
```

---

### Task 9: End-to-End Contracts, Drift Gates, and Final Standalone Documentation

**Files:**
- Create: `skills/kernelwiki/tests/test_contracts.py`
- Modify: `skills/kernelwiki/README.md`
- Modify: `skills/kernelwiki/SKILL.md`
- Modify: `skills/kernelwiki/index.md`

**Interfaces:**
- Produces one hardware-free acceptance suite for the standalone core.

- [ ] **Step 1: Write end-to-end contract tests**

Cover these exact assertions:

```python
class StandaloneContractTests(unittest.TestCase):
    def test_real_corpus_validates_and_generated_files_are_current(self):
        corpus = load_corpus(SKILL_ROOT)
        validate_corpus(corpus)
        assert_generated_outputs_current(corpus)

    def test_query_modules_have_no_network_imports(self):
        forbidden = {"urllib", "http", "requests", "socket"}
        for name in ("query.py", "get_page.py", "grep_wiki.py", "search.py"):
            imports = imported_top_level_names(SKILL_ROOT / "scripts" / name)
            self.assertTrue(forbidden.isdisjoint(imports), (name, imports & forbidden))

    def test_track2_names_are_not_card_ids_or_paths(self):
        forbidden = {"sparse_attn", "index_topk", "sinkhorn_normalize"}
        for card in load_corpus(SKILL_ROOT).cards.values():
            self.assertTrue(forbidden.isdisjoint({card.card_id, *card.path.parts}))
```

Also assert unique IDs/links, citation integrity, provenance hashes, no generated drift, no `claims/` directory, and no `KnowledgePacket` text in executable skill files. Scope loop isolation to role-neutral modules: `query.py`, `get_page.py`, `grep_wiki.py`, `search.py`, `catalog.py`, `corpus.py`, `provenance.py`, and core capture/generation code must not import or load `kernel-opt-loop`. Later plans may add only allowlisted read-only bridge modules, which neutral query paths must not import.

- [ ] **Step 2: Run the complete suite**

```bash
python3 -m unittest discover -s skills/kernelwiki/tests -p 'test_*.py' -v
```

Expected: all tests pass without network access or accelerator hardware.

- [ ] **Step 3: Run production smoke commands**

```bash
python3 skills/kernelwiki/scripts/validate.py
python3 skills/kernelwiki/scripts/generate_indices.py --check
python3 skills/kernelwiki/scripts/query.py "ascend launch overhead" --limit 5
python3 skills/kernelwiki/scripts/get_page.py technique-kernel-fusion --follow-sources
python3 skills/kernelwiki/scripts/grep_wiki.py "device.*wall" --scope wiki
```

Expected: each exits `0`; query/retrieval output is deterministic across two consecutive runs.

- [ ] **Step 4: Check performance and size budgets**

Run each query/retrieval command through `python3 -m timeit` or a small standard-library timing test over ten invocations. Assert median elapsed time is below two seconds on the checked-in initial corpus. Run provenance size validation and record measured total bytes in the test output; do not weaken budgets to make the test pass.

- [ ] **Step 5: Finalize docs and commit**

Document maintenance order:

```text
discover candidates -> curator edits reviewed ledger -> capture immutable Source -> author/review generic Card -> validate -> generate views -> review diff -> commit
```

Document that Phase C role-aware query admission and Phase D offline knowledge lift are separate plans and that no Phase E adapter exists.

```bash
git add skills/kernelwiki/tests/test_contracts.py skills/kernelwiki/README.md skills/kernelwiki/SKILL.md skills/kernelwiki/index.md
git commit -m "test(kernelwiki): gate standalone corpus contracts"
```

- [ ] **Step 6: Verify final standalone-core history and cleanliness**

```bash
git status --short
git log --oneline --decorate -9
```

Expected: clean worktree and one focused commit per task. Do not begin the Phase C plan until this complete suite is green and the generated corpus diff has been reviewed.

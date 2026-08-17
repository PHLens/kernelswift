# KernelWiki v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone, Git-versioned KernelWiki skill that compiles
provenance-pinned sources into approved claims, readable wiki pages, deterministic
query indices, and safe per-round KnowledgePackets for optional
`kernel-opt-loop` consumption.

**Architecture:** Implement KernelWiki under `skills/kernelwiki/` as a
four-stage compiler pipeline: immutable source records feed approved atomic
claims; claims feed explanatory wiki pages; active claim and page frontmatter
feed generated indices. Pure Python CLIs validate the corpus, search it, build a
bounded packet, and manage proposal-to-promotion transitions. The existing loop
only gains an optional schema-v2 adapter that freezes one packet before Designer,
passes it to roles under explicit authority boundaries, and records a sidecar
Verifier evidence artifact.

**Tech Stack:** Python 3 standard library, PyYAML `>=6.0,<7`, `unittest`, JSON,
YAML, Markdown contracts, Git.

## Global Constraints

- Keep all KernelWiki code, data, source records, claims, pages, indices, tests,
  and documentation below `skills/kernelwiki/`.
- Use ASCII source and documentation content unless a quoted upstream source
  requires otherwise.
- Use `yaml.safe_load` and `yaml.safe_dump`; never execute YAML tags or dynamic
  expressions.
- Keep source evidence immutable and provenance-first. Active claims are the
  smallest approval, retrieval, snapshot, staleness, and supersession unit.
- Do not implement code RAG, raw artifact injection, vector retrieval, automatic
  promotion, remote synchronization, target-profile mutation, or active-project
  migration.
- Preserve `kernel-opt-loop` schema-v1 decisions and current campaigns. Only
  schema-v2 decisions opt into the KernelWiki consultation section and snapshot.
- Target profile facts remain L0 authority. KernelWiki may reference target
  profile constraints but must not add a capability fact through a claim.
- Coder receives only the frozen packet's `hard_constraints` and current-target
  `target_recipes`; raw sources, artifacts, external code, other target
  projections, and newer corpus revisions are prohibited.
- A missing or failed KernelWiki query must generate a schema-valid empty packet
  and keep the local optimization workflow valid.
- Every implementation task ends with its focused `unittest` command and a
  commit that stages only the files named in that task.

---

## Scope Check

KernelWiki v1 contains source/claim validation, generated navigation, bounded
retrieval, promotion governance, and an optional loop adapter. These are not
independent subprojects: each later deliverable consumes the exact IDs, schemas,
and ownership rules created earlier. This is one ordered implementation plan with
separate reviewable commits rather than separate plans.

## File Structure

| Path | Responsibility |
|---|---|
| `skills/kernelwiki/SKILL.md` | Standalone skill contract: Curate, Retrieve, Promote modes; role and ownership boundaries; CLI entry points. |
| `skills/kernelwiki/README.md` | Installation, corpus topology, command examples, and no-code-RAG rules. |
| `skills/kernelwiki/requirements.txt` | Pins the only non-stdlib runtime dependency, PyYAML. |
| `skills/kernelwiki/data/taxonomy.yaml` | Validator-enforced controlled vocabularies for target profiles, source kinds, claim kinds, statuses, regime tags, and page types. |
| `skills/kernelwiki/data/aliases.yaml` | Canonical term to alias mapping used only by exploratory search. |
| `skills/kernelwiki/data/version-claims.yaml` | Version-sensitive claim registry and bidirectional source/claim/page references. |
| `skills/kernelwiki/schemas/*.schema.json` | Human-readable, machine-tested schema contracts for source, claim, page, query, packet, and evidence records. |
| `skills/kernelwiki/scripts/kernelwiki_lib.py` | Shared YAML/JSON loading, SHA-256, root discovery, frontmatter parsing, ID lookup, canonical JSON, and stable errors. |
| `skills/kernelwiki/scripts/validate.py` | Corpus-wide schema, ID, taxonomy, provenance, lifecycle, projection, and version-link validator. |
| `skills/kernelwiki/scripts/generate_indices.py` | Deterministic generator for the `queries/*.md` navigation pages. |
| `skills/kernelwiki/scripts/query.py` | Exploratory natural-language keyword and structured filter search for Curators. |
| `skills/kernelwiki/scripts/get_page.py` | Resolves a page/source/claim ID and optionally expands cited source records. |
| `skills/kernelwiki/scripts/build_packet.py` | Production-only structured query resolver that emits a bounded, deterministic KnowledgePacket. |
| `skills/kernelwiki/scripts/validate_evidence.py` | Validates project-side `KernelEvidenceRecord` JSON without modifying the project. |
| `skills/kernelwiki/scripts/propose_claim.py` | Converts validated evidence into a non-active claim proposal under `claims/proposed/`. |
| `skills/kernelwiki/scripts/promote_claim.py` | Moves an approved proposal into `claims/active/`, records approval metadata, and rejects missing approval references. |
| `skills/kernelwiki/scripts/evaluate_holdout.py` | Aggregates leave-one-operator replay observations into required safety and quality metrics. |
| `skills/kernelwiki/sources/` | Immutable local and external source records. |
| `skills/kernelwiki/claims/` | `proposed/`, `active/`, and `archived/` atomic claim records. |
| `skills/kernelwiki/wiki/` | Reviewed Markdown pages that synthesize claims but never replace claim-level authority. |
| `skills/kernelwiki/queries/` | Generated Markdown indices. |
| `skills/kernelwiki/candidates/sources/` | Per-repository include/defer/exclude ledger for external source candidates. |
| `skills/kernelwiki/artifacts/` | Empty-by-default directory for reviewed, size-capped, provenance-pinned bundles. |
| `skills/kernelwiki/tests/` | Unit tests and isolated corpus fixtures. |
| `skills/kernel-opt-loop/references/kernelwiki-integration.md` | Adapter contract, query/snapshot paths, authority boundaries, empty fallback, and promotion handoff. |
| `skills/kernel-opt-loop/SKILL.md` | Orchestrator additions for optional schema-v2 KernelWiki snapshots and evidence proposals. |
| `skills/kernel-opt-loop/prompts/designer.md` | Snapshot consultation and `KernelWiki Consultation` decision requirements. |
| `skills/kernel-opt-loop/prompts/coder.md` | Frozen packet consumption and claim-use reporting rules. |
| `skills/kernel-opt-loop/prompts/verifier.md` | Sidecar evidence record ownership and no-direct-promotion rule. |
| `skills/kernel-opt-loop/references/decision-template.md` | Schema-v1 compatibility plus schema-v2 consultation section and metadata examples. |
| `skills/kernel-opt-loop/references/report-template.md` | Sidecar KernelEvidenceRecord reference and report-to-evidence hash requirements. |
| `skills/kernel-opt-loop/scripts/validate_decision.py` | Backward-compatible decision schema-v1/v2 validation and normalized consultation output. |
| `skills/kernel-opt-loop/tests/test_kernelwiki_integration.py` | Decision-v2, packet, empty fallback, target isolation, and evidence-sidecar contract tests. |
| `skills/kernel-opt-loop/tests/test_contracts.py` | Updated cross-file contract and Future Work boundary assertions. |

## Common Interface Definitions

All new CLIs use process exit `0` on success and `2` for malformed user input,
invalid records, or contract violations. They print one canonical JSON object to
stdout on success and one stderr line beginning with `error:` on failure.

The shared module `skills/kernelwiki/scripts/kernelwiki_lib.py` publishes:

- `KernelWikiError(ValueError)`
- `resolve_wiki_root(script_path: str | Path) -> Path`
- `load_yaml_mapping(path: Path) -> dict[str, object]`
- `load_json_mapping(path: Path) -> dict[str, object]`
- `sha256_file(path: Path) -> str`
- `canonical_json(value: object) -> str`
- `parse_frontmatter(path: Path) -> tuple[dict[str, object], str]`
- `collect_records(root: Path, relative_dir: str) -> dict[str, tuple[Path, dict[str, object]]]`

The corpus validator `skills/kernelwiki/scripts/validate.py` publishes
`CorpusValidationError(KernelWikiError)` and
`validate_corpus(root: Path) -> dict[str, int]`.

The packet builder `skills/kernelwiki/scripts/build_packet.py` publishes
`PacketBuildError(KernelWikiError)` and
`build_packet(root: Path, query: dict[str, object]) -> dict[str, object]`.

The evidence validator `skills/kernelwiki/scripts/validate_evidence.py` publishes
`EvidenceValidationError(KernelWikiError)` and
`validate_evidence(path: Path) -> dict[str, object]`.

## Implementation Tasks

### Task 1: Create the Standalone Skill Skeleton and Stable Shared Utilities

**Files:**
- Create: `skills/kernelwiki/SKILL.md`
- Create: `skills/kernelwiki/README.md`
- Create: `skills/kernelwiki/requirements.txt`
- Create: `skills/kernelwiki/index.md`
- Create: `skills/kernelwiki/data/taxonomy.yaml`
- Create: `skills/kernelwiki/data/aliases.yaml`
- Create: `skills/kernelwiki/data/version-claims.yaml`
- Create: `skills/kernelwiki/references/source-policy.md`
- Create: `skills/kernelwiki/references/promotion-rubric.md`
- Create: `skills/kernelwiki/references/integration-contract.md`
- Create: `skills/kernelwiki/scripts/kernelwiki_lib.py`
- Create: `skills/kernelwiki/tests/test_kernelwiki_lib.py`

**Interfaces:**
- Produces: `KernelWikiError`, `resolve_wiki_root`, `load_yaml_mapping`,
  `load_json_mapping`, `sha256_file`, `canonical_json`, `parse_frontmatter`, and
  `collect_records` for every later KernelWiki script.
- Produces: an installable standalone skill whose documented modes are Curate,
  Retrieve, and Promote.
- Consumes: no existing code; it must remain independent from
  `kernel-opt-loop` imports.

- [ ] **Step 1: Write failing utility and standalone-skill tests**

Create `skills/kernelwiki/tests/test_kernelwiki_lib.py`. Use a temporary corpus
with `SKILL.md`, `data/taxonomy.yaml`, and YAML/JSON files. Test root discovery,
canonical serialization, SHA-256, frontmatter, duplicate IDs, and safe YAML
loading.

```python
from pathlib import Path
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
import sys
sys.path.insert(0, str(SCRIPTS))

from kernelwiki_lib import KernelWikiError, canonical_json, parse_frontmatter, resolve_wiki_root


class KernelWikiLibTests(unittest.TestCase):
    def test_canonical_json_is_stable_and_disallows_nan(self):
        self.assertEqual('{"a":1,"b":2}', canonical_json({"b": 2, "a": 1}))
        with self.assertRaises(KernelWikiError):
            canonical_json({"latency": float("nan")})

    def test_parse_frontmatter_requires_one_mapping_fence(self):
        with tempfile.TemporaryDirectory() as directory:
            page = Path(directory) / "page.md"
            page.write_text("---\nid: page-launch-bound\n---\n# Launch bound\n", encoding="utf-8")
            frontmatter, body = parse_frontmatter(page)
        self.assertEqual("page-launch-bound", frontmatter["id"])
        self.assertEqual("# Launch bound\n", body)

    def test_root_is_resolved_from_script_location(self):
        root = resolve_wiki_root(SCRIPTS / "query.py")
        self.assertEqual(SCRIPTS.parent, root)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m unittest skills/kernelwiki/tests/test_kernelwiki_lib.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'kernelwiki_lib'`.

- [ ] **Step 3: Create the skill contract, policy files, and utility module**

Create `requirements.txt` containing exactly:

```text
PyYAML>=6.0,<7
```

In `SKILL.md`, define these mode boundaries:

```text
Curate: inspect explicit sources and create non-active proposals.
Retrieve: return pages or packets without modifying projects.
Promote: move a proposal only after an explicit user approval reference.

Never write candidate code, benchmark, transition team state, mutate a target profile,
or insert raw source code into a Coder packet.
```

Implement safe, deterministic utility behavior:

```python
def canonical_json(value: object) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise KernelWikiError(f"value is not canonical JSON: {error}") from error


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
```

Populate `taxonomy.yaml` with the exact initial controlled lists required by the
spec: `triton_mlu`, `triton_gcu`; all five claim kinds; all claim/projection
statuses; page types; source kinds; `local-verifier`,
`official-doc-and-upstream-code`, `source-reported`, `inferred`,
`experimental`; and the initial regime tags listed in spec Section 6.1.

`aliases.yaml` starts with canonical aliases for `moe`, `topk`, `grouped-topk`,
`kernel-fusion`, `tl-dot`, `mlu`, and `gcu`. `version-claims.yaml` starts with
`claims: []`. The policy files must state the explicit approval, metadata-only
external source, artifact size-cap, and no-code-RAG rules from the spec.

- [ ] **Step 4: Run focused utility and static contract tests**

Run:

```bash
python3 -m unittest skills/kernelwiki/tests/test_kernelwiki_lib.py -v
python3 -m py_compile skills/kernelwiki/scripts/kernelwiki_lib.py
```

Expected: PASS. `py_compile` emits no output.

- [ ] **Step 5: Commit the standalone foundation**

```bash
git add skills/kernelwiki/SKILL.md skills/kernelwiki/README.md skills/kernelwiki/requirements.txt skills/kernelwiki/index.md skills/kernelwiki/data skills/kernelwiki/references skills/kernelwiki/scripts/kernelwiki_lib.py skills/kernelwiki/tests/test_kernelwiki_lib.py
git commit -m "feat: add KernelWiki skill foundation"
```

### Task 2: Define Record Schemas and Validate the Corpus Pipeline

**Files:**
- Create: `skills/kernelwiki/schemas/source.schema.json`
- Create: `skills/kernelwiki/schemas/claim.schema.json`
- Create: `skills/kernelwiki/schemas/wiki-page.schema.json`
- Create: `skills/kernelwiki/schemas/knowledge-query.schema.json`
- Create: `skills/kernelwiki/schemas/knowledge-packet.schema.json`
- Create: `skills/kernelwiki/schemas/kernel-evidence-record.schema.json`
- Create: `skills/kernelwiki/scripts/validate.py`
- Create: `skills/kernelwiki/tests/test_validate.py`
- Create: `skills/kernelwiki/tests/fixtures/corpus-valid/data/taxonomy.yaml`
- Create: `skills/kernelwiki/tests/fixtures/corpus-valid/data/aliases.yaml`
- Create: `skills/kernelwiki/tests/fixtures/corpus-valid/data/version-claims.yaml`
- Create: `skills/kernelwiki/tests/fixtures/corpus-valid/sources/local/mlu/source-mlu-launch.yaml`
- Create: `skills/kernelwiki/tests/fixtures/corpus-valid/claims/active/claim-launch-collapse.yaml`
- Create: `skills/kernelwiki/tests/fixtures/corpus-valid/wiki/patterns/launch-bound.md`
- Create: `skills/kernelwiki/tests/fixtures/corpus-invalid/README.md`

**Interfaces:**
- Consumes: Task 1 shared loading, taxonomy, aliases, and exception base.
- Produces: `CorpusValidationError` and `validate_corpus(root) -> dict[str, int]`.
- Produces: schema-valid records with the IDs later consumed by index and packet
  code.

- [ ] **Step 1: Write failing corpus validation tests**

Create `test_validate.py` with a copied temporary fixture corpus. Assert that a
valid corpus yields deterministic counts and invalid cases fail with stable codes.

```python
from pathlib import Path
import shutil
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
import sys
sys.path.insert(0, str(SCRIPTS))

from validate import CorpusValidationError, validate_corpus

FIXTURE = Path(__file__).parent / "fixtures" / "corpus-valid"


class CorpusValidationTests(unittest.TestCase):
    def copy_fixture(self) -> tuple[tempfile.TemporaryDirectory, Path]:
        directory = tempfile.TemporaryDirectory()
        root = Path(directory.name) / "kernelwiki"
        shutil.copytree(FIXTURE, root)
        return directory, root

    def test_valid_corpus_has_stable_counts(self):
        directory, root = self.copy_fixture()
        with directory:
            self.assertEqual({"claims": 1, "pages": 1, "sources": 1}, validate_corpus(root))

    def test_active_claim_rejects_unknown_regime_tag(self):
        directory, root = self.copy_fixture()
        with directory:
            claim = root / "claims" / "active" / "claim-launch-collapse.yaml"
            claim.write_text(claim.read_text(encoding="utf-8").replace("launch-bound", "invented-tag"), encoding="utf-8")
            with self.assertRaisesRegex(CorpusValidationError, "taxonomy-unknown-regime-tag"):
                validate_corpus(root)

    def test_cross_backend_claim_requires_two_local_projections(self):
        directory, root = self.copy_fixture()
        with directory:
            claim = root / "claims" / "active" / "claim-launch-collapse.yaml"
            text = claim.read_text(encoding="utf-8")
            claim.write_text(
                text.replace(
                    "  triton_gcu:\n    status: local-qualified\n    evidence: [source-gcu-launch]\n",
                    "  triton_gcu:\n    status: unavailable\n    evidence: []\n",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(CorpusValidationError, "cross-backend-evidence-insufficient"):
                validate_corpus(root)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m unittest skills/kernelwiki/tests/test_validate.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'validate'`.

- [ ] **Step 3: Implement JSON schema documents and deterministic validation**

Use simple JSON Schema documents with `required`, `properties`, and enumerated
values as the published contract. Keep the Python validator dependency-free
beyond PyYAML: load the schemas, enforce required fields and primitive types,
then enforce cross-record invariants in code.

Implement these validation rules in `validate_corpus`:

```python
ACTIVE_PROJECTION_STATUSES = {"local-qualified", "local-replicated"}
LOCAL_TARGETS = {"triton_mlu", "triton_gcu"}


def validate_cross_backend_claim(claim: dict[str, object]) -> None:
    if claim["transfer_class"] != "cross-backend-replicated":
        return
    projections = claim["projections"]
    qualified = [
        target for target in LOCAL_TARGETS
        if target in projections
        and projections[target]["status"] in ACTIVE_PROJECTION_STATUSES
        and projections[target]["evidence"]
    ]
    if len(qualified) != 2:
        raise CorpusValidationError(
            "cross-backend-evidence-insufficient: expected qualified MLU and GCU projections"
        )
```

Also enforce unique IDs; active-claim approval metadata; source and claim links;
valid projection lifecycle values; exact source hash format; taxonomy membership;
version-claim bidirectional references; page claim/source links; and no active
external-only projection in a target recipe position. Emit sorted JSON counts in
the CLI.

Make the valid fixture contain one `claim-launch-collapse` core claim with both
`triton_mlu` and `triton_gcu` `local-qualified` projections sharing
`runtime_launch_count_per_call` as a causal observable. The only wiki page lists
that claim and source ID in frontmatter.

- [ ] **Step 4: Run corpus validation tests and CLI checks**

Run:

```bash
python3 -m unittest skills/kernelwiki/tests/test_validate.py -v
python3 skills/kernelwiki/scripts/validate.py --root skills/kernelwiki/tests/fixtures/corpus-valid
```

Expected: all tests PASS and the CLI prints exactly:

```json
{"claims":1,"pages":1,"sources":1}
```

- [ ] **Step 5: Commit the schema and validator deliverable**

```bash
git add skills/kernelwiki/schemas skills/kernelwiki/scripts/validate.py skills/kernelwiki/tests/test_validate.py skills/kernelwiki/tests/fixtures/corpus-valid skills/kernelwiki/tests/fixtures/corpus-invalid
git commit -m "feat: validate KernelWiki corpus records"
```

### Task 3: Compile Claim-Backed Wiki Pages into Deterministic Navigation

**Files:**
- Create: `skills/kernelwiki/scripts/generate_indices.py`
- Create: `skills/kernelwiki/scripts/query.py`
- Create: `skills/kernelwiki/scripts/get_page.py`
- Create: `skills/kernelwiki/tests/test_queries.py`
- Create: `skills/kernelwiki/tests/fixtures/corpus-valid/wiki/techniques/kernel-fusion.md`
- Create: `skills/kernelwiki/tests/fixtures/corpus-valid/wiki/measurement/runtime-launch-not-device-time.md`
- Create: `skills/kernelwiki/tests/fixtures/corpus-valid/queries/.gitkeep`

**Interfaces:**
- Consumes: `validate_corpus`, `collect_records`, aliases, and active claim/page
  frontmatter from Tasks 1-2.
- Produces: `generate_indices(root, check=False) -> list[Path]`,
  `search(root, text, filters) -> list[dict[str, object]]`, and
  `get_record(root, identifier, follow_sources=False) -> dict[str, object]`.
- Produces: seven deterministic index files defined in the v1 specification.

- [ ] **Step 1: Write failing index/search/page-resolution tests**

Create `test_queries.py`. The test must copy `corpus-valid`, run the generator,
and assert an exact generated index plus alias-aware query behavior.

```python
from pathlib import Path
import shutil
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
import sys
sys.path.insert(0, str(SCRIPTS))

from generate_indices import generate_indices
from get_page import get_record
from query import search

FIXTURE = Path(__file__).parent / "fixtures" / "corpus-valid"


class QueryTests(unittest.TestCase):
    def test_generator_is_deterministic_and_groups_by_technique(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "kernelwiki"
            shutil.copytree(FIXTURE, root)
            first = generate_indices(root)
            first_text = (root / "queries" / "by-technique.md").read_text(encoding="utf-8")
            second = generate_indices(root)
        self.assertEqual(first, second)
        self.assertIn("kernel-fusion", first_text)
        self.assertIn("claim-launch-collapse", first_text)

    def test_alias_search_returns_page_not_raw_artifact(self):
        result = search(FIXTURE, text="fusion", filters={"target_profile": "triton_mlu"})
        self.assertEqual("page-kernel-fusion", result[0]["id"])
        self.assertEqual("wiki/techniques/kernel-fusion.md", result[0]["path"])

    def test_get_page_can_expand_cited_sources_only_when_requested(self):
        compact = get_record(FIXTURE, "page-kernel-fusion", follow_sources=False)
        expanded = get_record(FIXTURE, "page-kernel-fusion", follow_sources=True)
        self.assertNotIn("expanded_sources", compact)
        self.assertEqual("source-mlu-launch", expanded["expanded_sources"][0]["id"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m unittest skills/kernelwiki/tests/test_queries.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `generate_indices`, `query`, and
`get_page`.

- [ ] **Step 3: Implement generated indices and Curator-only exploration CLIs**

`generate_indices.py` must validate first, sort by canonical ID, render a stable
Markdown table, and write only the seven `queries/by-*.md` files. `--check` must
compare generated bytes with checked-in bytes and fail when a file is stale.

Use title > tag > body keyword scoring in `query.py`, but never let a text match
override an explicit `target_profile`, `claim_kind`, `source_kind`, or `page_type`
filter. Alias expansion only adds canonical terms before scoring.

```python
def score_page(frontmatter: dict[str, object], body: str, keywords: list[str]) -> int:
    title = str(frontmatter.get("title", "")).lower()
    tags = {str(tag).lower() for tag in frontmatter.get("tags", [])}
    score = 0
    for keyword in keywords:
        score += 8 if keyword in title else 0
        score += 4 if keyword in tags else 0
        score += 1 if keyword in body.lower() else 0
    return score
```

`get_page.py` resolves only a source, claim, or wiki page by ID/path. It follows
only source IDs from a requested wiki page. It never reads `artifacts/` unless a
future user-initiated deep-dive command is implemented.

- [ ] **Step 4: Run index, search, and complete corpus tests**

Run:

```bash
python3 -m unittest skills/kernelwiki/tests/test_queries.py -v
python3 skills/kernelwiki/scripts/generate_indices.py --root skills/kernelwiki/tests/fixtures/corpus-valid --check
python3 -m unittest skills/kernelwiki/tests/test_kernelwiki_lib.py skills/kernelwiki/tests/test_validate.py skills/kernelwiki/tests/test_queries.py -v
```

Expected: PASS. The `--check` invocation emits no stale-index error.

- [ ] **Step 5: Commit the compiled-navigation deliverable**

```bash
git add skills/kernelwiki/scripts/generate_indices.py skills/kernelwiki/scripts/query.py skills/kernelwiki/scripts/get_page.py skills/kernelwiki/tests/test_queries.py skills/kernelwiki/tests/fixtures/corpus-valid/wiki skills/kernelwiki/tests/fixtures/corpus-valid/queries
git commit -m "feat: add KernelWiki compiled navigation"
```

### Task 4: Build Deterministic, Authority-Partitioned KnowledgePackets

**Files:**
- Create: `skills/kernelwiki/scripts/build_packet.py`
- Create: `skills/kernelwiki/tests/test_build_packet.py`
- Create: `skills/kernelwiki/tests/fixtures/queries/exact-mlu.json`
- Create: `skills/kernelwiki/tests/fixtures/queries/exact-gcu.json`
- Create: `skills/kernelwiki/tests/fixtures/queries/no-match.json`
- Modify: `skills/kernelwiki/tests/fixtures/corpus-valid/claims/active/claim-launch-collapse.yaml`
- Create: `skills/kernelwiki/tests/fixtures/corpus-valid/claims/active/claim-small-m-tl-dot.yaml`
- Create: `skills/kernelwiki/tests/fixtures/corpus-valid/claims/active/claim-output-lifetime.yaml`

**Interfaces:**
- Consumes: validated active claims, query schema, taxonomy, and canonical JSON.
- Produces: `build_packet(root, query)` and a CLI accepting `--query PATH`,
  `--output PATH`, and `--root PATH`.
- Produces: packets with `hard_constraints`, `target_recipes`,
  `designer_hypotheses`, `counterexamples`, `selected_claim_ids`,
  `excluded_claims`, `wiki_revision`, `query_sha256`, and `empty_reason`.

- [ ] **Step 1: Write failing packet authority and target-isolation tests**

Create `test_build_packet.py` to cover strict filtering, all four authority
sections, deterministic tie-breaking, empty fallback, and prohibition on
cross-target recipes.

```python
from pathlib import Path
import copy
import json
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
import sys
sys.path.insert(0, str(SCRIPTS))

from build_packet import PacketBuildError, build_packet

ROOT = Path(__file__).parent / "fixtures" / "corpus-valid"
QUERY = json.loads((Path(__file__).parent / "fixtures" / "queries" / "exact-mlu.json").read_text())


class PacketTests(unittest.TestCase):
    def test_mlu_packet_never_contains_gcu_recipe(self):
        packet = build_packet(ROOT, QUERY)
        recipe_targets = {item["target_profile"] for item in packet["target_recipes"]}
        self.assertEqual({"triton_mlu"}, recipe_targets)
        self.assertEqual(["claim-output-lifetime"], [item["id"] for item in packet["hard_constraints"]])

    def test_exact_counterexample_is_included_and_external_analogy_is_designer_only(self):
        packet = build_packet(ROOT, QUERY)
        self.assertEqual(["claim-small-m-tl-dot"], [item["id"] for item in packet["counterexamples"]])
        self.assertNotIn("claim-small-m-tl-dot", [item["id"] for item in packet["target_recipes"]])

    def test_identical_query_produces_identical_packet(self):
        self.assertEqual(build_packet(ROOT, QUERY), build_packet(ROOT, copy.deepcopy(QUERY)))

    def test_no_match_returns_schema_valid_empty_packet(self):
        no_match = json.loads((Path(__file__).parent / "fixtures" / "queries" / "no-match.json").read_text())
        packet = build_packet(ROOT, no_match)
        self.assertEqual([], packet["target_recipes"])
        self.assertEqual("no-active-claim-matched", packet["empty_reason"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python3 -m unittest skills/kernelwiki/tests/test_build_packet.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'build_packet'`.

- [ ] **Step 3: Implement hard filtering, ranking, and bounded packet rendering**

Reject malformed structured queries before reading claims. Filter by current
target profile, exact runtime compatibility, regime tags, semantic constraints,
change scope, and change family. Apply the following authority transformation:

```python
def packet_section(claim: dict[str, object], target: str) -> str:
    if claim["kind"] == "semantic-safety":
        return "hard_constraints"
    projection = claim["projections"].get(target)
    if claim["kind"] == "counterexample" and projection is not None:
        return "counterexamples"
    if projection and projection["status"] in {"local-qualified", "local-replicated"}:
        return "target_recipes"
    return "designer_hypotheses"
```

Only `semantic-safety` entries that match current semantic tags can enter
`hard_constraints`. Only a current target projection with `local-qualified` or
`local-replicated` can enter `target_recipes`. Claims marked
`external-candidate`, `analogy-only`, or with no current projection must remain
Designer-only. Exact counterexamples always appear, even beyond the normal
three-item cap.

Sort matches by stable claim ID ascending first, then stably sort the same list
by these ranks descending. This makes stable ID the final ascending tie-breaker
without defining a fake inverse string:

```python
matches.sort(key=lambda match: str(match["claim"]["id"]))
matches.sort(
    key=lambda match: (
        match["transfer_rank"],
        match["runtime_exact_rank"],
        match["regime_overlap_count"],
        match["source_confidence_rank"],
        match["last_verified_timestamp"],
    ),
    reverse=True,
)
```

Emit at most three recipes and three hypotheses. Use `canonical_json` to hash
the query and packet source revision. If no active claim matches, emit all four
empty arrays and `empty_reason: "no-active-claim-matched"`.

- [ ] **Step 4: Run packet and corpus test suites**

Run:

```bash
python3 -m unittest skills/kernelwiki/tests/test_build_packet.py -v
python3 -m unittest discover -s skills/kernelwiki/tests -v
```

Expected: PASS. The discovery command runs Tasks 1-4 tests without requiring
accelerator hardware or network access.

- [ ] **Step 5: Commit deterministic packet generation**

```bash
git add skills/kernelwiki/scripts/build_packet.py skills/kernelwiki/tests/test_build_packet.py skills/kernelwiki/tests/fixtures/queries skills/kernelwiki/tests/fixtures/corpus-valid/claims
git commit -m "feat: build constrained KernelWiki packets"
```

### Task 5: Add Evidence Validation, Proposal, Promotion, and Holdout Reporting

**Files:**
- Create: `skills/kernelwiki/scripts/validate_evidence.py`
- Create: `skills/kernelwiki/scripts/propose_claim.py`
- Create: `skills/kernelwiki/scripts/promote_claim.py`
- Create: `skills/kernelwiki/scripts/evaluate_holdout.py`
- Create: `skills/kernelwiki/tests/test_curation.py`
- Create: `skills/kernelwiki/tests/test_evaluate_holdout.py`
- Create: `skills/kernelwiki/tests/fixtures/evidence/accepted-round.json`
- Create: `skills/kernelwiki/tests/fixtures/evidence/malformed-round.json`
- Create: `skills/kernelwiki/tests/fixtures/replay/holdout-results.json`
- Create: `skills/kernelwiki/candidates/sources/.gitkeep`
- Create: `skills/kernelwiki/artifacts/.gitkeep`

**Interfaces:**
- Consumes: Task 2 validation and Task 4 claim/query constants.
- Produces: `validate_evidence(path)`, `propose_claim(root, evidence_path)`,
  `promote_claim(root, proposal_id, approval_ref, approved_by)`, and
  `evaluate_holdout(records) -> dict[str, object]`.
- Produces: non-active proposed claim files and explicit approval records before
  any move into `claims/active/`.

- [ ] **Step 1: Write failing curation and holdout tests**

Create `test_curation.py` and `test_evaluate_holdout.py` with explicit approval
and no-project-mutation assertions.

```python
from pathlib import Path
import shutil
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
import sys
sys.path.insert(0, str(SCRIPTS))

from promote_claim import PromotionError, promote_claim
from validate_evidence import EvidenceValidationError, validate_evidence

FIXTURES = Path(__file__).parent / "fixtures"


class CurationTests(unittest.TestCase):
    def test_evidence_requires_bound_artifact_hashes(self):
        with self.assertRaises(EvidenceValidationError):
            validate_evidence(FIXTURES / "evidence" / "malformed-round.json")

    def test_promotion_requires_explicit_approval_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "kernelwiki"
            shutil.copytree(FIXTURES / "corpus-valid", root)
            proposal = root / "claims" / "proposed" / "claim-proposed.yaml"
            proposal.parent.mkdir(parents=True, exist_ok=True)
            proposal.write_text("id: claim-proposed\nstatus: proposed\n", encoding="utf-8")
            with self.assertRaisesRegex(PromotionError, "approval reference"):
                promote_claim(root, "claim-proposed", approval_ref="", approved_by="")
```

```python
from evaluate_holdout import evaluate_holdout


def test_holdout_reports_required_safety_metrics():
    result = evaluate_holdout([
        {"mode": "empty-packet", "decision_valid": True, "compile_smoke_pass": True, "correctness_pass": True, "repeated_counterexample": True, "rounds_to_first_accepted": 3},
        {"mode": "kernelwiki", "decision_valid": True, "compile_smoke_pass": True, "correctness_pass": True, "repeated_counterexample": False, "rounds_to_first_accepted": 2},
    ])
    assert result["kernelwiki"]["repeated_counterexample_rate"] == 0.0
    assert result["kernelwiki"]["correctness_pass_rate"] == 1.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python3 -m unittest skills/kernelwiki/tests/test_curation.py skills/kernelwiki/tests/test_evaluate_holdout.py -v
```

Expected: FAIL with `ModuleNotFoundError` for the curation and holdout modules.

- [ ] **Step 3: Implement evidence, promotion, and replay contracts**

`validate_evidence.py` must reject missing report/decision/candidate/reference,
base, harness, runtime, or measurement hashes; unsupported terminal result;
non-list observables; proposal text that prescribes a next implementation; and
paths escaping the declared project root.

`propose_claim.py` writes one canonical YAML proposal under `claims/proposed/`
using the proposal's stable claim ID as the filename. It preserves source evidence
IDs, sets `status: proposed`, sets `approval: null`, and writes a `proposal`
mapping with the SHA-256 returned by `validate_evidence`, a UTC ISO-8601 creation
timestamp, and `generated_by: kernelwiki-propose-claim`.

`promote_claim.py` must require nonempty `approval_ref` and `approved_by`,
validate the proposal against the active-claim schema, set `status: active`, and
write immutable approval metadata before moving the file to `claims/active/`.
The metadata has exactly three fields: `approved_by: user`, the literal approval
reference supplied to the command, and a UTC ISO-8601 `approved_at` timestamp.

The CLI records approval but does not claim to authenticate a human. The
KernelWiki skill contract states that only a user-approved invocation is allowed.
Neither script may write beneath a project root.

`evaluate_holdout.py` validates per-case booleans and nullable rounds, groups by
`empty-packet` and `kernelwiki`, and returns decision-validity, compile-smoke,
correctness-pass, repeated-counterexample, terminal-classification, and
first-acceptance-round metrics. It must explicitly report missing mode data
rather than inventing a comparison.

- [ ] **Step 4: Run curation, holdout, and complete KernelWiki suites**

Run:

```bash
python3 -m unittest skills/kernelwiki/tests/test_curation.py skills/kernelwiki/tests/test_evaluate_holdout.py -v
python3 -m unittest discover -s skills/kernelwiki/tests -v
```

Expected: PASS. Promotion without both approval fields fails, and no test writes
outside its temporary corpus root.

- [ ] **Step 5: Commit curation governance and replay reporting**

```bash
git add skills/kernelwiki/scripts/validate_evidence.py skills/kernelwiki/scripts/propose_claim.py skills/kernelwiki/scripts/promote_claim.py skills/kernelwiki/scripts/evaluate_holdout.py skills/kernelwiki/tests/test_curation.py skills/kernelwiki/tests/test_evaluate_holdout.py skills/kernelwiki/tests/fixtures/evidence skills/kernelwiki/tests/fixtures/replay skills/kernelwiki/candidates skills/kernelwiki/artifacts
git commit -m "feat: add KernelWiki curation governance"
```

### Task 6: Add Backward-Compatible kernel-opt-loop Schema-v2 Integration

**Files:**
- Create: `skills/kernel-opt-loop/references/kernelwiki-integration.md`
- Create: `skills/kernel-opt-loop/tests/test_kernelwiki_integration.py`
- Create: `skills/kernel-opt-loop/tests/fixtures/kernelwiki-project/rounds/decision_001.md`
- Create: `skills/kernel-opt-loop/tests/fixtures/kernelwiki-project/state/kernelwiki_snapshot_001.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/packets/empty-packet.json`
- Modify: `skills/kernel-opt-loop/SKILL.md`
- Modify: `skills/kernel-opt-loop/prompts/designer.md`
- Modify: `skills/kernel-opt-loop/prompts/coder.md`
- Modify: `skills/kernel-opt-loop/prompts/verifier.md`
- Modify: `skills/kernel-opt-loop/references/decision-template.md`
- Modify: `skills/kernel-opt-loop/references/report-template.md`
- Modify: `skills/kernel-opt-loop/scripts/validate_decision.py`
- Modify: `skills/kernel-opt-loop/tests/test_validate_decision.py`
- Modify: `skills/kernel-opt-loop/tests/test_contracts.py`

**Interfaces:**
- Consumes: `build_packet.py` JSON output and `validate_evidence.py` schema from
  Tasks 4-5; existing decision schema-v1 and v2 role contracts.
- Produces: valid schema-v2 decisions with `KernelWiki Consultation`, frozen
  snapshot path/hash metadata, and normalized `kernelwiki_consultation` output.
- Produces: explicit adapter fallback behavior and project-side
  `rounds/kernel_evidence_NNN.json` ownership.

- [ ] **Step 1: Write failing schema-v2 decision and contract tests**

Create `test_kernelwiki_integration.py` and extend `test_validate_decision.py`.
The existing v1 fixtures must still validate without modification.

```python
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from validate_decision import DecisionValidationError, validate_decision

FIXTURES = Path(__file__).parent / "fixtures"
PROJECT_FIXTURE = FIXTURES / "kernelwiki-project"


class KernelWikiDecisionTests(unittest.TestCase):
    def test_schema_v1_fixture_remains_valid(self):
        result = validate_decision(FIXTURES / "decisions" / "kernel-valid.md")
        self.assertEqual(1, result["metadata"]["schema_version"])
        self.assertNotIn("kernelwiki_consultation", result)

    def test_schema_v2_requires_snapshot_hash_and_consultation(self):
        decision = PROJECT_FIXTURE / "rounds" / "decision_001.md"
        result = validate_decision(decision)
        self.assertEqual(2, result["metadata"]["schema_version"])
        self.assertEqual("state/kernelwiki_snapshot_001.json", result["metadata"]["kernelwiki_snapshot"])
        self.assertEqual(["claim-launch-collapse"], result["kernelwiki_consultation"]["selected_claim_ids"])

    def test_schema_v2_rejects_designer_only_claim_as_coder_recipe(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "project"
            shutil.copytree(PROJECT_FIXTURE, project)
            path = project / "rounds" / "decision_001.md"
            text = path.read_text(encoding="utf-8")
            path.write_text(
                text.replace(
                    '"target_recipe_claim_ids":["claim-launch-collapse"]',
                    '"target_recipe_claim_ids":["claim-external-cuda-analogy"]',
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(DecisionValidationError, "kernelwiki-recipe-claim-invalid"):
                validate_decision(path)
```

- [ ] **Step 2: Run integration tests to verify they fail**

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_kernelwiki_integration.py skills/kernel-opt-loop/tests/test_validate_decision.py -v
```

Expected: FAIL because schema version `2` and the `KernelWiki Consultation`
section are currently unsupported.

- [ ] **Step 3: Implement schema-v2 and optional adapter contracts**

Preserve `REQUIRED_SECTIONS_V1` exactly. Add this v2-only section after
`Pitfalls and Anti-pattern Consultation` and before `Rationale and Evidence`:

```text
KernelWiki Consultation
```

Add `METADATA_FIELDS_V2` with all v1 fields plus:

```python
"kernelwiki_snapshot": str,
"kernelwiki_snapshot_sha256": str,
"kernelwiki_revision": str,
```

The v2 consultation JSON object must contain:

```json
{
  "packet_schema_version": 1,
  "packet_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "selected_claim_ids": ["claim-launch-collapse"],
  "target_recipe_claim_ids": ["claim-launch-collapse"],
  "counterexample_claim_ids": [],
  "designer_hypothesis_claim_ids": [],
  "consultation_outcome": "adopted-target-recipe"
}
```

Validate IDs against the referenced frozen packet JSON beside the decision. The
validator must reject a target recipe ID not present in the packet's
`target_recipes`, reject a Coder-only use of a Designer hypothesis, and reject a
snapshot hash mismatch. Schema-v1 continues to reject unknown sections and does
not require a packet.

Update the new adapter reference and contracts with this exact phase order and
an explicit statement that schema-v1 campaigns continue to use
`references/anti-patterns.md` without KernelWiki:

```text
Orchestrator writes query -> KernelWiki builds packet -> Orchestrator writes snapshot
-> Designer cites snapshot -> validate decision -> Coder reads packet -> Verifier writes evidence sidecar
```

The fallback sequence is:

```text
KernelWiki unavailable or packet builder fails -> write schema-valid empty packet
with empty_reason -> hash and snapshot it -> continue current round without a
KernelWiki-derived recipe.
```

Update the roles so Designer reads all four sections but only uses hypotheses to
select an intervention; Coder reads only hard constraints and target recipes;
Verifier owns `rounds/kernel_evidence_NNN.json` and cannot promote a claim.
Update the report template with an evidence-sidecar identity/hash row. Update
`test_contracts.py` to remove the obsolete assertion that all KernelWiki API
references are future-only and replace it with schema-v1 compatibility,
schema-v2 adapter, no-code-RAG, and no-target-profile-mutation checks.

- [ ] **Step 4: Run loop integration and full existing loop suites**

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_kernelwiki_integration.py skills/kernel-opt-loop/tests/test_validate_decision.py -v
python3 -m unittest discover -s skills/kernel-opt-loop/tests -v
```

Expected: PASS. Existing schema-v1 fixtures remain valid, schema-v2 fixtures
validate only with a matching frozen packet, and no test requires accelerator
hardware.

- [ ] **Step 5: Commit the optional loop adapter**

```bash
git add skills/kernel-opt-loop/SKILL.md skills/kernel-opt-loop/prompts skills/kernel-opt-loop/references/kernelwiki-integration.md skills/kernel-opt-loop/references/decision-template.md skills/kernel-opt-loop/references/report-template.md skills/kernel-opt-loop/scripts/validate_decision.py skills/kernel-opt-loop/tests/test_kernelwiki_integration.py skills/kernel-opt-loop/tests/test_validate_decision.py skills/kernel-opt-loop/tests/test_contracts.py skills/kernel-opt-loop/tests/fixtures/kernelwiki-project skills/kernel-opt-loop/tests/fixtures/packets
git commit -m "feat: integrate KernelWiki decision snapshots"
```

### Task 7: Curate the Initial Local MLU/GCU Evidence as Non-Active Proposals

**Files:**
- Create: `skills/kernelwiki/sources/local/mlu/source-mlu-groupedtopk.yaml`
- Create: `skills/kernelwiki/sources/local/mlu/source-mlu-fused-moe.yaml`
- Create: `skills/kernelwiki/sources/local/mlu/source-mlu-flexattention.yaml`
- Create: `skills/kernelwiki/sources/local/mlu/source-mlu-sparse-pooler-tldot.yaml`
- Create: `skills/kernelwiki/sources/local/gcu/source-gcu-groupedtopk-launch.yaml`
- Create: `skills/kernelwiki/sources/local/gcu/source-gcu-runtime-launch-measurement.yaml`
- Create: `skills/kernelwiki/claims/proposed/claim-launch-collapse.yaml`
- Create: `skills/kernelwiki/claims/proposed/claim-harness-loader-authoritative.yaml`
- Create: `skills/kernelwiki/claims/proposed/claim-gcu-runtime-launch-not-device-time.yaml`
- Create: `skills/kernelwiki/claims/proposed/claim-mlu-small-m-tl-dot.yaml`
- Create: `skills/kernelwiki/claims/proposed/claim-mlu-topk-selection-counterexamples.yaml`
- Create: `skills/kernelwiki/candidates/sources/external-kernel-repos.yaml`
- Modify: `skills/kernelwiki/tests/test_validate.py`
- Modify: `skills/kernelwiki/tests/test_queries.py`

**Interfaces:**
- Consumes: all Corpus, packet, curation, and generator interfaces from Tasks
  1-5.
- Produces: a reviewable, non-active production evidence revision with MLU/GCU
  source records, target-scoped claim proposals, and an external-candidate ledger.
  It cannot affect any KnowledgePacket before the user approves a proposal.

- [ ] **Step 1: Write failing production-corpus tests**

Extend `test_validate.py` with repository-root corpus checks. The test must
assert that all five initial local claims remain non-active proposals and cannot
be selected into a production packet before promotion.

```python
from kernelwiki_lib import collect_records

REPO_ROOT = Path(__file__).resolve().parents[3]
PRODUCTION_ROOT = REPO_ROOT / "skills" / "kernelwiki"


def test_initial_local_claims_are_valid_non_active_proposals(self):
    counts = validate_corpus(PRODUCTION_ROOT)
    self.assertGreaterEqual(counts["sources"], 6)
    proposals = collect_records(PRODUCTION_ROOT, "claims/proposed")
    self.assertEqual(
        {
            "claim-gcu-runtime-launch-not-device-time",
            "claim-harness-loader-authoritative",
            "claim-launch-collapse",
            "claim-mlu-small-m-tl-dot",
            "claim-mlu-topk-selection-counterexamples",
        },
        set(proposals),
    )
    self.assertTrue(all(record[1]["status"] == "proposed" for record in proposals.values()))
```

- [ ] **Step 2: Run production-corpus tests to verify they fail**

Run:

```bash
python3 -m unittest skills/kernelwiki/tests/test_validate.py -v
```

Expected: FAIL because the production source records and non-active claim
proposals do not exist yet.

- [ ] **Step 3: Curate immutable source records and claim proposals**

Create each local source with an explicit project-relative evidence path,
committed source revision, current SHA-256 digest, runtime fingerprint facts,
measurement facts, source category `local-verifier`, and the exact source tags.
Obtain the committed revision and digest while creating the record:

```bash
git rev-parse HEAD
sha256sum mlu/groupedtopk/outcome.md mlu/fused_moe/outcome.md mlu/flexattention/outcome.md
sha256sum mlu/sparse_pooler/rounds/report_003.md s60/groupedtopk/rounds/report_001.md
```

Use the following bounded claim statements and scopes:

- `claim-launch-collapse`: graph-level core claim that a legal fusion boundary
  can reduce materialization and dispatch. Add `local-qualified` MLU and GCU
  projections with `runtime_launch_count_per_call` as shared observable; set
  `transfer_class: cross-backend-replicated` only after both source links and
  target-specific recipes validate.
- `claim-harness-loader-authoritative`: semantic-safety claim that direct import
  success cannot replace the actual harness loader gate. It has no target recipe
  and appears in packet `hard_constraints` only when the project declares an AST
  loader constraint.
- `claim-gcu-runtime-launch-not-device-time`: target-local measurement claim
  restricting GCU runtime-launch values from being used as device duration or a
  device ratio.
- `claim-mlu-small-m-tl-dot`: MLU counterexample scoped to small-M Triton dot
  fusion, with the sparse-pooler report as source and `reconsider_when` requiring
  a different shape regime, lowering, or matched microprobe.
- `claim-mlu-topk-selection-counterexamples`: MLU counterexample for the
  hierarchical winner tree, full sort network, dynamic gather, and cumsum paths;
  preserve each source condition and reconsideration boundary from the existing
  anti-pattern evidence.

Create the external ledger with `include`, `defer`, and `exclude` records for
KernelWiki-relevant upstream repositories; all entries remain
`external-candidate` and metadata-only. Do not create wiki pages, generated
indices, or an active claim in this task.

- [ ] **Step 4: Run source/proposal validation and focused tests**

Run:

```bash
python3 skills/kernelwiki/scripts/validate.py --root skills/kernelwiki
python3 -m unittest skills/kernelwiki/tests/test_validate.py -v
```

Expected: validation succeeds, every new claim is `proposed`, and the focused
unit test PASSes. No packet may select a proposed claim.

- [ ] **Step 5: Commit the reviewable proposal revision**

```bash
git add skills/kernelwiki/sources skills/kernelwiki/claims/proposed skills/kernelwiki/candidates/sources/external-kernel-repos.yaml skills/kernelwiki/tests/test_validate.py
git commit -m "docs: propose KernelWiki local evidence claims"
```

### Task 8: Promote Individually Approved Claims and Compile the Active Corpus

**Files:**
- Move on user approval: `skills/kernelwiki/claims/proposed/claim-launch-collapse.yaml` to `skills/kernelwiki/claims/active/claim-launch-collapse.yaml`
- Move on user approval: `skills/kernelwiki/claims/proposed/claim-harness-loader-authoritative.yaml` to `skills/kernelwiki/claims/active/claim-harness-loader-authoritative.yaml`
- Move on user approval: `skills/kernelwiki/claims/proposed/claim-gcu-runtime-launch-not-device-time.yaml` to `skills/kernelwiki/claims/active/claim-gcu-runtime-launch-not-device-time.yaml`
- Move on user approval: `skills/kernelwiki/claims/proposed/claim-mlu-small-m-tl-dot.yaml` to `skills/kernelwiki/claims/active/claim-mlu-small-m-tl-dot.yaml`
- Move on user approval: `skills/kernelwiki/claims/proposed/claim-mlu-topk-selection-counterexamples.yaml` to `skills/kernelwiki/claims/active/claim-mlu-topk-selection-counterexamples.yaml`
- Create: `skills/kernelwiki/wiki/techniques/kernel-fusion.md`
- Create: `skills/kernelwiki/wiki/patterns/launch-bound.md`
- Create: `skills/kernelwiki/wiki/runtime/harness-loader-contract.md`
- Create: `skills/kernelwiki/wiki/measurement/runtime-launch-not-device-time.md`
- Create: `skills/kernelwiki/wiki/techniques/tl-dot-shape-scope.md`
- Create: `skills/kernelwiki/wiki/techniques/topk-selection-counterexamples.md`
- Create: `skills/kernelwiki/queries/by-target-profile.md`
- Create: `skills/kernelwiki/queries/by-operator.md`
- Create: `skills/kernelwiki/queries/by-technique.md`
- Create: `skills/kernelwiki/queries/by-bottleneck.md`
- Create: `skills/kernelwiki/queries/by-symptom.md`
- Create: `skills/kernelwiki/queries/by-semantic-constraint.md`
- Create: `skills/kernelwiki/queries/by-source-repo.md`
- Modify: `skills/kernelwiki/tests/test_validate.py`
- Modify: `skills/kernelwiki/tests/test_queries.py`

**Interfaces:**
- Consumes: Task 7's committed source records and non-active proposals, plus a
  user-supplied approval reference for each promotion.
- Produces: approved active claims, claim-backed wiki pages, generated indices,
  and only current-target local recipes in production KnowledgePackets.

- [ ] **Step 1: Write failing active-corpus and target-isolation tests**

Extend `test_validate.py` and `test_queries.py` with repository-root checks. The
tests must assert that the promoted MLU/GCU claims validate, the GCU measurement
claim remains target-local, and no external candidate becomes an active Coder
recipe.

```python
from build_packet import build_packet
from kernelwiki_lib import collect_records


def test_active_production_corpus_validates_and_indices_are_current(self):
    counts = validate_corpus(PRODUCTION_ROOT)
    active = collect_records(PRODUCTION_ROOT, "claims/active")
    self.assertGreaterEqual(counts["sources"], 6)
    self.assertTrue(all(record[1]["status"] == "active" for record in active.values()))
    self.assertTrue(all(record[1]["approval"] for record in active.values()))
    self.assertEqual([], generate_indices(PRODUCTION_ROOT, check=True))


def test_external_ledger_does_not_create_an_active_recipe(self):
    packet = build_packet(PRODUCTION_ROOT, {
        "schema_version": 1,
        "target_profile": "triton_mlu",
        "runtime_fingerprint_ref": "fixture#runtime",
        "measurement_fingerprint": "0000000000000000000000000000000000000000000000000000000000000000",
        "operator_tags": ["moe"],
        "regime_tags": ["launch-bound"],
        "dtype_layout_tags": ["contiguous-row-major"],
        "bottleneck_tags": ["launch-bound"],
        "semantic_constraints": [],
        "change_scope": "kernel",
        "change_family": "kernel-fusion",
    })
    self.assertNotIn("claim-external-cuda-analogy", [item["id"] for item in packet["target_recipes"]])
    self.assertTrue(all(item["target_profile"] == "triton_mlu" for item in packet["target_recipes"]))
```

- [ ] **Step 2: Run the active-corpus tests to verify they fail**

Run:

```bash
python3 -m unittest skills/kernelwiki/tests/test_validate.py skills/kernelwiki/tests/test_queries.py -v
```

Expected: FAIL because all initial claims are still non-active proposals and the
production pages and generated indices do not exist yet.

- [ ] **Step 3: Obtain explicit approval and promote only approved proposals**

Present the following five proposal IDs, exact core statement, target projections,
source IDs, counterexamples, and reconsideration conditions to the user:

```text
claim-launch-collapse
claim-harness-loader-authoritative
claim-gcu-runtime-launch-not-device-time
claim-mlu-small-m-tl-dot
claim-mlu-topk-selection-counterexamples
```

Do not call `promote_claim.py` for a rejected or unanswered proposal. For each
approved ID, invoke the promotion CLI once with the literal approval reference
returned by the user and `--approved-by user`. The resulting active YAML must
contain its own approval object and preserve source content hashes. A rejected
proposal remains at `claims/proposed/` with `status: proposed`.

After promotion, create concise pages only for the approved active claims. Each
page must cite claim/source IDs, explain scope and counterexamples, and contain
no copied kernel source, launch parameter, raw artifact text, or acceptance
history. Generate all navigation pages; never hand-edit them:

```bash
python3 skills/kernelwiki/scripts/generate_indices.py --root skills/kernelwiki
```

- [ ] **Step 4: Run active-corpus, packet, index, and holdout checks**

Run:

```bash
python3 skills/kernelwiki/scripts/validate.py --root skills/kernelwiki
python3 skills/kernelwiki/scripts/generate_indices.py --root skills/kernelwiki --check
python3 -m unittest discover -s skills/kernelwiki/tests -v
python3 skills/kernelwiki/scripts/evaluate_holdout.py --records skills/kernelwiki/tests/fixtures/replay/holdout-results.json
```

Expected: validation succeeds; generated indices are current; all unit tests
PASS; and the holdout report includes both `empty-packet` and `kernelwiki` metric
groups without reporting an accelerator benchmark result.

- [ ] **Step 5: Commit the approved active corpus revision**

```bash
git add skills/kernelwiki/claims/active skills/kernelwiki/claims/proposed skills/kernelwiki/wiki skills/kernelwiki/queries skills/kernelwiki/tests/test_validate.py skills/kernelwiki/tests/test_queries.py
git commit -m "feat: promote approved KernelWiki local evidence"
```

### Task 9: Run End-to-End Contract Verification and Update the Operator Documentation

**Files:**
- Modify: `skills/kernelwiki/README.md`
- Modify: `skills/kernelwiki/index.md`
- Modify: `skills/kernelwiki/SKILL.md`
- Create: `docs/superpowers/plans/2026-08-17-kernelwiki-v1-verification.md`

**Interfaces:**
- Consumes: all prior implementations and generated corpus state.
- Produces: a documented operational verification record and explicit legacy
  anti-pattern relationship without rewriting project history.

- [ ] **Step 1: Write failing cross-skill contract assertions**

Extend `skills/kernel-opt-loop/tests/test_contracts.py` with assertions that the
legacy anti-pattern catalog remains available, but the contracts point to the new
adapter only for schema-v2 campaigns. Add checks that no KernelWiki document
contains a Coder instruction to read `artifacts/` or an external source path.

```python
def test_kernelwiki_adapter_preserves_legacy_anti_patterns_and_blocks_code_rag(self):
    adapter = read_reference("kernelwiki-integration.md")
    anti_patterns = read_reference("anti-patterns.md")
    wiki_skill = (REPO_ROOT / "skills" / "kernelwiki" / "SKILL.md").read_text(encoding="utf-8")
    self.assertTrue(anti_patterns.strip())
    self.assertIn("references/anti-patterns.md", adapter)
    self.assertIn("schema-v2", adapter)
    self.assertNotIn("Coder reads artifacts", adapter)
    self.assertNotIn("Coder reads artifacts", wiki_skill)
```

- [ ] **Step 2: Run cross-skill contract tests to verify they fail**

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_contracts.py -v
```

Expected: FAIL until the final contract and documentation wording is aligned with
the implemented adapter.

- [ ] **Step 3: Update operator docs and preserve legacy boundaries**

Update the KernelWiki README/index/SKILL with exact install, validation,
exploration, packet-building, proposal, promotion, and index-generation commands.
Document that `artifacts/` are Curator-only and that packet failures return an
empty packet.

Do not modify `references/anti-patterns.md`, existing target profiles, active
project artifacts, or the approved v1 design specification. The schema-v1 fallback
relationship is documented exclusively in `kernelwiki-integration.md` from Task 6.

Write `docs/superpowers/plans/2026-08-17-kernelwiki-v1-verification.md` as an
execution record containing the exact commands, commit IDs, corpus revision,
index check result, unit-test counts, and holdout fixture result. It must not
claim accelerator benchmarking occurred.

- [ ] **Step 4: Run the complete repository verification set**

Run:

```bash
python3 -m unittest discover -s skills/kernelwiki/tests -v
python3 -m unittest discover -s skills/kernel-opt-loop/tests -v
python3 skills/kernelwiki/scripts/validate.py --root skills/kernelwiki
python3 skills/kernelwiki/scripts/generate_indices.py --root skills/kernelwiki --check
git diff --check
git status --short
```

Expected: both test suites PASS, corpus validation succeeds, generated indices
are current, `git diff --check` has no output, and status contains only the
planned KernelWiki/loop documentation changes before their final commit.

- [ ] **Step 5: Commit final documentation and verification record**

```bash
git add skills/kernelwiki/README.md skills/kernelwiki/index.md skills/kernelwiki/SKILL.md skills/kernel-opt-loop/tests/test_contracts.py docs/superpowers/plans/2026-08-17-kernelwiki-v1-verification.md
git commit -m "docs: verify KernelWiki v1 contracts"
```

## Plan Self-Review

### Spec coverage

| v1 design requirement | Plan tasks |
|---|---|
| Standalone self-contained skill | Tasks 1-3 |
| `sources -> claims -> wiki -> queries` compiler model | Tasks 2-3 and 7-8 |
| Controlled taxonomy, aliases, version claims | Tasks 1-2 |
| Atomic claim lifecycle and cross-backend projections | Tasks 2, 4, 5, 7, and 8 |
| No code RAG and Curator-only artifacts | Tasks 1, 3, 5, 6, and 9 |
| Structured bounded KnowledgePacket | Task 4 |
| Explicit promotion approval | Tasks 5 and 8 |
| KernelEvidenceRecord and optional loop adapter | Tasks 5-6 |
| Schema-v1 campaign compatibility | Task 6 |
| Initial local MLU/GCU corpus and external candidate ledger | Task 7 |
| Generated indices and deterministic validation | Tasks 2-4 and 8 |
| Holdout replay reporting | Tasks 5 and 8 |
| Cross-skill safety and final verification | Task 9 |

### Placeholder scan

No placeholder markers or undefined function references remain. Dynamic source
digests in Task 7 are deliberately produced by exact commands at curation time
because they must bind the then-current committed local artifact bytes.

### Type consistency

All later tasks use the interfaces produced earlier: `KernelWikiError` from Task
1; `validate_corpus` from Task 2; index/search/page functions from Task 3;
`build_packet` from Task 4; curation and replay functions from Task 5; and the
schema-v2 adapter from Task 6. No later task requires a name not defined in an
earlier task.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-17-kernelwiki-v1-implementation.md`. Two execution options:

1. **Subagent-Driven (recommended)** - Dispatch a fresh subagent per task and review between tasks for fast, bounded iteration.
2. **Inline Execution** - Execute tasks in this session using `superpowers:executing-plans`, with checkpoints for review.

Choose one approach before implementation begins.

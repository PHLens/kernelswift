# KernelWiki Role-Aware Query Admission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add standalone Designer/Coder query admission that classifies broad research evidence safely and exposes Coder implementation guidance only for exact, validated profile/runtime/Sketch contexts.

**Architecture:** Extend the standalone core with a versioned role-query context, a read-only bridge to the latest checked-in `kernel-opt-loop` validators, and an admission engine that runs before ranking. Designer receives broad explicitly classified evidence; Coder receives only exact-profile Cards and independently approved assets bound to frozen Sketch statements. Output remains an ephemeral query result or caller-saved receipt, never a campaign artifact or required dossier.

**Tech Stack:** Python 3 standard library, the pinned PyYAML dependency from the standalone-core plan, Markdown/YAML Card metadata, current checked-in `kernel-opt-loop` Python validators as read-only imports, `unittest`, Git.

**Spec:** `docs/superpowers/specs/2026-08-17-kernelwiki-v1-design.md`

**Depends on:** Completed and green `docs/superpowers/plans/2026-08-21-kernelwiki-standalone-core.md`.

## Execution Granularity

Each named test method below is one red/green micro-step: add only that test, run its exact test module, implement the smallest behavior that passes it, and rerun before adding the next method. Checkbox steps are review gates that group these 2–5 minute micro-steps; do not batch a prose list into one implementation jump.

## Global Constraints

- Modify only `skills/kernelwiki/` and KernelWiki documentation. Do not modify `skills/kernel-opt-loop/`, its prompts, schemas, validators, templates, profiles, or campaign artifacts.
- Consume the latest checked-in `kernel-opt-loop` authority at implementation time and pin the exact artifact hashes/contract version in each role-query request. Unsupported or invalid contracts fail closed.
- Designer admission may include exact, family, backend, analogy-only, counterexample, measurement, Source, and capability-gap evidence, but every result exposes its match class.
- Coder admission requires exact target, implementation profile, runtime, current version qualification, non-Unknown/non-Unsupported capability, preserved frozen Sketch/Decision semantics, explicit Sketch-statement binding, approved provenance/license, and approved asset mode.
- A Coder query never falls back across target, backend, implementation profile, language, or runtime.
- Card admission and asset/example/snippet admission are independent. A readable Card never grants all cited assets automatically.
- Admission happens before lexical ranking and before result limits. Do not rank everything, truncate top N, and then filter.
- Positive results, counterexamples, and capability gaps remain separate groups so positive ranking cannot suppress the latter two.
- No persisted dossier is required. A caller may save canonical query JSON as a receipt, but that file has no campaign authority and is not a Designer-to-Coder handoff.
- Missing Wiki/profile/capability returns a schema-valid empty or capability-gap result and never blocks the local optimization workflow.
- Do not create/register a fake canonical AscendC profile. Real missing AscendC authority must continue to produce an exact-profile empty Coder result.
- Do not modify Designer/Coder prompts, Decisions, Sketches, `coder_result`, Orchestrator state, or consultation-record formats. Those remain Phase E concerns.
- Use the sealed holdout manifest created before standalone-core taxonomy/ranking work. Do not edit holdout judgments to improve retrieval metrics.

## Planned File Map

```text
skills/kernelwiki/
  references/
    role-query-contract.md           # Context/result/admission contract; no campaign writes
    evaluation-protocol.md           # Extended with role-aware final evaluation

  data/
    schemas.yaml                     # Add role_query_context/result schema versions
    version-claims.yaml              # Existing registry consumed by admission
    track2-development-queries.yaml  # Non-holdout structured query contexts

  scripts/
    kernel_opt_bridge.py             # Read-only loader for current loop validators
    role_context.py                  # Designer/Coder context parsing and authority pinning
    admission.py                     # Card/item/asset admission and match classes
    role_search.py                   # Admission-first grouped search orchestration
    evaluate_holdout.py              # Final Track 2/adversarial evaluator
    query.py                         # Add --context and grouped role output
    get_page.py                      # Apply page/item/asset admission
    search.py                        # Reuse neutral scoring after admission
    corpus.py                        # Parse optional coder_access sections
    validate.py                      # Validate coder_access/version references

  tests/
    test_kernel_opt_bridge.py
    role_fixture_factory.py
    test_role_context.py
    test_admission.py
    test_role_search.py
    test_role_contracts.py
    fixtures/role/
      designer-context.json
      coder-missing-profile.json
    fixtures/cards/
      exact-coder-card.md
      analogy-designer-card.md
      mixed-asset-card.md
    fixtures/track2/
      sparse-attn-development.json
      index-topk-development.json
      adversarial-dot-scope.json
      adversarial-output-reuse.json
      adversarial-device-wall.json
      adversarial-topk-transfer.json
      adversarial-profiler-evidence.json
```

### Stable Role Interfaces

```python
@dataclass(frozen=True)
class ArtifactRef:
    path: Path
    sha256: str

@dataclass(frozen=True)
class LoopContractIdentity:
    repository_commit: str
    skill_tree_sha: str
    validator_sha256: Mapping[str, str]
    schema_sha256: Mapping[str, str]

@dataclass(frozen=True)
class RoleQueryContext:
    schema_version: int
    contract_version: int | None
    role: str  # designer | coder
    target_id: str
    implementation_profile_id: str | None
    implementation_profile_status: str
    runtime_fingerprint: str | None
    languages: tuple[str, ...]
    dtypes: tuple[str, ...]
    operator_tags: tuple[str, ...]
    kernel_types: tuple[str, ...]
    semantic_features: tuple[str, ...]
    shape_signature: Mapping[str, Any]
    current_bottlenecks: tuple[str, ...]
    project_root: Path | None
    artifacts: Mapping[str, ArtifactRef]
    guidance_bindings: Mapping[str, tuple[str, ...]]
    loop_contract_identity: LoopContractIdentity | None

@dataclass(frozen=True)
class RoleQueryRequest:
    text: str
    filters: Mapping[str, tuple[str, ...]]
    scope: str
    group_limits: Mapping[str, int]
    show_excluded: bool

@dataclass(frozen=True)
class AuthoritySnapshot:
    contract_version: int
    loop_contract_identity: LoopContractIdentity
    profile: Mapping[str, Any]
    project_claim: Mapping[str, Any]
    sketch_result: Mapping[str, Any]
    decision_result: Mapping[str, Any]
    artifact_hashes: Mapping[str, str]

@dataclass(frozen=True)
class ValidatedGuidanceBinding:
    guidance_id: str
    sketch_statement_ids: tuple[str, ...]
    permitted_change_family: str
    protected_fields: tuple[str, ...]

@dataclass(frozen=True)
class AdmissionDecision:
    status: str
    reasons: tuple[str, ...]
    match_class: str
    admitted_guidance_ids: tuple[str, ...]
    admitted_example_ids: tuple[str, ...]
    admitted_asset_ids: tuple[str, ...]

@dataclass(frozen=True)
class RoleSearchResult:
    schema_version: int
    context_sha256: str
    loop_contract_identity: LoopContractIdentity | None
    authority_hashes: Mapping[str, str]
    groups: Mapping[str, tuple[Mapping[str, Any], ...]]
```

Required stable result groups:

```text
admitted
conditional
analogy_only
counterexamples
capability_gaps
excluded
```

Required stable exclusion reasons:

```text
audience-mismatch
target-mismatch
profile-missing
profile-version-mismatch
runtime-mismatch
capability-unknown
capability-unsupported
sketch-binding-required
sketch-change-required
version-stale
artifact-designer-only
license-unapproved
source-broken
contract-unsupported
```

---

### Task 1: Versioned Role Contexts and Read-Only `kernel-opt-loop` Authority Bridge

**Files:**
- Create: `skills/kernelwiki/scripts/kernel_opt_bridge.py`
- Create: `skills/kernelwiki/scripts/role_context.py`
- Create: `skills/kernelwiki/tests/test_kernel_opt_bridge.py`
- Create: `skills/kernelwiki/tests/test_role_context.py`
- Create: `skills/kernelwiki/tests/fixtures/role/designer-context.json`
- Create: `skills/kernelwiki/tests/fixtures/role/coder-missing-profile.json`
- Create: `skills/kernelwiki/tests/role_fixture_factory.py`
- Modify: `skills/kernelwiki/data/schemas.yaml`

**Interfaces:**
- Produces: `load_role_context(path)`, `load_authority_snapshot(context)`, `load_loop_module(name)`, `compute_loop_contract_identity()`.
- Consumes current functions without copying schemas: `load_profile`, `validate_project_claim`, `validate_sketch`, and `validate_decision`.

- [ ] **Step 1: Write failing Designer/Coder context tests**

Use canonical JSON context files. The Designer fixture contains no campaign artifacts and must load. The missing-profile Coder fixture must parse structurally but fail authority loading with `profile-missing`.

```python
class RoleContextTests(unittest.TestCase):
    def test_designer_context_requires_no_loop_artifacts(self):
        context = load_role_context(FIXTURES / "role" / "designer-context.json")
        self.assertEqual("designer", context.role)
        self.assertEqual({}, context.artifacts)

    def test_coder_context_requires_frozen_authority_refs(self):
        context = load_role_context(FIXTURES / "role" / "coder-missing-profile.json")
        with self.assertRaisesRegex(KernelWikiError, "profile-missing"):
            load_authority_snapshot(context)
```

Add exact tests `test_non_allowlisted_loop_module_is_denied`, `test_context_cannot_override_loop_root`, `test_validator_hash_mismatch_is_contract_unsupported`, `test_loop_identity_round_trips_in_result`, `test_invalid_role_fails`, `test_artifact_path_escape_fails`, `test_malformed_artifact_sha_fails`, `test_coder_runtime_fingerprint_required`, `test_missing_decision_or_sketch_ref_fails`, `test_unsupported_contract_version_fails`, and `test_artifact_hash_mismatch_fails`.

- [ ] **Step 2: Run the tests and verify failure**

```bash
python3 -m unittest skills/kernelwiki/tests/test_role_context.py skills/kernelwiki/tests/test_kernel_opt_bridge.py -v
```

Expected: missing modules.

- [ ] **Step 3: Implement safe dynamic loading of current validators**

`kernel_opt_bridge.py` loads modules from the sibling skill without modifying `sys.path` globally after initialization:

```python
LOOP_ROOT = Path(__file__).resolve().parents[2] / "kernel-opt-loop"
ALLOWED_MODULES = frozenset({"validate_profile", "validate_sketch", "validate_decision"})
CONSUMED_SCHEMA_FILES: tuple[str, ...] = ()  # Current contract embeds schema checks in validators.


def load_loop_module(name: str) -> ModuleType:
    if name not in ALLOWED_MODULES:
        raise KernelWikiError("contract-module-denied", f"module {name!r} is not allowlisted")
    scripts = require_within(LOOP_ROOT, LOOP_ROOT / "scripts")
    path = require_within(scripts, scripts / f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"kernelwiki_bridge_{name}", path)
    if spec is None or spec.loader is None:
        raise KernelWikiError("contract-module-missing", f"cannot load {name}", path)
    module = importlib.util.module_from_spec(spec)
    scripts_text = str(scripts)
    inserted = scripts_text not in sys.path
    if inserted:
        sys.path.insert(0, scripts_text)
    try:
        spec.loader.exec_module(module)
    finally:
        if inserted:
            sys.path.remove(scripts_text)
    return module
```

`compute_loop_contract_identity()` runs read-only Git commands for `git log -1 --format=%H -- skills/kernel-opt-loop` and `git rev-parse HEAD:skills/kernel-opt-loop`, hashes every allowlisted validator and consumed external schema, and returns `LoopContractIdentity`; the current embedded-schema contract records `schema_sha256: {}`. Coder contexts pin this full identity; any mismatch returns `contract-unsupported`. Tests prove no CLI or context field can select another loop root or module.

- [ ] **Step 4: Implement context parsing and authority validation**

A Designer context requires: `schema_version`, `role`, `target_id`, `implementation_profile_id`, `implementation_profile_status`, `languages`, `dtypes`, `operator_tags`, `kernel_types`, `semantic_features`, `shape_signature`, and `current_bottlenecks`. A normal Coder context also requires `contract_version`, runtime/artifact authority, and pinned loop identity. One fail-closed conditional is schema-valid: when `implementation_profile_status == "missing"`, Coder may set `runtime_fingerprint`, `project_root`, and `loop_contract_identity` to null and `artifacts`/`guidance_bindings` to empty mappings; `load_authority_snapshot` returns `profile-missing` before importing any validator or inspecting fallback paths. No other Coder context may omit authority.

A Coder context additionally requires a complete materialized test project. Implement `role_fixture_factory.py` by copying the same current vNext fixtures used by `skills/kernel-opt-loop/tests/test_validate_decision.py::materialize_vnext_project`, replacing Decision hash markers, writing `state/runtime-snapshot.json` as `{"target_id":"mlu590","implementation_profile_id":"triton_mlu","triton_version":"3.6.0","device_arch":"mlu-arch"}`, and constructing the context with computed hashes:

```python
def build_coder_context(project_root: Path) -> dict[str, Any]:
    refs = {
        "profile": project_root / "state" / "implementation_profile_snapshot" / "profile.yaml",
        "runtime_snapshot": project_root / "state" / "runtime-snapshot.json",
        "project_claim": project_root / "state" / "project_capability_claim.json",
        "sketch": project_root / "rounds" / "sketch_001.json",
        "decision": project_root / "rounds" / "decision_001.md",
    }
    return {
        "schema_version": 1,
        "role": "coder",
        "contract_version": 3,
        "target_id": "mlu590",
        "implementation_profile_id": "triton_mlu",
        "implementation_profile_status": "partial",
        "runtime_fingerprint": "triton 3.6.0 / CoreX 4.4.0",
        "languages": ["triton"],
        "dtypes": ["fp32"],
        "operator_tags": ["topk", "selection"],
        "kernel_types": ["topk", "reduction"],
        "semantic_features": ["left-tie-breaking"],
        "shape_signature": {"T": 83, "E": 256, "K": 8},
        "current_bottlenecks": ["reduction"],
        "project_root": str(project_root),
        "artifacts": {
            name: {"path": path.relative_to(project_root).as_posix(), "sha256": sha256_file(path)}
            for name, path in refs.items()
        },
        "guidance_bindings": {"guidance-test-exact": ["op.load.row"]},
        "loop_contract_identity": asdict(compute_loop_contract_identity()),
    }
```

This is a clearly test-only MLU authority fixture and must never be registered as an Ascend profile. `load_authority_snapshot` verifies every hash, then calls current `load_profile`, `validate_project_claim`, `validate_sketch`, and `validate_decision`; it returns normalized outputs and exact hashes.

- [ ] **Step 5: Add explicit real Ascend-missing regression**

Point a Coder context at `implementation_profile_id: ascendc` with no profile artifact. At this task boundary assert only `profile-missing` and no attempt to open `triton_ascend.md`, CUDA profiles, or any cross-backend profile/module. Move grouped empty-result assertions to Task 3 after `role_search` exists.

- [ ] **Step 6: Run tests and commit**

```bash
python3 -m unittest skills/kernelwiki/tests/test_role_context.py skills/kernelwiki/tests/test_kernel_opt_bridge.py -v
git add skills/kernelwiki/scripts/kernel_opt_bridge.py skills/kernelwiki/scripts/role_context.py skills/kernelwiki/tests/role_fixture_factory.py skills/kernelwiki/tests/test_kernel_opt_bridge.py skills/kernelwiki/tests/test_role_context.py skills/kernelwiki/tests/fixtures/role skills/kernelwiki/data/schemas.yaml
git commit -m "feat(kernelwiki): validate role query authority"
```

---

### Task 2: Card, Guidance, Example, and Asset Admission Engine

**Files:**
- Create: `skills/kernelwiki/scripts/admission.py`
- Create: `skills/kernelwiki/tests/test_admission.py`
- Create: `skills/kernelwiki/tests/fixtures/cards/analogy-designer-card.md`
- Create: `skills/kernelwiki/tests/fixtures/cards/mixed-asset-card.md`
- Create: `skills/kernelwiki/tests/fixtures/sources/source-exact-coder.md`
- Create: `skills/kernelwiki/tests/fixtures/sources/source-analogy-only.md`
- Modify: `skills/kernelwiki/scripts/corpus.py`
- Modify: `skills/kernelwiki/scripts/validate.py`
- Modify: `skills/kernelwiki/references/schema.md`

**Interfaces:**
- Produces: `admit_card(card, context, authority)`, `admit_source(source, context, authority)`, `admit_candidate(candidate: SearchCandidate, context, authority)`, `admit_asset(card, asset_id, context, authority)`, `classify_designer_match(card, context)`, and `validate_guidance_binding`.
- Admission returns `AdmissionDecision`; it never changes profile, claim, Sketch, Decision, Card, or provenance data.

- [ ] **Step 1: Write failing rule-table tests**

Use subtests for every stable reason:

```python
class AdmissionTests(unittest.TestCase):
    def test_designer_sees_analogy_with_explicit_class(self):
        decision = admit_card(analogy_card(), designer_context(), None)
        self.assertEqual("analogy_only", decision.status)
        self.assertEqual("analogy-only", decision.match_class)

    def test_coder_rejects_exact_page_without_statement_binding(self):
        decision = admit_card(exact_card(), coder_context(bindings={}), valid_authority())
        self.assertEqual("excluded", decision.status)
        self.assertIn("sketch-binding-required", decision.reasons)

    def test_page_admission_does_not_admit_designer_only_asset(self):
        page = admit_card(mixed_asset_card(), valid_coder_context(), valid_authority())
        asset = admit_asset(mixed_asset_card(), "asset-full-kernel", valid_coder_context(), valid_authority())
        self.assertEqual("admitted", page.status)
        self.assertIn("artifact-designer-only", asset.reasons)
```

Define `analogy_card()` and `mixed_asset_card()` by parsing the named Markdown fixtures. Define `exact_coder_card()` by asking `role_fixture_factory.py` to write a temporary Card containing `build_exact_guidance(sketch_result, decision_result)`. Define `valid_coder_context()` and `valid_authority()` by calling `materialize_vnext_project`, `build_coder_context`, `load_role_context`, and `load_authority_snapshot` inside one `TemporaryDirectory`. Add methods `test_target_family_backend_unknown_classification`, `test_exact_target_mismatch_excludes_coder`, `test_profile_version_mismatch_excludes_coder`, `test_runtime_mismatch_excludes_coder`, `test_unknown_or_unsupported_capability_excludes_coder`, `test_stale_version_excludes_coder`, `test_unapproved_license_denies_asset`, `test_broken_source_excludes_candidate`, `test_forbidden_sketch_change_is_rejected`, and `test_approved_snippet_is_exposed_only_after_asset_admission`.

- [ ] **Step 2: Run tests and verify failure**

```bash
python3 -m unittest skills/kernelwiki/tests/test_admission.py -v
```

Expected: missing `admission` module.

- [ ] **Step 3: Add optional Coder metadata without widening generic Cards**

Extend Card validation with optional `coder_access`. Build the exact-profile test Card dynamically after validating the current Sketch/Decision:

```python
PROTECTED_FIELDS = ("algorithm", "dataflow", "precision", "effects", "aliases", "host-plan", "public-interface")
ALLOWED_CHANGE_FAMILIES = frozenset({"implementation-spelling", "loop-structure-preserving", "memory-access-spelling"})


def protected_projection(sketch_result: Mapping[str, Any], decision_result: Mapping[str, Any]) -> Mapping[str, Any]:
    sketch = sketch_result["sketch"]
    return {
        "algorithm": {"scope_kind": sketch["scope"]["kind"], "operation_kinds": [item["kind"] for item in sketch["operations"]], "causal_nodes": sketch["causal_nodes"]},
        "dataflow": {"declarations": sketch["declarations"], "operations": [{key: item.get(key) for key in ("id", "inputs", "outputs", "index_domain", "mask")} for item in sketch["operations"]], "control": sketch["control"]},
        "precision": [{"id": item["id"], "dtype": item["dtype"]} for item in sketch["declarations"]],
        "effects": {"top": sketch["effects"], "operations": [{"id": item["id"], "effects": item["effects"]} for item in sketch["operations"]]},
        "aliases": sketch["effects"]["aliases"],
        "host-plan": decision_result["host_plan"],
        "public-interface": {"entrypoints": sketch["scope"]["entrypoints"], "unchanged_boundary": sketch["scope"]["unchanged_boundary"]},
    }


def build_exact_guidance(sketch_result: Mapping[str, Any], decision_result: Mapping[str, Any]) -> Mapping[str, Any]:
    projection_sha = sha256_bytes(canonical_json_bytes(protected_projection(sketch_result, decision_result)))
    return {
        "id": "guidance-test-exact",
        "implementation_profile_ids": ["triton_mlu"],
        "target_ids": ["mlu590"],
        "runtime_fingerprints": ["triton 3.6.0 / CoreX 4.4.0"],
        "languages": ["triton"],
        "dtypes": ["fp32"],
        "shape_constraints": {"T": {"min": 1, "max": 4096}, "E": {"exact": 256}, "K": {"exact": 8}},
        "required_capabilities": ["memory.load.contiguous-fp32"],
        "preserves": list(PROTECTED_FIELDS),
        "implementation_delta": {
            "statement_ids": ["op.load.row"],
            "change_family": "memory-access-spelling",
            "protected_projection_sha256": projection_sha,
            "changed_protected_fields": [],
        },
        "eligible_example_ids": ["example-test-exact"],
        "eligible_asset_ids": ["asset-short-snippet"],
        "version_claim_ids": [],
    }
```

This block is absent from initial Ascend Cards. Validation requires exact IDs, all seven protected semantic fields, known example/asset IDs, and valid version references. It does not make a generic page or all its assets Coder-eligible.

Implement exact binding validation:

```python
def validate_guidance_binding(
    guidance: Mapping[str, Any],
    binding_ids: Sequence[str],
    sketch_result: Mapping[str, Any],
    decision_result: Mapping[str, Any],
) -> tuple[ValidatedGuidanceBinding, ...]:
    delta = guidance["implementation_delta"]
    declared_ids = tuple(sorted(delta["statement_ids"]))
    context_ids = tuple(sorted(binding_ids))
    if not context_ids or context_ids != declared_ids:
        raise KernelWikiError("sketch-binding-required", "context binding must equal guidance statement_ids")
    if tuple(guidance["preserves"]) != PROTECTED_FIELDS:
        raise KernelWikiError("sketch-change-required", "guidance must preserve all protected fields")
    if delta["change_family"] not in ALLOWED_CHANGE_FAMILIES:
        raise KernelWikiError("sketch-change-required", "change family is not implementation-only")
    if delta["changed_protected_fields"]:
        raise KernelWikiError("sketch-change-required", "protected semantic fields may not change")
    for statement_id in declared_ids:
        if not statement_id.startswith(("op.", "ctrl.", "guard.")) or statement_id not in sketch_result["statement_index"]:
            raise KernelWikiError("sketch-binding-required", f"unknown Sketch statement {statement_id}")
    actual_sha = sha256_bytes(canonical_json_bytes(protected_projection(sketch_result, decision_result)))
    if delta["protected_projection_sha256"] != actual_sha:
        raise KernelWikiError("sketch-change-required", "protected semantic projection changed")
    return (
        ValidatedGuidanceBinding(
            guidance_id=guidance["id"],
            sketch_statement_ids=declared_ids,
            permitted_change_family=delta["change_family"],
            protected_fields=PROTECTED_FIELDS,
        ),
    )
```

Add literal tests `test_binding_ids_required`, `test_unknown_binding_id_fails`, `test_partial_binding_fails`, `test_changed_precision_or_dataflow_fails`, `test_projection_hash_mismatch_fails`, and `test_valid_memory_access_spelling_binding`.

- [ ] **Step 4: Implement Designer classification**

Use this order: exact target, device family, backend, analogy-only, unknown. Counterexample and capability-gap are orthogonal result roles, not weaker target matches. Designer never promotes analogy or unknown target disposition into capability authority.

- [ ] **Step 5: Implement Source and fail-closed Coder admission**

`admit_source` applies Source target disposition, repository/version state, provenance, license, and allowed-audience metadata. Designer may receive Source body/excerpts with explicit match class. Coder receives Source metadata only when the Source frontmatter itself declares exact target/profile/runtime, `audiences` contains `coder`, and license/provenance checks pass; Source body, snippet, and asset exposure still require separate admission. Add tests `test_designer_admits_backend_source`, `test_coder_rejects_analogy_source`, `test_coder_metadata_does_not_expose_code`, and `test_broken_source_is_excluded`.

`match_dtype_shape_regime` requires every context dtype to appear in guidance `dtypes`. Each declared shape constraint is either `{exact: value}` or `{min: value, max: value}` and is evaluated against the same named `shape_signature` field. Missing or out-of-range fields exclude Coder with `target-mismatch`; Designer retains the result as `conditional`. Runtime matching is byte-exact after trimming surrounding whitespace.

Coder admission checks, in order:

```text
validated authority present
Card audiences contains coder
one guidance item matches exact profile/target/runtime/language/dtype/shape regime
context implementation_profile_status equals validated profile status; partial is allowed only when every referenced capability is proven
all required capabilities are supported or constrained, never unknown/unsupported/prohibited
guidance preserves all protected fields
context binds that guidance ID to existing frozen Sketch statement IDs
all version claims current
each selected example/asset separately passes target/profile/runtime/license/provenance/mode checks
```

Return every applicable exclusion reason in stable sorted order. Do not stop at the first reason in library output; the CLI may show a concise primary reason plus the complete list.

- [ ] **Step 6: Run tests and commit**

```bash
python3 -m unittest skills/kernelwiki/tests/test_admission.py skills/kernelwiki/tests/test_corpus.py skills/kernelwiki/tests/test_provenance.py -v
git add skills/kernelwiki/scripts/admission.py skills/kernelwiki/scripts/corpus.py skills/kernelwiki/scripts/validate.py skills/kernelwiki/tests/test_admission.py skills/kernelwiki/tests/fixtures/cards skills/kernelwiki/tests/fixtures/sources skills/kernelwiki/references/schema.md
git commit -m "feat(kernelwiki): admit role-scoped knowledge"
```

---

### Task 3: Admission-First Grouped Search and Safe Page Retrieval

**Files:**
- Create: `skills/kernelwiki/scripts/role_search.py`
- Create: `skills/kernelwiki/tests/test_role_search.py`
- Modify: `skills/kernelwiki/scripts/query.py`
- Modify: `skills/kernelwiki/scripts/get_page.py`
- Modify: `skills/kernelwiki/scripts/search.py`

**Interfaces:**
- Produces: `role_search(corpus: Corpus, request: RoleQueryRequest, context: RoleQueryContext, authority: AuthoritySnapshot | None) -> RoleSearchResult`, `role_get_page(corpus: Corpus, record_id: str, context: RoleQueryContext, authority: AuthoritySnapshot | None, *, follow_sources: bool, access: str) -> Mapping[str, Any]`, and `rank_role_candidates(candidates: Sequence[tuple[SearchCandidate, AdmissionDecision]], context: RoleQueryContext, request: RoleQueryRequest) -> tuple[SearchCandidate, ...]`.
- Uses the neutral lexical score as only one late component after admission and grouping.

- [ ] **Step 1: Write failing ordering/grouping tests**

```python
def test_admission_happens_before_limit(self):
    corpus = many_high_scoring_ineligible_cards_plus_one_exact_card()
    result = role_search(corpus, request(limit=1), valid_coder_context(), valid_authority())
    self.assertEqual("exact-eligible-card", result.groups["admitted"][0]["id"])


def test_counterexamples_and_gaps_have_independent_limits(self):
    result = role_search(mixed_corpus(), designer_request(limit=1), designer_context(), None)
    self.assertEqual(1, len(result.groups["admitted"]))
    self.assertEqual(1, len(result.groups["counterexamples"]))
    self.assertEqual(1, len(result.groups["capability_gaps"]))
```

Implement `many_high_scoring_ineligible_cards_plus_one_exact_card()` and `mixed_corpus()` in `test_role_search.py` by cloning the core temporary corpus fixture, writing five high-scoring target-mismatch Cards plus one exact Card, and writing one positive, one counterexample, and one capability-gap Card. Reuse the valid context/authority helpers from `role_fixture_factory.py`. Add methods `test_group_order_is_deterministic`, `test_show_excluded_preserves_reasons`, `test_source_results_use_source_admission`, `test_analogy_only_is_separate`, `test_missing_profile_coder_result_is_schema_valid_empty`, and `test_role_result_json_is_byte_identical`.

- [ ] **Step 2: Run tests and verify failure**

```bash
python3 -m unittest skills/kernelwiki/tests/test_role_search.py -v
```

Expected: missing `role_search` module.

- [ ] **Step 3: Implement role search pipeline**

`rank_role_candidates` sorts by this exact tuple, descending for numeric components and ascending for the final stable path/ID strings:

```python
(
    target_specificity,          # exact=4, family=3, backend=2, analogy-only=1, unknown=0
    profile_runtime_exactness,   # both exact=2, one exact=1, neither/missing=0
    kernel_type_overlap_count,
    dtype_overlap_count,
    semantic_feature_overlap_count + normalized_shape_regime_score,
    evidence_rank,               # local=5, official+code=4, source=3, inferred=2, experimental=1
    reproduction_rank,           # benchmarked=5, runnable=4, snippet=3, pseudocode=2, concept=1
    freshness_ordinal,           # parsed checked-in last_verified_at/captured_at date; missing=0
    neutral_lexical_score,
    stable_path,
    stable_id,
)
```

`dtype_overlap_count` is the set intersection between context dtypes and the matched guidance/observation/example dtype, with missing dtype scoring `0`. `normalized_shape_regime_score` uses matched guidance `shape_constraints` or matched example `shape`: exact equality for every shared named dimension scores `2`, range-compatible guidance scores `1`, missing shape metadata scores `0`, and any conflict scores `-1` for Designer while Coder was already excluded by admission. General Card body prose never contributes shape or dtype authority.

For a Card, evidence/reproduction/freshness come only from admitted matched observations/examples, never the strongest unrelated item on the page. Dates use checked-in ISO `YYYY-MM-DD` values; filesystem mtime and current time are forbidden. Add one focused ordering test for each component plus missing-dtype, missing-shape, conflict-shape, and stable-ID tie tests.

`role_get_page` first builds the core `SearchCandidate`, runs `admit_candidate`, and only then calls core `retrieve_page`. Its canonical mapping contains exactly `schema_version`, `context_sha256`, `loop_contract_identity`, `authority_hashes`, `admission` (all `AdmissionDecision` fields), and `page` (the serialized core `PageResult`). Designer may request `metadata|approved-assets`; Coder `approved-assets` is filtered again through `admit_asset`, and excluded assets are returned only as metadata with reasons.

The exact pipeline is:

```python
def role_search(corpus, request, context, authority):
    candidates = collect_unlimited_candidates(corpus, request)
    decisions = [(candidate, admit_candidate(candidate, context, authority)) for candidate in candidates]
    groups = group_by_decision(decisions)
    ranked = {name: rank_candidates(items, request) for name, items in groups.items()}
    limited = apply_per_group_limits(ranked, request.group_limits)
    return build_role_result(context, authority, limited)
```

`collect_unlimited_candidates` may use cheap exact field preselection but cannot apply lexical top-N truncation before admission. `excluded` defaults to metadata-only records with reasons; `--show-excluded` controls whether they appear in CLI output.

- [ ] **Step 4: Extend `query.py` with role contexts**

Support:

```bash
python3 skills/kernelwiki/scripts/query.py \
  "topk reduction" \
  --context skills/kernelwiki/tests/fixtures/role/designer-context.json \
  --group-limit admitted=12 \
  --group-limit counterexamples=8 \
  --group-limit capability_gaps=8

python3 skills/kernelwiki/scripts/query.py \
  "ascendc implementation spelling" \
  --context skills/kernelwiki/tests/fixtures/role/coder-missing-profile.json \
  --show-excluded
```

Without `--context`, retain Phase A/B role-neutral schema-v1 output. With `--context`, emit role-query-result schema `1`, exact context/artifact hashes, grouped results, and no persisted file. Users who need a receipt redirect stdout or use `--output` to an explicit path outside active campaign state.

- [ ] **Step 5: Extend `get_page.py` with independent item/asset admission**

When `--context` is supplied, page retrieval first admits the page, then separately evaluates requested `--example`, `--guidance`, and `--asset` IDs. A denied item is omitted from the returned body/excerpts and appears under `denied_items` with stable reasons. `--include-code` alone never bypasses asset admission.

- [ ] **Step 6: Run tests and commit**

```bash
python3 -m unittest skills/kernelwiki/tests/test_role_search.py skills/kernelwiki/tests/test_admission.py skills/kernelwiki/tests/test_search.py -v
git add skills/kernelwiki/scripts/role_search.py skills/kernelwiki/scripts/query.py skills/kernelwiki/scripts/get_page.py skills/kernelwiki/scripts/search.py skills/kernelwiki/tests/test_role_search.py
git commit -m "feat(kernelwiki): add admission-first role search"
```

---

### Task 4: Version Claims, Capability Gaps, and Exact-Profile Empty Behavior

**Files:**
- Modify: `skills/kernelwiki/data/version-claims.yaml`
- Modify: `skills/kernelwiki/scripts/admission.py`
- Modify: `skills/kernelwiki/scripts/validate.py`
- Modify: `skills/kernelwiki/tests/test_admission.py`
- Modify: `skills/kernelwiki/tests/test_role_search.py`

**Interfaces:**
- Produces: `resolve_version_claim`, `resolve_capability_status`, and deterministic gap records.

- [ ] **Step 1: Write failing version/gap tests**

Test current, stale, unknown-version, Unknown capability, Unsupported capability, and missing profile. Require different reasons:

```python
self.assertEqual("version-stale", stale_decision.reasons[0])
self.assertEqual("capability-unknown", unknown_decision.reasons[0])
self.assertEqual("capability-unsupported", unsupported_decision.reasons[0])
self.assertEqual("profile-missing", missing_profile_decision.reasons[0])
```

Unknown must never be rewritten to Unsupported.

- [ ] **Step 2: Run focused tests and verify failure**

```bash
python3 -m unittest skills/kernelwiki/tests/test_admission.py skills/kernelwiki/tests/test_role_search.py -v
```

Expected: assertions fail because version/gap resolution is absent.

- [ ] **Step 3: Implement version-claim resolution**

Each version claim contains `id`, `subject`, `supported_versions`, `last_verified_at`, `source_ids`, `status: current|stale|unknown`, and `replacement_claim_id`. Validation requires bidirectional replacement links and resolved Sources. Coder admits only `current`; Designer receives stale/unknown as `conditional` with the claim ID and reasons.

- [ ] **Step 4: Implement capability-gap result generation**

When exact Coder admission fails because profile/capability authority is missing or Unknown, search for matching `pattern`/`language`/`runtime` Cards tagged `capability-gap`. Return those under `capability_gaps` with Designer-readable metadata only. Never return a guidance item, snippet, or cross-backend recipe as the gap substitute.

- [ ] **Step 5: Prove real AscendC remains empty**

Run:

```bash
python3 skills/kernelwiki/scripts/query.py \
  "ascendc topk implementation" \
  --context skills/kernelwiki/tests/fixtures/role/coder-missing-profile.json
```

Expected: exit `0`; `admitted` is empty; `capability_gaps` is present when matching generic gap Cards exist; no result contains `triton`, `cuda`, or an implementation asset.

- [ ] **Step 6: Run tests and commit**

```bash
python3 -m unittest skills/kernelwiki/tests/test_admission.py skills/kernelwiki/tests/test_role_search.py -v
git add skills/kernelwiki/data/version-claims.yaml skills/kernelwiki/scripts/admission.py skills/kernelwiki/scripts/validate.py skills/kernelwiki/tests/test_admission.py skills/kernelwiki/tests/test_role_search.py
git commit -m "feat(kernelwiki): preserve version and capability gaps"
```

---

### Task 5: Track 2 Development Contexts and Adversarial Isolation Tests

**Files:**
- Create: `skills/kernelwiki/data/track2-development-queries.yaml`
- Create: `skills/kernelwiki/tests/fixtures/track2/sparse-attn-development.json`
- Create: `skills/kernelwiki/tests/fixtures/track2/index-topk-development.json`
- Create: `skills/kernelwiki/tests/fixtures/track2/adversarial-dot-scope.json`
- Create: `skills/kernelwiki/tests/fixtures/track2/adversarial-output-reuse.json`
- Create: `skills/kernelwiki/tests/fixtures/track2/adversarial-device-wall.json`
- Create: `skills/kernelwiki/tests/fixtures/track2/adversarial-topk-transfer.json`
- Create: `skills/kernelwiki/tests/fixtures/track2/adversarial-profiler-evidence.json`
- Create: `skills/kernelwiki/scripts/evaluate_holdout.py`
- Create: `skills/kernelwiki/tests/test_role_contracts.py`
- Modify: `skills/kernelwiki/references/evaluation-protocol.md`

**Interfaces:**
- Produces: `evaluate_queries(corpus, cases) -> Mapping[str, Any]` and the final holdout CLI.
- Consumes the sealed `sinkhorn_normalize` gold fixture from the standalone-core plan without modifying it.

- [ ] **Step 1: Encode development contexts without authoring operator Cards**

`track2-development-queries.yaml` contains `sparse_attn` and `index_topk` structured contexts only. Each records target, language, kernel types, semantic features, dtypes, shape signature, and expected general knowledge categories. It contains no source code, recipe, or Card body.

- [ ] **Step 2: Write failing adversarial tests**

Require:

```text
generic tl.dot evidence does not satisfy dtype/shape-specific capability
positive output reuse does not hide a conflicting counterexample
device-time improvement does not imply wall-time improvement
grouped-top-k evidence remains bounded when querying index-top-k
raw torch profiler evidence does not become CANN device attribution
```

For every case, assert unsafe Coder admissions `0`, Unknown promotions `0`, and cross-target recipe leaks `0`.

- [ ] **Step 3: Run adversarial tests and verify failure**

```bash
python3 -m unittest skills/kernelwiki/tests/test_role_contracts.py -v
```

Expected: evaluator or fixture loader is missing.

- [ ] **Step 4: Implement deterministic evaluation**

`evaluate_queries` runs admission-first search and reports:

```json
{
  "schema_version": 1,
  "case_count": 5,
  "unsafe_coder_admissions": 0,
  "unknown_promotions": 0,
  "cross_target_recipe_leaks": 0,
  "capability_gap_recall": 1.0,
  "top5_relevant_card_recall": 1.0,
  "counterexample_recall": 1.0
}
```

Values are computed, not hardcoded. `top5_relevant_card_recall` is the number of sealed `gold.relevant_card_ids` in the first five admitted Designer Card IDs divided by `top5_relevant_denominator=4`; counterexample and capability-gap recall use their explicit one-ID lists and denominators. Safety metrics count actual forbidden admissions/promotions/leaks across the Designer and missing-profile Coder contexts. The final holdout command reads the sealed fixture only after implementation is complete:

```bash
python3 skills/kernelwiki/scripts/evaluate_holdout.py \
  --manifest skills/kernelwiki/data/evaluation-holdouts.yaml \
  --gold skills/kernelwiki/tests/fixtures/holdout/track2-sinkhorn-gold.yaml
```

It first verifies the sealed SHA-256 and exits `2` on mismatch.

- [ ] **Step 5: Meet safety gates without tuning the holdout**

If safety metrics fail, fix admission logic. If retrieval metrics fail, do not edit the holdout, aliases, or taxonomy based on hidden answer details. Record the failure and open a separate source-backed curation change; rerun only after that change has independent review.

- [ ] **Step 6: Commit Track 2 evaluation**

```bash
git add skills/kernelwiki/data/track2-development-queries.yaml skills/kernelwiki/tests/fixtures/track2 skills/kernelwiki/scripts/evaluate_holdout.py skills/kernelwiki/tests/test_role_contracts.py skills/kernelwiki/references/evaluation-protocol.md
git commit -m "test(kernelwiki): gate role query isolation"
```

---

### Task 6: Role-Query Contracts, Full Regression, and Documentation

**Files:**
- Create: `skills/kernelwiki/references/role-query-contract.md`
- Modify: `skills/kernelwiki/SKILL.md`
- Modify: `skills/kernelwiki/README.md`
- Modify: `skills/kernelwiki/index.md`
- Modify: `skills/kernelwiki/tests/test_contracts.py`

**Interfaces:**
- Finalizes role-query schema and documented CLI behavior.

- [ ] **Step 1: Add contract tests for forbidden integration artifacts**

Assert that Phase C creates none of:

```text
validate_consultation.py
rounds/kernelwiki_consultation_*.json
coder_result schema changes
Designer/Coder prompt edits
kernel-opt-loop file changes
KnowledgePacket or required dossier paths
```

Also assert role-query contexts/results are versioned, context/artifact hashes are present, and Coder result items include explicit guidance/Sketch bindings.

- [ ] **Step 2: Run the full KernelWiki suite**

```bash
python3 -m unittest discover -s skills/kernelwiki/tests -p 'test_*.py' -v
```

Expected: all standalone-core and role-aware tests pass without accelerator or network access.

- [ ] **Step 3: Run production validation and smoke queries**

```bash
python3 skills/kernelwiki/scripts/validate.py
python3 skills/kernelwiki/scripts/generate_indices.py --check
python3 skills/kernelwiki/scripts/query.py "ascend launch" --context skills/kernelwiki/tests/fixtures/role/designer-context.json
python3 skills/kernelwiki/scripts/query.py "ascendc implementation" --context skills/kernelwiki/tests/fixtures/role/coder-missing-profile.json
```

Expected: Designer query exposes match classes; real AscendC Coder query is exact-profile empty with visible gap reasons.

- [ ] **Step 4: Document active access and non-authority**

`role-query-contract.md` documents context fields, bridge validation, admission ordering, stable groups/reasons, Card-versus-asset admission, exact-profile no-fallback behavior, output limits, and explicit `--output` receipts. It states that a receipt is neither a Decision nor a campaign artifact and does not modify role prompts.

Update `SKILL.md` and `README.md` with Designer/Coder examples and a warning that missing canonical AscendC authority is expected to return no implementation guidance.

- [ ] **Step 5: Commit and verify cleanliness**

```bash
git add skills/kernelwiki/references/role-query-contract.md skills/kernelwiki/SKILL.md skills/kernelwiki/README.md skills/kernelwiki/index.md skills/kernelwiki/tests/test_contracts.py
git commit -m "docs(kernelwiki): document role-aware query admission"
git status --short
```

Expected: clean worktree. Phase D may execute independently after the standalone-core plan; it must not depend on this admission engine.

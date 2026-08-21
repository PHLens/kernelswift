# backend-probe Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independent `backend-probe` skill that discovers backend capabilities, emits per-family machine-readable evidence artifacts, produces a Markdown promotion note, and ends cleanly without requiring a full optimization campaign.

**Architecture:** Keep `backend-probe` separate from `kernel-opt-loop` and model it as a pre-campaign fact-discovery workflow. Use standard-library Python validators and route-recommendation helpers around per-family JSON artifacts stored under a run-local probe directory. Preserve canonical backend profiles as reviewed contracts by emitting promotion notes and optional recommendation metadata instead of mutating profiles directly.

**Tech Stack:** Python 3 standard library (`json`, `hashlib`, `pathlib`, `subprocess`, `tempfile`, `unittest`), Markdown contract files, JSON artifacts, Git.

**Spec:** `docs/superpowers/specs/2026-08-21-backend-probe-skill-design.md`

## Global Constraints

- Implement `backend-probe` as an independent skill under `skills/backend-probe/`; do not hide it inside `kernel-opt-loop`.
- Do not generate candidate kernels, benchmark rankings, accepted implementations, round reports, or any `kernel-opt-loop` campaign state.
- Emit one machine-readable file per probe family and one separate Markdown promotion note.
- Store probe artifacts as project-local or run-local observations; do not mutate canonical backend profiles directly.
- Promotion remains explicit: the implementation may emit recommendation metadata, but canonical profile changes require later human approval.
- Support the six v1 families from the spec only: `dot`, `launch`, `profiler`, `tuning`, `core-primitives`, and `reductions-mixed-precision`.
- Route recommendations must distinguish at least `fusion-first`, `launch-first`, `hybrid-first`, and `not-ready-for-gemm`.
- Tests must run without accelerator hardware. Use fixtures and stubbed command output rather than live probes.
- Do not introduce a second mutable workflow state store, a daemon, or network calls.
- `kernel-opt-loop` may later consume probe artifacts, but this implementation must not require vNext semantic-attribution machinery to exist first.

---

## File Structure

| Path | Responsibility |
|---|---|
| `skills/backend-probe/SKILL.md` | Runtime-neutral contract for running backend capability probes and terminating without a full optimization campaign. |
| `skills/backend-probe/README.md` | Human-facing overview of the skill, its families, artifact layout, and relationship to `kernel-opt-loop`. |
| `skills/backend-probe/scripts/backend_probe_common.py` | Shared JSON loading, SHA-256, reference, and run-directory helpers for the skill. |
| `skills/backend-probe/scripts/validate_family_result.py` | Deterministic validator for one family artifact (`dot.json`, `launch.json`, etc.). |
| `skills/backend-probe/scripts/recommend_route.py` | Deterministic route recommendation and end-state classifier from validated family artifacts. |
| `skills/backend-probe/scripts/render_promotion_note.py` | Render a Markdown promotion note from validated family artifacts and recommendation metadata. |
| `skills/backend-probe/references/family-result-template.json` | Template shape for machine-readable family artifacts. |
| `skills/backend-probe/references/promotion-note-template.md` | Template for the Markdown promotion note. |
| `skills/backend-probe/bundles/triton-common-v1.json` | Canonical definition of the six v1 probe families. |
| `skills/backend-probe/targets/triton_mlu.json` | MLU runtime/bootstrap metadata and the families enabled for MLU probes. |
| `skills/backend-probe/targets/triton_gcu.json` | GCU runtime/bootstrap metadata and family selection. |
| `skills/backend-probe/targets/triton_maca.json` | MACA runtime/bootstrap metadata and family selection. |
| `skills/backend-probe/targets/triton_cuda.json` | BI150/CoreX runtime/bootstrap metadata and family selection. |
| `skills/backend-probe/targets/triton_ascend.json` | Ascend runtime/bootstrap metadata and family selection. |
| `skills/backend-probe/tests/test_backend_probe_common.py` | Unit tests for common helpers and run-local path handling. |
| `skills/backend-probe/tests/test_validate_family_result.py` | Family artifact validation tests. |
| `skills/backend-probe/tests/test_recommend_route.py` | Route recommendation and end-state classification tests. |
| `skills/backend-probe/tests/test_contracts.py` | Cross-file contract checks for bundles, targets, docs, and templates. |
| `skills/backend-probe/tests/fixtures/families/` | Valid and invalid family result fixtures. |
| `skills/backend-probe/tests/fixtures/recommendation/` | Multi-family fixture sets for route recommendation tests. |
| `skills/backend-probe/tests/fixtures/notes/` | Promotion-note rendering inputs and golden outputs. |
| `skills/README.md` | Adds `backend-probe` to the repo skill catalog. |
| `README.md` | Documents `backend-probe` as the pre-campaign backend capability step. |
| `docs/backend-registry.md` | Notes that canonical backend profiles may be informed by `backend-probe` promotion notes. |
| `docs/competition/track1-triton.md` | Describes `backend-probe` as an optional backend onboarding step before campaign creation. |
| `skills/kernel-opt-loop/SKILL.md` | Adds a concise note that existing probe artifacts may be consumed in Phase 0 but are optional. |

No JSON Schema registry is added in this first implementation; the spec explicitly leaves that for a later phase.

## Acceptance Criteria

- AC-1: `backend-probe` exists as an independent skill and does not allocate optimization rounds or candidate deliverables.
- AC-2: A probe run can terminate successfully without starting `kernel-opt-loop` and can classify at least `ready-for-campaign`, `not-ready-yet`, `not-worth-pursuing-now`, and `promotion-only`.
- AC-3: Each v1 family artifact validates independently and records runtime identity, scope, command/evidence, and an observation bounded to that scope.
- AC-4: A promotion note is emitted separately from machine artifacts and never mutates canonical backend profiles.
- AC-5: Route recommendation is deterministic from validated family facts and can emit at least `fusion-first`, `launch-first`, `hybrid-first`, and `not-ready-for-gemm`.
- AC-6: Target and bundle documents accurately represent the five current Triton backend targets and the six v1 probe families.
- AC-7: `kernel-opt-loop` documentation may refer to existing probe artifacts as optional prior facts, but no vNext dependency is introduced.
- AC-8: All tests pass without accelerator access.

## Path Boundaries

### Upper Bound

Implement a complete first-pass `backend-probe` skill with validators, route recommendation, promotion-note rendering, target descriptors for the current Triton backends, documentation integration, and tests.

### Lower Bound

At minimum, provide:

- the skill contract;
- per-family artifact validation;
- deterministic route recommendation;
- Markdown promotion-note rendering;
- one bundle document covering the six v1 families;
- one target document per current Triton backend; and
- cross-file tests proving docs and target lists stay aligned.

### Allowed Choices

- Use JSON documents plus Python validators rather than adding a schema library.
- Keep route recommendation rule-based rather than model-based.
- Treat family artifacts as read-only evidence objects with exact per-family semantics.
- Keep target descriptors narrow and factual; do not overstate capabilities beyond current repo evidence.

### Prohibited Choices

- Do not auto-edit `skills/kernel-opt-loop/prompts/coder_targets/*.md` or canonical backend profiles from the skill itself.
- Do not create candidate kernels, benchmark ranking calculators, or full campaign manifests.
- Do not add network calls, background schedulers, or a second global mutable artifact ledger.
- Do not depend on hardware execution in tests.

## Dependencies and Sequence

1. Common helpers and fixtures before validators.
2. Family validation before route recommendation.
3. Route recommendation before promotion-note rendering.
4. Core scripts before the skill contract and repo docs.
5. Cross-file contract checks and full regression last.

## Implementation Tasks

### Task 1: Add backend-probe scaffolding and common helpers

**Files:**
- Create: `skills/backend-probe/SKILL.md`
- Create: `skills/backend-probe/README.md`
- Create: `skills/backend-probe/scripts/backend_probe_common.py`
- Create: `skills/backend-probe/tests/test_backend_probe_common.py`
- Create: `skills/backend-probe/tests/fixtures/.gitkeep`

**Interfaces:**
- Produces `ProbeContractError(code: str, message: str, path: Path | None = None)`.
- Produces `load_json_object(path: Path, *, artifact: str) -> dict[str, Any]`.
- Produces `sha256_file(path: Path) -> str`.
- Produces `ensure_probe_run_dir(root: Path, backend: str, probe_run_id: str) -> Path`.
- Produces `require_relative_artifact(root: Path, reference: str) -> Path`.

- [ ] **Step 1: Write the failing helper tests**

```python
from pathlib import Path
import tempfile
import unittest

from backend_probe_common import (
    ProbeContractError,
    ensure_probe_run_dir,
    load_json_object,
    require_relative_artifact,
    sha256_file,
)


class BackendProbeCommonTests(unittest.TestCase):
    def test_load_json_object_rejects_non_object(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "array.json"
            path.write_text("[]\n", encoding="utf-8")
            with self.assertRaisesRegex(ProbeContractError, "JSON object"):
                load_json_object(path, artifact="family artifact")

    def test_ensure_probe_run_dir_creates_backend_scoped_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = ensure_probe_run_dir(root, "triton_gcu", "probe-001")
            self.assertTrue(path.is_dir())
            self.assertEqual(root / "probes" / "triton_gcu" / "probe-001", path)

    def test_reference_cannot_escape_probe_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            safe = root / "note.md"
            safe.write_text("ok\n", encoding="utf-8")
            self.assertEqual(safe.resolve(), require_relative_artifact(root, "note.md"))
            with self.assertRaisesRegex(ProbeContractError, "relative artifact"):
                require_relative_artifact(root, "../outside.md")
```

- [ ] **Step 2: Run the helper test to verify it fails**

Run:

```bash
python3 -m unittest skills/backend-probe/tests/test_backend_probe_common.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'backend_probe_common'`.

- [ ] **Step 3: Implement the common helper module**

```python
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


class ProbeContractError(ValueError):
    def __init__(self, code: str, message: str, path: Path | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.path = path


def load_json_object(path: Path, *, artifact: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ProbeContractError("artifact-read", f"cannot read {artifact}", path) from error
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ProbeContractError("artifact-json-invalid", f"{artifact} is not valid JSON", path) from error
    if not isinstance(value, dict):
        raise ProbeContractError("artifact-object-required", f"{artifact} must be a JSON object", path)
    return value


def sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ProbeContractError("artifact-read", f"cannot read file {path}", path) from error


def ensure_probe_run_dir(root: Path, backend: str, probe_run_id: str) -> Path:
    path = root / "probes" / backend / probe_run_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_relative_artifact(root: Path, reference: str) -> Path:
    pure = PurePosixPath(reference)
    if pure.is_absolute() or ".." in pure.parts:
        raise ProbeContractError("artifact-reference-invalid", "relative artifact reference must stay under the probe root")
    root = root.resolve()
    candidate = (root / Path(*pure.parts)).resolve()
    if root not in (candidate, *candidate.parents) or not candidate.is_file():
        raise ProbeContractError("artifact-reference-invalid", "relative artifact reference must name an existing file")
    return candidate
```

- [ ] **Step 4: Run helper tests and create the skill scaffolding**

Run:

```bash
python3 -m unittest skills/backend-probe/tests/test_backend_probe_common.py -v
```

Expected: PASS.

Write `skills/backend-probe/README.md` and `skills/backend-probe/SKILL.md` as skeletal documents that:

- describe the skill as pre-campaign;
- list the six v1 families;
- state that the skill may finish without starting `kernel-opt-loop`; and
- state that canonical profiles are never edited directly.

- [ ] **Step 5: Commit scaffolding and common helpers**

```bash
git add skills/backend-probe/SKILL.md \
  skills/backend-probe/README.md \
  skills/backend-probe/scripts/backend_probe_common.py \
  skills/backend-probe/tests/test_backend_probe_common.py \
  skills/backend-probe/tests/fixtures/.gitkeep
git commit -m "skills: scaffold backend-probe skill"
```

### Task 2: Validate per-family machine artifacts

**Files:**
- Create: `skills/backend-probe/scripts/validate_family_result.py`
- Create: `skills/backend-probe/references/family-result-template.json`
- Create: `skills/backend-probe/tests/test_validate_family_result.py`
- Create: `skills/backend-probe/tests/fixtures/families/valid-dot.json`
- Create: `skills/backend-probe/tests/fixtures/families/valid-launch.json`
- Create: `skills/backend-probe/tests/fixtures/families/invalid-missing-scope.json`
- Create: `skills/backend-probe/tests/fixtures/families/invalid-wrong-family.json`

**Interfaces:**
- Produces `FamilyResultValidationError`, a `ProbeContractError` subclass.
- Produces `validate_family_result(path: Path, *, expected_family: str | None = None) -> dict[str, Any]`.
- Normalized result contains `family`, `backend`, `scope`, `observation`, `evidence_refs`, and `result_level`.

- [ ] **Step 1: Write the failing validation tests**

```python
from pathlib import Path
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from validate_family_result import FamilyResultValidationError, validate_family_result

FIXTURES = Path(__file__).parent / "fixtures" / "families"


class ValidateFamilyResultTests(unittest.TestCase):
    def test_valid_dot_artifact_returns_normalized_fields(self):
        result = validate_family_result(FIXTURES / "valid-dot.json", expected_family="dot")
        self.assertEqual("dot", result["family"])
        self.assertEqual("triton_cuda", result["backend"])
        self.assertEqual("observed", result["result_level"])

    def test_missing_scope_is_rejected(self):
        with self.assertRaisesRegex(FamilyResultValidationError, "scope"):
            validate_family_result(FIXTURES / "invalid-missing-scope.json")

    def test_wrong_expected_family_is_rejected(self):
        with self.assertRaisesRegex(FamilyResultValidationError, "expected family"):
            validate_family_result(FIXTURES / "valid-launch.json", expected_family="dot")
```

- [ ] **Step 2: Run the validator test to verify it fails**

Run:

```bash
python3 -m unittest skills/backend-probe/tests/test_validate_family_result.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'validate_family_result'`.

- [ ] **Step 3: Implement the validator and template**

Use a stable artifact shape like this in `family-result-template.json`:

```json
{
  "schema_version": 1,
  "family": "dot",
  "backend": "triton_cuda",
  "runtime_identity": {
    "interpreter": "/abs/python",
    "device": "cuda:0",
    "toolchain": "corex-4.4.0"
  },
  "scope": {
    "shape": ["32", "32", "32"],
    "dtype": "bf16",
    "layout": "row_major"
  },
  "probe": {
    "command": "python3 scripts/bi150_tl_dot_probe_bf16.py",
    "probe_id": "bi150-dot-bf16-001"
  },
  "observation": {
    "result_level": "observed",
    "status": "success",
    "summary": "bf16 inputs with fp32 accumulate succeeded"
  },
  "evidence_refs": ["stdout.log", "trace.json"]
}
```

Implement the validator so it requires:

- top-level object with `schema_version == 1`;
- `family` in the six supported family names;
- nonempty `backend`, `runtime_identity`, `scope`, `probe`, `observation`, `evidence_refs`;
- `observation.result_level` in `observed|inferred|unknown`;
- `observation.status` in `success|failure|unavailable|not-run`;
- if `expected_family` is passed, exact match.

- [ ] **Step 4: Run validation tests and JSON parse checks**

Run:

```bash
python3 -m unittest skills/backend-probe/tests/test_validate_family_result.py -v
python3 -m json.tool skills/backend-probe/references/family-result-template.json >/dev/null
```

Expected: PASS.

- [ ] **Step 5: Commit family-result validation**

```bash
git add skills/backend-probe/scripts/validate_family_result.py \
  skills/backend-probe/references/family-result-template.json \
  skills/backend-probe/tests/test_validate_family_result.py \
  skills/backend-probe/tests/fixtures/families
git commit -m "skills: validate backend-probe family artifacts"
```

### Task 3: Add deterministic route recommendation and end-state classification

**Files:**
- Create: `skills/backend-probe/scripts/recommend_route.py`
- Create: `skills/backend-probe/tests/test_recommend_route.py`
- Create: `skills/backend-probe/tests/fixtures/recommendation/fusion-first/`
- Create: `skills/backend-probe/tests/fixtures/recommendation/launch-first/`
- Create: `skills/backend-probe/tests/fixtures/recommendation/hybrid-first/`
- Create: `skills/backend-probe/tests/fixtures/recommendation/not-ready-for-gemm/`

**Interfaces:**
- Produces `recommend_route(family_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]`.
- Returns `route`, `end_state`, `reasons`, and `missing_families`.
- Valid routes: `fusion-first|launch-first|hybrid-first|not-ready-for-gemm`.
- Valid end states: `ready-for-campaign|not-ready-yet|not-worth-pursuing-now|promotion-only`.

- [ ] **Step 1: Write the failing recommendation tests**

```python
from pathlib import Path
import json
import sys
import unittest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from recommend_route import recommend_route

FIXTURES = Path(__file__).parent / "fixtures" / "recommendation"


def load_family_set(directory: Path) -> dict[str, dict]:
    results = {}
    for path in directory.glob("*.json"):
        results[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return results


class RecommendRouteTests(unittest.TestCase):
    def test_launch_first_when_profiler_is_weak_and_launch_path_is_clear(self):
        result = recommend_route(load_family_set(FIXTURES / "launch-first"))
        self.assertEqual("launch-first", result["route"])
        self.assertEqual("ready-for-campaign", result["end_state"])

    def test_not_ready_for_gemm_when_dot_is_unknown(self):
        result = recommend_route(load_family_set(FIXTURES / "not-ready-for-gemm"))
        self.assertEqual("not-ready-for-gemm", result["route"])
        self.assertEqual("not-ready-yet", result["end_state"])
```

- [ ] **Step 2: Run the recommendation test to verify it fails**

Run:

```bash
python3 -m unittest skills/backend-probe/tests/test_recommend_route.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'recommend_route'`.

- [ ] **Step 3: Implement rule-based recommendation**

Implement explicit, readable rules rather than hidden scoring. For example:

```python
from __future__ import annotations

from typing import Any, Mapping


def recommend_route(family_results: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    missing = [name for name in ("dot", "launch", "profiler", "tuning", "core-primitives", "reductions-mixed-precision") if name not in family_results]
    if missing:
        return {
            "route": "not-ready-for-gemm",
            "end_state": "not-ready-yet",
            "reasons": [f"missing family: {name}" for name in missing],
            "missing_families": missing,
        }

    dot = family_results["dot"]["observation"]
    launch = family_results["launch"]["observation"]
    profiler = family_results["profiler"]["observation"]

    if dot["status"] != "success":
        return {
            "route": "not-ready-for-gemm",
            "end_state": "not-ready-yet",
            "reasons": ["matrix path not established"],
            "missing_families": [],
        }
    if profiler.get("summary") == "launch-only evidence" and launch["status"] == "success":
        return {
            "route": "launch-first",
            "end_state": "ready-for-campaign",
            "reasons": ["launch path is known while device-duration evidence is weak"],
            "missing_families": [],
        }
    return {
        "route": "fusion-first",
        "end_state": "ready-for-campaign",
        "reasons": ["dot path and profiler evidence are both established"],
        "missing_families": [],
    }
```

Expand this minimal logic to cover all four routes and all four end states, including:

- `promotion-only` when new facts are observed but the route does not justify immediate campaign start;
- `not-worth-pursuing-now` when the probe explicitly shows a backend path is available but strategically weak for the current competition stage.

- [ ] **Step 4: Run recommendation tests**

Run:

```bash
python3 -m unittest skills/backend-probe/tests/test_recommend_route.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit route recommendation**

```bash
git add skills/backend-probe/scripts/recommend_route.py \
  skills/backend-probe/tests/test_recommend_route.py \
  skills/backend-probe/tests/fixtures/recommendation
git commit -m "skills: recommend backend probe routes"
```

### Task 4: Render Markdown promotion notes and target descriptors

**Files:**
- Create: `skills/backend-probe/scripts/render_promotion_note.py`
- Create: `skills/backend-probe/references/promotion-note-template.md`
- Create: `skills/backend-probe/bundles/triton-common-v1.json`
- Create: `skills/backend-probe/targets/triton_mlu.json`
- Create: `skills/backend-probe/targets/triton_gcu.json`
- Create: `skills/backend-probe/targets/triton_maca.json`
- Create: `skills/backend-probe/targets/triton_cuda.json`
- Create: `skills/backend-probe/targets/triton_ascend.json`
- Create: `skills/backend-probe/tests/test_contracts.py`
- Create: `skills/backend-probe/tests/fixtures/notes/golden-promotion-note.md`

**Interfaces:**
- Produces `render_promotion_note(family_results: Mapping[str, Mapping[str, Any]], recommendation: Mapping[str, Any]) -> str`.
- Target descriptors declare `backend_id`, `profile_name`, runtime/bootstrap facts, and the enabled family list.
- `triton-common-v1.json` declares the six family names and a short purpose for each.

- [ ] **Step 1: Write the failing contracts and note-rendering tests**

```python
from pathlib import Path
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from render_promotion_note import render_promotion_note


class BackendProbeContractTests(unittest.TestCase):
    def test_target_files_and_bundle_agree(self):
        bundle = json.loads((ROOT / "bundles" / "triton-common-v1.json").read_text(encoding="utf-8"))
        families = set(bundle["families"])
        for name in ("triton_mlu", "triton_gcu", "triton_maca", "triton_cuda", "triton_ascend"):
            target = json.loads((ROOT / "targets" / f"{name}.json").read_text(encoding="utf-8"))
            self.assertTrue(set(target["families"]).issubset(families))

    def test_render_promotion_note_mentions_scope_and_conservative_status(self):
        text = render_promotion_note(
            {
                "dot": {
                    "family": "dot",
                    "backend": "triton_cuda",
                    "scope": {"shape": ["32", "32", "32"], "dtype": "bf16", "layout": "row_major"},
                    "observation": {"result_level": "observed", "status": "success", "summary": "bf16 dot succeeded"},
                    "evidence_refs": ["dot.log"],
                }
            },
            {"route": "fusion-first", "end_state": "promotion-only", "reasons": ["new dot fact"]},
        )
        self.assertIn("bf16 dot succeeded", text)
        self.assertIn("constrained", text)
```

- [ ] **Step 2: Run the contracts test to verify it fails**

Run:

```bash
python3 -m unittest skills/backend-probe/tests/test_contracts.py -v
```

Expected: FAIL because the bundle, targets, and renderer do not exist yet.

- [ ] **Step 3: Implement rendering, bundle, and targets**

Write `triton-common-v1.json` like:

```json
{
  "schema_version": 1,
  "bundle_name": "triton-common-v1",
  "families": [
    "dot",
    "launch",
    "profiler",
    "tuning",
    "core-primitives",
    "reductions-mixed-precision"
  ]
}
```

Write each target JSON as a narrow factual descriptor, for example `triton_gcu.json`:

```json
{
  "schema_version": 1,
  "backend_id": "triton_gcu",
  "profile_name": "triton_gcu",
  "families": ["dot", "launch", "profiler", "tuning", "core-primitives", "reductions-mixed-precision"],
  "runtime_bootstrap": {
    "device": "gcu:0",
    "required_imports": ["torch_gcu", "triton_gcu"],
    "sync_api": "torch.gcu.synchronize()"
  }
}
```

Implement `render_promotion_note()` so the output always contains:

- backend id;
- per-family observation summary;
- exact scope lines;
- recommendation route and end state;
- a conservative profile suggestion line such as `Suggested profile status: constrained` when the observation is first-time evidence.

- [ ] **Step 4: Run contract tests and JSON parsing checks**

Run:

```bash
python3 -m unittest skills/backend-probe/tests/test_contracts.py -v
python3 -m json.tool skills/backend-probe/bundles/triton-common-v1.json >/dev/null
python3 -m json.tool skills/backend-probe/targets/triton_gcu.json >/dev/null
```

Expected: PASS.

- [ ] **Step 5: Commit promotion-note rendering and target descriptors**

```bash
git add skills/backend-probe/scripts/render_promotion_note.py \
  skills/backend-probe/references/promotion-note-template.md \
  skills/backend-probe/bundles/triton-common-v1.json \
  skills/backend-probe/targets \
  skills/backend-probe/tests/test_contracts.py \
  skills/backend-probe/tests/fixtures/notes
git commit -m "skills: add backend probe targets and promotion notes"
```

### Task 5: Integrate documentation and optional kernel-opt-loop consumption

**Files:**
- Modify: `skills/README.md`
- Modify: `README.md`
- Modify: `docs/backend-registry.md`
- Modify: `docs/competition/track1-triton.md`
- Modify: `skills/kernel-opt-loop/SKILL.md`

**Interfaces:**
- Documents `backend-probe` as an optional pre-campaign onboarding step.
- Documents that `kernel-opt-loop` Phase 0 may consume existing probe artifacts as prior facts but does not require them.
- Does not overstate that probe outputs are canonical profile updates.

- [ ] **Step 1: Write the failing documentation consistency test**

Add to `skills/backend-probe/tests/test_contracts.py`:

```python
    def test_repo_docs_reference_backend_probe_consistently(self):
        root_readme = (Path(__file__).resolve().parents[2] / "README.md").read_text(encoding="utf-8")
        skills_readme = (Path(__file__).resolve().parents[1].parent / "README.md").read_text(encoding="utf-8")
        registry = (Path(__file__).resolve().parents[2] / "docs" / "backend-registry.md").read_text(encoding="utf-8")
        self.assertIn("backend-probe", root_readme)
        self.assertIn("backend-probe", skills_readme)
        self.assertIn("promotion note", registry)
```

- [ ] **Step 2: Run the consistency test to verify it fails**

Run:

```bash
python3 -m unittest skills/backend-probe/tests/test_contracts.py -v
```

Expected: FAIL because the repo docs do not mention the new skill yet.

- [ ] **Step 3: Update documentation boundaries exactly**

Apply these factual updates:

- `skills/README.md`: add a `backend-probe` section describing it as a pre-campaign capability-discovery skill.
- `README.md`: in the “如何新增一个 campaign” section, note that a backend may first be onboarded via `backend-probe`, but campaign creation still happens through `kernel-opt-loop`.
- `docs/backend-registry.md`: note that canonical backend profiles may later absorb approved facts from `backend-probe` promotion notes; do not claim automatic mutation.
- `docs/competition/track1-triton.md`: describe `backend-probe` as an optional step before campaign creation when a backend is new or weakly understood.
- `skills/kernel-opt-loop/SKILL.md`: add one short sentence in Phase 0 inputs or initialization stating that existing probe artifacts may be consumed as prior backend facts when their runtime scope matches, but they are optional and do not bypass campaign-local validation.

- [ ] **Step 4: Run documentation consistency checks**

Run:

```bash
python3 -m unittest skills/backend-probe/tests/test_contracts.py -v
python3 - <<'PY'
from pathlib import Path
for path in (Path('skills/backend-probe').rglob('*.md')):
    assert path.read_text(encoding='utf-8').count('```') % 2 == 0, path
PY
```

Expected: PASS.

- [ ] **Step 5: Commit docs and optional-consumption note**

```bash
git add skills/README.md README.md docs/backend-registry.md \
  docs/competition/track1-triton.md skills/kernel-opt-loop/SKILL.md
git commit -m "docs: integrate backend-probe workflow"
```

### Task 6: Full regression and plan closeout

**Files:**
- Modify: `docs/superpowers/plans/2026-08-21-backend-probe-skill.md`

**Interfaces:**
- Confirms the implementation satisfies the spec and leaves no hardware-only test gap.

- [ ] **Step 1: Run the full backend-probe test suite**

Run:

```bash
python3 -m unittest discover -s skills/backend-probe/tests -v
```

Expected: PASS.

- [ ] **Step 2: Run repo-wide hygiene checks**

Run:

```bash
git diff --check
python3 -m unittest skills/kernel-opt-loop/tests/test_contracts.py -v
```

Expected: PASS. The new skill does not break existing `kernel-opt-loop` contract checks.

- [ ] **Step 3: Verify only intended files changed**

Run:

```bash
git status --short
```

Expected: only `skills/backend-probe/`, the listed docs, and the current plan file if its checklist is being updated.

- [ ] **Step 4: Update the plan checklist and self-review**

Check off completed tasks in this plan, then confirm:

- no step still contains placeholders;
- the six probe families are handled consistently everywhere;
- the documentation never claims profile auto-mutation; and
- the skill can terminate without a campaign.

- [ ] **Step 5: Commit final verification and plan updates**

```bash
git add skills/backend-probe docs/superpowers/plans/2026-08-21-backend-probe-skill.md \
  README.md docs/backend-registry.md docs/competition/track1-triton.md \
  skills/README.md skills/kernel-opt-loop/SKILL.md
git commit -m "test: verify backend-probe skill implementation"
```

## Plan Self-Review

- **Spec coverage:** The plan covers the independent skill boundary, six v1 families, run-local artifact location, promotion-note output, conservative promotion boundary, optional `kernel-opt-loop` consumption, and normal non-campaign end states.
- **Placeholder scan:** Every task names exact files, public interfaces, tests, commands, and expected outcomes. No step says “TBD”, “implement later”, or “add appropriate validation”.
- **Type consistency:** The same family names (`dot`, `launch`, `profiler`, `tuning`, `core-primitives`, `reductions-mixed-precision`) and end states (`ready-for-campaign`, `not-ready-yet`, `not-worth-pursuing-now`, `promotion-only`) are used consistently across tasks.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-21-backend-probe-skill.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**

# kernel-opt-loop vNext Semantic Attribution Implementation Plan

**Goal:** Add executable pre-campaign implementation-profile probes, a typed Unified Sketch, machine-readable capability profiles, deterministic source conformance, auditable Orchestrator attribution, and one final bounded configuration-tuning gate to new kernel-opt-loop campaigns.

**Architecture:** Use one profile/evidence contract and the Markdown campaign ledger. Kernel-opt-loop profile onboarding runs versioned probe definitions through a bounded command runner, emits run-local hashed evidence plus a proposed promotion candidate, and may finish without allocating campaign state. Reviewed facts enter the JSON-compatible YAML implementation profile frozen by Phase 0. Campaign rounds add typed Sketch, binding, Verifier facts, and verdict artifacts. Submission finalization reuses the accepted Sketch and the Decision/report/binding/verdict chain to search a finite profile-legal configuration space, pin one selected candidate, and rerun complete verification without introducing finalization-specific state or artifact families.

**Tech Stack:** Python 3 standard library (`ast`, `datetime`, `hashlib`, `json`, `pathlib`, `re`, `subprocess`, `tempfile`, `time`, `unittest`), Markdown contracts, JSON Schema documents as versioned artifact definitions, Git.

**Spec:** `docs/superpowers/specs/2026-08-19-kernel-opt-loop-vnext-semantic-attribution-design.md`

## Global Constraints

- Enable the vNext campaign contract only for new runs. Existing v1/v2 campaign artifacts remain readable and are never rewritten.
- Allow profile probing to complete before or without a campaign. It must not create `team-state.md`, rounds, Decisions, candidates, reports, verdicts, benchmark rankings, or accepted implementation pointers.
- Do not create `skills/backend-probe/`, a second skill contract, a database, daemon, scheduler, network dependency, or another mutable workflow state store.
- Do not modify `base.py`, `auto_bench.py`, benchmark timing semantics, or official competition measurement behavior.
- Do not add PyYAML or `jsonschema`; store `profile.yaml` as JSON-compatible YAML parsed by `json.loads`.
- Keep the canonical profile layout as `profile.yaml + schema/ + probes/ + evidence/`. The profile root contains byte-identical `schema/profile.schema.json` and `schema/shared-profile.schema.json` copies, not path-only pointers; `profile.yaml` records both refs/hashes and the shared schema version, while canonical contract tests compare the shared copy with the global source schema. `probes/` contains versioned definitions and declared input artifacts; `evidence/` contains only reviewed reusable records/attachments, never raw run output.
- Phase 0 freezes the entire root-confined implementation-profile closure byte-for-byte under `state/implementation_profile_snapshot/`; validation must still pass after the canonical profile directory is changed or deleted.
- Use schema version `1` for probe definitions, probe runs/results, promotion candidates, Sketch, profile, project claim, binding, and verdict artifacts. Use `metadata.schema_version: 2` for vNext Decision Markdown and preserve schema-v1 Decision behavior.
- Keep `target_id` distinct from `implementation_profile_id`. API compatibility or a shared runtime namespace does not transfer evidence across vendors, devices, architectures, or toolchains.
- A profile may have `profile_status: partial`; capability entries use exactly `supported|constrained|unknown|unsupported|prohibited`. `unknown` never satisfies a required Sketch capability and is never interpreted as unavailable.
- Before Phase 0, demand-scoped selection examines only explicit `must-resolve|before-fallback` requirements. An Unknown primary with an algorithm-substitution fallback must run the unique exact-scope catalog probe before fallback; unrelated Unknowns are not probed and ambiguous matches fail.
- Numerically checked success stops as onboarding disposition `promotion-pending`—not a probe summary or campaign terminal result—with no campaign state until a maintainer promotes the exact scope or explicitly authorizes fallback. Partial/failure/probe-specific block/no-match leaves the primary Unknown; fallback requires durable provenance and never consumes raw evidence as profile authority, while a campaign-global target/profile/runtime identity block remains blocking.
- `state/project_capability_claim.json` is the immutable maintainer-authorization authority for fallback. It embeds the normalized requirement, disposition, confirmation, and optional probe hashes; it forbids raw probe-result refs and remains independently valid after the pre-campaign run is deleted.
- Probe definitions use structured argv arrays. The runner never invokes a shell, requires validated absolute roots and runtime paths, applies bounded timeouts, creates run directories exclusively, and writes normalized artifacts atomically.
- Probe results and promotion candidates never mutate canonical profiles. Approval and profile updates occur in a separate maintainer commit, and approved scope may not exceed observed scope.
- Probe summaries report evidence readiness and gaps only. They do not infer competition ROI, global backend priority, or a mandatory GEMM/fusion/launch route.
- Preserve the campaign terminal result enum: `accepted|no-improvement|screened-out|design-rejected|candidate-failed|aborted`. Attribution remains separate: `design-error|code-error|lowering-unknown|evidence-gap|none`.
- `lowering-unknown` maps to terminal `design-rejected` but does not increment `failed_attempt_streak`; explicit `design-error` does increment it.
- Use SHA-256 for every cross-artifact reference. Candidate source spans are evidence locations, not identity; stable identity is `statement_id` plus the candidate hash.
- Keep raw profiler logs, probe stdout/stderr, environment secrets, and runtime sessions in project-local gitignored areas. Normalized result documents carry relative paths, byte counts, and hashes.
- All tests run without accelerator hardware. Probe-runner tests use fixture payloads and stub commands; a real profile probe payload may exist without claiming an unobserved hardware success.
- The implementation-profile schema and command runner must not require Triton primitives or Python execution, although the first source-binding analyzer remains Python/Triton-specific.
- Each Triton submission snapshot runs one submission-finalization configuration gate. `submission_snapshot_id` hashes the accepted candidate/binding, Sketch, profile, claim, runtime snapshot, official measurement fingerprint, harness, and base/reference; Decision, report, and verdict carry the same ID, and validated verdict scanning prevents duplicate input or already-finalized output identities.
- Finalization allocates the maximum occupied artifact index plus one across standard artifact families, or resumes the same valid incomplete Decision for the same snapshot ID. The index is not a campaign round and updates no round pointer, terminal field, attribution, streak, or run-policy counter.
- The finite deterministic configuration domain contains the accepted configuration as fallback/control and is covered by reviewed exact-scope `supported|constrained` profile legality. Missing or Unknown legality blocks; a singleton is valid only when the profile explicitly covers the accepted configuration.
- Tuning fields are Decision-declared `preferred|exploratory` launch options or compile-time meta-parameters. The accepted Sketch, algorithm, precision, effects, aliases, Host Plan, public interface, and semantic layout remain immutable.
- Search runs one accepted source hash through temporary configuration injection. The existing gitignored `log/final-tuning/` boundary may contain configuration tables, compiler caches/binaries, and raw output, but no derived candidate-language source.
- Verifier returns normalized trials to Orchestrator without a persisted selection artifact; Orchestrator runs the pure selector, Coder pins once or confirms the accepted fallback, Verifier performs final official verification, and only then writes the report atomically. Orchestrator reruns the selector from the sealed report before verdict creation.
- Finalization uses a pure submission-promotion predicate and routes only `submission-ready|blocked`. An improved winner atomically advances the existing `last_accepted_kernel`/`last_accepted_report` pair to the pinned source/sealed report while preserving `last_accepted_round`; fallback-retained changes neither pointer, and partial pair updates are invalid. The verdict omits attribution, terminal-result, counter-effect, and run-policy fields; the existing report/verdict chain is the only persisted finalization authority.
- The final candidate contains one fixed selected configuration and no runtime/online `@triton.autotune`, adaptive search, or autotune-cache selection dependency.

---

## File Structure

| Path | Responsibility |
|---|---|
| `skills/kernel-opt-loop/schemas/probe-definition.schema.json` | Versioned profile-local probe definition. |
| `skills/kernel-opt-loop/schemas/probe-run.schema.json` | One pre-campaign or campaign-local probe run manifest. |
| `skills/kernel-opt-loop/schemas/probe-result.schema.json` | Normalized per-probe observation, scope, command, and evidence hash chain. |
| `skills/kernel-opt-loop/schemas/profile-promotion-candidate.schema.json` | Proposed capability/profile changes awaiting explicit review. |
| `skills/kernel-opt-loop/schemas/sketch.schema.json` | Versioned normative shape for the typed Sketch JSON artifact. |
| `skills/kernel-opt-loop/schemas/profile.schema.json` | Versioned canonical profile shape, including exact-scope legal configuration constraints for final tuning. |
| `skills/kernel-opt-loop/schemas/project-capability-claim.schema.json` | Shape for the run-local operator applicability claim. |
| `skills/kernel-opt-loop/schemas/binding.schema.json` | Shape for the Coder source-level binding ledger. |
| `skills/kernel-opt-loop/schemas/verdict.schema.json` | Shape for the Orchestrator attribution verdict. |
| `skills/kernel-opt-loop/scripts/vnext_common.py` | Shared errors, JSON loading, SHA-256, relative-reference, atomic-write, source-span, and JSON-compatible YAML helpers. |
| `skills/kernel-opt-loop/scripts/validate_probe.py` | Probe definition/run/result validation plus pure demand-scoped exact-probe selection. |
| `skills/kernel-opt-loop/scripts/run_profile_probe.py` | Shell-free bounded command runner that writes an isolated run-local probe directory. |
| `skills/kernel-opt-loop/scripts/render_profile_promotion.py` | Deterministically derives a proposed promotion candidate and Markdown note without editing a profile. |
| `skills/kernel-opt-loop/scripts/validate_sketch.py` | Typed Sketch structural and semantic checker. |
| `skills/kernel-opt-loop/scripts/validate_profile.py` | Canonical implementation-profile, legal configuration-domain, and project-capability-claim checker. |
| `skills/kernel-opt-loop/scripts/validate_binding.py` | Profile-selected source analyzer and Sketch-to-source binding conformance checker; Python/Triton is the first adapter. |
| `skills/kernel-opt-loop/scripts/validate_verdict.py` | Causal graph validation, Verifier fact-pack extraction, final-configuration selection/post-pin gates, attribution rule evaluation, and verdict validation. |
| `skills/kernel-opt-loop/scripts/validate_decision.py` | v1 compatibility plus vNext Decision references, hashes, causal graph, final-tuning contract, and artifact-existence validation. |
| `skills/kernel-opt-loop/scripts/evaluate_run_policy.py` | Existing terminal evaluator extended with explicit attribution counter effects. |
| `skills/kernel-opt-loop/profiles/triton_mlu/profile.yaml` | Canonical machine-readable `triton_mlu` profile with initial `partial` status and reviewed-evidence promotion rules. |
| `skills/kernel-opt-loop/profiles/triton_mlu/schema/profile.schema.json` | Profile-local byte-for-byte vendored schema used by `load_profile()`. |
| `skills/kernel-opt-loop/profiles/triton_mlu/schema/shared-profile.schema.json` | Frozen shared-source copy; canonical tests compare it with the global schema, and snapshots remain closed. |
| `skills/kernel-opt-loop/profiles/triton_mlu/probes/basic-memory.json` | Versioned definition for a real MLU basic-memory probe. |
| `skills/kernel-opt-loop/profiles/triton_mlu/probes/basic_memory.py` | Hardware probe payload; emits a normalized result payload when run in a matched MLU environment. |
| `skills/kernel-opt-loop/profiles/triton_mlu/evidence/README.md` | Reviewed-evidence naming, scope, archival, and promotion rules; no fabricated success. |
| `skills/kernel-opt-loop/references/profile-promotion-note-template.md` | Human review rendering for a proposed machine-readable promotion candidate. |
| `skills/kernel-opt-loop/references/project-capability-claim-template.json` | Template for the operator-specific applicability claim. |
| `skills/kernel-opt-loop/references/decision-template.md` | vNext Decision metadata, Sketch reference/rendering, hint modalities, causal graph, mixed sub-contracts, and finite final-tuning contract. |
| `skills/kernel-opt-loop/references/report-template.md` | vNext structured Verifier fact pack, observed-lowering evidence, normalized final-tuning trials, and post-pin official evidence. |
| `skills/kernel-opt-loop/references/team-state-template.md` | vNext run identity, frozen implementation-profile snapshot, campaign pointers, atomic `last_accepted_kernel`/`last_accepted_report` submission pair, attribution fields, and explicit contract version; no finalization-specific field. |
| `skills/kernel-opt-loop/references/invariants.md` | Probe/profile boundaries, `*_context.md` ownership, source/binding/lowering boundaries, attribution counter rules, and config-only finalization boundary. |
| `skills/kernel-opt-loop/references/role-context-template.md` | Single `*_context.md` naming convention; no `*_state.md` aliases. |
| `skills/kernel-opt-loop/prompts/designer.md` | Designer typed Sketch, causal graph, capability, hint-modality, and config-only final-tuning Decision responsibilities. |
| `skills/kernel-opt-loop/prompts/coder.md` | Decision-local capability probe, binding ledger, conformance/repair, and one-time selected-config pinning responsibilities; no pre-campaign ownership or tuning-source variants. |
| `skills/kernel-opt-loop/prompts/verifier.md` | Verifier structured facts, observed lowering, bounded configuration comparison, and post-pin official verification without blame assignment. |
| `skills/kernel-opt-loop/SKILL.md` | Pre-campaign profile onboarding, vNext artifact gates/routing, final bounded configuration tuning, implementation-profile snapshot, verdict creation, and legacy boundary. |
| `skills/kernel-opt-loop/prompts/coder_targets/*.md` | Legacy/rendered human explanations that refer to canonical implementation profiles when migrated. |
| `skills/kernel-opt-loop/tests/test_vnext_common.py` | Unit tests for common document, atomic-write, hash, reference, and source-span helpers. |
| `skills/kernel-opt-loop/tests/test_validate_probe.py` | Probe validation plus demand selection, no-sweep, no-match, and ambiguity fixtures. |
| `skills/kernel-opt-loop/tests/test_run_profile_probe.py` | CLI selection, bounded subprocess, timeout, output confinement, and atomic artifact tests. |
| `skills/kernel-opt-loop/tests/test_profile_promotion.py` | Promotion rendering, scope narrowing, no-mutation, and approval-boundary tests. |
| `skills/kernel-opt-loop/tests/test_validate_sketch.py` | Typed Sketch semantic fixtures. |
| `skills/kernel-opt-loop/tests/test_validate_profile.py` | Implementation-profile, legal configuration-domain, and project-claim fixtures, including a non-Triton profile. |
| `skills/kernel-opt-loop/tests/test_validate_binding.py` | Candidate source, statement-level binding, and selected-config pin/hash fixtures. |
| `skills/kernel-opt-loop/tests/test_validate_verdict.py` | Causal graph, fact-pack, deterministic final-tuning selection/post-pin gates, atomic accepted kernel/report pair promotion, verdict rule, and terminal/counter fixtures. |
| `skills/kernel-opt-loop/tests/test_validate_decision.py` | Legacy plus vNext Decision, final-tuning contract, and actual-reference/anchor validation. |
| `skills/kernel-opt-loop/tests/test_run_policy.py` | Attribution-specific `failed_attempt_streak` tests. |
| `skills/kernel-opt-loop/tests/test_contracts.py` | Cross-file contract, profile registry, public script/CLI presence, naming, schema, and documentation consistency assertions. |
| `skills/kernel-opt-loop/tests/fixtures/vnext/probes/` | Fake definitions, payloads, results, logs, promotion candidates, and malformed hash/scope cases. |
| `skills/kernel-opt-loop/tests/fixtures/vnext/` | Self-contained profiles, claims, Sketches, candidate sources, bindings, reports, and verdicts. |
| `README.md` | Implementation-profile list and vNext new-run/profile-onboarding statement. |
| `docs/backend-registry.md` | Target-to-implementation-profile mapping, onboarding status, registry location, and snapshot behavior. |

## Dependency Sequence

1. Shared deterministic loading, atomic writes, probe schemas, and fixture layout before any runner or artifact checker.
2. Implementation-profile and probe-definition validation before executing a probe.
3. Demand-scoped exact-probe selector, runner, and promotion boundary before any profile claims a usable versioned probe suite.
4. Profile and Sketch validators before vNext Decision validation.
5. Binding validator before Coder contract can emit `candidate-ready`.
6. Verifier fact-pack and verdict validator before Orchestrator routing and policy changes.
7. Profile-legal configuration constraints and final-tuning Decision validation before submission finalization can begin.
8. Deterministic bounded trial normalization/selection before Coder pins a configuration; fresh binding before post-pin correctness and official measurement.
9. Templates, prompts, state machine, registry, and documentation after deterministic interfaces are tested.
10. Separate pre-campaign, campaign, and submission-finalization integration fixtures plus full regression last.

## Implementation Tasks

### Task 1: Add Versioned Artifact Primitives and Fixture Harness

**Files:**
- Create: `skills/kernel-opt-loop/scripts/vnext_common.py`
- Create: `skills/kernel-opt-loop/schemas/probe-definition.schema.json`
- Create: `skills/kernel-opt-loop/schemas/probe-run.schema.json`
- Create: `skills/kernel-opt-loop/schemas/probe-result.schema.json`
- Create: `skills/kernel-opt-loop/schemas/profile-promotion-candidate.schema.json`
- Create: `skills/kernel-opt-loop/schemas/sketch.schema.json`
- Create: `skills/kernel-opt-loop/schemas/profile.schema.json`
- Create: `skills/kernel-opt-loop/schemas/project-capability-claim.schema.json`
- Create: `skills/kernel-opt-loop/schemas/binding.schema.json`
- Create: `skills/kernel-opt-loop/schemas/verdict.schema.json`
- Create: `skills/kernel-opt-loop/tests/test_vnext_common.py`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/.gitkeep`

**Interfaces:**
- Produces `ContractValidationError(code: str, message: str, path: Path | None = None)`.
- Produces `load_json_document(path: Path, *, artifact: str) -> dict[str, Any]`.
- Produces `load_json_yaml_document(path: Path, *, artifact: str) -> dict[str, Any]`.
- Produces `sha256_file(path: Path) -> str`, `sha256_canonical_json(value: Mapping[str, Any]) -> str`, `compute_submission_snapshot_id(anchors: Mapping[str, str]) -> str`, `require_relative_artifact(root: Path, reference: str) -> Path`, and `validate_source_span(source: str, span: Mapping[str, Any]) -> None`.
- Produces `create_exclusive_directory(path: Path) -> Path` and `write_json_atomic(path: Path, value: Mapping[str, Any]) -> None` for run-local probe artifacts.
- Later validators and the runner import only these helpers; no component duplicates path traversal, exclusive-create, atomic-write, or hash logic.

- [x] **Step 1: Write failing common-helper tests**

Create `test_vnext_common.py` using `sys.path.insert(0, str(SCRIPTS))`. Add isolated temporary-directory tests for JSON object loading, JSON-compatible YAML loading, SHA-256 stability, canonical `submission_snapshot_id`, root confinement, exclusive directory creation, atomic sorted-JSON writes, and source spans. The snapshot helper requires exactly accepted candidate/binding, Sketch, profile, claim, runtime snapshot, official measurement fingerprint, harness, and base/reference hashes; Decision hash and artifact index are invalid inputs.

```python
from pathlib import Path
import tempfile
import unittest

from vnext_common import (
    ContractValidationError,
    create_exclusive_directory,
    load_json_document,
    load_json_yaml_document,
    require_relative_artifact,
    sha256_canonical_json,
    sha256_file,
    validate_source_span,
    write_json_atomic,
)


class VNextCommonTests(unittest.TestCase):
    def test_json_yaml_accepts_json_subset_and_rejects_yaml_only_syntax(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "profile.yaml"
            valid.write_text('{"schema_version":1,"name":"triton_mlu"}\n', encoding="utf-8")
            self.assertEqual("triton_mlu", load_json_yaml_document(valid, artifact="profile")["name"])

            invalid = root / "invalid.yaml"
            invalid.write_text("schema_version: 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractValidationError, "json-compatible YAML"):
                load_json_yaml_document(invalid, artifact="profile")

    def test_reference_cannot_escape_project_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rounds").mkdir()
            path = root / "rounds" / "sketch_001.json"
            path.write_text("{}", encoding="utf-8")
            self.assertEqual(path, require_relative_artifact(root, "rounds/sketch_001.json"))
            with self.assertRaisesRegex(ContractValidationError, "relative artifact"):
                require_relative_artifact(root, "../outside.json")

    def test_exclusive_directory_and_atomic_json_never_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = create_exclusive_directory(Path(directory) / "probe-001")
            write_json_atomic(run_dir / "run.json", {"b": 2, "a": 1})
            self.assertEqual('{"a":1,"b":2}\n', (run_dir / "run.json").read_text(encoding="utf-8"))
            with self.assertRaises(ContractValidationError):
                create_exclusive_directory(run_dir)

    def test_canonical_json_hash_is_key_order_independent(self):
        self.assertEqual(sha256_canonical_json({"b": 2, "a": 1}), sha256_canonical_json({"a": 1, "b": 2}))

    def test_source_span_requires_existing_one_based_range(self):
        validate_source_span("first\nsecond\n", {"start": [2, 1], "end": [2, 7]})
        with self.assertRaisesRegex(ContractValidationError, "source span"):
            validate_source_span("first\n", {"start": [2, 1], "end": [2, 2]})
```

- [x] **Step 2: Run the helper test to verify it fails**

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_vnext_common.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'vnext_common'`.

- [x] **Step 3: Implement common document and reference helpers**

Create `vnext_common.py` with the public surface below. Treat a missing, non-UTF-8, malformed, non-object, absolute, escaping, or absent reference as a stable `ContractValidationError`; never allow a raw `OSError`, `UnicodeDecodeError`, or `JSONDecodeError` out of a public validator.

```python
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any


class ContractValidationError(ValueError):
    def __init__(self, code: str, message: str, path: Path | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.path = path


def sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ContractValidationError("artifact-read", f"cannot read {path}", path) from error


def sha256_canonical_json(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json_document(path: Path, *, artifact: str) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise ContractValidationError("artifact-read", f"cannot read {artifact}", path) from error
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ContractValidationError("artifact-json-invalid", f"{artifact} is not valid JSON", path) from error
    if not isinstance(value, dict):
        raise ContractValidationError("artifact-object-required", f"{artifact} must be a JSON object", path)
    return value


def load_json_yaml_document(path: Path, *, artifact: str) -> dict[str, Any]:
    try:
        return load_json_document(path, artifact=artifact)
    except ContractValidationError as error:
        raise ContractValidationError(error.code, f"{artifact} must use JSON-compatible YAML: {error.message}", path) from error


def require_relative_artifact(root: Path, reference: str) -> Path:
    if not isinstance(reference, str) or not reference:
        raise ContractValidationError("artifact-reference-invalid", "relative artifact reference is required")
    pure = PurePosixPath(reference)
    if pure.is_absolute() or ".." in pure.parts:
        raise ContractValidationError("artifact-reference-invalid", "relative artifact reference must remain under project root")
    root = root.resolve()
    candidate = (root / Path(*pure.parts)).resolve()
    if root not in (candidate, *candidate.parents) or not candidate.is_file():
        raise ContractValidationError("artifact-reference-invalid", "relative artifact reference must name an existing file")
    return candidate


def validate_source_span(source: str, span: Mapping[str, Any]) -> None:
    start, end = span.get("start"), span.get("end")
    if not all(isinstance(point, list) and len(point) == 2 for point in (start, end)):
        raise ContractValidationError("source-span-invalid", "source span must contain start and end [line, column]")
    if any(isinstance(value, bool) or not isinstance(value, int) for point in (start, end) for value in point):
        raise ContractValidationError("source-span-invalid", "source span coordinates must be integers")
    lines = source.splitlines(keepends=True)
    if start[0] < 1 or end[0] < start[0] or end[0] > len(lines):
        raise ContractValidationError("source-span-invalid", "source span line range is invalid")
    if start[1] < 1 or end[1] < 1 or (end[0], end[1]) <= (start[0], start[1]):
        raise ContractValidationError("source-span-invalid", "source span must be a nonempty range")
    if start[1] > len(lines[start[0] - 1]) + 1 or end[1] > len(lines[end[0] - 1]) + 1:
        raise ContractValidationError("source-span-invalid", "source span column is outside source text")
```

Add `create_exclusive_directory()` using `Path.mkdir(parents=True, exist_ok=False)` with stable collision/path errors, and `write_json_atomic()` using a same-directory temporary file, `json.dumps(..., sort_keys=True, separators=(",", ":")) + "\n"`, flush/close, and `Path.replace()`. Probe code must never partially overwrite a prior run.

Create nine schema documents. Each must have a draft marker, title, top-level `type: object`, `required`, `additionalProperties: false`, and a `schema_version` property constrained to integer `1`. The Python validators remain the enforcement mechanism; the JSON Schema files are versioned definitions and documentation. Probe schemas must make `target_id`, `implementation_profile_id`, definition/result hashes, runtime fingerprint, argv array, bounded execution state, observed scope, and evidence path/byte/hash records explicit. The promotion schema must require `review_status: proposed`, permit `onboarding_disposition: promotion-pending` only for eligible demand-selected success, and forbid a recommended scope wider than the source result through Python validation. The project-claim schema must define an embedded normalized requirement plus canonical-JSON hash, primary/fallback contracts and signatures, `fallback_kind`, `probe_policy`, onboarding outcome, promotion disposition, `fallback_authorized`, reason, maintainer confirmation, optional probe id/definition/result hashes, and `primary_remains_unknown`; raw probe-result refs are forbidden. The profile schema permits exact-scope finite configuration constraints, and the verdict schema defines a separate `final-autotune` branch with `submission_snapshot_id`, normalized report-fact hash, selected/final configuration hashes, post-pin gates, and `submission-ready|blocked` route while forbidding campaign attribution, terminal, counter, and run-policy fields. These extensions create no additional artifact family. Ordinary binding/verdict branches retain `round`; finalization binding/report/verdict metadata require `artifact_kind: submission-finalization` and `artifact_index` matching the filename while forbidding campaign `round`.

For example, `schemas/binding.schema.json` must declare the stable fields used by the binding ledger. The branch-specific discriminator is intentional: the first `oneOf` branch is for ordinary campaign rounds and the second is for submission finalization. The shared `required` list deliberately excludes `round`; finalization requires `artifact_kind` plus `artifact_index` and forbids campaign `round`.

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "kernel-opt-loop binding ledger",
  "type": "object",
  "additionalProperties": false,
  "required": ["schema_version", "decision_sha256", "sketch_sha256", "candidate_path", "candidate_sha256", "bindings"],
  "properties": {
    "schema_version": {"const": 1},
    "round": {"type": "string", "pattern": "^[0-9]{3}$"},
    "artifact_kind": {"const": "submission-finalization"},
    "artifact_index": {"type": "string", "pattern": "^[0-9]{3}$"},
    "decision_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    "sketch_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    "candidate_path": {"type": "string"},
    "candidate_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    "bindings": {"type": "array", "minItems": 1}
  },
  "oneOf": [
    {"required": ["round"]},
    {
      "required": ["artifact_kind", "artifact_index"],
      "not": {"required": ["round"]}
    }
  ]
}
```

- [x] **Step 4: Run common tests and schema JSON parsing**

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_vnext_common.py -v
for schema in skills/kernel-opt-loop/schemas/*.schema.json; do
  python3 -m json.tool "$schema" >/dev/null
done
```

Expected: PASS; every schema parses as JSON and every helper rejects malformed inputs through `ContractValidationError`.

- [x] **Step 5: Commit artifact primitives**

```bash
git add skills/kernel-opt-loop/scripts/vnext_common.py \
  skills/kernel-opt-loop/schemas \
  skills/kernel-opt-loop/tests/test_vnext_common.py \
  skills/kernel-opt-loop/tests/fixtures/vnext/.gitkeep
git commit -m "skills: add vnext artifact primitives"
```

### Task 2: Implement Typed Unified Sketch Validation

**Files:**
- Create: `skills/kernel-opt-loop/scripts/validate_sketch.py`
- Create: `skills/kernel-opt-loop/tests/test_validate_sketch.py`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/sketches/valid-kernel.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/sketches/invalid-duplicate-definition.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/sketches/invalid-undefined-use.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/sketches/invalid-unbounded-store.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/sketches/invalid-undeclared-alias.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/sketches/invalid-hint-modality.json`

**Interfaces:**
- Produces `SketchValidationError`, a `ContractValidationError` subclass.
- Produces `validate_sketch(path: Path, *, expected_round: str | None = None) -> dict[str, Any]`.
- The normalized result contains `statement_index`, `value_definitions`, `effect_outputs`, `required_hints`, `preferred_hints`, `exploratory_hints`, and `causal_node_ids`.
- `validate_decision.py` and `validate_binding.py` consume the normalized validation result as their Sketch contract input.

- [x] **Step 1: Write failing semantic tests**

Create a valid fixture with two tensor declarations, a register tile, a `load`, `compute`, and `store` operation, one `parallel` domain, a connected `guard`, declared output effect, a `required` `num_warps` hint, and one causal mechanism node. Add these tests:

```python
from pathlib import Path
import sys
import unittest

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

from validate_sketch import SketchValidationError, validate_sketch

FIXTURES = Path(__file__).parent / "fixtures" / "vnext" / "sketches"


class ValidateSketchTests(unittest.TestCase):
    def test_valid_kernel_sketch_returns_statement_and_effect_indexes(self):
        result = validate_sketch(FIXTURES / "valid-kernel.json", expected_round="001")
        self.assertEqual("load", result["statement_index"]["op.load.row"]["kind"])
        self.assertEqual("row", result["value_definitions"]["row"])
        self.assertEqual(["topk_values"], result["effect_outputs"])
        self.assertEqual(["num_warps"], result["required_hints"])

    def test_duplicate_value_definition_is_design_error(self):
        with self.assertRaisesRegex(SketchValidationError, "duplicate value definition"):
            validate_sketch(FIXTURES / "invalid-duplicate-definition.json")

    def test_load_or_store_requires_guarded_index_domain(self):
        with self.assertRaisesRegex(SketchValidationError, "bounded index"):
            validate_sketch(FIXTURES / "invalid-unbounded-store.json")
```

Add separate assertions for undefined operation input, undeclared output alias/mutation, missing operation effect declaration, unknown hint modality, duplicate statement ID, and a causal node referenced by no operation or output observable.

- [x] **Step 2: Run the Sketch module to verify it fails**

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_validate_sketch.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'validate_sketch'`.

- [x] **Step 3: Implement the checker in explicit semantic passes**

Create `validate_sketch.py`. Keep each pass pure and take only normalized mappings; do not import Triton or execute kernels.

```python
from pathlib import Path
from typing import Any

from vnext_common import ContractValidationError, load_json_document


class SketchValidationError(ContractValidationError):
    pass


def validate_sketch(path: Path, *, expected_round: str | None = None) -> dict[str, Any]:
    sketch = load_json_document(path, artifact="sketch")
    _validate_header(sketch, expected_round)
    declarations = _index_declarations(sketch["declarations"])
    statement_index = _index_statements(sketch["operations"], sketch["control"])
    value_definitions = _validate_ssa_and_operation_signatures(declarations, sketch["operations"])
    _validate_control_and_bounds(sketch["operations"], sketch["control"])
    effect_outputs = _validate_effects_and_aliases(sketch["effects"], sketch["operations"], value_definitions)
    hint_groups = _validate_hint_modalities(sketch["hints"])
    causal_node_ids = _validate_causal_nodes(sketch["causal_nodes"], sketch["operations"], effect_outputs)
    return {
        "valid": True,
        "sketch": sketch,
        "statement_index": statement_index,
        "value_definitions": value_definitions,
        "effect_outputs": effect_outputs,
        "causal_node_ids": causal_node_ids,
        **hint_groups,
    }
```

Apply these concrete rules in the named helpers:

- declarations have unique `id`, `kind` in `tensor|tile|scalar`, nonempty shape, dtype, layout, and memory;
- each operation has a unique `id`, recognized `kind` in `alloc|load|compute|store`, nonempty `inputs`, `outputs`, and exact `effects.reads`/`effects.writes` lists;
- every input is either a declaration or exactly one earlier operation output; no output may redefine an existing declaration or operation output;
- operation edge declarations agree on dtype, layout, and memory unless the `compute` operation includes an explicit `conversion` object;
- every `load` and `store` has nonempty `index_domain` and `mask`, and that mask string references a declared `guard` or the operation's parallel domain;
- every `store` target is listed in `effects.outputs`, `effects.mutations`, or `effects.aliases`; aliases name both source and target and may not be implicit;
- hints have unique names and `modality` in `required|preferred|exploratory`;
- causal nodes have unique `id`, nonempty `kind`/`expected`, and are referenced by at least one operation `causal_nodes` list or an effect observable.

Use stable error codes such as `sketch-duplicate-definition`, `sketch-undefined-value`, `sketch-index-unbounded`, `sketch-effect-undeclared`, and `sketch-hint-modality-invalid`.

- [x] **Step 4: Run semantic tests and legacy decision tests**

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_validate_sketch.py -v
python3 -m unittest skills/kernel-opt-loop/tests/test_validate_decision.py -v
```

Expected: PASS. The module accepts the typed JSON fixtures, and schema-v1 Decision validation continues to pass.

- [x] **Step 5: Commit the typed Sketch gate**

```bash
git add skills/kernel-opt-loop/scripts/validate_sketch.py \
  skills/kernel-opt-loop/tests/test_validate_sketch.py \
  skills/kernel-opt-loop/tests/fixtures/vnext/sketches
git commit -m "skills: validate typed unified sketches"
```

### Task 3: Add Machine-Readable Implementation Profiles and Project Capability Claims

**Files:**
- Create: `skills/kernel-opt-loop/scripts/validate_profile.py`
- Create: `skills/kernel-opt-loop/profiles/triton_mlu/profile.yaml`
- Create: `skills/kernel-opt-loop/profiles/triton_mlu/schema/profile.schema.json`
- Create: `skills/kernel-opt-loop/profiles/triton_mlu/schema/shared-profile.schema.json`
- Create: `skills/kernel-opt-loop/profiles/triton_mlu/evidence/README.md`
- Create: `skills/kernel-opt-loop/references/project-capability-claim-template.json`
- Create: `skills/kernel-opt-loop/tests/test_validate_profile.py`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/profiles/valid-partial/profile.yaml`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/profiles/valid-partial/schema/profile.schema.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/profiles/valid-partial/schema/shared-profile.schema.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/profiles/valid-clike-partial/profile.yaml`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/profiles/valid-clike-partial/schema/profile.schema.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/profiles/valid-clike-partial/schema/shared-profile.schema.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/profiles/unknown-required/profile.yaml`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/profiles/unknown-required/schema/profile.schema.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/profiles/unknown-required/schema/shared-profile.schema.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/profiles/valid-partial/evidence/approved-result.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/claims/valid-claim.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/claims/runtime-mismatch.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/claims/target-id-mismatch.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/claims/implementation-profile-mismatch.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/claims/valid-fallback-disposition.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/claims/silent-algorithm-substitution.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/claims/raw-probe-ref-fallback.json`
- Modify: `skills/kernel-opt-loop/prompts/coder_targets/triton_mlu.md`

**Interfaces:**
- Produces `ProfileValidationError`.
- Produces `load_profile(path: Path) -> dict[str, Any]` and `validate_project_claim(path: Path, *, profile: Mapping[str, Any], snapshot: Mapping[str, Any]) -> dict[str, Any]`.
- Produces `require_capability(profile: Mapping[str, Any], contract_name: str, signature: Mapping[str, Any], modality: str) -> dict[str, Any]`.
- Produces `validate_configuration_domain(profile: Mapping[str, Any], fields: Sequence[Mapping[str, Any]], scope: Mapping[str, Any]) -> tuple[dict[str, Any], ...]`, which returns the finite deterministic domain after exact-scope legality and cross-field checks.
- New artifacts use `implementation_profile_id`, `implementation_profile_version`, and `implementation_profile_sha256`; legacy `target_profile` remains read-only compatibility input and is normalized internally, never emitted alongside the new fields.
- `validate_project_claim()` checks the claim's implementation-profile ID/version/hash and concrete `target_id` against the immutable run snapshot; it never writes or promotes the canonical profile.
- `required` returns `capability-miss` for `unknown`, `unsupported`, or `prohibited`; target/runtime/profile identity mismatch raises `environment-blocked`. `require_capability()` reports only the primary status and never selects a fallback.
- `validate_project_claim()` validates normalized primary/fallback contracts and exact signatures, `fallback_kind: semantic-accommodation|algorithm-substitution`, `probe_policy: optional|before-fallback|must-resolve`, and embedded `qualification_dispositions`. Each disposition contains the full normalized requirement and `sha256_canonical_json(requirement)` hash, onboarding outcome, promotion disposition, `fallback_authorized`, reason, maintainer confirmation identity, UTC RFC 3339 timestamp, and method `explicit-user-instruction|maintainer-reviewed-commit`, optional probe id/definition/result hashes, and `primary_remains_unknown`; raw probe-result refs are forbidden. `qualification_disposition_sha256` is `sha256_canonical_json()` of the complete embedded disposition object, including id, requirement/hash, outcomes, authorization, reason, confirmation, optional probe hashes, and Unknown marker; the object contains no self-hash field. A silent algorithm substitution, unresolved `before-fallback`, or unconfirmed fallback is invalid.
- A profile declares `source_conformance.analyzer` and `binding_model`. The schema accepts non-Triton implementations even though Task 6 initially implements only `python-ast-triton`.

- [x] **Step 1: Write failing profile and claim tests**

Add profile tests covering a structurally complete partial Triton profile, a structurally complete C-like profile with no `tl.*` symbols, all five capability statuses, profile-selected source-conformance metadata, status-specific required-hint behavior, target/profile/runtime scope matching, run-local claims, valid embedded maintainer-authorized fallback disposition, finite legal configuration fields/values and cross-field exclusions, and rejection of silent/unresolved substitutions, missing confirmation, requirement-hash mismatch, any raw probe-result reference, disposition-hash mismatch after changing confirmation, reason, outcome, or an optional probe hash, Unknown/prohibited tuning values, duplicate configurations, open-ended ranges, and an exact-scope mismatch.

```python
from validate_profile import ProfileValidationError, load_profile, require_capability, validate_project_claim


class ValidateProfileTests(unittest.TestCase):
    def test_partial_profile_is_usable_but_unknown_cannot_satisfy_required(self):
        profile = load_profile(FIXTURES / "profiles" / "valid-partial" / "profile.yaml")
        self.assertEqual("partial", profile["profile_status"])
        with self.assertRaisesRegex(ProfileValidationError, "unproven required capability"):
            require_capability(
                profile,
                "matrix.dot",
                {"dtype": "fp32", "layout": "row_major", "shape": ["M", "N"]},
                "required",
            )

    def test_runtime_identity_mismatch_is_environment_blocked(self):
        profile = load_profile(FIXTURES / "profiles" / "valid-partial" / "profile.yaml")
        with self.assertRaisesRegex(ProfileValidationError, "environment-blocked"):
            validate_project_claim(
                FIXTURES / "claims" / "runtime-mismatch.json",
                profile=profile,
                snapshot={"target_id": "mlu590", "implementation_profile_id": "triton_mlu", "triton_version": "3.6.0", "device_arch": "different-arch"},
            )
```

Add tests that a profile requires `implementation_profile_id`, `implementation_profile_version`, `profile_schema_ref`, `profile_schema_sha256`, `shared_profile_schema_ref`, `shared_profile_schema_version`, `shared_profile_schema_sha256`, `implementation`, `identity_match`, `source_conformance`, `runtime_launcher`, `capability_matrix`, `probe_catalog`, `fallback_and_unknown_policy`, and `profiler_evidence`; permits optional `resource_constraints`, `configuration_constraints`, and `host_lifecycle`; rejects duplicated semantic capability IDs; and requires every evidence item to declare target, toolchain, device, signature, launcher/runner context, definition/result hashes, `review_status: approved`, a root-confined `archived_result_ref`, its SHA-256, and `observed|inferred|unknown` provenance. The Task 3 gate validates archived paths and hashes. Probe-result semantics and approved-scope conservation are Task 4 gates implemented through `validate_probe.py`. The C-like fixture uses semantic contracts such as `memory.copy` with an implementation symbol such as `DataCopy` and proves the schema does not require Triton.

- [x] **Step 2: Run the profile test to verify it fails**

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_validate_profile.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'validate_profile'`.

- [x] **Step 3: Implement profile and claim validation**

Create `validate_profile.py` using `load_json_yaml_document`; it must not import `yaml`. `_validate_probe_catalog()` validates unique probe ids, root-confined relative definition paths, file existence, and recorded SHA-256. An empty catalog is valid for a `partial` profile. Semantic definition validation belongs to `validate_probe_definition()` in Task 4, where the first runnable catalog entry, payload, and runner are added as one unit.

```python
CAPABILITY_STATUSES = frozenset({"supported", "constrained", "unknown", "unsupported", "prohibited"})
HINT_MODALITIES = frozenset({"required", "preferred", "exploratory"})


def load_profile(path: Path) -> dict[str, Any]:
    profile = load_json_yaml_document(path, artifact="implementation profile")
    _validate_required_sections(profile)
    _validate_identity(profile["identity_match"])
    _validate_capability_matrix(profile["capability_matrix"])
    _validate_configuration_constraints(profile.get("configuration_constraints", {}))
    _validate_probe_catalog(profile["probe_catalog"])
    _validate_evidence_scopes(profile)
    return profile


def require_capability(
    profile: Mapping[str, Any],
    contract_name: str,
    signature: Mapping[str, Any],
    modality: str,
) -> dict[str, Any]:
    entry = _match_capability(profile["capability_matrix"], contract_name, signature)
    if modality == "required" and entry["status"] in {"unknown", "unsupported", "prohibited"}:
        raise ProfileValidationError("profile-required-capability-unproven", "unproven required capability")
    return entry
```

`configuration_constraints` defines only reviewed exact-scope legal configuration fields, finite values, temporary-injection method, field kind (`launch-option|compile-time-meta`), and cross-field exclusions. It contains no performance ranking. `validate_configuration_domain()` requires the Decision domain to be a subset, rejects missing/Unknown legality and duplicate normalized configurations, preserves declared deterministic order, and confirms the accepted configuration is explicitly covered as fallback/control. A singleton is valid only when the profile proves that value. Canonical profiles omit unreviewed legality and therefore block submission finalization as `profile-legality-unavailable`; fixture profiles provide synthetic reviewed domains for hardware-free tests.

Write `profiles/triton_mlu/profile.yaml` as valid JSON text with `implementation_profile_id: "triton_mlu"`, `implementation_profile_version: 1`, `profile_status: "partial"`, explicit language/backend/runner/source-analyzer fields, `profile_schema_ref: "schema/profile.schema.json"`, `profile_schema_sha256`, `shared_profile_schema_ref: "schema/shared-profile.schema.json"`, `shared_profile_schema_version`, `shared_profile_schema_sha256`, and identity information sourced from `prompts/coder_targets/triton_mlu.md`. Vendor the global shared schema byte-for-byte into both local schema paths. `load_profile()` compares both recorded hashes and bytes inside the profile root; canonical contract tests compare the shared copy with `skills/kernel-opt-loop/schemas/profile.schema.json`. Record semantic contracts such as `memory.load`, `memory.store`, `index.range`, `parallel.program-id`, `matrix.dot`, `reduction.argmax`, `layout.reshape`, `tensor.zeros`, `resource.num-warps`, `resource.num-stages`, and `resource.vectorize`, with their Triton `implementation_symbol` values. Treat Markdown evidence as non-authoritative human context. Without approved archived probe-result records, capabilities use `unknown` or policy-only `constrained` and carry no observed or inferred support claim.

Copy `skills/kernel-opt-loop/schemas/profile.schema.json` byte-for-byte to both `profiles/triton_mlu/schema/profile.schema.json` and `profiles/triton_mlu/schema/shared-profile.schema.json`. Record each relative path and SHA-256 in `profile.yaml`, and require the two local bytes/hashes to match. Canonical contract tests additionally compare the shared-profile copy with the global source; snapshot validation resolves only the two copied root-confined files. Include positive fixtures with computed digests and negative fixtures for changed local bytes, changed shared-copy bytes, and all-zero recorded hashes; reject them as `profile-schema-hash-mismatch`.

Set `probe_catalog` to an empty array in Task 3. Task 4 adds the definition, payload, semantic validator, runner, catalog entry, and cross-contract tests in one independently executable unit before the profile advertises a versioned probe suite.

Update `triton_mlu.md` to say the Markdown page is explanatory and that machine-readable authority is `profiles/triton_mlu/profile.yaml`; retain its historical evidence descriptions.

- [x] **Step 4: Run profile tests and JSON parsing checks**

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_validate_profile.py -v
python3 -m json.tool skills/kernel-opt-loop/profiles/triton_mlu/profile.yaml >/dev/null
```

Expected: PASS. A partial profile loads; required implementation needs and final-tuning fields must match exact-scope supported/constrained profile facts, and canonical profiles carry no unreviewed configuration legality.

- [x] **Step 5: Commit the profile seam**

```bash
git add skills/kernel-opt-loop/scripts/validate_profile.py \
  skills/kernel-opt-loop/profiles/triton_mlu \
  skills/kernel-opt-loop/references/project-capability-claim-template.json \
  skills/kernel-opt-loop/prompts/coder_targets/triton_mlu.md \
  skills/kernel-opt-loop/tests/test_validate_profile.py \
  skills/kernel-opt-loop/tests/fixtures/vnext/profiles \
  skills/kernel-opt-loop/tests/fixtures/vnext/claims
git commit -m "skills: add machine readable implementation profiles"
```

### Task 4: Execute Pre-Campaign Profile Probes and Render Promotion Candidates

**Files:**
- Create: `skills/kernel-opt-loop/scripts/validate_probe.py`
- Create: `skills/kernel-opt-loop/scripts/run_profile_probe.py`
- Create: `skills/kernel-opt-loop/scripts/render_profile_promotion.py`
- Modify: `skills/kernel-opt-loop/scripts/validate_profile.py`
- Modify: `skills/kernel-opt-loop/tests/test_validate_profile.py`
- Modify: `skills/kernel-opt-loop/profiles/triton_mlu/profile.yaml`
- Create: `skills/kernel-opt-loop/profiles/triton_mlu/probes/basic-memory.json`
- Create: `skills/kernel-opt-loop/profiles/triton_mlu/probes/basic_memory.py`
- Create: `skills/kernel-opt-loop/references/profile-promotion-note-template.md`
- Create: `skills/kernel-opt-loop/tests/test_validate_probe.py`
- Create: `skills/kernel-opt-loop/tests/test_run_profile_probe.py`
- Create: `skills/kernel-opt-loop/tests/test_profile_promotion.py`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/probes/profile/profile.yaml`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/probes/profile/schema/profile.schema.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/probes/profile/schema/shared-profile.schema.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/probes/profile/probes/basic-memory.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/probes/profile/probes/fake-success.py`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/probes/profile/probes/fake-timeout.py`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/probes/profile/probes/fake-nonzero.py`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/probes/profile/probes/fake-malformed.py`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/probes/runtime-snapshot.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/probes/valid-result-payload.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/probes/invalid-scope-promotion.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/probes/qualification/s60/profile.yaml`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/probes/qualification/s60/schema/profile.schema.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/probes/qualification/s60/schema/shared-profile.schema.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/probes/qualification/s60/probes/dot-fp16.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/probes/qualification/s60/probes/fake-dot-success.py`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/probes/qualification/s60/evidence/groupedtopk-result.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/probes/qualification/s60/requirements/attention-dot-before-fallback.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/probes/qualification/s60/requirements/reduction-only.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/probes/qualification/s60/probes/dot-fp16-ambiguous.json`

**Interfaces:**
- Produces `ProbeValidationError`, a `ContractValidationError` subclass.
- Produces `validate_probe_definition(path: Path, *, profile: Mapping[str, Any]) -> dict[str, Any]`.
- Produces `validate_probe_run(run_dir: Path) -> dict[str, Any]`; the run directory contains frozen profile, definition, runtime, optional qualification requirement, and payload inputs, so validation is self-contained after canonical files change.
- Produces `select_profile_probes(profile: Mapping[str, Any], requirements: Sequence[Mapping[str, Any]], runtime_snapshot: Mapping[str, Any]) -> QualificationPlan` as a pure demand selector. The plan contains ordered `selections` plus one disposition per explicit requirement, including `no-exact-probe` when no runner work exists.
- Produces `run_profile_probe(*, profile_path: Path, probe_id: str, target_id: str, runtime_snapshot_path: Path, output_root: Path, qualification_requirement_path: Path | None = None, run_id: str | None = None) -> Path`.
- Produces `render_profile_promotion(run_dir: Path, *, profile_path: Path) -> tuple[Path, Path]`, returning `promotion-candidate.json` and `promotion-note.md` without modifying the profile. For an eligible demand-selected success, the candidate records `onboarding_disposition: promotion-pending`.
- Runner CLI: `run_profile_probe.py --profile <absolute> --probe-id <id> --target-id <id> --runtime-snapshot <absolute> --output-root <absolute> [--qualification-requirement <absolute-normalized-json>] [--run-id <safe-id>]`.
- Renderer CLI: `render_profile_promotion.py --run-dir <absolute-completed-run> --profile <absolute-current-profile>` and prints the two output paths as sorted JSON.
- `target_id` and explicit `run_id` accept only `[A-Za-z0-9._-]+`. When `run_id` is omitted, generate UTC `YYYYMMDDTHHMMSSffffffZ`; any collision fails without retry, overwrite, or reuse.
- Normal execution summaries are exactly `evidence-ready|partial|environment-blocked|probe-failed`; these are not campaign terminal results.

- [x] **Step 1: Write failing probe-runner and promotion tests**

Use complete co-located fixture profile trees only. The S60-named qualification tree is routing-only and uses fixture evidence/payloads; it proves the contract without claiming that a real S60 run occurred. The generic runner tree catalog points to `profile/probes/basic-memory.json`, whose declared input is `fake-success.py`; timeout/nonzero/malformed cases materialize a temporary definition variant and recompute both definition and catalog hashes before launch. The success fixture writes the result payload path supplied by the runner and exits zero. Tests must prove:

- the fixture S60 profile may carry reviewed groupedtopk-derived reduction evidence while exact-scope `matrix.dot` remains `unknown`; absence from groupedtopk never becomes negative dot evidence;
- an attention/MoE `before-fallback` requirement selects only the unique exact-scope dot probe, ignores unrelated Unknowns, returns `no-exact-probe` for zero matches, and rejects two exact matches as ambiguous;
- a reduction-only or `probe_policy: optional` requirement does not trigger a dot probe sweep;
- the profile, definition, target id, runtime snapshot, and absolute interpreter are validated before subprocess launch;
- argv is executed with `shell=False`, a timeout, a confined working directory, and no user-controlled path interpolation outside validated placeholders;
- an existing run id is never overwritten;
- success freezes profile/definition/runtime bytes and every declared payload/build input under `inputs/`, captures stdout/stderr/exit/duration, and records relative path, byte count, and SHA-256 for every input and evidence object;
- timeout, nonzero exit, malformed payload, target/profile mismatch, definition-hash mismatch, and path escape receive stable classifications;
- a successful run creates no `team-state.md`, `rounds/`, Decision, candidate, report, verdict, or benchmark artifact;
- promotion output never changes `profile.yaml`, remains `review_status: proposed`, never recommends `supported` in v1, and rejects scope widening;
- numerically checked selected-dot success yields `promotion-pending`, creates no campaign, and cannot make `require_capability()` pass before an explicit profile commit;
- partial/failure/block/no-match leaves dot `unknown`; any later sum substitution requires an explicit onboarding disposition stating that the primary capability remains Unknown.

```python
class RunProfileProbeTests(unittest.TestCase):
    def test_fixture_probe_finishes_without_campaign_state(self):
        run_dir = run_profile_probe(
            profile_path=PROBES / "profile" / "profile.yaml",
            probe_id="fixture-basic-memory-001",
            target_id="fixture-device",
            runtime_snapshot_path=PROBES / "runtime-snapshot.json",
            output_root=self.output_root,
            run_id="probe-001",
        )
        result = validate_probe_run(run_dir)
        self.assertEqual("evidence-ready", result["summary"])
        self.assertFalse((self.output_root / "team-state.md").exists())
        self.assertFalse((self.output_root / "rounds").exists())

    def test_promotion_is_proposed_and_profile_bytes_do_not_change(self):
        before = self.profile_path.read_bytes()
        candidate_path, note_path = render_profile_promotion(self.run_dir, profile_path=self.profile_path)
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        self.assertEqual("proposed", candidate["review_status"])
        self.assertEqual(before, self.profile_path.read_bytes())
        self.assertTrue(note_path.is_file())
```

- [x] **Step 2: Run the new tests to verify they fail**

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_validate_probe.py -v
python3 -m unittest skills/kernel-opt-loop/tests/test_validate_profile.py -v
python3 -m unittest skills/kernel-opt-loop/tests/test_run_profile_probe.py -v
python3 -m unittest skills/kernel-opt-loop/tests/test_profile_promotion.py -v
```

Expected: FAIL with missing runner, validator, renderer, payload, and fixture errors.

- [x] **Step 3: Implement probe validation, bounded execution, and the real profile payload**

Add `basic-memory.json`, `basic_memory.py`, and the matching path/hash entry to `triton_mlu/profile.yaml` together. The declarative definition invokes the payload directly:

```json
{
  "schema_version": 1,
  "probe_id": "triton-mlu-basic-memory-001",
  "implementation_profile_id": "triton_mlu",
  "family": "core-primitives",
  "purpose": "Compile and execute masked load/store with one-dimensional indexing",
  "capability_ids": ["memory.load.contiguous-fp32", "memory.store.contiguous-fp32", "index.range.one-dimensional", "parallel.program-id.axis0"],
  "scope_kind": "device-family",
  "scope_template": {"dtype": "fp32", "layout": "contiguous", "shape": ["N"]},
  "input_artifacts": [{"path": "probes/basic_memory.py", "sha256": "...", "run_path": "basic_memory.py"}],
  "runner": {
    "kind": "command",
    "argv": ["{interpreter}", "{probe_inputs_root}/basic_memory.py", "--result-json", "{result_payload_path}", "--target-id", "{target_id}", "--runtime-snapshot", "{runtime_snapshot_path}"],
    "cwd": "{probe_run_dir}",
    "timeout_seconds": 120
  },
  "required_runtime_fields": ["interpreter", "device", "toolchain", "device_arch", "runner_adapter", "bootstrap_modules", "synchronize_api"]
}
```

`probe_catalog` records the definition's root-confined relative path and SHA-256. `validate_probe_definition()` validates every `input_artifacts` path/hash/run-path, placeholder, capability id, selection scope, runtime requirement, and timeout. Add a cross-contract test requiring every definition named by a canonical catalog to validate. Matrix probe selection scope includes target/profile/version, runtime/toolchain, device architecture, input and accumulator dtype, layout, tile/shape regime, and launcher context where relevant. Extend `load_profile()` so every approved archived evidence entry validates as a `probe-result` artifact, its archived hash equals the referenced result hash, and approved scope is no broader than the observation scope.

Implement `select_profile_probes()` without filesystem mutation or subprocesses. It evaluates only explicit requirements sorted by `requirement_id`: `must-resolve` selects an exact probe for an Unknown primary; `before-fallback` also requires a supported/constrained algorithm-substitution fallback; `optional` never auto-selects. Exactly one definition must cover the primary signature and current runtime identity. Zero matches return a stable `no-exact-probe` disposition, multiple matches raise `ProbeValidationError("ambiguous-profile-probe-selection")`, and unrelated profile Unknowns are never enumerated into work.

`run_profile_probe.py` must load the canonical profile and its root-confined vendored schema copies, resolve the selected definition from `probe_catalog`, verify its SHA-256, validate `implementation_profile_id`, validate a safe concrete `target_id`, and require every runtime field named by the definition. `runtime-snapshot.json` records an absolute interpreter or executable path, selected device, toolchain identity, device architecture, runner adapter, allowlisted bootstrap modules, and synchronization API; secret environment values are neither accepted nor serialized.

Create `<output-root>/probes/<target-id>/<run-id>/` with exclusive creation. Before execution, atomically copy the validated bytes to `inputs/profile.snapshot.yaml`, `inputs/probe-definition.json`, `inputs/runtime-snapshot.json`, optional `inputs/qualification-requirement.json`, and `inputs/payload/<run_path>` for every declared input artifact; `run.json` hashes these frozen inputs so `validate_probe_run(run_dir)` remains self-contained after canonical files change. Resolve only `{interpreter}`, `{probe_inputs_root}`, `{probe_run_dir}`, `{result_payload_path}`, `{runtime_snapshot_path}`, `{target_id}`, and allowlisted required runtime fields such as `{device}` placeholders, with payload and runtime paths pointing to frozen input copies. Execute the argv array with `subprocess.run(..., shell=False, capture_output=True, timeout=<definition bound>, check=False)`. Write stdout/stderr first, then validate the payload, compute evidence metadata, and atomically write `results/<probe-id>.json` and `run.json`.

Classification is exact: prelaunch target/profile/runtime mismatch, missing interpreter/executable, or unavailable required bootstrap/device/toolchain is `environment-blocked`; a started payload that times out, exits nonzero, emits malformed/mismatched output, or reports compile/execution/correctness failure is `probe-failed`; a valid run with at least one inferred/unknown/unavailable observation is `partial`; only all-declared numerically checked observed success is `evidence-ready`. Invalid profile or definition contracts fail before a completed run is claimed.

The payload result contains the same probe/profile/target ids, observed scope, and one observation per declared capability id. The runner wrapper adds profile/definition/runtime hashes and execution/evidence provenance. A result may be `observed|inferred|unknown`; only an observed, numerically checked success is eligible for a support recommendation.

Create `profiles/triton_mlu/probes/basic_memory.py` as a real file-backed MLU probe that accepts `--result-json`, imports the matched MLU runtime, compiles and executes masked contiguous fp32 load/store/indexing, checks output numerically, and writes the normalized payload. Unit tests never execute this hardware payload and no committed evidence claims that it passed.

- [x] **Step 4: Implement promotion rendering and run all probe tests**

`render_profile_promotion.py` first calls `validate_probe_run()`, rejects `environment-blocked` and `probe-failed` runs plus partial runs with no observed fact, loads the unchanged canonical profile, and derives recommendations by capability id. It records current status, recommended status, exact source scope, probe-definition hash, result hash, evidence refs, rationale, unresolved gaps, and `review_status: proposed`. The v1 renderer may recommend only `constrained`, unchanged status, or additional evidence; it never recommends `supported`. It rejects a recommendation whose target/toolchain/device/shape/dtype/layout/launcher scope is broader than the validated result. Render the Markdown note from the candidate JSON so the JSON remains authoritative. When the run contains a `before-fallback` qualification requirement and produces eligible observed success, return the onboarding disposition `promotion-pending`; this disposition stops before Phase 0 and cannot be converted into fallback eligibility without an explicit maintainer decline/defer record.

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_validate_probe.py -v
python3 -m unittest skills/kernel-opt-loop/tests/test_validate_profile.py -v
python3 -m unittest skills/kernel-opt-loop/tests/test_run_profile_probe.py -v
python3 -m unittest skills/kernel-opt-loop/tests/test_profile_promotion.py -v
python3 -m json.tool skills/kernel-opt-loop/profiles/triton_mlu/probes/basic-memory.json >/dev/null
python3 skills/kernel-opt-loop/scripts/run_profile_probe.py --help >/dev/null
python3 skills/kernel-opt-loop/scripts/render_profile_promotion.py --help >/dev/null
```

Expected: PASS without accelerator access. The fixture runner produces the documented layout and no test mutates a canonical profile or creates campaign state.

- [x] **Step 5: Commit the profile-probe lifecycle**

```bash
git add skills/kernel-opt-loop/scripts/validate_probe.py \
  skills/kernel-opt-loop/scripts/run_profile_probe.py \
  skills/kernel-opt-loop/scripts/validate_profile.py \
  skills/kernel-opt-loop/scripts/render_profile_promotion.py \
  skills/kernel-opt-loop/profiles/triton_mlu/profile.yaml \
  skills/kernel-opt-loop/profiles/triton_mlu/probes \
  skills/kernel-opt-loop/references/profile-promotion-note-template.md \
  skills/kernel-opt-loop/tests/test_validate_probe.py \
  skills/kernel-opt-loop/tests/test_validate_profile.py \
  skills/kernel-opt-loop/tests/test_run_profile_probe.py \
  skills/kernel-opt-loop/tests/test_profile_promotion.py \
  skills/kernel-opt-loop/tests/fixtures/vnext/probes
git commit -m "skills: add implementation profile probes"
```

### Task 5: Integrate vNext Decisions, Artifact References, and Context Naming

**Files:**
- Modify: `skills/kernel-opt-loop/scripts/validate_decision.py`
- Modify: `skills/kernel-opt-loop/references/decision-template.md`
- Modify: `skills/kernel-opt-loop/references/team-state-template.md`
- Modify: `skills/kernel-opt-loop/references/project-template.md`
- Modify: `skills/kernel-opt-loop/references/role-context-template.md`
- Modify: `skills/kernel-opt-loop/references/invariants.md`
- Modify: `skills/kernel-opt-loop/tests/test_validate_decision.py`
- Modify: `skills/kernel-opt-loop/tests/test_contracts.py`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/decisions/valid-vnext.md`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/decisions/invalid-sketch-hash.md`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/decisions/valid-explicit-fallback.md`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/decisions/invalid-silent-fallback.md`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/decisions/valid-final-tuning.md`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/decisions/invalid-final-tuning-semantic-field.md`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/decisions/invalid-final-tuning-profile-domain.md`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/projects/README.md`

**Interfaces:**
- Extends `validate_decision(decision: Path, expected_profile: str | None = None, *, project_root: Path | None = None, expected_implementation_profile: str | None = None) -> dict[str, Any]` without breaking schema-v1 callers. Schema-v1 uses `expected_profile`; schema-v2 requires `expected_implementation_profile`, and passing both is invalid.
- For schema-v2, returns `decision_kind`, `sketch_ref`, `sketch_sha256`, `implementation_profile_snapshot_ref`, `implementation_profile_snapshot_sha256`, `project_capability_claim_ref`, `project_capability_claim_sha256`, optional normalized `fallback_provenance`, optional normalized `final_tuning_contract`, and `causal_graph` fields. Phase 0 copies the entire validated canonical profile directory to `state/implementation_profile_snapshot/`; the snapshot ref points to its `profile.yaml`, and all local schema, probe definition/input, and approved evidence/attachment references must remain inside that copied root and validate by hash.
- Schema-v1 uses the D/O/C/H parser. Schema-v2 requires the JSON Sketch reference and treats Markdown rendering as non-authoritative.

- [x] **Step 1: Write failing legacy-safe and vNext Decision tests**

Add a helper that creates a temporary project root with `rounds/`, `state/`, `baseline_adapter.py`, `project.md`, `rounds/report_000.md`, a frozen implementation-profile snapshot closure, claim, and Sketch fixture. Fixture tests materialize every referenced file and hash. Add a closure test that loads the copied snapshot with no canonical profile path available and proves every schema, probe catalog/input, and approved evidence reference resolves under `state/implementation_profile_snapshot/`.

```python
def materialize_vnext_project(root: Path) -> Path:
    import hashlib
    import shutil

    fixture_root = Path(__file__).parent / "fixtures" / "vnext"
    (root / "rounds").mkdir()
    (root / "state").mkdir()
    (root / "baseline_adapter.py").write_text("class ModelNew: pass\n", encoding="utf-8")
    (root / "rounds" / "report_000.md").write_text("# Report 000\n", encoding="utf-8")
    (root / "project.md").write_text("# Project\n\n## runtime-fingerprint\n\nfixture\n", encoding="utf-8")
    shutil.copyfile(fixture_root / "sketches" / "valid-kernel.json", root / "rounds" / "sketch_001.json")
    shutil.copytree(fixture_root / "profiles" / "valid-partial", root / "state" / "implementation_profile_snapshot")
    shutil.copyfile(fixture_root / "claims" / "valid-claim.json", root / "state" / "project_capability_claim.json")
    decision = root / "rounds" / "decision_001.md"
    text = (fixture_root / "decisions" / "valid-vnext.md").read_text(encoding="utf-8")
    replacements = {
        "__SKETCH_SHA256__": hashlib.sha256((root / "rounds" / "sketch_001.json").read_bytes()).hexdigest(),
        "__PROFILE_SHA256__": hashlib.sha256((root / "state" / "implementation_profile_snapshot" / "profile.yaml").read_bytes()).hexdigest(),
        "__CLAIM_SHA256__": hashlib.sha256((root / "state" / "project_capability_claim.json").read_bytes()).hexdigest(),
    }
    for marker, digest in replacements.items():
        text = text.replace(marker, digest)
    decision.write_text(text, encoding="utf-8")
    return decision


def test_vnext_decision_requires_existing_hashed_sketch(self):
    with tempfile.TemporaryDirectory() as directory:
        decision = materialize_vnext_project(Path(directory))
        result = validate_decision(decision, project_root=decision.parents[1], expected_implementation_profile="triton_mlu")
        self.assertEqual("rounds/sketch_001.json", result["sketch_ref"])
        self.assertTrue(result["valid"])
```

Add negative tests for missing reference file, wrong SHA-256, missing `project.md#runtime-fingerprint` heading, schema-v2 using a legacy text Sketch as authority, profile-snapshot mismatch, invalid causal edge, an algorithm-substitution fallback missing primary/fallback signatures or onboarding disposition, an unresolved/`promotion-pending` fallback, and a v1 fixture continuing to normalize as before. Final-tuning fixtures cover immutable accepted candidate/binding/Sketch/profile/claim/measurement/harness/base hashes, a finite deterministically ordered domain, accepted-config fallback, budget, warmup/repeat, mutation reset, selection/tie rules, and rejection of open-ended ranges, duplicate configurations, missing/profile-illegal/Unknown legality, stale anchors, missing control, and fields that alter algorithm, dataflow, precision, effects, aliases, Host Plan, public interface, or semantic layout.

- [x] **Step 2: Run Decision tests to verify failures**

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_validate_decision.py -v
```

Expected: FAIL with missing `project_root`, referenced-file, hash, anchor, and vNext artifact validation.

- [x] **Step 3: Extend Decision validation without changing v1 behavior**

Use `parse_sketch()` as the schema-v1 code path and add a schema-version branch after Metadata parsing:

```python
def validate_decision(
    decision: Path,
    expected_profile: str | None = None,
    *,
    project_root: Path | None = None,
    expected_implementation_profile: str | None = None,
) -> dict[str, Any]:
    sections = extract_sections(decision.read_text(encoding="utf-8"))
    metadata_section = sections["Metadata"]
    metadata = parse_single_json_block(metadata_section)
    root = _resolve_project_root(decision, project_root)
    if metadata.get("schema_version") == 1:
        if expected_implementation_profile is not None:
            raise DecisionValidationError("implementation-profile-v1-invalid", "schema-v1 uses expected_profile")
        _validate_metadata_v1(metadata, metadata_section, expected_profile)
        return _validate_v1_decision(sections, metadata, expected_profile)
    if metadata.get("schema_version") == 2:
        if expected_profile is not None or expected_implementation_profile is None:
            raise DecisionValidationError("implementation-profile-v2-required", "schema-v2 requires expected_implementation_profile only")
        _validate_metadata_v2(metadata, metadata_section, expected_implementation_profile)
        _validate_referenced_artifacts(root, metadata, sections)
        return _validate_vnext_decision(root, sections, metadata, expected_implementation_profile)
    raise DecisionValidationError("metadata-schema-version", "schema_version must be 1 or 2", metadata_section.line)
```

For schema-v2 Metadata require these exact keys alongside the identity fields:

```json
{
  "decision_kind": "optimization",
  "sketch_ref": "rounds/sketch_001.json",
  "sketch_sha256": "7b2d6b7ed8ac4f4b8d0a1a6b57d96d4a4e05b9c47d0f17cf0c2d7782f5d4a1c3",
  "implementation_profile_snapshot_ref": "state/implementation_profile_snapshot/profile.yaml",
  "implementation_profile_snapshot_sha256": "b03b0b53f0d45d2478b9f2f7fd0ea8d9a5e9b5f7b355ac8d8a4ed6f8c8d9e0a1",
  "project_capability_claim_ref": "state/project_capability_claim.json",
  "project_capability_claim_sha256": "a92d13e2b7d8c6f54b198a40f4234ae5711e9c2346d1766c7b1da3b98f7cc102"
}
```

Require the `## Unified Sketch` section to contain one JSON object with `artifact`, `sha256`, and optional `rendering`; compare both first two fields to Metadata and invoke `validate_sketch()`. Require `## Evaluation Contract` to carry `causal_graph.nodes` and `causal_graph.edges`. Task 5 checks graph structure; Task 7 evaluates verdict rules. When the Decision uses an algorithm substitution for a previously Unknown primary, require `fallback_provenance` with `fallback_from`, exact primary/fallback signatures, `fallback_kind`, `probe_policy`, `qualification_disposition_id`, `qualification_disposition_sha256`, `primary_remains_unknown: true`, and expected causal/performance consequence. Compute `qualification_disposition_sha256` with `sha256_canonical_json()` over the complete embedded disposition object (which has no self-hash field). Cross-check the id/hash and signatures against the embedded maintainer-authorized disposition in the validated project claim; reject `unresolved|promotion-pending`, missing confirmation, or any raw probe-result ref.

For `decision_kind: final-autotune`, Metadata uses `artifact_index` matching the filename and omits campaign `round`. Require a `## Final Configuration Tuning` JSON object with canonical `submission_snapshot_id`; accepted candidate/binding, Sketch, profile, project-claim, runtime-snapshot, official-measurement, harness, and base/reference refs/hashes; finite ordered `configurations`; accepted `fallback_configuration`; `max_trials`, `max_wall_seconds`, search warmup/repeat settings, mutation-reset policy, comparison metric, deterministic winner/tie rule, and `pin_selected_config: true`. Recompute the snapshot ID, invoke `validate_configuration_domain()`, block missing/Unknown legality, and require every changed field to name an accepted-Sketch `preferred|exploratory` launch/meta hint classified as configuration-only. Reject any field or configuration that changes semantic operations, dependency/fusion boundaries, precision, effects, aliases, Host Plan, public interface, or semantic layout.

Replace `_validate_reference()`'s format-only behavior with a helper that checks a root-confined file exists. For required anchors, parse H2 headings and reject absent exact anchors. Preserve source line numbers in `DecisionValidationError`.

Update templates and state fields:

```yaml
schema_version: 2
skill_version: 3.0.0
contract_version: 3
semantic_contract: typed-sketch-v1
attribution_contract: verdict-v1
implementation_profile_snapshot_ref: null
implementation_profile_snapshot_sha256: null
project_capability_claim_ref: null
project_capability_claim_sha256: null
last_completed_sketch: null
last_completed_binding: null
last_completed_verdict: null
last_attribution: null
```

Keep `team-state.md` free of finalization bookkeeping fields. Finalization is discovered by scanning `decision_kind: final-autotune` artifacts and matching the canonical `submission_snapshot_id` or a current accepted candidate/binding already recorded as a final output under unchanged anchors. `artifact_index` never updates campaign-round pointers.

Use only `state/designer_context.md`, `state/coder_context.md`, and `state/verifier_context.md` in `invariants.md`, templates, and tests. Delete references to `*_state.md`; do not create compatibility aliases.

- [x] **Step 4: Run Decision and durable-contract tests**

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_validate_decision.py -v
python3 -m unittest skills/kernel-opt-loop/tests/test_contracts.py -v
```

Expected: PASS. Schema-v1 fixture behavior remains unchanged; schema-v2 validates ordinary and final-autotune Decisions, rejects stale/mismatched anchors, and admits only finite profile-legal config-only domains.

- [x] **Step 5: Commit Decision and durable-state integration**

```bash
git add skills/kernel-opt-loop/scripts/validate_decision.py \
  skills/kernel-opt-loop/references/decision-template.md \
  skills/kernel-opt-loop/references/team-state-template.md \
  skills/kernel-opt-loop/references/project-template.md \
  skills/kernel-opt-loop/references/role-context-template.md \
  skills/kernel-opt-loop/references/invariants.md \
  skills/kernel-opt-loop/tests/test_validate_decision.py \
  skills/kernel-opt-loop/tests/test_contracts.py \
  skills/kernel-opt-loop/tests/fixtures/vnext/decisions \
  skills/kernel-opt-loop/tests/fixtures/vnext/projects
git commit -m "skills: gate vnext decisions on typed artifacts"
```

### Task 6: Add Deterministic Candidate Binding Conformance

**Files:**
- Create: `skills/kernel-opt-loop/scripts/validate_binding.py`
- Create: `skills/kernel-opt-loop/tests/test_validate_binding.py`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/candidates/valid_candidate.py`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/bindings/valid-many-to-many.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/bindings/missing-required-statement.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/bindings/stale-candidate-hash.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/bindings/invalid-source-primitive.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/candidates/final-tuning-pinned.py`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/bindings/final-tuning-pinned.json`

**Interfaces:**
- Produces `BindingValidationError`.
- Produces `validate_binding(binding_path: Path, *, project_root: Path, sketch_result: Mapping[str, Any], profile: Mapping[str, Any], candidate_path: Path, accepted_candidate_path: Path | None = None, final_tuning_contract: Mapping[str, Any] | None = None) -> dict[str, Any]`; the optional pair is required together for final tuning.
- Returns `source_analyzer`, `binding_model`, `coverage`, `source_symbols`, `required_hint_bindings`, and normalized `bindings`.
- Supports `implemented-by`, `fused-into`, `expanded-into`, and `elided-by`; all multi-source or multi-statement relationships require nonempty `reason`.
- Selects the analyzer from `profile.source_conformance`. The first production adapter is `python-ast-triton`; an unavailable declared analyzer fails as `profile-source-analyzer-unavailable`, not as a Python syntax or candidate error.

- [x] **Step 1: Write failing source-analyzer and binding tests**

Use a fixture candidate that is syntactically ordinary Python and does not import Triton at test time:

```python
class tl:
    @staticmethod
    def load(pointer, mask):
        return pointer

    @staticmethod
    def store(pointer, value, mask):
        return None


def kernel(scores, output, token, expert, e):
    row = tl.load(scores, mask=expert < e)
    tl.store(output, row, mask=expert < e)
```

Test exact source span validation, candidate hash validation, semantic contract to implementation-symbol discovery from Python AST call expressions, full required statement coverage, missing coverage, stale hash, nonmatching declared implementation symbol, valid `fused-into` relation, invalid multi-relation without reason, and `required` hint capability validation. Final-tuning cases prove that pinning an authorized configuration changes the candidate hash, requires a fresh binding and source spans, preserves statement coverage and semantic symbols, and rejects a stale accepted-candidate binding or any pin that introduces a new operation, fusion boundary, precision, effect, alias, Host Plan, public interface, or semantic layout. Add a non-Triton profile declaring `fixture-clike-symbols` and assert the first implementation returns the stable unavailable-analyzer classification without calling `ast.parse` on C-like source.

```python
from validate_binding import BindingValidationError, validate_binding


def test_valid_many_to_many_binding_covers_every_required_statement(self):
    result = validate_binding(
        BINDINGS / "valid-many-to-many.json",
        project_root=PROJECT_ROOT,
        sketch_result=VALID_SKETCH,
        profile=VALID_PROFILE,
        candidate_path=CANDIDATES / "valid_candidate.py",
    )
    self.assertEqual({"op.load.row", "op.store.output"}, set(result["coverage"]))
    self.assertEqual("python-ast-triton", result["source_analyzer"])
    self.assertIn("tl.load", result["source_symbols"])
```

- [x] **Step 2: Run the binding tests to verify they fail**

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_validate_binding.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'validate_binding'`.

- [x] **Step 3: Implement source-level conformance only**

Create `validate_binding.py` with a small analyzer dispatch selected from `profile["source_conformance"]["analyzer"]`. Implement only `python-ast-triton` in vNext: it parses the candidate with `ast.parse` and collects dotted call names with one-based `(lineno, col_offset + 1, end_lineno, end_col_offset + 1)` spans. Do not import, compile, run, or profile the candidate. Unknown analyzers fail before reading candidate syntax and leave the profile incomplete for automatic binding.

```python
RELATIONS = frozenset({"implemented-by", "fused-into", "expanded-into", "elided-by"})


def validate_binding(
    binding_path: Path,
    *,
    project_root: Path,
    sketch_result: Mapping[str, Any],
    profile: Mapping[str, Any],
    candidate_path: Path,
    accepted_candidate_path: Path | None = None,
    final_tuning_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    binding = load_json_document(binding_path, artifact="binding")
    _validate_hashes(binding, candidate_path, sketch_result)
    _validate_final_tuning_diff(candidate_path, accepted_candidate_path, final_tuning_contract)
    analyzer_name, binding_model, analyzer = _select_source_analyzer(profile)
    source = candidate_path.read_text(encoding="utf-8")
    source_symbols = analyzer(source)
    normalized = _validate_bindings(binding["bindings"], sketch_result, source_symbols, source, profile)
    _validate_required_coverage(normalized, sketch_result["statement_index"])
    _validate_required_hints(binding, sketch_result, profile)
    return {"valid": True, "source_analyzer": analyzer_name, "binding_model": binding_model, "coverage": _coverage(normalized), "source_symbols": source_symbols, "bindings": normalized}
```

Implement exact behavior:

- `candidate_path` must be root-confined and match `binding.candidate_path`; `candidate_sha256`, `sketch_sha256`, and Decision hash must match supplied artifacts;
- every binding references a Sketch `statement_id`, known `relation`, semantic `contract_name`, profile-mapped `implementation_symbol`, and source span;
- under `python-ast-triton`, the declared implementation symbol must appear as an AST call at the exact span, except `elided-by`, which must name an existing replacement statement and reason;
- one source span may bind multiple statements and one statement may bind multiple source spans, but cardinality above one requires a nonempty `reason`;
- every `load`, `compute`, and `store` operation in the Sketch is covered;
- `required` hints must carry a binding record and pass `require_capability`; `preferred` accommodations must be explicitly recorded; `exploratory` hints only require observation intent;
- reject bindings to `base.py`, the harness, decision artifacts, or paths outside candidate ownership;
- for final tuning, require `artifact_kind: submission-finalization` and Decision-matching `artifact_index` with no campaign `round`; compare accepted and pinned candidates with a profile/analyzer-selected normalized source diff so only Decision-authorized launch/meta values may change, while implementation symbols, calls, control/dataflow structure, signatures, effects, and bound statement coverage remain identical.

- [x] **Step 4: Run binding, Sketch, and profile tests**

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_validate_binding.py -v
python3 -m unittest skills/kernel-opt-loop/tests/test_validate_sketch.py -v
python3 -m unittest skills/kernel-opt-loop/tests/test_validate_profile.py -v
```

Expected: PASS. The checker proves source conformance and config-only pinning against fresh hashes; observed compiled structure remains a Verifier claim.

- [x] **Step 5: Commit the Coder conformance gate**

```bash
git add skills/kernel-opt-loop/scripts/validate_binding.py \
  skills/kernel-opt-loop/tests/test_validate_binding.py \
  skills/kernel-opt-loop/tests/fixtures/vnext/candidates \
  skills/kernel-opt-loop/tests/fixtures/vnext/bindings
git commit -m "skills: check sketch source bindings"
```

### Task 7: Add Causal Graph, Verifier Fact Pack, and Verdict Rules

**Files:**
- Create: `skills/kernel-opt-loop/scripts/validate_verdict.py`
- Create: `skills/kernel-opt-loop/tests/test_validate_verdict.py`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/reports/valid-report.md`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/reports/missing-observable.md`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/verdicts/design-error.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/verdicts/code-error.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/verdicts/lowering-unknown.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/verdicts/evidence-gap.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/reports/valid-final-tuning-report.md`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/verdicts/final-tuning-submission-ready.json`
- Modify: `skills/kernel-opt-loop/references/report-template.md`

**Interfaces:**
- Produces `VerdictValidationError`.
- Produces `validate_causal_graph(graph: Mapping[str, Any], *, intervention: str, observable_names: Collection[str]) -> None`.
- Produces `extract_verifier_fact_pack(report_path: Path) -> dict[str, Any]`.
- Produces `select_final_tuning_configuration(contract: Mapping[str, Any], trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]` as a pure deterministic selector.
- Produces `resolve_finalization_slot(project_root: Path, submission_snapshot_id: str) -> FinalizationSlot`, which validates completed verdicts and resumable Decisions before allocating or resuming an artifact index.
- Produces `apply_submission_promotion(state: Mapping[str, Any], verdict: Mapping[str, Any]) -> dict[str, Any]`, a pure atomic update of the existing accepted kernel/report pair that never touches campaign-round or counter fields.
- Produces `validate_verdict(verdict_path: Path, *, inputs: Mapping[str, Any]) -> dict[str, Any]`.
- `inputs` contains normalized Decision, Sketch, binding, profile, and fact-pack results; no verdict function opens candidate code or profiler traces directly.

- [x] **Step 1: Write failing attribution-rule tests**

Add fixtures whose `## vNext Fact Pack` contains exactly one `json` fence. Test each deterministic rule and its terminal/counter mapping.

```python
from validate_verdict import (
    VerdictValidationError,
    extract_verifier_fact_pack,
    validate_causal_graph,
    validate_verdict,
)


def test_lowering_unknown_requires_all_static_gates_and_absent_observed_mechanism(self):
    report = extract_verifier_fact_pack(REPORTS / "valid-report.md")
    result = validate_verdict(
        VERDICTS / "lowering-unknown.json",
        inputs={
            "decision": VALID_DECISION,
            "sketch": VALID_SKETCH,
            "binding": VALID_BINDING,
            "profile": VALID_PROFILE,
            "facts": report,
        },
    )
    self.assertEqual("lowering-unknown", result["classification"])
    self.assertEqual("design-rejected", result["terminal_result"])
    self.assertEqual("unchanged", result["failed_attempt_effect"])
```

Add tests that:

- an invalid or disconnected causal graph is `DESIGN.CAUSAL.INVALID`;
- a missing binding is `CODE.BINDING.MISSING` and allows one repair;
- a failed correctness fact is `CODE.CORRECTNESS.FAIL` and terminates `candidate-failed` after repair exhaustion;
- `lowering-unknown` is rejected when static gates did not pass;
- a missing required observable is `EVIDENCE.OBSERVABLE.MISSING`, maps to `blocked` when fact-pack cause is environment, and maps to `design-rejected` with classification `design-error` when the Decision omitted a measurable observable definition;
- a correct mechanism-improved candidate with insufficient accepted-to-candidate e2e improvement produces `no-improvement` and `classification: none`;
- verdict artifact hashes and precondition pass/fail/missing entries must match inputs;
- final-tuning trials match the Decision domain and deterministic order, stay within trial/time budgets, carry compile/correctness/reset/comparable-measurement eligibility, and include the accepted fallback/control;
- the pure selector applies the declared metric and tie order deterministically, rejecting duplicate/missing trials, undeclared adaptive expansion, stale fingerprints, and ambiguous ties;
- `submission-ready` requires the selected configuration to equal the pinned values or accepted fallback/control, validated binding on the exact final candidate hash, post-pin correctness, lowering, promotion, and official competition facts on that hash;
- search-only measurements, a failed pin/confirmation, or missing post-pin official evidence cannot authorize submission, and finalization verdicts reject attribution, terminal-result, counter-effect, and run-policy fields;
- finalization-slot tests cover fresh allocation, same-ID Decision-only resume, same-ID Decision plus sealed report resume, invalid/incomplete report block, conflicting Decision hash block, completed input-ID rejection, and current accepted candidate/binding matching a prior final output;
- submission-promotion tests prove an improved winner atomically changes both accepted kernel/report pointers, fallback-retained changes neither, `last_accepted_round` and campaign counters remain byte-for-byte unchanged, and any partial pair update is rejected.

- [x] **Step 2: Run verdict tests to verify they fail**

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_validate_verdict.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'validate_verdict'`.

- [x] **Step 3: Implement structured facts and rule-precondition validation**

Add this exact fact-pack shape to `report-template.md` under `## vNext Fact Pack`:

```json
{
  "schema_version": 1,
  "candidate_sha256": "88c41c1f1d6ee5fb35a55f1f8638f3dd3f4b27c63a4a2d91b54f5b9a6d8c7e31",
  "correctness": {
    "status": "pass",
    "evidence": ["python3 auto_bench.py --check-correctness --v1_file candidate.py"]
  },
  "observables": [
    {
      "name": "external-kernel-count",
      "status": "observed",
      "value": "3 -> 2",
      "confidence": "high",
      "evidence": ["log/profiler_candidate_summary.json"]
    }
  ],
  "lowering": {
    "status": "observed",
    "expected_mechanism": "absent",
    "evidence_contract": "mlu-kernel-summary-v1",
    "evidence": ["log/profiler_candidate_summary.json"]
  },
  "evidence_gap_cause": "none"
}
```

For `decision_kind: final-autotune`, report and verdict metadata require `artifact_kind: submission-finalization` and the Decision-matching `artifact_index`, and forbid campaign `round`. Verifier writes the report atomically only after pinning or accepted-fallback confirmation and final verification. The same fact pack adds `final_configuration_tuning` with `submission_snapshot_id`, immutable contract hashes, `search_trials`, selected configuration and selector rule, `selection_outcome: improved|fallback-retained`, proof that temporary storage contains no derived candidate source, final candidate/binding hashes, and `post_pin_official` correctness/lowering/promotion/competition facts. Each search trial contains only the declared configuration, order index, compile/correctness/reset status, comparable measurement count/statistic, eligibility, and normalized rejection code; raw output remains under the existing gitignored runtime boundary. `select_final_tuning_configuration()` includes the accepted configuration as deterministic fallback, enforces budgets and declared order, and returns no selection when eligibility or comparison preconditions fail. Orchestrator reruns the selector from the sealed report and checks its result against the exact final source values.

Create `validate_verdict.py` with a declarative rule table. The validator must not let free-form explanation select a class.

```python
RULES = {
    "DESIGN.SKETCH.INVALID": {"classification": "design-error", "terminal_result": "design-rejected", "failed_attempt_effect": "increment"},
    "DESIGN.CAUSAL.INVALID": {"classification": "design-error", "terminal_result": "design-rejected", "failed_attempt_effect": "increment"},
    "CODE.BINDING.MISSING": {"classification": "code-error", "terminal_result": "candidate-failed", "failed_attempt_effect": "increment", "repairable": True},
    "CODE.BINDING.VIOLATION": {"classification": "code-error", "terminal_result": "candidate-failed", "failed_attempt_effect": "increment", "repairable": True},
    "CODE.CORRECTNESS.FAIL": {"classification": "code-error", "terminal_result": "candidate-failed", "failed_attempt_effect": "increment"},
    "LOWERING.EXPECTED.ABSENT": {"classification": "lowering-unknown", "terminal_result": "design-rejected", "failed_attempt_effect": "unchanged"},
    "EVIDENCE.OBSERVABLE.MISSING": {
        "classification": "evidence-gap",
        "terminal_result": "blocked",
        "failed_attempt_effect": "unchanged",
        "cause_overrides": {
            "decision": {"classification": "design-error", "terminal_result": "design-rejected", "failed_attempt_effect": "increment"}
        },
    },
}
```

Entries with `repairable: true` define the post-repair terminal branch only; their table `terminal_result` and `failed_attempt_effect` values must not be applied to the first repair route. A first occurrence may produce `route: coder-repair`, `repair_exhausted: false`, and `terminal_result: null` with no counter effect; the table's terminal result and counter effect apply only when `repair_exhausted: true`. For evidence gaps, require `bounded_probe_attempted: true`; the table's `blocked`/counter-neutral values are the default for fact-pack cause `environment`, while `cause_overrides["decision"]` explicitly resolves the omitted-measurable-observable case to `design-error`/`design-rejected`/increment.

A `final-autotune` verdict uses a separate schema branch with `route: submission-ready|blocked`; it rejects `classification`, `terminal_result`, `failed_attempt_effect`, and every run-policy projection field. `submission-ready` requires deterministic selection, no derived source in temporary storage, config-only pin conformance or accepted-source confirmation, validated binding, full correctness, required promotion evidence, and official competition evidence on the exact final hash. Improved and fallback-retained selections use the same pure submission-promotion predicate. The former atomically advances `last_accepted_kernel` and `last_accepted_report` to the exact final source and sealed report while preserving `last_accepted_round`; the latter changes neither pointer, and a partial pair update is invalid. `resolve_finalization_slot()` rejects a prior matching `submission_snapshot_id` and a current accepted candidate/binding already recorded as final output under unchanged anchors.

- [x] **Step 4: Run verdict and report-template tests**

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_validate_verdict.py -v
python3 -m unittest skills/kernel-opt-loop/tests/test_contracts.py -v
```

Expected: PASS. Ordinary reports/verdicts retain deterministic attribution; finalization uses a separate counter-free verdict branch for bounded selection and exact-source submission verification.

- [x] **Step 5: Commit deterministic attribution**

```bash
git add skills/kernel-opt-loop/scripts/validate_verdict.py \
  skills/kernel-opt-loop/references/report-template.md \
  skills/kernel-opt-loop/tests/test_validate_verdict.py \
  skills/kernel-opt-loop/tests/fixtures/vnext/reports \
  skills/kernel-opt-loop/tests/fixtures/vnext/verdicts
git commit -m "skills: add deterministic attribution verdicts"
```

### Task 8: Apply Attribution Effects to Run Policy and Contracts

**Files:**
- Modify: `skills/kernel-opt-loop/scripts/evaluate_run_policy.py`
- Modify: `skills/kernel-opt-loop/tests/test_run_policy.py`
- Modify: `skills/kernel-opt-loop/SKILL.md`
- Modify: `skills/kernel-opt-loop/prompts/designer.md`
- Modify: `skills/kernel-opt-loop/prompts/coder.md`
- Modify: `skills/kernel-opt-loop/prompts/verifier.md`
- Modify: `skills/kernel-opt-loop/references/invariants.md`
- Modify: `skills/kernel-opt-loop/references/team-state-template.md`
- Modify: `skills/kernel-opt-loop/tests/test_contracts.py`

**Interfaces:**
- Extends `evaluate_terminal(state: Mapping[str, Any], result: str, *, target_reached: bool = False, user_stop_requested: bool = False, attribution: str | None = None, failed_attempt_effect: str | None = None) -> dict[str, Any]`.
- New vNext campaign-terminal calls require the verdict's `failed_attempt_effect`; finalization verdicts never call this interface, and legacy calls retain current `FAILED_RESULTS` behavior when both optional arguments are omitted.
- `evaluate_terminal()` output includes `attribution` and `failed_attempt_effect` when supplied.

- [x] **Step 1: Write failing policy and contract assertions**

Add tests to `test_run_policy.py` for the explicit exception to the legacy `design-rejected` rule:

```python
def test_lowering_unknown_design_rejection_does_not_increment_failed_streak(self):
    result = evaluate_terminal(
        state(failed_attempt_streak=1),
        "design-rejected",
        attribution="lowering-unknown",
        failed_attempt_effect="unchanged",
    )
    self.assertEqual(1, result["failed_attempt_streak"])
    self.assertEqual("lowering-unknown", result["attribution"])


def test_explicit_design_error_still_increments_failed_streak(self):
    result = evaluate_terminal(
        state(failed_attempt_streak=1),
        "design-rejected",
        attribution="design-error",
        failed_attempt_effect="increment",
    )
    self.assertEqual(2, result["failed_attempt_streak"])
```

Add contract assertions that profile onboarding and each campaign role name exactly their vNext artifact ownership:

- Before campaign state exists, Orchestrator owns pre-campaign profile onboarding and may write only its normalized qualification input plus isolated probe run, results, evidence, promotion candidate, and note; it invokes selector/runner/validator/renderer, reports completion, creates no campaign state, and never edits `profile.yaml`.
- Orchestrator probes only explicit `must-resolve|before-fallback` requirements, rejects ambiguous exact matches, stops on `promotion-pending`, and requires maintainer promotion or a confirmed fallback disposition embedded in the project claim before Phase 0; it never stores a raw probe-result ref in campaign state.
- Orchestrator materializes the initial project capability claim and frozen implementation-profile snapshot in Phase 0, then owns verdict, manifest, pointers, and commits.
- In capability preflight, Designer returns explicit optimization-critical primary/fallback pairs from operator semantics without writing campaign files; Orchestrator validates/materializes the normalized qualification input. After Phase 0, Designer writes Decision, Sketch, causal graph, and explicit fallback provenance but no runtime fact pack or verdict; it never equates Unknown with unavailable.
- Coder writes only Decision-local probe references and the binding ledger; it does not own pre-campaign qualification, the canonical profile, the initial project claim, or a Verdict.
- Verifier writes the fact pack inside the report and observed lowering but no design/code blame, profile mutation, or promotion approval.
- Final tuning begins only after normal campaign termination with an accepted fingerprint-stable submission snapshot and reviewed exact-scope configuration legality. Designer freezes one config-only Decision; Verifier evaluates injected configurations against one accepted source and returns normalized trials without a persisted selection artifact; Orchestrator selects; Coder pins once or confirms fallback; Verifier seals the report after final official verification; Orchestrator reselects and routes.
- `submission_snapshot_id` uses the complete canonical anchor set. Finalization allocates or resumes an artifact-slot index without changing campaign-round pointers, terminal fields, attribution, or counters. `team-state.md` gains no finalization-specific field, and temporary storage contains no derived candidate source.

- [x] **Step 2: Run policy and contract tests to verify failures**

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_run_policy.py -v
python3 -m unittest skills/kernel-opt-loop/tests/test_contracts.py -v
```

Expected: FAIL with missing attribution arguments and vNext artifact-gate declarations.

- [x] **Step 3: Implement explicit counter effect and routing text**

Update the policy evaluator without changing the legacy CLI contract. Validate the optional pair together: if either attribution or effect is supplied, both must be supplied and must be one of the defined values.

```python
ATTRIBUTIONS = frozenset({"design-error", "code-error", "lowering-unknown", "evidence-gap", "none"})
FAILED_ATTEMPT_EFFECTS = frozenset({"increment", "unchanged", "reset"})


def _apply_failed_attempt_effect(current: int, effect: str) -> int:
    if effect == "increment":
        return current + 1
    if effect == "reset":
        return 0
    return current
```

When explicit effect is present, use it instead of `result in FAILED_RESULTS`; preserve existing counter behavior for legacy callers. Add CLI flags `--attribution` and `--failed-attempt-effect`, require both together, and emit them in sorted JSON output.

Update `SKILL.md` routing in this order:

1. For an explicit profile-onboarding request, Orchestrator validates profile/probe/runtime inputs, runs one bounded pre-campaign probe lifecycle, emits proposed promotion artifacts, reports them to the user, and may stop without entering Phase 0.
2. When a campaign is requested, Orchestrator obtains explicit requirements from the user/maintainer or Designer read-only capability preflight, validates/materializes them without inventing design claims, and invokes `select_profile_probes()` before campaign state or snapshot creation.
3. A unique exact-scope Unknown primary with `must-resolve|before-fallback` runs one bounded probe; unrelated Unknowns are ignored and ambiguous matches fail deterministically.
4. Eligible observed success emits promotion artifacts and stops as `promotion-pending`. Phase 0 resumes only after maintainer promotion, or after explicit decline/defer plus fallback authorization. Partial/failure/block/no-match leaves the primary Unknown.
5. Phase 0 validates only reviewed exact-scope prior evidence, rediscovers current target/runtime identity, materializes the project capability claim including any complete maintainer-confirmed fallback disposition, rejects raw probe refs, and freezes the implementation-profile snapshot. Claim/Decision validation must not depend on the pre-campaign run directory.
6. A schema-v2 Decision validates Sketch, references, frozen implementation-profile snapshot, claim, causal graph, and any fallback provenance before Coder dispatch.
7. Coder runs only Decision-scoped capability/compile probes and the binding checker before `candidate-ready`; its results stay under campaign-local `log/probes/`.
8. Verifier writes authoritative runtime facts, correctness, observed lowering, and performance only.
9. Orchestrator validates `verdict_NNN.json`; it may route one `code-error` repair.
10. `design-error` terminates as `design-rejected` with increment.
11. `lowering-unknown` terminates as `design-rejected` with unchanged failed streak.
12. Bounded `evidence-gap` routes environment absence to `blocked`; a Decision missing a measurable contract terminates as `design-rejected` with explicit design-error effect. A correct candidate with insufficient accepted-to-candidate e2e improvement remains `no-improvement` and attribution `none`.
13. At submission finalization, Orchestrator computes the canonical `submission_snapshot_id`, calls `resolve_finalization_slot()`, and opens or resumes one artifact index only when no qualification, promotion, repair, fingerprint transition, missing profile legality, completed identical input, or already-finalized current output exists.
14. Designer reuses the accepted Sketch and declares a finite reviewed profile-legal config-only domain, accepted fallback/control, search budget/protocol, objective, and deterministic order/tie rule.
15. Verifier executes the accepted source through temporary launch/meta-parameter injection under measurement exclusivity. It returns normalized trials to Orchestrator without writing a selection artifact or report and verifies that temporary storage contains no derived candidate source.
16. Orchestrator runs the pure selector and sends only the normalized selection to Coder. For an improved winner, Coder emits exactly one pinned candidate derived from the accepted source; for the fallback/control, Coder emits no source. Binding validation then runs on the exact final source.
17. Verifier reruns full correctness, lowering, required promotion evidence, and official competition measurements on that source, then atomically writes the report once with separate `search_trials` and `post_pin_official` sections.
18. Orchestrator reruns the selector from the sealed report and evaluates the pure submission-promotion predicate. An improved winner atomically advances `last_accepted_kernel` and `last_accepted_report` to the pinned source and sealed report while preserving `last_accepted_round`; a fallback-retained winner changes neither pointer; both require all final gates, and partial pair updates are rejected.
19. Orchestrator emits only `submission-ready|blocked` through the separate finalization verdict branch. It never calls `evaluate_terminal()` and the verdict rejects attribution, campaign terminal, counter-effect, and run-policy fields.
20. The final candidate contains one fixed selected configuration and no runtime/online `@triton.autotune`, adaptive search, or autotune-cache selection dependency.

Update role contracts with exact file ownership, required hashes, measurement exclusivity, temporary configuration injection, non-persistent selection handoff, one-time config pinning, sealed-report timing, and the profile/fact boundaries defined above. Update `team-state-template.md` to state that `last_attribution` and `last_completed_verdict` refer only to terminal campaign verdicts; finalization never overwrites them and is discovered through artifact scanning. `last_accepted_kernel` and `last_accepted_report` form one atomic submission pair, while `last_accepted_round` remains campaign-owned. No finalization-specific state field is added.

- [x] **Step 4: Run policy, contract, and validator suites**

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_run_policy.py -v
python3 -m unittest skills/kernel-opt-loop/tests/test_contracts.py -v
python3 -m unittest skills/kernel-opt-loop/tests/test_validate_decision.py -v
python3 -m unittest skills/kernel-opt-loop/tests/test_validate_verdict.py -v
```

Expected: PASS. Attribution effects remain explicit; final tuning has deterministic role sequencing, persists only existing artifact families, and cannot affect campaign counters.

- [x] **Step 5: Commit vNext orchestration contracts**

```bash
git add skills/kernel-opt-loop/scripts/evaluate_run_policy.py \
  skills/kernel-opt-loop/tests/test_run_policy.py \
  skills/kernel-opt-loop/SKILL.md \
  skills/kernel-opt-loop/prompts/designer.md \
  skills/kernel-opt-loop/prompts/coder.md \
  skills/kernel-opt-loop/prompts/verifier.md \
  skills/kernel-opt-loop/references/invariants.md \
  skills/kernel-opt-loop/references/team-state-template.md \
  skills/kernel-opt-loop/tests/test_contracts.py
git commit -m "skills: route vnext attribution outcomes"
```

### Task 9: Publish Registry, Documentation, and New-Run Boundary

**Files:**
- Modify: `README.md`
- Modify: `docs/backend-registry.md`
- Modify: `docs/competition/track2-clike.md`
- Modify: `skills/README.md`
- Modify: `skills/kernel-opt-loop/SKILL.md`
- Modify: `skills/kernel-opt-loop/prompts/coder_targets/triton_gcu.md`
- Modify: `skills/kernel-opt-loop/prompts/coder_targets/triton_cuda.md`
- Modify: `skills/kernel-opt-loop/prompts/coder_targets/triton_maca.md`
- Modify: `skills/kernel-opt-loop/prompts/coder_targets/triton_ascend.md`
- Modify: `skills/kernel-opt-loop/tests/test_contracts.py`

**Interfaces:**
- Documents each migrated implementation profile under `skills/kernel-opt-loop/profiles/` as canonical capability source and the matching legacy `prompts/coder_targets/*.md` page as rendered explanation.
- Documents `triton_mlu` as the first vNext registry candidate and does not claim that unconverted Triton or C-like profiles are complete.
- Documents target ids separately from implementation-profile ids and places pre-campaign profile onboarding in the kernel-opt-loop profile subsystem, with completion allowed before campaign creation.
- Documents existing v1/v2 campaigns as read-only and vNext activation as a new-run choice.
- Documents one offline bounded config-only finalization for each eligible Triton submission snapshot, with exact-source confirmation and post-pin official verification. A profile missing reviewed exact-scope configuration legality is `profile-legality-unavailable` and cannot be described as submission-finalization ready. The submitted candidate contains one fixed configuration and no runtime autotune, first-use search, or cache-dependent selection.
- Documents that the schema accepts future C-like profiles while no Track 2 profile is complete until its build/runner/profiler and source-analyzer contracts are implemented and probed.

- [x] **Step 1: Write failing consistency tests**

Add table-driven `test_contracts.py` assertions that every implementation profile named as machine-readable authority has a matching canonical file under `profiles/`, every migrated Markdown page points to that canonical source, target ids and implementation-profile ids remain distinct in the registry, repository docs place pre-campaign profile onboarding under the kernel-opt-loop profile subsystem, final tuning is documented as offline/bounded/config-only with exact-source confirmation and post-pin official verification, the submitted candidate contains no runtime autotune or cache-dependent selector, and Track 2 docs do not claim a complete profile before its adapters and probes exist.

```python
def test_profile_registry_and_human_docs_agree(self):
    canonical = SKILL_ROOT / "profiles" / "triton_mlu" / "profile.yaml"
    self.assertTrue(canonical.is_file())
    mlu_doc = (PROMPTS / "coder_targets" / "triton_mlu.md").read_text(encoding="utf-8")
    self.assertIn("profiles/triton_mlu/profile.yaml", mlu_doc)
    self.assertIn("triton_cuda", (REPO_ROOT / "README.md").read_text(encoding="utf-8"))
    registry = (REPO_ROOT / "docs" / "backend-registry.md").read_text(encoding="utf-8")
    self.assertIn("triton_maca", registry)
    self.assertIn("target id", registry.lower())
    self.assertNotIn("skills/backend-probe", (REPO_ROOT / "README.md").read_text(encoding="utf-8"))
    self.assertIn("implementation profile", (REPO_ROOT / "docs" / "competition" / "track2-clike.md").read_text(encoding="utf-8").lower())
```

- [x] **Step 2: Run the consistency test to verify it fails**

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_contracts.py -v
```

Expected: FAIL on implementation-profile registry and machine-readable coverage assertions.

- [x] **Step 3: Update exact documentation boundaries**

Update repository README and backend registry to list the human-readable `triton_mlu`, `triton_gcu`, `triton_cuda`, `triton_maca`, and `triton_ascend` documents accurately, but state only `triton_mlu` has a vNext canonical implementation profile until each other profile receives its own reviewed `profile.yaml`, executable versioned probe suite, and approved evidence. Show concrete target ids such as `bi150` or `ascend910b` separately from profile ids such as `triton_cuda` or `triton_ascend`.

Update `skills/README.md` and `SKILL.md` to describe both boundaries:

```text
Profile onboarding may run versioned probes, emit hashed run-local evidence and a
proposed promotion candidate, and stop without creating a campaign. It never edits
the canonical implementation profile.

A vNext campaign records contract_version: 3, a frozen implementation-profile
snapshot hash, a project capability claim, typed Sketch, binding ledger, and
verdict artifact. Existing v1/v2 campaigns remain historical and are not migrated.

Each Triton submission snapshot runs one offline bounded configuration-only search over
profile-legal fields. The selected configuration is pinned into one candidate and
must pass fresh binding, correctness, lowering, promotion, and official benchmark
gates. The workflow adds no finalization-specific state or artifact family. The final
candidate contains one fixed configuration and no runtime/online autotune,
first-use search, or cache-dependent configuration selection.
```

Update `docs/competition/track2-clike.md` to state that C-like backends reuse this implementation-profile qualification lifecycle with their own build/runner/profiler payloads and source analyzer; they do not require a copied skill. Update the four unconverted Triton Markdown profiles with their own migration status without inventing nonexistent canonical paths. In `triton_gcu.md`, state explicitly that groupedtopk-derived evidence proves only its observed capabilities: it does not establish negative evidence for `tl.dot`, and a future S60 attention/MoE algorithm substitution must use demand-scoped dot qualification before treating `tl.sum` as the fallback.

- [x] **Step 4: Run documentation contract checks and full Markdown fence check**

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_contracts.py -v
python3 - <<'PY'
from pathlib import Path
for path in Path('skills/kernel-opt-loop').rglob('*.md'):
    assert path.read_text(encoding='utf-8').count('```') % 2 == 0, path
PY
```

Expected: PASS. Public documentation does not overstate vNext profile coverage.

- [x] **Step 5: Commit documentation and registry alignment**

```bash
git add README.md docs/backend-registry.md docs/competition/track2-clike.md skills/README.md \
  skills/kernel-opt-loop/SKILL.md \
  skills/kernel-opt-loop/prompts/coder_targets \
  skills/kernel-opt-loop/tests/test_contracts.py \
  docs/superpowers/specs/2026-08-19-kernel-opt-loop-vnext-semantic-attribution-design.md
git commit -m "docs: describe vnext semantic contract"
```

### Task 10: Verify Profile-Probe, Campaign, and Submission-Finalization Flows

**Files:**
- Modify: `skills/kernel-opt-loop/tests/test_run_profile_probe.py`
- Modify: `skills/kernel-opt-loop/tests/test_profile_promotion.py`
- Modify: `skills/kernel-opt-loop/tests/test_validate_decision.py`
- Modify: `skills/kernel-opt-loop/tests/test_validate_binding.py`
- Modify: `skills/kernel-opt-loop/tests/test_validate_verdict.py`
- Modify: `skills/kernel-opt-loop/tests/test_contracts.py`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/integration/profile-probe/definition.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/integration/profile-probe/runtime.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/integration/profile-probe/payload.py`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/integration/campaign/decision_001.md`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/integration/campaign/sketch_001.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/integration/campaign/binding_001.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/integration/campaign/report_001.md`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/integration/campaign/verdict_001.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/integration/final-tuning/decision_002.md`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/integration/final-tuning/accepted_candidate.py`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/integration/final-tuning/pinned_candidate.py`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/integration/final-tuning/binding_002.json`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/integration/final-tuning/report_002.md`
- Create: `skills/kernel-opt-loop/tests/fixtures/vnext/integration/final-tuning/verdict_002.json`
- Modify: `docs/superpowers/plans/2026-08-19-kernel-opt-loop-vnext-semantic-attribution.md`

**Interfaces:**
- The profile-probe integration flow is self-contained, uses a fixture command, produces validated evidence and a proposed promotion candidate, and terminates with no campaign state.
- The campaign integration flow uses a separately materialized reviewed implementation-profile snapshot and exercises Decision -> Sketch -> Profile/claim -> Binding -> Verifier fact pack -> Verdict -> Run policy.
- Campaign fixtures materialize approved canonical evidence before creating the snapshot; profile-probe raw output is excluded from campaign authority.
- An S60-shaped demand-qualification flow starts from groupedtopk-derived reduction evidence plus Unknown `matrix.dot`, selects only the exact dot probe before a sum algorithm substitution, and stops as `promotion-pending` or requires explicit fallback disposition.
- The submission-finalization flow reuses the accepted Sketch/profile/claim, validates one finite profile-legal config-only Decision, evaluates one accepted source through temporary injection, selects deterministically, pins or confirms one final source, revalidates binding, requires post-pin correctness/lowering/promotion/official evidence, and emits a counter-free submission route through the existing verdict family.

- [x] **Step 1: Write the failing pre-campaign profile-probe integration test**

Create a fixture profile/definition/runtime/payload, recompute definition and profile hashes, invoke the public runner, validate the run, and render the proposed promotion artifacts.

```python
def test_profile_probe_flow_stops_before_campaign_and_never_mutates_profile(self):
    with materialized_profile_probe() as fixture:
        before = fixture.profile_path.read_bytes()
        run_dir = run_profile_probe(
            profile_path=fixture.profile_path,
            probe_id="integration-memory-001",
            target_id="fixture-target",
            runtime_snapshot_path=fixture.runtime_path,
            output_root=fixture.output_root,
            run_id="integration-001",
        )
        result = validate_probe_run(run_dir)
        candidate_path, note_path = render_profile_promotion(run_dir, profile_path=fixture.profile_path)
        self.assertEqual("evidence-ready", result["summary"])
        self.assertEqual("proposed", json.loads(candidate_path.read_text(encoding="utf-8"))["review_status"])
        self.assertEqual(before, fixture.profile_path.read_bytes())
        self.assertTrue(note_path.is_file())
        self.assertFalse((fixture.output_root / "team-state.md").exists())
        self.assertFalse((fixture.output_root / "rounds").exists())
```

Add failure variants for timeout and target/profile mismatch, and assert neither creates campaign state or an approved capability.

Add the demand-scoped S60 regression without executing hardware:

```python
def test_s60_unknown_dot_is_selected_before_sum_substitution(self):
    fixture = materialized_s60_qualification_fixture()
    profile = load_profile(fixture.profile_path)
    requirement = json.loads(fixture.requirement_path.read_text(encoding="utf-8"))
    plan = select_profile_probes(profile, [requirement], fixture.runtime_snapshot)
    self.assertEqual(["s60-dot-fp16-001"], [item.probe_id for item in plan.selections])

    run_dir = run_profile_probe(
        profile_path=fixture.profile_path,
        probe_id=plan.selections[0].probe_id,
        target_id="s60",
        runtime_snapshot_path=fixture.runtime_path,
        qualification_requirement_path=fixture.requirement_path,
        output_root=fixture.output_root,
        run_id="s60-dot-qualification-001",
    )
    candidate_path, _ = render_profile_promotion(run_dir, profile_path=fixture.profile_path)
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    self.assertEqual("promotion-pending", candidate["onboarding_disposition"])
    dot = next(item for item in load_profile(fixture.profile_path)["capability_matrix"] if item["id"] == "matrix.dot.fp16-fp16-fp32")
    self.assertEqual("unknown", dot["status"])
    self.assertFalse((fixture.output_root / "team-state.md").exists())
```

Add variants proving unrelated Unknowns are not selected, an ambiguous second dot definition is rejected, and failure leaves dot Unknown. For the authorized-fallback variant, embed the complete requirement, canonical requirement hash, failed outcome, promotion disposition, `fallback_authorized: true`, reason, maintainer confirmation, optional probe hashes, and `primary_remains_unknown: true` in the project claim; forbid any raw result ref. Delete the entire pre-campaign run directory before materializing the campaign, then prove the frozen claim, profile snapshot, and sum-fallback Decision still validate. The Decision carries only disposition id/hash plus causal consequence and is rejected when confirmation or hashes are missing. Mutating the embedded confirmation, reason, onboarding outcome, or optional probe hash after the Decision is written must invalidate `qualification_disposition_sha256`.

- [x] **Step 2: Write campaign and submission-finalization integration flows**

Materialize a separate fake campaign whose frozen implementation-profile snapshot already contains explicitly approved fixture evidence. Recompute all copied SHA-256 fields and invoke public validators in workflow order.

```python
def test_vnext_campaign_flow_reaches_lowering_unknown_without_failed_streak_increment(self):
    with materialized_integration_campaign() as project_root:
        decision = validate_decision(
            project_root / "rounds" / "decision_001.md",
            project_root=project_root,
            expected_implementation_profile="triton_mlu",
        )
        sketch = validate_sketch(project_root / decision["sketch_ref"], expected_round="001")
        profile = load_profile(project_root / decision["implementation_profile_snapshot_ref"])
        claim = validate_project_claim(project_root / decision["project_capability_claim_ref"], profile=profile, snapshot=RUNTIME_SNAPSHOT)
        binding = validate_binding(project_root / "rounds" / "binding_001.json", project_root=project_root, sketch_result=sketch, profile=profile, candidate_path=project_root / "candidate.py")
        facts = extract_verifier_fact_pack(project_root / "rounds" / "report_001.md")
        verdict = validate_verdict(project_root / "rounds" / "verdict_001.json", inputs={"decision": decision, "sketch": sketch, "profile": profile, "claim": claim, "binding": binding, "facts": facts})
        policy = evaluate_terminal(BASE_STATE, verdict["terminal_result"], attribution=verdict["classification"], failed_attempt_effect=verdict["failed_attempt_effect"])
        self.assertEqual("lowering-unknown", verdict["classification"])
        self.assertEqual(0, policy["failed_attempt_streak"])


def test_final_tuning_selects_pinned_candidate_and_preserves_campaign_state(self):
    with materialized_final_tuning() as project_root:
        decision = validate_decision(
            project_root / "rounds" / "decision_002.md",
            project_root=project_root,
            expected_implementation_profile="fixture-triton",
        )
        profile = load_profile(project_root / decision["implementation_profile_snapshot_ref"])
        facts = extract_verifier_fact_pack(project_root / "rounds" / "report_002.md")
        self.assertEqual(decision["final_tuning_contract"]["submission_snapshot_id"], facts["final_configuration_tuning"]["submission_snapshot_id"])
        selected = select_final_tuning_configuration(decision["final_tuning_contract"], facts["final_configuration_tuning"]["search_trials"])
        self.assertEqual({"num_warps": 2, "num_stages": 2}, selected["configuration"])
        binding = validate_binding(
            project_root / "rounds" / "binding_002.json",
            project_root=project_root,
            sketch_result=validate_sketch(project_root / decision["sketch_ref"]),
            profile=profile,
            candidate_path=project_root / "pinned_candidate.py",
            accepted_candidate_path=project_root / "accepted_candidate.py",
            final_tuning_contract=decision["final_tuning_contract"],
        )
        verdict = validate_verdict(project_root / "rounds" / "verdict_002.json", inputs={"decision": decision, "profile": profile, "binding": binding, "facts": facts})
        self.assertEqual("submission-ready", verdict["route"])
        self.assertNotIn("classification", verdict)
        self.assertNotIn("terminal_result", verdict)
        self.assertNotIn("failed_attempt_effect", verdict)
        before = {"last_accepted_kernel": "accepted_candidate.py", "last_accepted_report": "rounds/report_001.md", "last_accepted_round": "001", "failed_attempt_streak": 0}
        after = apply_submission_promotion(before, verdict)
        self.assertEqual("pinned_candidate.py", after["last_accepted_kernel"])
        self.assertEqual("rounds/report_002.md", after["last_accepted_report"])
        self.assertEqual("001", after["last_accepted_round"])
        self.assertEqual(0, after["failed_attempt_streak"])
        self.assertFalse((project_root / "state" / "final-tuning.json").exists())
        self.assertFalse((project_root / "log" / "final-tuning").exists())
```

Add a missing-required-binding fixture that routes `coder-repair` and an environment evidence gap that leaves counters neutral. Add a fallback-retained finalization variant that reruns full final official verification, emits no candidate source, and leaves both accepted kernel/report pointers unchanged. Final-tuning negative variants cover a semantic field disguised as configuration, missing/profile-illegal/Unknown legality, stale accepted candidate/binding/Sketch/profile/claim/runtime/measurement/harness/base hashes, incorrect `submission_snapshot_id`, exceeded budget, undeclared or duplicate trials, nondeterministic tie, any derived source under `log/final-tuning/`, pin values that differ from selection, stale binding, post-pin correctness failure, missing official evidence, any runtime/online autotune, first-use search, or cache-dependent configuration selector, completed input-ID reuse, a current accepted output already finalized under unchanged anchors, conflicting Decision hash at a reserved index, partial accepted kernel/report pointer update, and any `evaluate_terminal()` call or finalization verdict carrying campaign projection fields.

Run:

```bash
python3 -m unittest skills/kernel-opt-loop/tests/test_run_profile_probe.py -v
python3 -m unittest skills/kernel-opt-loop/tests/test_profile_promotion.py -v
python3 -m unittest skills/kernel-opt-loop/tests/test_validate_verdict.py -v
```

Expected: FAIL until the profile-probe, campaign, and submission-finalization flows satisfy all hashes, role boundaries, and post-pin gates.

- [x] **Step 3: Complete fixture materialization and public script contract checks**

Implement `materialized_profile_probe()`, `materialized_integration_campaign()`, and `materialized_final_tuning()` with temporary directories, copy-only fixture setup, and recomputed hashes. The campaign helper must first archive a reviewed evidence record into a canonical fixture profile, then copy that entire profile closure—`profile.yaml`, vendored schema, probe definitions/inputs, and reviewed evidence/attachments—to `state/implementation_profile_snapshot/`; it may not consume the raw run directory directly. Delete or rename the canonical fixture copy and delete any pre-campaign raw run before campaign validation; prove the frozen profile snapshot, embedded project-claim disposition, and Decision still load without external dependencies. Resolve failures by correcting schemas, validators, or fixtures, never by weakening target/profile identity, closure, scope, hash, source-span, or rule-precondition checks.

The final-tuning helper starts from an accepted candidate/binding, accepted Sketch, frozen profile/claim, runtime snapshot, official measurement fingerprint, and harness/base hashes, then recomputes `submission_snapshot_id`. Search injects configuration values into the accepted source without creating derived source files; `log/final-tuning/` may contain only non-source temporary data and is removed before verdict validation. The helper models the non-persistent trial handoff, pins one candidate or confirms the fallback, validates the exact final binding, and atomically materializes `report_002.md` once with `search_trials` and `post_pin_official`. It creates no finalization-specific state or artifact family and no persistent source variant.

Add a final `test_contracts.py` assertion that every JSON/JSON-compatible YAML fixture parses and every public runner/validator/renderer script exists as a regular file. Scripts are invoked explicitly with `python3`; executable mode bits are not part of the contract:

```python
def test_vnext_fixture_documents_and_scripts_are_loadable(self):
    for path in (SKILL_ROOT / "tests" / "fixtures" / "vnext").rglob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))
    for path in (SKILL_ROOT / "profiles").rglob("*.yaml"):
        json.loads(path.read_text(encoding="utf-8"))
    for name in (
        "run_profile_probe.py",
        "validate_probe.py",
        "render_profile_promotion.py",
        "validate_sketch.py",
        "validate_profile.py",
        "validate_binding.py",
        "validate_verdict.py",
    ):
        self.assertTrue((SKILL_ROOT / "scripts" / name).is_file())
```

CLI-specific tests invoke `python3 <script> --help` where applicable; import-only validators are exercised through their public Python functions.

- [x] **Step 4: Run full regression, whitespace, and artifact checks**

Run:

```bash
python3 -m unittest discover -s skills/kernel-opt-loop/tests -v
git diff --check
for schema in skills/kernel-opt-loop/schemas/*.schema.json; do
  python3 -m json.tool "$schema" >/dev/null
done
python3 -m json.tool skills/kernel-opt-loop/profiles/triton_mlu/profile.yaml >/dev/null
python3 -m json.tool skills/kernel-opt-loop/tests/fixtures/vnext/integration/campaign/verdict_001.json >/dev/null
python3 -m json.tool skills/kernel-opt-loop/tests/fixtures/vnext/integration/final-tuning/verdict_002.json >/dev/null
```

Expected: PASS with no accelerator access. Confirm `git status --short` contains only vNext skill, tests, spec, plan, and documentation files; preserve but do not stage unrelated pre-existing files.

- [x] **Step 5: Commit integration verification**

```bash
git add skills/kernel-opt-loop/tests \
  docs/superpowers/plans/2026-08-19-kernel-opt-loop-vnext-semantic-attribution.md
git commit -m "test: cover probe campaign and finalization flows"
```

## Acceptance Coverage

| Spec requirement | Implementation tasks |
|---|---|
| Pre-campaign probe completes without campaign state | Tasks 1, 4, 8, 10 |
| Probe catalog path/hash plus definition/run/result hash chain, bounded runner, and failure classification | Tasks 1, 3, 4, 10 |
| Proposed promotion candidate, scope conservation, and no automatic profile mutation | Tasks 1, 4, 10 |
| Promotion ownership and no campaign-time approval | Tasks 8, 10 |
| Demand-scoped Unknown selection probes only explicit primary capabilities before algorithm substitution | Tasks 3, 4, 8, 10 |
| Successful qualification stops as promotion-pending; raw evidence cannot satisfy Phase 0 | Tasks 4, 8, 10 |
| Failed/unavailable qualification leaves Unknown and requires explicit fallback provenance | Tasks 3, 5, 8, 10 |
| Maintainer fallback authorization remains self-contained after raw-run deletion | Tasks 1, 3, 5, 10 |
| Concrete target identity separate from implementation-profile identity | Tasks 1, 3, 4, 5, 10 |
| Language-neutral implementation profile and future C-like seam | Tasks 1, 3, 6, 9 |
| Typed Sketch structural semantics: shape/type/layout/memory, SSA, bounds, effects/aliases | Tasks 1, 2, 5, 10 |
| JSON Sketch normative; Markdown rendering non-authoritative | Tasks 2, 5, 9 |
| Required/preferred/exploratory hints | Tasks 2, 3, 6 |
| Machine-readable profile, scoped approved evidence, five capability statuses, partial profile | Tasks 1, 3, 4, 9 |
| Profile/project claim/campaign snapshot matching | Tasks 3, 5, 8, 10 |
| Frozen implementation-profile dependency closure survives canonical deletion | Tasks 3, 5, 10 |
| Statement-level source binding and many-to-many relations | Task 6 |
| Profile-selected analyzer; source binding separate from observed lowering | Tasks 3, 6, 7, 8 |
| Decision causal graph references and structural gate | Task 5 |
| Verifier fact pack plus causal/verdict evaluation | Task 7 |
| Deterministic design/code/lowering/evidence attribution | Tasks 7, 8, 10 |
| Existing terminal enums retained; lowering unknown counter-neutral | Tasks 8, 10 |
| One local code repair; no Verifier blame or profile authority | Tasks 7, 8, 10 |
| Probe input/result/evidence hash validation | Tasks 1, 4, 10 |
| Decision references, runtime anchors, and context naming repair | Tasks 1, 5, 10 |
| Exact-scope finite reviewed configuration legality with accepted fallback/control; missing/Unknown legality blocks | Tasks 3, 5, 10 |
| Final tuning preserves Sketch/algorithm/precision/effects/Host Plan/public interface | Tasks 5, 6, 8, 10 |
| Deterministic bounded comparison uses non-persistent handoff and one sealed report without finalization-specific state | Tasks 5, 7, 8, 10 |
| Selected configuration is pinned once or accepted fallback confirmed; exact final source is bound and fully verified | Tasks 6, 7, 8, 10 |
| Improved finalization atomically advances accepted kernel/report pair; fallback advances neither; accepted round remains campaign-owned | Tasks 7, 8, 10 |
| Canonical submission identity and recoverable artifact index make finalization one-shot and counter-free; search evidence cannot authorize submission | Tasks 1, 5, 7, 8, 10 |
| Final candidate has one fixed config and no runtime autotune/cache selector | Tasks 8, 9, 10 |
| Existing campaigns remain read-only; docs do not overclaim profile migration | Task 9 |
| No hardware-dependent test requirement | Every task; verified in Tasks 4 and 10 |

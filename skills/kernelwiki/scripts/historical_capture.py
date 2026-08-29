from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import json
import re
import subprocess
from typing import Any

import yaml

from corpus import load_corpus, validate_corpus
from experience import ExperienceProposal
from kernelwiki_common import (
    KernelWikiError,
    canonical_json_bytes,
    load_yaml_document,
    sha256_bytes,
    validate_root_relative_posix_path,
)
from lift_schema import validate_lift_document
from validate_lift import validate_proposal, validate_review


SKILL_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_PATH = SKILL_ROOT / "data" / "schemas.yaml"
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_ID = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_MISSING = re.compile(r"(?:sketch|binding|verdict|measurement|runtime|profile|other):[a-z0-9][a-z0-9-]*")
_ARTIFACT_FIELDS = {"path", "role", "sha256"}
_MEASUREMENT_FIELDS = {"status", "fingerprint", "reason"}
_OBSERVATION_FIELDS = {"metric", "value", "statistic", "unit", "evidence_ref"}
_BOUNDARY_KEYS = {"target", "profile", "runtime", "shape", "dtype", "round", "measurement"}

@dataclass(frozen=True)
class HistoricalArtifact:
    path: str
    role: str
    sha256: str

@dataclass(frozen=True)
class HistoricalManifest:
    schema_version: int
    source_id: str
    historical_contract_version: int
    repository_commit: str
    project_path: str
    local_locator: str
    captured_at: str
    repository_id: str
    languages: tuple[str, ...]
    kernel_types: tuple[str, ...]
    techniques: tuple[str, ...]
    hardware_features: tuple[str, ...]
    tags: tuple[str, ...]
    license_state: str
    asset_mode: str
    allowed_audiences: tuple[str, ...]
    target_id: str
    implementation_profile_id: str
    profile_authority: str
    terminal_result: str
    artifacts: tuple[HistoricalArtifact, ...]
    measurement: Mapping[str, Any]
    observations: tuple[Mapping[str, Any], ...]
    transfer_boundaries: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    audiences: tuple[str, ...]
    strict_vnext_validated: bool
    repository_root: Path


def _fail(message: str, path: Path | None = None) -> None:
    raise KernelWikiError("historical-capture-invalid", message, path)


def _mapping(value: Any, label: str, path: Path) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _fail(f"{label} must be an object", path)
    return value


def _texts(value: Any, label: str, path: Path, *, nonempty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        _fail(f"{label} must be a list of nonempty strings", path)
    if nonempty and not value:
        _fail(f"{label} must not be empty", path)
    if len(value) != len(set(value)):
        _fail(f"{label} must not contain duplicates", path)
    return tuple(value)


def _relative(value: Any, label: str, path: Path) -> str:
    try:
        return validate_root_relative_posix_path(value)
    except ValueError as error:
        _fail(f"{label} is invalid: {error}", path)
    raise AssertionError("unreachable")


def _git_bytes(root: Path, *args: str) -> bytes:
    completed = subprocess.run(["git", "-C", str(root), *args], check=False, capture_output=True)
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", "replace").strip() or "git command failed"
        raise KernelWikiError("historical-git-invalid", message, root)
    return completed.stdout


def _validate_capture_time(value: Any, path: Path) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        _fail("captured_at must be an RFC3339 UTC timestamp", path)
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        _fail(f"captured_at is invalid: {error}", path)
    return value


def _validate_artifacts(value: Any, project_path: str, path: Path) -> tuple[HistoricalArtifact, ...]:
    if not isinstance(value, list) or not value:
        _fail("artifacts must be a nonempty list", path)
    artifacts: list[HistoricalArtifact] = []
    roles: set[str] = set()
    paths: set[str] = set()
    project_prefix = project_path + "/"
    for index, raw in enumerate(value):
        item = _mapping(raw, f"artifacts[{index}]", path)
        if set(item) != _ARTIFACT_FIELDS:
            _fail(f"artifacts[{index}] must contain exactly path, role, and sha256", path)
        artifact_path = _relative(item["path"], f"artifacts[{index}].path", path)
        if not artifact_path.startswith(project_prefix):
            _fail(f"artifacts[{index}].path must stay under project_path", path)
        role = item["role"]
        digest = item["sha256"]
        if not isinstance(role, str) or _ID.fullmatch(role) is None:
            _fail(f"artifacts[{index}].role is invalid", path)
        if not isinstance(digest, str) or _HEX64.fullmatch(digest) is None:
            _fail(f"artifacts[{index}].sha256 is invalid", path)
        if role in roles or artifact_path in paths:
            _fail("artifact roles and paths must be unique", path)
        roles.add(role)
        paths.add(artifact_path)
        artifacts.append(HistoricalArtifact(artifact_path, role, digest))
    return tuple(artifacts)


def _validate_measurement(value: Any, path: Path) -> dict[str, Any]:
    measurement = dict(_mapping(value, "measurement", path))
    if set(measurement) != _MEASUREMENT_FIELDS:
        _fail("measurement must contain exactly status, fingerprint, and reason", path)
    if measurement["status"] == "available":
        if not isinstance(measurement["fingerprint"], str) or _HEX64.fullmatch(measurement["fingerprint"]) is None:
            _fail("available measurement requires a SHA-256 fingerprint", path)
        if measurement["reason"] is not None:
            _fail("available measurement reason must be null", path)
    elif measurement["status"] == "missing":
        if measurement["fingerprint"] is not None:
            _fail("missing measurement fingerprint must be null", path)
        if not isinstance(measurement["reason"], str) or not measurement["reason"].strip():
            _fail("missing measurement requires a reason", path)
    else:
        _fail("measurement status must be available or missing", path)
    return measurement


def _validate_observations(value: Any, artifact_paths: set[str], path: Path) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        _fail("observations must be a list", path)
    observations: list[dict[str, Any]] = []
    for index, raw in enumerate(value):
        item = dict(_mapping(raw, f"observations[{index}]", path))
        if set(item) != _OBSERVATION_FIELDS:
            _fail(f"observations[{index}] has unknown or missing fields", path)
        for label in ("metric", "statistic", "unit", "evidence_ref"):
            if not isinstance(item[label], str) or not item[label].strip():
                _fail(f"observations[{index}].{label} must be nonempty text", path)
        if isinstance(item["value"], (Mapping, list)):
            _fail(f"observations[{index}].value must be a scalar", path)
        if item["evidence_ref"] not in artifact_paths:
            _fail(f"observations[{index}].evidence_ref must name a selected artifact", path)
        observations.append(item)
    return tuple(observations)


def _validate_boundaries(value: Any, path: Path) -> tuple[str, ...]:
    boundaries = _texts(value, "transfer_boundaries", path)
    for item in boundaries:
        key, separator, scoped = item.partition("=")
        if separator != "=" or key not in _BOUNDARY_KEYS or not scoped:
            _fail("transfer boundaries must be explicit key=value scope statements", path)
        lowered = scoped.lower()
        if any(word in lowered for word in ("all-target", "any-target", "portable", "universal")):
            _fail("transfer boundary exceeds the recorded scope", path)
    return boundaries


def load_historical_manifest(path: Path, *, repository_root: Path | None = None) -> HistoricalManifest:
    manifest_path = Path(path)
    document = validate_lift_document(
        "historical_capture",
        load_yaml_document(manifest_path),
        SCHEMA_PATH,
        path=manifest_path,
    )
    root = Path(repository_root or REPOSITORY_ROOT).resolve()
    source_id = document["source_id"]
    if not isinstance(source_id, str) or _ID.fullmatch(source_id) is None:
        _fail("source_id is invalid", manifest_path)
    version = document["historical_contract_version"]
    if type(version) is not int or version < 1:
        _fail("historical_contract_version must be a positive integer", manifest_path)
    commit = document["repository_commit"]
    if not isinstance(commit, str) or _HEX40.fullmatch(commit) is None:
        _fail("repository_commit must be 40 lowercase hex characters", manifest_path)
    project_path = _relative(document["project_path"], "project_path", manifest_path)
    locator = document["local_locator"]
    if not isinstance(locator, str) or not locator.startswith(project_path):
        _fail("local_locator must identify the selected project", manifest_path)
    if document["repository_id"] != "local":
        _fail("repository_id must be local", manifest_path)
    if document["asset_mode"] not in {"metadata-only", "selected-files"}:
        _fail("asset_mode must be metadata-only or selected-files", manifest_path)
    if document["profile_authority"] != "historical-noncanonical":
        _fail("profile_authority must be historical-noncanonical", manifest_path)
    if document["strict_vnext_validated"] is not False:
        _fail("strict_vnext_validated must be false", manifest_path)
    allowed = _texts(document["allowed_audiences"], "allowed_audiences", manifest_path)
    audiences = _texts(document["audiences"], "audiences", manifest_path)
    if allowed != ("designer",) or audiences != ("designer",):
        _fail("historical evidence is Designer-only", manifest_path)
    for label in ("target_id", "implementation_profile_id", "terminal_result", "license_state"):
        if not isinstance(document[label], str) or not document[label].strip():
            _fail(f"{label} must be nonempty text", manifest_path)
    artifacts = _validate_artifacts(document["artifacts"], project_path, manifest_path)
    measurement = _validate_measurement(document["measurement"], manifest_path)
    observations = _validate_observations(document["observations"], {item.path for item in artifacts}, manifest_path)
    boundaries = _validate_boundaries(document["transfer_boundaries"], manifest_path)
    missing = _texts(document["missing_evidence"], "missing_evidence", manifest_path)
    if any(_MISSING.fullmatch(item) is None for item in missing):
        _fail("missing_evidence entries must be typed", manifest_path)
    return HistoricalManifest(
        schema_version=1,
        source_id=source_id,
        historical_contract_version=version,
        repository_commit=commit,
        project_path=project_path,
        local_locator=locator,
        captured_at=_validate_capture_time(document["captured_at"], manifest_path),
        repository_id="local",
        languages=_texts(document["languages"], "languages", manifest_path),
        kernel_types=_texts(document["kernel_types"], "kernel_types", manifest_path),
        techniques=_texts(document["techniques"], "techniques", manifest_path),
        hardware_features=_texts(document["hardware_features"], "hardware_features", manifest_path),
        tags=_texts(document["tags"], "tags", manifest_path),
        license_state=document["license_state"],
        asset_mode=document["asset_mode"],
        allowed_audiences=allowed,
        target_id=document["target_id"],
        implementation_profile_id=document["implementation_profile_id"],
        profile_authority="historical-noncanonical",
        terminal_result=document["terminal_result"],
        artifacts=artifacts,
        measurement=measurement,
        observations=observations,
        transfer_boundaries=boundaries,
        missing_evidence=missing,
        audiences=audiences,
        strict_vnext_validated=False,
        repository_root=root,
    )


def _verify_artifacts(manifest: HistoricalManifest) -> None:
    _git_bytes(manifest.repository_root, "cat-file", "-e", f"{manifest.repository_commit}^{{commit}}")
    for artifact in manifest.artifacts:
        data = _git_bytes(manifest.repository_root, "show", f"{manifest.repository_commit}:{artifact.path}")
        if sha256_bytes(data) != artifact.sha256:
            raise KernelWikiError("historical-artifact-hash", f"artifact hash mismatch: {artifact.role}", manifest.repository_root / artifact.path)


def build_historical_proposal(manifest: HistoricalManifest) -> ExperienceProposal:
    _verify_artifacts(manifest)
    measurement_fingerprint = manifest.measurement.get("fingerprint")
    proposal = ExperienceProposal(
        schema_version=1,
        proposal_id=f"experience-historical-{manifest.source_id}",
        source_lane="historical-manual",
        contract_version=manifest.historical_contract_version,
        loop_contract_identity=None,
        artifact_hashes={item.role: item.sha256 for item in sorted(manifest.artifacts, key=lambda item: item.role)},
        terminal={
            "result": manifest.terminal_result,
            "commit": manifest.repository_commit,
            "project_path": manifest.project_path,
            "local_locator": manifest.local_locator,
            "strict_vnext_validated": False,
        },
        scope={
            "source_id": manifest.source_id,
            "captured_at": manifest.captured_at,
            "repository_id": manifest.repository_id,
            "target_id": manifest.target_id,
            "implementation_profile_id": manifest.implementation_profile_id,
            "profile_authority": manifest.profile_authority,
            "languages": list(manifest.languages),
            "kernel_types": list(manifest.kernel_types),
            "techniques": list(manifest.techniques),
            "hardware_features": list(manifest.hardware_features),
            "tags": list(manifest.tags),
            "license_state": manifest.license_state,
            "asset_mode": manifest.asset_mode,
            "allowed_audiences": ["designer"],
            "audiences": ["designer"],
            "measurement_fingerprint": measurement_fingerprint,
            "comparability": "historical-local",
        },
        expected={},
        observed=manifest.observations,
        suggested_publication={
            "decision": "defer",
            "mode": "existing-card-example",
            "role": None,
            "subtype": None,
            "card_ids": [],
            "tags": ["historical-local"],
        },
        transfer_boundaries=manifest.transfer_boundaries,
        reconsider_when=tuple(f"evidence becomes available: {item}" for item in manifest.missing_evidence),
        missing_evidence=manifest.missing_evidence,
    )
    validate_lift_document("experience_proposal", proposal.to_document(), SCHEMA_PATH)
    return proposal


def write_historical_proposal(proposal: ExperienceProposal, output_path: Path) -> str:
    path = Path(output_path)
    document = proposal.to_document()
    validate_lift_document("experience_proposal", document, SCHEMA_PATH, path=path)
    data = canonical_json_bytes(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_file() and path.read_bytes() == data:
            return sha256_bytes(data)
        raise KernelWikiError("proposal-exists", "different proposal output already exists", path)
    try:
        with path.open("xb") as handle:
            handle.write(data)
    except FileExistsError as error:
        raise KernelWikiError("proposal-exists", "proposal output already exists", path) from error
    return sha256_bytes(data)


def _boundary_value(boundaries: Any, name: str, proposal_path: Path) -> str:
    prefix = name + "="
    matches = [item[len(prefix):] for item in boundaries if isinstance(item, str) and item.startswith(prefix)]
    if len(matches) != 1 or not matches[0]:
        raise KernelWikiError(
            "reviewed-historical-scope",
            f"proposal requires exactly one {name}= transfer boundary",
            proposal_path,
        )
    return matches[0]


def _historical_source_bytes(proposal: Mapping[str, Any], review: Mapping[str, Any]) -> bytes:
    scope = proposal["scope"]
    terminal = proposal["terminal"]
    runtime = _boundary_value(proposal["transfer_boundaries"], "runtime", Path(str(proposal["proposal_id"])))
    source_id = scope["source_id"]
    title = f"Reviewed historical local campaign evidence: {terminal['local_locator']}"
    metadata = {
        "schema_version": 1,
        "id": source_id,
        "source_kind": "local-campaign",
        "title": title,
        "url": f"local://{terminal['commit']}/{terminal['project_path']}",
        "repository_id": "local",
        "captured_at": scope["captured_at"],
        "target_disposition": "exact",
        "target_ids": [scope["target_id"]],
        "implementation_profile_ids": [scope["implementation_profile_id"]],
        "runtime_fingerprints": [runtime],
        "languages": sorted(scope["languages"]),
        "kernel_types": sorted(scope["kernel_types"]),
        "techniques": sorted(scope["techniques"]),
        "hardware_features": sorted(scope["hardware_features"]),
        "tags": sorted(scope["tags"]),
        "license_state": scope["license_state"],
        "audiences": ["designer"],
        "profile_authority": "historical-noncanonical",
        "strict_vnext_validated": False,
        "missing_evidence": sorted(proposal["missing_evidence"]),
    }
    review_record = {
        "decision": review["decision"],
        "proposal_id": review["proposal_id"],
        "proposal_sha256": review["proposal_sha256"],
        "publication_target": review["publication_target"],
        "rationale": review["rationale"],
        "reviewed_at": review["reviewed_at"],
        "reviewed_by": review["reviewed_by"],
    }
    evidence_record = {
        "artifact_hashes": proposal["artifact_hashes"],
        "missing_evidence": proposal["missing_evidence"],
        "observed": proposal["observed"],
        "scope": proposal["scope"],
        "terminal": proposal["terminal"],
        "transfer_boundaries": proposal["transfer_boundaries"],
    }
    frontmatter = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True).rstrip()
    review_json = json.dumps(review_record, indent=2, sort_keys=True, ensure_ascii=False)
    evidence_json = json.dumps(evidence_record, indent=2, sort_keys=True, ensure_ascii=False)
    body = (
        f"# {title}\n\n"
        "Immutable reviewed historical campaign evidence. This Source is metadata-only, "
        "Designer-only, and not validated against the current vNext contract.\n\n"
        "## Curator review\n\n"
        f"```json\n{review_json}\n```\n\n"
        "## Reviewed proposal evidence\n\n"
        f"```json\n{evidence_json}\n```\n"
    )
    return f"---\n{frontmatter}\n---\n{body}".encode("utf-8")


def materialize_reviewed_historical_source(
    proposal_path: Path,
    review_path: Path,
    skill_root: Path = SKILL_ROOT,
) -> Path:
    root = Path(skill_root).resolve()
    proposal = validate_proposal(Path(proposal_path))
    review = validate_review(Path(review_path), proposal)
    document = proposal.document
    if document["source_lane"] != "historical-manual":
        raise KernelWikiError("reviewed-historical-lane", "proposal is not historical-manual", proposal.path)
    if review["decision"] != "include":
        raise KernelWikiError("reviewed-historical-not-included", "review decision must be include", Path(review_path))
    target = review["publication_target"]
    if not isinstance(target, Mapping):
        raise KernelWikiError("reviewed-historical-target", "include review requires a publication target", Path(review_path))

    corpus = load_corpus(root)
    validate_corpus(corpus)
    if target["mode"] == "existing-card-example" and target["card_id"] not in corpus.cards:
        raise KernelWikiError("reviewed-historical-target", "review target Card does not exist", Path(review_path))

    scope = document["scope"]
    if scope["asset_mode"] != "metadata-only":
        raise KernelWikiError(
            "reviewed-historical-asset-mode",
            "selected-files publication requires a separate reviewed artifact capture",
            proposal.path,
        )
    source_id = scope["source_id"]
    if not isinstance(source_id, str) or _ID.fullmatch(source_id) is None:
        raise KernelWikiError("reviewed-historical-source-id", "proposal source_id is invalid", proposal.path)
    target_id = str(scope["target_id"])
    if _ID.fullmatch(target_id) is None:
        raise KernelWikiError("reviewed-historical-target", "proposal target_id is not a valid identifier", proposal.path)

    destination = root / "sources" / "local" / target_id / f"{source_id}.md"
    data = _historical_source_bytes(document, review)
    destination.parent.mkdir(parents=True, exist_ok=True)
    created = False
    if destination.exists():
        if not destination.is_file() or destination.read_bytes() != data:
            raise KernelWikiError("source-exists", "different immutable Source already exists", destination)
    else:
        try:
            with destination.open("xb") as handle:
                handle.write(data)
            created = True
        except FileExistsError as error:
            raise KernelWikiError("source-exists", "Source already exists", destination) from error

    try:
        validate_corpus(load_corpus(root))
    except Exception:
        if created:
            destination.unlink(missing_ok=True)
        raise
    return destination

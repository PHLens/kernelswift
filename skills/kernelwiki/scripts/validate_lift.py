from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import yaml

from kernelwiki_common import KernelWikiError, run_cli, sha256_bytes
from lift_schema import validate_lift_document


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = SKILL_ROOT / "data" / "schemas.yaml"
_FORBIDDEN_KEYS = {"next_candidate", "recommended_next_change", "implementation_instruction"}
_PROPOSAL_PUBLICATION_FIELDS = {"decision", "mode", "role", "subtype", "card_ids", "tags"}
_STRICT_SCOPE_FIELDS = {
    "target_id",
    "implementation_profile_id",
    "implementation_profile_version",
    "profile_status",
    "runtime_fingerprint",
    "device_architectures",
    "language",
    "backend",
    "shape_signatures",
    "dtypes",
    "measurement_fingerprint",
    "comparability",
}
_HISTORICAL_SCOPE_FIELDS = {
    "source_id",
    "captured_at",
    "repository_id",
    "target_id",
    "implementation_profile_id",
    "profile_authority",
    "languages",
    "kernel_types",
    "techniques",
    "hardware_features",
    "tags",
    "license_state",
    "asset_mode",
    "allowed_audiences",
    "audiences",
    "measurement_fingerprint",
    "comparability",
}
_OPERATOR_TERMS = (
    "groupedtopk",
    "grouped top-k",
    "grouped top k",
    "flexattention",
    "flex attention",
    "mhc post layer mix",
    "mm encoder attention",
    "sparse pooler",
    "sinkhorn normalize",
    "index top-k",
    "sparse attention",
)


@dataclass(frozen=True)
class ValidatedProposal:
    path: Path
    document: Mapping[str, Any]
    sha256: str


def _fail(code: str, message: str, path: Path | None = None) -> None:
    raise KernelWikiError(code, message, path)


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    candidate = Path(path)
    try:
        data = candidate.read_bytes()
        value = json.loads(data)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(f"{label}-invalid", str(error), candidate)
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        _fail(f"{label}-invalid", f"{label} must be a JSON object", candidate)
    return value, data


def _load_review_document(path: Path) -> dict[str, Any]:
    candidate = Path(path)
    try:
        value = yaml.safe_load(candidate.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as error:
        _fail("lift-review-invalid", str(error), candidate)
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        _fail("lift-review-invalid", "lift review must be an object", candidate)
    return value


def _mapping(value: Any, label: str, path: Path) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        _fail("lift-proposal-invalid", f"{label} must be an object", path)
    return value


def _text(value: Any, label: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail("lift-proposal-invalid", f"{label} must be nonempty text", path)
    return value


def _text_list(value: Any, label: str, path: Path, *, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item.strip() for item in value):
        _fail("lift-proposal-invalid", f"{label} must be a list of nonempty strings", path)
    if not allow_empty and not value:
        _fail("lift-proposal-invalid", f"{label} must not be empty", path)
    if len(value) != len(set(value)):
        _fail("lift-proposal-invalid", f"{label} must not contain duplicates", path)
    return value


def _walk_forbidden(value: Any, path: Path) -> None:
    if isinstance(value, Mapping):
        found = _FORBIDDEN_KEYS & set(value)
        if found:
            _fail("lift-proposal-invalid", f"forbidden fields: {', '.join(sorted(found))}", path)
        for nested in value.values():
            _walk_forbidden(nested, path)
    elif isinstance(value, list):
        for nested in value:
            _walk_forbidden(nested, path)


def _validate_scope(document: Mapping[str, Any], path: Path) -> None:
    scope = _mapping(document["scope"], "scope", path)
    lane = document["source_lane"]
    expected = _STRICT_SCOPE_FIELDS if lane == "strict-current-vnext" else _HISTORICAL_SCOPE_FIELDS
    if set(scope) != expected:
        _fail("lift-proposal-invalid", f"{lane} scope must contain its exact fields", path)
    for label in ("target_id", "implementation_profile_id"):
        _text(scope[label], f"scope.{label}", path)
    if lane == "strict-current-vnext":
        _text(scope["runtime_fingerprint"], "scope.runtime_fingerprint", path)
        for label in ("device_architectures", "shape_signatures", "dtypes"):
            _text_list(scope[label], f"scope.{label}", path)
    else:
        if scope["repository_id"] != "local" or scope["profile_authority"] != "historical-noncanonical":
            _fail("lift-proposal-invalid", "historical scope must remain local and noncanonical", path)
        if scope["allowed_audiences"] != ["designer"] or scope["audiences"] != ["designer"]:
            _fail("lift-proposal-invalid", "historical proposals are Designer-only", path)
        for label in ("languages", "kernel_types", "techniques", "hardware_features", "tags"):
            _text_list(scope[label], f"scope.{label}", path, allow_empty=False)


def _validate_publication_suggestion(value: Any, path: Path) -> None:
    suggestion = _mapping(value, "suggested_publication", path)
    if set(suggestion) != _PROPOSAL_PUBLICATION_FIELDS:
        _fail("lift-proposal-invalid", "suggested_publication must contain its exact fields", path)
    if suggestion["decision"] not in {"include", "defer", "exclude"}:
        _fail("lift-proposal-invalid", "suggested publication decision is invalid", path)
    if suggestion["mode"] not in {None, "existing-card-example", "new-general-card"}:
        _fail("lift-proposal-invalid", "suggested publication mode is invalid", path)
    for label in ("card_ids", "tags"):
        _text_list(suggestion[label], f"suggested_publication.{label}", path)


def validate_proposal(path: Path) -> ValidatedProposal:
    proposal_path = Path(path)
    document, data = _load_json(proposal_path, "lift-proposal")
    validated = validate_lift_document("experience_proposal", document, SCHEMA_PATH, path=proposal_path)
    _walk_forbidden(validated, proposal_path)
    _validate_scope(validated, proposal_path)
    hashes = _mapping(validated["artifact_hashes"], "artifact_hashes", proposal_path)
    if not hashes:
        _fail("lift-proposal-invalid", "artifact_hashes must not be empty", proposal_path)
    _mapping(validated["terminal"], "terminal", proposal_path)
    _mapping(validated["expected"], "expected", proposal_path)
    if not isinstance(validated["observed"], list):
        _fail("lift-proposal-invalid", "observed must be a list", proposal_path)
    boundaries = _text_list(validated["transfer_boundaries"], "transfer_boundaries", proposal_path, allow_empty=False)
    for prefix in ("target=", "profile=", "runtime="):
        if not any(item.startswith(prefix) for item in boundaries):
            _fail("lift-proposal-invalid", f"transfer_boundaries requires {prefix[:-1]} scope", proposal_path)
    missing = _text_list(validated["missing_evidence"], "missing_evidence", proposal_path)
    _text_list(validated["reconsider_when"], "reconsider_when", proposal_path)
    _validate_publication_suggestion(validated["suggested_publication"], proposal_path)

    lane = validated["source_lane"]
    if lane == "strict-current-vnext":
        if validated["loop_contract_identity"] is None:
            _fail("lift-proposal-invalid", "strict proposal requires LoopContractIdentity", proposal_path)
    elif lane == "historical-manual":
        if validated["loop_contract_identity"] is not None:
            _fail("lift-proposal-invalid", "historical proposal identity must be null", proposal_path)
        terminal = _mapping(validated["terminal"], "terminal", proposal_path)
        if terminal.get("strict_vnext_validated") is not False:
            _fail("lift-proposal-invalid", "historical proposal must record strict_vnext_validated false", proposal_path)
        if not missing:
            _fail("lift-proposal-invalid", "historical proposal must preserve missing evidence", proposal_path)
    else:
        _fail("lift-proposal-invalid", "source_lane is invalid", proposal_path)
    return ValidatedProposal(proposal_path, validated, sha256_bytes(data))


def _validate_review_time(value: Any, path: Path) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        _fail("lift-review-invalid", "reviewed_at must be an RFC3339 UTC timestamp", path)
    try:
        datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        _fail("lift-review-invalid", f"reviewed_at is invalid: {error}", path)


def _operator_specific(title: str) -> bool:
    lowered = " ".join(title.lower().replace("_", " ").split())
    return any(term in lowered for term in _OPERATOR_TERMS)


def _validate_publication_target(target: Any, path: Path) -> None:
    value = _mapping(target, "publication_target", path)
    mode = value.get("mode")
    if mode == "existing-card-example":
        if set(value) != {"mode", "card_id"}:
            _fail("lift-review-invalid", "existing-card-example target fields are invalid", path)
        _text(value["card_id"], "publication_target.card_id", path)
    elif mode == "new-general-card":
        if set(value) != {"mode", "title", "independent_teaching_value"}:
            _fail("lift-review-invalid", "new-general-card target fields are invalid", path)
        title = _text(value["title"], "publication_target.title", path)
        if value["independent_teaching_value"] is not True:
            _fail("lift-review-invalid", "new general Card requires independent teaching value", path)
        if _operator_specific(title):
            _fail("lift-review-invalid", "new Card title must describe a reusable general mechanism", path)
    else:
        _fail("lift-review-invalid", "include target must be existing-card-example or new-general-card", path)


def validate_review(path: Path, proposal: ValidatedProposal) -> dict[str, Any]:
    review_path = Path(path)
    document = _load_review_document(review_path)
    validated = validate_lift_document("experience_review", document, SCHEMA_PATH, path=review_path)
    if validated["proposal_id"] != proposal.document["proposal_id"]:
        _fail("lift-review-invalid", "review proposal_id does not match proposal", review_path)
    if validated["proposal_sha256"] != proposal.sha256:
        _fail("lift-review-invalid", "review proposal_sha256 does not match proposal bytes", review_path)
    _validate_review_time(validated["reviewed_at"], review_path)
    decision = validated["decision"]
    target = validated["publication_target"]
    if decision == "include":
        if target is None:
            _fail("lift-review-invalid", "include review requires a publication target", review_path)
        _validate_publication_target(target, review_path)
    elif target is not None:
        _fail("lift-review-invalid", "defer/exclude review publication_target must be null", review_path)
    return validated


def validate_experience_tree(root: Path) -> dict[str, int]:
    skill_root = Path(root)
    candidate_root = skill_root / "candidates" / "experience"
    if not candidate_root.exists():
        return {"proposals": 0, "reviews": 0, "included": 0}
    if not candidate_root.is_dir():
        _fail("lift-tree-invalid", "experience candidate path must be a directory", candidate_root)

    proposals: dict[str, ValidatedProposal] = {}
    for path in sorted(candidate_root.glob("*.json")):
        proposal = validate_proposal(path)
        proposal_id = proposal.document["proposal_id"]
        if not isinstance(proposal_id, str) or not proposal_id:
            _fail("lift-tree-invalid", "proposal_id must be nonempty text", path)
        if path.stem != proposal_id:
            _fail("lift-tree-invalid", "proposal filename must match proposal_id", path)
        if proposal_id in proposals:
            _fail("lift-tree-invalid", f"duplicate proposal_id: {proposal_id}", path)
        proposals[proposal_id] = proposal

    review_root = candidate_root / "reviews"
    reviews: dict[str, Path] = {}
    included = 0
    if review_root.exists():
        if not review_root.is_dir():
            _fail("lift-tree-invalid", "experience reviews path must be a directory", review_root)
        review_paths = sorted((*review_root.glob("*.json"), *review_root.glob("*.yaml"), *review_root.glob("*.yml")))
        for path in review_paths:
            raw = _load_review_document(path)
            proposal_id = raw.get("proposal_id")
            if not isinstance(proposal_id, str) or proposal_id not in proposals:
                _fail("lift-tree-invalid", f"review references missing proposal: {proposal_id}", path)
            if proposal_id in reviews:
                _fail("lift-tree-invalid", f"duplicate review for proposal: {proposal_id}", path)
            review = validate_review(path, proposals[proposal_id])
            reviews[proposal_id] = path
            included += int(review["decision"] == "include")
    return {"proposals": len(proposals), "reviews": len(reviews), "included": included}


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=SKILL_ROOT)
    args = parser.parse_args(argv)
    result = validate_experience_tree(args.root)
    print(json.dumps({"schema_version": 1, "valid": True, **result}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))

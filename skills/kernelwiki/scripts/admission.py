from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

from catalog import card_to_catalog_record
from corpus import (
    ID_RE,
    PROTECTED_FIELDS,
    Corpus,
    GuidanceSchemaError,
    SourceRecord,
    WikiCard,
    load_corpus,
    validate_corpus,
    validate_guidance_schema,
)
from kernelwiki_common import (
    KernelWikiError,
    canonical_json_bytes,
    load_yaml_document,
    parse_markdown,
    require_within,
    sha256_bytes,
)
from provenance import CODE_ROLES, ProvenanceBundle, ProvenanceFile, load_provenance, validate_provenance
from role_context import (
    AuthoritySnapshot,
    RoleQueryContext,
    require_validated_authority_snapshot,
    require_validated_role_context,
)
from search import SearchCandidate, build_card_candidate, build_source_candidate


ADMISSION_STATUSES = frozenset({"admitted", "conditional", "analogy_only", "excluded"})
MATCH_CLASSES = frozenset({"exact", "family", "backend", "analogy-only", "unknown"})


@dataclass(frozen=True)
class ValidatedGuidanceBinding:
    guidance_id: str
    sketch_statement_ids: tuple[str, ...]
    permitted_change_family: str
    protected_fields: tuple[str, ...]


_ADMISSION_VALIDATION_TOKEN = object()


@dataclass(frozen=True)
class AdmissionDecision:
    status: str
    reasons: tuple[str, ...]
    match_class: str
    admitted_guidance_ids: tuple[str, ...]
    admitted_example_ids: tuple[str, ...]
    admitted_asset_ids: tuple[str, ...]
    _validation_token: object | None = field(default=None, init=False, repr=False, compare=False)
    _validation_fingerprint: str | None = field(default=None, init=False, repr=False, compare=False)
    _record_fingerprint: str | None = field(default=None, init=False, repr=False, compare=False)
    _candidate_fingerprint: str | None = field(default=None, init=False, repr=False, compare=False)


@dataclass(frozen=True)
class VersionClaimResolution:
    claim_id: str
    subject: str
    status: str
    supported_versions: tuple[str, ...]
    last_verified_at: str | None
    source_ids: tuple[str, ...]
    replacement_claim_id: str | None
    reason: str | None


@dataclass(frozen=True)
class CapabilityStatusResolution:
    capability_id: str
    status: str
    reason: str | None


@dataclass(frozen=True)
class _AssetRecord:
    source: SourceRecord
    bundle: ProvenanceBundle
    file: ProvenanceFile


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    return value


def protected_projection(sketch_result: Mapping[str, Any], decision_result: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        sketch = sketch_result["sketch"]
        operations = sketch["operations"]
        declarations = sketch["declarations"]
        projection = {
            "algorithm": {
                "scope_kind": sketch["scope"]["kind"],
                "operation_kinds": [item["kind"] for item in operations],
                "causal_nodes": sketch["causal_nodes"],
            },
            "dataflow": {
                "declarations": declarations,
                "operations": [
                    {key: item.get(key) for key in ("id", "inputs", "outputs", "index_domain", "mask")}
                    for item in operations
                ],
                "control": sketch["control"],
            },
            "precision": [{"id": item["id"], "dtype": item["dtype"]} for item in declarations],
            "effects": {
                "top": sketch["effects"],
                "operations": [{"id": item["id"], "effects": item["effects"]} for item in operations],
            },
            "aliases": sketch["effects"]["aliases"],
            "host-plan": decision_result["host_plan"],
            "public-interface": {
                "entrypoints": sketch["scope"]["entrypoints"],
                "unchanged_boundary": sketch["scope"]["unchanged_boundary"],
            },
        }
    except (KeyError, TypeError) as error:
        raise KernelWikiError("sketch-change-required", "validated Sketch/Decision cannot form protected projection") from error
    return _plain(projection)


def build_exact_guidance(sketch_result: Mapping[str, Any], decision_result: Mapping[str, Any]) -> Mapping[str, Any]:
    projection_sha = sha256_bytes(canonical_json_bytes(protected_projection(sketch_result, decision_result)))
    return {
        "id": "guidance-test-exact",
        "implementation_profile_ids": ["triton_mlu"],
        "target_ids": ["mlu590"],
        "runtime_fingerprints": ["triton 3.6.0 / CoreX 4.4.0"],
        "languages": ["triton"],
        "dtypes": ["fp32"],
        "shape_constraints": {"E": {"exact": 256}, "K": {"exact": 8}, "T": {"min": 1, "max": 4096}},
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


def _validate_guidance_schema(guidance: Any):
    try:
        return validate_guidance_schema(guidance)
    except GuidanceSchemaError as error:
        raise KernelWikiError("sketch-change-required", error.message) from error


def validate_guidance_binding(
    guidance: Mapping[str, Any],
    binding_ids: Sequence[str],
    sketch_result: Mapping[str, Any],
    decision_result: Mapping[str, Any],
) -> tuple[ValidatedGuidanceBinding, ...]:
    validated = _validate_guidance_schema(guidance)
    guidance_id = validated.guidance_id
    declared_ids = validated.statement_ids
    if isinstance(binding_ids, (str, bytes)) or not isinstance(binding_ids, Sequence):
        raise KernelWikiError("sketch-binding-required", "binding IDs must be a sequence")
    context_ids = tuple(binding_ids)
    if (
        not context_ids
        or any(not isinstance(item, str) or ID_RE.fullmatch(item) is None for item in context_ids)
        or context_ids != tuple(sorted(set(context_ids)))
        or context_ids != declared_ids
    ):
        raise KernelWikiError("sketch-binding-required", "context binding must equal sorted guidance statement_ids")
    statement_index = sketch_result.get("statement_index") if isinstance(sketch_result, Mapping) else None
    if not isinstance(statement_index, Mapping):
        raise KernelWikiError("sketch-binding-required", "validated Sketch statement index is missing")
    for statement_id in declared_ids:
        if not statement_id.startswith(("op.", "ctrl.", "guard.")) or statement_id not in statement_index:
            raise KernelWikiError("sketch-binding-required", f"unknown Sketch statement {statement_id}")
    actual_sha = sha256_bytes(canonical_json_bytes(protected_projection(sketch_result, decision_result)))
    if validated.protected_projection_sha256 != actual_sha:
        raise KernelWikiError("sketch-change-required", "protected semantic projection changed")
    return (
        ValidatedGuidanceBinding(
            guidance_id=guidance_id,
            sketch_statement_ids=declared_ids,
            permitted_change_family=validated.change_family,
            protected_fields=PROTECTED_FIELDS,
        ),
    )


def _record_fingerprint(record: WikiCard | SourceRecord) -> str:
    if type(record) is WikiCard:
        record_kind = "card"
        record_id = record.card_id
    elif type(record) is SourceRecord:
        record_kind = "source"
        record_id = record.source_id
    else:
        raise KernelWikiError("source-broken", "admission record type is invalid")
    payload = {
        "record_kind": record_kind,
        "record_id": record_id,
        "path": record.path,
        "metadata": record.metadata,
        "body": record.body,
    }
    return sha256_bytes(canonical_json_bytes(_plain(payload)))


def _candidate_fingerprint(candidate: SearchCandidate) -> str:
    if type(candidate) is not SearchCandidate or not isinstance(candidate.record, (WikiCard, SourceRecord)):
        raise KernelWikiError("source-broken", "admission candidate type is invalid")
    payload = {
        "record_kind": candidate.record_kind,
        "record_id": candidate.record_id,
        "path": candidate.path,
        "title": candidate.title,
        "record_type": candidate.record_type,
        "structured_fields": candidate.structured_fields,
        "body": candidate.body,
        "record_fingerprint": _record_fingerprint(candidate.record),
    }
    return sha256_bytes(canonical_json_bytes(_plain(payload)))


def _decision_public_fingerprint(decision: AdmissionDecision) -> str:
    payload = {
        "status": decision.status,
        "reasons": decision.reasons,
        "match_class": decision.match_class,
        "admitted_guidance_ids": decision.admitted_guidance_ids,
        "admitted_example_ids": decision.admitted_example_ids,
        "admitted_asset_ids": decision.admitted_asset_ids,
    }
    return sha256_bytes(canonical_json_bytes(_plain(payload)))


def _seal_decision(
    decision: AdmissionDecision,
    *,
    record: WikiCard | SourceRecord | None = None,
    candidate: SearchCandidate | None = None,
) -> AdmissionDecision:
    if candidate is not None:
        record = candidate.record
    object.__setattr__(decision, "_validation_token", _ADMISSION_VALIDATION_TOKEN)
    object.__setattr__(decision, "_validation_fingerprint", _decision_public_fingerprint(decision))
    object.__setattr__(
        decision,
        "_record_fingerprint",
        _record_fingerprint(record) if record is not None else None,
    )
    object.__setattr__(
        decision,
        "_candidate_fingerprint",
        _candidate_fingerprint(candidate) if candidate is not None else None,
    )
    return decision


def require_validated_admission_decision(
    value: Any,
    *,
    record: WikiCard | SourceRecord | None = None,
    candidate: SearchCandidate | None = None,
) -> AdmissionDecision:
    if type(value) is not AdmissionDecision or value._validation_token is not _ADMISSION_VALIDATION_TOKEN:
        raise KernelWikiError("role-query-invalid", "decision was not issued by admission")
    if value.status not in ADMISSION_STATUSES or value.match_class not in MATCH_CLASSES:
        raise KernelWikiError("role-query-invalid", "validated decision fields are invalid")
    for collection in (
        value.reasons,
        value.admitted_guidance_ids,
        value.admitted_example_ids,
        value.admitted_asset_ids,
    ):
        if (
            not isinstance(collection, tuple)
            or any(not isinstance(item, str) or not item for item in collection)
            or collection != tuple(sorted(set(collection)))
        ):
            raise KernelWikiError("role-query-invalid", "validated decision collections are invalid")
    if value._validation_fingerprint != _decision_public_fingerprint(value):
        raise KernelWikiError("role-query-invalid", "validated decision changed after admission")
    if candidate is not None and record is not None:
        raise KernelWikiError("role-query-invalid", "decision validation accepts either record or candidate, not both")
    if candidate is not None:
        expected_record = _record_fingerprint(candidate.record)
        expected_candidate = _candidate_fingerprint(candidate)
        if value._record_fingerprint != expected_record or value._candidate_fingerprint != expected_candidate:
            raise KernelWikiError("role-query-invalid", "decision does not match its canonical candidate")
    elif record is not None:
        if value._record_fingerprint != _record_fingerprint(record) or value._candidate_fingerprint is not None:
            raise KernelWikiError("role-query-invalid", "decision does not match its canonical record")
    return value


def _decision(
    status: str,
    *,
    reasons: Sequence[str] = (),
    match_class: str = "unknown",
    guidance_ids: Sequence[str] = (),
    example_ids: Sequence[str] = (),
    asset_ids: Sequence[str] = (),
    record: WikiCard | SourceRecord | None = None,
    candidate: SearchCandidate | None = None,
) -> AdmissionDecision:
    if status not in ADMISSION_STATUSES or match_class not in MATCH_CLASSES:
        raise KernelWikiError("contract-unsupported", "invalid internal admission decision")
    decision = AdmissionDecision(
        status=status,
        reasons=tuple(sorted(set(reasons))),
        match_class=match_class,
        admitted_guidance_ids=tuple(sorted(set(guidance_ids))),
        admitted_example_ids=tuple(sorted(set(example_ids))),
        admitted_asset_ids=tuple(sorted(set(asset_ids))),
    )
    return _seal_decision(decision, record=record, candidate=candidate)


def _bind_decision_to_candidate(decision: AdmissionDecision, candidate: SearchCandidate) -> AdmissionDecision:
    require_validated_admission_decision(decision)
    return _decision(
        decision.status,
        reasons=decision.reasons,
        match_class=decision.match_class,
        guidance_ids=decision.admitted_guidance_ids,
        example_ids=decision.admitted_example_ids,
        asset_ids=decision.admitted_asset_ids,
        candidate=candidate,
    )


def _require_context(context: Any) -> RoleQueryContext:
    return require_validated_role_context(context)


def _require_authority(authority: Any) -> AuthoritySnapshot:
    return require_validated_authority_snapshot(authority)


def _backend_token(target_id: str) -> str:
    token = target_id.strip().casefold()
    for backend in ("ascend", "mlu", "cuda", "rocm", "cpu"):
        if token == backend or re.fullmatch(rf"{backend}[0-9][a-z0-9]*", token):
            return backend
    return token


def _classify_target(disposition: Any, targets: Any, context: RoleQueryContext) -> str:
    if disposition not in MATCH_CLASSES or not isinstance(targets, Sequence) or isinstance(targets, (str, bytes)):
        return "unknown"
    if any(not isinstance(item, str) or not item.strip() for item in targets):
        return "unknown"
    target_values = tuple(item.strip().casefold() for item in targets)
    context_target = context.target_id.strip().casefold()
    context_backend = _backend_token(context_target)
    target_backends = {_backend_token(item) for item in target_values}
    if disposition == "exact":
        return "exact" if context_target in target_values else "analogy-only"
    if disposition == "family":
        return "family" if context_backend in target_backends else "analogy-only"
    if disposition == "backend":
        return "backend" if context_backend in target_backends else "analogy-only"
    if disposition == "analogy-only":
        return "analogy-only"
    return "unknown"


def classify_designer_match(card: WikiCard, context: RoleQueryContext) -> str:
    context = _require_context(context)
    _, canonical = _canonical_card(card)
    return _classify_target(canonical.metadata.get("target_match"), canonical.metadata.get("targets"), context)


def match_dtype_shape_regime(guidance: Mapping[str, Any], context: RoleQueryContext) -> bool:
    dtypes = guidance.get("dtypes")
    constraints = guidance.get("shape_constraints")
    if not isinstance(dtypes, Sequence) or isinstance(dtypes, (str, bytes)) or not isinstance(constraints, Mapping):
        return False
    if not set(context.dtypes) <= {str(item) for item in dtypes}:
        return False
    for dimension, raw_constraint in constraints.items():
        if dimension not in context.shape_signature:
            return False
        value = context.shape_signature[dimension]
        if isinstance(value, bool) or not isinstance(value, int):
            return False
        if not isinstance(raw_constraint, Mapping):
            return False
        if set(raw_constraint) == {"exact"}:
            if value != raw_constraint["exact"]:
                return False
        elif set(raw_constraint) == {"min", "max"}:
            minimum = raw_constraint["min"]
            maximum = raw_constraint["max"]
            if any(isinstance(item, bool) or not isinstance(item, int) for item in (minimum, maximum)):
                return False
            if value < minimum or value > maximum:
                return False
        else:
            return False
    return True


def _root_for_path(path: Path) -> Path | None:
    try:
        path = Path(path).resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    for parent in path.parents:
        if (parent / "data" / "taxonomy.yaml").is_file() and (parent / "data" / "version-claims.yaml").is_file():
            return parent
    return None


def _load_authoritative_corpus(path: Path) -> Corpus:
    root = _root_for_path(path)
    if root is None:
        raise KernelWikiError("source-broken", "record has no validated KernelWiki root", path)
    try:
        corpus = load_corpus(root)
        validate_corpus(corpus)
    except KernelWikiError as error:
        raise KernelWikiError("source-broken", f"corpus authority is invalid: {error.code}", path) from error
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise KernelWikiError("source-broken", "corpus authority cannot be loaded", path) from error
    return corpus


def _canonical_card(card: WikiCard) -> tuple[Corpus, WikiCard]:
    if type(card) is not WikiCard or not isinstance(card.metadata, Mapping):
        raise KernelWikiError("source-broken", "card must be a WikiCard")
    corpus = _load_authoritative_corpus(card.path)
    card_id = card.metadata.get("id")
    canonical = corpus.cards.get(card_id) if isinstance(card_id, str) else None
    if (
        canonical is None
        or card.path != canonical.path
        or dict(card.metadata) != dict(canonical.metadata)
        or card.body != canonical.body
    ):
        raise KernelWikiError("source-broken", "Card differs from validated on-disk authority", card.path)
    return corpus, canonical


def _canonical_source(source: SourceRecord) -> tuple[Corpus, SourceRecord]:
    if type(source) is not SourceRecord or not isinstance(source.metadata, Mapping):
        raise KernelWikiError("source-broken", "source must be a SourceRecord")
    corpus = _load_authoritative_corpus(source.path)
    source_id = source.metadata.get("id")
    canonical = corpus.sources.get(source_id) if isinstance(source_id, str) else None
    if (
        canonical is None
        or source.path != canonical.path
        or dict(source.metadata) != dict(canonical.metadata)
        or source.body != canonical.body
    ):
        raise KernelWikiError("source-broken", "Source differs from validated on-disk authority", source.path)
    return corpus, canonical


def _record_is_intact(record: WikiCard | SourceRecord) -> bool:
    try:
        if isinstance(record, WikiCard):
            _canonical_card(record)
        elif isinstance(record, SourceRecord):
            _canonical_source(record)
        else:
            return False
    except (KernelWikiError, OSError, RuntimeError, TypeError, ValueError):
        return False
    return True


def _source_bundle(source: SourceRecord) -> ProvenanceBundle | None:
    artifact_dir = source.metadata.get("artifact_dir")
    if artifact_dir is None:
        return None
    root = _root_for_path(source.path)
    if root is None:
        raise KernelWikiError("source-broken", "Source has no validated KernelWiki root", source.path)
    try:
        bundle_dir = require_within(root, root / str(artifact_dir))
    except (KernelWikiError, OSError, RuntimeError, ValueError) as error:
        raise KernelWikiError("source-broken", "Source artifact path is invalid", source.path) from error
    manifest = bundle_dir / "PROVENANCE.yaml"
    bundle = load_provenance(manifest)
    validate_provenance(bundle, root)
    if source.source_id not in bundle.source_ids:
        raise KernelWikiError("source-broken", "Source is absent from provenance source_ids", manifest)
    return bundle


def _load_card_sources(card: WikiCard) -> tuple[SourceRecord, ...]:
    corpus, canonical = _canonical_card(card)
    sources: list[SourceRecord] = []
    for source_id in canonical.metadata.get("sources", ()):
        source = corpus.sources.get(source_id)
        if source is None:
            raise KernelWikiError("source-broken", f"Card Source {source_id} is missing", canonical.path)
        _, canonical_source = _canonical_source(source)
        sources.append(canonical_source)
    return tuple(sources)


def _source_reasons(source: SourceRecord, context: RoleQueryContext, *, require_coder_scope: bool) -> set[str]:
    reasons: set[str] = set()
    try:
        _, source = _canonical_source(source)
    except (KernelWikiError, OSError, RuntimeError, TypeError, ValueError):
        return {"source-broken"}
    metadata = source.metadata
    if require_coder_scope:
        if metadata.get("license_state") != "approved":
            reasons.add("license-unapproved")
        if metadata.get("target_disposition") != "exact" or context.target_id not in metadata.get("target_ids", ()):
            reasons.add("target-mismatch")
        audiences = metadata.get("audiences", ())
        if "coder" not in audiences:
            reasons.add("audience-mismatch")
        if context.implementation_profile_id not in metadata.get("implementation_profile_ids", ()):
            reasons.add("profile-version-mismatch")
        runtimes = metadata.get("runtime_fingerprints", ())
        if context.runtime_fingerprint is None or context.runtime_fingerprint.strip() not in runtimes:
            reasons.add("runtime-mismatch")
        languages = metadata.get("languages", ())
        if not set(context.languages) <= set(languages):
            reasons.add("target-mismatch")
        if metadata.get("source_kind") == "local-campaign":
            if metadata.get("profile_authority") != "current-vnext" or metadata.get("strict_vnext_validated") is not True:
                reasons.add("profile-version-mismatch")
            if metadata.get("missing_evidence"):
                reasons.add("capability-unknown")
    artifact_dir = metadata.get("artifact_dir")
    if artifact_dir is not None:
        try:
            bundle = _source_bundle(source)
            if bundle is None or bundle.license_state != "approved":
                reasons.add("license-unapproved")
        except (KernelWikiError, OSError, RuntimeError, TypeError, ValueError):
            reasons.add("source-broken")
    return reasons


def _designer_source_reasons(source: SourceRecord) -> tuple[set[str], SourceRecord | None]:
    try:
        _, canonical = _canonical_source(source)
    except (KernelWikiError, OSError, RuntimeError, TypeError, ValueError):
        return {"source-broken"}, None
    reasons: set[str] = set()
    if canonical.metadata.get("artifact_dir") is not None:
        if canonical.metadata.get("license_state") != "approved":
            reasons.add("license-unapproved")
        try:
            bundle = _source_bundle(canonical)
            if bundle is None or bundle.license_state != "approved" or "designer" not in bundle.allowed_audiences:
                reasons.add("license-unapproved")
        except (KernelWikiError, OSError, RuntimeError, TypeError, ValueError):
            reasons.add("source-broken")
    return reasons, canonical


def _authority_reasons(context: RoleQueryContext, authority: AuthoritySnapshot | None) -> set[str]:
    if context.implementation_profile_status == "missing":
        return {"profile-missing"}
    try:
        authority = _require_authority(authority)
    except KernelWikiError:
        return {"contract-unsupported"}
    context_artifact_hashes = {name: reference.sha256 for name, reference in context.artifacts.items()}
    if context_artifact_hashes != dict(authority.artifact_hashes):
        return {"contract-unsupported"}
    reasons: set[str] = set()
    if context.contract_version != authority.contract_version or context.loop_contract_identity != authority.loop_contract_identity:
        return {"contract-unsupported"}
    profile = authority.profile
    if (
        context.implementation_profile_id != profile.get("implementation_profile_id")
        or context.implementation_profile_status != profile.get("profile_status")
    ):
        reasons.add("profile-version-mismatch")
    implementation = profile.get("implementation")
    toolchain = implementation.get("toolchain") if isinstance(implementation, Mapping) else None
    if context.runtime_fingerprint is None or not isinstance(toolchain, str) or context.runtime_fingerprint.strip() != toolchain.strip():
        reasons.add("runtime-mismatch")
    identity_match = profile.get("identity_match")
    permitted_targets = identity_match.get("permitted_target_ids", ()) if isinstance(identity_match, Mapping) else ()
    if context.target_id not in permitted_targets:
        reasons.add("target-mismatch")
    return reasons


def resolve_capability_status(
    authority: AuthoritySnapshot | None,
    capability_id: str,
) -> CapabilityStatusResolution:
    if not isinstance(capability_id, str) or ID_RE.fullmatch(capability_id) is None:
        raise KernelWikiError("capability-unknown", "capability ID must be a valid identifier")
    if authority is None:
        return CapabilityStatusResolution(capability_id, "unknown", "profile-missing")
    validated = require_validated_authority_snapshot(authority)
    matrix = validated.profile.get("capability_matrix", ())
    if not isinstance(matrix, (list, tuple)):
        return CapabilityStatusResolution(capability_id, "unknown", "capability-unknown")
    status = "unknown"
    for item in matrix:
        if isinstance(item, Mapping) and item.get("id") == capability_id:
            raw_status = item.get("status")
            status = raw_status if isinstance(raw_status, str) else "unknown"
            break
    if status in {"supported", "constrained"}:
        reason = None
    elif status in {"unsupported", "prohibited"}:
        reason = "capability-unsupported"
    else:
        status = "unknown"
        reason = "capability-unknown"
    return CapabilityStatusResolution(capability_id, status, reason)


def _canonical_corpus_for_version_resolution(corpus: Corpus) -> Corpus:
    if type(corpus) is not Corpus:
        raise KernelWikiError("version-stale", "version resolution requires a validated Corpus")
    try:
        canonical = load_corpus(corpus.root)
        validate_corpus(canonical)
    except (KernelWikiError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise KernelWikiError("version-stale", "version registry is not valid") from error
    return canonical


def resolve_version_claim(corpus: Corpus, claim_id: str) -> VersionClaimResolution:
    if not isinstance(claim_id, str) or ID_RE.fullmatch(claim_id) is None:
        raise KernelWikiError("version-stale", "version claim ID must be valid")
    canonical = _canonical_corpus_for_version_resolution(corpus)
    claim = next((item for item in canonical.version_claims if item.get("id") == claim_id), None)
    if claim is None:
        raise KernelWikiError("version-stale", f"version claim {claim_id} is missing")
    status = str(claim["status"])
    return VersionClaimResolution(
        claim_id=claim_id,
        subject=str(claim["subject"]),
        status=status,
        supported_versions=tuple(claim["supported_versions"]),
        last_verified_at=claim["last_verified_at"],
        source_ids=tuple(claim["source_ids"]),
        replacement_claim_id=claim["replacement_claim_id"],
        reason=None if status == "current" else "version-stale",
    )


def _capability_reasons(guidance: Mapping[str, Any], authority: AuthoritySnapshot) -> set[str]:
    reasons: set[str] = set()
    for capability in guidance.get("required_capabilities", ()):
        resolution = resolve_capability_status(authority, capability)
        if resolution.reason is not None:
            reasons.add(resolution.reason)
    return reasons


def _version_resolutions(card: WikiCard, guidance: Mapping[str, Any] | None = None) -> tuple[VersionClaimResolution, ...]:
    corpus, canonical = _canonical_card(card)
    raw_claim_ids = canonical.metadata.get("version_sensitive", ())
    if not isinstance(raw_claim_ids, (list, tuple)) or any(not isinstance(item, str) for item in raw_claim_ids):
        raise KernelWikiError("version-stale", "Card version claims are malformed", canonical.path)
    claim_ids = set(raw_claim_ids)
    if guidance is not None:
        raw_guidance_ids = guidance.get("version_claim_ids", ())
        if not isinstance(raw_guidance_ids, list) or any(not isinstance(item, str) for item in raw_guidance_ids):
            raise KernelWikiError("version-stale", "guidance version claims are malformed", canonical.path)
        claim_ids.update(raw_guidance_ids)
    return tuple(resolve_version_claim(corpus, claim_id) for claim_id in sorted(claim_ids))


def _version_reasons(card: WikiCard, guidance: Mapping[str, Any] | None = None) -> set[str]:
    try:
        resolutions = _version_resolutions(card, guidance)
    except (KernelWikiError, OSError, RuntimeError, TypeError, ValueError):
        return {"version-stale"}
    return {resolution.reason for resolution in resolutions if resolution.reason is not None}


def _example_reasons(card: WikiCard, example_id: str, context: RoleQueryContext) -> set[str]:
    raw_examples = card.metadata.get("examples", ()) if isinstance(card.metadata, Mapping) else ()
    if not isinstance(raw_examples, (list, tuple)):
        return {"source-broken"}
    examples = {
        item["id"]: item
        for item in raw_examples
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    example = examples.get(example_id)
    if example is None:
        return {"source-broken"}
    reasons: set[str] = set()
    if example.get("target_id") != context.target_id:
        reasons.add("target-mismatch")
    if example.get("implementation_profile_id") != context.implementation_profile_id:
        reasons.add("profile-version-mismatch")
    runtime = example.get("runtime_fingerprint")
    if not isinstance(runtime, str) or context.runtime_fingerprint is None or runtime.strip() != context.runtime_fingerprint.strip():
        reasons.add("runtime-mismatch")
    if example.get("dtype") not in context.dtypes:
        reasons.add("target-mismatch")
    shape = example.get("shape")
    if not isinstance(shape, Mapping):
        reasons.add("target-mismatch")
    else:
        for dimension, size in shape.items():
            if not isinstance(size, int) or context.shape_signature.get(dimension) != size:
                reasons.add("target-mismatch")
    if example.get("profile_authority") != "current-vnext":
        reasons.add("profile-version-mismatch")
    try:
        source = next(item for item in _load_card_sources(card) if item.source_id == example.get("source_id"))
    except (KernelWikiError, StopIteration):
        reasons.add("source-broken")
    else:
        reasons.update(_source_reasons(source, context, require_coder_scope=True))
    return reasons


def _asset_records(card: WikiCard) -> Mapping[str, _AssetRecord]:
    records: dict[str, _AssetRecord] = {}
    for source in _load_card_sources(card):
        bundle = _source_bundle(source)
        if bundle is None:
            continue
        for item in bundle.files:
            if item.local_path in records:
                raise KernelWikiError("source-broken", f"duplicate Card asset ID {item.local_path}", card.path)
            records[item.local_path] = _AssetRecord(source=source, bundle=bundle, file=item)
    return records


def _asset_reasons(
    card: WikiCard,
    asset_id: str,
    context: RoleQueryContext,
    eligible_ids: Sequence[str],
) -> set[str]:
    if asset_id not in eligible_ids:
        return {"artifact-designer-only"}
    try:
        record = _asset_records(card).get(asset_id)
    except (KernelWikiError, OSError, RuntimeError, ValueError):
        return {"source-broken"}
    if record is None:
        return {"source-broken"}
    reasons = _source_reasons(record.source, context, require_coder_scope=True)
    if record.bundle.license_state != "approved":
        reasons.add("license-unapproved")
    if "coder" not in record.bundle.allowed_audiences or record.bundle.coder_access == "denied":
        reasons.add("artifact-designer-only")
    if record.bundle.coder_access == "snippet-only" and record.file.role != "snippet":
        reasons.add("artifact-designer-only")
    if record.file.role not in CODE_ROLES:
        reasons.add("artifact-designer-only")
    return reasons


def _guidance_reasons(
    card: WikiCard,
    guidance: Mapping[str, Any],
    context: RoleQueryContext,
    authority: AuthoritySnapshot | None,
) -> tuple[set[str], tuple[str, ...], str | None]:
    reasons: set[str] = set()
    try:
        guidance_id = _validate_guidance_schema(guidance).guidance_id
    except KernelWikiError:
        return {"sketch-change-required"}, (), None
    if context.implementation_profile_id not in guidance["implementation_profile_ids"]:
        reasons.add("profile-version-mismatch")
    if context.target_id not in guidance["target_ids"] or card.metadata.get("target_match") != "exact":
        reasons.add("target-mismatch")
    if context.runtime_fingerprint is None or context.runtime_fingerprint.strip() not in guidance["runtime_fingerprints"]:
        reasons.add("runtime-mismatch")
    if not set(context.languages) <= set(guidance["languages"]):
        reasons.add("target-mismatch")
    if not guidance["dtypes"] or not match_dtype_shape_regime(guidance, context):
        reasons.add("target-mismatch")

    validated_authority: AuthoritySnapshot | None
    try:
        validated_authority = _require_authority(authority)
    except KernelWikiError:
        validated_authority = None
    if validated_authority is None:
        if guidance["required_capabilities"]:
            reasons.add("capability-unknown")
        bindings = context.guidance_bindings.get(guidance_id, ())
        if not bindings:
            reasons.add("sketch-binding-required")
    else:
        reasons.update(_capability_reasons(guidance, validated_authority))
        bindings = context.guidance_bindings.get(guidance_id, ())
        try:
            validate_guidance_binding(
                guidance,
                bindings,
                validated_authority.sketch_result,
                validated_authority.decision_result,
            )
        except KernelWikiError as error:
            reasons.add(
                error.code
                if error.code in {"sketch-binding-required", "sketch-change-required"}
                else "sketch-change-required"
            )
    reasons.update(_version_reasons(card, guidance))
    admitted_examples: list[str] = []
    for example_id in guidance["eligible_example_ids"]:
        example_reasons = _example_reasons(card, example_id, context)
        reasons.update(example_reasons)
        if not example_reasons:
            admitted_examples.append(example_id)
    for asset_id in guidance["eligible_asset_ids"]:
        reasons.update(_asset_reasons(card, asset_id, context, guidance["eligible_asset_ids"]))
    return reasons, tuple(sorted(admitted_examples)), guidance_id


def relevant_unknown_capabilities(
    card: WikiCard,
    guidance: Mapping[str, Any],
    context: RoleQueryContext,
    authority: AuthoritySnapshot,
) -> tuple[CapabilityStatusResolution, ...]:
    """Return Unknown capabilities only when guidance is otherwise exact and admissible."""
    try:
        context = _require_context(context)
        authority = _require_authority(authority)
        _, canonical = _canonical_card(card)
    except (KernelWikiError, OSError, RuntimeError, TypeError, ValueError):
        return ()
    if context.role != "coder" or context.implementation_profile_status == "missing":
        return ()
    base_reasons = _authority_reasons(context, authority)
    metadata = canonical.metadata
    if "coder" not in metadata.get("audiences", ()):
        base_reasons.add("audience-mismatch")
    if metadata.get("target_match") != "exact" or context.target_id not in metadata.get("targets", ()):
        base_reasons.add("target-mismatch")
    try:
        for source in _load_card_sources(canonical):
            base_reasons.update(_source_reasons(source, context, require_coder_scope=True))
    except (KernelWikiError, OSError, RuntimeError, TypeError, ValueError):
        base_reasons.add("source-broken")
    if base_reasons:
        return ()
    item_reasons, _, guidance_id = _guidance_reasons(canonical, guidance, context, authority)
    if guidance_id is None or "capability-unknown" not in item_reasons:
        return ()
    if item_reasons - {"capability-unknown"}:
        return ()
    resolutions = tuple(
        resolve_capability_status(authority, capability_id)
        for capability_id in guidance["required_capabilities"]
    )
    return tuple(
        resolution
        for resolution in resolutions
        if resolution.reason == "capability-unknown"
    )


def _admit_card_impl(
    card: WikiCard,
    context: RoleQueryContext,
    authority: AuthoritySnapshot | None,
) -> AdmissionDecision:
    context = _require_context(context)
    if context.role == "coder" and context.implementation_profile_status == "missing":
        return _decision("excluded", reasons=("profile-missing",), match_class="unknown")

    try:
        _, canonical = _canonical_card(card)
    except (KernelWikiError, OSError, RuntimeError, TypeError, ValueError):
        canonical = None

    if context.role == "designer":
        if canonical is None:
            return _decision("excluded", reasons=("source-broken",), match_class="unknown")
        match_class = _classify_target(
            canonical.metadata.get("target_match"), canonical.metadata.get("targets"), context
        )
        reasons = _version_reasons(canonical)
        if "designer" not in canonical.metadata.get("audiences", ()):
            reasons.add("audience-mismatch")
        if "audience-mismatch" in reasons:
            return _decision("excluded", reasons=reasons, match_class=match_class)
        status = (
            "conditional"
            if reasons or match_class == "unknown"
            else "analogy_only"
            if match_class == "analogy-only"
            else "admitted"
        )
        return _decision(status, reasons=reasons, match_class=match_class)

    reasons = _authority_reasons(context, authority)
    guidance_authority = None if "contract-unsupported" in reasons else authority
    working = canonical if canonical is not None else card
    if canonical is None:
        reasons.add("source-broken")
    metadata = working.metadata if isinstance(working, WikiCard) and isinstance(working.metadata, Mapping) else {}
    audiences = metadata.get("audiences", ())
    if not isinstance(audiences, (list, tuple)) or any(not isinstance(item, str) for item in audiences):
        reasons.add("source-broken")
        audiences = ()
    if "coder" not in audiences:
        reasons.add("audience-mismatch")
    targets = metadata.get("targets", ())
    if not isinstance(targets, (list, tuple)) or any(not isinstance(item, str) for item in targets):
        reasons.add("source-broken")
        targets = ()
    if metadata.get("target_match") != "exact" or context.target_id not in targets:
        reasons.add("target-mismatch")
    if isinstance(working, WikiCard):
        reasons.update(_version_reasons(working))

    if canonical is not None:
        try:
            referenced_sources = _load_card_sources(canonical)
        except (KernelWikiError, OSError, RuntimeError, TypeError, ValueError):
            reasons.add("source-broken")
        else:
            for source in referenced_sources:
                reasons.update(_source_reasons(source, context, require_coder_scope=True))

    access = metadata.get("coder_access") if canonical is not None else None
    passing_guidance: list[str] = []
    passing_examples: set[str] = set()
    failed_reasons: set[str] = set()
    if not isinstance(access, Mapping) or set(access) != {"page", "guidance"} or access.get("page") != "exact-profile":
        reasons.add("profile-version-mismatch")
        guidance_items: Any = ()
    else:
        guidance_items = access.get("guidance")
    if not isinstance(guidance_items, list) or not guidance_items:
        failed_reasons.add("profile-version-mismatch")
    else:
        for guidance in guidance_items:
            if not isinstance(guidance, Mapping):
                failed_reasons.add("sketch-change-required")
                continue
            item_reasons, example_ids, guidance_id = _guidance_reasons(
                working, guidance, context, guidance_authority
            )
            if item_reasons:
                failed_reasons.update(item_reasons)
            elif guidance_id is not None:
                passing_guidance.append(guidance_id)
                passing_examples.update(example_ids)
    if not passing_guidance:
        reasons.update(failed_reasons or {"profile-version-mismatch"})
    if reasons:
        return _decision(
            "excluded",
            reasons=reasons,
            match_class="exact" if "target-mismatch" not in reasons else "unknown",
        )
    return _decision(
        "admitted",
        match_class="exact",
        guidance_ids=passing_guidance,
        example_ids=tuple(passing_examples),
    )


def admit_card(
    card: WikiCard,
    context: RoleQueryContext,
    authority: AuthoritySnapshot | None,
) -> AdmissionDecision:
    decision = _admit_card_impl(card, context, authority)
    if context.role == "coder" and context.implementation_profile_status == "missing":
        return decision
    try:
        _, canonical = _canonical_card(card)
    except (KernelWikiError, OSError, RuntimeError, TypeError, ValueError):
        return decision
    return _decision(
        decision.status,
        reasons=decision.reasons,
        match_class=decision.match_class,
        guidance_ids=decision.admitted_guidance_ids,
        example_ids=decision.admitted_example_ids,
        asset_ids=decision.admitted_asset_ids,
        record=canonical,
    )


def _admit_source_impl(
    source: SourceRecord,
    context: RoleQueryContext,
    authority: AuthoritySnapshot | None,
) -> AdmissionDecision:
    context = _require_context(context)
    if context.role == "coder" and context.implementation_profile_status == "missing":
        return _decision("excluded", reasons=("profile-missing",), match_class="unknown")
    if context.role == "designer":
        reasons, canonical = _designer_source_reasons(source)
        if canonical is None:
            return _decision("excluded", reasons=reasons, match_class="unknown")
        match_class = _classify_target(
            canonical.metadata.get("target_disposition"), canonical.metadata.get("target_ids", ()), context
        )
        audiences = canonical.metadata.get("audiences")
        if audiences is not None and "designer" not in audiences:
            reasons.add("audience-mismatch")
        if reasons:
            return _decision("excluded", reasons=reasons, match_class=match_class)
        status = (
            "analogy_only"
            if match_class == "analogy-only"
            else "conditional"
            if match_class == "unknown"
            else "admitted"
        )
        return _decision(status, match_class=match_class)

    reasons = _authority_reasons(context, authority)
    reasons.update(_source_reasons(source, context, require_coder_scope=True))
    root = _root_for_path(source.path) if isinstance(source, SourceRecord) else None
    if root is not None:
        try:
            claims_doc = load_yaml_document(root / "data" / "version-claims.yaml")
            claims = claims_doc.get("claims", ()) if isinstance(claims_doc, Mapping) else ()
            source_id = source.metadata.get("id") if isinstance(source.metadata, Mapping) else None
            for claim in claims:
                if isinstance(claim, Mapping) and source_id in claim.get("source_ids", ()) and claim.get("status") != "current":
                    reasons.add("version-stale")
        except (KernelWikiError, AttributeError, OSError, RuntimeError, TypeError, ValueError):
            reasons.add("source-broken")
    match_class = "exact" if "target-mismatch" not in reasons else "unknown"
    if reasons:
        return _decision("excluded", reasons=reasons, match_class=match_class)
    return _decision("admitted", match_class="exact")


def admit_source(
    source: SourceRecord,
    context: RoleQueryContext,
    authority: AuthoritySnapshot | None,
) -> AdmissionDecision:
    decision = _admit_source_impl(source, context, authority)
    if context.role == "coder" and context.implementation_profile_status == "missing":
        return decision
    try:
        _, canonical = _canonical_source(source)
    except (KernelWikiError, OSError, RuntimeError, TypeError, ValueError):
        return decision
    return _decision(
        decision.status,
        reasons=decision.reasons,
        match_class=decision.match_class,
        guidance_ids=decision.admitted_guidance_ids,
        example_ids=decision.admitted_example_ids,
        asset_ids=decision.admitted_asset_ids,
        record=canonical,
    )


def admit_candidate(
    candidate: SearchCandidate,
    context: RoleQueryContext,
    authority: AuthoritySnapshot | None,
) -> AdmissionDecision:
    context = _require_context(context)
    if type(candidate) is not SearchCandidate:
        raise KernelWikiError("source-broken", "candidate must be a neutral SearchCandidate")
    try:
        if candidate.record_kind == "card" and isinstance(candidate.record, WikiCard):
            corpus, record = _canonical_card(candidate.record)
            expected = build_card_candidate(record, card_to_catalog_record(record, corpus), corpus)
            if candidate != expected:
                raise KernelWikiError("source-broken", "SearchCandidate differs from canonical Card candidate")
            return _bind_decision_to_candidate(admit_card(record, context, authority), expected)
        if candidate.record_kind == "source" and isinstance(candidate.record, SourceRecord):
            corpus, record = _canonical_source(candidate.record)
            expected = build_source_candidate(record, corpus)
            if candidate != expected:
                raise KernelWikiError("source-broken", "SearchCandidate differs from canonical Source candidate")
            return _bind_decision_to_candidate(admit_source(record, context, authority), expected)
    except (KernelWikiError, OSError, RuntimeError, TypeError, ValueError):
        return _decision("excluded", reasons=("source-broken",), match_class="unknown")
    return _decision("excluded", reasons=("source-broken",), match_class="unknown")


def admit_asset(
    card: WikiCard,
    asset_id: str,
    context: RoleQueryContext,
    authority: AuthoritySnapshot | None,
) -> AdmissionDecision:
    context = _require_context(context)
    if not isinstance(asset_id, str) or not asset_id.strip():
        raise KernelWikiError("source-broken", "asset_id must be a nonempty string")
    if context.role == "designer":
        try:
            record = _asset_records(card).get(asset_id)
        except (KernelWikiError, OSError, RuntimeError, ValueError):
            record = None
        if record is None:
            return _decision("excluded", reasons=("source-broken",), match_class=classify_designer_match(card, context))
        if record.source.metadata.get("license_state") != "approved" or record.bundle.license_state != "approved":
            return _decision("excluded", reasons=("license-unapproved",), match_class=classify_designer_match(card, context))
        source_audiences = record.source.metadata.get("audiences")
        if (source_audiences is not None and "designer" not in source_audiences) or "designer" not in record.bundle.allowed_audiences:
            return _decision("excluded", reasons=("artifact-designer-only",), match_class=classify_designer_match(card, context))
        return _decision("admitted", match_class=classify_designer_match(card, context), asset_ids=(asset_id,))
    if context.implementation_profile_status == "missing":
        return _decision("excluded", reasons=("profile-missing",), match_class="unknown")
    page = admit_card(card, context, authority)
    eligible_ids: tuple[str, ...] = ()
    try:
        _, canonical = _canonical_card(card)
        access = canonical.metadata.get("coder_access")
        if isinstance(access, Mapping) and isinstance(access.get("guidance"), list):
            values: list[str] = []
            for guidance in access["guidance"]:
                if (
                    isinstance(guidance, Mapping)
                    and guidance.get("id") in page.admitted_guidance_ids
                    and isinstance(guidance.get("eligible_asset_ids"), list)
                ):
                    values.extend(item for item in guidance["eligible_asset_ids"] if isinstance(item, str))
            eligible_ids = tuple(sorted(set(values)))
    except (KernelWikiError, OSError, RuntimeError, TypeError, ValueError):
        pass
    reasons = set(page.reasons)
    reasons.update(_asset_reasons(card, asset_id, context, eligible_ids))
    if reasons:
        return _decision("excluded", reasons=reasons, match_class=page.match_class)
    return _decision("admitted", match_class="exact", asset_ids=(asset_id,))

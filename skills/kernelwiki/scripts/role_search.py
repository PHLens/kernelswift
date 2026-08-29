from __future__ import annotations

import base64
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

from admission import (
    AdmissionDecision,
    _asset_records,
    _authority_reasons,
    _decision,
    admit_asset,
    admit_candidate,
    admit_source,
    classify_designer_match,
    relevant_unknown_capabilities,
    require_validated_admission_decision,
)
from corpus import Corpus, SourceRecord, WikiCard, load_corpus
from kernel_opt_bridge import LoopContractIdentity
from kernelwiki_common import KernelWikiError, canonical_json_bytes, require_within, sha256_bytes
from role_context import (
    AuthoritySnapshot,
    RoleQueryContext,
    _context_fingerprint,
    require_validated_authority_snapshot,
    require_validated_role_context,
)
from search import (
    FILTER_FIELDS,
    PageResult,
    QueryRequest,
    SearchCandidate,
    SearchHit,
    authoritative_search_corpus,
    collect_unlimited_candidates,
    page_payload,
    parse_query_request,
    retrieve_page,
    score_search_candidate,
)


ROLE_GROUPS = (
    "admitted",
    "conditional",
    "analogy_only",
    "counterexamples",
    "capability_gaps",
    "excluded",
)
DEFAULT_GROUP_LIMITS = {
    "admitted": 20,
    "conditional": 20,
    "analogy_only": 20,
    "counterexamples": 8,
    "capability_gaps": 8,
    "excluded": 20,
}
TARGET_SPECIFICITY = {"exact": 4, "family": 3, "backend": 2, "analogy-only": 1, "unknown": 0}
EVIDENCE_RANK = {
    "local-verifier": 5,
    "official-doc-and-upstream-code": 4,
    "source-reported": 3,
    "inferred": 2,
    "experimental": 1,
}
REPRODUCTION_RANK = {"benchmarked": 5, "runnable": 4, "snippet": 3, "pseudocode": 2, "concept": 1}
ROLE_PAGE_ITEM_MAX_BYTES = 16384
ROLE_PAGE_BODY_MAX_BYTES = 65536


@dataclass(frozen=True)
class RoleQueryRequest:
    text: str
    filters: Mapping[str, tuple[str, ...]]
    scope: str
    group_limits: Mapping[str, int]
    show_excluded: bool


@dataclass(frozen=True)
class RoleSearchResult:
    schema_version: int
    context_sha256: str
    loop_contract_identity: LoopContractIdentity | None
    authority_hashes: Mapping[str, str]
    groups: Mapping[str, tuple[Mapping[str, Any], ...]]


@dataclass(frozen=True)
class CapabilityGapRecord:
    gap_id: str
    reason: str
    capability_id: str | None
    capability_status: str
    target_id: str
    implementation_profile_id: str | None
    card_id: str


@dataclass(frozen=True)
class _RankedCandidate:
    candidate: SearchCandidate
    decision: AdmissionDecision
    hit: SearchHit
    numeric: tuple[int, ...]


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, LoopContractIdentity):
        return {
            "repository_commit": value.repository_commit,
            "skill_tree_sha": value.skill_tree_sha,
            "validator_sha256": _plain(value.validator_sha256),
            "schema_sha256": _plain(value.schema_sha256),
        }
    return value


def _decision_payload(decision: AdmissionDecision) -> dict[str, Any]:
    decision = require_validated_admission_decision(decision)
    return {
        "status": decision.status,
        "reasons": list(decision.reasons),
        "match_class": decision.match_class,
        "admitted_guidance_ids": list(decision.admitted_guidance_ids),
        "admitted_example_ids": list(decision.admitted_example_ids),
        "admitted_asset_ids": list(decision.admitted_asset_ids),
    }


def parse_role_query_request(
    text: str,
    filters: Mapping[str, Sequence[str]] | None = None,
    scope: str = "both",
    group_limits: Mapping[str, int] | None = None,
    show_excluded: bool = False,
) -> RoleQueryRequest:
    neutral = parse_query_request(text, filters, scope, 1)
    if group_limits is None:
        group_limits = {}
    if not isinstance(group_limits, Mapping):
        raise KernelWikiError("role-query-invalid", "group_limits must be a mapping")
    if any(not isinstance(name, str) for name in group_limits):
        raise KernelWikiError("role-query-invalid", "group limit names must be strings")
    unknown = sorted(set(group_limits) - set(ROLE_GROUPS))
    if unknown:
        raise KernelWikiError("role-query-invalid", f"unknown result groups: {', '.join(unknown)}")
    normalized_limits = dict(DEFAULT_GROUP_LIMITS)
    for name, value in group_limits.items():
        if type(value) is not int or value <= 0:
            raise KernelWikiError("role-query-invalid", f"group limit {name} must be a positive integer")
        normalized_limits[name] = value
    if not isinstance(show_excluded, bool):
        raise KernelWikiError("role-query-invalid", "show_excluded must be boolean")
    return RoleQueryRequest(
        text=neutral.text,
        filters=dict(neutral.filters),
        scope=neutral.scope,
        group_limits={name: normalized_limits[name] for name in ROLE_GROUPS},
        show_excluded=show_excluded,
    )


def _require_request(request: Any) -> RoleQueryRequest:
    if type(request) is not RoleQueryRequest:
        raise KernelWikiError("role-query-invalid", "request must be a RoleQueryRequest")
    return parse_role_query_request(
        request.text,
        request.filters,
        request.scope,
        request.group_limits,
        request.show_excluded,
    )


def _candidate_root(candidate: SearchCandidate) -> Path:
    record = candidate.record
    path = getattr(record, "path", None)
    if not isinstance(path, Path):
        raise KernelWikiError("source-broken", "candidate record path is invalid")
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError, ValueError) as error:
        raise KernelWikiError("source-broken", "candidate record path cannot be resolved", path) from error
    for parent in resolved.parents:
        if (parent / "data" / "taxonomy.yaml").is_file() and (parent / "compiled" / "catalog.jsonl").is_file():
            return parent
    raise KernelWikiError("source-broken", "candidate is not inside a validated KernelWiki root", path)


def _validate_candidate_projection(candidate: SearchCandidate) -> None:
    record = candidate.record
    if isinstance(record, WikiCard):
        metadata = record.metadata
        expected_kind = "card"
        expected_id = record.card_id
        expected_type = metadata.get("type")
        projected_fields = {
            "type": (str(metadata.get("type")),),
            "tags": tuple(metadata.get("tags", ())),
            "language": tuple(metadata.get("languages", ())),
            "target": tuple(metadata.get("targets", ())),
            "target-match": (str(metadata.get("target_match")),),
            "symptom": tuple(metadata.get("symptoms", ())),
            "kernel-type": tuple(metadata.get("kernel_types", ())),
            "audience": tuple(metadata.get("audiences", ())),
            "techniques": tuple(metadata.get("techniques", ())),
            "hardware-features": tuple(metadata.get("hardware_features", ())),
            "candidate-techniques": tuple(metadata.get("candidate_techniques", ())),
        }
    elif isinstance(record, SourceRecord):
        metadata = record.metadata
        expected_kind = "source"
        expected_id = record.source_id
        expected_type = metadata.get("source_kind")
        projected_fields = {
            "type": (str(metadata.get("source_kind")),),
            "tags": tuple(metadata.get("tags", ())),
            "repository": (str(metadata.get("repository_id")),),
            "language": tuple(metadata.get("languages", ())),
            "target": tuple(metadata.get("target_ids", ())),
            "target-match": (str(metadata.get("target_disposition")),),
            "symptom": (),
            "kernel-type": tuple(metadata.get("kernel_types", ())),
            "audience": tuple(metadata.get("audiences", ())),
            "techniques": tuple(metadata.get("techniques", ())),
            "hardware-features": tuple(metadata.get("hardware_features", ())),
            "candidate-techniques": (),
        }
    else:
        raise KernelWikiError("role-query-invalid", "rank candidate record type is invalid")
    if (
        candidate.record_kind != expected_kind
        or candidate.record_id != expected_id
        or candidate.title != metadata.get("title")
        or candidate.record_type != expected_type
        or candidate.body != record.body
    ):
        raise KernelWikiError("source-broken", "rank candidate differs from its canonical record projection")
    for field, expected in projected_fields.items():
        if tuple(candidate.structured_fields.get(field, ())) != expected:
            raise KernelWikiError("source-broken", f"rank candidate field {field} differs from its canonical record")


def _validate_rank_inputs(candidates: Sequence[tuple[SearchCandidate, AdmissionDecision]]) -> None:
    for item in candidates:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise KernelWikiError("role-query-invalid", "rank inputs must be candidate/decision pairs")
        candidate, decision = item
        if type(candidate) is not SearchCandidate or type(decision) is not AdmissionDecision:
            raise KernelWikiError("role-query-invalid", "rank inputs must contain SearchCandidate/AdmissionDecision pairs")
        _validate_candidate_projection(candidate)
        require_validated_admission_decision(decision, candidate=candidate)


def _date_ordinal(value: Any) -> int:
    if not isinstance(value, str):
        return 0
    try:
        return date.fromisoformat(value[:10]).toordinal() if len(value) >= 10 else 0
    except ValueError:
        return 0


def _backend_token(target_id: str) -> str:
    folded = target_id.casefold()
    for backend in ("ascend", "mlu", "cuda", "rocm", "cpu"):
        if folded == backend or folded.startswith(backend):
            return backend
    return folded


def _target_matches(item: Mapping[str, Any], context: RoleQueryContext) -> bool:
    target = item.get("target_id")
    if target is None:
        return True
    if not isinstance(target, str):
        return False
    if target == context.target_id:
        return True
    return context.role == "designer" and _backend_token(target) == _backend_token(context.target_id)


def _profile_runtime_matches(item: Mapping[str, Any], context: RoleQueryContext) -> bool:
    profile = item.get("implementation_profile_id")
    runtime = item.get("runtime_fingerprint")
    if profile is not None and context.implementation_profile_id is not None and profile != context.implementation_profile_id:
        return False
    if profile is not None and context.implementation_profile_id is None and context.role == "coder":
        return False
    if runtime is not None and context.runtime_fingerprint is not None and runtime != context.runtime_fingerprint:
        return False
    if runtime is not None and context.runtime_fingerprint is None and context.role == "coder":
        return False
    return True


def _shape_regime_score(shape: Any, context: RoleQueryContext, *, constraints: bool) -> int:
    if not isinstance(shape, Mapping) or not shape:
        return 0
    shared = sorted(set(shape).intersection(context.shape_signature))
    if not shared:
        return 0
    score = 2
    for dimension in shared:
        actual = context.shape_signature[dimension]
        expected = shape[dimension]
        if constraints:
            if not isinstance(expected, Mapping):
                return -1
            if set(expected) == {"exact"}:
                if actual != expected["exact"]:
                    return -1
            elif set(expected) == {"min", "max"}:
                minimum, maximum = expected["min"], expected["max"]
                if any(type(item) is not int for item in (actual, minimum, maximum)) or not minimum <= actual <= maximum:
                    return -1
                score = min(score, 1)
            else:
                return -1
        elif actual != expected:
            return -1
    return score


def _guidance_matches(item: Mapping[str, Any], context: RoleQueryContext) -> bool:
    targets = item.get("target_ids", ())
    profiles = item.get("implementation_profile_ids", ())
    runtimes = item.get("runtime_fingerprints", ())
    if not isinstance(targets, (list, tuple)) or context.target_id not in targets:
        return False
    if profiles:
        if context.implementation_profile_id is None or context.implementation_profile_id not in profiles:
            return False
    if runtimes:
        if context.runtime_fingerprint is None or context.runtime_fingerprint not in runtimes:
            return False
    return True


def _card_items(
    card: WikiCard,
    decision: AdmissionDecision,
    context: RoleQueryContext,
    item_role: str | None = None,
) -> tuple[Mapping[str, Any], ...]:
    metadata = card.metadata
    examples = metadata.get("examples", ())
    observations = metadata.get("observations", ())
    items: list[Mapping[str, Any]] = []
    admitted_examples = set(decision.admitted_example_ids)
    for item in examples if isinstance(examples, list) else ():
        if not isinstance(item, Mapping):
            continue
        if context.role == "coder" and item.get("id") not in admitted_examples:
            continue
        if item_role is not None and item.get("role") != item_role:
            continue
        if _target_matches(item, context) and _profile_runtime_matches(item, context):
            items.append(item)
    if item_role is None:
        for item in observations if isinstance(observations, list) else ():
            if isinstance(item, Mapping) and _target_matches(item, context) and _profile_runtime_matches(item, context):
                items.append(item)
        access = metadata.get("coder_access")
        guidance_items = access.get("guidance", ()) if isinstance(access, Mapping) else ()
        admitted_guidance = set(decision.admitted_guidance_ids)
        for item in guidance_items if isinstance(guidance_items, list) else ():
            if not isinstance(item, Mapping):
                continue
            if context.role == "coder" and item.get("id") not in admitted_guidance:
                continue
            if _guidance_matches(item, context):
                items.append(item)
    return tuple(items)


def _candidate_items(
    candidate: SearchCandidate,
    decision: AdmissionDecision,
    context: RoleQueryContext,
    item_role: str | None = None,
) -> tuple[Mapping[str, Any], ...]:
    if isinstance(candidate.record, WikiCard):
        return _card_items(candidate.record, decision, context, item_role)
    if isinstance(candidate.record, SourceRecord):
        return () if item_role is not None else (candidate.record.metadata,)
    return ()


def _profile_runtime_exactness(items: Sequence[Mapping[str, Any]], context: RoleQueryContext) -> int:
    profiles: set[str] = set()
    runtimes: set[str] = set()
    for item in items:
        profile = item.get("implementation_profile_id")
        if isinstance(profile, str):
            profiles.add(profile)
        profiles.update(value for value in item.get("implementation_profile_ids", ()) if isinstance(value, str))
        runtime = item.get("runtime_fingerprint")
        if isinstance(runtime, str):
            runtimes.add(runtime)
        runtimes.update(value for value in item.get("runtime_fingerprints", ()) if isinstance(value, str))
    profile_exact = context.implementation_profile_id is not None and context.implementation_profile_id in profiles
    runtime_exact = context.runtime_fingerprint is not None and context.runtime_fingerprint in runtimes
    return int(profile_exact) + int(runtime_exact)


def _dtype_overlap(items: Sequence[Mapping[str, Any]], context: RoleQueryContext) -> int:
    values: set[str] = set()
    for item in items:
        dtype = item.get("dtype")
        if isinstance(dtype, str):
            values.add(dtype)
        values.update(value for value in item.get("dtypes", ()) if isinstance(value, str))
    return len(values.intersection(context.dtypes))


def _shape_score(items: Sequence[Mapping[str, Any]], context: RoleQueryContext) -> int:
    scores: list[int] = []
    for item in items:
        if "shape_constraints" in item:
            scores.append(_shape_regime_score(item.get("shape_constraints"), context, constraints=True))
        elif "shape" in item:
            scores.append(_shape_regime_score(item.get("shape"), context, constraints=False))
    return max(scores, default=0)


def _exact_rank_items(
    items: Sequence[Mapping[str, Any]],
    context: RoleQueryContext,
) -> tuple[Mapping[str, Any], ...]:
    exact: list[Mapping[str, Any]] = []
    for item in items:
        target_id = item.get("target_id")
        if target_id is not None and target_id != context.target_id:
            continue
        target_ids = item.get("target_ids")
        if isinstance(target_ids, (list, tuple)) and target_ids and context.target_id not in target_ids:
            continue
        exact.append(item)
    return tuple(exact)


def _evidence_reproduction_freshness(
    candidate: SearchCandidate,
    items: Sequence[Mapping[str, Any]],
) -> tuple[int, int, int]:
    evidence = max((EVIDENCE_RANK.get(str(item.get("evidence_level")), 0) for item in items), default=0)
    reproduction = max((REPRODUCTION_RANK.get(str(item.get("reproduction")), 0) for item in items), default=0)
    if isinstance(candidate.record, SourceRecord):
        freshness = _date_ordinal(candidate.record.metadata.get("captured_at"))
    else:
        freshness = max((_date_ordinal(item.get("last_verified_at")) for item in items), default=0)
    return evidence, reproduction, freshness


def _semantic_overlap(candidate: SearchCandidate, context: RoleQueryContext) -> int:
    values: set[str] = set()
    for field in ("tags", "techniques", "hardware-features", "candidate-techniques", "symptom"):
        values.update(candidate.structured_fields.get(field, ()))
    return len(values.intersection(context.semantic_features))


def _safe_coder_search_body(candidate: SearchCandidate, decision: AdmissionDecision) -> str:
    if not isinstance(candidate.record, WikiCard) or decision.status == "excluded":
        return ""
    metadata = candidate.record.metadata
    admitted_guidance = set(decision.admitted_guidance_ids)
    admitted_examples = set(decision.admitted_example_ids)
    access = metadata.get("coder_access")
    guidance = access.get("guidance", ()) if isinstance(access, Mapping) else ()
    safe_items = [
        item
        for item in guidance
        if isinstance(item, Mapping) and item.get("id") in admitted_guidance
    ]
    safe_items.extend(
        item
        for item in metadata.get("examples", ())
        if isinstance(item, Mapping) and item.get("id") in admitted_examples
    )
    return canonical_json_bytes(_plain(safe_items)).decode("utf-8") if safe_items else ""


def _neutral_lexical_score(hit: SearchHit) -> int:
    value = 0
    for component in hit.score:
        value = value * 1000 + min(component, 999)
    return value


def _ranked_candidate(
    candidate: SearchCandidate,
    decision: AdmissionDecision,
    hit: SearchHit,
    context: RoleQueryContext,
    item_role: str | None = None,
) -> _RankedCandidate:
    items = _candidate_items(candidate, decision, context, item_role)
    exact_items = _exact_rank_items(items, context)
    evidence, reproduction, freshness = _evidence_reproduction_freshness(candidate, exact_items)
    numeric = (
        TARGET_SPECIFICITY[decision.match_class],
        _profile_runtime_exactness(exact_items, context),
        len(set(candidate.structured_fields.get("kernel-type", ())).intersection(context.kernel_types)),
        _dtype_overlap(exact_items, context),
        _semantic_overlap(candidate, context) + _shape_score(exact_items, context),
        evidence,
        reproduction,
        freshness,
        _neutral_lexical_score(hit),
    )
    return _RankedCandidate(candidate, decision, hit, numeric)


def _rank_key(item: _RankedCandidate) -> tuple[Any, ...]:
    return tuple(-value for value in item.numeric) + (item.candidate.path, item.candidate.record_id)


def _neutral_request(request: RoleQueryRequest) -> QueryRequest:
    return parse_query_request(request.text, request.filters, request.scope, 1)


def _rank_pairs(
    candidates: Sequence[tuple[SearchCandidate, AdmissionDecision]],
    context: RoleQueryContext,
    request: RoleQueryRequest,
    corpus: Corpus,
    hits: Mapping[tuple[str, str], SearchHit] | None = None,
    *,
    include_body: bool = True,
    item_role: str | None = None,
) -> tuple[_RankedCandidate, ...]:
    ranked: list[_RankedCandidate] = []
    neutral = _neutral_request(request)
    for candidate, decision in candidates:
        key = (candidate.path, candidate.record_id)
        hit = hits.get(key) if hits is not None else None
        if hit is None:
            body_projection = (
                _safe_coder_search_body(candidate, decision)
                if context.role == "coder" and decision.status != "excluded"
                else None
            )
            hit = score_search_candidate(
                corpus,
                candidate,
                neutral,
                include_body=include_body,
                body_projection=body_projection,
            )
        if hit is not None:
            ranked.append(_ranked_candidate(candidate, decision, hit, context, item_role))
    ranked.sort(key=_rank_key)
    return tuple(ranked)


def _canonical_rank_pairs(
    pairs: Sequence[tuple[SearchCandidate, AdmissionDecision]],
) -> tuple[tuple[tuple[SearchCandidate, AdmissionDecision], ...], Corpus]:
    roots = {_candidate_root(candidate) for candidate, _ in pairs}
    if len(roots) != 1:
        raise KernelWikiError("role-query-invalid", "rank candidates must share one canonical corpus root")
    root = next(iter(roots))
    try:
        corpus = authoritative_search_corpus(load_corpus(root))
        neutral = parse_query_request("", {}, "both", 1)
        canonical = {
            (candidate.record_kind, candidate.record_id): candidate
            for candidate in collect_unlimited_candidates(corpus, neutral)
        }
    except KernelWikiError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise KernelWikiError("source-broken", "rank corpus authority cannot be loaded", root) from error
    rebound: list[tuple[SearchCandidate, AdmissionDecision]] = []
    for candidate, decision in pairs:
        expected = canonical.get((candidate.record_kind, candidate.record_id))
        if expected is None or candidate != expected:
            raise KernelWikiError("source-broken", "rank candidate differs from neutral corpus authority")
        require_validated_admission_decision(decision, candidate=expected)
        rebound.append((expected, decision))
    return tuple(rebound), corpus


def rank_role_candidates(
    candidates: Sequence[tuple[SearchCandidate, AdmissionDecision]],
    context: RoleQueryContext,
    request: RoleQueryRequest,
) -> tuple[SearchCandidate, ...]:
    context = require_validated_role_context(context)
    request = _require_request(request)
    if isinstance(candidates, (str, bytes)) or not isinstance(candidates, Sequence):
        raise KernelWikiError("role-query-invalid", "candidates must be a sequence")
    pairs = tuple(candidates)
    if not pairs:
        return ()
    _validate_rank_inputs(pairs)
    rebound, scoring_corpus = _canonical_rank_pairs(pairs)
    return tuple(item.candidate for item in _rank_pairs(rebound, context, request, scoring_corpus))


def _special_groups(
    candidate: SearchCandidate,
    decision: AdmissionDecision,
    context: RoleQueryContext,
) -> tuple[str, ...]:
    if not isinstance(candidate.record, WikiCard):
        return ()
    examples = candidate.record.metadata.get("examples", ())
    admitted_ids = set(decision.admitted_example_ids)
    roles = {
        item.get("role")
        for item in examples
        if isinstance(examples, list)
        and isinstance(item, Mapping)
        and (context.role == "designer" or item.get("id") in admitted_ids)
        and item.get("target_id") == context.target_id
        and _profile_runtime_matches(item, context)
    }
    groups: list[str] = []
    if "counterexample" in roles:
        groups.append("counterexamples")
    if "capability-gap" in roles:
        groups.append("capability_gaps")
    return tuple(groups)


def _group_pairs(
    decisions: Sequence[tuple[SearchCandidate, AdmissionDecision]],
    context: RoleQueryContext,
    *,
    show_excluded: bool,
) -> dict[str, list[tuple[SearchCandidate, AdmissionDecision]]]:
    groups = {name: [] for name in ROLE_GROUPS}
    for candidate, decision in decisions:
        primary = decision.status
        if primary == "excluded":
            if show_excluded:
                groups["excluded"].append((candidate, decision))
            continue
        if primary not in {"admitted", "conditional", "analogy_only"}:
            raise KernelWikiError("contract-unsupported", f"unsupported admission status {primary}")
        groups[primary].append((candidate, decision))
        if primary in {"admitted", "conditional"}:
            for name in _special_groups(candidate, decision, context):
                groups[name].append((candidate, decision))
    return groups


def _entry_version_claims(item: _RankedCandidate, corpus: Corpus) -> list[dict[str, Any]]:
    if not isinstance(item.candidate.record, WikiCard):
        return []
    claim_ids = set(item.candidate.record.metadata.get("version_sensitive", ()))
    admitted_guidance = set(item.decision.admitted_guidance_ids)
    access = item.candidate.record.metadata.get("coder_access")
    if isinstance(access, Mapping):
        for guidance in access.get("guidance", ()):
            if isinstance(guidance, Mapping) and guidance.get("id") in admitted_guidance:
                claim_ids.update(guidance.get("version_claim_ids", ()))
    claims = {str(claim.get("id")): claim for claim in corpus.version_claims if isinstance(claim, Mapping)}
    payload: list[dict[str, Any]] = []
    for claim_id in sorted(claim_ids):
        claim = claims.get(claim_id)
        if claim is None:
            continue
        status = str(claim["status"])
        payload.append({
            "id": claim_id,
            "subject": str(claim["subject"]),
            "status": status,
            "supported_versions": list(claim["supported_versions"]),
            "last_verified_at": claim["last_verified_at"],
            "source_ids": list(claim["source_ids"]),
            "replacement_claim_id": claim["replacement_claim_id"],
            "reason": None if status == "current" else "version-stale",
        })
    return payload


def _metadata_only_gap_admission(
    decision: AdmissionDecision, gaps: Sequence[CapabilityGapRecord]
) -> dict[str, Any]:
    require_validated_admission_decision(decision)
    return {
        "status": "conditional",
        "reasons": sorted({gap.reason for gap in gaps}),
        "match_class": decision.match_class,
        "admitted_guidance_ids": [],
        "admitted_example_ids": [],
        "admitted_asset_ids": [],
    }


def _entry(
    item: _RankedCandidate,
    corpus: Corpus,
    *,
    metadata_only: bool = False,
    gaps: Sequence[CapabilityGapRecord] = (),
    generated_gap: bool = False,
) -> dict[str, Any]:
    metadata_only = metadata_only or generated_gap
    matched_fields = tuple(field for field in item.hit.matched_fields if field != "body") if metadata_only else item.hit.matched_fields
    return {
        "id": item.candidate.record_id,
        "record_kind": item.candidate.record_kind,
        "path": item.candidate.path,
        "title": item.candidate.title,
        "type": item.candidate.record_type,
        "admission": (
            _metadata_only_gap_admission(item.decision, gaps)
            if generated_gap
            else _decision_payload(item.decision)
        ),
        "version_claims": _entry_version_claims(item, corpus),
        "capability_gaps": [asdict(gap) for gap in gaps],
        "rank": {
            "target_specificity": item.numeric[0],
            "profile_runtime_exactness": item.numeric[1],
            "kernel_type_overlap_count": item.numeric[2],
            "dtype_overlap_count": item.numeric[3],
            "semantic_shape_score": item.numeric[4],
            "evidence_rank": item.numeric[5],
            "reproduction_rank": item.numeric[6],
            "freshness_ordinal": item.numeric[7],
            "neutral_lexical_score": item.numeric[8],
        },
        "matched_fields": list(matched_fields),
        "excerpt": "" if metadata_only else item.hit.excerpt,
    }


def _bind_context_authority(
    context: RoleQueryContext,
    authority: AuthoritySnapshot | None,
) -> AuthoritySnapshot | None:
    if context.role == "designer":
        if authority is not None:
            raise KernelWikiError("authority-context-invalid", "Designer context cannot consume loop authority")
        return None
    if context.implementation_profile_status == "missing":
        if authority is not None:
            raise KernelWikiError("authority-context-invalid", "missing-profile Coder cannot consume fallback authority")
        return None
    if authority is None:
        raise KernelWikiError("authority-context-invalid", "non-missing Coder requires validated loop authority")
    validated = require_validated_authority_snapshot(authority)
    reasons = _authority_reasons(context, validated)
    if reasons:
        raise KernelWikiError(
            "authority-context-invalid",
            "authority does not match context: " + ", ".join(sorted(reasons)),
        )
    return validated


def _gap_triggers(
    decisions: Sequence[tuple[SearchCandidate, AdmissionDecision]],
    context: RoleQueryContext,
    authority: AuthoritySnapshot | None,
) -> tuple[tuple[str, str | None, str], ...]:
    if context.role != "coder":
        return ()
    if context.implementation_profile_status == "missing":
        return (("profile-missing", None, "unknown"),)
    if authority is None:
        return ()
    triggers: set[tuple[str, str | None, str]] = set()
    for candidate, _decision_value in decisions:
        if not isinstance(candidate.record, WikiCard):
            continue
        access = candidate.record.metadata.get("coder_access")
        if not isinstance(access, Mapping):
            continue
        for guidance in access.get("guidance", ()):
            if not isinstance(guidance, Mapping):
                continue
            for resolution in relevant_unknown_capabilities(
                candidate.record, guidance, context, authority
            ):
                triggers.add((resolution.reason or "capability-unknown", resolution.capability_id, resolution.status))
    return tuple(sorted(triggers, key=lambda item: (item[0], item[1] or "", item[2])))


def _is_matching_gap_candidate(candidate: SearchCandidate, context: RoleQueryContext) -> bool:
    if not isinstance(candidate.record, WikiCard):
        return False
    metadata = candidate.record.metadata
    if metadata.get("type") not in {"pattern", "language", "runtime"}:
        return False
    if "capability-gap" not in set(metadata.get("tags", ())) | set(metadata.get("symptoms", ())):
        return False
    languages = set(metadata.get("languages", ()))
    if context.languages and not languages.intersection(context.languages):
        return False
    disposition = metadata.get("target_match")
    if disposition not in {"exact", "family", "backend"}:
        return False
    return classify_designer_match(candidate.record, context) == disposition


def _gap_records(
    candidate: SearchCandidate,
    triggers: Sequence[tuple[str, str | None, str]],
    context: RoleQueryContext,
) -> tuple[CapabilityGapRecord, ...]:
    records: list[CapabilityGapRecord] = []
    for reason, capability_id, status in triggers:
        identity = {
            "reason": reason,
            "capability_id": capability_id,
            "capability_status": status,
            "target_id": context.target_id,
            "implementation_profile_id": context.implementation_profile_id,
            "card_id": candidate.record_id,
        }
        records.append(CapabilityGapRecord(
            gap_id="gap-" + sha256_bytes(canonical_json_bytes(identity))[:16],
            reason=reason,
            capability_id=capability_id,
            capability_status=status,
            target_id=context.target_id,
            implementation_profile_id=context.implementation_profile_id,
            card_id=candidate.record_id,
        ))
    return tuple(records)


def role_search(
    corpus: Corpus,
    request: RoleQueryRequest,
    context: RoleQueryContext,
    authority: AuthoritySnapshot | None,
) -> RoleSearchResult:
    context = require_validated_role_context(context)
    request = _require_request(request)
    validated_authority = _bind_context_authority(context, authority)
    neutral_request = _neutral_request(request)
    candidates = collect_unlimited_candidates(corpus, neutral_request)
    decisions = tuple(
        (candidate, admit_candidate(candidate, context, validated_authority))
        for candidate in candidates
    )
    grouped = _group_pairs(decisions, context, show_excluded=request.show_excluded)
    gap_triggers = _gap_triggers(decisions, context, validated_authority)
    generated_gap_ids: set[str] = set()
    if gap_triggers:
        existing_ids = {candidate.record_id for candidate, _ in grouped["capability_gaps"]}
        for candidate, decision in decisions:
            if _is_matching_gap_candidate(candidate, context) and candidate.record_id not in existing_ids:
                grouped["capability_gaps"].append((candidate, decision))
                existing_ids.add(candidate.record_id)
                generated_gap_ids.add(candidate.record_id)
    result_groups: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for name in ROLE_GROUPS:
        metadata_only = name == "excluded" or context.role == "coder"
        ranked = _rank_pairs(
            grouped[name],
            context,
            request,
            corpus,
            include_body=not metadata_only,
            item_role=(
                "counterexample"
                if name == "counterexamples"
                else "capability-gap"
                if name == "capability_gaps" and not gap_triggers
                else None
            ),
        )
        result_groups[name] = tuple(
            _entry(
                item,
                corpus,
                metadata_only=metadata_only,
                gaps=(
                    _gap_records(item.candidate, gap_triggers, context)
                    if name == "capability_gaps" and item.candidate.record_id in generated_gap_ids
                    else ()
                ),
                generated_gap=(name == "capability_gaps" and item.candidate.record_id in generated_gap_ids),
            )
            for item in ranked[: request.group_limits[name]]
        )
    return RoleSearchResult(
        schema_version=1,
        context_sha256=_context_fingerprint(context),
        loop_contract_identity=(
            validated_authority.loop_contract_identity
            if validated_authority is not None
            else context.loop_contract_identity
        ),
        authority_hashes=(
            dict(sorted(validated_authority.artifact_hashes.items()))
            if validated_authority is not None
            else {}
        ),
        groups=result_groups,
    )


def role_result_payload(result: RoleSearchResult) -> dict[str, Any]:
    if type(result) is not RoleSearchResult:
        raise KernelWikiError("role-query-invalid", "result must be a RoleSearchResult")
    return {
        "schema_version": result.schema_version,
        "context_sha256": result.context_sha256,
        "loop_contract_identity": _plain(result.loop_contract_identity),
        "authority_hashes": dict(sorted(result.authority_hashes.items())),
        "groups": {name: [_plain(item) for item in result.groups[name]] for name in ROLE_GROUPS},
    }


def _resolve_candidate(corpus: Corpus, selector: str) -> SearchCandidate:
    neutral = parse_query_request("", {}, "both", 1)
    matches = [
        candidate
        for candidate in collect_unlimited_candidates(corpus, neutral)
        if candidate.record_id == selector or candidate.path == selector
    ]
    if len(matches) != 1:
        raise KernelWikiError("page-not-found", f"no Card or Source matches {selector!r}", corpus.root)
    return matches[0]


def _item_ids(card: WikiCard) -> tuple[set[str], set[str]]:
    examples = {
        str(item["id"])
        for item in card.metadata.get("examples", ())
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    access = card.metadata.get("coder_access")
    guidance = access.get("guidance", ()) if isinstance(access, Mapping) else ()
    guidance_ids = {
        str(item["id"])
        for item in guidance
        if isinstance(item, Mapping) and isinstance(item.get("id"), str)
    }
    return examples, guidance_ids


def _requested_item_decision(
    kind: str,
    item_id: str,
    candidate: SearchCandidate,
    page_decision: AdmissionDecision,
    context: RoleQueryContext,
    authority: AuthoritySnapshot | None,
) -> AdmissionDecision:
    if not isinstance(candidate.record, WikiCard):
        return _decision("excluded", reasons=("source-broken",), match_class=page_decision.match_class)
    examples, guidance = _item_ids(candidate.record)
    if page_decision.status == "excluded":
        return _decision("excluded", reasons=page_decision.reasons, match_class=page_decision.match_class)
    if kind == "asset":
        return admit_asset(candidate.record, item_id, context, authority)
    existing = examples if kind == "example" else guidance
    if item_id not in existing:
        return _decision("excluded", reasons=("source-broken",), match_class=page_decision.match_class)
    if context.role == "designer":
        return _decision(
            "admitted",
            match_class=page_decision.match_class,
            example_ids=(item_id,) if kind == "example" else (),
        )
    allowed = (
        item_id in page_decision.admitted_example_ids
        if kind == "example"
        else item_id in page_decision.admitted_guidance_ids
    )
    if allowed:
        return _decision(
            "admitted",
            match_class=page_decision.match_class,
            guidance_ids=(item_id,) if kind == "guidance" else (),
            example_ids=(item_id,) if kind == "example" else (),
        )
    reasons = page_decision.reasons or (("sketch-change-required",) if kind == "guidance" else ("profile-version-mismatch",))
    return _decision("excluded", reasons=tuple(reasons), match_class=page_decision.match_class)


def _asset_sources(card: WikiCard) -> Mapping[str, str]:
    try:
        return {
            asset_id: record.source.source_id
            for asset_id, record in _asset_records(card).items()
        }
    except (KernelWikiError, OSError, RuntimeError, TypeError, ValueError):
        return {}


def _selected_metadata_item(card: WikiCard, kind: str, item_id: str) -> Mapping[str, Any]:
    if kind == "example":
        values = card.metadata.get("examples", ())
    else:
        access = card.metadata.get("coder_access")
        values = access.get("guidance", ()) if isinstance(access, Mapping) else ()
    for item in values if isinstance(values, (list, tuple)) else ():
        if isinstance(item, Mapping) and item.get("id") == item_id:
            return item
    raise KernelWikiError("source-broken", f"admitted {kind} {item_id} is absent from the canonical Card", card.path)


def _read_admitted_asset(candidate: SearchCandidate, asset_id: str) -> Mapping[str, Any]:
    if not isinstance(candidate.record, WikiCard):
        raise KernelWikiError("source-broken", "admitted asset selection requires a Card")
    card = candidate.record
    try:
        record = _asset_records(card).get(asset_id)
    except (KernelWikiError, OSError, RuntimeError, TypeError, ValueError) as error:
        if isinstance(error, KernelWikiError):
            raise
        raise KernelWikiError("source-broken", f"admitted asset {asset_id} could not be resolved", card.path) from error
    if record is None:
        raise KernelWikiError("source-broken", f"admitted asset {asset_id} is absent from provenance", card.path)
    root = _candidate_root(candidate)
    path = record.bundle.path.parent / record.file.local_path
    try:
        require_within(root, path)
        if path.is_symlink() or not path.is_file():
            raise KernelWikiError("source-broken", f"admitted asset {asset_id} is not a regular file", path)
        data = path.read_bytes()
    except KernelWikiError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise KernelWikiError("source-broken", f"admitted asset {asset_id} could not be read", path) from error
    if len(data) > ROLE_PAGE_ITEM_MAX_BYTES:
        raise KernelWikiError("page-item-too-large", f"admitted asset {asset_id} exceeds the role page item limit", path)
    if sha256_bytes(data) != record.file.sha256:
        raise KernelWikiError("source-broken", f"admitted asset {asset_id} differs from validated provenance", path)
    try:
        content = data.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        content = base64.b64encode(data).decode("ascii")
        encoding = "base64"
    return {
        "id": asset_id,
        "source_id": record.source.source_id,
        "sha256": record.file.sha256,
        "encoding": encoding,
        "content": content,
    }


def _render_selected_page_body(
    candidate: SearchCandidate,
    selected_items: Sequence[Mapping[str, Any]],
) -> str:
    if not selected_items:
        return ""
    if not isinstance(candidate.record, WikiCard):
        raise KernelWikiError("source-broken", "selected role items require a Card", getattr(candidate.record, "path", None))
    sections: list[str] = []
    for selected in selected_items:
        kind = selected["kind"]
        item_id = selected["id"]
        if kind == "asset":
            value = _read_admitted_asset(candidate, item_id)
        else:
            value = _plain(_selected_metadata_item(candidate.record, kind, item_id))
        encoded = canonical_json_bytes(value)
        if len(encoded) > ROLE_PAGE_ITEM_MAX_BYTES:
            raise KernelWikiError("page-item-too-large", f"admitted {kind} {item_id} exceeds the role page item limit")
        sections.append(f"## Admitted {kind} `{item_id}`\n\n```json\n{encoded.decode('utf-8').rstrip()}\n```")
    body = "\n\n".join(sections) + "\n"
    if len(body.encode("utf-8")) > ROLE_PAGE_BODY_MAX_BYTES:
        raise KernelWikiError("page-body-too-large", "admitted role page body exceeds the bounded rendering limit")
    return body


def _sanitize_page(
    corpus: Corpus,
    page: PageResult,
    candidate: SearchCandidate,
    page_decision: AdmissionDecision,
    context: RoleQueryContext,
    authority: AuthoritySnapshot | None,
    requested: Mapping[str, tuple[str, ...]],
    access: str,
) -> dict[str, Any]:
    payload = page_payload(page)
    visible_page = page_decision.status != "excluded"
    if not visible_page or context.role == "coder":
        payload["body"] = ""
    followed: list[dict[str, Any]] = []
    for source_payload in payload["followed_sources"]:
        source = candidate.record if isinstance(candidate.record, SourceRecord) else None
        if source is None and isinstance(candidate.record, WikiCard):
            source_id = source_payload["source_id"]
            if source_id in candidate.record.metadata.get("sources", ()):
                source = corpus.sources.get(source_id)
        decision = (
            admit_source(source, context, authority)
            if isinstance(source, SourceRecord)
            else _decision("excluded", reasons=("source-broken",), match_class="unknown")
        )
        item = dict(source_payload)
        item["admission"] = _decision_payload(decision)
        if page_decision.status == "excluded" or decision.status == "excluded" or context.role == "coder":
            item["body"] = ""
        followed.append(item)
    payload["followed_sources"] = followed

    selected_items: list[dict[str, Any]] = []
    denied_items: list[dict[str, Any]] = []
    for kind in ("guidance", "example", "asset"):
        for item_id in requested[kind]:
            decision = _requested_item_decision(kind, item_id, candidate, page_decision, context, authority)
            if kind == "asset" and access != "approved-assets" and decision.status == "admitted":
                decision = _decision(
                    "excluded",
                    reasons=("artifact-designer-only",),
                    match_class=decision.match_class,
                )
            item = {"kind": kind, "id": item_id, "admission": _decision_payload(decision)}
            if decision.status == "admitted":
                selected_items.append(item)
            else:
                denied_items.append(item)
    payload["selected_items"] = selected_items
    payload["denied_items"] = denied_items
    denied_ids = [item["id"] for item in denied_items]
    if denied_ids:
        payload["body"] = ""

    if isinstance(candidate.record, WikiCard):
        source_by_asset = _asset_sources(candidate.record)
    else:
        source_by_asset = {}
    selected_assets_by_source: dict[str, list[str]] = {}
    for item in selected_items:
        if item["kind"] != "asset":
            continue
        source_id = source_by_asset.get(item["id"])
        if source_id is not None:
            selected_assets_by_source.setdefault(source_id, []).append(item["id"])
    selected_asset_ids = {
        asset_id
        for values in selected_assets_by_source.values()
        for asset_id in values
    }
    if set(source_by_asset) - selected_asset_ids:
        payload["body"] = ""
    for followed_source in payload["followed_sources"]:
        source_assets = {
            asset_id
            for asset_id, source_id in source_by_asset.items()
            if source_id == followed_source["source_id"]
        }
        if source_assets - selected_asset_ids:
            followed_source["body"] = ""
    role_asset_access: list[dict[str, Any]] = []
    for raw_access in payload["asset_access"]:
        item = dict(raw_access)
        admitted_ids = sorted(selected_assets_by_source.get(item["source_id"], ()))
        item["admitted_asset_ids"] = admitted_ids
        item["code_visible"] = bool(admitted_ids)
        item["reason"] = "role-admitted" if admitted_ids else "item-admission-required"
        role_asset_access.append(item)
    payload["asset_access"] = role_asset_access

    metadata = dict(payload["metadata"])
    if context.role == "coder" and isinstance(candidate.record, WikiCard):
        admitted_example_ids = {item["id"] for item in selected_items if item["kind"] == "example"}
        admitted_guidance_ids = {item["id"] for item in selected_items if item["kind"] == "guidance"}
        metadata["examples"] = [
            item
            for item in metadata.get("examples", ())
            if isinstance(item, Mapping) and item.get("id") in admitted_example_ids
        ]
        access = metadata.get("coder_access")
        if isinstance(access, Mapping):
            access_copy = dict(access)
            access_copy["guidance"] = [
                item
                for item in access.get("guidance", ())
                if isinstance(item, Mapping) and item.get("id") in admitted_guidance_ids
            ]
            metadata["coder_access"] = access_copy
    payload["metadata"] = metadata
    if selected_items:
        payload["body"] = _render_selected_page_body(candidate, selected_items)
    return payload


def role_get_page(
    corpus: Corpus,
    record_id: str,
    context: RoleQueryContext,
    authority: AuthoritySnapshot | None,
    *,
    follow_sources: bool,
    access: str,
    example_ids: Sequence[str] = (),
    guidance_ids: Sequence[str] = (),
    asset_ids: Sequence[str] = (),
) -> Mapping[str, Any]:
    context = require_validated_role_context(context)
    validated_authority = _bind_context_authority(context, authority)
    if not isinstance(record_id, str) or not record_id:
        raise KernelWikiError("page-selector-invalid", "record_id must be a nonempty string")
    if not isinstance(follow_sources, bool):
        raise KernelWikiError("page-follow-invalid", "follow_sources must be boolean")
    if access not in {"metadata", "approved-assets"}:
        raise KernelWikiError("page-access-invalid", "access must be metadata or approved-assets")
    requested: dict[str, tuple[str, ...]] = {}
    for name, values in (("example", example_ids), ("guidance", guidance_ids), ("asset", asset_ids)):
        if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
            raise KernelWikiError("page-item-invalid", f"{name} IDs must be a sequence")
        normalized = tuple(values)
        if any(not isinstance(item, str) or not item.strip() for item in normalized) or normalized != tuple(sorted(set(normalized))):
            raise KernelWikiError("page-item-invalid", f"{name} IDs must be sorted unique nonempty strings")
        requested[name] = normalized
    candidate = _resolve_candidate(corpus, record_id)
    decision = admit_candidate(candidate, context, validated_authority)
    core_access = access if context.role == "designer" and decision.status != "excluded" else "metadata"
    page = retrieve_page(corpus, record_id, follow_sources=follow_sources, access=core_access)
    page_mapping = _sanitize_page(corpus, page, candidate, decision, context, validated_authority, requested, access)
    return {
        "schema_version": 1,
        "context_sha256": _context_fingerprint(context),
        "loop_contract_identity": _plain(
            validated_authority.loop_contract_identity
            if validated_authority is not None
            else context.loop_contract_identity
        ),
        "authority_hashes": (
            dict(sorted(validated_authority.artifact_hashes.items()))
            if validated_authority is not None
            else {}
        ),
        "admission": _decision_payload(decision),
        "page": page_mapping,
    }


def role_payload_bytes(result: RoleSearchResult | Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(role_result_payload(result) if isinstance(result, RoleSearchResult) else _plain(result))

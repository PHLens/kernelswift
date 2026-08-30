from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import heapq
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

from catalog import CATALOG_FIELDS, card_to_catalog_record
from corpus import Corpus, SourceRecord, WikiCard, load_corpus, validate_corpus
from kernelwiki_common import KernelWikiError, require_within
from provenance import CODE_ROLES, load_provenance, validate_provenance


TOKEN_RE = re.compile(r"[\w.+-]+", re.UNICODE)
QUERY_SCOPES = frozenset({"cards", "sources", "both"})
GREP_SCOPES = frozenset({"wiki", "sources", "both"})
ACCESS_VALUES = frozenset({"metadata", "approved-assets"})
FILTER_FIELDS = (
    "type",
    "tag",
    "repository",
    "language",
    "target",
    "target-match",
    "symptom",
    "kernel-type",
    "evidence-level",
    "reproduction",
    "audience",
    "has-code",
)
FILTER_TO_STRUCTURED = {
    "type": "type",
    "tag": "tags",
    "repository": "repository",
    "language": "language",
    "target": "target",
    "target-match": "target-match",
    "symptom": "symptom",
    "kernel-type": "kernel-type",
    "evidence-level": "evidence-level",
    "reproduction": "reproduction",
    "audience": "audience",
    "has-code": "has-code",
}
STRUCTURED_PRIORITY = (
    "tags",
    "type",
    "repository",
    "language",
    "target",
    "target-match",
    "symptom",
    "kernel-type",
    "evidence-level",
    "reproduction",
    "audience",
    "has-code",
    "techniques",
    "hardware-features",
    "candidate-techniques",
)


@dataclass(frozen=True)
class QueryRequest:
    text: str
    filters: Mapping[str, tuple[str, ...]]
    scope: str
    limit: int


@dataclass(frozen=True)
class SearchCandidate:
    record_kind: str
    record_id: str
    path: str
    title: str
    record_type: str
    structured_fields: Mapping[str, tuple[str, ...]]
    body: str
    record: WikiCard | SourceRecord


@dataclass(frozen=True)
class SearchHit:
    record_kind: str
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


def _require_corpus(value: Any) -> Corpus:
    if not isinstance(value, Corpus):
        raise KernelWikiError("search-corpus-invalid", "corpus must be a Corpus")
    return value


def _authoritative_corpus(value: Any) -> Corpus:
    corpus = _require_corpus(value)
    root = corpus.root
    if not isinstance(root, Path) or root.is_symlink() or not root.is_dir():
        raise KernelWikiError("search-corpus-invalid", "corpus root must be a real directory")
    _checked_authority_file(root, "data/aliases.yaml", code="search-authority-invalid")
    try:
        fresh = load_corpus(root)
        validate_corpus(fresh)
    except KernelWikiError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise KernelWikiError("search-corpus-invalid", "corpus authority could not be reloaded", root) from error
    comparable = (
        corpus.root == fresh.root
        and dict(corpus.cards) == dict(fresh.cards)
        and dict(corpus.sources) == dict(fresh.sources)
        and dict(corpus.taxonomy) == dict(fresh.taxonomy)
        and dict(corpus.aliases) == dict(fresh.aliases)
        and tuple(corpus.version_claims) == tuple(fresh.version_claims)
        and corpus.repository_ids == fresh.repository_ids
    )
    if not comparable:
        raise KernelWikiError("search-corpus-stale", "corpus differs from validated on-disk authority", root)
    return fresh


def authoritative_search_corpus(value: Any) -> Corpus:
    """Rebind to the complete neutral search authority, including aliases and catalog bytes."""
    corpus = _authoritative_corpus(value)
    _load_catalog_records(corpus)
    return corpus


def _require_request(value: Any) -> QueryRequest:
    if not isinstance(value, QueryRequest):
        raise KernelWikiError("query-request-invalid", "request must be a QueryRequest")
    return value


def _authoritative_card(value: Any, corpus: Corpus) -> WikiCard:
    if not isinstance(value, WikiCard) or not isinstance(value.metadata, Mapping):
        raise KernelWikiError("search-card-invalid", "card must be a WikiCard from the Corpus")
    card_id = value.metadata.get("id")
    if not isinstance(card_id, str) or not card_id:
        raise KernelWikiError("search-card-invalid", "card has no valid ID")
    authoritative = corpus.cards.get(card_id)
    if authoritative is None or authoritative != value:
        raise KernelWikiError("search-card-invalid", "card is not the authoritative Corpus record", value.path)
    return authoritative


def _authoritative_source(value: Any, corpus: Corpus) -> SourceRecord:
    if not isinstance(value, SourceRecord) or not isinstance(value.metadata, Mapping):
        raise KernelWikiError("search-source-invalid", "source must be a SourceRecord from the Corpus")
    source_id = value.metadata.get("id")
    if not isinstance(source_id, str) or not source_id:
        raise KernelWikiError("search-source-invalid", "source has no valid ID")
    authoritative = corpus.sources.get(source_id)
    if authoritative is None or authoritative != value:
        raise KernelWikiError("search-source-invalid", "source is not the authoritative Corpus record", value.path)
    return authoritative


def _validate_search_catalog_record(
    value: Any, card: WikiCard, corpus: Corpus
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise KernelWikiError("catalog-record-invalid", "catalog record must be a mapping", card.path)
    if set(value) != set(CATALOG_FIELDS):
        raise KernelWikiError("catalog-record-invalid", "catalog record fields do not match schema", card.path)
    expected = card_to_catalog_record(card, corpus)
    if dict(value) != expected:
        raise KernelWikiError("catalog-card-mismatch", "catalog record differs from authoritative Card", card.path)
    return value


def _nonempty_string(value: Any, code: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise KernelWikiError(code, f"{label} must be a nonempty string")
    return value


def _checked_authority_file(root: Path, relative_path: str, *, code: str) -> Path:
    pure = PurePosixPath(relative_path)
    current = root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise KernelWikiError(code, f"authority path contains a symlink: {relative_path}", current)
    if not current.is_file():
        raise KernelWikiError(code, f"authority file is missing: {relative_path}", current)
    try:
        require_within(root, current)
    except KernelWikiError as error:
        raise KernelWikiError(code, error.message, current) from error
    return current


def _checked_relative_path(root: Path, value: str, *, code: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise KernelWikiError(code, "path must be a normalized repository-relative POSIX path", root)
    pure = PurePosixPath(value)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise KernelWikiError(code, "path must be a normalized repository-relative POSIX path", root)
    if pure.as_posix() != value:
        raise KernelWikiError(code, "path must be a normalized repository-relative POSIX path", root)
    return require_within(root, root.joinpath(*pure.parts))


def _tokenize(value: str) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall(value.casefold()))


def _tuple_strings(value: Any, code: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise KernelWikiError(code, "filter values must be a sequence of strings")
    items = tuple(value)
    if not items or any(not isinstance(item, str) or not item.strip() for item in items):
        raise KernelWikiError(code, "filter values must be nonempty strings")
    return tuple(sorted(set(items)))


def parse_query_request(
    text: str,
    filters: Mapping[str, Sequence[str]] | None = None,
    scope: str = "both",
    limit: int = 20,
) -> QueryRequest:
    if not isinstance(text, str):
        raise KernelWikiError("query-text-invalid", "query text must be a string")
    if not isinstance(scope, str) or scope not in QUERY_SCOPES:
        raise KernelWikiError("query-scope-invalid", "scope must be cards, sources, or both")
    if type(limit) is not int or limit <= 0:
        raise KernelWikiError("query-limit-invalid", "limit must be a positive integer")
    if filters is None:
        filters = {}
    if not isinstance(filters, Mapping):
        raise KernelWikiError("query-filter-invalid", "filters must be a mapping")
    if any(not isinstance(key, str) for key in filters):
        raise KernelWikiError("query-filter-invalid", "filter keys must be strings")
    unknown = sorted(set(filters) - set(FILTER_FIELDS))
    if unknown:
        raise KernelWikiError("query-filter-invalid", f"unknown filters: {', '.join(unknown)}")
    normalized = {
        key: _tuple_strings(filters[key], "query-filter-invalid")
        for key in sorted(filters)
    }
    if "has-code" in normalized and not set(normalized["has-code"]) <= {"true", "false"}:
        raise KernelWikiError("query-filter-invalid", "has-code values must be true or false")
    return QueryRequest(text=text, filters=normalized, scope=scope, limit=limit)


def _load_catalog_records(corpus: Corpus) -> dict[str, Mapping[str, Any]]:
    path = _checked_authority_file(
        corpus.root, "compiled/catalog.jsonl", code="catalog-read-invalid"
    )
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise KernelWikiError("catalog-read-invalid", str(error), path) from error
    records: dict[str, Mapping[str, Any]] = {}
    for line_number, line in enumerate(lines, 1):
        if not line:
            raise KernelWikiError("catalog-read-invalid", f"blank line {line_number}", path)
        try:
            record = json.loads(line)
        except (json.JSONDecodeError, TypeError) as error:
            raise KernelWikiError("catalog-read-invalid", f"invalid JSON line {line_number}", path) from error
        if not isinstance(record, Mapping):
            raise KernelWikiError("catalog-read-invalid", f"line {line_number} must be an object", path)
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id or record_id in records:
            raise KernelWikiError("catalog-read-invalid", f"invalid or duplicate id on line {line_number}", path)
        records[record_id] = record
    if set(records) != set(corpus.cards):
        raise KernelWikiError("catalog-read-invalid", "catalog must contain every Card exactly once", path)
    return records


def _source_code_visible(source: SourceRecord, corpus: Corpus) -> bool:
    artifact_dir = source.metadata.get("artifact_dir")
    if artifact_dir is None or source.metadata.get("license_state") != "approved":
        return False
    bundle_dir = _checked_relative_path(corpus.root, str(artifact_dir), code="page-asset-path-invalid")
    manifest = bundle_dir / "PROVENANCE.yaml"
    if manifest.is_symlink() or not manifest.is_file():
        raise KernelWikiError("provenance-missing", "approved artifact requires PROVENANCE.yaml", source.path)
    bundle = load_provenance(manifest)
    validate_provenance(bundle, corpus.root)
    if source.source_id not in bundle.source_ids:
        raise KernelWikiError(
            "provenance-source-mismatch",
            f"Source {source.source_id} is absent from provenance source_ids",
            manifest,
        )
    return (
        bundle.license_state == "approved"
        and "designer" in bundle.allowed_audiences
        and bundle.asset_mode in {"verbatim", "extracted", "derived"}
        and any(item.role in CODE_ROLES for item in bundle.files)
    )


def _card_has_code(card: WikiCard, corpus: Corpus) -> bool:
    return any(_source_code_visible(corpus.sources[source_id], corpus) for source_id in card.metadata["sources"])


def build_card_candidate(
    card: WikiCard, catalog_record: Mapping[str, Any], corpus: Corpus
) -> SearchCandidate:
    corpus = _authoritative_corpus(corpus)
    card = _authoritative_card(card, corpus)
    catalog_record = _validate_search_catalog_record(catalog_record, card, corpus)
    if catalog_record.get("id") != card.card_id:
        raise KernelWikiError("catalog-card-mismatch", "catalog record does not match Card", card.path)
    expected_path = card.path.relative_to(corpus.root).as_posix()
    if catalog_record.get("path") != expected_path:
        raise KernelWikiError("catalog-card-mismatch", "catalog path does not match Card", card.path)
    structured = {
        "type": (str(catalog_record["type"]),),
        "tags": tuple(catalog_record["tags"]),
        "repository": tuple(catalog_record["source_repositories"]),
        "language": tuple(catalog_record["languages"]),
        "target": tuple(catalog_record["targets"]),
        "target-match": (str(catalog_record["target_match"]),),
        "symptom": tuple(catalog_record["symptoms"]),
        "kernel-type": tuple(catalog_record["kernel_types"]),
        "evidence-level": tuple(catalog_record["evidence_levels"]),
        "reproduction": tuple(catalog_record["reproduction_levels"]),
        "audience": tuple(catalog_record["audiences"]),
        "has-code": ("true" if _card_has_code(card, corpus) else "false",),
        "techniques": tuple(catalog_record["techniques"]),
        "hardware-features": tuple(catalog_record["hardware_features"]),
        "candidate-techniques": tuple(catalog_record["candidate_techniques"]),
    }
    return SearchCandidate(
        record_kind="card",
        record_id=card.card_id,
        path=expected_path,
        title=str(catalog_record["title"]),
        record_type=str(catalog_record["type"]),
        structured_fields=structured,
        body=card.body,
        record=card,
    )


def build_source_candidate(source: SourceRecord, corpus: Corpus) -> SearchCandidate:
    corpus = _authoritative_corpus(corpus)
    source = _authoritative_source(source, corpus)
    metadata = source.metadata
    structured = {
        "type": (str(metadata["source_kind"]),),
        "tags": tuple(metadata["tags"]),
        "repository": (str(metadata["repository_id"]),),
        "language": tuple(metadata["languages"]),
        "target": tuple(metadata.get("target_ids", ())),
        "target-match": (str(metadata["target_disposition"]),),
        "symptom": (),
        "kernel-type": tuple(metadata["kernel_types"]),
        "evidence-level": (),
        "reproduction": (),
        "audience": tuple(metadata.get("audiences", ())),
        "has-code": ("true" if _source_code_visible(source, corpus) else "false",),
        "techniques": tuple(metadata["techniques"]),
        "hardware-features": tuple(metadata["hardware_features"]),
        "candidate-techniques": (),
    }
    return SearchCandidate(
        record_kind="source",
        record_id=source.source_id,
        path=source.path.relative_to(corpus.root).as_posix(),
        title=str(metadata["title"]),
        record_type=str(metadata["source_kind"]),
        structured_fields=structured,
        body=source.body,
        record=source,
    )


def collect_unlimited_candidates(corpus: Corpus, request: QueryRequest) -> tuple[SearchCandidate, ...]:
    corpus = authoritative_search_corpus(corpus)
    request = _require_request(request)
    request = parse_query_request(request.text, request.filters, request.scope, request.limit)
    candidates: list[SearchCandidate] = []
    if request.scope in {"cards", "both"}:
        records = _load_catalog_records(corpus)
        candidates.extend(build_card_candidate(corpus.cards[card_id], records[card_id], corpus) for card_id in sorted(corpus.cards))
    if request.scope in {"sources", "both"}:
        candidates.extend(build_source_candidate(corpus.sources[source_id], corpus) for source_id in sorted(corpus.sources))
    return tuple(sorted(candidates, key=lambda item: (item.path, item.record_id)))


def _alias_variants(corpus: Corpus, query_tokens: Sequence[str]) -> dict[str, frozenset[str]]:
    alias_to_canonical: dict[str, str] = {}
    canonical_to_aliases: dict[str, set[str]] = {}
    for canonical, aliases in corpus.aliases.items():
        canonical_key = canonical.casefold()
        canonical_to_aliases.setdefault(canonical_key, set()).add(canonical_key)
        alias_to_canonical[canonical_key] = canonical_key
        for alias in aliases:
            alias_key = alias.casefold()
            alias_to_canonical[alias_key] = canonical_key
            canonical_to_aliases[canonical_key].add(alias_key)
    variants: dict[str, frozenset[str]] = {}
    for token in query_tokens:
        canonical = alias_to_canonical.get(token)
        variants[token] = (
            frozenset({token})
            if canonical is None
            else frozenset(canonical_to_aliases[canonical])
        )
    return variants


def _value_tokens(values: Sequence[str]) -> frozenset[str]:
    tokens: set[str] = set()
    for value in values:
        folded = value.casefold()
        tokens.add(folded)
        tokens.update(_tokenize(folded))
    return frozenset(tokens)


def _excerpt(body: str, variants: Mapping[str, frozenset[str]], *, max_chars: int = 240) -> str:
    single = " ".join(body.split())
    if not single:
        return ""
    positions = [
        single.casefold().find(variant)
        for choices in variants.values()
        for variant in choices
        if single.casefold().find(variant) >= 0
    ]
    start = max(0, (min(positions) if positions else 0) - max_chars // 3)
    excerpt = single[start : start + max_chars]
    if start:
        excerpt = "…" + excerpt[1:]
    if start + max_chars < len(single):
        excerpt = excerpt[:-1] + "…"
    return excerpt


def _passes_filters(candidate: SearchCandidate, filters: Mapping[str, tuple[str, ...]]) -> bool:
    for filter_name, requested in filters.items():
        field = FILTER_TO_STRUCTURED[filter_name]
        actual = set(candidate.structured_fields[field])
        if not actual.intersection(requested):
            return False
    return True


def _score_candidate(candidate: SearchCandidate, corpus: Corpus, text: str) -> SearchHit | None:
    query_tokens = _tokenize(text)
    variants = _alias_variants(corpus, query_tokens)
    title_tokens = _value_tokens((candidate.title,))
    title_matched = {token for token in query_tokens if token in title_tokens}

    transparent_concepts: set[str] = set()
    structured_field_count = 0
    matched_fields: list[str] = ["title"] if title_matched else []
    all_candidate_tokens = set(title_tokens)
    for field in STRUCTURED_PRIORITY:
        values = candidate.structured_fields.get(field, ())
        field_tokens = _value_tokens(values)
        all_candidate_tokens.update(field_tokens)
        field_concepts = {
            token for token in query_tokens if variants[token].intersection(field_tokens)
        }
        if field_concepts:
            structured_field_count += 1
            if field_concepts - transparent_concepts:
                matched_fields.append(field)
                transparent_concepts.update(field_concepts)

    body_tokens = _tokenize(candidate.body)
    body_count = sum(
        sum(1 for body_token in body_tokens if body_token in choices)
        for choices in variants.values()
    )
    if body_count and not matched_fields:
        matched_fields.append("body")

    direct_tokens = all_candidate_tokens.union(body_tokens)
    alias_count = sum(
        1
        for token, choices in variants.items()
        if token not in direct_tokens and (choices - {token}).intersection(direct_tokens)
    )
    score = (len(title_matched), structured_field_count, alias_count, min(body_count, 8))
    if query_tokens and not any(score):
        return None
    return SearchHit(
        record_kind=candidate.record_kind,
        record_id=candidate.record_id,
        path=candidate.path,
        title=candidate.title,
        record_type=candidate.record_type,
        score=score,
        matched_fields=tuple(matched_fields),
        excerpt=_excerpt(candidate.body, variants),
    )


def score_search_candidate(
    corpus: Corpus,
    candidate: SearchCandidate,
    request: QueryRequest,
    *,
    include_body: bool = True,
    body_projection: str | None = None,
) -> SearchHit | None:
    """Score one canonical neutral candidate without applying request.limit."""
    corpus = authoritative_search_corpus(corpus)
    request = _require_request(request)
    request = parse_query_request(request.text, request.filters, request.scope, request.limit)
    if type(candidate) is not SearchCandidate:
        raise KernelWikiError("search-candidate-invalid", "candidate must be a SearchCandidate")
    if not isinstance(include_body, bool):
        raise KernelWikiError("search-candidate-invalid", "include_body must be boolean")
    if body_projection is not None and not isinstance(body_projection, str):
        raise KernelWikiError("search-candidate-invalid", "body_projection must be null or a string")
    if candidate.record_kind == "card" and isinstance(candidate.record, WikiCard):
        records = _load_catalog_records(corpus)
        record = _authoritative_card(candidate.record, corpus)
        expected = build_card_candidate(record, records[record.card_id], corpus)
    elif candidate.record_kind == "source" and isinstance(candidate.record, SourceRecord):
        record = _authoritative_source(candidate.record, corpus)
        expected = build_source_candidate(record, corpus)
    else:
        raise KernelWikiError("search-candidate-invalid", "candidate record kind/type is invalid")
    if candidate != expected:
        raise KernelWikiError("search-candidate-invalid", "candidate differs from neutral corpus authority")
    if not _passes_filters(candidate, request.filters):
        return None
    scored = replace(candidate, body=body_projection) if body_projection is not None else candidate if include_body else replace(candidate, body="")
    return _score_candidate(scored, corpus, request.text)


def search_records(corpus: Corpus, request: QueryRequest) -> tuple[SearchHit, ...]:
    corpus = authoritative_search_corpus(corpus)
    request = _require_request(request)
    request = parse_query_request(request.text, request.filters, request.scope, request.limit)
    hits: list[SearchHit] = []
    for candidate in collect_unlimited_candidates(corpus, request):
        if not _passes_filters(candidate, request.filters):
            continue
        hit = _score_candidate(candidate, corpus, request.text)
        if hit is not None:
            hits.append(hit)
    hits.sort(key=lambda item: tuple(-part for part in item.score) + (item.path, item.record_id))
    return tuple(hits[: request.limit])


def _hit_payload(hit: SearchHit) -> dict[str, Any]:
    return {
        "record_kind": hit.record_kind,
        "record_id": hit.record_id,
        "path": hit.path,
        "title": hit.title,
        "record_type": hit.record_type,
        "score": list(hit.score),
        "matched_fields": list(hit.matched_fields),
        "excerpt": hit.excerpt,
    }


def query_payload(corpus: Corpus, request: QueryRequest) -> Mapping[str, Any]:
    corpus = _authoritative_corpus(corpus)
    request = _require_request(request)
    request = parse_query_request(request.text, request.filters, request.scope, request.limit)
    return {
        "schema_version": 1,
        "query": request.text,
        "filters": {key: list(values) for key, values in request.filters.items()},
        "scope": request.scope,
        "results": [_hit_payload(hit) for hit in search_records(corpus, request)],
    }


def _exact_record_path(root: Path, selector: str) -> Path:
    if not isinstance(selector, str) or not selector or "\\" in selector or "\x00" in selector:
        raise KernelWikiError("page-selector-invalid", "record path must be normalized relative POSIX", root)
    pure = PurePosixPath(selector)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise KernelWikiError("page-selector-invalid", "record path must be normalized relative POSIX", root)
    if pure.as_posix() != selector:
        raise KernelWikiError("page-selector-invalid", "record path must be normalized relative POSIX", root)
    current = root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise KernelWikiError("page-selector-invalid", "record path must not contain symlinks", current)
    return current


def _resolve_record(corpus: Corpus, selector: str) -> tuple[str, WikiCard | SourceRecord]:
    _nonempty_string(selector, "page-selector-invalid", "record selector")
    if selector in corpus.cards:
        return "card", corpus.cards[selector]
    if selector in corpus.sources:
        return "source", corpus.sources[selector]
    path = _exact_record_path(corpus.root, selector)
    matches = [
        ("card", card) for card in corpus.cards.values() if card.path == path
    ] + [
        ("source", source) for source in corpus.sources.values() if source.path == path
    ]
    if len(matches) != 1:
        raise KernelWikiError("page-not-found", f"no Card or Source matches {selector!r}", corpus.root)
    return matches[0]


def _followed_source(source: SourceRecord, corpus: Corpus) -> FollowedSource:
    return FollowedSource(
        source_id=source.source_id,
        path=source.path.relative_to(corpus.root).as_posix(),
        title=str(source.metadata["title"]),
        metadata=dict(source.metadata),
        body=source.body,
    )


def _asset_access(source: SourceRecord, corpus: Corpus, access: str) -> AssetAccess:
    artifact = source.metadata.get("artifact_dir")
    artifact_dir = None if artifact is None else str(artifact)
    if access == "metadata":
        return AssetAccess(source.source_id, artifact_dir, True, False, "metadata-request")
    if source.metadata.get("license_state") != "approved":
        return AssetAccess(source.source_id, artifact_dir, True, False, "license-not-approved")
    if artifact_dir is None:
        return AssetAccess(source.source_id, None, True, False, "asset-missing")
    bundle_dir = _checked_relative_path(corpus.root, artifact_dir, code="page-asset-path-invalid")
    manifest = bundle_dir / "PROVENANCE.yaml"
    bundle = load_provenance(manifest)
    validate_provenance(bundle, corpus.root)
    if source.source_id not in bundle.source_ids:
        raise KernelWikiError(
            "provenance-source-mismatch",
            f"Source {source.source_id} is absent from provenance source_ids",
            manifest,
        )
    if "designer" not in bundle.allowed_audiences:
        return AssetAccess(source.source_id, artifact_dir, True, False, "audience-denied")
    if bundle.asset_mode not in {"verbatim", "extracted", "derived"}:
        return AssetAccess(source.source_id, artifact_dir, True, False, "asset-mode-denied")
    visible = any(item.role in CODE_ROLES for item in bundle.files)
    return AssetAccess(
        source.source_id,
        artifact_dir,
        True,
        visible,
        "approved" if visible else "no-code-asset",
    )


def retrieve_page(
    corpus: Corpus,
    record_id: str,
    *,
    follow_sources: bool,
    access: str,
) -> PageResult:
    corpus = _authoritative_corpus(corpus)
    if not isinstance(follow_sources, bool):
        raise KernelWikiError("page-follow-invalid", "follow_sources must be boolean")
    if not isinstance(access, str) or access not in ACCESS_VALUES:
        raise KernelWikiError("page-access-invalid", "access must be metadata or approved-assets")
    kind, record = _resolve_record(corpus, record_id)
    if kind == "card":
        assert isinstance(record, WikiCard)
        source_ids = tuple(record.metadata["sources"])
        title = str(record.metadata["title"])
        record_type = "card"
        stable_id = record.card_id
    else:
        assert isinstance(record, SourceRecord)
        source_ids = (record.source_id,)
        title = str(record.metadata["title"])
        record_type = "source"
        stable_id = record.source_id
    followed = tuple(_followed_source(corpus.sources[source_id], corpus) for source_id in source_ids) if follow_sources else ()
    assets = tuple(_asset_access(corpus.sources[source_id], corpus, access) for source_id in source_ids)
    return PageResult(
        schema_version=1,
        record_kind=record_type,
        record_id=stable_id,
        path=record.path.relative_to(corpus.root).as_posix(),
        title=title,
        metadata=dict(record.metadata),
        body=record.body,
        followed_sources=followed,
        asset_access=assets,
    )


def _markdown_body_start_line(path: Path, expected_body: str, root: Path) -> int:
    try:
        if path.is_symlink():
            raise KernelWikiError("search-corpus-stale", "record path must not be a symlink", path)
        require_within(root, path)
        text = path.read_text(encoding="utf-8")
    except KernelWikiError:
        raise
    except (OSError, RuntimeError, UnicodeError, ValueError) as error:
        raise KernelWikiError("search-corpus-invalid", "record Markdown could not be read", path) from error
    if not text.startswith("---\n"):
        raise KernelWikiError("search-corpus-stale", "record Markdown lost frontmatter", path)
    marker = text.find("\n---\n", 4)
    if marker < 0:
        raise KernelWikiError("search-corpus-stale", "record Markdown frontmatter is unclosed", path)
    body_index = marker + 5
    if text[body_index:] != expected_body:
        raise KernelWikiError("search-corpus-stale", "record body differs from validated corpus", path)
    return text.count("\n", 0, body_index) + 1


def grep_corpus(
    corpus: Corpus,
    pattern: str,
    *,
    scope: str,
    max_matches: int,
    context_chars: int,
) -> tuple[GrepMatch, ...]:
    corpus = _authoritative_corpus(corpus)
    if not isinstance(pattern, str):
        raise KernelWikiError("regex-invalid", "pattern must be a string")
    if not isinstance(scope, str) or scope not in GREP_SCOPES:
        raise KernelWikiError("grep-scope-invalid", "scope must be wiki, sources, or both")
    if type(max_matches) is not int or max_matches <= 0:
        raise KernelWikiError("grep-limit-invalid", "max_matches must be a positive integer")
    if type(context_chars) is not int or context_chars < 0:
        raise KernelWikiError("grep-context-invalid", "context_chars must be a nonnegative integer")
    try:
        compiled = re.compile(pattern)
    except re.error as error:
        raise KernelWikiError("regex-invalid", str(error)) from error
    records: list[tuple[str, str, str, str, int]] = []
    if scope in {"wiki", "both"}:
        records.extend(
            (
                "card",
                card.card_id,
                card.path.relative_to(corpus.root).as_posix(),
                card.body,
                _markdown_body_start_line(card.path, card.body, corpus.root),
            )
            for card in corpus.cards.values()
        )
    if scope in {"sources", "both"}:
        records.extend(
            (
                "source",
                source.source_id,
                source.path.relative_to(corpus.root).as_posix(),
                source.body,
                _markdown_body_start_line(source.path, source.body, corpus.root),
            )
            for source in corpus.sources.values()
        )
    def raw_matches():
        for kind, stable_id, path, body, body_start_line in sorted(records, key=lambda item: (item[2], item[1])):
            for match in compiled.finditer(body):
                line_number = body_start_line + body.count("\n", 0, match.start())
                start = max(0, match.start() - context_chars)
                end = min(len(body), match.end() + context_chars)
                excerpt = body[start:end].replace("\n", " ")
                yield (path, line_number, stable_id, excerpt, kind)

    selected = heapq.nsmallest(max_matches, raw_matches(), key=lambda item: item[:4])
    return tuple(
        GrepMatch(kind, stable_id, path, line_number, excerpt)
        for path, line_number, stable_id, excerpt, kind in selected
    )


def page_payload(page: PageResult) -> dict[str, Any]:
    return {
        "schema_version": page.schema_version,
        "record_kind": page.record_kind,
        "record_id": page.record_id,
        "path": page.path,
        "title": page.title,
        "metadata": dict(page.metadata),
        "body": page.body,
        "followed_sources": [asdict(item) for item in page.followed_sources],
        "asset_access": [asdict(item) for item in page.asset_access],
    }


def grep_payload(pattern: str, scope: str, matches: Sequence[GrepMatch]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pattern": pattern,
        "scope": scope,
        "matches": [asdict(item) for item in matches],
    }

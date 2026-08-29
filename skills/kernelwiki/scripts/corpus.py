from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any

from kernelwiki_common import KernelWikiError, load_yaml_document, parse_markdown, require_within


ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
SYMBOLIC_DIMENSION_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

SOURCE_REQUIRED = frozenset({
    "schema_version", "id", "source_kind", "title", "url", "repository_id", "captured_at",
    "target_disposition", "languages", "kernel_types", "techniques", "hardware_features", "tags",
    "license_state",
})
SOURCE_OPTIONAL = frozenset({
    "artifact_dir", "implementation_profile_ids", "runtime_fingerprints", "audiences",
    "profile_authority", "strict_vnext_validated", "missing_evidence",
})
CARD_REQUIRED = frozenset({
    "schema_version", "id", "title", "type", "audiences", "authority", "summary", "targets",
    "target_match", "languages", "kernel_types", "techniques", "hardware_features", "tags", "symptoms",
    "sources", "related", "prerequisites", "version_sensitive", "observations", "examples",
})
CARD_OPTIONAL = frozenset({"candidate_techniques"})
OBSERVATION_FIELDS = frozenset({
    "id", "text", "source_id", "locator", "evidence_level", "reproduction", "targets", "target_match",
    "implementation_profile_id", "runtime_fingerprint", "versions", "transfer_boundaries",
})
EXAMPLE_BASE_FIELDS = frozenset({
    "id", "role", "subtype", "source_id", "locator", "evidence_level", "reproduction", "target_id",
    "implementation_profile_id", "profile_authority", "runtime_fingerprint", "operator_family", "shape",
    "dtype", "terminal_classification", "comparability", "measurement_fingerprint", "baseline_id",
    "candidate_id", "observed", "transfer_boundary", "reconsider_when",
})
CAPABILITY_GAP_FIELDS = frozenset({"capability_id", "capability_status", "required_probe_or_authority"})
MEASUREMENT_FIELDS = frozenset({"metric", "value", "statistic", "unit"})
VERSION_CLAIM_FIELDS = frozenset({
    "id", "card_ids", "status", "supported_versions", "last_verified_at", "source_ids",
})

BASE_CARD_HEADINGS = (
    "Summary", "Problem or symptom", "Mechanism", "Applicability", "Implementation approaches",
    "Expected observables", "Risks and counterexamples", "Examples", "Transfer boundaries",
    "Required local checks", "Sources",
)
KERNEL_EXTRA_HEADINGS = (
    "Shape and contract", "Implementation structure", "Source excerpt or snippet", "Measured claims",
    "What transfers", "What does not transfer",
)


@dataclass(frozen=True)
class SourceRecord:
    path: Path
    metadata: Mapping[str, Any]
    body: str

    @property
    def source_id(self) -> str:
        return str(self.metadata["id"])


@dataclass(frozen=True)
class WikiCard:
    path: Path
    metadata: Mapping[str, Any]
    body: str

    @property
    def card_id(self) -> str:
        return str(self.metadata["id"])


@dataclass(frozen=True)
class Corpus:
    root: Path
    sources: Mapping[str, SourceRecord]
    cards: Mapping[str, WikiCard]
    taxonomy: Mapping[str, tuple[str, ...]]
    aliases: Mapping[str, tuple[str, ...]]
    version_claims: tuple[Mapping[str, Any], ...]
    repository_ids: frozenset[str]


def _fail(code: str, message: str, path: Path) -> None:
    raise KernelWikiError(code, message, path)


def _require_mapping(value: Any, code: str, path: Path) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code, "expected a mapping", path)
    return value


def _require_list(value: Any, code: str, path: Path) -> list[Any]:
    if not isinstance(value, list):
        _fail(code, "expected a list", path)
    return value


def _require_exact_fields(
    value: Mapping[str, Any], required: frozenset[str], optional: frozenset[str], prefix: str, path: Path
) -> None:
    missing = sorted(required - value.keys())
    if missing:
        _fail(f"{prefix}-field-required", f"missing fields: {', '.join(missing)}", path)
    unknown = sorted(value.keys() - required - optional)
    if unknown:
        _fail(f"{prefix}-field-unknown", f"unknown fields: {', '.join(unknown)}", path)


def _require_string(value: Any, code: str, path: Path, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        _fail(code, "expected a nonempty string" if nonempty else "expected a string", path)
    return value


def _require_nullable_string(value: Any, code: str, path: Path) -> str | None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        _fail(code, "expected a nonempty string or null", path)
    return value


def _require_id(value: Any, path: Path) -> str:
    value = _require_string(value, "id-invalid", path)
    if not ID_RE.fullmatch(value):
        _fail("id-invalid", f"invalid ID {value!r}", path)
    return value


def _require_sorted_unique_strings(
    value: Any, code: str, path: Path, *, nonempty: bool = False, identifiers: bool = False
) -> tuple[str, ...]:
    items = _require_list(value, code, path)
    if nonempty and not items:
        _fail(code, "list must not be empty", path)
    for item in items:
        _require_id(item, path) if identifiers else _require_string(item, code, path)
    if items != sorted(set(items)):
        _fail(code, "list must be sorted and unique", path)
    return tuple(items)


def _taxonomy_values(corpus: Corpus, key: str) -> tuple[str, ...]:
    if key not in corpus.taxonomy:
        _fail("taxonomy-key-missing", f"missing taxonomy key {key}", corpus.root / "data" / "taxonomy.yaml")
    return corpus.taxonomy[key]


def _require_taxonomy(value: Any, key: str, corpus: Corpus, path: Path) -> str:
    value = _require_string(value, "taxonomy-unknown", path)
    if value not in _taxonomy_values(corpus, key):
        _fail("taxonomy-unknown", f"{value!r} is not in taxonomy {key}", path)
    return value


def _require_taxonomy_list(
    value: Any, key: str, corpus: Corpus, path: Path, *, nonempty: bool = False
) -> tuple[str, ...]:
    items = _require_sorted_unique_strings(value, "taxonomy-list-invalid", path, nonempty=nonempty)
    allowed = set(_taxonomy_values(corpus, key))
    unknown = sorted(set(items) - allowed)
    if unknown:
        _fail("taxonomy-unknown", f"unknown {key}: {', '.join(unknown)}", path)
    return items


def _load_schema_versions(path: Path) -> None:
    document = _require_mapping(load_yaml_document(path), "schemas-invalid", path)
    expected = {"source": 1, "card": 1, "catalog": 1, "query_result": 1}
    if dict(document) != expected:
        _fail("schemas-invalid", "schema version registry does not match v1", path)


def _load_taxonomy(path: Path) -> dict[str, tuple[str, ...]]:
    document = _require_mapping(load_yaml_document(path), "taxonomy-invalid", path)
    if document.get("schema_version") != 1:
        _fail("taxonomy-schema-invalid", "taxonomy schema_version must be 1", path)
    taxonomy: dict[str, tuple[str, ...]] = {}
    for key, value in document.items():
        if key == "schema_version":
            continue
        taxonomy[key] = _require_sorted_unique_strings(value, "taxonomy-invalid", path, nonempty=True)
    return taxonomy


def _load_aliases(path: Path, taxonomy: Mapping[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    document = _require_mapping(load_yaml_document(path), "aliases-invalid", path)
    known = {item for values in taxonomy.values() for item in values}
    aliases: dict[str, tuple[str, ...]] = {}
    for canonical, values in document.items():
        _require_string(canonical, "aliases-invalid", path)
        if canonical not in known:
            _fail("alias-canonical-unknown", f"unknown canonical term {canonical}", path)
        aliases[canonical] = _require_sorted_unique_strings(values, "aliases-invalid", path, nonempty=True)
    if list(aliases) != sorted(aliases):
        _fail("aliases-invalid", "canonical alias keys must be sorted", path)
    return aliases


def _load_version_claims(path: Path) -> tuple[Mapping[str, Any], ...]:
    document = _require_mapping(load_yaml_document(path), "version-registry-invalid", path)
    if set(document) != {"schema_version", "claims"} or document.get("schema_version") != 1:
        _fail("version-registry-invalid", "version registry must contain schema_version 1 and claims", path)
    claims = _require_list(document["claims"], "version-registry-invalid", path)
    return tuple(_require_mapping(item, "version-claim-invalid", path) for item in claims)


def _load_repository_ids(root: Path) -> frozenset[str]:
    path = root / "data" / "source-repositories.yaml"
    if not path.exists():
        return frozenset({"local"})
    document = _require_mapping(load_yaml_document(path), "repository-registry-invalid", path)
    records = document.get("repositories", document.get("records", []))
    values = _require_list(records, "repository-registry-invalid", path)
    ids = {"local"}
    for record in values:
        record = _require_mapping(record, "repository-registry-invalid", path)
        ids.add(_require_id(record.get("id"), path))
    return frozenset(ids)


def _load_records(directory: Path, record_type: type[SourceRecord] | type[WikiCard]) -> dict[str, Any]:
    records: dict[str, Any] = {}
    if not directory.exists():
        return records
    for path in sorted(directory.rglob("*.md")):
        require_within(directory, path)
        metadata, body = parse_markdown(path)
        record_id = _require_id(metadata.get("id"), path)
        if record_id in records:
            _fail("id-duplicate", f"duplicate ID {record_id}", path)
        records[record_id] = record_type(path=path, metadata=metadata, body=body)
    return records


def _reject_cross_kind_duplicate_ids(
    sources: Mapping[str, SourceRecord], cards: Mapping[str, WikiCard]
) -> None:
    duplicate = sorted(set(sources) & set(cards))
    if duplicate:
        _fail("id-duplicate", f"IDs used by Source and Card: {', '.join(duplicate)}", cards[duplicate[0]].path)


def load_corpus(root: Path) -> Corpus:
    root = Path(root).resolve()
    _load_schema_versions(root / "data" / "schemas.yaml")
    taxonomy = _load_taxonomy(root / "data" / "taxonomy.yaml")
    aliases = _load_aliases(root / "data" / "aliases.yaml", taxonomy)
    sources = _load_records(root / "sources", SourceRecord)
    cards = _load_records(root / "wiki", WikiCard)
    _reject_cross_kind_duplicate_ids(sources, cards)
    return Corpus(
        root=root,
        sources=sources,
        cards=cards,
        taxonomy=taxonomy,
        aliases=aliases,
        version_claims=_load_version_claims(root / "data" / "version-claims.yaml"),
        repository_ids=_load_repository_ids(root),
    )


def _validate_schema_version(metadata: Mapping[str, Any], kind: str, path: Path) -> None:
    if metadata.get("schema_version") != 1:
        _fail(f"{kind}-schema-invalid", "schema_version must be 1", path)


def _validate_relative_path(value: Any, root: Path, code: str, path: Path) -> None:
    text = _require_string(value, code, path)
    candidate = Path(text)
    if candidate.is_absolute():
        _fail("path-absolute", f"absolute path is forbidden: {text}", path)
    require_within(root, root / candidate)


def _validate_source(source: SourceRecord, corpus: Corpus) -> None:
    metadata = source.metadata
    path = source.path
    _require_exact_fields(metadata, SOURCE_REQUIRED, SOURCE_OPTIONAL, "source", path)
    _validate_schema_version(metadata, "source", path)
    _require_id(metadata["id"], path)
    _require_taxonomy(metadata["source_kind"], "source_kinds", corpus, path)
    for key in ("title", "url", "repository_id"):
        _require_string(metadata[key], f"source-{key}-invalid", path)
    captured_at = metadata["captured_at"]
    if not isinstance(captured_at, str) or not captured_at.strip():
        _fail("source-captured-at-invalid", "captured_at must be a checked-in string", path)
    _require_taxonomy(metadata["target_disposition"], "target_matches", corpus, path)
    _require_taxonomy_list(metadata["languages"], "languages", corpus, path)
    _require_taxonomy_list(metadata["kernel_types"], "kernel_types", corpus, path)
    _require_taxonomy_list(metadata["techniques"], "techniques", corpus, path)
    _require_taxonomy_list(metadata["hardware_features"], "hardware_features", corpus, path)
    _require_taxonomy_list(metadata["tags"], "tags", corpus, path)
    _require_taxonomy(metadata["license_state"], "license_states", corpus, path)
    if metadata["repository_id"] not in corpus.repository_ids:
        _fail("repository-id-missing", f"unknown repository_id {metadata['repository_id']}", path)
    if "artifact_dir" in metadata:
        _validate_relative_path(metadata["artifact_dir"], corpus.root, "source-artifact-dir-invalid", path)
    if "implementation_profile_ids" in metadata:
        _require_sorted_unique_strings(metadata["implementation_profile_ids"], "source-profile-list-invalid", path)
    if "runtime_fingerprints" in metadata:
        _require_sorted_unique_strings(metadata["runtime_fingerprints"], "source-runtime-list-invalid", path)
    if "audiences" in metadata:
        _require_taxonomy_list(metadata["audiences"], "audiences", corpus, path, nonempty=True)
    if not source.body.strip():
        _fail("source-body-required", "Source body must not be empty", path)

    if metadata["source_kind"] == "local-campaign":
        required = {"profile_authority", "strict_vnext_validated", "missing_evidence", "audiences"}
        missing = sorted(required - metadata.keys())
        if missing:
            _fail("local-campaign-field-required", f"missing fields: {', '.join(missing)}", path)
        authority = metadata["profile_authority"]
        if authority not in {"current-vnext", "historical-noncanonical"}:
            _fail("local-campaign-authority-invalid", "invalid local profile authority", path)
        if not isinstance(metadata["strict_vnext_validated"], bool):
            _fail("local-campaign-strict-invalid", "strict_vnext_validated must be boolean", path)
        _require_sorted_unique_strings(metadata["missing_evidence"], "local-campaign-missing-evidence-invalid", path)
        audiences = metadata["audiences"]
        if authority == "historical-noncanonical" and (
            metadata["strict_vnext_validated"] is not False or audiences != ["designer"]
        ):
            _fail(
                "local-campaign-authority-invalid",
                "historical local evidence must be non-strict and Designer-only",
                path,
            )
    else:
        local_fields = {"profile_authority", "strict_vnext_validated", "missing_evidence"} & metadata.keys()
        if local_fields:
            _fail("source-field-conditional", "local-campaign fields require source_kind local-campaign", path)


def _validate_observation(observation: Any, card: WikiCard, corpus: Corpus) -> None:
    path = card.path
    item = _require_mapping(observation, "observation-invalid", path)
    _require_exact_fields(item, OBSERVATION_FIELDS, frozenset(), "observation", path)
    _require_id(item["id"], path)
    for key in ("text", "source_id", "locator"):
        _require_string(item[key], f"observation-{key}-invalid", path)
    _require_taxonomy(item["evidence_level"], "evidence_levels", corpus, path)
    _require_taxonomy(item["reproduction"], "reproduction_levels", corpus, path)
    _require_sorted_unique_strings(item["targets"], "observation-targets-invalid", path, nonempty=True)
    _require_taxonomy(item["target_match"], "target_matches", corpus, path)
    _require_nullable_string(item["implementation_profile_id"], "observation-nullable-string", path)
    _require_nullable_string(item["runtime_fingerprint"], "observation-nullable-string", path)
    _require_sorted_unique_strings(item["versions"], "observation-versions-invalid", path)
    _require_sorted_unique_strings(item["transfer_boundaries"], "observation-transfer-boundaries-invalid", path, nonempty=True)


def _validate_measurement(item: Any, corpus: Corpus, path: Path) -> None:
    measurement = _require_mapping(item, "measurement-invalid", path)
    _require_exact_fields(measurement, MEASUREMENT_FIELDS, frozenset(), "measurement", path)
    _require_taxonomy(measurement["metric"], "measurement_metrics", corpus, path)
    value = measurement["value"]
    if isinstance(value, bool):
        pass
    elif not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        _fail("measurement-value-invalid", "measurement value must be finite numeric or boolean", path)
    _require_taxonomy(measurement["statistic"], "measurement_statistics", corpus, path)
    _require_taxonomy(measurement["unit"], "measurement_units", corpus, path)


def _validate_example(example: Any, card: WikiCard, corpus: Corpus) -> None:
    path = card.path
    item = _require_mapping(example, "example-invalid", path)
    role = item.get("role")
    expected = EXAMPLE_BASE_FIELDS | (CAPABILITY_GAP_FIELDS if role == "capability-gap" else frozenset())
    if role == "capability-gap":
        missing_gap = sorted(CAPABILITY_GAP_FIELDS - item.keys())
        if missing_gap:
            _fail("capability-gap-field-required", f"missing fields: {', '.join(missing_gap)}", path)
    _require_exact_fields(item, expected, frozenset(), "example", path)
    _require_id(item["id"], path)
    _require_taxonomy(role, "example_roles", corpus, path)
    _require_taxonomy(item["subtype"], "example_subtypes", corpus, path)
    for key in ("source_id", "locator", "target_id", "operator_family", "transfer_boundary"):
        _require_string(item[key], "example-transfer-boundary-required" if key == "transfer_boundary" else f"example-{key}-invalid", path)
    _require_taxonomy(item["evidence_level"], "evidence_levels", corpus, path)
    _require_taxonomy(item["reproduction"], "reproduction_levels", corpus, path)
    _require_nullable_string(item["implementation_profile_id"], "example-nullable-string", path)
    _require_taxonomy(item["profile_authority"], "profile_authorities", corpus, path)
    _require_nullable_string(item["runtime_fingerprint"], "example-nullable-string", path)
    shape = _require_mapping(item["shape"], "example-shape-invalid", path)
    if list(shape) != sorted(shape):
        _fail("example-shape-order", "shape keys must be sorted", path)
    for dimension, size in shape.items():
        _require_string(dimension, "example-shape-invalid", path)
        if isinstance(size, bool) or not (
            isinstance(size, int) and size > 0
            or isinstance(size, str) and SYMBOLIC_DIMENSION_RE.fullmatch(size)
        ):
            _fail("example-shape-invalid", f"invalid shape value for {dimension}", path)
    _require_taxonomy(item["dtype"], "dtypes", corpus, path)
    _require_taxonomy(item["terminal_classification"], "terminal_classifications", corpus, path)
    _require_taxonomy(item["comparability"], "comparability_classes", corpus, path)
    for key in ("measurement_fingerprint", "baseline_id", "candidate_id"):
        _require_nullable_string(item[key], "example-nullable-string", path)
    observed = _require_list(item["observed"], "example-observed-invalid", path)
    for measurement in observed:
        _validate_measurement(measurement, corpus, path)
    metrics = [measurement["metric"] for measurement in observed]
    if metrics != sorted(metrics):
        _fail("example-observed-order", "observed measurements must be sorted by metric", path)
    _require_sorted_unique_strings(item["reconsider_when"], "example-reconsider-when-invalid", path, nonempty=True)

    if role in {"positive", "counterexample"} and not observed:
        _fail("example-observed-required", f"{role} examples require observations", path)
    if item["evidence_level"] == "local-verifier":
        required_identity = (
            "measurement_fingerprint", "baseline_id", "candidate_id", "implementation_profile_id", "runtime_fingerprint"
        )
        if any(item[key] is None for key in required_identity):
            _fail("example-local-identity-required", "local examples require complete measurement identity", path)
    if role == "capability-gap":
        if observed:
            _fail("capability-gap-observed-forbidden", "capability gaps require observed: []", path)
        _require_id(item["capability_id"], path)
        if item["capability_status"] not in {"unknown", "unsupported"}:
            _fail("capability-gap-status-invalid", "invalid capability status", path)
        _require_string(item["required_probe_or_authority"], "capability-gap-authority-required", path)


def _heading_names(body: str) -> frozenset[str]:
    return frozenset(line[3:].strip() for line in body.splitlines() if line.startswith("## "))


def _validate_card(card: WikiCard, corpus: Corpus) -> None:
    metadata = card.metadata
    path = card.path
    _require_exact_fields(metadata, CARD_REQUIRED, CARD_OPTIONAL, "card", path)
    _validate_schema_version(metadata, "card", path)
    _require_id(metadata["id"], path)
    _require_string(metadata["title"], "card-title-invalid", path)
    card_type = _require_taxonomy(metadata["type"], "card_types", corpus, path)
    _require_taxonomy_list(metadata["audiences"], "audiences", corpus, path, nonempty=True)
    _require_taxonomy(metadata["authority"], "authorities", corpus, path)
    _require_string(metadata["summary"], "card-summary-invalid", path)
    _require_sorted_unique_strings(metadata["targets"], "card-targets-invalid", path, nonempty=True)
    _require_taxonomy(metadata["target_match"], "target_matches", corpus, path)
    _require_taxonomy_list(metadata["languages"], "languages", corpus, path)
    _require_taxonomy_list(metadata["kernel_types"], "kernel_types", corpus, path)
    techniques = _require_taxonomy_list(metadata["techniques"], "techniques", corpus, path)
    if card_type == "technique" and not techniques:
        _fail(
            "technique-identity-invalid",
            "technique Cards must bind at least one taxonomy technique",
            path,
        )
    _require_taxonomy_list(metadata["hardware_features"], "hardware_features", corpus, path)
    _require_taxonomy_list(metadata["tags"], "tags", corpus, path)
    _require_taxonomy_list(metadata["symptoms"], "symptoms", corpus, path)
    for key in ("sources", "related", "prerequisites", "version_sensitive"):
        _require_sorted_unique_strings(metadata[key], f"card-{key}-invalid", path, identifiers=True)
    observations = _require_list(metadata["observations"], "card-observations-invalid", path)
    examples = _require_list(metadata["examples"], "card-examples-invalid", path)
    for observation in observations:
        _validate_observation(observation, card, corpus)
    for example in examples:
        _validate_example(example, card, corpus)
    observation_ids = [item.get("id") for item in observations if isinstance(item, Mapping)]
    example_ids = [item.get("id") for item in examples if isinstance(item, Mapping)]
    if len(observation_ids) != len(set(observation_ids)) or len(example_ids) != len(set(example_ids)):
        _fail("id-duplicate", "duplicate observation or example ID", path)

    if card_type == "pattern":
        if "candidate_techniques" not in metadata:
            _fail("card-field-required", "pattern Cards require candidate_techniques", path)
        _require_sorted_unique_strings(metadata["candidate_techniques"], "candidate-techniques-invalid", path, identifiers=True)
    elif "candidate_techniques" in metadata:
        _fail("card-field-conditional", "candidate_techniques requires type pattern", path)

    if not card.body.strip():
        _fail("card-body-required", "Card body must not be empty", path)
    headings = _heading_names(card.body)
    required_headings: Sequence[str] = BASE_CARD_HEADINGS if card_type in {"technique", "pattern", "kernel"} else ()
    if card_type == "kernel":
        required_headings = (*BASE_CARD_HEADINGS, *KERNEL_EXTRA_HEADINGS)
    missing_headings = [heading for heading in required_headings if heading not in headings]
    if missing_headings:
        _fail("heading-required", f"missing headings: {', '.join(missing_headings)}", path)


def _track2_name_forms(value: str) -> frozenset[str]:
    raw = value.casefold()
    return frozenset({raw, raw.replace("_", "-"), raw.replace("-", "_")})


def _validate_track2_boundaries(corpus: Corpus) -> None:
    path = corpus.root / "data" / "evaluation-holdouts.yaml"
    if not path.exists():
        return
    document = _require_mapping(load_yaml_document(path), "holdout-invalid", path)
    track2 = _require_mapping(document.get("track2"), "holdout-invalid", path)
    contexts = [*track2.get("development_contexts", []), *track2.get("holdout_contexts", [])]
    context_names = tuple(_require_string(context, "holdout-invalid", path) for context in contexts)
    forbidden = frozenset(form for context in context_names for form in _track2_name_forms(context))
    for card in corpus.cards.values():
        relative_path = card.path.relative_to(corpus.root)
        identities = (card.card_id, *relative_path.parts)
        for identity in identities:
            normalized = _track2_name_forms(identity)
            if any(token in candidate for token in forbidden for candidate in normalized):
                _fail("track2-card-forbidden", f"Track 2 operator Card is forbidden: {card.card_id}", card.path)


def _validate_links(corpus: Corpus) -> None:
    for card in corpus.cards.values():
        metadata = card.metadata
        for source_id in metadata["sources"]:
            if source_id not in corpus.sources:
                _fail("source-missing", f"missing Source {source_id}", card.path)
        for related_id in metadata["related"]:
            if related_id not in corpus.cards:
                _fail("related-missing", f"missing related Card {related_id}", card.path)
        for prerequisite_id in metadata["prerequisites"]:
            if prerequisite_id not in corpus.cards:
                _fail("prerequisite-missing", f"missing prerequisite Card {prerequisite_id}", card.path)
        if metadata["type"] == "pattern":
            for technique_id in metadata["candidate_techniques"]:
                target = corpus.cards.get(technique_id)
                if target is None or target.metadata["type"] != "technique":
                    _fail("candidate-technique-missing", f"candidate technique {technique_id} is not a technique Card", card.path)
        for observation in metadata["observations"]:
            source_id = observation["source_id"]
            if source_id not in corpus.sources:
                _fail("observation-source-missing", f"missing observation Source {source_id}", card.path)
            if source_id not in metadata["sources"]:
                _fail("observation-source-scope", f"observation Source {source_id} is outside Card sources", card.path)
        for example in metadata["examples"]:
            source_id = example["source_id"]
            if source_id not in corpus.sources:
                _fail("example-source-missing", f"missing example Source {source_id}", card.path)
            if source_id not in metadata["sources"]:
                _fail("example-source-scope", f"example Source {source_id} is outside Card sources", card.path)


def _validate_version_registry(corpus: Corpus) -> None:
    path = corpus.root / "data" / "version-claims.yaml"
    claim_ids: set[str] = set()
    previous_id = ""
    for claim in corpus.version_claims:
        _require_exact_fields(claim, VERSION_CLAIM_FIELDS, frozenset(), "version-claim", path)
        claim_id = _require_id(claim["id"], path)
        if claim_id in claim_ids:
            _fail("id-duplicate", f"duplicate version claim {claim_id}", path)
        if claim_id < previous_id:
            _fail("version-claim-order", "version claims must be sorted by id", path)
        previous_id = claim_id
        claim_ids.add(claim_id)
        card_ids = _require_sorted_unique_strings(claim["card_ids"], "version-card-ids-invalid", path, identifiers=True)
        source_ids = _require_sorted_unique_strings(claim["source_ids"], "version-source-ids-invalid", path, identifiers=True)
        _require_sorted_unique_strings(claim["supported_versions"], "version-supported-invalid", path)
        if claim["status"] not in {"current", "stale", "unknown"}:
            _fail("version-status-invalid", "invalid version status", path)
        verified = claim["last_verified_at"]
        if verified is not None and (not isinstance(verified, str) or not DATE_RE.fullmatch(verified)):
            _fail("version-date-invalid", "last_verified_at must be YYYY-MM-DD or null", path)
        for card_id in card_ids:
            if card_id not in corpus.cards:
                _fail("version-card-missing", f"missing Card {card_id}", path)
        for source_id in source_ids:
            if source_id not in corpus.sources:
                _fail("version-source-missing", f"missing Source {source_id}", path)
    for card in corpus.cards.values():
        for claim_id in card.metadata["version_sensitive"]:
            if claim_id not in claim_ids:
                _fail("version-claim-missing", f"missing version claim {claim_id}", card.path)
            claim = next(item for item in corpus.version_claims if item["id"] == claim_id)
            if card.card_id not in claim["card_ids"]:
                _fail("version-card-backref-missing", f"version claim {claim_id} does not refer to Card", card.path)


def validate_examples_document(path: Path, corpus: Corpus) -> tuple[Mapping[str, Any], ...]:
    document = _require_mapping(load_yaml_document(path), "examples-document-invalid", path)
    validated: list[Mapping[str, Any]] = []
    example_ids: set[str] = set()
    synthetic_card = WikiCard(path=path, metadata={}, body="")
    for name, raw_example in document.items():
        _require_string(name, "examples-document-invalid", path)
        example = _require_mapping(raw_example, "example-invalid", path)
        _validate_example(example, synthetic_card, corpus)
        example_id = str(example["id"])
        if example_id in example_ids:
            _fail("id-duplicate", f"duplicate example ID {example_id}", path)
        example_ids.add(example_id)
        source_id = str(example["source_id"])
        if source_id not in corpus.sources:
            _fail("example-source-missing", f"missing example Source {source_id}", path)
        validated.append(example)
    return tuple(validated)


def _validate_examples(corpus: Corpus) -> None:
    path = corpus.root / "examples.yaml"
    if path.exists():
        validate_examples_document(path, corpus)


def validate_corpus(corpus: Corpus) -> None:
    for source in corpus.sources.values():
        _validate_source(source, corpus)
    _validate_track2_boundaries(corpus)
    for card in corpus.cards.values():
        _validate_card(card, corpus)
    _validate_links(corpus)
    _validate_examples(corpus)
    _validate_version_registry(corpus)

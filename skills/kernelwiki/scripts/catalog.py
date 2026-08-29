from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
from typing import Any

from corpus import Corpus, SourceRecord, WikiCard, load_corpus, validate_corpus
from kernelwiki_common import KernelWikiError, canonical_json_bytes, sha256_bytes


CATALOG_FIELDS = (
    "schema_version",
    "id",
    "path",
    "body_sha256",
    "type",
    "title",
    "summary",
    "audiences",
    "targets",
    "target_match",
    "languages",
    "kernel_types",
    "techniques",
    "hardware_features",
    "candidate_techniques",
    "tags",
    "symptoms",
    "source_ids",
    "source_repositories",
    "version_claims",
    "source_count",
    "evidence_levels",
    "reproduction_levels",
    "positive_example_count",
    "counterexample_count",
    "capability_gap_count",
)

QUERY_VIEW_NAMES = (
    "by-problem.md",
    "by-technique.md",
    "by-hardware-feature.md",
    "by-kernel-type.md",
    "by-language.md",
    "by-target.md",
    "by-source-repo.md",
    "by-version.md",
    "by-evidence-level.md",
)

GENERATED_OUTPUT_PATHS = (
    "compiled/catalog.jsonl",
    *(f"queries/{name}" for name in QUERY_VIEW_NAMES),
)
GENERATED_STAGING_PREFIX = ".generated-staging-"

TARGET_MATCH_ORDER = ("exact", "family", "backend", "analogy-only", "unknown")
VERSION_STATUS_ORDER = ("current", "stale", "unknown")


def one_line(text: str) -> str:
    return " ".join(str(text).split())


def join(values: Sequence[str]) -> str:
    rendered = ", ".join(sorted(str(value) for value in values))
    return rendered or "none"


def resolve_source_repositories(card: WikiCard, corpus: Corpus) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                str(corpus.sources[source_id].metadata["repository_id"])
                for source_id in card.metadata["sources"]
            }
        )
    )


def resolve_version_claims(
    card: WikiCard, corpus: Corpus
) -> tuple[Mapping[str, Any], ...]:
    requested = set(card.metadata["version_sensitive"])
    claims = []
    for claim in corpus.version_claims:
        if claim["id"] not in requested:
            continue
        claims.append(
            {
                "id": claim["id"],
                "status": claim["status"],
                "last_verified_at": claim["last_verified_at"],
                "supported_versions": sorted(claim["supported_versions"]),
            }
        )
    return tuple(sorted(claims, key=lambda claim: str(claim["id"])))


def resolve_evidence_levels(card: WikiCard) -> tuple[str, ...]:
    values = {str(item["evidence_level"]) for item in card.metadata["observations"]}
    values.update(str(item["evidence_level"]) for item in card.metadata["examples"])
    return tuple(sorted(values))


def resolve_reproduction_levels(card: WikiCard) -> tuple[str, ...]:
    values = {str(item["reproduction"]) for item in card.metadata["observations"]}
    values.update(str(item["reproduction"]) for item in card.metadata["examples"])
    return tuple(sorted(values))


def count_examples(card: WikiCard, role: str) -> int:
    return sum(1 for example in card.metadata["examples"] if example["role"] == role)


def card_to_catalog_record(card: WikiCard, corpus: Corpus) -> dict[str, Any]:
    record = {
        "schema_version": 1,
        "id": card.card_id,
        "path": card.path.relative_to(corpus.root).as_posix(),
        "body_sha256": sha256_bytes(card.body.encode("utf-8")),
        "type": card.metadata["type"],
        "title": card.metadata["title"],
        "summary": card.metadata["summary"],
        "audiences": sorted(card.metadata["audiences"]),
        "targets": sorted(card.metadata["targets"]),
        "target_match": card.metadata["target_match"],
        "languages": sorted(card.metadata["languages"]),
        "kernel_types": sorted(card.metadata["kernel_types"]),
        "techniques": sorted(card.metadata["techniques"]),
        "hardware_features": sorted(card.metadata["hardware_features"]),
        "candidate_techniques": sorted(card.metadata.get("candidate_techniques", [])),
        "tags": sorted(card.metadata["tags"]),
        "symptoms": sorted(card.metadata.get("symptoms", [])),
        "source_ids": sorted(card.metadata["sources"]),
        "source_repositories": list(resolve_source_repositories(card, corpus)),
        "version_claims": list(resolve_version_claims(card, corpus)),
        "source_count": len(card.metadata["sources"]),
        "evidence_levels": list(resolve_evidence_levels(card)),
        "reproduction_levels": list(resolve_reproduction_levels(card)),
        "positive_example_count": count_examples(card, "positive"),
        "counterexample_count": count_examples(card, "counterexample"),
        "capability_gap_count": count_examples(card, "capability-gap"),
    }
    if tuple(record) != CATALOG_FIELDS:
        raise KernelWikiError(
            "catalog-field-invalid",
            "catalog record fields do not match schema",
            card.path,
        )
    return record


def build_catalog(corpus: Corpus) -> tuple[dict[str, Any], ...]:
    return tuple(
        card_to_catalog_record(corpus.cards[card_id], corpus)
        for card_id in sorted(corpus.cards)
    )


def card_row(record: Mapping[str, Any]) -> str:
    return (
        f"- [{record['id']}](../{record['path']}) — {one_line(record['summary'])} "
        f"— target `{record['target_match']}:{join(record['targets'])}` "
        f"— evidence `{join(record['evidence_levels'])}` "
        f"— reproduction `{join(record['reproduction_levels'])}` "
        f"— sources `{record['source_count']}`"
    )


def source_row(source: SourceRecord, corpus: Corpus) -> str:
    path = source.path.relative_to(corpus.root).as_posix()
    return (
        f"- [{source.source_id}](../{path}) — {one_line(source.metadata['title'])} "
        f"— captured `{source.metadata['captured_at']}` "
        f"— license `{source.metadata['license_state']}` "
        f"— target `{source.metadata['target_disposition']}`"
    )


def generated_header(title: str) -> list[str]:
    return [
        "<!-- GENERATED by scripts/generate_indices.py; DO NOT EDIT. -->",
        f"# {title}",
        "",
    ]


def _append_section(lines: list[str], heading: str, rows: Sequence[str]) -> None:
    lines.extend([heading, ""])
    lines.extend(rows or ("- _None._",))
    lines.append("")


def _render(lines: list[str]) -> bytes:
    while lines and lines[-1] == "":
        lines.pop()
    return ("\n".join(lines) + "\n").encode("utf-8")


def _normalize_records(
    corpus: Corpus, records: Sequence[Mapping[str, Any]]
) -> tuple[Mapping[str, Any], ...]:
    normalized: list[Mapping[str, Any]] = []
    ids: list[str] = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) != set(CATALOG_FIELDS):
            raise KernelWikiError(
                "catalog-record-invalid",
                "catalog record fields do not match schema",
                corpus.root,
            )
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise KernelWikiError(
                "catalog-record-invalid",
                "catalog record id must be a nonempty string",
                corpus.root,
            )
        ids.append(record_id)
        normalized.append(record)
    if len(ids) != len(set(ids)):
        raise KernelWikiError(
            "catalog-record-duplicate", "catalog record IDs must be unique", corpus.root
        )
    if set(ids) != set(corpus.cards):
        raise KernelWikiError(
            "catalog-record-set",
            "catalog records must cover every Card exactly once",
            corpus.root,
        )
    return tuple(sorted(normalized, key=lambda record: str(record["id"])))


def _records_by_id(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return {str(record["id"]): record for record in records}


def _render_by_problem(
    corpus: Corpus, records: Sequence[Mapping[str, Any]]
) -> bytes:
    lines = generated_header("By Problem")
    by_id = _records_by_id(records)
    for symptom in corpus.taxonomy["symptoms"]:
        lines.extend([f"## {symptom}", ""])
        patterns = [
            record
            for record in records
            if record["type"] == "pattern" and symptom in record["symptoms"]
        ]
        _append_section(lines, "### Patterns", [card_row(record) for record in patterns])
        candidate_ids = sorted(
            {candidate for record in patterns for candidate in record["candidate_techniques"]}
        )
        _append_section(
            lines,
            "### Candidate techniques",
            [
                f"- [{candidate_id}](../{by_id[candidate_id]['path']})"
                for candidate_id in candidate_ids
            ],
        )
    return _render(lines)


def _render_by_simple_taxonomy(
    corpus: Corpus,
    records: Sequence[Mapping[str, Any]],
    *,
    title: str,
    taxonomy_key: str,
    record_key: str,
) -> bytes:
    lines = generated_header(title)
    for value in corpus.taxonomy[taxonomy_key]:
        rows = [card_row(record) for record in records if value in record[record_key]]
        _append_section(lines, f"## {value}", rows)
    return _render(lines)


def _render_by_technique(
    corpus: Corpus, records: Sequence[Mapping[str, Any]]
) -> bytes:
    lines = generated_header("By Technique")
    for value in corpus.taxonomy["techniques"]:
        rows = [
            card_row(record)
            for record in records
            if value in record["techniques"]
            or (record["type"] == "technique" and record["id"] == f"technique-{value}")
        ]
        _append_section(lines, f"## {value}", rows)
    return _render(lines)


def _render_by_target(records: Sequence[Mapping[str, Any]]) -> bytes:
    lines = generated_header("By Target")
    targets = sorted({target for record in records for target in record["targets"]})
    for target in targets:
        lines.extend([f"## {target}", ""])
        for target_match in TARGET_MATCH_ORDER:
            rows = [
                card_row(record)
                for record in records
                if target in record["targets"] and record["target_match"] == target_match
            ]
            _append_section(lines, f"### {target_match}", rows)
    return _render(lines)


def _render_by_source_repo(
    corpus: Corpus, records: Sequence[Mapping[str, Any]]
) -> bytes:
    lines = generated_header("By Source Repository")
    for repository_id in sorted(corpus.repository_ids):
        lines.extend([f"## {repository_id}", ""])
        sources = [
            corpus.sources[source_id]
            for source_id in sorted(corpus.sources)
            if corpus.sources[source_id].metadata["repository_id"] == repository_id
        ]
        _append_section(
            lines, "### Sources", [source_row(source, corpus) for source in sources]
        )
        cards = [
            record
            for record in records
            if repository_id in record["source_repositories"]
        ]
        _append_section(lines, "### Cards", [card_row(record) for record in cards])
    return _render(lines)


def _render_by_version(
    corpus: Corpus, records: Sequence[Mapping[str, Any]]
) -> bytes:
    lines = generated_header("By Version")
    by_id = _records_by_id(records)
    for status in VERSION_STATUS_ORDER:
        rows: list[str] = []
        for claim in sorted(corpus.version_claims, key=lambda item: str(item["id"])):
            if claim["status"] != status:
                continue
            card_links = ", ".join(
                f"[{card_id}](../{by_id[card_id]['path']})"
                for card_id in sorted(claim["card_ids"])
            ) or "none"
            verified = claim["last_verified_at"] or "unknown"
            rows.append(
                f"- {claim['id']} — versions `{join(claim['supported_versions'])}` "
                f"— verified `{verified}` — Cards: {card_links}"
            )
        _append_section(lines, f"## {status}", rows)
    return _render(lines)


def _render_by_evidence(
    corpus: Corpus, records: Sequence[Mapping[str, Any]]
) -> bytes:
    lines = generated_header("By Evidence Level")
    for evidence in corpus.taxonomy["evidence_levels"]:
        rows = [
            f"{card_row(record)} — examples `positive={record['positive_example_count']}, "
            f"counterexample={record['counterexample_count']}, capability-gap={record['capability_gap_count']}`"
            for record in records
            if evidence in record["evidence_levels"]
        ]
        _append_section(lines, f"## {evidence}", rows)
    return _render(lines)


def render_query_views(
    corpus: Corpus, records: Sequence[Mapping[str, Any]] | None = None
) -> dict[str, bytes]:
    records = _normalize_records(
        corpus, build_catalog(corpus) if records is None else records
    )
    return {
        "queries/by-problem.md": _render_by_problem(corpus, records),
        "queries/by-technique.md": _render_by_technique(corpus, records),
        "queries/by-hardware-feature.md": _render_by_simple_taxonomy(
            corpus,
            records,
            title="By Hardware Feature",
            taxonomy_key="hardware_features",
            record_key="hardware_features",
        ),
        "queries/by-kernel-type.md": _render_by_simple_taxonomy(
            corpus,
            records,
            title="By Kernel Type",
            taxonomy_key="kernel_types",
            record_key="kernel_types",
        ),
        "queries/by-language.md": _render_by_simple_taxonomy(
            corpus,
            records,
            title="By Language",
            taxonomy_key="languages",
            record_key="languages",
        ),
        "queries/by-target.md": _render_by_target(records),
        "queries/by-source-repo.md": _render_by_source_repo(corpus, records),
        "queries/by-version.md": _render_by_version(corpus, records),
        "queries/by-evidence-level.md": _render_by_evidence(corpus, records),
    }


def build_generated_outputs(corpus: Corpus) -> dict[str, bytes]:
    records = build_catalog(corpus)
    catalog_bytes = b"".join(canonical_json_bytes(record) for record in records)
    return {"compiled/catalog.jsonl": catalog_bytes, **render_query_views(corpus, records)}


def _validate_output_set(outputs: Mapping[str, bytes]) -> None:
    expected = set(GENERATED_OUTPUT_PATHS)
    actual = set(outputs)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise KernelWikiError(
            "generated-output-set", f"missing={missing}, unknown={unknown}"
        )
    for relative_path, payload in outputs.items():
        path = PurePosixPath(relative_path)
        if path.is_absolute() or ".." in path.parts or "." in path.parts:
            raise KernelWikiError(
                "generated-output-path", f"invalid generated path {relative_path}"
            )
        if not isinstance(payload, bytes):
            raise KernelWikiError(
                "generated-output-bytes",
                f"generated output {relative_path} must be bytes",
            )


def _resolve_generated_root(root: Path, *, create: bool, code: str) -> Path:
    supplied = Path(root)
    if supplied.is_symlink():
        raise KernelWikiError(code, "generated root must not be a symlink", supplied)
    try:
        if create:
            supplied.mkdir(parents=True, exist_ok=True)
        resolved = supplied.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise KernelWikiError(code, "generated root is missing or invalid", supplied) from error
    if not resolved.is_dir():
        raise KernelWikiError(code, "generated root must be a directory", supplied)
    return resolved


def _managed_file_set(root: Path, *, code: str) -> set[str]:
    paths: set[str] = set()
    for directory_name in ("compiled", "queries"):
        directory = root / directory_name
        if not directory.exists():
            continue
        if not directory.is_dir():
            raise KernelWikiError(code, f"{directory_name} must be a directory", directory)
        for path in directory.iterdir():
            if not path.is_file():
                raise KernelWikiError(code, "generated outputs must be regular files", path)
            paths.add(path.relative_to(root).as_posix())
    return paths


def _load_outputs(root: Path) -> dict[str, bytes]:
    corpus = load_corpus(root)
    validate_corpus(corpus)
    return build_generated_outputs(corpus)


def write_generated_outputs(
    root: Path,
    outputs: Mapping[str, bytes] | None = None,
    *,
    failure_hook: Callable[[str], None] | None = None,
) -> tuple[Path, ...]:
    root = _resolve_generated_root(root, create=True, code="generated-output-invalid")
    outputs = _load_outputs(root) if outputs is None else dict(outputs)
    _validate_output_set(outputs)
    stage = Path(tempfile.mkdtemp(prefix=GENERATED_STAGING_PREFIX, dir=root))
    try:
        for relative_path, payload in sorted(outputs.items()):
            staged = stage / relative_path
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(payload)
        for relative_path in sorted(outputs):
            if failure_hook is not None:
                failure_hook(relative_path)
            final = root / relative_path
            final.parent.mkdir(parents=True, exist_ok=True)
            os.replace(stage / relative_path, final)
        stale = _managed_file_set(root, code="generated-output-invalid") - set(outputs)
        for relative_path in sorted(stale):
            (root / relative_path).unlink()
    except KernelWikiError:
        raise
    except Exception as error:
        raise KernelWikiError("generated-write-failed", str(error), root) from error
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return tuple(root / path for path in GENERATED_OUTPUT_PATHS)


def assert_generated_outputs_current(
    root: Path, outputs: Mapping[str, bytes] | None = None
) -> None:
    root = _resolve_generated_root(root, create=False, code="generated-drift")
    outputs = _load_outputs(root) if outputs is None else dict(outputs)
    _validate_output_set(outputs)
    actual_paths = _managed_file_set(root, code="generated-drift")
    expected_paths = set(GENERATED_OUTPUT_PATHS)
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        raise KernelWikiError(
            "generated-drift",
            f"generated file set differs: missing={missing}, extra={extra}",
            root,
        )
    for relative_path in sorted(outputs):
        path = root / relative_path
        try:
            actual = path.read_bytes()
        except OSError as error:
            raise KernelWikiError(
                "generated-drift", f"cannot read generated output {relative_path}", path
            ) from error
        if actual != outputs[relative_path]:
            raise KernelWikiError(
                "generated-drift", f"generated output differs: {relative_path}", path
            )

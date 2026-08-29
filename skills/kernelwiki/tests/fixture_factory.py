from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from typing import Any

import yaml


SKILL_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
VALID_CORPUS = FIXTURES / "valid-corpus"


def _parse_markdown(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    marker = text.find("\n---\n", 4)
    return yaml.safe_load(text[4:marker]), text[marker + 5 :]


def _write_markdown(path: Path, metadata: dict[str, Any], body: str) -> None:
    frontmatter = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True).rstrip()
    path.write_text(f"---\n{frontmatter}\n---\n{body}", encoding="utf-8")


def card_path(root: Path) -> Path:
    return root / "wiki" / "techniques" / "kernel-fusion.md"


def source_path(root: Path) -> Path:
    return root / "sources" / "docs" / "source-valid-manual.md"


def mutate_card(root: Path, mutate) -> None:
    path = card_path(root)
    metadata, body = _parse_markdown(path)
    mutate(metadata, body)
    _write_markdown(path, metadata, body)


def mutate_source(root: Path, mutate) -> None:
    path = source_path(root)
    metadata, body = _parse_markdown(path)
    mutate(metadata, body)
    _write_markdown(path, metadata, body)


def make_valid_corpus(
    *,
    duplicate_card_id: bool = False,
    extra_tag: str | None = None,
    example_source: str | None = None,
    related_id: str | None = None,
    prerequisite_id: str | None = None,
    card_type: str | None = None,
    missing_heading: str | None = None,
    observation_evidence: str | None = None,
    example_reproduction: str | None = None,
    target_disposition: str | None = None,
    local_example_without_transfer: bool = False,
    card_id: str | None = None,
) -> Path:
    root = Path(tempfile.mkdtemp(prefix="kernelwiki-corpus-"))
    shutil.copytree(VALID_CORPUS, root, dirs_exist_ok=True)

    card = card_path(root)
    metadata, body = _parse_markdown(card)
    if extra_tag is not None:
        metadata["tags"] = sorted({*metadata["tags"], extra_tag})
    if example_source is not None:
        metadata["examples"][0]["source_id"] = example_source
    if related_id is not None:
        metadata["related"] = [related_id]
    if prerequisite_id is not None:
        metadata["prerequisites"] = [prerequisite_id]
    if card_type is not None:
        metadata["type"] = card_type
    if missing_heading is not None:
        section = f"## {missing_heading}\n"
        body = body.replace(section, "## Removed heading\n", 1)
    if observation_evidence is not None:
        metadata["observations"][0]["evidence_level"] = observation_evidence
    if example_reproduction is not None:
        metadata["examples"][0]["reproduction"] = example_reproduction
    if local_example_without_transfer:
        example = metadata["examples"][0]
        example.update(
            evidence_level="local-verifier",
            reproduction="benchmarked",
            profile_authority="current-vnext",
            implementation_profile_id="profile-test",
            runtime_fingerprint="runtime-test",
            measurement_fingerprint="measurement-test",
            baseline_id="baseline.py",
            candidate_id="candidate.py",
            transfer_boundary="",
        )
    if card_id is not None:
        metadata["id"] = card_id
    _write_markdown(card, metadata, body)

    if duplicate_card_id:
        duplicate = root / "wiki" / "patterns" / "duplicate.md"
        duplicate.parent.mkdir(parents=True, exist_ok=True)
        duplicate.write_text(card.read_text(encoding="utf-8"), encoding="utf-8")

    if target_disposition is not None:
        source = source_path(root)
        source_metadata, source_body = _parse_markdown(source)
        source_metadata["target_disposition"] = target_disposition
        _write_markdown(source, source_metadata, source_body)

    return root


def make_catalog_corpus() -> Path:
    root = make_valid_corpus()
    shutil.copy2(SKILL_ROOT / "data" / "source-repositories.yaml", root / "data" / "source-repositories.yaml")

    source = source_path(root)
    source_metadata, source_body = _parse_markdown(source)
    source_metadata["repository_id"] = "vllm-ascend"
    _write_markdown(source, source_metadata, source_body)

    technique = card_path(root)
    technique_metadata, technique_body = _parse_markdown(technique)
    technique_metadata["version_sensitive"] = ["claim-current"]
    _write_markdown(technique, technique_metadata, technique_body)

    examples = yaml.safe_load((root / "examples.yaml").read_text(encoding="utf-8"))
    pattern_metadata = dict(technique_metadata)
    pattern_metadata.update(
        id="pattern-launch-bound",
        title="Launch-bound materialization",
        type="pattern",
        summary="Repeated launch and materialization boundaries can dominate otherwise small device work.",
        targets=["ascend910b"],
        target_match="exact",
        languages=["ascendc", "triton"],
        kernel_types=["reduction", "selection"],
        techniques=["launch-collapse"],
        hardware_features=["execution-pipeline"],
        candidate_techniques=["technique-kernel-fusion"],
        tags=["launch-bound", "launch-collapse", "materialization-overhead"],
        symptoms=["launch-bound", "materialization-overhead"],
        sources=[
            "source-ascendc-programming-model-cann-900beta1",
            "source-local-ascend-flexattention-round-003",
            "source-local-ascend-groupedtopk-round-001",
            "source-valid-manual",
        ],
        version_sensitive=["claim-stale"],
        observations=[{
            "id": "observation-launch-bound",
            "text": "Reviewed upstream code and documentation expose a repeated launch boundary.",
            "source_id": "source-valid-manual",
            "locator": "Reviewed fusion note",
            "evidence_level": "official-doc-and-upstream-code",
            "reproduction": "concept",
            "targets": ["ascend910b"],
            "target_match": "exact",
            "implementation_profile_id": None,
            "runtime_fingerprint": None,
            "versions": ["CANN 8.0"],
            "transfer_boundaries": ["the launch topology and target must match"],
        }],
        examples=[examples["positive"], examples["counterexample"], examples["capability_gap"]],
        related=[],
        prerequisites=[],
    )
    pattern_path = root / "wiki" / "patterns" / "launch-bound-materialization.md"
    pattern_path.parent.mkdir(parents=True, exist_ok=True)
    _write_markdown(
        pattern_path,
        pattern_metadata,
        technique_body.replace("# Kernel fusion", "# Launch-bound materialization", 1),
    )

    claims = {
        "schema_version": 1,
        "claims": [
            {
                "id": "claim-current",
                "card_ids": ["technique-kernel-fusion"],
                "status": "current",
                "supported_versions": ["CANN 9.0"],
                "last_verified_at": "2026-08-21",
                "source_ids": ["source-valid-manual"],
            },
            {
                "id": "claim-stale",
                "card_ids": ["pattern-launch-bound"],
                "status": "stale",
                "supported_versions": ["CANN 8.0"],
                "last_verified_at": "2026-08-20",
                "source_ids": ["source-valid-manual"],
            },
        ],
    }
    (root / "data" / "version-claims.yaml").write_text(
        yaml.safe_dump(claims, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return root


def remove_tree(root: Path) -> None:
    shutil.rmtree(root)

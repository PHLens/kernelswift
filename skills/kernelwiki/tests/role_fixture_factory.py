from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import yaml


SKILL_ROOT = Path(__file__).resolve().parents[1]
LOOP_ROOT = SKILL_ROOT.parent / "kernel-opt-loop"
VNEXT_FIXTURES = LOOP_ROOT / "tests" / "fixtures" / "vnext"


def _scripts_imports() -> None:
    import sys

    scripts = str(SKILL_ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)


def _sha256_file(path: Path) -> str:
    _scripts_imports()
    from kernelwiki_common import sha256_file

    return sha256_file(path)


def _replace_decision_markers(text: str, root: Path) -> str:
    replacements = {
        "__SKETCH_SHA256__": _sha256_file(root / "rounds" / "sketch_001.json"),
        "__PROFILE_SHA256__": _sha256_file(root / "state" / "implementation_profile_snapshot" / "profile.yaml"),
        "__CLAIM_SHA256__": _sha256_file(root / "state" / "project_capability_claim.json"),
    }
    for marker, digest in replacements.items():
        text = text.replace(marker, digest)
    return text


def materialize_vnext_project(root: Path) -> Path:
    root = Path(root)
    (root / "rounds").mkdir(parents=True)
    (root / "state").mkdir()
    (root / "baseline_adapter.py").write_text("class ModelNew: pass\n", encoding="utf-8")
    (root / "rounds" / "report_000.md").write_text("# Report 000\n", encoding="utf-8")
    (root / "project.md").write_text(
        "# Project\n\n## runtime-fingerprint\n\ntriton 3.6.0 / CoreX 4.4.0\n",
        encoding="utf-8",
    )
    shutil.copyfile(VNEXT_FIXTURES / "sketches" / "valid-kernel.json", root / "rounds" / "sketch_001.json")
    shutil.copytree(
        VNEXT_FIXTURES / "profiles" / "valid-partial",
        root / "state" / "implementation_profile_snapshot",
    )
    shutil.copyfile(
        VNEXT_FIXTURES / "claims" / "valid-claim.json",
        root / "state" / "project_capability_claim.json",
    )
    runtime = {
        "target_id": "mlu590",
        "implementation_profile_id": "triton_mlu",
        "triton_version": "3.6.0",
        "device_arch": "mlu-arch",
    }
    (root / "state" / "runtime-snapshot.json").write_text(
        json.dumps(runtime, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    decision_text = (VNEXT_FIXTURES / "decisions" / "valid-vnext.md").read_text(encoding="utf-8")
    decision_text = _replace_decision_markers(decision_text, root)
    (root / "rounds" / "decision_001.md").write_text(decision_text, encoding="utf-8")
    return root


def build_coder_context(project_root: Path) -> dict[str, Any]:
    _scripts_imports()
    from kernel_opt_bridge import compute_loop_contract_identity

    project_root = Path(project_root).resolve()
    refs = {
        "profile": project_root / "state" / "implementation_profile_snapshot" / "profile.yaml",
        "runtime_snapshot": project_root / "state" / "runtime-snapshot.json",
        "project_claim": project_root / "state" / "project_capability_claim.json",
        "project_document": project_root / "project.md",
        "sketch": project_root / "rounds" / "sketch_001.json",
        "decision": project_root / "rounds" / "decision_001.md",
    }
    identity = compute_loop_contract_identity()
    identity_document = {
        "repository_commit": identity.repository_commit,
        "skill_tree_sha": identity.skill_tree_sha,
        "validator_sha256": dict(identity.validator_sha256),
        "schema_sha256": dict(identity.schema_sha256),
    }
    return {
        "schema_version": 1,
        "role": "coder",
        "contract_version": 3,
        "target_id": "mlu590",
        "implementation_profile_id": "triton_mlu",
        "implementation_profile_status": "partial",
        "runtime_fingerprint": "triton 3.6.0 / CoreX 4.4.0",
        "languages": ["triton"],
        "dtypes": ["fp32"],
        "operator_tags": ["topk", "selection"],
        "kernel_types": ["topk", "reduction"],
        "semantic_features": ["left-tie-breaking"],
        "shape_signature": {"T": 83, "E": 256, "K": 8},
        "current_bottlenecks": ["reduction"],
        "project_root": str(project_root),
        "artifacts": {
            name: {"path": path.relative_to(project_root).as_posix(), "sha256": _sha256_file(path)}
            for name, path in refs.items()
        },
        "guidance_bindings": {"guidance-test-exact": ["op.load.row"]},
        "loop_contract_identity": identity_document,
    }


def write_coder_context(project_root: Path, destination: Path) -> Path:
    document = build_coder_context(project_root)
    destination = Path(destination)
    destination.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return destination


def _plain(value: Any) -> Any:
    from collections.abc import Mapping

    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def role_context_document(context: Any) -> dict[str, Any]:
    identity = context.loop_contract_identity
    return {
        "schema_version": context.schema_version,
        "contract_version": context.contract_version,
        "role": context.role,
        "target_id": context.target_id,
        "implementation_profile_id": context.implementation_profile_id,
        "implementation_profile_status": context.implementation_profile_status,
        "runtime_fingerprint": context.runtime_fingerprint,
        "languages": list(context.languages),
        "dtypes": list(context.dtypes),
        "operator_tags": list(context.operator_tags),
        "kernel_types": list(context.kernel_types),
        "semantic_features": list(context.semantic_features),
        "shape_signature": _plain(context.shape_signature),
        "current_bottlenecks": list(context.current_bottlenecks),
        "project_root": None if context.project_root is None else str(context.project_root),
        "artifacts": {
            name: {"path": reference.path.as_posix(), "sha256": reference.sha256}
            for name, reference in context.artifacts.items()
        },
        "guidance_bindings": {key: list(value) for key, value in context.guidance_bindings.items()},
        "loop_contract_identity": None
        if identity is None
        else {
            "repository_commit": identity.repository_commit,
            "skill_tree_sha": identity.skill_tree_sha,
            "validator_sha256": dict(identity.validator_sha256),
            "schema_sha256": dict(identity.schema_sha256),
        },
    }


def write_role_context_variant(context: Any, destination: Path, **changes: Any) -> Path:
    document = role_context_document(context)
    document.update(changes)
    destination = Path(destination)
    destination.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    return destination


def _read_markdown(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    marker = text.find("\n---\n", 4)
    return yaml.safe_load(text[4:marker]), text[marker + 5 :]


def _write_markdown(path: Path, metadata: dict[str, Any], body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True).rstrip()
    path.write_text(f"---\n{frontmatter}\n---\n{body}", encoding="utf-8")


def materialize_exact_role_corpus(
    corpus_root: Path,
    sketch_result: dict[str, Any] | Any,
    decision_result: dict[str, Any] | Any,
) -> Path:
    _scripts_imports()
    from admission import build_exact_guidance

    corpus_root = Path(corpus_root)
    fixture_root = Path(__file__).resolve().parent / "fixtures"
    shutil.copyfile(SKILL_ROOT / "data" / "source-repositories.yaml", corpus_root / "data" / "source-repositories.yaml")
    exact_source = corpus_root / "sources" / "commits" / "source-exact-coder.md"
    analogy_source = corpus_root / "sources" / "docs" / "source-analogy-only.md"
    exact_source.parent.mkdir(parents=True, exist_ok=True)
    analogy_source.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(fixture_root / "sources" / "source-exact-coder.md", exact_source)
    shutil.copyfile(fixture_root / "sources" / "source-analogy-only.md", analogy_source)

    card_fixture = fixture_root / "cards" / "mixed-asset-card.md"
    metadata, body = _read_markdown(card_fixture)
    metadata["coder_access"] = {
        "page": "exact-profile",
        "guidance": [build_exact_guidance(sketch_result, decision_result)],
    }
    card_path = corpus_root / "wiki" / "languages" / "mixed-asset-card.md"
    _write_markdown(card_path, metadata, body)
    shutil.copyfile(
        fixture_root / "cards" / "analogy-designer-card.md",
        corpus_root / "wiki" / "languages" / "analogy-designer-card.md",
    )

    artifact_dir = corpus_root / "artifacts" / "source-exact-coder"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    full_asset = artifact_dir / "asset-full-kernel"
    snippet_asset = artifact_dir / "asset-short-snippet"
    full_asset.write_text("def full_kernel():\n    return 'designer-only full implementation'\n", encoding="utf-8")
    snippet_asset.write_text("value = load_row(pointer)\n", encoding="utf-8")
    provenance = {
        "schema_version": 1,
        "origin_url": "https://github.com/Ascend/triton-ascend/commit/1111111111111111111111111111111111111111",
        "upstream_repo": "Ascend/triton-ascend",
        "upstream_sha": "1111111111111111111111111111111111111111",
        "license_state": "approved",
        "retrieved_at": "2026-08-21T00:00:00Z",
        "asset_mode": "verbatim",
        "allowed_audiences": ["coder", "designer"],
        "coder_access": "exact-profile",
        "source_ids": ["source-exact-coder"],
        "files": [
            {
                "local_path": "asset-full-kernel",
                "upstream_path": "python/asset-full-kernel",
                "heading_path": None,
                "role": "upstream-file",
                "mode": "verbatim",
                "sha256": _sha256_file(full_asset),
            },
            {
                "local_path": "asset-short-snippet",
                "upstream_path": "python/asset-short-snippet",
                "heading_path": None,
                "role": "snippet",
                "mode": "verbatim",
                "sha256": _sha256_file(snippet_asset),
            },
        ],
    }
    (artifact_dir / "PROVENANCE.yaml").write_text(
        yaml.safe_dump(provenance, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    return card_path

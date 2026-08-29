from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Sequence

from catalog import assert_generated_outputs_current, build_generated_outputs
from corpus import Corpus, load_corpus, validate_corpus
from kernelwiki_common import KernelWikiError, require_within, run_cli
from provenance import load_provenance, validate_provenance, validate_size_budget
from validate_lift import validate_experience_tree


def validate_artifact_bundles(corpus: Corpus) -> None:
    for source_id in sorted(corpus.sources):
        source = corpus.sources[source_id]
        artifact_dir = source.metadata.get("artifact_dir")
        if artifact_dir is None:
            continue
        bundle_dir = require_within(corpus.root, corpus.root / str(artifact_dir))
        manifest_path = bundle_dir / "PROVENANCE.yaml"
        if not bundle_dir.is_dir() or not manifest_path.is_file():
            raise KernelWikiError(
                "provenance-missing",
                f"Source {source_id} requires {manifest_path.relative_to(corpus.root).as_posix()}",
                source.path,
            )
        bundle = load_provenance(manifest_path)
        validate_provenance(bundle, corpus.root)
        if source_id not in bundle.source_ids:
            raise KernelWikiError(
                "provenance-source-mismatch",
                f"Source {source_id} is absent from provenance source_ids",
                manifest_path,
            )
    validate_size_budget(corpus.root)


def validate_coder_access_assets(corpus: Corpus) -> None:
    for card in corpus.cards.values():
        access = card.metadata.get("coder_access")
        if access is None:
            continue
        asset_owners: dict[str, str] = {}
        for source_id in card.metadata["sources"]:
            source = corpus.sources[source_id]
            artifact_dir = source.metadata.get("artifact_dir")
            if artifact_dir is None:
                continue
            bundle_dir = require_within(corpus.root, corpus.root / str(artifact_dir))
            bundle = load_provenance(bundle_dir / "PROVENANCE.yaml")
            validate_provenance(bundle, corpus.root)
            for item in bundle.files:
                previous = asset_owners.setdefault(item.local_path, source_id)
                if previous != source_id:
                    raise KernelWikiError(
                        "coder-asset-id-duplicate",
                        f"asset ID {item.local_path!r} is provided by multiple Card Sources",
                        card.path,
                    )
        eligible_ids = {
            asset_id
            for guidance in access["guidance"]
            for asset_id in guidance["eligible_asset_ids"]
        }
        missing = sorted(eligible_ids - asset_owners.keys())
        if missing:
            raise KernelWikiError(
                "coder-asset-missing",
                f"unknown eligible assets: {', '.join(missing)}",
                card.path,
            )


def _validated_root_path(root: Path) -> Path:
    try:
        candidate = Path(root)
    except (TypeError, ValueError) as error:
        raise KernelWikiError("validation-root-invalid", "skill root must be a valid path") from error
    try:
        if "\x00" in str(candidate):
            raise KernelWikiError("validation-root-invalid", "skill root contains an embedded NUL", candidate)
        if candidate.is_symlink():
            raise KernelWikiError("validation-root-invalid", "skill root must not be a symlink", candidate)
        resolved = candidate.resolve()
        if not resolved.is_dir():
            raise KernelWikiError("validation-root-invalid", "skill root must be a directory", candidate)
        return resolved
    except KernelWikiError:
        raise
    except (OSError, RuntimeError, ValueError) as error:
        raise KernelWikiError("validation-root-invalid", "skill root could not be resolved", candidate) from error


def validate_skill_root(root: Path, *, check_generated: bool = True) -> Corpus:
    root = _validated_root_path(root)
    corpus = load_corpus(root)
    validate_corpus(corpus)
    validate_artifact_bundles(corpus)
    validate_coder_access_assets(corpus)
    if check_generated:
        assert_generated_outputs_current(corpus.root, build_generated_outputs(corpus))
    validate_experience_tree(corpus.root)
    return corpus


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    corpus = validate_skill_root(args.root)
    print(
        json.dumps(
            {
                "schema_version": 1,
                "valid": True,
                "sources": len(corpus.sources),
                "cards": len(corpus.cards),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))

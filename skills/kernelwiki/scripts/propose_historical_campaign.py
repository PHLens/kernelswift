from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from historical_capture import build_historical_proposal, load_historical_manifest, write_historical_proposal
from kernelwiki_common import KernelWikiError, canonical_json_bytes, run_cli


SKILL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = SKILL_ROOT / "candidates" / "experience"


def _within(root: Path, path: Path) -> bool:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def _validated_output(path: Path) -> Path:
    output = (path if path.is_absolute() else Path.cwd() / path).resolve()
    if _within(SKILL_ROOT, output) and not _within(DEFAULT_OUTPUT_ROOT, output):
        raise KernelWikiError("proposal-output-forbidden", "KernelWiki output must stay under candidates/experience", output)
    if "kernels" in output.parts or any(part.lower() in {"state", "rounds", "campaign", "campaigns"} for part in output.parts):
        raise KernelWikiError("proposal-output-forbidden", "historical proposal must not target a campaign path", output)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a Designer-only proposal from historical local campaign evidence")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repository-root", type=Path)
    return parser


def _main(argv: Sequence[str]) -> int:
    arguments = _parser().parse_args(argv)
    manifest = load_historical_manifest(arguments.manifest, repository_root=arguments.repository_root)
    proposal = build_historical_proposal(manifest)
    output = _validated_output(arguments.output or DEFAULT_OUTPUT_ROOT / f"{proposal.proposal_id}.json")
    proposal_sha256 = write_historical_proposal(proposal, output)
    print(canonical_json_bytes({"output_path": str(output), "proposal_sha256": proposal_sha256}).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(_main))

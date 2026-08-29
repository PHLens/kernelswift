from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from campaign_import import load_terminal_bundle, validate_campaign
from experience import build_experience_proposal, write_proposal
from kernelwiki_common import KernelWikiError, canonical_json_bytes, run_cli


SKILL_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = SKILL_ROOT / "candidates" / "experience"


def _within(root: Path, path: Path) -> bool:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def _validated_output(path: Path, *, repository_root: Path, project_root: Path) -> Path:
    output = path if path.is_absolute() else Path.cwd() / path
    output = output.resolve()
    if _within(repository_root, output) or _within(project_root, output):
        raise KernelWikiError("proposal-output-forbidden", "output must not be inside the selected campaign repository", output)
    if any(part.lower() in {"state", "campaign", "campaigns"} for part in output.parts):
        raise KernelWikiError("proposal-output-forbidden", "output must not target a campaign or state path", output)
    if _within(SKILL_ROOT, output) and not _within(DEFAULT_OUTPUT_ROOT, output):
        raise KernelWikiError("proposal-output-forbidden", "KernelWiki output must remain under candidates/experience", output)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a review-only KernelWiki proposal from a strict terminal bundle")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser


def _main(argv: Sequence[str]) -> int:
    arguments = _parser().parse_args(argv)
    bundle = load_terminal_bundle(arguments.bundle)
    validated = validate_campaign(bundle)
    proposal = build_experience_proposal(validated)
    requested = arguments.output or DEFAULT_OUTPUT_ROOT / f"{proposal.proposal_id}.json"
    output = _validated_output(requested, repository_root=bundle.repository_root, project_root=bundle.project_root)
    proposal_sha256 = write_proposal(proposal, output)
    print(canonical_json_bytes({"output_path": str(output), "proposal_sha256": proposal_sha256}).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(_main))

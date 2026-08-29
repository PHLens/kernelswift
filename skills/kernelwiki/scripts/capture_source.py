from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from kernelwiki_common import KernelWikiError, canonical_json_bytes, run_cli
from source_capture import (
    GitHubClient,
    GitHubCommitCaptureRequest,
    GitHubPRCaptureRequest,
    ManualCaptureRequest,
    StableArgumentParser,
    capture_github_commit,
    capture_github_pr,
    capture_manual_source,
    capture_result_document,
    load_github_capture_manifest,
    main as discovery_main,
    validate_capture_skill_root,
)


def main(argv: Sequence[str]) -> int:
    parser = StableArgumentParser(description="Discover or capture immutable KernelWiki Source evidence")
    subparsers = parser.add_subparsers(
        dest="command", required=True, parser_class=StableArgumentParser
    )

    discover = subparsers.add_parser("discover")
    discover.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    discover.add_argument("--repository", required=True)
    discover.add_argument("--term", action="append", dest="terms")
    discover.add_argument("--limit", type=int, default=100)

    github_pr = subparsers.add_parser("github-pr")
    github_pr.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    github_pr.add_argument("--metadata", type=Path, required=True)
    github_pr.add_argument("--repo", required=True)
    github_pr.add_argument("--pr", type=int, required=True)

    github_commit = subparsers.add_parser("github-commit")
    github_commit.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    github_commit.add_argument("--metadata", type=Path, required=True)
    github_commit.add_argument("--repo", required=True)
    github_commit.add_argument("--sha", required=True)

    manual = subparsers.add_parser("manual")
    manual.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    manual.add_argument("--metadata", type=Path, required=True)

    args = parser.parse_args(list(argv))
    root = validate_capture_skill_root(args.root)
    if args.command == "discover":
        delegated = [
            "discover",
            "--root",
            str(root),
            "--repository",
            args.repository,
            "--limit",
            str(args.limit),
        ]
        for term in args.terms or ():
            delegated.extend(("--term", term))
        return discovery_main(delegated)
    if args.command == "github-pr":
        metadata, selections = load_github_capture_manifest(args.metadata)
        result = capture_github_pr(
            GitHubPRCaptureRequest(root, metadata, args.repo, args.pr, selections),
            GitHubClient(),
        )
    elif args.command == "github-commit":
        metadata, selections = load_github_capture_manifest(args.metadata)
        result = capture_github_commit(
            GitHubCommitCaptureRequest(root, metadata, args.repo, args.sha, selections),
            GitHubClient(),
        )
    elif args.command == "manual":
        result = capture_manual_source(ManualCaptureRequest(root, args.metadata))
    else:
        raise KernelWikiError("capture-command-invalid", f"unsupported command {args.command}")
    print(canonical_json_bytes(capture_result_document(result, root)).decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli(main))

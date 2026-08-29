from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from kernelwiki_common import KernelWikiError, canonical_json_bytes, run_cli
from search import FILTER_FIELDS, parse_query_request, query_payload
from validate import validate_skill_root


class StableArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise KernelWikiError("cli-input-invalid", message)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("expected a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _parser() -> StableArgumentParser:
    parser = StableArgumentParser(description="Search the local KernelWiki corpus")
    parser.add_argument("text", nargs="?", default="")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--scope", choices=("cards", "sources", "both"), default="both")
    parser.add_argument("--limit", type=_positive_int, default=20)
    parser.add_argument("--format", choices=("json", "markdown"), default="json")
    parser.add_argument("--profile-snapshot", type=Path)
    parser.add_argument("--type", action="append")
    parser.add_argument("--tag", action="append")
    parser.add_argument("--repository", "--repo", dest="repository", action="append")
    parser.add_argument("--language", action="append")
    parser.add_argument("--target", action="append")
    parser.add_argument("--target-match", action="append")
    parser.add_argument("--symptom", action="append")
    parser.add_argument("--kernel-type", action="append")
    parser.add_argument("--evidence-level", action="append")
    parser.add_argument("--reproduction", action="append")
    parser.add_argument("--audience", action="append")
    parser.add_argument("--has-code", choices=("true", "false"), action="append")
    return parser


def _filters(args: argparse.Namespace) -> dict[str, tuple[str, ...]]:
    values = {}
    for field in FILTER_FIELDS:
        attribute = field.replace("-", "_")
        selected = getattr(args, attribute)
        if selected:
            values[field] = tuple(sorted(set(selected)))
    return values


def _markdown(payload) -> str:
    lines = [f"# KernelWiki search: {payload['query'] or '(filters only)'}", ""]
    if not payload["results"]:
        return "\n".join([*lines, "- _No matches._", ""])
    for result in payload["results"]:
        lines.append(
            f"- [{result['record_id']}]({result['path']}) — {result['title']} — {result['excerpt']}"
        )
    return "\n".join([*lines, ""])


def _main(argv: Sequence[str]) -> int:
    args = _parser().parse_args(list(argv))
    if args.profile_snapshot is not None:
        raise KernelWikiError(
            "phase-c-required",
            "--profile-snapshot requires Phase C role-aware admission",
            args.profile_snapshot,
        )
    corpus = validate_skill_root(args.root)
    request = parse_query_request(args.text, _filters(args), args.scope, args.limit)
    payload = query_payload(corpus, request)
    if args.format == "markdown":
        print(_markdown(payload), end="")
    else:
        print(canonical_json_bytes(payload).decode("utf-8"), end="")
    return 0


def main(argv: Sequence[str]) -> int:
    return run_cli(_main, argv)


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))

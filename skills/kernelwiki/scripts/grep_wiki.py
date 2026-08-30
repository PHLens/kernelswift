from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from kernelwiki_common import KernelWikiError, canonical_json_bytes, run_cli
from search import grep_corpus, grep_payload
from validate import validate_skill_root


class StableArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise KernelWikiError("cli-input-invalid", message)


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("expected a nonnegative integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("expected a nonnegative integer")
    return parsed


def _positive_int(value: str) -> int:
    parsed = _nonnegative_int(value)
    if parsed == 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def _parser() -> StableArgumentParser:
    parser = StableArgumentParser(description="Regex search the local KernelWiki corpus")
    parser.add_argument("pattern")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--scope", choices=("wiki", "sources", "both"), default="both")
    parser.add_argument("--max-matches", type=_positive_int, default=100)
    parser.add_argument("--context-chars", type=_nonnegative_int, default=80)
    return parser


def _main(argv: Sequence[str]) -> int:
    args = _parser().parse_args(list(argv))
    corpus = validate_skill_root(args.root)
    matches = grep_corpus(
        corpus,
        args.pattern,
        scope=args.scope,
        max_matches=args.max_matches,
        context_chars=args.context_chars,
    )
    print(canonical_json_bytes(grep_payload(args.pattern, args.scope, matches)).decode("utf-8"), end="")
    return 0


def main(argv: Sequence[str]) -> int:
    return run_cli(_main, argv)


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))

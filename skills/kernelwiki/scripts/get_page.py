from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from kernelwiki_common import KernelWikiError, canonical_json_bytes, run_cli
from search import page_payload, retrieve_page
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
    parser = StableArgumentParser(description="Retrieve one local KernelWiki Card or Source")
    parser.add_argument("record")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--frontmatter", action="store_true")
    parser.add_argument("--follow-sources", action="store_true")
    parser.add_argument("--source-excerpt-lines", type=_positive_int, default=40)
    parser.add_argument("--include-code", action="store_true")
    return parser


def _main(argv: Sequence[str]) -> int:
    args = _parser().parse_args(list(argv))
    corpus = validate_skill_root(args.root)
    page = retrieve_page(
        corpus,
        args.record,
        follow_sources=args.follow_sources,
        access="approved-assets" if args.include_code else "metadata",
    )
    if args.frontmatter:
        payload = dict(page.metadata)
    else:
        payload = page_payload(page)
        for source in payload["followed_sources"]:
            source["body"] = "\n".join(source["body"].splitlines()[: args.source_excerpt_lines])
    print(canonical_json_bytes(payload).decode("utf-8"), end="")
    return 0


def main(argv: Sequence[str]) -> int:
    return run_cli(_main, argv)


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from kernelwiki_common import KernelWikiError, canonical_json_bytes, run_cli
from role_context import load_authority_snapshot, load_role_context
from role_search import role_get_page
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
    parser.add_argument("--context", type=Path)
    parser.add_argument("--example", action="append", default=[])
    parser.add_argument("--guidance", action="append", default=[])
    parser.add_argument("--asset", action="append", default=[])
    return parser


def _main(argv: Sequence[str]) -> int:
    args = _parser().parse_args(list(argv))
    corpus = validate_skill_root(args.root)
    access = "approved-assets" if args.include_code else "metadata"
    if args.context is None:
        if args.example or args.guidance or args.asset:
            raise KernelWikiError("cli-input-invalid", "--example/--guidance/--asset require --context")
        page = retrieve_page(
            corpus,
            args.record,
            follow_sources=args.follow_sources,
            access=access,
        )
        if args.frontmatter:
            payload = dict(page.metadata)
        else:
            payload = page_payload(page)
            for source in payload["followed_sources"]:
                source["body"] = "\n".join(source["body"].splitlines()[: args.source_excerpt_lines])
    else:
        context = load_role_context(args.context)
        authority = (
            load_authority_snapshot(context)
            if context.role == "coder" and context.implementation_profile_status != "missing"
            else None
        )
        payload = dict(
            role_get_page(
                corpus,
                args.record,
                context,
                authority,
                follow_sources=args.follow_sources,
                access=access,
                example_ids=tuple(sorted(set(args.example))),
                guidance_ids=tuple(sorted(set(args.guidance))),
                asset_ids=tuple(sorted(set(args.asset))),
            )
        )
        page_payload_value = payload["page"]
        for source in page_payload_value["followed_sources"]:
            source["body"] = "\n".join(source["body"].splitlines()[: args.source_excerpt_lines])
        if args.frontmatter:
            page_payload_value["body"] = ""
            page_payload_value["followed_sources"] = []
    print(canonical_json_bytes(payload).decode("utf-8"), end="")
    return 0


def main(argv: Sequence[str]) -> int:
    return run_cli(_main, argv)


if __name__ == "__main__":
    raise SystemExit(main(__import__("sys").argv[1:]))

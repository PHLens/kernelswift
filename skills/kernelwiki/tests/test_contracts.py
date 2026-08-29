from __future__ import annotations

import ast
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import sys
import time
import unittest
from unittest.mock import patch
import urllib.request

TESTS = Path(__file__).resolve().parent
SKILL_ROOT = TESTS.parent
SCRIPTS = SKILL_ROOT / "scripts"
REPOSITORY_ROOT = SKILL_ROOT.parents[1]
sys.path.insert(0, str(SCRIPTS))

from catalog import (  # noqa: E402
    GENERATED_OUTPUT_PATHS,
    assert_generated_outputs_current,
    build_generated_outputs,
)
from corpus import load_corpus, validate_corpus  # noqa: E402
from get_page import main as get_page_main  # noqa: E402
from grep_wiki import main as grep_main  # noqa: E402
from kernelwiki_common import load_yaml_document, sha256_file  # noqa: E402
from provenance import load_provenance, validate_provenance, validate_size_budget  # noqa: E402
from query import main as query_main  # noqa: E402
from validate import validate_artifact_bundles, validate_skill_root  # noqa: E402


NETWORK_IMPORTS = frozenset({"urllib", "http", "requests", "socket"})
ROLE_NEUTRAL_MODULES = (
    "query.py",
    "get_page.py",
    "grep_wiki.py",
    "search.py",
    "catalog.py",
    "corpus.py",
    "provenance.py",
    "kernelwiki_common.py",
    "source_capture.py",
    "capture_source.py",
    "generate_indices.py",
    "validate.py",
    "validate_provenance.py",
)
MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def imported_top_level_names(path: Path) -> frozenset[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".", 1)[0])
    return frozenset(names)


def run_main(main, argv: list[str]) -> tuple[int, str, str]:
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        result = main(argv)
    return result, stdout.getvalue(), stderr.getvalue()


def subprocess_output(*arguments: str) -> str:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"command failed ({completed.returncode}): {arguments}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    if completed.stderr:
        raise AssertionError(f"command emitted stderr: {arguments}\n{completed.stderr}")
    return completed.stdout


class StandaloneContractTests(unittest.TestCase):
    def test_real_corpus_validates_and_generated_files_are_current(self):
        corpus = load_corpus(SKILL_ROOT)
        validate_corpus(corpus)
        validate_artifact_bundles(corpus)
        expected = build_generated_outputs(corpus)
        self.assertEqual(set(GENERATED_OUTPUT_PATHS), set(expected))
        assert_generated_outputs_current(corpus.root, expected)
        validated = validate_skill_root(SKILL_ROOT)
        self.assertEqual(set(corpus.cards), set(validated.cards))
        self.assertEqual(set(corpus.sources), set(validated.sources))



    def test_generated_outputs_are_byte_deterministic(self):
        corpus = load_corpus(SKILL_ROOT)
        validate_corpus(corpus)
        first = build_generated_outputs(corpus)
        second = build_generated_outputs(load_corpus(SKILL_ROOT))
        self.assertEqual(first, second)
        for relative_path, expected in first.items():
            self.assertEqual(expected, (SKILL_ROOT / relative_path).read_bytes(), relative_path)


    def test_role_neutral_modules_do_not_load_kernel_opt_loop(self):
        for name in ROLE_NEUTRAL_MODULES:
            path = SCRIPTS / name
            text = path.read_text(encoding="utf-8")
            imports = imported_top_level_names(path)
            self.assertNotIn("kernel_opt_loop", imports, name)
            self.assertNotIn("kernel-opt-loop", text, name)
            self.assertNotIn("kernel_opt_loop", text, name)

    def test_track2_names_are_not_card_ids_or_paths(self):
        manifest = load_yaml_document(SKILL_ROOT / "data" / "evaluation-holdouts.yaml")
        names = {
            *manifest["track2"]["development_contexts"],
            *manifest["track2"]["holdout_contexts"],
        }
        forbidden = names | {name.replace("_", "-") for name in names}
        corpus = load_corpus(SKILL_ROOT)
        for card in corpus.cards.values():
            identity = {card.card_id, *card.path.relative_to(SKILL_ROOT).parts}
            self.assertTrue(forbidden.isdisjoint(identity), (card.card_id, forbidden & identity))
        self.assertTrue(
            all(source.metadata["repository_id"] != manifest["repository_holdout"]["repository_id"]
                for source in corpus.sources.values())
        )


    def test_production_smoke_commands_exit_zero_and_are_deterministic(self):
        validate_output = subprocess_output("skills/kernelwiki/scripts/validate.py")
        corpus = load_corpus(SKILL_ROOT)
        self.assertEqual(
            {
                "cards": len(corpus.cards),
                "schema_version": 1,
                "sources": len(corpus.sources),
                "valid": True,
            },
            json.loads(validate_output),
        )
        generated_output = subprocess_output(
            "skills/kernelwiki/scripts/generate_indices.py", "--check"
        )
        self.assertEqual(
            {"checked": 10, "root": ".", "schema_version": 1},
            json.loads(generated_output),
        )

        commands = (
            ("skills/kernelwiki/scripts/query.py", "ascend launch overhead", "--limit", "5"),
            ("skills/kernelwiki/scripts/get_page.py", "technique-kernel-fusion", "--follow-sources"),
            ("skills/kernelwiki/scripts/grep_wiki.py", "device.*wall", "--scope", "wiki"),
        )
        for command in commands:
            with self.subTest(command=command[0]):
                first = subprocess_output(*command)
                second = subprocess_output(*command)
                self.assertEqual(first, second)
                payload = json.loads(first)
                self.assertEqual(1, payload["schema_version"])

    def test_query_page_and_grep_are_offline(self):
        with (
            patch.object(socket, "socket", side_effect=AssertionError("network forbidden")),
            patch.object(urllib.request, "urlopen", side_effect=AssertionError("network forbidden")),
        ):
            for main, argv in (
                (query_main, ["ascend launch overhead", "--limit", "5"]),
                (get_page_main, ["technique-kernel-fusion", "--follow-sources"]),
                (grep_main, ["device.*wall", "--scope", "wiki"]),
            ):
                with self.subTest(main=main.__module__):
                    result, stdout, stderr = run_main(main, argv)
                    self.assertEqual(0, result)
                    self.assertEqual("", stderr)
                    self.assertEqual(1, json.loads(stdout)["schema_version"])



if __name__ == "__main__":
    unittest.main()

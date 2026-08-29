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
import tempfile
import time
import unittest
from unittest.mock import patch
import urllib.request

TESTS = Path(__file__).resolve().parent
SKILL_ROOT = TESTS.parent
SCRIPTS = SKILL_ROOT / "scripts"
REPOSITORY_ROOT = SKILL_ROOT.parents[1]
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(TESTS))

from catalog import (  # noqa: E402
    GENERATED_OUTPUT_PATHS,
    assert_generated_outputs_current,
    build_generated_outputs,
    write_generated_outputs,
)
from corpus import load_corpus, validate_corpus  # noqa: E402
from fixture_factory import make_valid_corpus, remove_tree  # noqa: E402
from get_page import main as get_page_main  # noqa: E402
from grep_wiki import main as grep_main  # noqa: E402
from kernelwiki_common import KernelWikiError, load_yaml_document, sha256_file  # noqa: E402
from provenance import load_provenance, validate_provenance, validate_size_budget  # noqa: E402
from propose_from_campaign import _validated_output  # noqa: E402
from query import main as query_main  # noqa: E402
from role_context import load_authority_snapshot, load_role_context  # noqa: E402
from role_fixture_factory import (  # noqa: E402
    materialize_exact_role_corpus,
    materialize_vnext_project,
    write_coder_context,
)
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

    def test_phase_c_creates_no_loop_or_campaign_integration(self):
        loop_tree = subprocess.check_output(
            ["git", "rev-parse", "HEAD:skills/kernel-opt-loop"],
            cwd=REPOSITORY_ROOT,
            text=True,
        ).strip()
        branch_base = subprocess.check_output(
            ["git", "merge-base", "HEAD", "origin/dev"],
            cwd=REPOSITORY_ROOT,
            text=True,
        ).strip()
        base_loop_tree = subprocess.check_output(
            ["git", "rev-parse", f"{branch_base}:skills/kernel-opt-loop"],
            cwd=REPOSITORY_ROOT,
            text=True,
        ).strip()
        relative_paths = tuple(
            path.relative_to(SKILL_ROOT).as_posix().lower()
            for path in SKILL_ROOT.rglob("*")
        )
        checks = (
            ("consultation validator", not (SCRIPTS / "validate_consultation.py").exists()),
            ("consultation artifact", not tuple(SKILL_ROOT.rglob("kernelwiki_consultation_*.json"))),
            ("coder_result schema change", loop_tree == base_loop_tree),
            ("Designer/Coder prompt edit", loop_tree == base_loop_tree),
            ("kernel-opt-loop change", loop_tree == base_loop_tree),
            ("KnowledgePacket path", all("knowledgepacket" not in path for path in relative_paths)),
            ("required dossier path", all("dossier" not in path for path in relative_paths)),
            (
                "campaign/state write path",
                all(not (SKILL_ROOT / name).exists() for name in ("campaigns", "rounds", "state")),
            ),
        )
        for label, passed in checks:
            with self.subTest(contract=label):
                self.assertTrue(passed, label)

    def test_role_query_receipts_bind_context_authority_and_guidance(self):
        designer_context = TESTS / "fixtures" / "role" / "designer-context.json"
        exit_code, stdout, stderr = run_main(
            query_main,
            ["ascend launch", "--root", str(SKILL_ROOT), "--context", str(designer_context), "--limit", "2"],
        )
        self.assertEqual(0, exit_code)
        self.assertEqual("", stderr)
        designer_receipt = json.loads(stdout)
        self.assertEqual(1, designer_receipt["schema_version"])
        self.assertRegex(designer_receipt["context_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual({}, designer_receipt["authority_hashes"])
        self.assertIsNone(designer_receipt["loop_contract_identity"])

        root = make_valid_corpus()
        try:
            project = materialize_vnext_project(root / "role-project")
            context_path = write_coder_context(project, root / "coder-context.json")
            context = load_role_context(context_path)
            authority = load_authority_snapshot(context)
            materialize_exact_role_corpus(root, authority.sketch_result, authority.decision_result)
            write_generated_outputs(root)
            receipt_path = root / "exact-coder-receipt.json"
            exit_code, stdout, stderr = run_main(
                query_main,
                [
                    "exact profile mixed assets",
                    "--root",
                    str(root),
                    "--context",
                    str(context_path),
                    "--scope",
                    "cards",
                    "--limit",
                    "5",
                    "--output",
                    str(receipt_path),
                ],
            )
            self.assertEqual(0, exit_code)
            self.assertEqual("", stdout)
            self.assertEqual("", stderr)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            context_document = json.loads(context_path.read_text(encoding="utf-8"))
            self.assertEqual(1, receipt["schema_version"])
            self.assertRegex(receipt["context_sha256"], r"^[0-9a-f]{64}$")
            self.assertEqual(
                {name: reference["sha256"] for name, reference in context_document["artifacts"].items()},
                receipt["authority_hashes"],
            )
            self.assertEqual(3, context_document["contract_version"])
            self.assertEqual(
                context_document["loop_contract_identity"]["skill_tree_sha"],
                receipt["loop_contract_identity"]["skill_tree_sha"],
            )
            item = next(entry for entry in receipt["groups"]["admitted"] if entry["id"] == "language-mixed-asset")
            self.assertIn("guidance-test-exact", item["admission"]["admitted_guidance_ids"])
            self.assertEqual(["op.load.row"], context_document["guidance_bindings"]["guidance-test-exact"])
        finally:
            remove_tree(root)

    def test_reviewed_examples_preserve_contradiction_visibility(self):
        corpus = load_corpus(SKILL_ROOT)
        validate_corpus(corpus)
        fusion = corpus.cards["technique-kernel-fusion"]
        mismatch = corpus.cards["pattern-device-win-wall-loss"]
        self.assertEqual(
            {"source-local-ascend-groupedtopk-round-001"},
            {item["source_id"] for item in fusion.metadata["examples"] if item["role"] == "positive"},
        )
        self.assertEqual(
            {"source-local-ascend-flexattention-round-003"},
            {item["source_id"] for item in mismatch.metadata["examples"] if item["role"] == "counterexample"},
        )
        catalog = {
            item["id"]: item
            for item in (
                json.loads(line)
                for line in (SKILL_ROOT / "compiled" / "catalog.jsonl").read_text(encoding="utf-8").splitlines()
            )
        }
        self.assertEqual(1, catalog["technique-kernel-fusion"]["positive_example_count"])
        self.assertEqual(1, catalog["pattern-device-win-wall-loss"]["counterexample_count"])
        for query, expected in (
            ("reviewed historical fusion", "technique-kernel-fusion"),
            ("device win wall loss", "pattern-device-win-wall-loss"),
        ):
            with self.subTest(query=query):
                _, stdout, _ = run_main(query_main, [query, "--root", str(SKILL_ROOT), "--scope", "cards", "--limit", "5"])
                self.assertIn(expected, [item["record_id"] for item in json.loads(stdout)["results"]])

    def test_two_local_campaign_holdout_rows_surface_generic_categories(self):
        candidate_root = SKILL_ROOT / "candidates" / "experience"
        decisions = {
            path.stem: load_yaml_document(path)["decision"]
            for path in sorted((candidate_root / "reviews").glob("*.yaml"))
        }
        self.assertEqual(2, tuple(decisions.values()).count("include"))
        self.assertEqual(1, tuple(decisions.values()).count("defer"))
        holdout = load_yaml_document(SKILL_ROOT / "data" / "local-campaign-holdout.yaml")
        rows = (
            (
                "kernels/track1-triton/mm_encoder_attention/ascend",
                "layout materialization kernel count wall time",
                {"pattern-launch-bound-materialization"},
            ),
            (
                "kernels/track1-triton/sparse_pooler/ascend",
                "output allocation reuse wall time",
                {"pattern-device-win-wall-loss", "pattern-launch-bound-materialization"},
            ),
        )
        self.assertEqual([item[0] for item in rows], holdout["holdout_campaigns"])
        for campaign, query, expected in rows:
            with self.subTest(campaign=campaign):
                _, stdout, stderr = run_main(
                    query_main,
                    [query, "--root", str(SKILL_ROOT), "--scope", "cards", "--limit", "5"],
                )
                self.assertEqual("", stderr)
                ids = {item["record_id"] for item in json.loads(stdout)["results"]}
                self.assertTrue(expected <= ids)

    def test_final_lift_nonintegration_and_output_paths(self):
        historical_text = (SCRIPTS / "historical_capture.py").read_text(encoding="utf-8")
        capture_text = (SCRIPTS / "capture_source.py").read_text(encoding="utf-8")
        checks = (
            ("no Orchestrator hook", "orchestrator" not in historical_text.lower()),
            ("no final-stop callback", "final_stop_callback" not in historical_text),
            ("no consultation record", "consultation" not in historical_text.lower()),
            ("no automatic Card publisher", "write_generated_outputs" not in historical_text and "wiki/" not in historical_text),
            ("no Coder auto-promotion", '"coder"' not in historical_text),
            ("explicit reviewed Source command", '"reviewed-historical"' in capture_text),
        )
        for label, passed in checks:
            with self.subTest(contract=label):
                self.assertTrue(passed, label)

        with tempfile.TemporaryDirectory(prefix="kernelwiki-output-") as temporary:
            root = Path(temporary)
            repository = root / "repository"
            project = repository / "project"
            project.mkdir(parents=True)
            valid = root / "review-output" / "proposal.json"
            self.assertEqual(valid.resolve(), _validated_output(valid, repository_root=repository, project_root=project))
            for invalid in (project / "proposal.json", SKILL_ROOT / "state" / "proposal.json"):
                with self.subTest(output=str(invalid)):
                    with self.assertRaises(KernelWikiError):
                        _validated_output(invalid, repository_root=repository, project_root=project)


if __name__ == "__main__":
    unittest.main()

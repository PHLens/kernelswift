from __future__ import annotations

import ast
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

TESTS = Path(__file__).resolve().parent
SKILL_ROOT = TESTS.parent
SCRIPTS = SKILL_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(TESTS))

from catalog import build_generated_outputs, write_generated_outputs  # noqa: E402
from corpus import SourceRecord, WikiCard, load_corpus, validate_corpus  # noqa: E402
from fixture_factory import make_catalog_corpus, remove_tree  # noqa: E402
from get_page import main as get_page_main  # noqa: E402
from grep_wiki import main as grep_main  # noqa: E402
from kernelwiki_common import KernelWikiError, sha256_bytes  # noqa: E402
from query import main as query_main  # noqa: E402
import search as search_module  # noqa: E402
from validate import validate_skill_root  # noqa: E402
from search import (  # noqa: E402
    QueryRequest,
    build_card_candidate,
    build_source_candidate,
    collect_unlimited_candidates,
    grep_corpus,
    parse_query_request,
    query_payload,
    retrieve_page,
    search_records,
)


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.roots: list[Path] = []

    def tearDown(self):
        for root in self.roots:
            remove_tree(root)

    def make_root(self) -> Path:
        root = make_catalog_corpus()
        corpus = load_corpus(root)
        validate_corpus(corpus)
        write_generated_outputs(root, build_generated_outputs(corpus))
        self.roots.append(root)
        return root

    def load_fixture(self):
        root = self.make_root()
        corpus = load_corpus(root)
        validate_corpus(corpus)
        return root, corpus

    def approve_asset(self, root: Path, source_id: str = "source-valid-manual") -> None:
        source_path = root / "sources" / "docs" / "source-valid-manual.md"
        text = source_path.read_text(encoding="utf-8")
        marker = text.find("\n---\n", 4)
        metadata = yaml.safe_load(text[4:marker])
        body = text[marker + 5 :]
        metadata["license_state"] = "approved"
        metadata["artifact_dir"] = f"artifacts/{source_id}"
        rendered = yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True).rstrip()
        source_path.write_text(f"---\n{rendered}\n---\n{body}", encoding="utf-8")
        bundle = root / "artifacts" / source_id
        bundle.mkdir(parents=True)
        code = b"def fused_kernel():\n    return 1\n"
        (bundle / "snippet.py").write_bytes(code)
        provenance = {
            "schema_version": 1,
            "origin_url": "https://example.invalid/derived",
            "upstream_repo": None,
            "upstream_sha": None,
            "license_state": "approved",
            "retrieved_at": "2026-08-21T00:00:00Z",
            "asset_mode": "derived",
            "allowed_audiences": ["designer"],
            "coder_access": "denied",
            "source_ids": [source_id],
            "files": [{
                "local_path": "snippet.py",
                "upstream_path": None,
                "heading_path": None,
                "role": "snippet",
                "mode": "derived",
                "sha256": sha256_bytes(code),
            }],
        }
        (bundle / "PROVENANCE.yaml").write_text(
            yaml.safe_dump(provenance, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )

    def test_title_and_structured_matches_rank_before_body_only(self):
        _root, corpus = self.load_fixture()
        result = search_records(corpus, QueryRequest("kernel fusion", {}, "both", 10))
        self.assertEqual("technique-kernel-fusion", result[0].record_id)
        self.assertEqual(("title", "tags"), result[0].matched_fields)


    def test_each_filter_field(self):
        _root, corpus = self.load_fixture()
        cases = {
            "type": "pattern",
            "tag": "launch-bound",
            "repository": "vllm-ascend",
            "language": "ascendc",
            "target": "ascend910b",
            "target-match": "exact",
            "symptom": "launch-bound",
            "kernel-type": "selection",
            "evidence-level": "local-verifier",
            "reproduction": "benchmarked",
            "audience": "designer",
            "has-code": "false",
        }
        for key, value in cases.items():
            with self.subTest(filter=key):
                hits = search_records(corpus, QueryRequest("", {key: (value,)}, "cards", 20))
                self.assertTrue(hits, key)
                self.assertTrue(all(hit.record_kind == "card" for hit in hits))


    def test_alias_normalization(self):
        _root, corpus = self.load_fixture()
        hit = search_records(corpus, QueryRequest("fused-kernel", {}, "cards", 10))[0]
        self.assertEqual("technique-kernel-fusion", hit.record_id)
        self.assertGreater(hit.score[2], 0)





    def test_source_candidates_never_enter_card_catalog(self):
        root, corpus = self.load_fixture()
        catalog_ids = {json.loads(line)["id"] for line in (root / "compiled" / "catalog.jsonl").read_text().splitlines()}
        candidates = collect_unlimited_candidates(corpus, QueryRequest("", {}, "both", 1))
        self.assertEqual(set(corpus.cards), catalog_ids)
        self.assertTrue(any(candidate.record_kind == "source" for candidate in candidates))
        self.assertTrue(catalog_ids.isdisjoint(corpus.sources))



    def test_stable_path_and_id_tie_break(self):
        _root, corpus = self.load_fixture()
        hits = search_records(corpus, QueryRequest("source", {}, "sources", 20))
        expected = sorted(hits, key=lambda hit: (hit.path, hit.record_id))
        equal_score_groups = {}
        for hit in hits:
            equal_score_groups.setdefault(hit.score, []).append(hit)
        for group in equal_score_groups.values():
            self.assertEqual(sorted(group, key=lambda hit: (hit.path, hit.record_id)), group)
        self.assertEqual(sorted({hit.record_id for hit in hits}), sorted({hit.record_id for hit in expected}))






    def test_retrieve_page_resolves_id_and_relative_path(self):
        _root, corpus = self.load_fixture()
        by_id = retrieve_page(corpus, "technique-kernel-fusion", follow_sources=False, access="metadata")
        by_path = retrieve_page(corpus, "wiki/techniques/kernel-fusion.md", follow_sources=False, access="metadata")
        self.assertEqual(by_id, by_path)
        self.assertEqual("card", by_id.record_kind)





    def test_approved_designer_asset_is_visible(self):
        root, _corpus = self.load_fixture()
        self.approve_asset(root)
        corpus = load_corpus(root)
        validate_corpus(corpus)
        page = retrieve_page(corpus, "source-valid-manual", follow_sources=False, access="approved-assets")
        self.assertTrue(page.asset_access[0].code_visible)
        self.assertEqual("approved", page.asset_access[0].reason)





    def test_grep_reports_checked_in_markdown_line_numbers_but_searches_bodies_only(self):
        root, corpus = self.load_fixture()
        cases = (
            (
                "Fusion can increase register pressure",
                "wiki",
                "wiki/techniques/kernel-fusion.md",
                "technique-kernel-fusion",
            ),
            (
                "This metadata-only source records",
                "sources",
                "sources/docs/source-valid-manual.md",
                "source-valid-manual",
            ),
        )
        for pattern, scope, relative_path, record_id in cases:
            with self.subTest(path=relative_path):
                checked_in_lines = (root / relative_path).read_text(encoding="utf-8").splitlines()
                expected_line = next(
                    index for index, line in enumerate(checked_in_lines, 1) if pattern in line
                )
                matches = grep_corpus(
                    corpus, pattern, scope=scope, max_matches=20, context_chars=0
                )
                match = next(item for item in matches if item.record_id == record_id)
                self.assertEqual(expected_line, match.line_number)
        self.assertEqual(
            (),
            grep_corpus(corpus, "schema_version", scope="both", max_matches=20, context_chars=0),
        )





    def test_query_json_and_markdown_outputs(self):
        root = self.make_root()
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(0, query_main(["fusion", "--root", str(root)]))
        payload = json.loads(stdout.getvalue())
        self.assertEqual(1, payload["schema_version"])
        self.assertTrue(payload["results"])
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            self.assertEqual(0, query_main(["fusion", "--root", str(root), "--format", "markdown"]))
        self.assertIn("[technique-kernel-fusion]", stdout.getvalue())




    def test_production_query_paths_are_offline(self):
        _root, corpus = self.load_fixture()
        with patch.object(socket, "socket", side_effect=AssertionError("network forbidden")):
            with patch("urllib.request.urlopen", side_effect=AssertionError("network forbidden")):
                query_payload(corpus, QueryRequest("fusion", {}, "both", 10))
                retrieve_page(corpus, "technique-kernel-fusion", follow_sources=True, access="metadata")
                grep_corpus(corpus, "fusion", scope="both", max_matches=20, context_chars=20)






if __name__ == "__main__":
    unittest.main()

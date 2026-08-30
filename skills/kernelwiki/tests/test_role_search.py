from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import copy
from dataclasses import replace
import io
import json
from pathlib import Path
import sys
import unittest
from unittest import mock

import yaml

SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
TESTS = SKILL_ROOT / "tests"
FIXTURES = TESTS / "fixtures"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(TESTS))

from catalog import write_generated_outputs  # noqa: E402
from corpus import load_corpus, validate_corpus  # noqa: E402
from fixture_factory import make_valid_corpus, remove_tree  # noqa: E402
from get_page import main as get_page_main  # noqa: E402
from kernelwiki_common import KernelWikiError, canonical_json_bytes, parse_markdown  # noqa: E402
import query as query_module  # noqa: E402
from query import main as query_main  # noqa: E402
from role_context import load_authority_snapshot, load_role_context  # noqa: E402
from role_fixture_factory import (  # noqa: E402
    materialize_exact_role_corpus,
    materialize_vnext_project,
    write_coder_context,
    write_role_context_variant,
)
from role_search import (  # noqa: E402
    ROLE_GROUPS,
    _ranked_candidate,
    parse_role_query_request,
    rank_role_candidates,
    role_get_page,
    role_result_payload,
    role_search,
)
from search import SearchCandidate, SearchHit, collect_unlimited_candidates, parse_query_request  # noqa: E402
from validate import validate_skill_root  # noqa: E402


def _write_markdown(path: Path, metadata, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "---\n" + yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True).rstrip() + "\n---\n" + body,
        encoding="utf-8",
    )


class RoleSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.roots: list[Path] = []

    def tearDown(self) -> None:
        for root in self.roots:
            remove_tree(root)

    def designer_context(self):
        return load_role_context(FIXTURES / "role" / "designer-context.json")

    def missing_context(self):
        return load_role_context(FIXTURES / "role" / "coder-missing-profile.json")

    def exact_environment(self):
        root = make_valid_corpus()
        self.roots.append(root)
        project = materialize_vnext_project(root / "role-project")
        context_path = write_coder_context(project, root / "coder-context.json")
        context = load_role_context(context_path)
        authority = load_authority_snapshot(context)
        materialize_exact_role_corpus(root, authority.sketch_result, authority.decision_result)
        write_generated_outputs(root)
        corpus = validate_skill_root(root)
        return root, corpus, context, authority

    def designer_variant(self, root: Path, **changes):
        base = self.designer_context()
        path = root / f"designer-{len(tuple(root.glob('designer-*.json'))):03d}.json"
        write_role_context_variant(base, path, **changes)
        return load_role_context(path)

    def add_high_scoring_ineligible_cards(self, root: Path, count: int = 5) -> None:
        path = root / "wiki" / "languages" / "mixed-asset-card.md"
        metadata, body = parse_markdown(path)
        for index in range(count):
            item = copy.deepcopy(metadata)
            item["id"] = f"language-ineligible-{index:02d}"
            item["title"] = f"Topk reduction exact high score {index:02d}"
            item["targets"] = ["mlu580"]
            target = root / "wiki" / "languages" / f"ineligible-{index:02d}.md"
            _write_markdown(target, item, body)
        write_generated_outputs(root)

    def add_special_cards(self, root: Path) -> None:
        base_path = root / "wiki" / "techniques" / "kernel-fusion.md"
        base, body = parse_markdown(base_path)
        examples = yaml.safe_load((FIXTURES / "valid-corpus" / "examples.yaml").read_text(encoding="utf-8"))
        for role, key, source_id in (
            ("counterexample", "counterexample", "source-local-ascend-flexattention-round-003"),
            ("capability-gap", "capability_gap", "source-ascendc-programming-model-cann-900beta1"),
        ):
            metadata = copy.deepcopy(base)
            metadata.update(
                id=f"pattern-role-{role}",
                title=f"Role grouped {role} topk reduction",
                type="pattern",
                candidate_techniques=["technique-kernel-fusion"],
                sources=[source_id],
                observations=[],
                examples=[copy.deepcopy(examples[key])],
            )
            _write_markdown(root / "wiki" / "patterns" / f"role-{role}.md", metadata, body)
        write_generated_outputs(root)

    def test_admission_happens_before_limit(self):
        root, _, context, authority = self.exact_environment()
        self.add_high_scoring_ineligible_cards(root)
        corpus = validate_skill_root(root)
        request = parse_role_query_request("topk reduction", scope="cards", group_limits={"admitted": 1})
        result = role_search(corpus, request, context, authority)
        self.assertEqual("language-mixed-asset", result.groups["admitted"][0]["id"])
        self.assertEqual(1, len(result.groups["admitted"]))




    def test_counterexamples_and_gaps_have_independent_limits(self):
        root = make_valid_corpus()
        self.roots.append(root)
        self.add_special_cards(root)
        corpus = validate_skill_root(root)
        request = parse_role_query_request(
            "topk reduction",
            scope="cards",
            group_limits={"admitted": 1, "counterexamples": 1, "capability_gaps": 1},
        )
        result = role_search(corpus, request, self.designer_context(), None)
        self.assertEqual(1, len(result.groups["admitted"]))
        self.assertEqual("pattern-role-counterexample", result.groups["counterexamples"][0]["id"])
        self.assertEqual("pattern-role-capability-gap", result.groups["capability_gaps"][0]["id"])



    def test_group_order_is_deterministic_and_byte_identical(self):
        root = make_valid_corpus()
        self.roots.append(root)
        self.add_special_cards(root)
        corpus = validate_skill_root(root)
        request = parse_role_query_request("", scope="both", show_excluded=True)
        first = role_result_payload(role_search(corpus, request, self.designer_context(), None))
        second = role_result_payload(role_search(corpus, request, self.designer_context(), None))
        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))
        self.assertEqual(tuple(first["groups"]), ROLE_GROUPS)
        for group in ROLE_GROUPS:
            paths = [(item["path"], item["id"]) for item in first["groups"][group]]
            ranks = [tuple(item["rank"].values()) for item in first["groups"][group]]
            self.assertEqual(len(paths), len(ranks))


    def test_missing_profile_coder_result_is_schema_valid_empty(self):
        _, corpus, _, _ = self.exact_environment()
        request = parse_role_query_request("topk", scope="both")
        payload = role_result_payload(role_search(corpus, request, self.missing_context(), None))
        self.assertEqual(1, payload["schema_version"])
        self.assertEqual({}, payload["authority_hashes"])
        self.assertIsNone(payload["loop_contract_identity"])
        self.assertTrue(all(payload["groups"][name] == [] for name in ROLE_GROUPS))








    def test_source_results_use_source_admission(self):
        _, corpus, context, authority = self.exact_environment()
        request = parse_role_query_request("exact mlu coder", scope="sources")
        result = role_search(corpus, request, context, authority)
        self.assertEqual("source-exact-coder", result.groups["admitted"][0]["id"])
        self.assertEqual("source", result.groups["admitted"][0]["record_kind"])

    def test_analogy_only_is_separate(self):
        _, corpus, _, _ = self.exact_environment()
        request = parse_role_query_request("analogy", scope="both")
        result = role_search(corpus, request, self.designer_context(), None)
        ids = {item["id"] for item in result.groups["analogy_only"]}
        self.assertIn("language-analogy-designer", ids)
        self.assertFalse(ids.intersection(item["id"] for item in result.groups["admitted"]))




    def test_same_backend_different_target_items_do_not_boost_exact_rank(self):
        root, _, _, _ = self.exact_environment()
        card_path = root / "wiki" / "languages" / "mixed-asset-card.md"
        metadata, body = parse_markdown(card_path)
        exact = copy.deepcopy(metadata["examples"][0])
        exact.update(
            evidence_level="experimental",
            reproduction="concept",
            implementation_profile_id=None,
            profile_authority="source-only",
            runtime_fingerprint=None,
        )
        cross_target = copy.deepcopy(exact)
        cross_target.update(
            id="example-cross-mlu580",
            target_id="mlu580",
            dtype="bf16",
            shape={"E": 256, "K": 8, "T": 64},
            evidence_level="local-verifier",
            reproduction="benchmarked",
            implementation_profile_id="triton_mlu",
            profile_authority="current-vnext",
            runtime_fingerprint="triton 3.6.0 / CoreX 4.4.0",
            measurement_fingerprint="1" * 64,
            baseline_id="baseline-mlu580",
            candidate_id="candidate-mlu580",
        )
        metadata["examples"] = [cross_target, exact]
        metadata.pop("coder_access", None)
        _write_markdown(card_path, metadata, body)
        source_path = root / "sources" / "commits" / "source-exact-coder.md"
        source_metadata, source_body = parse_markdown(source_path)
        source_metadata["target_ids"] = ["mlu580", "mlu590"]
        _write_markdown(source_path, source_metadata, source_body)
        write_generated_outputs(root)
        corpus = validate_skill_root(root)
        request = parse_role_query_request("topk reduction", scope="cards")
        role_authority = {
            "implementation_profile_id": "triton_mlu",
            "implementation_profile_status": "partial",
            "runtime_fingerprint": "triton 3.6.0 / CoreX 4.4.0",
        }
        mlu590 = self.designer_variant(
            root,
            target_id="mlu590",
            dtypes=["bf16"],
            shape_signature={"E": 256, "K": 8, "T": 64},
            semantic_features=[],
            **role_authority,
        )
        mlu580 = self.designer_variant(
            root,
            target_id="mlu580",
            dtypes=["bf16"],
            shape_signature={"E": 256, "K": 8, "T": 64},
            semantic_features=[],
            **role_authority,
        )

        exact_entry = next(
            item
            for item in role_search(corpus, request, mlu590, None).groups["admitted"]
            if item["id"] == "language-mixed-asset"
        )
        backend_result = role_search(corpus, request, mlu580, None)
        backend_entry = next(
            item
            for group in ("admitted", "conditional", "analogy_only")
            for item in backend_result.groups[group]
            if item["id"] == "language-mixed-asset"
        )
        self.assertEqual(0, exact_entry["rank"]["profile_runtime_exactness"])
        self.assertEqual(0, exact_entry["rank"]["dtype_overlap_count"])
        self.assertLessEqual(exact_entry["rank"]["semantic_shape_score"], 0)
        self.assertEqual(1, exact_entry["rank"]["evidence_rank"])
        self.assertEqual(1, exact_entry["rank"]["reproduction_rank"])
        self.assertEqual(2, backend_entry["rank"]["profile_runtime_exactness"])
        self.assertEqual(1, backend_entry["rank"]["dtype_overlap_count"])
        self.assertEqual(2, backend_entry["rank"]["semantic_shape_score"])
        self.assertEqual(5, backend_entry["rank"]["evidence_rank"])
        self.assertEqual(5, backend_entry["rank"]["reproduction_rank"])










    def test_role_get_page_exact_envelope_and_separate_asset_admission(self):
        _, corpus, context, authority = self.exact_environment()
        payload = role_get_page(
            corpus,
            "language-mixed-asset",
            context,
            authority,
            follow_sources=True,
            access="approved-assets",
            example_ids=("example-test-exact",),
            guidance_ids=("guidance-test-exact",),
            asset_ids=("asset-full-kernel", "asset-short-snippet"),
        )
        self.assertEqual(
            {"schema_version", "context_sha256", "loop_contract_identity", "authority_hashes", "admission", "page"},
            set(payload),
        )
        self.assertEqual("admitted", payload["admission"]["status"])
        selected = {(item["kind"], item["id"]) for item in payload["page"]["selected_items"]}
        denied = {(item["kind"], item["id"]) for item in payload["page"]["denied_items"]}
        self.assertIn(("asset", "asset-short-snippet"), selected)
        self.assertIn(("asset", "asset-full-kernel"), denied)
        asset_access = payload["page"]["asset_access"][0]
        self.assertTrue(asset_access["code_visible"])
        self.assertEqual(["asset-short-snippet"], asset_access["admitted_asset_ids"])
        self.assertEqual("role-admitted", asset_access["reason"])
        self.assertIn("value = load_row(pointer)", payload["page"]["body"])
        self.assertIn("guidance-test-exact", payload["page"]["body"])
        self.assertIn("example-test-exact", payload["page"]["body"])
        self.assertNotIn("designer-only full implementation", payload["page"]["body"])
        self.assertTrue(payload["page"]["followed_sources"])
        self.assertEqual("admitted", payload["page"]["followed_sources"][0]["admission"]["status"])





    def test_unrequested_asset_content_is_hidden_from_coder_card_and_followed_source_bodies(self):
        root, _, context, authority = self.exact_environment()
        card_path = root / "wiki" / "languages" / "mixed-asset-card.md"
        metadata, body = parse_markdown(card_path)
        _write_markdown(card_path, metadata, body + "\n## asset-full-kernel\nUNREQUESTED_CARD_SECRET\n")
        source_path = root / "sources" / "commits" / "source-exact-coder.md"
        source_metadata, source_body = parse_markdown(source_path)
        _write_markdown(source_path, source_metadata, source_body + "\nasset-full-kernel\nUNREQUESTED_SOURCE_SECRET\n")
        write_generated_outputs(root)
        corpus = validate_skill_root(root)
        payload = role_get_page(
            corpus,
            "language-mixed-asset",
            context,
            authority,
            follow_sources=True,
            access="approved-assets",
        )
        self.assertEqual("", payload["page"]["body"])
        self.assertEqual("", payload["page"]["followed_sources"][0]["body"])
        serialized = canonical_json_bytes(payload).decode()
        self.assertNotIn("UNREQUESTED_CARD_SECRET", serialized)
        self.assertNotIn("UNREQUESTED_SOURCE_SECRET", serialized)











    def test_query_cli_missing_profile_can_show_excluded(self):
        root, _, _, _ = self.exact_environment()
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = query_main([
                "topk",
                "--root",
                str(root),
                "--context",
                str(FIXTURES / "role" / "coder-missing-profile.json"),
                "--show-excluded",
            ])
        self.assertEqual(0, exit_code)
        self.assertEqual("", stderr.getvalue())
        payload = json.loads(stdout.getvalue())
        self.assertTrue(payload["groups"]["excluded"])
        self.assertIn("profile-missing", payload["groups"]["excluded"][0]["admission"]["reasons"])


if __name__ == "__main__":
    unittest.main()

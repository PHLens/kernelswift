from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import yaml

SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
TESTS = SKILL_ROOT / "tests"
FIXTURES = TESTS / "fixtures"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(TESTS))

from admission import (  # noqa: E402
    AdmissionDecision,
    PROTECTED_FIELDS,
    admit_asset,
    admit_candidate,
    admit_card,
    admit_source,
    build_exact_guidance,
    classify_designer_match,
    protected_projection,
    require_validated_admission_decision,
    resolve_capability_status,
    resolve_version_claim,
    validate_guidance_binding,
)
from corpus import (  # noqa: E402
    GuidanceSchemaError,
    SourceRecord,
    WikiCard,
    load_corpus,
    validate_corpus,
    validate_guidance_schema,
)
from fixture_factory import make_valid_corpus, remove_tree  # noqa: E402
from kernelwiki_common import KernelWikiError, parse_markdown  # noqa: E402
from role_context import AuthoritySnapshot, RoleQueryContext, load_authority_snapshot, load_role_context  # noqa: E402
from role_fixture_factory import (  # noqa: E402
    materialize_exact_role_corpus,
    materialize_vnext_project,
    write_coder_context,
    write_role_context_variant,
)
from search import build_source_candidate  # noqa: E402
from validate import validate_skill_root  # noqa: E402


def _card(path: Path) -> WikiCard:
    metadata, body = parse_markdown(path)
    return WikiCard(path=path, metadata=metadata, body=body)


def _source(path: Path) -> SourceRecord:
    metadata, body = parse_markdown(path)
    return SourceRecord(path=path, metadata=metadata, body=body)


class AdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._roots: list[Path] = []

    def tearDown(self) -> None:
        for root in self._roots:
            remove_tree(root)

    def designer_context(self):
        return load_role_context(FIXTURES / "role" / "designer-context.json")

    def sealed_context(self, root: Path, context, **changes):
        counter = len(tuple(root.glob("context-variant-*.json")))
        path = root / f"context-variant-{counter:03d}.json"
        write_role_context_variant(context, path, **changes)
        return load_role_context(path)

    def rewrite_card(self, root: Path, card_id: str, mutate):
        corpus = load_corpus(root)
        card = corpus.cards[card_id]
        metadata, body = parse_markdown(card.path)
        mutate(metadata)
        card.path.write_text(
            "---\n" + yaml.safe_dump(metadata, sort_keys=False).rstrip() + "\n---\n" + body,
            encoding="utf-8",
        )
        refreshed = load_corpus(root)
        validate_corpus(refreshed)
        return refreshed.cards[card_id]

    def rewrite_source(self, root: Path, source_id: str, mutate):
        corpus = load_corpus(root)
        source = corpus.sources[source_id]
        metadata, body = parse_markdown(source.path)
        mutate(metadata)
        source.path.write_text(
            "---\n" + yaml.safe_dump(metadata, sort_keys=False).rstrip() + "\n---\n" + body,
            encoding="utf-8",
        )
        refreshed = load_corpus(root)
        validate_corpus(refreshed)
        return refreshed.sources[source_id]

    @contextmanager
    def exact_environment(self):
        root = make_valid_corpus()
        self._roots.append(root)
        project = materialize_vnext_project(root / "role-project")
        context_path = write_coder_context(project, root / "coder-context.json")
        context = load_role_context(context_path)
        authority = load_authority_snapshot(context)
        card_path = materialize_exact_role_corpus(root, authority.sketch_result, authority.decision_result)
        corpus = load_corpus(root)
        validate_corpus(corpus)
        yield root, corpus, corpus.cards["language-mixed-asset"], context, authority, card_path

    def test_designer_sees_analogy_with_explicit_class(self):
        with self.exact_environment() as (_, corpus, _, _, _, _):
            card = corpus.cards["language-analogy-designer"]
            decision = admit_card(card, self.designer_context(), None)
            self.assertEqual("analogy_only", decision.status)
            self.assertEqual("analogy-only", decision.match_class)


    def test_coder_rejects_exact_page_without_statement_binding(self):
        with self.exact_environment() as (root, _, card, context, authority, _):
            context = self.sealed_context(root, context, guidance_bindings={})
            decision = admit_card(card, context, authority)
            self.assertEqual("excluded", decision.status)
            self.assertIn("sketch-binding-required", decision.reasons)








    def test_exact_target_mismatch_excludes_coder(self):
        with self.exact_environment() as (root, _, card, context, authority, _):
            decision = admit_card(card, self.sealed_context(root, context, target_id="mlu580"), authority)
            self.assertIn("target-mismatch", decision.reasons)



    def test_unknown_or_unsupported_capability_excludes_coder(self):
        with self.exact_environment() as (root, _, _, context, authority, _):
            for capability, expected in (("matrix.dot.fp32", "capability-unknown"), ("memory.copy.device-to-device", "capability-unsupported")):
                with self.subTest(capability=capability):
                    card = self.rewrite_card(
                        root,
                        "language-mixed-asset",
                        lambda metadata, capability=capability: metadata["coder_access"]["guidance"][0].update(
                            required_capabilities=[capability]
                        ),
                    )
                    decision = admit_card(card, context, authority)
                    self.assertIn(expected, decision.reasons)








    def test_stale_version_excludes_coder(self):
        with self.exact_environment() as (root, corpus, card, context, authority, _):
            claims = {
                "schema_version": 1,
                "claims": [{
                    "id": "claim-stale-test",
                    "card_ids": [card.card_id],
                    "subject": "fixture-runtime",
                    "status": "stale",
                    "supported_versions": ["fixture"],
                    "last_verified_at": "2026-08-21",
                    "source_ids": ["source-exact-coder"],
                    "replacement_claim_id": None,
                }],
            }
            (root / "data" / "version-claims.yaml").write_text(yaml.safe_dump(claims, sort_keys=False), encoding="utf-8")
            metadata = dict(card.metadata)
            metadata["version_sensitive"] = ["claim-stale-test"]
            access = dict(metadata["coder_access"])
            guidance = dict(access["guidance"][0])
            guidance["version_claim_ids"] = ["claim-stale-test"]
            access["guidance"] = [guidance]
            metadata["coder_access"] = access
            path = root / "wiki" / "languages" / "mixed-asset-card.md"
            _, body = parse_markdown(path)
            path.write_text("---\n" + yaml.safe_dump(metadata, sort_keys=False).rstrip() + "\n---\n" + body, encoding="utf-8")
            fresh = load_corpus(root)
            validate_corpus(fresh)
            decision = admit_card(fresh.cards[card.card_id], context, authority)
            self.assertIn("version-stale", decision.reasons)

    def test_page_admission_does_not_admit_designer_only_asset(self):
        with self.exact_environment() as (_, _, card, context, authority, _):
            page = admit_card(card, context, authority)
            asset = admit_asset(card, "asset-full-kernel", context, authority)
            self.assertEqual("admitted", page.status)
            self.assertEqual((), page.admitted_asset_ids)
            self.assertIn("artifact-designer-only", asset.reasons)

    def test_approved_snippet_is_exposed_only_after_asset_admission(self):
        with self.exact_environment() as (_, _, card, context, authority, _):
            page = admit_card(card, context, authority)
            asset = admit_asset(card, "asset-short-snippet", context, authority)
            self.assertEqual((), page.admitted_asset_ids)
            self.assertEqual("admitted", asset.status)
            self.assertEqual(("asset-short-snippet",), asset.admitted_asset_ids)

    def test_unapproved_license_denies_asset(self):
        with self.exact_environment() as (root, _, card, context, authority, _):
            source_path = root / "sources" / "commits" / "source-exact-coder.md"
            metadata, body = parse_markdown(source_path)
            metadata["license_state"] = "unknown"
            source_path.write_text("---\n" + yaml.safe_dump(metadata, sort_keys=False).rstrip() + "\n---\n" + body, encoding="utf-8")
            card = load_corpus(root).cards[card.card_id]
            decision = admit_asset(card, "asset-short-snippet", context, authority)
            self.assertIn("license-unapproved", decision.reasons)





    def test_broken_source_is_excluded(self):
        with self.exact_environment() as (root, corpus, _, context, authority, _):
            source = corpus.sources["source-exact-coder"]
            source.path.unlink()
            decision = admit_source(source, context, authority)
            self.assertEqual("excluded", decision.status)
            self.assertIn("source-broken", decision.reasons)







    def test_constrained_capability_is_admitted(self):
        with self.exact_environment() as (root, _, _, context, authority, _):
            card = self.rewrite_card(
                root,
                "language-mixed-asset",
                lambda metadata: metadata["coder_access"]["guidance"][0].update(
                    required_capabilities=["memory.store.contiguous-fp32"]
                ),
            )
            decision = admit_card(card, context, authority)
            self.assertEqual("admitted", decision.status)



























if __name__ == "__main__":
    unittest.main()

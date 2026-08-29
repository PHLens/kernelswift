from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest

import yaml

SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
TESTS = SKILL_ROOT / "tests"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(TESTS))

from corpus import load_corpus, load_version_claim_registry, validate_corpus, validate_examples_document  # noqa: E402
from fixture_factory import (  # noqa: E402
    _parse_markdown,
    _write_markdown,
    card_path,
    make_valid_corpus,
    mutate_card,
    mutate_source,
    remove_tree,
    source_path,
)
from kernelwiki_common import KernelWikiError, load_yaml_document  # noqa: E402
from validate import main as validate_main  # noqa: E402


class CorpusTests(unittest.TestCase):
    def make_root(self, **kwargs) -> Path:
        root = make_valid_corpus(**kwargs)
        self.addCleanup(remove_tree, root)
        return root

    def assert_invalid(self, code: str, mutate) -> None:
        root = self.make_root()
        mutate(root)
        with self.assertRaisesRegex(KernelWikiError, code):
            validate_corpus(load_corpus(root))

    def write_version_claims(self, root: Path, claims: list[dict]) -> None:
        registry = {"schema_version": 1, "claims": claims}
        (root / "data" / "version-claims.yaml").write_text(
            yaml.safe_dump(registry, sort_keys=False), encoding="utf-8"
        )

    def valid_version_claim(self, claim_id: str = "claim-runtime-one") -> dict:
        return {
            "id": claim_id,
            "card_ids": ["technique-kernel-fusion"],
            "subject": "fixture-runtime",
            "status": "current",
            "supported_versions": ["runtime-1"],
            "last_verified_at": "2026-08-21",
            "source_ids": ["source-valid-manual"],
            "replacement_claim_id": None,
        }

    def test_valid_card_resolves_source_and_related_links(self):
        root = self.make_root()
        corpus = load_corpus(root)
        validate_corpus(corpus)
        self.assertIn("technique-kernel-fusion", corpus.cards)
        self.assertIn("source-valid-manual", corpus.sources)

    def test_duplicate_ids_fail(self):
        root = self.make_root(duplicate_card_id=True)
        with self.assertRaisesRegex(KernelWikiError, "id-duplicate"):
            load_corpus(root)

    def test_unknown_taxonomy_value_fails(self):
        root = self.make_root(extra_tag="not-in-taxonomy")
        with self.assertRaisesRegex(KernelWikiError, "taxonomy-unknown"):
            validate_corpus(load_corpus(root))

    def test_example_requires_existing_source_and_scope(self):
        root = self.make_root(example_source="missing-source")
        with self.assertRaisesRegex(KernelWikiError, "example-source-missing"):
            validate_corpus(load_corpus(root))

    def test_unresolved_related_fails(self):
        root = self.make_root(related_id="missing-card")
        with self.assertRaisesRegex(KernelWikiError, "related-missing"):
            validate_corpus(load_corpus(root))










    def test_local_example_without_transfer_boundary_fails(self):
        root = self.make_root(local_example_without_transfer=True)
        with self.assertRaisesRegex(KernelWikiError, "example-transfer-boundary-required"):
            validate_corpus(load_corpus(root))












    def test_capability_gap_conditionals_are_enforced(self):
        def base_gap(metadata, body):
            example = metadata["examples"][0]
            example.update(
                role="capability-gap",
                subtype="profile",
                observed=[],
                terminal_classification="not-applicable",
                comparability="not-comparable",
            )

        self.assert_invalid("capability-gap-field-required", lambda root: mutate_card(root, base_gap))

        def observed_gap(metadata, body):
            base_gap(metadata, body)
            example = metadata["examples"][0]
            example.update(
                capability_id="profile.missing",
                capability_status="unknown",
                required_probe_or_authority="reviewed authority",
                observed=[{"metric": "correctness_pass", "value": False, "statistic": "exact", "unit": "boolean"}],
            )
        self.assert_invalid("capability-gap-observed-forbidden", lambda root: mutate_card(root, observed_gap))

        def invalid_status(metadata, body):
            base_gap(metadata, body)
            metadata["examples"][0].update(
                capability_id="profile.missing",
                capability_status="assumed",
                required_probe_or_authority="reviewed authority",
            )
        self.assert_invalid("capability-gap-status-invalid", lambda root: mutate_card(root, invalid_status))

        self.assert_invalid(
            "example-field-unknown",
            lambda root: mutate_card(
                root, lambda metadata, body: metadata["examples"][0].__setitem__("capability_id", "not.allowed")
            ),
        )


    def test_local_campaign_authority_rules_fail_closed(self):
        def mutate(metadata, body):
            metadata.update(
                source_kind="local-campaign",
                profile_authority="historical-noncanonical",
                strict_vnext_validated=True,
                missing_evidence=[],
                audiences=["designer"],
            )
        self.assert_invalid("local-campaign-authority-invalid", lambda root: mutate_source(root, mutate))











    def test_current_version_claim_requires_complete_source_backing(self):
        cases = (
            ("supported_versions", [], "version-current-unbacked"),
            ("source_ids", [], "version-current-unbacked"),
            ("last_verified_at", None, "version-current-unbacked"),
        )
        for field, value, code in cases:
            with self.subTest(field=field):
                root = self.make_root()
                claim = self.valid_version_claim()
                claim[field] = value
                self.write_version_claims(root, [claim])
                with self.assertRaisesRegex(KernelWikiError, code):
                    validate_corpus(load_corpus(root))












    def test_literal_examples_pass_complete_schema_and_source_link_validation(self):
        root = self.make_root()
        corpus = load_corpus(root)
        validate_corpus(corpus)
        examples = validate_examples_document(root / "examples.yaml", corpus)
        self.assertEqual(
            {
                "example-ascendc-profile-missing",
                "example-flexattention-device-wall-loss",
                "example-grouped-topk-fusion",
            },
            {example["id"] for example in examples},
        )



    def test_seed_card_frontmatter_is_closed_taxonomy(self):
        relative_paths = (
            "wiki/hardware/ascend-execution-and-memory.md",
            "wiki/languages/triton-ascend-backend.md",
            "wiki/languages/mskl-kernel-authoring.md",
            "wiki/languages/ascendc-programming-model.md",
            "wiki/runtimes/ascend-kernel-integration.md",
            "wiki/techniques/kernel-fusion.md",
            "wiki/techniques/tiling-and-work-partitioning.md",
            "wiki/techniques/topk-selection-and-reduction.md",
            "wiki/patterns/launch-bound-materialization.md",
            "wiki/measurement/cann-device-attribution.md",
            "wiki/patterns/device-win-wall-loss.md",
            "wiki/patterns/ascend-capability-gap.md",
        )
        taxonomy = load_yaml_document(SKILL_ROOT / "data" / "taxonomy.yaml")
        for relative_path in relative_paths:
            with self.subTest(path=relative_path):
                metadata, _ = _parse_markdown(SKILL_ROOT / relative_path)
                for field in (
                    "tags",
                    "languages",
                    "kernel_types",
                    "techniques",
                    "symptoms",
                    "hardware_features",
                ):
                    self.assertTrue(set(metadata[field]) <= set(taxonomy[field]), (field, metadata[field]))




    def test_validator_cli_outputs_stable_summary(self):
        output = io.StringIO()
        fixture = TESTS / "fixtures" / "valid-corpus"
        with redirect_stdout(output):
            self.assertEqual(0, validate_main(["--root", str(fixture)]))
        self.assertEqual('{"cards": 1, "schema_version": 1, "sources": 4, "valid": true}\n', output.getvalue())


if __name__ == "__main__":
    unittest.main()

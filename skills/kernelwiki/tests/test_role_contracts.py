from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from evaluate_holdout import (  # noqa: E402
    ADVERSARIAL_CASES,
    canonical_report_bytes,
    evaluate_holdout_report,
    evaluate_queries,
    load_evaluation_cases,
    main as evaluate_main,
    verify_holdout_inputs,
)
from validate import validate_skill_root  # noqa: E402


MANIFEST = SKILL_ROOT / "data" / "evaluation-holdouts.yaml"
GOLD = SKILL_ROOT / "tests" / "fixtures" / "holdout" / "track2-sinkhorn-gold.yaml"
DEVELOPMENT = SKILL_ROOT / "data" / "track2-development-queries.yaml"
TRACK2 = SKILL_ROOT / "tests" / "fixtures" / "track2"


class RoleContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = validate_skill_root(SKILL_ROOT)
        cls.cases = load_evaluation_cases(SKILL_ROOT)
        manifest, gold = verify_holdout_inputs(MANIFEST.read_bytes(), GOLD.read_bytes())
        cls.gold = gold
        cls.adversarial = evaluate_queries(cls.corpus, cls.cases)
        cls.first = evaluate_holdout_report(cls.corpus, cls.cases, manifest, gold, cls.adversarial)
        cls.second = evaluate_holdout_report(cls.corpus, cls.cases, manifest, gold, cls.adversarial)

    def test_five_adversarial_cases_preserve_safety(self):
        development = yaml.safe_load(DEVELOPMENT.read_text(encoding="utf-8"))
        self.assertEqual(["index_topk", "sparse_attn"], [item["id"] for item in development["contexts"]])
        for item in development["contexts"]:
            serialized = json.dumps(item, sort_keys=True).casefold()
            self.assertFalse(any(term in serialized for term in ("source_code", "recipe", "card_body")))
            fixture_name = "index-topk-development.json" if item["id"] == "index_topk" else "sparse-attn-development.json"
            self.assertEqual(item["id"], json.loads((TRACK2 / fixture_name).read_text(encoding="utf-8"))["case_id"])

        adversarial = self.adversarial
        self.assertEqual(5, adversarial["case_count"])
        by_id = {item["case_id"]: item for item in adversarial["cases"]}
        for case_id, contract in ADVERSARIAL_CASES.items():
            with self.subTest(case_id=case_id):
                item = by_id[case_id]
                self.assertEqual(contract, item["contract"])
                self.assertEqual(0, item["unsafe_coder_admissions"])
                self.assertEqual(0, item["unknown_promotions"])
                self.assertEqual(0, item["cross_target_recipe_leaks"])
                self.assertTrue(item["passed"])
        self.assertTrue(adversarial["safety_gate_passed"])

    def test_sealed_hash_mismatch_exits_two(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.yaml"
            gold = root / "gold.yaml"
            manifest.write_bytes(MANIFEST.read_bytes())
            gold.write_bytes(GOLD.read_bytes() + b"tampered\n")
            stdout = io.StringIO()
            stderr = io.StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                status = evaluate_main(["--root", str(SKILL_ROOT), "--manifest", str(manifest), "--gold", str(gold)])
            self.assertEqual(2, status)
            self.assertEqual("", stdout.getvalue())
            self.assertIn("error[holdout-sha-mismatch]", stderr.getvalue())

    def test_report_is_deterministic_and_uses_gold_denominators(self):
        first = canonical_report_bytes(self.first)
        second = canonical_report_bytes(self.second)
        self.assertEqual(first, second)
        metrics = self.first["metrics"]
        holdout = self.first["holdout"]
        judgments = self.gold["gold"]

        expected_top5 = len(set(judgments["relevant_card_ids"]) & set(holdout["top5_admitted_card_ids"]))
        expected_counterexamples = len(set(judgments["counterexample_card_ids"]) & set(holdout["designer_group_card_ids"]["counterexamples"]))
        expected_gaps = len(set(judgments["capability_gap_card_ids"]) & set(holdout["capability_gap_card_ids"]))
        self.assertEqual((expected_top5, 4, 0.75), (metrics["top5_relevant_numerator"], metrics["top5_relevant_denominator"], metrics["top5_relevant_card_recall"]))
        self.assertEqual((expected_counterexamples, 1, 0.0), (metrics["counterexample_numerator"], metrics["counterexample_denominator"], metrics["counterexample_recall"]))
        self.assertEqual((expected_gaps, 1, 0.0), (metrics["capability_gap_numerator"], metrics["capability_gap_denominator"], metrics["capability_gap_recall"]))
        self.assertEqual(0, metrics["unsafe_coder_admissions"])
        self.assertEqual(0, metrics["unknown_promotions"])
        self.assertEqual(0, metrics["cross_target_recipe_leaks"])
        self.assertEqual("recorded-no-tuning", metrics["retrieval_gate_status"])


if __name__ == "__main__":
    unittest.main()

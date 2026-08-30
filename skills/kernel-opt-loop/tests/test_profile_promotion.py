import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

from render_profile_promotion import ProbeValidationError, render_profile_promotion, validate_promotion_candidate
from run_profile_probe import run_profile_probe
from validate_profile import load_profile

FIXTURES = Path(__file__).parent / "fixtures" / "vnext"
PROBES = FIXTURES / "probes"
S60 = PROBES / "qualification" / "s60"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ProfilePromotionTests(unittest.TestCase):
    def setUp(self):
        self.output_root = Path(tempfile.mkdtemp(prefix="promotion-runs-"))
        self.profile_root = Path(tempfile.mkdtemp(prefix="promotion-profile-"))
        shutil.copytree(PROBES / "profile", self.profile_root / "profile")
        self.profile_path = self.profile_root / "profile" / "profile.yaml"
        self.runtime_snapshot_path = PROBES / "runtime-snapshot.json"
        self.run_dir = run_profile_probe(
            profile_path=self.profile_path,
            probe_id="fixture-basic-memory-001",
            target_id="fixture-device",
            runtime_snapshot_path=self.runtime_snapshot_path,
            output_root=self.output_root,
            run_id="promotion-001",
        )

    def tearDown(self):
        shutil.rmtree(self.output_root, ignore_errors=True)
        shutil.rmtree(self.profile_root, ignore_errors=True)

    def test_promotion_is_proposed_and_profile_bytes_do_not_change(self):
        before = self.profile_path.read_bytes()
        candidate_path, note_path = render_profile_promotion(self.run_dir, profile_path=self.profile_path)
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        self.assertEqual("proposed", candidate["review_status"])
        self.assertEqual(before, self.profile_path.read_bytes())
        self.assertTrue(note_path.is_file())
        self.assertEqual("unknown", candidate["recommendations"][0]["current_status"])
        self.assertEqual("constrained", candidate["recommendations"][0]["recommended_status"])
        self.assertNotIn("onboarding_disposition", candidate)

    def test_promotion_never_recommends_supported(self):
        candidate_path, _ = render_profile_promotion(self.run_dir, profile_path=self.profile_path)
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        statuses = {recommendation["recommended_status"] for recommendation in candidate["recommendations"]}
        self.assertNotIn("supported", statuses)

    def test_blocked_and_failed_runs_cannot_be_promoted(self):
        from run_profile_probe import run_profile_probe as run

        blocked_dir = run(
            profile_path=self.profile_path,
            probe_id="fixture-basic-memory-001",
            target_id="wrong-target",
            runtime_snapshot_path=self.runtime_snapshot_path,
            output_root=self.output_root,
            run_id="blocked-promo-001",
        )
        with self.assertRaisesRegex(ProbeValidationError, "cannot promote"):
            render_profile_promotion(blocked_dir, profile_path=self.profile_path)

    def test_candidate_rejects_widened_scope_on_revalidation(self):
        candidate_path, _ = render_profile_promotion(self.run_dir, profile_path=self.profile_path)
        profile = load_profile(self.profile_path)
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        candidate["recommendations"][0]["source_scope"]["shape"] = ["M", "N", "K"]
        with self.assertRaisesRegex(ProbeValidationError, "scope"):
            validate_promotion_candidate(candidate, run_dir=self.run_dir, profile=profile)

    def test_candidate_rejects_supported_recommendation(self):
        candidate_path, _ = render_profile_promotion(self.run_dir, profile_path=self.profile_path)
        profile = load_profile(self.profile_path)
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        candidate["recommendations"][0]["recommended_status"] = "supported"
        with self.assertRaisesRegex(ProbeValidationError, "never recommends supported"):
            validate_promotion_candidate(candidate, run_dir=self.run_dir, profile=profile)

    def test_contradictory_observed_scope_is_rejected(self):
        # A payload whose observed scope contradicts the definition template must not render.
        payload_template = json.loads((PROBES / "invalid-scope-promotion.json").read_text(encoding="utf-8"))
        payload_source = f'''
import argparse, json, sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--runtime-snapshot", required=True)
    args = parser.parse_args()
    payload = {json.dumps(payload_template)}
    payload["target_id"] = args.target_id
    with open(args.result_json, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
'''
        (self.profile_root / "profile" / "probes" / "fake-success.py").write_text(payload_source, encoding="utf-8")
        definition_path = self.profile_root / "profile" / "probes" / "basic-memory.json"
        definition = json.loads(definition_path.read_text(encoding="utf-8"))
        for artifact in definition["input_artifacts"]:
            artifact["sha256"] = sha256_file(self.profile_root / "profile" / artifact["path"])
        definition_path.write_text(json.dumps(definition, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        profile = json.loads(self.profile_path.read_text(encoding="utf-8"))
        profile["probe_catalog"] = [
            {"probe_id": "fixture-basic-memory-001", "definition_path": "probes/basic-memory.json", "definition_sha256": sha256_file(definition_path)}
        ]
        self.profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        run_dir = run_profile_probe(
            profile_path=self.profile_path,
            probe_id="fixture-basic-memory-001",
            target_id="fixture-device",
            runtime_snapshot_path=self.runtime_snapshot_path,
            output_root=self.output_root,
            run_id="scope-promo-001",
        )
        with self.assertRaisesRegex(ProbeValidationError, "scope"):
            render_profile_promotion(run_dir, profile_path=self.profile_path)


class S60PromotionTests(unittest.TestCase):
    def setUp(self):
        self.output_root = Path(tempfile.mkdtemp(prefix="s60-promo-"))
        self.s60_root = Path(tempfile.mkdtemp(prefix="s60-promo-profile-"))
        shutil.copytree(S60, self.s60_root / "s60")
        self.profile_path = self.s60_root / "s60" / "profile.yaml"
        self.runtime_path = self.s60_root / "s60" / "runtime-snapshot.json"
        self.requirement_path = self.s60_root / "s60" / "requirements" / "attention-dot-before-fallback.json"

    def tearDown(self):
        shutil.rmtree(self.output_root, ignore_errors=True)
        shutil.rmtree(self.s60_root, ignore_errors=True)

    def test_eligible_demand_selected_success_is_promotion_pending(self):
        from validate_probe import select_profile_probes

        profile = load_profile(self.profile_path)
        requirement = json.loads(self.requirement_path.read_text(encoding="utf-8"))
        snapshot = json.loads(self.runtime_path.read_text(encoding="utf-8"))
        plan = select_profile_probes(profile, [requirement], snapshot)
        run_dir = run_profile_probe(
            profile_path=self.profile_path,
            probe_id=plan.selections[0].probe_id,
            target_id="s60",
            runtime_snapshot_path=self.runtime_path,
            qualification_requirement_path=self.requirement_path,
            output_root=self.output_root,
            run_id="s60-promo-001",
        )
        candidate_path, note_path = render_profile_promotion(run_dir, profile_path=self.profile_path)
        candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
        self.assertEqual("promotion-pending", candidate["onboarding_disposition"])
        self.assertTrue(note_path.is_file())
        # The canonical profile still carries dot unknown; promotion is not an approval.
        dot = next(
            item for item in load_profile(self.profile_path)["capability_matrix"]
            if item["id"] == "matrix.dot.fp16-fp16-fp32"
        )
        self.assertEqual("unknown", dot["status"])
        self.assertFalse((self.output_root / "team-state.md").exists())


if __name__ == "__main__":
    unittest.main()

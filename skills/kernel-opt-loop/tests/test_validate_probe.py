import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

from validate_probe import (
    ProbeValidationError,
    validate_probe_definition,
    validate_probe_run,
    select_profile_probes,
)
from validate_profile import ProfileValidationError, load_profile

FIXTURES = Path(__file__).parent / "fixtures" / "vnext"
PROBES = FIXTURES / "probes"
S60 = PROBES / "qualification" / "s60"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialized_profile() -> tuple[Path, Path]:
    """Copy the generic runner fixture tree to a temp dir and return (root, profile_path)."""
    root = Path(tempfile.mkdtemp(prefix="probe-profile-"))
    shutil.copytree(PROBES / "profile", root / "profile")
    return root, root / "profile" / "profile.yaml"


class ValidateProbeDefinitionTests(unittest.TestCase):
    def setUp(self):
        self.root, self.profile_path = materialized_profile()
        self.profile = load_profile(self.profile_path)

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_definition_validates_against_profile(self):
        result = validate_probe_definition(
            self.root / "profile" / "probes" / "basic-memory.json",
            profile=self.profile,
        )
        self.assertTrue(result["valid"])
        self.assertEqual("fixture-basic-memory-001", result["definition"]["probe_id"])

    def test_definition_profile_mismatch_is_rejected(self):
        definition_path = self.root / "profile" / "probes" / "basic-memory.json"
        definition = json.loads(definition_path.read_text(encoding="utf-8"))
        definition["implementation_profile_id"] = "other_profile"
        definition_path.write_text(json.dumps(definition), encoding="utf-8")
        with self.assertRaisesRegex(ProbeValidationError, "does not match the profile"):
            validate_probe_definition(definition_path, profile=self.profile)

    def test_definition_unknown_capability_is_rejected(self):
        definition_path = self.root / "profile" / "probes" / "basic-memory.json"
        definition = json.loads(definition_path.read_text(encoding="utf-8"))
        definition["capability_ids"] = ["no.such.capability"]
        definition_path.write_text(json.dumps(definition), encoding="utf-8")
        with self.assertRaisesRegex(ProbeValidationError, "unknown capability"):
            validate_probe_definition(definition_path, profile=self.profile)

    def test_definition_input_hash_mismatch_is_rejected(self):
        definition_path = self.root / "profile" / "probes" / "basic-memory.json"
        definition = json.loads(definition_path.read_text(encoding="utf-8"))
        definition["input_artifacts"][0]["sha256"] = "0" * 64
        definition_path.write_text(json.dumps(definition), encoding="utf-8")
        with self.assertRaisesRegex(ProbeValidationError, "hash mismatch"):
            validate_probe_definition(definition_path, profile=self.profile)

    def test_definition_disallowed_placeholder_is_rejected(self):
        definition_path = self.root / "profile" / "probes" / "basic-memory.json"
        definition = json.loads(definition_path.read_text(encoding="utf-8"))
        definition["runner"]["argv"].append("{injected_shell}")
        definition_path.write_text(json.dumps(definition), encoding="utf-8")
        with self.assertRaisesRegex(ProbeValidationError, "disallowed placeholder"):
            validate_probe_definition(definition_path, profile=self.profile)


class S60SelectionTests(unittest.TestCase):
    def setUp(self):
        self.profile = load_profile(S60 / "profile.yaml")
        self.snapshot = json.loads((S60 / "runtime-snapshot.json").read_text(encoding="utf-8"))

    def test_before_fallback_selects_only_the_unique_exact_scope_dot_probe(self):
        requirement = json.loads((S60 / "requirements" / "attention-dot-before-fallback.json").read_text(encoding="utf-8"))
        plan = select_profile_probes(self.profile, [requirement], self.snapshot)
        self.assertEqual(["s60-dot-fp16-001"], [item.probe_id for item in plan.selections])
        self.assertEqual([("attention-dot-before-fallback", "selected")], [(d.requirement_id, d.outcome) for d in plan.dispositions])

    def test_optional_requirement_never_triggers_a_probe_sweep(self):
        requirement = json.loads((S60 / "requirements" / "reduction-only.json").read_text(encoding="utf-8"))
        plan = select_profile_probes(self.profile, [requirement], self.snapshot)
        self.assertEqual([], list(plan.selections))

    def test_zero_matches_return_no_exact_probe(self):
        requirement = {
            "requirement_id": "copy-before-fallback",
            "primary_contract": "memory.copy",
            "primary_signature": {"dtype": "fp32"},
            "fallback_contract": "memory.move",
            "fallback_signature": {"dtype": "fp32"},
            "fallback_kind": "algorithm-substitution",
            "probe_policy": "before-fallback",
        }
        plan = select_profile_probes(self.profile, [requirement], self.snapshot)
        self.assertEqual([], list(plan.selections))
        self.assertEqual([("copy-before-fallback", "no-exact-probe")], [(d.requirement_id, d.outcome) for d in plan.dispositions])

    def test_two_exact_matches_are_ambiguous(self):
        requirement = json.loads((S60 / "requirements" / "attention-dot-before-fallback.json").read_text(encoding="utf-8"))
        root = Path(tempfile.mkdtemp(prefix="s60-ambiguity-"))
        shutil.copytree(S60, root / "s60")
        try:
            profile_path = root / "s60" / "profile.yaml"
            profile = json.loads(profile_path.read_text(encoding="utf-8"))
            ambiguous_definition = json.loads((root / "s60" / "probes" / "dot-fp16-ambiguous.json").read_text(encoding="utf-8"))
            profile["probe_catalog"] = profile["probe_catalog"] + [
                {
                    "probe_id": ambiguous_definition["probe_id"],
                    "definition_path": "probes/dot-fp16-ambiguous.json",
                    "definition_sha256": sha256_file(root / "s60" / "probes" / "dot-fp16-ambiguous.json"),
                }
            ]
            profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            modified = load_profile(profile_path)
            with self.assertRaisesRegex(ProbeValidationError, "ambiguous"):
                select_profile_probes(modified, [requirement], self.snapshot)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_unrelated_unknowns_are_never_enumerated_into_work(self):
        requirement = json.loads((S60 / "requirements" / "attention-dot-before-fallback.json").read_text(encoding="utf-8"))
        plan = select_profile_probes(self.profile, [requirement], self.snapshot)
        self.assertEqual(1, len(plan.selections))
        self.assertNotIn("layout.reshape.logical", [s.probe_id for s in plan.selections])

    def test_runtime_identity_mismatch_blocks_selection(self):
        requirement = json.loads((S60 / "requirements" / "attention-dot-before-fallback.json").read_text(encoding="utf-8"))
        snapshot = {**self.snapshot, "target_id": "other-device"}
        with self.assertRaisesRegex(ProbeValidationError, "environment-blocked"):
            select_profile_probes(self.profile, [requirement], snapshot)


if __name__ == "__main__":
    unittest.main()

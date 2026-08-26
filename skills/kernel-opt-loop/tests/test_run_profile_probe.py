import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

from run_profile_probe import run_profile_probe
from validate_probe import ProbeValidationError, validate_probe_run
from validate_profile import ProfileValidationError, load_profile

FIXTURES = Path(__file__).parent / "fixtures" / "vnext"
PROBES = FIXTURES / "probes"
S60 = PROBES / "qualification" / "s60"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RunProfileProbeTests(unittest.TestCase):
    def setUp(self):
        self.output_root = Path(tempfile.mkdtemp(prefix="probe-runs-"))
        self.profile_root = Path(tempfile.mkdtemp(prefix="probe-profile-"))
        shutil.copytree(PROBES / "profile", self.profile_root / "profile")
        self.profile_path = self.profile_root / "profile" / "profile.yaml"
        self.runtime_snapshot_path = PROBES / "runtime-snapshot.json"

    def tearDown(self):
        shutil.rmtree(self.output_root, ignore_errors=True)
        shutil.rmtree(self.profile_root, ignore_errors=True)

    def materialize_variant(self, payload_name: str, *, timeout_seconds: float = 2.0) -> None:
        shutil.copyfile(
            PROBES / "profile" / "probes" / payload_name,
            self.profile_root / "profile" / "probes" / "fake-success.py",
        )
        definition_path = self.profile_root / "profile" / "probes" / "basic-memory.json"
        definition = json.loads(definition_path.read_text(encoding="utf-8"))
        for artifact in definition["input_artifacts"]:
            artifact["sha256"] = sha256_file(self.profile_root / "profile" / artifact["path"])
        definition["runner"]["timeout_seconds"] = timeout_seconds
        definition_path.write_text(json.dumps(definition, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        profile = json.loads(self.profile_path.read_text(encoding="utf-8"))
        profile["probe_catalog"] = [
            {"probe_id": "fixture-basic-memory-001", "definition_path": "probes/basic-memory.json", "definition_sha256": sha256_file(definition_path)}
        ]
        self.profile_path.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def test_fixture_probe_finishes_without_campaign_state(self):
        run_dir = run_profile_probe(
            profile_path=self.profile_path,
            probe_id="fixture-basic-memory-001",
            target_id="fixture-device",
            runtime_snapshot_path=self.runtime_snapshot_path,
            output_root=self.output_root,
            run_id="probe-001",
        )
        result = validate_probe_run(run_dir)
        self.assertEqual("evidence-ready", result["summary"])
        self.assertFalse((self.output_root / "team-state.md").exists())
        self.assertFalse((self.output_root / "rounds").exists())
        run_document = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        self.assertEqual("probe-001", run_document["run_id"])
        self.assertTrue(all(record["byte_count"] > 0 for record in run_document["inputs"].values()))

    def test_existing_run_id_is_never_overwritten(self):
        from vnext_common import ContractValidationError

        run_profile_probe(
            profile_path=self.profile_path,
            probe_id="fixture-basic-memory-001",
            target_id="fixture-device",
            runtime_snapshot_path=self.runtime_snapshot_path,
            output_root=self.output_root,
            run_id="probe-001",
        )
        with self.assertRaisesRegex(ContractValidationError, "refusing to reuse"):
            run_profile_probe(
                profile_path=self.profile_path,
                probe_id="fixture-basic-memory-001",
                target_id="fixture-device",
                runtime_snapshot_path=self.runtime_snapshot_path,
                output_root=self.output_root,
                run_id="probe-001",
            )

    def test_timeout_is_probe_failed(self):
        self.materialize_variant("fake-timeout.py", timeout_seconds=1)
        run_dir = run_profile_probe(
            profile_path=self.profile_path,
            probe_id="fixture-basic-memory-001",
            target_id="fixture-device",
            runtime_snapshot_path=self.runtime_snapshot_path,
            output_root=self.output_root,
            run_id="timeout-001",
        )
        self.assertEqual("probe-failed", validate_probe_run(run_dir)["summary"])

    def test_nonzero_exit_is_probe_failed(self):
        self.materialize_variant("fake-nonzero.py")
        run_dir = run_profile_probe(
            profile_path=self.profile_path,
            probe_id="fixture-basic-memory-001",
            target_id="fixture-device",
            runtime_snapshot_path=self.runtime_snapshot_path,
            output_root=self.output_root,
            run_id="nonzero-001",
        )
        self.assertEqual("probe-failed", validate_probe_run(run_dir)["summary"])

    def test_malformed_payload_is_probe_failed(self):
        self.materialize_variant("fake-malformed.py")
        run_dir = run_profile_probe(
            profile_path=self.profile_path,
            probe_id="fixture-basic-memory-001",
            target_id="fixture-device",
            runtime_snapshot_path=self.runtime_snapshot_path,
            output_root=self.output_root,
            run_id="malformed-001",
        )
        self.assertEqual("probe-failed", validate_probe_run(run_dir)["summary"])

    def test_target_profile_mismatch_is_environment_blocked(self):
        run_dir = run_profile_probe(
            profile_path=self.profile_path,
            probe_id="fixture-basic-memory-001",
            target_id="wrong-target",
            runtime_snapshot_path=self.runtime_snapshot_path,
            output_root=self.output_root,
            run_id="blocked-001",
        )
        self.assertEqual("environment-blocked", validate_probe_run(run_dir)["summary"])

    def test_missing_interpreter_is_environment_blocked(self):
        snapshot_path = self.profile_root / "runtime-snapshot.json"
        snapshot = json.loads(self.runtime_snapshot_path.read_text(encoding="utf-8"))
        snapshot["interpreter"] = "/nonexistent/interpreter-binary"
        snapshot_path.write_text(json.dumps(snapshot), encoding="utf-8")
        run_dir = run_profile_probe(
            profile_path=self.profile_path,
            probe_id="fixture-basic-memory-001",
            target_id="fixture-device",
            runtime_snapshot_path=snapshot_path,
            output_root=self.output_root,
            run_id="blocked-002",
        )
        self.assertEqual("environment-blocked", validate_probe_run(run_dir)["summary"])

    def test_frozen_inputs_validate_after_canonical_change(self):
        run_dir = run_profile_probe(
            profile_path=self.profile_path,
            probe_id="fixture-basic-memory-001",
            target_id="fixture-device",
            runtime_snapshot_path=self.runtime_snapshot_path,
            output_root=self.output_root,
            run_id="probe-002",
        )
        # Mutate the canonical profile: the frozen run must still validate.
        profile = json.loads(self.profile_path.read_text(encoding="utf-8"))
        profile["profile_status"] = "complete"
        self.profile_path.write_text(json.dumps(profile), encoding="utf-8")
        self.assertEqual("evidence-ready", validate_probe_run(run_dir)["summary"])

    def test_successful_run_creates_no_campaign_or_benchmark_artifacts(self):
        run_dir = run_profile_probe(
            profile_path=self.profile_path,
            probe_id="fixture-basic-memory-001",
            target_id="fixture-device",
            runtime_snapshot_path=self.runtime_snapshot_path,
            output_root=self.output_root,
            run_id="probe-003",
        )
        for name in ("team-state.md", "project.md", "rounds", "state", "baseline_adapter.py"):
            self.assertFalse((self.output_root / name).exists(), name)
        self.assertTrue((run_dir / "results" / "fixture-basic-memory-001.json").is_file())


class S60QualificationRunTests(unittest.TestCase):
    def setUp(self):
        self.output_root = Path(tempfile.mkdtemp(prefix="s60-runs-"))
        self.s60_root = Path(tempfile.mkdtemp(prefix="s60-profile-"))
        shutil.copytree(S60, self.s60_root / "s60")
        self.profile_path = self.s60_root / "s60" / "profile.yaml"
        self.runtime_path = self.s60_root / "s60" / "runtime-snapshot.json"
        self.requirement_path = self.s60_root / "s60" / "requirements" / "attention-dot-before-fallback.json"

    def tearDown(self):
        shutil.rmtree(self.output_root, ignore_errors=True)
        shutil.rmtree(self.s60_root, ignore_errors=True)

    def test_s60_dot_qualification_run_has_qualification_requirement_frozen(self):
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
            run_id="s60-dot-qualification-001",
        )
        self.assertEqual("evidence-ready", validate_probe_run(run_dir)["summary"])
        self.assertTrue((run_dir / "inputs" / "qualification-requirement.json").is_file())
        self.assertFalse((self.output_root / "team-state.md").exists())


if __name__ == "__main__":
    unittest.main()

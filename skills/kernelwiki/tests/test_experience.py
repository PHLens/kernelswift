from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


TESTS = Path(__file__).resolve().parent
SKILL_ROOT = TESTS.parent
SCRIPTS = SKILL_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(TESTS))

from campaign_contract_bridge import LoopContractIdentity  # noqa: E402
from campaign_fixture_factory import materialize_vnext_bundle  # noqa: E402
from campaign_import import TerminalBundle, TerminalStateEvidence, ValidatedCampaign  # noqa: E402
from experience import build_experience_proposal, write_proposal  # noqa: E402
from kernelwiki_common import KernelWikiError, canonical_json_bytes, sha256_bytes  # noqa: E402


FORBIDDEN = {"next_candidate", "recommended_next_change", "implementation_instruction"}


def make_campaign(
    *,
    terminal: str = "accepted",
    classification: str = "none",
    route: str = "proceed",
    facts: dict[str, object] | None = None,
    missing: tuple[str, ...] = (),
) -> ValidatedCampaign:
    identity = LoopContractIdentity(
        repository_commit="a" * 40,
        skill_tree_sha="b" * 40,
        validator_sha256={"validate_profile": "c" * 64},
        schema_sha256={},
    )
    bundle = TerminalBundle(
        schema_version=1,
        proposal_id="ignored-input-id",
        repository_root=Path("/tmp/example-repository"),
        project_root=Path("/tmp/example-repository/project"),
        contract_version=3,
        loop_contract_identity=identity,
        round_id="001",
        terminal_commit="d" * 40,
        terminal_result=terminal,
        measurement_exclusive=False,
        artifacts={},
        canonical_candidate_ref="candidate.py",
        canonical_report_ref="rounds/report_001.md",
    )
    base_facts: dict[str, object] = {
        "measurement_fingerprint": "bench-fixture-sha256",
        "comparability": "comparable",
        "performance_status": "improved",
        "correctness": {"status": "pass"},
        "lowering": {"status": "observed"},
        "measurements": [
            {"metric": "wall_time", "value": -10.0, "statistic": "delta_pct", "unit": "percent"},
            {"metric": "device_time", "value": -8.0, "statistic": "delta_pct", "unit": "percent"},
        ],
    }
    if facts:
        base_facts.update(facts)
    return ValidatedCampaign(
        bundle=bundle,
        loop_contract_identity=identity,
        normalized_profile={
            "implementation_profile_id": "triton_mlu",
            "implementation_profile_version": 1,
            "profile_status": "partial",
            "implementation": {"language": "python", "backend": "triton"},
            "identity_match": {"permitted_device_architectures": ["mlu590"]},
        },
        normalized_claim={
            "claim": {
                "target_id": "mlu590",
                "runtime_fingerprint": "triton 3.6.0 / CoreX 4.4.0",
                "qualification_dispositions": [],
            }
        },
        normalized_sketch={
            "sketch": {
                "declarations": [{"dtype": "fp32", "shape": ["M", "N"]}],
                "operations": [{"id": "op.load.row"}, {"id": "op.store.output"}],
            }
        },
        normalized_decision={
            "metadata": {"decision": route, "change_family": "kernel-fusion"},
            "optimization_intent": {
                "bottleneck_class": "launch-bound",
                "intervention": "fuse the measured launch sequence",
                "expected_wall_improvement_pct": 10,
            },
            "evaluation_contract": {"causal_graph": {"nodes": ["m.fusion", "o.kernel-count", "p.wall-time"]}},
        },
        normalized_binding={"valid": True},
        fact_pack=base_facts,
        normalized_verdict={
            "valid": True,
            "classification": classification,
            "terminal_result": terminal,
            "route": route,
        },
        terminal_state=TerminalStateEvidence(
            workflow_status="running",
            phase="ready",
            last_completed_round="001",
            last_result=terminal,
            measurement_exclusive=False,
            last_accepted_candidate="candidate.py",
            last_accepted_report="rounds/report_001.md",
        ),
        artifact_hashes={"candidate": "e" * 64, "report": "f" * 64},
        missing_evidence=missing,
    )


def snapshot_trees() -> dict[str, bytes]:
    snapshot: dict[str, bytes] = {}
    for root_name in ("sources", "wiki", "queries", "compiled"):
        root = SKILL_ROOT / root_name
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            snapshot[path.relative_to(SKILL_ROOT).as_posix()] = path.read_bytes()
    return snapshot


def assert_no_forbidden(test: unittest.TestCase, value: object) -> None:
    if isinstance(value, dict):
        test.assertFalse(FORBIDDEN & set(value))
        for nested in value.values():
            assert_no_forbidden(test, nested)
    elif isinstance(value, list):
        for nested in value:
            assert_no_forbidden(test, nested)


class ExperienceTests(unittest.TestCase):
    def test_version_three_mapping_table(self):
        cases = (
            ("accepted comparable improvement", {}, ("include", "positive", "performance")),
            ("slower no improvement", {"terminal": "no-improvement", "facts": {"performance_status": "no-improvement", "measurements": [{"metric": "wall_time", "value": 0.0, "statistic": "delta_pct", "unit": "percent"}]}}, ("include", "counterexample", "performance")),
            ("screened out", {"terminal": "screened-out"}, ("include", "counterexample", "screening")),
            ("device win wall loss", {"facts": {"measurements": [{"metric": "device_time", "value": -5.0, "statistic": "delta_pct", "unit": "percent"}, {"metric": "wall_time", "value": 4.0, "statistic": "delta_pct", "unit": "percent"}]}}, ("include", "counterexample", "device-wall-mismatch")),
            ("Designer semantic rejection", {"terminal": "design-rejected", "classification": "semantic-rejection", "route": "abort"}, ("include", "counterexample", "design-pitfall")),
            ("Coder implementation failure", {"terminal": "implementation-failed", "classification": "implementation-failure", "route": "proceed"}, ("include", "counterexample", "implementation-pitfall")),
            ("Unknown capability", {"facts": {"required_capability_status": "unknown"}}, ("include", "capability-gap", "profile")),
            ("unsupported capability", {"facts": {"required_capability_status": "unsupported"}}, ("include", "capability-gap", "profile")),
            ("probe only", {"terminal": "probe-only"}, ("defer", None, None)),
            ("environment blocked", {"terminal": "environment-blocked"}, ("defer", None, None)),
            ("incomplete", {"missing": ("incomplete-evidence",)}, ("defer", None, None)),
            ("measurement fingerprint missing", {"facts": {"measurement_fingerprint": None}}, ("defer", None, None)),
        )
        for label, options, expected in cases:
            with self.subTest(row=label):
                proposal = build_experience_proposal(make_campaign(**options))
                publication = proposal.suggested_publication
                self.assertEqual(expected, (publication["decision"], publication["role"], publication["subtype"]))
                if label == "Unknown capability":
                    statuses = [item["value"] for item in proposal.observed if item["metric"] == "required_capability_status"]
                    self.assertEqual(["unknown"], statuses)

    def test_deterministic_id_scope_hashes_and_identity(self):
        campaign = make_campaign()
        first = build_experience_proposal(campaign)
        second = build_experience_proposal(campaign)
        expected_id = "experience-" + sha256_bytes(canonical_json_bytes({
            "contract_version": 3,
            "terminal_commit": "d" * 40,
            "round_id": "001",
            "terminal_result": "accepted",
        }))[:20]
        self.assertEqual(expected_id, first.proposal_id)
        self.assertEqual(first, second)
        self.assertEqual("mlu590", first.scope["target_id"])
        self.assertEqual("triton_mlu", first.scope["implementation_profile_id"])
        self.assertEqual(["fp32"], first.scope["dtypes"])
        self.assertEqual("bench-fixture-sha256", first.scope["measurement_fingerprint"])
        self.assertEqual("comparable", first.scope["comparability"])
        self.assertIn("attribution", first.terminal)
        self.assertEqual({"candidate": "e" * 64, "report": "f" * 64}, first.artifact_hashes)
        self.assertEqual("b" * 40, first.loop_contract_identity["skill_tree_sha"])

    def test_recursive_forbidden_instruction_keys_are_rejected(self):
        proposal = build_experience_proposal(make_campaign())
        assert_no_forbidden(self, proposal.to_document())
        poisoned = replace(proposal, expected={"nested": {"implementation_instruction": "change the next kernel"}})
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(KernelWikiError) as caught:
                write_proposal(poisoned, Path(temporary) / "proposal.json")
        self.assertEqual("proposal-forbidden", caught.exception.code)

    def test_proposal_only_trees_remain_immutable_and_cli_has_no_publisher_import(self):
        before = snapshot_trees()
        proposal = build_experience_proposal(make_campaign())
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "proposal.json"
            write_proposal(proposal, output)
            poisoned = replace(proposal, expected={"next_candidate": "forbidden"})
            with self.assertRaises(KernelWikiError):
                write_proposal(poisoned, Path(temporary) / "bad.json")
        self.assertEqual(before, snapshot_trees())

        tree = ast.parse((SCRIPTS / "propose_from_campaign.py").read_text(encoding="utf-8"))
        imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and isinstance(node.module, str)
        }
        self.assertFalse(imports & {"catalog", "corpus", "source_capture", "capture_source", "generate_indices"})

    def test_cli_smoke_rejects_overwrite_and_campaign_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, manifest = materialize_vnext_bundle(Path(temporary) / "repo")
            output = Path(temporary) / "experience.json"
            command = [sys.executable, str(SCRIPTS / "propose_from_campaign.py"), "--bundle", str(manifest), "--output", str(output)]
            completed = subprocess.run(command, cwd=SKILL_ROOT.parent.parent, capture_output=True, text=True, check=False)
            self.assertEqual(0, completed.returncode, completed.stderr)
            receipt = json.loads(completed.stdout)
            self.assertEqual(str(output), receipt["output_path"])
            self.assertEqual(sha256_bytes(output.read_bytes()), receipt["proposal_sha256"])

            repeated = subprocess.run(command, cwd=SKILL_ROOT.parent.parent, capture_output=True, text=True, check=False)
            self.assertEqual(2, repeated.returncode)
            self.assertIn("error[proposal-exists]", repeated.stderr)

            forbidden = command[:-1] + [str(root / "project" / "state" / "proposal.json")]
            denied = subprocess.run(forbidden, cwd=SKILL_ROOT.parent.parent, capture_output=True, text=True, check=False)
            self.assertEqual(2, denied.returncode)
            self.assertIn("error[proposal-output-forbidden]", denied.stderr)


if __name__ == "__main__":
    unittest.main()

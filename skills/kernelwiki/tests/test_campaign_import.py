from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest

import yaml


TESTS = Path(__file__).resolve().parent
SCRIPTS = TESTS.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(TESTS))

from campaign_fixture_factory import (  # noqa: E402
    materialize_terminal_bundle,
    materialize_vnext_bundle,
    recommit_artifact,
    run_git,
    write_manifest,
)
from campaign_contract_bridge import (  # noqa: E402
    LoopContractIdentity,
    compute_loop_contract_identity,
    load_validator_module,
    loop_root,
)
from campaign_import import (  # noqa: E402
    coder_result_required,
    load_committed_artifact,
    load_terminal_bundle,
    validate_campaign,
    validate_git_identity,
)
from kernelwiki_common import KernelWikiError, sha256_bytes  # noqa: E402


class CampaignImportTests(unittest.TestCase):
    def test_valid_bundle_pins_terminal_commit_and_committed_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, manifest = materialize_terminal_bundle(Path(temporary) / "repo")
            bundle = load_terminal_bundle(manifest)

            self.assertEqual((), validate_git_identity(bundle))
            self.assertEqual(run_git(root, "rev-parse", "HEAD"), bundle.terminal_commit)
            self.assertEqual(root / "project", bundle.project_root)
            self.assertEqual(
                bundle.artifacts["candidate"].sha256,
                sha256_bytes(load_committed_artifact(bundle, "candidate")),
            )

    def test_post_commit_worktree_change_is_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, manifest = materialize_terminal_bundle(Path(temporary) / "repo")
            bundle = load_terminal_bundle(manifest)
            bundle.artifacts["candidate"].path.write_text("changed after terminal commit\n", encoding="utf-8")

            diagnostics = validate_git_identity(bundle)

            self.assertIn("worktree-diverged:candidate", diagnostics)
            self.assertEqual(
                bundle.artifacts["candidate"].sha256,
                sha256_bytes(load_committed_artifact(bundle, "candidate")),
            )

    def test_committed_bundle_failures(self):
        cases = (
            ("absent commit", "terminal-commit-absent"),
            ("absent artifact", "artifact-absent"),
            ("hash mismatch", "artifact-hash-mismatch"),
            ("canonical pointer", "canonical-pointer-invalid"),
        )
        for label, expected_code in cases:
            with self.subTest(row=label), tempfile.TemporaryDirectory() as temporary:
                _, manifest = materialize_terminal_bundle(Path(temporary) / "repo")
                document = yaml.safe_load(manifest.read_text(encoding="utf-8"))
                if label == "absent commit":
                    document["terminal_commit"] = "f" * 40
                elif label == "absent artifact":
                    document["artifacts"]["candidate"]["path"] = "project/absent.py"
                    document["artifacts"]["candidate"]["sha256"] = "0" * 64
                    document["canonical_candidate_ref"] = "absent.py"
                elif label == "hash mismatch":
                    document["artifacts"]["candidate"]["sha256"] = "0" * 64
                else:
                    document["canonical_candidate_ref"] = "rounds/report_001.md"
                write_manifest(manifest, document)

                with self.assertRaises(KernelWikiError) as caught:
                    bundle = load_terminal_bundle(manifest)
                    validate_git_identity(bundle)
                self.assertEqual(expected_code, caught.exception.code)

    def test_bundle_root_confinement(self):
        cases = ("project path", "artifact path", "symlink escape")
        for label in cases:
            with self.subTest(row=label), tempfile.TemporaryDirectory() as temporary:
                root, manifest = materialize_terminal_bundle(Path(temporary) / "repo")
                document = deepcopy(yaml.safe_load(manifest.read_text(encoding="utf-8")))
                if label == "project path":
                    document["project_root"] = "../outside"
                    write_manifest(manifest, document)
                elif label == "artifact path":
                    document["artifacts"]["candidate"]["path"] = "../outside.py"
                    write_manifest(manifest, document)
                else:
                    outside = Path(temporary) / "outside.py"
                    outside.write_text("outside\n", encoding="utf-8")
                    candidate = root / "project" / "candidate.py"
                    candidate.unlink()
                    candidate.symlink_to(outside)

                with self.assertRaises(KernelWikiError) as caught:
                    load_terminal_bundle(manifest)
                self.assertIn(caught.exception.code, {"lift-schema-invalid", "bundle-path-invalid", "bundle-path-escape"})

    def test_valid_vnext_chain_normalizes_all_stages(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, manifest = materialize_vnext_bundle(Path(temporary) / "repo")
            validated = validate_campaign(load_terminal_bundle(manifest))

            self.assertEqual("triton_mlu", validated.normalized_profile["implementation_profile_id"])
            self.assertEqual("mlu590", validated.normalized_claim["claim"]["target_id"])
            self.assertEqual("accepted", validated.normalized_verdict["terminal_result"])
            self.assertEqual("001", validated.terminal_state.last_completed_round)
            self.assertEqual(compute_loop_contract_identity(), validated.loop_contract_identity)
            self.assertIn("verdict-round-profile-target-not-modeled", validated.missing_evidence)

    def test_each_validator_stage_fails_closed(self):
        cases = (
            ("profile", "implementation_profile", lambda _data: b"{}\n", "campaign-profile-invalid"),
            ("claim", "project_claim", lambda _data: b"{}\n", "campaign-claim-invalid"),
            ("Sketch", "sketch", lambda _data: b"{}\n", "campaign-sketch-invalid"),
            ("Decision", "decision", lambda _data: b"# Broken Decision\n", "campaign-decision-invalid"),
            ("binding", "binding", lambda _data: b"{}\n", "campaign-binding-invalid"),
            ("fact pack", "report", lambda _data: b"# Report 001\n", "campaign-fact-pack-invalid"),
            ("verdict", "verdict", lambda _data: b"{}\n", "campaign-verdict-invalid"),
        )
        for label, artifact, transform, expected_code in cases:
            with self.subTest(stage=label), tempfile.TemporaryDirectory() as temporary:
                root, manifest = materialize_vnext_bundle(Path(temporary) / "repo")
                recommit_artifact(root, manifest, artifact, transform)
                with self.assertRaises(KernelWikiError) as caught:
                    validate_campaign(load_terminal_bundle(manifest))
                self.assertEqual(expected_code, caught.exception.code)

    def test_terminal_state_mismatches_fail_stably(self):
        cases = (
            ("round", b'last_completed_round: "002"', "terminal-round-mismatch"),
            ("result", b"last_result: no-improvement", "terminal-result-mismatch"),
            ("canonical pointer", b"last_accepted_kernel: other.py", "canonical-pointer-mismatch"),
            ("measurement", b"measurement_exclusive: true", "measurement-exclusive"),
        )
        for label, replacement, expected_code in cases:
            with self.subTest(row=label), tempfile.TemporaryDirectory() as temporary:
                root, manifest = materialize_vnext_bundle(Path(temporary) / "repo")

                def mutate(data: bytes, *, row: str = label, value: bytes = replacement) -> bytes:
                    needles = {
                        "round": b'last_completed_round: "001"',
                        "result": b"last_result: accepted",
                        "canonical pointer": b"last_accepted_kernel: candidate.py",
                        "measurement": b"measurement_exclusive: false",
                    }
                    return data.replace(needles[row], value)

                recommit_artifact(root, manifest, "team_state", mutate)
                with self.assertRaises(KernelWikiError) as caught:
                    validate_campaign(load_terminal_bundle(manifest))
                self.assertEqual(expected_code, caught.exception.code)

    def test_contract_authority_is_fixed_and_versioned(self):
        expected_root = Path(__file__).resolve().parents[2] / "kernel-opt-loop"
        self.assertEqual(expected_root, loop_root())
        with self.assertRaises(KernelWikiError) as denied:
            load_validator_module("os")
        self.assertEqual("contract-validator-denied", denied.exception.code)

        with tempfile.TemporaryDirectory() as temporary:
            _, manifest = materialize_vnext_bundle(Path(temporary) / "repo")
            bundle = load_terminal_bundle(manifest)
            validated = validate_campaign(bundle)
            self.assertEqual(bundle.loop_contract_identity, validated.loop_contract_identity)

            mismatch = replace(
                bundle,
                loop_contract_identity=LoopContractIdentity(
                    repository_commit=bundle.loop_contract_identity.repository_commit,
                    skill_tree_sha=bundle.loop_contract_identity.skill_tree_sha,
                    validator_sha256={"validate_profile": "0" * 64},
                    schema_sha256={},
                ),
            )
            with self.assertRaises(KernelWikiError) as identity_error:
                validate_campaign(mismatch)
            self.assertEqual("contract-unsupported", identity_error.exception.code)

        with self.assertRaises(KernelWikiError) as version_error:
            coder_result_required(4, "accepted", "proceed")
        self.assertEqual("contract-unsupported", version_error.exception.code)

    def test_coder_result_requirement_matrix(self):
        cases = (
            ("proceed", "accepted", True),
            ("proceed", "no-improvement", True),
            ("proceed", "screened-out", True),
            ("abort", "accepted", False),
            ("proceed", "environment-blocked", False),
            ("proceed", "design-rejected", None),
        )
        for route, terminal, expected in cases:
            with self.subTest(route=route, terminal=terminal):
                if expected is None:
                    with self.assertRaises(KernelWikiError) as caught:
                        coder_result_required(3, terminal, route)
                    self.assertEqual("contract-unsupported", caught.exception.code)
                else:
                    self.assertIs(expected, coder_result_required(3, terminal, route))


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
TESTS = SKILL_ROOT / "tests"
FIXTURES = TESTS / "fixtures"
sys.path.insert(0, str(SCRIPTS))
sys.path.insert(0, str(TESTS))

from corpus import load_corpus  # noqa: E402
from kernelwiki_common import KernelWikiError, canonical_json_bytes, load_yaml_document, sha256_file  # noqa: E402
import role_context  # noqa: E402
from role_context import load_authority_snapshot, load_role_context  # noqa: E402
from role_fixture_factory import build_coder_context, materialize_vnext_project  # noqa: E402


class RoleContextTests(unittest.TestCase):
    def _write(self, root: Path, document: dict, name: str = "context.json") -> Path:
        path = root / name
        path.write_text(json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        return path

    def _valid_document(self, root: Path) -> tuple[Path, dict]:
        project = materialize_vnext_project(root / "project")
        return project, build_coder_context(project)

    def _refresh_artifact_hash(self, project: Path, document: dict, name: str) -> None:
        relative = Path(document["artifacts"][name]["path"])
        document["artifacts"][name]["sha256"] = sha256_file(project / relative)

    def _replace_decision_text(self, project: Path, document: dict, old: str, new: str) -> None:
        decision = project / "rounds" / "decision_001.md"
        text = decision.read_text(encoding="utf-8")
        self.assertIn(old, text)
        decision.write_text(text.replace(old, new, 1), encoding="utf-8")
        self._refresh_artifact_hash(project, document, "decision")

    def _assert_decision_artifact_field_rejected(self, artifact_name: str, field: str, value: str) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, document = self._valid_document(root)
            context = load_role_context(self._write(root, document))
            profile_module = role_context.load_loop_module("validate_profile")
            sketch_module = role_context.load_loop_module("validate_sketch")
            real_decision_module = role_context.load_loop_module("validate_decision")
            decision_module = ModuleType("mutated-decision")

            def validate_decision(*args, **kwargs):
                result = dict(real_decision_module.validate_decision(*args, **kwargs))
                result["metadata"] = dict(result["metadata"])
                result[field] = value
                result["metadata"][field] = value
                return result

            decision_module.validate_decision = validate_decision

            @contextmanager
            def fake_modules(names):
                self.assertEqual(
                    ("validate_profile", "validate_sketch", "validate_decision"),
                    tuple(names),
                )
                yield context.loop_contract_identity, {
                    "validate_profile": profile_module,
                    "validate_sketch": sketch_module,
                    "validate_decision": decision_module,
                }

            with mock.patch("role_context.load_loop_modules", fake_modules):
                with self.assertRaisesRegex(
                    KernelWikiError,
                    rf"authority-invalid.*{field}.*{artifact_name}",
                ):
                    load_authority_snapshot(context)

    def test_designer_context_requires_no_loop_artifacts(self):
        context = load_role_context(FIXTURES / "role" / "designer-context.json")
        self.assertEqual("designer", context.role)
        self.assertEqual({}, context.artifacts)
        self.assertIsNone(context.loop_contract_identity)





    def test_missing_profile_does_not_inspect_validators_or_fallbacks(self):
        context = load_role_context(FIXTURES / "role" / "coder-missing-profile.json")
        with mock.patch("role_context.compute_loop_contract_identity", side_effect=AssertionError("identity inspected")), mock.patch(
            "role_context.load_loop_module", side_effect=AssertionError("validator inspected")
        ):
            with self.assertRaisesRegex(KernelWikiError, "profile-missing"):
                load_authority_snapshot(context)





    def test_artifact_path_escape_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, document = self._valid_document(root)
            document["artifacts"]["sketch"]["path"] = "../outside.json"
            with self.assertRaisesRegex(KernelWikiError, "artifact-path-escape"):
                load_role_context(self._write(root, document))







    def test_artifact_hash_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project, document = self._valid_document(root)
            context = load_role_context(self._write(root, document))
            (project / "rounds" / "sketch_001.json").write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(KernelWikiError, "artifact-hash-mismatch"):
                load_authority_snapshot(context)





    def test_valid_coder_authority_uses_current_validators(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, document = self._valid_document(root)
            context = load_role_context(self._write(root, document))
            result = load_authority_snapshot(context)
            self.assertEqual("triton_mlu", result.profile["implementation_profile_id"])
            self.assertEqual("partial", result.profile["profile_status"])
            self.assertTrue(result.project_claim["valid"])
            self.assertTrue(result.sketch_result["valid"])
            self.assertTrue(result.decision_result["valid"])
            self.assertEqual(set(document["artifacts"]), set(result.artifact_hashes))







    def test_coder_languages_must_exactly_match_profile_backend(self):
        for languages in (["python"], ["triton", "python"]):
            with self.subTest(languages=languages), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                _, document = self._valid_document(root)
                document["languages"] = languages
                context = load_role_context(self._write(root, document))
                with self.assertRaisesRegex(KernelWikiError, "profile-version-mismatch"):
                    load_authority_snapshot(context)

    def test_decision_sketch_ref_must_match_context_artifact(self):
        self._assert_decision_artifact_field_rejected(
            "sketch",
            "sketch_ref",
            "rounds/alternate-sketch.json",
        )











    def test_runtime_fingerprint_must_use_snapshot_triton_version(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project, document = self._valid_document(root)
            runtime_path = project / "state" / "runtime-snapshot.json"
            runtime_document = json.loads(runtime_path.read_text(encoding="utf-8"))
            runtime_document["triton_version"] = "3.7.0"
            runtime_path.write_text(json.dumps(runtime_document, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
            self._refresh_artifact_hash(project, document, "runtime_snapshot")
            context = load_role_context(self._write(root, document))
            with self.assertRaisesRegex(KernelWikiError, "runtime-mismatch"):
                load_authority_snapshot(context)








if __name__ == "__main__":
    unittest.main()

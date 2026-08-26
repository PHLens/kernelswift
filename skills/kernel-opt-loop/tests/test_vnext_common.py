from pathlib import Path
import tempfile
import unittest

import sys

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

from vnext_common import (
    ContractValidationError,
    create_exclusive_directory,
    load_json_document,
    load_json_yaml_document,
    require_relative_artifact,
    sha256_canonical_json,
    sha256_file,
    validate_source_span,
    write_json_atomic,
)


class VNextCommonTests(unittest.TestCase):
    def test_json_yaml_accepts_json_subset_and_rejects_yaml_only_syntax(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "profile.yaml"
            valid.write_text('{"schema_version":1,"name":"triton_mlu"}\n', encoding="utf-8")
            self.assertEqual("triton_mlu", load_json_yaml_document(valid, artifact="profile")["name"])

            invalid = root / "invalid.yaml"
            invalid.write_text("schema_version: 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractValidationError, "json-compatible YAML"):
                load_json_yaml_document(invalid, artifact="profile")

    def test_reference_cannot_escape_project_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "rounds").mkdir()
            path = root / "rounds" / "sketch_001.json"
            path.write_text("{}", encoding="utf-8")
            self.assertEqual(path, require_relative_artifact(root, "rounds/sketch_001.json"))
            with self.assertRaisesRegex(ContractValidationError, "relative artifact"):
                require_relative_artifact(root, "../outside.json")

    def test_exclusive_directory_and_atomic_json_never_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = create_exclusive_directory(Path(directory) / "probe-001")
            write_json_atomic(run_dir / "run.json", {"b": 2, "a": 1})
            self.assertEqual('{"a":1,"b":2}\n', (run_dir / "run.json").read_text(encoding="utf-8"))
            with self.assertRaises(ContractValidationError):
                create_exclusive_directory(run_dir)

    def test_canonical_json_hash_is_key_order_independent(self):
        self.assertEqual(sha256_canonical_json({"b": 2, "a": 1}), sha256_canonical_json({"a": 1, "b": 2}))

    def test_source_span_requires_existing_one_based_range(self):
        validate_source_span("first\nsecond\n", {"start": [2, 1], "end": [2, 7]})
        with self.assertRaisesRegex(ContractValidationError, "source span"):
            validate_source_span("first\n", {"start": [2, 1], "end": [2, 2]})

    def test_json_document_requires_utf8_object(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            malformed = root / "malformed.json"
            malformed.write_bytes(b"\xff\xfe")
            with self.assertRaises(ContractValidationError):
                load_json_document(malformed, artifact="run")
            array = root / "array.json"
            array.write_text("[1,2]\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractValidationError, "JSON object"):
                load_json_document(array, artifact="run")

    def test_sha256_file_matches_read_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "blob.bin"
            path.write_bytes(b"kernel bytes")
            self.assertEqual(hashlib_sha256(b"kernel bytes"), sha256_file(path))

    def test_atomic_write_survives_directory_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = create_exclusive_directory(Path(directory) / "probe-002")
            nested = run_dir / "results"
            nested.mkdir()
            write_json_atomic(nested / "result.json", {"level": "observed"})
            self.assertEqual(
                {"level": "observed"},
                load_json_document(nested / "result.json", artifact="result"),
            )


def hashlib_sha256(payload: bytes) -> str:
    import hashlib

    return hashlib.sha256(payload).hexdigest()


if __name__ == "__main__":
    unittest.main()

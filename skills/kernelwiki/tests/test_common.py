from contextlib import redirect_stderr
import io
from pathlib import Path
import sys
import tempfile
import unittest

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

from kernelwiki_common import (  # noqa: E402
    KernelWikiError,
    canonical_json_bytes,
    load_yaml_document,
    parse_markdown,
    require_within,
    run_cli,
    sha256_bytes,
    sha256_file,
    write_text_atomic,
)


class CommonTests(unittest.TestCase):
    def test_parse_markdown_returns_frontmatter_and_body(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "page.md"
            path.write_text("---\nid: card-one\ntags: [fusion]\n---\n# Body\n", encoding="utf-8")
            metadata, body = parse_markdown(path)
            self.assertEqual("card-one", metadata["id"])
            self.assertEqual(["fusion"], metadata["tags"])
            self.assertEqual("# Body\n", body)

    def test_yaml_unsafe_python_tag_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.yaml"
            path.write_text("!!python/object/apply:os.system ['false']\n", encoding="utf-8")
            with self.assertRaisesRegex(KernelWikiError, "yaml-invalid"):
                load_yaml_document(path)

    def test_require_within_rejects_parent_escape(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(KernelWikiError, "path-escape"):
                require_within(root, root / ".." / "outside")

    def test_parse_markdown_requires_frontmatter(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "page.md"
            path.write_text("# Body\n", encoding="utf-8")
            with self.assertRaisesRegex(KernelWikiError, "frontmatter-missing"):
                parse_markdown(path)





    def test_canonical_json_bytes_are_sorted_and_newline_terminated(self):
        self.assertEqual(b'{"a":2,"z":1}\n', canonical_json_bytes({"z": 1, "a": 2}))

    def test_sha256_bytes_and_file_are_stable(self):
        expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        self.assertEqual(expected, sha256_bytes(b"abc"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "payload.bin"
            path.write_bytes(b"abc")
            self.assertEqual(expected, sha256_file(path))

    def test_write_text_atomic_replaces_content_without_temporary_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "nested" / "output.txt"
            write_text_atomic(path, "first\n")
            write_text_atomic(path, "second\n")
            self.assertEqual("second\n", path.read_text(encoding="utf-8"))
            self.assertEqual([path], sorted(item for item in root.rglob("*") if item.is_file()))


    def test_run_cli_formats_kernelwiki_error_and_returns_two(self):
        def failing_main(argv):
            self.assertEqual(["--bad"], argv)
            raise KernelWikiError("input-invalid", "bad request", Path("request.json"))

        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = run_cli(failing_main, ["--bad"])
        self.assertEqual(2, result)
        self.assertEqual("error[input-invalid]: bad request (request.json)\n", stderr.getvalue())



if __name__ == "__main__":
    unittest.main()

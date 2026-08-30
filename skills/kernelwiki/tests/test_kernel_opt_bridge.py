from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest import mock


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import kernel_opt_bridge as bridge  # noqa: E402
from kernel_opt_bridge import (  # noqa: E402
    ALLOWED_MODULES,
    CONSUMED_SCHEMA_FILES,
    LOOP_ROOT,
    compute_loop_contract_identity,
    load_loop_module,
)
from kernelwiki_common import KernelWikiError  # noqa: E402


class KernelOptBridgeTests(unittest.TestCase):
    def test_non_allowlisted_loop_module_is_denied(self):
        with self.assertRaisesRegex(KernelWikiError, "contract-module-denied"):
            load_loop_module("validate_binding")


    def test_loop_root_is_fixed_sibling(self):
        self.assertEqual(SKILL_ROOT.parent / "kernel-opt-loop", LOOP_ROOT)


    def test_identity_pins_loop_commit_and_tree(self):
        identity = compute_loop_contract_identity()
        self.assertRegex(identity.repository_commit, r"^[0-9a-f]{40}$")
        self.assertRegex(identity.skill_tree_sha, r"^[0-9a-f]{40}$")






    def test_mutable_worktree_root_is_not_execution_authority(self):
        with tempfile.TemporaryDirectory() as directory:
            poisoned_root = Path(directory) / "kernel-opt-loop"
            (poisoned_root / "scripts").mkdir(parents=True)
            (poisoned_root / "scripts" / "validate_sketch.py").write_text("raise AssertionError('poison')\n", encoding="utf-8")
            with mock.patch("kernel_opt_bridge.LOOP_ROOT", poisoned_root):
                module = load_loop_module("validate_sketch")
        self.assertTrue(callable(module.validate_sketch))


    def _synthetic_authority(self):
        module_bytes = {
            "vnext_common": b"",
            "validate_probe": b"MARKER = 'snapshot'\n",
            "validate_profile": (
                b"PUBLIC = {'items': [1], 'tags': {'snapshot'}}\n"
                b"def load_profile(path):\n"
                b"    import validate_probe\n"
                b"    return validate_probe.MARKER\n"
                b"class Factory:\n"
                b"    def __new__(cls):\n"
                b"        import validate_probe\n"
                b"        return validate_probe.MARKER\n"
            ),
            "validate_sketch": b"def validate_sketch(path):\n    return {'valid': True}\n",
            "validate_decision": b"def validate_decision(path, **kwargs):\n    return {'valid': True}\n",
        }
        validators = {
            name: hashlib.sha256(module_bytes[name]).hexdigest()
            for name in sorted(ALLOWED_MODULES)
        }
        identity = bridge.LoopContractIdentity(
            repository_commit="1" * 40,
            skill_tree_sha="2" * 40,
            validator_sha256=validators,
            schema_sha256={},
        )
        return bridge._CommittedLoopAuthority(identity=identity, module_bytes=module_bytes)

    def test_proxy_callable_reopens_fresh_snapshot_for_lazy_imports(self):
        authority = self._synthetic_authority()
        poisoned = ModuleType("validate_probe")
        poisoned.MARKER = "ambient"
        poisoned.__file__ = "/tmp/validate_probe.py"
        with mock.patch.object(bridge, "_resolve_committed_authority", return_value=authority):
            module = load_loop_module("validate_profile")
            with mock.patch.dict(sys.modules, {"validate_probe": poisoned}):
                self.assertEqual("snapshot", module.load_profile(None))
                self.assertEqual("snapshot", module.Factory())
                self.assertIs(poisoned, sys.modules["validate_probe"])
            self.assertEqual("snapshot", module.load_profile(None))





if __name__ == "__main__":
    unittest.main()

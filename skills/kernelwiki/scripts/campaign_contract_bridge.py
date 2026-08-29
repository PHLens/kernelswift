from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from typing import Mapping

from kernelwiki_common import KernelWikiError


LOOP_ROOT = Path(__file__).resolve().parents[2] / "kernel-opt-loop"
LOOP_SCRIPTS = LOOP_ROOT / "scripts"
CHECKOUT_ROOT = LOOP_ROOT.parents[1]
VALIDATOR_MODULES: Mapping[str, tuple[str, ...]] = {
    "validate_profile": ("load_profile", "validate_project_claim"),
    "validate_sketch": ("validate_sketch",),
    "validate_decision": ("validate_decision",),
    "validate_binding": ("validate_binding",),
    "validate_verdict": ("extract_verifier_fact_pack", "validate_verdict"),
}
_MODULE_CACHE: dict[str, ModuleType] = {}


@dataclass(frozen=True)
class LoopContractIdentity:
    repository_commit: str
    skill_tree_sha: str
    validator_sha256: Mapping[str, str]
    schema_sha256: Mapping[str, str]


def _fail(code: str, message: str, path: Path | None = None) -> None:
    raise KernelWikiError(code, message, path)


def _git_text(*args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(CHECKOUT_ROOT), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        _fail("contract-unsupported", completed.stderr.strip() or "cannot resolve loop contract", LOOP_ROOT)
    return completed.stdout.strip()


def loop_root() -> Path:
    return LOOP_ROOT


def load_validator_module(name: str) -> ModuleType:
    if name not in VALIDATOR_MODULES:
        _fail("contract-validator-denied", f"validator module is not allowlisted: {name}", LOOP_ROOT)
    if name in _MODULE_CACHE:
        return _MODULE_CACHE[name]

    scripts = str(LOOP_SCRIPTS)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    try:
        module = importlib.import_module(name)
    except Exception as error:
        _fail("contract-unsupported", f"cannot import {name}: {error}", LOOP_ROOT / "scripts" / f"{name}.py")

    missing = [function for function in VALIDATOR_MODULES[name] if not callable(getattr(module, function, None))]
    if missing:
        _fail("contract-unsupported", f"{name} is missing validator functions: {', '.join(missing)}")
    _MODULE_CACHE[name] = module
    return module


def compute_loop_contract_identity() -> LoopContractIdentity:
    validator_hashes = {
        name: hashlib.sha256((LOOP_SCRIPTS / f"{name}.py").read_bytes()).hexdigest()
        for name in sorted(VALIDATOR_MODULES)
    }
    return LoopContractIdentity(
        repository_commit=_git_text("log", "-1", "--format=%H", "--", "skills/kernel-opt-loop"),
        skill_tree_sha=_git_text("rev-parse", "HEAD:skills/kernel-opt-loop"),
        validator_sha256=validator_hashes,
        schema_sha256={},
    )


def validator_modules() -> Mapping[str, ModuleType]:
    return {name: load_validator_module(name) for name in VALIDATOR_MODULES}

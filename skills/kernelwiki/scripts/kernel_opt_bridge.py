from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import importlib
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
from types import MappingProxyType, ModuleType

from kernelwiki_common import KernelWikiError, sha256_bytes


LOOP_ROOT = Path(__file__).resolve().parents[2] / "kernel-opt-loop"
REPOSITORY_ROOT = LOOP_ROOT.parents[1]
ALLOWED_MODULES = frozenset({"validate_profile", "validate_sketch", "validate_decision"})
CONSUMED_SCHEMA_FILES: tuple[str, ...] = ()
_INTERNAL_RUNTIME_DEPENDENCIES = {
    "validate_profile": ("vnext_common", "validate_probe"),
    "validate_sketch": ("vnext_common",),
    "validate_decision": ("vnext_common", "validate_profile", "validate_sketch", "validate_probe"),
}
_SNAPSHOT_IMPORT_ORDER = (
    "vnext_common",
    "validate_probe",
    "validate_profile",
    "validate_sketch",
    "validate_decision",
)
_MANAGED_MODULES = frozenset(_SNAPSHOT_IMPORT_ORDER)
_GIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
_TREE_SHA_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_IMPORT_LOCK = threading.RLock()
_MISSING = object()


class _FrozenDigestMapping(Mapping[str, str]):
    __slots__ = ("_items", "_values")

    def __init__(self, value: Mapping[str, str], label: str):
        if not isinstance(value, Mapping):
            raise KernelWikiError("contract-identity-invalid", f"{label} must be an object")
        items: list[tuple[str, str]] = []
        for key, digest in value.items():
            if not isinstance(key, str) or not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                raise KernelWikiError("contract-identity-invalid", f"{label} contains an invalid digest")
            items.append((key, digest))
        object.__setattr__(self, "_items", tuple(sorted(items)))
        object.__setattr__(self, "_values", MappingProxyType(dict(self._items)))

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("loop contract digest mappings are immutable")

    def __delattr__(self, name: str) -> None:
        raise AttributeError("loop contract digest mappings are immutable")

    def __getitem__(self, key: str) -> str:
        return self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __deepcopy__(self, memo):
        return dict(self._items)


@dataclass(frozen=True)
class LoopContractIdentity:
    repository_commit: str
    skill_tree_sha: str
    validator_sha256: Mapping[str, str]
    schema_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        if not isinstance(self.repository_commit, str) or _GIT_SHA_RE.fullmatch(self.repository_commit) is None:
            raise KernelWikiError("contract-identity-invalid", "repository_commit must be a Git commit")
        if not isinstance(self.skill_tree_sha, str) or _TREE_SHA_RE.fullmatch(self.skill_tree_sha) is None:
            raise KernelWikiError("contract-identity-invalid", "skill_tree_sha must be a Git tree")
        object.__setattr__(
            self,
            "validator_sha256",
            _FrozenDigestMapping(self.validator_sha256, "validator_sha256"),
        )
        object.__setattr__(
            self,
            "schema_sha256",
            _FrozenDigestMapping(self.schema_sha256, "schema_sha256"),
        )


@dataclass(frozen=True)
class _CommittedLoopAuthority:
    identity: LoopContractIdentity
    module_bytes: Mapping[str, bytes]

    def __post_init__(self) -> None:
        if not isinstance(self.identity, LoopContractIdentity):
            raise KernelWikiError("contract-identity-invalid", "committed authority identity has an invalid type")
        identity = LoopContractIdentity(
            repository_commit=self.identity.repository_commit,
            skill_tree_sha=self.identity.skill_tree_sha,
            validator_sha256=self.identity.validator_sha256,
            schema_sha256=self.identity.schema_sha256,
        )
        if not isinstance(self.module_bytes, Mapping) or set(self.module_bytes) != set(_SNAPSHOT_IMPORT_ORDER):
            raise KernelWikiError("contract-module-invalid", "committed authority snapshot has invalid module fields")
        frozen_bytes: dict[str, bytes] = {}
        for name in _SNAPSHOT_IMPORT_ORDER:
            data = self.module_bytes[name]
            if not isinstance(data, bytes):
                raise KernelWikiError("contract-module-invalid", f"committed module {name!r} is not bytes")
            frozen_bytes[name] = bytes(data)
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "module_bytes", MappingProxyType(frozen_bytes))


def _run_git(*arguments: str) -> bytes:
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except (OSError, ValueError) as error:
        raise KernelWikiError("contract-git-failed", "cannot execute git for loop authority") from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise KernelWikiError("contract-git-failed", detail or "git could not resolve loop authority")
    return completed.stdout


def _git_ascii(*arguments: str, code: str = "contract-git-failed") -> str:
    try:
        return _run_git(*arguments).decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise KernelWikiError(code, "git returned non-ASCII authority identity") from error


def _resolve_committed_authority() -> _CommittedLoopAuthority:
    repository_head = _git_ascii("rev-parse", "HEAD")
    if _GIT_SHA_RE.fullmatch(repository_head) is None:
        raise KernelWikiError("contract-git-failed", "git returned invalid repository HEAD")
    tree = _git_ascii("rev-parse", f"{repository_head}:skills/kernel-opt-loop")
    commit = _git_ascii("log", "-1", "--format=%H", repository_head, "--", "skills/kernel-opt-loop")
    if _TREE_SHA_RE.fullmatch(tree) is None or _GIT_SHA_RE.fullmatch(commit) is None:
        raise KernelWikiError("contract-git-failed", "git returned invalid loop identity")

    module_bytes: dict[str, bytes] = {}
    for name in _SNAPSHOT_IMPORT_ORDER:
        module_bytes[name] = _run_git("show", f"{tree}:scripts/{name}.py")

    validator_hashes = {
        name: sha256_bytes(module_bytes[name])
        for name in sorted(ALLOWED_MODULES)
    }
    schema_hashes: dict[str, str] = {}
    for relative in CONSUMED_SCHEMA_FILES:
        schema_hashes[relative] = sha256_bytes(_run_git("show", f"{tree}:{relative}"))

    identity = LoopContractIdentity(
        repository_commit=commit,
        skill_tree_sha=tree,
        validator_sha256=dict(sorted(validator_hashes.items())),
        schema_sha256=dict(sorted(schema_hashes.items())),
    )
    return _CommittedLoopAuthority(
        identity=identity,
        module_bytes=MappingProxyType(dict(module_bytes)),
    )


def compute_loop_contract_identity() -> LoopContractIdentity:
    return _resolve_committed_authority().identity


def _validate_requested_modules(names: Sequence[str]) -> tuple[str, ...]:
    if isinstance(names, (str, bytes)):
        raise KernelWikiError("contract-module-denied", "module names must be a sequence")
    output: list[str] = []
    for name in names:
        if not isinstance(name, str) or name not in ALLOWED_MODULES:
            raise KernelWikiError("contract-module-denied", f"module {name!r} is not allowlisted")
        if name not in output:
            output.append(name)
    if not output:
        raise KernelWikiError("contract-module-denied", "at least one allowlisted module is required")
    return tuple(output)


def _restore_snapshot_permissions(scripts: Path) -> None:
    try:
        scripts.chmod(0o700)
    except OSError:
        return
    for path in scripts.glob("*.py"):
        try:
            path.chmod(0o600)
        except OSError:
            pass


@contextmanager
def _load_authority_modules(
    authority: _CommittedLoopAuthority,
    names: Sequence[str],
) -> Iterator[Mapping[str, ModuleType]]:
    requested = _validate_requested_modules(names)
    with tempfile.TemporaryDirectory(prefix="kernelwiki-loop-snapshot-") as directory:
        scripts = Path(directory) / "scripts"
        scripts.mkdir(mode=0o700)
        for name, data in authority.module_bytes.items():
            path = scripts / f"{name}.py"
            path.write_bytes(data)
            path.chmod(0o400)
        scripts.chmod(0o500)

        with _IMPORT_LOCK:
            previous_path = list(sys.path)
            previous_modules = {name: sys.modules.get(name, _MISSING) for name in _MANAGED_MODULES}
            previous_dont_write_bytecode = sys.dont_write_bytecode
            for name in _MANAGED_MODULES:
                sys.modules.pop(name, None)
            sys.path.insert(0, str(scripts))
            sys.dont_write_bytecode = True
            loaded: dict[str, ModuleType] = {}
            try:
                for name in _SNAPSHOT_IMPORT_ORDER:
                    module = importlib.import_module(name)
                    module_file = getattr(module, "__file__", None)
                    expected_path = scripts / f"{name}.py"
                    if not isinstance(module_file, str) or Path(module_file).resolve() != expected_path.resolve():
                        raise KernelWikiError(
                            "contract-module-invalid",
                            f"module {name!r} did not load from the committed private snapshot",
                        )
                    try:
                        loaded_bytes = expected_path.read_bytes()
                    except OSError as error:
                        raise KernelWikiError("contract-module-invalid", f"cannot verify module {name!r}") from error
                    if loaded_bytes != authority.module_bytes[name]:
                        raise KernelWikiError("contract-unsupported", f"snapshot bytes changed for module {name!r}")
                    loaded[name] = module
                yield MappingProxyType({name: loaded[name] for name in requested})
            except KernelWikiError:
                raise
            except Exception as error:
                raise KernelWikiError("contract-module-invalid", f"cannot execute committed loop snapshot: {error}") from error
            finally:
                for name in _MANAGED_MODULES:
                    sys.modules.pop(name, None)
                for name, previous in previous_modules.items():
                    if previous is not _MISSING:
                        sys.modules[name] = previous
                sys.path[:] = previous_path
                sys.dont_write_bytecode = previous_dont_write_bytecode
                _restore_snapshot_permissions(scripts)


@contextmanager
def load_loop_modules(names: Sequence[str]) -> Iterator[tuple[LoopContractIdentity, Mapping[str, ModuleType]]]:
    authority = _resolve_committed_authority()
    with _load_authority_modules(authority, names) as modules:
        yield authority.identity, modules


def _freeze_exported_value(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_exported_value(item) for key, item in value.items()})
    if isinstance(value, tuple):
        return tuple(_freeze_exported_value(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze_exported_value(item) for item in value)
    if isinstance(value, frozenset):
        return frozenset(_freeze_exported_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_exported_value(item) for item in value)
    return value


class _SnapshotModuleProxy(ModuleType):
    def __init__(self, name: str):
        super().__init__(name)
        ModuleType.__setattr__(self, "_snapshot_proxy_sealed", False)

    def __getattribute__(self, name: str):
        if name == "__dict__" and ModuleType.__getattribute__(self, "_snapshot_proxy_sealed"):
            return MappingProxyType(ModuleType.__getattribute__(self, "__dict__"))
        return ModuleType.__getattribute__(self, name)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_snapshot_proxy_sealed", False):
            raise AttributeError("committed validator proxy is immutable")
        ModuleType.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        if getattr(self, "_snapshot_proxy_sealed", False):
            raise AttributeError("committed validator proxy is immutable")
        ModuleType.__delattr__(self, name)


def _snapshot_callable(
    module_name: str,
    attribute_name: str,
    captured_authority: _CommittedLoopAuthority,
):
    def invoke(*args, **kwargs):
        current_identity = _resolve_committed_authority().identity
        if current_identity != captured_authority.identity:
            raise KernelWikiError(
                "contract-unsupported",
                "committed loop authority changed after validator proxy creation",
            )
        with _load_authority_modules(captured_authority, (module_name,)) as modules:
            target = getattr(modules[module_name], attribute_name, None)
            if not callable(target):
                raise KernelWikiError(
                    "contract-module-invalid",
                    f"committed export {module_name}.{attribute_name} is no longer callable",
                )
            return target(*args, **kwargs)

    invoke.__name__ = attribute_name
    invoke.__qualname__ = f"{module_name}.{attribute_name}"
    invoke.__module__ = f"kernelwiki_bridge_{module_name}"
    return invoke


def load_loop_module(name: str) -> ModuleType:
    if not isinstance(name, str) or name not in ALLOWED_MODULES:
        raise KernelWikiError("contract-module-denied", f"module {name!r} is not allowlisted")
    authority = _resolve_committed_authority()
    with _load_authority_modules(authority, (name,)) as modules:
        source = modules[name]
        proxy = _SnapshotModuleProxy(f"kernelwiki_bridge_{name}")
        ModuleType.__setattr__(proxy, "__file__", f"git:{authority.identity.skill_tree_sha}:scripts/{name}.py")
        ModuleType.__setattr__(proxy, "_loop_contract_identity", authority.identity)
        exported: list[str] = []
        for attribute_name, value in sorted(vars(source).items()):
            if attribute_name.startswith("_") or isinstance(value, ModuleType):
                continue
            if callable(value):
                exported_value = _snapshot_callable(name, attribute_name, authority)
            else:
                exported_value = _freeze_exported_value(value)
            ModuleType.__setattr__(proxy, attribute_name, exported_value)
            exported.append(attribute_name)
        ModuleType.__setattr__(proxy, "__all__", tuple(exported))
        ModuleType.__setattr__(proxy, "_snapshot_proxy_sealed", True)
        return proxy


def parse_loop_contract_identity(value: object) -> LoopContractIdentity:
    if not isinstance(value, dict) or set(value) != {
        "repository_commit",
        "skill_tree_sha",
        "validator_sha256",
        "schema_sha256",
    }:
        raise KernelWikiError("contract-identity-invalid", "loop_contract_identity has invalid fields")
    commit = value["repository_commit"]
    tree = value["skill_tree_sha"]
    validators = value["validator_sha256"]
    schemas = value["schema_sha256"]
    if not isinstance(commit, str) or _GIT_SHA_RE.fullmatch(commit) is None:
        raise KernelWikiError("contract-identity-invalid", "repository_commit must be a Git commit")
    if not isinstance(tree, str) or _TREE_SHA_RE.fullmatch(tree) is None:
        raise KernelWikiError("contract-identity-invalid", "skill_tree_sha must be a Git tree")
    if not isinstance(validators, dict) or set(validators) != ALLOWED_MODULES:
        raise KernelWikiError("contract-identity-invalid", "validator_sha256 must pin every allowlisted validator")
    if not isinstance(schemas, dict) or set(schemas) != set(CONSUMED_SCHEMA_FILES):
        raise KernelWikiError("contract-identity-invalid", "schema_sha256 does not match the current contract")
    for mapping, label in ((validators, "validator_sha256"), (schemas, "schema_sha256")):
        for key, digest in mapping.items():
            if not isinstance(key, str) or not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
                raise KernelWikiError("contract-identity-invalid", f"{label} contains an invalid digest")
    return LoopContractIdentity(
        repository_commit=commit,
        skill_tree_sha=tree,
        validator_sha256=dict(sorted(validators.items())),
        schema_sha256=dict(sorted(schemas.items())),
    )

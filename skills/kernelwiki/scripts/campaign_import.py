from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any

from campaign_contract_bridge import (
    LoopContractIdentity,
    compute_loop_contract_identity,
    validator_modules,
)
from kernelwiki_common import (
    KernelWikiError,
    load_yaml_document,
    parse_markdown,
    require_within,
    sha256_bytes,
    validate_root_relative_posix_path,
)
from lift_schema import validate_lift_document


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "data" / "schemas.yaml"


@dataclass(frozen=True)
class BundleArtifact:
    name: str
    path: Path
    sha256: str
    required: bool


@dataclass(frozen=True)
class TerminalBundle:
    schema_version: int
    proposal_id: str
    repository_root: Path
    project_root: Path
    contract_version: int
    loop_contract_identity: LoopContractIdentity
    round_id: str
    terminal_commit: str
    terminal_result: str
    measurement_exclusive: bool
    artifacts: Mapping[str, BundleArtifact]
    canonical_candidate_ref: str | None
    canonical_report_ref: str | None


@dataclass(frozen=True)
class TerminalStateEvidence:
    workflow_status: str
    phase: str
    last_completed_round: str
    last_result: str
    measurement_exclusive: bool
    last_accepted_candidate: str | None
    last_accepted_report: str | None


@dataclass(frozen=True)
class ValidatedCampaign:
    bundle: TerminalBundle
    loop_contract_identity: LoopContractIdentity
    normalized_profile: Mapping[str, Any]
    normalized_claim: Mapping[str, Any]
    normalized_sketch: Mapping[str, Any]
    normalized_decision: Mapping[str, Any]
    normalized_binding: Mapping[str, Any]
    fact_pack: Mapping[str, Any]
    normalized_verdict: Mapping[str, Any]
    terminal_state: TerminalStateEvidence
    artifact_hashes: Mapping[str, str]
    missing_evidence: tuple[str, ...]


def _fail(code: str, message: str, path: Path | None = None) -> None:
    raise KernelWikiError(code, message, path)


def _text(value: Any, label: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        _fail("bundle-invalid", f"{label} must be nonempty trimmed text", path)
    return value


def _relative_path(value: Any, label: str, path: Path) -> str:
    try:
        return validate_root_relative_posix_path(value)
    except ValueError as error:
        _fail("bundle-path-invalid", f"{label}: {error}", path)


def _resolve_within(root: Path, relative: str, label: str, path: Path) -> Path:
    candidate = root / relative
    try:
        return require_within(root, candidate)
    except KernelWikiError as error:
        _fail("bundle-path-escape", f"{label} escapes repository root", path)


def _canonical_ref(
    value: Any,
    label: str,
    project_root: Path,
    expected: BundleArtifact,
    manifest_path: Path,
) -> str | None:
    if value is None:
        return None
    relative = _relative_path(value, label, manifest_path)
    try:
        resolved = require_within(project_root, project_root / relative)
    except KernelWikiError:
        _fail("canonical-pointer-invalid", f"{label} escapes project root", manifest_path)
    if resolved != expected.path:
        _fail("canonical-pointer-invalid", f"{label} does not identify {expected.name}", manifest_path)
    return relative


def load_terminal_bundle(path: Path) -> TerminalBundle:
    manifest_path = Path(path)
    document = validate_lift_document(
        "terminal_bundle",
        load_yaml_document(manifest_path),
        SCHEMA_PATH,
        path=manifest_path,
    )

    repository_value = _text(document["repository_root"], "repository_root", manifest_path)
    repository_root = Path(repository_value)
    if not repository_root.is_absolute():
        _fail("bundle-path-invalid", "repository_root must be absolute", manifest_path)
    repository_root = repository_root.resolve()

    project_relative = _relative_path(document["project_root"], "project_root", manifest_path)
    project_root = _resolve_within(repository_root, project_relative, "project_root", manifest_path)

    raw_identity = document["loop_contract_identity"]
    identity = LoopContractIdentity(
        repository_commit=raw_identity["repository_commit"],
        skill_tree_sha=raw_identity["skill_tree_sha"],
        validator_sha256=dict(raw_identity["validator_sha256"]),
        schema_sha256=dict(raw_identity["schema_sha256"]),
    )

    artifacts: dict[str, BundleArtifact] = {}
    for name, raw in document["artifacts"].items():
        relative = _relative_path(raw["path"], f"artifacts.{name}.path", manifest_path)
        artifacts[name] = BundleArtifact(
            name=name,
            path=_resolve_within(repository_root, relative, f"artifacts.{name}.path", manifest_path),
            sha256=raw["sha256"],
            required=raw["required"],
        )

    candidate_ref = _canonical_ref(
        document["canonical_candidate_ref"],
        "canonical_candidate_ref",
        project_root,
        artifacts["candidate"],
        manifest_path,
    )
    report_ref = _canonical_ref(
        document["canonical_report_ref"],
        "canonical_report_ref",
        project_root,
        artifacts["report"],
        manifest_path,
    )

    return TerminalBundle(
        schema_version=document["schema_version"],
        proposal_id=_text(document["proposal_id"], "proposal_id", manifest_path),
        repository_root=repository_root,
        project_root=project_root,
        contract_version=document["contract_version"],
        loop_contract_identity=identity,
        round_id=_text(document["round_id"], "round_id", manifest_path),
        terminal_commit=document["terminal_commit"],
        terminal_result=_text(document["terminal_result"], "terminal_result", manifest_path),
        measurement_exclusive=document["measurement_exclusive"],
        artifacts=artifacts,
        canonical_candidate_ref=candidate_ref,
        canonical_report_ref=report_ref,
    )


def _git_bytes(root: Path, *args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", "replace").strip()
        _fail("git-command-failed", message or "git command failed", root)
    return completed.stdout


def _artifact_git_path(bundle: TerminalBundle, artifact: BundleArtifact) -> str:
    try:
        return artifact.path.relative_to(bundle.repository_root).as_posix()
    except ValueError:
        _fail("bundle-path-escape", f"artifact {artifact.name} escapes repository root", artifact.path)


def load_committed_artifact(bundle: TerminalBundle, name: str) -> bytes:
    artifact = bundle.artifacts.get(name)
    if artifact is None:
        _fail("artifact-unknown", f"unknown bundle artifact: {name}", bundle.repository_root)
    git_path = _artifact_git_path(bundle, artifact)
    try:
        return _git_bytes(bundle.repository_root, "show", f"{bundle.terminal_commit}:{git_path}")
    except KernelWikiError as error:
        if error.code == "git-command-failed":
            _fail("artifact-absent", f"artifact {name} is absent from terminal commit", artifact.path)
        raise


def validate_git_identity(bundle: TerminalBundle) -> tuple[str, ...]:
    try:
        _git_bytes(bundle.repository_root, "cat-file", "-e", f"{bundle.terminal_commit}^{{commit}}")
    except KernelWikiError as error:
        if error.code == "git-command-failed":
            _fail("terminal-commit-absent", "terminal_commit does not exist", bundle.repository_root)
        raise

    diagnostics: list[str] = []
    for name in sorted(bundle.artifacts):
        artifact = bundle.artifacts[name]
        committed = load_committed_artifact(bundle, name)
        if sha256_bytes(committed) != artifact.sha256:
            _fail("artifact-hash-mismatch", f"committed artifact hash differs for {name}", artifact.path)
        try:
            current = artifact.path.read_bytes()
        except OSError:
            diagnostics.append(f"worktree-diverged:{name}")
        else:
            if sha256_bytes(current) != artifact.sha256:
                diagnostics.append(f"worktree-diverged:{name}")
    return tuple(diagnostics)


def coder_result_required(contract_version: int, terminal_result: str, route: str) -> bool:
    if contract_version != 3:
        _fail("contract-unsupported", f"unsupported campaign contract version: {contract_version}")
    if route == "proceed" and terminal_result in {"accepted", "no-improvement", "screened-out"}:
        return True
    if route == "abort" or terminal_result == "environment-blocked":
        return False
    _fail("contract-unsupported", f"unsupported Coder-result matrix: route={route}, terminal={terminal_result}")


def _same_contract_identity(left: LoopContractIdentity, right: LoopContractIdentity) -> bool:
    return (
        left.repository_commit == right.repository_commit
        and left.skill_tree_sha == right.skill_tree_sha
        and dict(left.validator_sha256) == dict(right.validator_sha256)
        and dict(left.schema_sha256) == dict(right.schema_sha256)
    )


def _write_committed(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def _materialize_committed_snapshot(bundle: TerminalBundle, destination: Path) -> Path:
    for artifact in bundle.artifacts.values():
        relative = _artifact_git_path(bundle, artifact)
        _write_committed(destination / relative, load_committed_artifact(bundle, artifact.name))

    profile = bundle.artifacts["implementation_profile"]
    profile_prefix = Path(_artifact_git_path(bundle, profile)).parent.as_posix()
    listing = _git_bytes(
        bundle.repository_root,
        "ls-tree",
        "-r",
        "--name-only",
        bundle.terminal_commit,
        "--",
        profile_prefix,
    ).decode("utf-8")
    for relative in (line for line in listing.splitlines() if line):
        _write_committed(
            destination / relative,
            _git_bytes(bundle.repository_root, "show", f"{bundle.terminal_commit}:{relative}"),
        )

    project_relative = bundle.project_root.relative_to(bundle.repository_root)
    project_root = destination / project_relative
    # Current Decision fixtures reference these project-local files. Their bytes
    # are duplicated from selected committed bundle artifacts, never the worktree.
    _write_committed(project_root / "baseline_adapter.py", load_committed_artifact(bundle, "base"))
    _write_committed(project_root / "rounds" / "report_000.md", load_committed_artifact(bundle, "report"))
    return project_root


def _validator_call(stage: str, function: Any, *args: Any, **kwargs: Any) -> Any:
    try:
        return function(*args, **kwargs)
    except KernelWikiError:
        raise
    except Exception as error:
        detail = getattr(error, "code", type(error).__name__)
        _fail(f"campaign-{stage}-invalid", f"{detail}: {error}")


def _load_json(path: Path, stage: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        _fail(f"campaign-{stage}-invalid", str(error), path)
    if not isinstance(value, dict):
        _fail(f"campaign-{stage}-invalid", f"{stage} must be a JSON object", path)
    return value


def _report_round(report_path: Path) -> str | None:
    text = report_path.read_text(encoding="utf-8")
    for block in re.findall(r"```json\s*\n(.*?)\n```", text, flags=re.DOTALL):
        try:
            value = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("round"), str):
            return value["round"]
    return None


def parse_terminal_state(path: Path) -> TerminalStateEvidence:
    try:
        metadata, _body = parse_markdown(path)
    except (KernelWikiError, OSError) as error:
        _fail("terminal-state-invalid", str(error), path)
    workflow_status = metadata.get("workflow_status")
    phase = metadata.get("phase")
    if (workflow_status, phase) not in {("running", "ready"), ("stopped", "stopped")}:
        _fail("terminal-state-invalid", "terminal state must be running/ready or stopped/stopped", path)
    if metadata.get("measurement_exclusive") is not False:
        _fail("measurement-exclusive", "terminal state must not be measurement-exclusive", path)
    for field in ("last_completed_round", "last_result"):
        if not isinstance(metadata.get(field), str) or not metadata[field]:
            _fail("terminal-state-invalid", f"terminal state requires {field}", path)
    for field in ("last_accepted_kernel", "last_accepted_report"):
        value = metadata.get(field)
        if value is not None:
            try:
                validate_root_relative_posix_path(value)
            except ValueError as error:
                _fail("terminal-state-invalid", f"{field}: {error}", path)
    return TerminalStateEvidence(
        workflow_status=workflow_status,
        phase=phase,
        last_completed_round=metadata["last_completed_round"],
        last_result=metadata["last_result"],
        measurement_exclusive=False,
        last_accepted_candidate=metadata.get("last_accepted_kernel"),
        last_accepted_report=metadata.get("last_accepted_report"),
    )


def _validate_coder_result(
    path: Path,
    *,
    round_id: str,
    candidate_sha256: str,
    profile_id: str,
) -> None:
    try:
        metadata, _body = parse_markdown(path)
    except (KernelWikiError, OSError) as error:
        _fail("campaign-coder-result-invalid", str(error), path)
    expected = {
        "schema_version": 1,
        "round": round_id,
        "candidate_sha256": candidate_sha256,
        "implementation_profile_id": profile_id,
        "status": "complete",
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        _fail("campaign-coder-result-invalid", "Coder result identity does not match the campaign", path)


def _check_campaign_identities(
    bundle: TerminalBundle,
    *,
    profile: Mapping[str, Any],
    claim: Mapping[str, Any],
    sketch: Mapping[str, Any],
    decision: Mapping[str, Any],
    binding_path: Path,
    fact_pack: Mapping[str, Any],
    verdict: Mapping[str, Any],
    terminal_state: TerminalStateEvidence,
    report_path: Path,
) -> tuple[str, ...]:
    profile_id = profile["implementation_profile_id"]
    target_id = claim["target_id"]
    candidate_sha = bundle.artifacts["candidate"].sha256
    raw_binding = _load_json(binding_path, "binding")

    if claim.get("implementation_profile_id") != profile_id:
        _fail("campaign-identity-mismatch", "claim profile does not match implementation profile")
    if target_id not in profile.get("identity_match", {}).get("permitted_target_ids", []):
        _fail("campaign-identity-mismatch", "claim target is outside implementation profile")
    if sketch.get("sketch", {}).get("round") != bundle.round_id:
        _fail("terminal-round-mismatch", "Sketch round does not match bundle")
    if decision.get("metadata", {}).get("round") != bundle.round_id:
        _fail("terminal-round-mismatch", "Decision round does not match bundle")
    if raw_binding.get("round") != bundle.round_id or _report_round(report_path) != bundle.round_id:
        _fail("terminal-round-mismatch", "binding/report round does not match bundle")
    if raw_binding.get("candidate_sha256") != candidate_sha or fact_pack.get("candidate_sha256") != candidate_sha:
        _fail("candidate-hash-mismatch", "candidate hash differs across manifest, binding, or report")
    if terminal_state.last_completed_round != bundle.round_id:
        _fail("terminal-round-mismatch", "terminal state round does not match bundle")
    if verdict.get("terminal_result") != bundle.terminal_result or terminal_state.last_result != bundle.terminal_result:
        _fail("terminal-result-mismatch", "verdict or terminal state result does not match bundle")
    if terminal_state.last_accepted_candidate != bundle.canonical_candidate_ref:
        _fail("canonical-pointer-mismatch", "terminal candidate pointer does not match bundle")
    if terminal_state.last_accepted_report != bundle.canonical_report_ref:
        _fail("canonical-pointer-mismatch", "terminal report pointer does not match bundle")

    missing: list[str] = []
    if not fact_pack.get("observables"):
        missing.append("report-observables-not-recorded")
    # The current verdict schema has no round/profile/target fields; the hashes
    # validated by validate_verdict are the current authority for those links.
    missing.append("verdict-round-profile-target-not-modeled")
    return tuple(missing)


def validate_campaign(bundle: TerminalBundle) -> ValidatedCampaign:
    if bundle.contract_version != 3:
        _fail("contract-unsupported", f"unsupported campaign contract version: {bundle.contract_version}")
    current_identity = compute_loop_contract_identity()
    if not _same_contract_identity(bundle.loop_contract_identity, current_identity):
        _fail("contract-unsupported", "bundle loop contract identity does not match the checked-in authority")
    validate_git_identity(bundle)

    modules = validator_modules()
    with tempfile.TemporaryDirectory(prefix="kernelwiki-campaign-") as temporary:
        snapshot_root = Path(temporary)
        project_root = _materialize_committed_snapshot(bundle, snapshot_root)
        paths = {
            name: snapshot_root / artifact.path.relative_to(bundle.repository_root)
            for name, artifact in bundle.artifacts.items()
        }

        profile = _validator_call("profile", modules["validate_profile"].load_profile, paths["implementation_profile"])
        runtime_snapshot = _load_json(paths["runtime_snapshot"], "runtime-snapshot")
        claim_result = _validator_call(
            "claim",
            modules["validate_profile"].validate_project_claim,
            paths["project_claim"],
            profile=profile,
            snapshot=runtime_snapshot,
        )
        sketch = _validator_call(
            "sketch",
            modules["validate_sketch"].validate_sketch,
            paths["sketch"],
            expected_round=bundle.round_id,
        )
        decision = _validator_call(
            "decision",
            modules["validate_decision"].validate_decision,
            paths["decision"],
            project_root=project_root,
            expected_implementation_profile=profile["implementation_profile_id"],
        )
        binding = _validator_call(
            "binding",
            modules["validate_binding"].validate_binding,
            paths["binding"],
            project_root=project_root,
            sketch_result=sketch,
            profile=profile,
            candidate_path=paths["candidate"],
        )
        facts = _validator_call(
            "fact-pack",
            modules["validate_verdict"].extract_verifier_fact_pack,
            paths["report"],
        )
        verdict = _validator_call(
            "verdict",
            modules["validate_verdict"].validate_verdict,
            paths["verdict"],
            inputs={
                "decision": decision,
                "sketch": sketch,
                "claim": claim_result,
                "binding": binding,
                "profile": profile,
                "facts": facts,
            },
        )
        terminal_state = parse_terminal_state(paths["team_state"])
        missing = list(
            _check_campaign_identities(
                bundle,
                profile=profile,
                claim=claim_result["claim"],
                sketch=sketch,
                decision=decision,
                binding_path=paths["binding"],
                fact_pack=facts,
                verdict=verdict,
                terminal_state=terminal_state,
                report_path=paths["report"],
            )
        )
        route = decision.get("metadata", {}).get("decision")
        required = coder_result_required(bundle.contract_version, bundle.terminal_result, route)
        coder_path = paths.get("coder_result")
        if required:
            if coder_path is None:
                _fail("campaign-coder-result-required", "terminal campaign requires a Coder result")
            _validate_coder_result(
                coder_path,
                round_id=bundle.round_id,
                candidate_sha256=bundle.artifacts["candidate"].sha256,
                profile_id=profile["implementation_profile_id"],
            )
        elif coder_path is None:
            missing.append("coder-result-not-produced")

        return ValidatedCampaign(
            bundle=bundle,
            loop_contract_identity=current_identity,
            normalized_profile=profile,
            normalized_claim=claim_result,
            normalized_sketch=sketch,
            normalized_decision=decision,
            normalized_binding=binding,
            fact_pack=facts,
            normalized_verdict=verdict,
            terminal_state=terminal_state,
            artifact_hashes={name: artifact.sha256 for name, artifact in sorted(bundle.artifacts.items())},
            missing_evidence=tuple(sorted(set(missing))),
        )

# KernelWiki Offline Knowledge Lift Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline, proposal-only knowledge-lift pipeline that validates an explicitly selected terminal campaign bundle and produces a reviewable, exactly scoped KernelWiki experience candidate.

**Architecture:** Keep two ingestion lanes strictly separate. The strict current-vNext lane validates the full available artifact chain through the latest checked-in `kernel-opt-loop` validators, verifies terminal Git identity, and writes only an experience proposal. The historical manual lane captures old noncanonical evidence with explicit missing fields and Designer-only scope; it never pretends historical campaigns passed vNext validation. Curator-reviewed Source/Card publication remains a separate Git change using the standalone-core validators and generators.

**Tech Stack:** Python 3 standard library, pinned PyYAML from the standalone-core plan, current checked-in `kernel-opt-loop` validators as read-only imports, Markdown/YAML/JSON, temporary Git repositories in `unittest`, Git.

**Spec:** `docs/superpowers/specs/2026-08-17-kernelwiki-v1-design.md`

**Depends on:** Completed and green `docs/superpowers/plans/2026-08-21-kernelwiki-standalone-core.md`. This plan does not depend on the role-aware query plan.

## Execution Granularity

Each named test method below is one red/green micro-step: add only that test, run its exact test module, implement the smallest behavior that passes it, and rerun before adding the next method. Checkbox steps are review gates that group these 2–5 minute micro-steps; do not batch a prose list into one implementation jump.

## Global Constraints

- Modify only `skills/kernelwiki/` and KernelWiki documentation. Do not modify `skills/kernel-opt-loop/`, active campaigns, project `state/`, profiles, base files, harnesses, or role artifacts.
- Input is always an explicit caller-selected terminal bundle. No daemon, hook, Orchestrator invocation, final-stop callback, or live-loop write is permitted.
- Strict vNext extraction consumes the latest checked-in control-plane contract at implementation time, pins exact artifact/contract hashes per import, and fails closed on unsupported versions or validator API changes.
- Historical manual capture is not vNext extraction. It records the historical contract version, noncanonical profile authority, exact available hashes, and missing evidence; it never migrates or upgrades old artifacts.
- The extractor writes only a deterministic JSON filename under `skills/kernelwiki/candidates/experience/` or one explicit caller-provided output path. It never writes Source records, Wiki Cards, query views, catalog files, active campaign files, or project state.
- Extraction never equals publication. Publication requires a separate Curator `include|defer|exclude` review, manual Source/Card change, generated-view regeneration, Git diff review, and commit.
- Included local evidence defaults to Designer-only. Coder eligibility never auto-promotes and remains a separate exact-profile review.
- A local proposal defaults to a scoped example on an existing general Card. A new Card requires a previously uncovered reusable mechanism or independent teaching value and cannot be named after a competition operator merely because a campaign exists.
- Preserve exact target/profile/runtime/shape/dtype/measurement scope, missing evidence, terminal classification, comparability, artifact hashes, transfer boundaries, and reconsideration conditions.
- Preserve validated control-plane vocabulary. Unknown remains Unknown; a slow result does not gain an invented causal explanation; a single code failure or environment block defaults to `defer`.
- Contradictory examples remain visible. Never rewrite an immutable historical Source to reconcile later evidence.
- A proposal contains no instruction for the next optimization candidate.
- Seal the local-campaign holdout before selecting development examples or writing mapping logic.

## Planned File Map

```text
skills/kernelwiki/
  references/
    knowledge-lift-contract.md       # Strict lane, historical lane, review boundary
    evaluation-protocol.md           # Local-campaign holdout procedure

  data/
    schemas.yaml                     # Add terminal bundle/proposal/review versions
    local-campaign-holdout.yaml      # Sealed before mapping implementation

  candidates/experience/
    .gitkeep
    reviews/                         # Curator decision files; no automatic publication

  scripts/
    campaign_contract_bridge.py      # Read-only current-vNext validator adapter
    campaign_import.py               # Terminal bundle/hash/Git/authority validation
    experience.py                    # Proposal normalization and outcome mapping
    propose_from_campaign.py         # Strict proposal-only CLI
    propose_historical_campaign.py   # Historical proposal-only CLI
    validate_lift.py                 # Proposal/review validator CLI
    historical_capture.py            # Explicit legacy/noncanonical candidate/capture library
    capture_source.py                # Add reviewed-historical maintenance subcommand
    validate.py                      # Validate experience proposals/reviews/local Sources

  tests/
    campaign_fixture_factory.py
    test_campaign_import.py
    test_experience.py
    test_historical_capture.py
    test_lift_contracts.py
    fixtures/lift/                    # Static source fragments only; manifests are generated in temp dirs
```

### Stable Lift Interfaces

```python
@dataclass(frozen=True)
class LoopContractIdentity:
    repository_commit: str
    skill_tree_sha: str
    validator_sha256: Mapping[str, str]
    schema_sha256: Mapping[str, str]

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

@dataclass(frozen=True)
class ExperienceProposal:
    schema_version: int
    proposal_id: str
    source_lane: str
    contract_version: int
    loop_contract_identity: LoopContractIdentity | None
    artifact_hashes: Mapping[str, str]
    terminal: Mapping[str, Any]
    scope: Mapping[str, Any]
    expected: Mapping[str, Any]
    observed: tuple[Mapping[str, Any], ...]
    suggested_publication: Mapping[str, Any]
    transfer_boundaries: tuple[str, ...]
    reconsider_when: tuple[str, ...]
    missing_evidence: tuple[str, ...]
```

Strict bundle artifact names:

```text
implementation_profile
runtime_snapshot
project_claim
sketch
decision
binding
candidate
coder_result
report
verdict
team_state
project
base
harness
```

`coder_result` may be declared optional only when the pinned current contract marks it unavailable for the selected terminal result. Every other missing required artifact fails closed.

---

### Task 1: Seal Local Holdouts and Define Bundle/Proposal/Review Schemas

**Files:**
- Create: `skills/kernelwiki/data/local-campaign-holdout.yaml`
- Create: `skills/kernelwiki/candidates/experience/.gitkeep`
- Create: `skills/kernelwiki/candidates/experience/reviews/.gitkeep`
- Create: `skills/kernelwiki/references/knowledge-lift-contract.md`
- Modify: `skills/kernelwiki/data/schemas.yaml`
- Modify: `skills/kernelwiki/references/evaluation-protocol.md`
- Create: `skills/kernelwiki/tests/test_lift_contracts.py`

**Interfaces:**
- Establishes schema versions `terminal_bundle: 1`, `experience_proposal: 1`, `experience_review: 1`, and `historical_capture: 1`.

- [ ] **Step 1: Write the sealed local-campaign boundary before mapping code**

Create exactly:

```yaml
schema_version: 1
sealed_at: 2026-08-21T00:00:00Z
development_campaigns:
  - kernels/track1-triton/groupedtopk/ascend
  - kernels/track1-triton/flexattention/ascend
  - kernels/track1-triton/mhc_post_layer_mix/ascend
holdout_campaigns:
  - kernels/track1-triton/mm_encoder_attention/ascend
  - kernels/track1-triton/sparse_pooler/ascend
rules:
  - holdout campaigns do not influence outcome mapping or publication defaults
  - holdout campaigns are evaluated only after strict and historical lanes pass tests
  - all historical campaigns remain noncanonical and Designer-only
```

Commit this file before implementing Tasks 2–5. Later changes to the holdout require a separate design review, not a metric-tuning commit.

- [ ] **Step 2: Write failing schema/contract tests**

Test that schemas require the exact stable fields and reject:

```text
unsupported contract version
non-40-hex terminal commit
absolute or escaping artifact path
non-64-hex artifact hash
measurement_exclusive=true
missing required artifact
proposal containing next_candidate, recommended_next_change, or implementation_instruction
review without reviewer/rationale/proposal hash
```

- [ ] **Step 3: Run tests and verify failure**

```bash
python3 -m unittest skills/kernelwiki/tests/test_lift_contracts.py -v
```

Expected: lift schema/validator code is missing.

- [ ] **Step 4: Document the two lanes and immutable boundary**

`knowledge-lift-contract.md` must state:

```text
strict current-vNext bundle -> validated proposal only
historical manual manifest -> noncanonical local Source only after Curator review
proposal -> review decision -> manual Source/Card change -> validate/generate -> Git commit
```

It explicitly forbids strict-lane fallback to historical parsing and historical-lane claims of vNext validation.

- [ ] **Step 5: Commit the sealed boundary and contracts**

```bash
git add skills/kernelwiki/data/local-campaign-holdout.yaml skills/kernelwiki/data/schemas.yaml skills/kernelwiki/candidates/experience skills/kernelwiki/references/knowledge-lift-contract.md skills/kernelwiki/references/evaluation-protocol.md skills/kernelwiki/tests/test_lift_contracts.py
git commit -m "docs(kernelwiki): seal knowledge lift boundaries"
```

---

### Task 2: Strict Terminal Bundle, Git Identity, and Root-Confinement Validation

**Files:**
- Create: `skills/kernelwiki/scripts/campaign_import.py`
- Create: `skills/kernelwiki/tests/campaign_fixture_factory.py`
- Create: `skills/kernelwiki/tests/test_campaign_import.py`

**Interfaces:**
- Produces: `load_terminal_bundle(path) -> TerminalBundle`, `validate_git_identity(bundle)`, `load_committed_artifact(bundle, name) -> bytes`.
- Does not yet interpret control-plane artifacts; Task 3 adds validator integration.

- [ ] **Step 1: Write failing bundle/Git tests**

Create a temporary Git repository in `campaign_fixture_factory.py`, materialize a complete test campaign, commit it, and emit a manifest with relative paths and computed SHA-256 values.

```python
class CampaignImportTests(unittest.TestCase):
    def test_valid_bundle_pins_terminal_commit_and_committed_bytes(self):
        root, manifest = materialize_terminal_bundle()
        bundle = load_terminal_bundle(manifest)
        validate_git_identity(bundle)
        self.assertEqual(run_git(root, "rev-parse", "HEAD"), bundle.terminal_commit)
        self.assertEqual(bundle.artifacts["candidate"].sha256, sha256_bytes(load_committed_artifact(bundle, "candidate")))

    def test_post_commit_worktree_change_is_diagnostic_only(self):
        root, manifest = materialize_terminal_bundle()
        bundle = load_terminal_bundle(manifest)
        candidate = bundle.project_root / "candidate.py"
        candidate.write_text("changed after terminal commit\n", encoding="utf-8")
        diagnostics = validate_git_identity(bundle)
        self.assertIn("worktree-diverged:candidate", diagnostics)
        self.assertEqual(bundle.artifacts["candidate"].sha256, sha256_bytes(load_committed_artifact(bundle, "candidate")))
```

Add methods `test_terminal_commit_absent_fails`, `test_required_artifact_absent_from_commit_fails`, `test_committed_hash_mismatch_fails`, `test_bundle_path_escape_fails`, `test_bundle_symlink_escape_fails`, `test_malformed_canonical_pointer_fails`, and `test_measurement_exclusive_must_be_boolean`. Task 3 cross-checks their meaning against validated control-plane artifacts.

- [ ] **Step 2: Run tests and verify failure**

```bash
python3 -m unittest skills/kernelwiki/tests/test_campaign_import.py -v
```

Expected: missing `campaign_import` module.

- [ ] **Step 3: Implement terminal-bundle parsing**

The checked-in tests generate manifests from a temporary repository instead of storing fake hashes. Implement this factory shape:

```python
def build_bundle_manifest(repository_root: Path) -> dict[str, Any]:
    project = repository_root / "project"
    artifact_paths = {
        "implementation_profile": project / "state" / "implementation_profile_snapshot" / "profile.yaml",
        "runtime_snapshot": project / "state" / "runtime-snapshot.json",
        "project_claim": project / "state" / "project_capability_claim.json",
        "sketch": project / "rounds" / "sketch_001.json",
        "decision": project / "rounds" / "decision_001.md",
        "binding": project / "rounds" / "binding_001.json",
        "candidate": project / "candidate.py",
        "coder_result": project / "rounds" / "coder_result_001.md",
        "report": project / "rounds" / "report_001.md",
        "verdict": project / "rounds" / "verdict_001.json",
        "team_state": project / "team-state.md",
        "project": project / "project.md",
        "base": repository_root / "base.py",
        "harness": repository_root / "auto_bench.py",
    }
    return {
        "schema_version": 1,
        "proposal_id": "experience-test-round-001",
        "repository_root": str(repository_root),
        "project_path": "project",
        "contract_version": 3,
        "loop_contract_identity": build_expected_loop_contract_identity(),
        "round_id": "001",
        "terminal_commit": run_git(repository_root, "rev-parse", "HEAD"),
        "terminal_result": "accepted",
        "measurement_exclusive": False,
        "canonical_candidate_ref": "candidate.py",
        "canonical_report_ref": "rounds/report_001.md",
        "artifacts": {
            name: {
                "path": path.relative_to(repository_root).as_posix(),
                "sha256": sha256_file(path),
                "required": True,
            }
            for name, path in artifact_paths.items()
        },
    }
```

The test-only `build_expected_loop_contract_identity()` computes Git/tree and file hashes directly from the fixed sibling `skills/kernel-opt-loop` without importing it; Task 3 implements and compares the production identity. `load_terminal_bundle` requires `terminal_commit` to match `[0-9a-f]{40}` and every artifact hash to match `[0-9a-f]{64}`. All artifact paths are repository-relative and root-confined; `project_root` is resolved as `repository_root / project_path`.

- [ ] **Step 4: Validate committed identity without shell use**

Use argv-only Git subprocesses:

```python
def git_bytes(root: Path, *args: str) -> bytes:
    completed = subprocess.run(["git", "-C", str(root), *args], check=False, capture_output=True)
    if completed.returncode != 0:
        raise KernelWikiError("git-command-failed", completed.stderr.decode("utf-8", "replace").strip(), root)
    return completed.stdout
```

Verify `terminal_commit` exists, each required artifact exists in that commit, and committed bytes match manifest hashes. Parse and validate the committed bytes, not the current checkout. Current-worktree divergence is recorded as a nonblocking `worktree-diverged` diagnostic; the validator is read-only and never checks out or resets the project.

- [ ] **Step 5: Run tests and commit**

```bash
python3 -m unittest skills/kernelwiki/tests/test_campaign_import.py -v
git add skills/kernelwiki/scripts/campaign_import.py skills/kernelwiki/tests/campaign_fixture_factory.py skills/kernelwiki/tests/test_campaign_import.py
git commit -m "feat(kernelwiki): validate terminal campaign bundles"
```

---

### Task 3: Read-Only Current-vNext Artifact Chain Validation

**Files:**
- Create: `skills/kernelwiki/scripts/campaign_contract_bridge.py`
- Modify: `skills/kernelwiki/scripts/campaign_import.py`
- Modify: `skills/kernelwiki/tests/campaign_fixture_factory.py`
- Modify: `skills/kernelwiki/tests/test_campaign_import.py`

**Interfaces:**
- Produces: `validate_campaign(bundle) -> ValidatedCampaign`, `compute_loop_contract_identity()`, and `coder_result_required(contract_version, terminal_result, route)`.
- Uses current checked-in functions read-only: `load_profile`, `validate_project_claim`, `validate_sketch`, `validate_decision`, `validate_binding`, `extract_verifier_fact_pack`, and `validate_verdict`.

- [ ] **Step 1: Extend the fixture factory with a complete current vNext chain**

Copy the smallest current fixtures from `skills/kernel-opt-loop/tests/fixtures/vnext/` into the temporary project:

```text
profiles/valid-partial
claims/valid-claim.json
sketches/valid-kernel.json
decisions/valid-vnext.md
bindings/valid-many-to-many.json
candidates/valid_candidate.py
integration/campaign/report_001.md
integration/campaign/verdict_001.json
```

Use the same marker-replacement strategy as current loop tests to update profile, claim, Sketch, candidate, binding, fact-pack, Decision, and verdict hashes. Write `state/runtime-snapshot.json` as `{"target_id":"mlu590","implementation_profile_id":"triton_mlu","triton_version":"3.6.0","device_arch":"mlu-arch"}`. Add a minimal `coder_result_001.md`, `team-state.md`, `project.md`, immutable base, and harness. Commit the complete tree before generating the strict bundle manifest.

- [ ] **Step 2: Write failing full-chain tests**

Require successful normalization plus separate failures for invalid profile, claim, Sketch, Decision, binding, fact pack, verdict, invalid terminal-state matrix, canonical pointer mismatch, terminal round/result mismatch, and `measurement_exclusive: true`. Add exact tests for nonallowlisted validator denial, rejection of a caller-supplied loop root, validator/schema hash mismatch, and LoopContractIdentity round-trip into the proposal. Assert unsupported `contract_version: 4` fails with `contract-unsupported` rather than using v3 logic.

- [ ] **Step 3: Run tests and verify failure**

```bash
python3 -m unittest skills/kernelwiki/tests/test_campaign_import.py -v
```

Expected: bridge functions are missing.

- [ ] **Step 4: Implement the current-contract bridge**

Resolve the loop root only as `Path(__file__).resolve().parents[2] / "kernel-opt-loop"`; no CLI or manifest field may override it. Allowlist exactly `validate_profile`, `validate_sketch`, `validate_decision`, `validate_binding`, and `validate_verdict`. `compute_loop_contract_identity()` records `git log -1 --format=%H -- skills/kernel-opt-loop`, `git rev-parse HEAD:skills/kernel-opt-loop`, every allowlisted validator hash, and every consumed external schema hash; the current contract embeds schema checks in validators, so `schema_sha256` is `{}` until an external schema is actually consumed. The bundle pins that identity and validation fails `contract-unsupported` on mismatch. Keep this bridge self-contained so Phase D does not depend on Phase C.

Validation order is exact:

```text
profile -> project claim -> Sketch -> Decision -> binding -> report fact pack -> verdict -> cross-artifact identity checks
```

Materialize each selected file into a temporary validation tree with `git_bytes(bundle.repository_root, "show", f"{bundle.terminal_commit}:{artifact.path.as_posix()}")`; validators receive only this committed snapshot. Current checkout bytes are never authority.

Call signatures:

```python
profile = validate_profile.load_profile(profile_path)
runtime_snapshot = json.loads(runtime_snapshot_path.read_text(encoding="utf-8"))
claim_result = validate_profile.validate_project_claim(claim_path, profile=profile, snapshot=runtime_snapshot)
sketch_result = validate_sketch.validate_sketch(sketch_path, expected_round=bundle.round_id)
decision_result = validate_decision.validate_decision(
    decision_path,
    project_root=snapshot.project_root,
    expected_implementation_profile=profile["implementation_profile_id"],
)
binding_result = validate_binding.validate_binding(
    binding_path,
    project_root=snapshot.project_root,
    sketch_result=sketch_result,
    profile=profile,
    candidate_path=candidate_path,
)
facts = validate_verdict.extract_verifier_fact_pack(report_path)
verdict_result = validate_verdict.validate_verdict(
    verdict_path,
    inputs={"decision": decision_result, "sketch": sketch_result, "binding": binding_result, "profile": profile, "facts": facts},
)
```

Read `runtime_snapshot` from a manifest field or pinned project artifact required by the then-current claim validator. If the current API differs, fail `contract-unsupported` and update this plan through review; do not silently adapt fields.

- [ ] **Step 5: Enforce cross-artifact identities and missing-evidence reporting**

Implement `parse_terminal_state(team_state_bytes) -> TerminalStateEvidence` from committed Markdown YAML frontmatter. Accept only `(workflow_status=running, phase=ready)` for a committed terminal round or `(workflow_status=stopped, phase=stopped)` for final stop. In both cases, `last_completed_round` equals the bundle round, `last_result` equals the validated verdict terminal result, and `measurement_exclusive` is false. `last_accepted_kernel`/`last_accepted_report` are root-confined relative paths and must equal the bundle's canonical pointer duplicates. Stable errors are `terminal-state-invalid`, `terminal-round-mismatch`, `terminal-result-mismatch`, `measurement-exclusive`, and `canonical-pointer-mismatch`.

Require candidate hashes to agree across manifest, fact pack, binding, and committed bytes. Require round/profile/target identities to agree across bundle, profile, claim, Sketch, Decision, report, verdict, and terminal state. Record optional unavailable facts as `missing_evidence`; never synthesize them.

For contract v3, `coder_result_required` returns true when `route == "proceed"` and `terminal_result` is `accepted`, `no-improvement`, or `screened-out`; it returns false for `route == "abort"` or terminal `environment-blocked`. Any other combination is `contract-unsupported`. When required, validate the Coder result's round, candidate hash, profile, and status; when optional and absent, record `coder-result-not-produced` rather than inventing content. Add one test per matrix row.

- [ ] **Step 6: Run tests and commit**

```bash
python3 -m unittest skills/kernelwiki/tests/test_campaign_import.py -v
git add skills/kernelwiki/scripts/campaign_contract_bridge.py skills/kernelwiki/scripts/campaign_import.py skills/kernelwiki/tests/campaign_fixture_factory.py skills/kernelwiki/tests/test_campaign_import.py
git commit -m "feat(kernelwiki): validate vNext evidence chains"
```

---

### Task 4: Deterministic Proposal Mapping and Proposal-Only CLI

**Files:**
- Create: `skills/kernelwiki/scripts/experience.py`
- Create: `skills/kernelwiki/scripts/propose_from_campaign.py`
- Create: `skills/kernelwiki/tests/test_experience.py`

**Interfaces:**
- Produces: `build_experience_proposal(validated) -> ExperienceProposal`, `write_proposal(proposal, output_path)`.
- Writes no Source/Card/catalog/query files.

- [ ] **Step 1: Write failing outcome-mapping tests**

Use validated synthetic campaigns for:

```text
accepted improvement -> positive example proposal
no-improvement wall/device mismatch -> counterexample proposal
screened-out slower result -> slower-result proposal without causal invention
design/capability gap -> design-pitfall or capability-gap proposal only when verdict supports it
lowering/capability Unknown -> Unknown/probe proposal
stable exact-profile repeated code pitfall -> implementation-pitfall proposal
environment block or incomplete chain -> defer
```

Assert every proposal preserves exact scope/hashes and lacks forbidden next-candidate keys.

- [ ] **Step 2: Run tests and verify failure**

```bash
python3 -m unittest skills/kernelwiki/tests/test_experience.py -v
```

Expected: missing `experience` module.

- [ ] **Step 3: Implement proposal normalization**

`build_experience_proposal` derives a deterministic ID from contract version, terminal commit, round, and terminal result:

```python
proposal_id = "experience-" + sha256_bytes(
    canonical_json_bytes({
        "contract_version": validated.bundle.contract_version,
        "terminal_commit": validated.bundle.terminal_commit,
        "round_id": validated.bundle.round_id,
        "terminal_result": validated.bundle.terminal_result,
    })
)[:20]
```

The proposal includes:

```text
expected intervention and causal observables from Decision/Sketch
actual correctness, lowering, profiler, kernel-count, device-time, and wall facts as a sorted list of core `{metric,value,statistic,unit}` measurement records without reinterpretation
validated terminal classification and attribution
target/profile/runtime/shape/dtype/measurement/comparability scope
suggested existing general Card IDs or generic tags
suggested example role
transfer boundaries
reconsideration conditions
missing evidence
all artifact hashes, terminal commit, and pinned LoopContractIdentity
```

If no current-project measurement fingerprint is available, mark `measurement_fingerprint: null`, add `measurement-fingerprint-missing`, and default publication decision recommendation to `defer`.

- [ ] **Step 4: Implement version-specific mapping without redefining vocabulary**

Put mapping logic behind a contract adapter keyed by the validated exact contract version. It reads `verdict_result["terminal_result"]`, `classification`, route, fact-pack statuses, and evidence gaps. Unknown remains Unknown. No branch infers a cause from latency numbers alone.

Use this closed publication mapping:

```text
accepted + comparable improvement                       -> positive
accepted/no-improvement + slower or no improvement      -> counterexample:performance
screened-out                                             -> counterexample:screening
accepted + device win / wall loss                        -> counterexample:device-wall-mismatch
aborted after Designer semantic rejection                -> counterexample:design-pitfall
aborted after valid proceed / Coder failure              -> counterexample:implementation-pitfall
Unknown or unsupported required capability               -> capability-gap:profile
probe-only, environment-blocked, or incomplete evidence  -> defer (not publishable)
```

Add one cross-plan schema test per row. Every publishable proposal uses core role `positive|counterexample|capability-gap` plus the listed subtype; every other outcome carries suggested decision `defer`.

- [ ] **Step 5: Implement the proposal-only CLI**

Support:

```bash
python3 skills/kernelwiki/scripts/propose_from_campaign.py \
  --bundle /absolute/path/to/terminal-bundle.yaml \
  --output skills/kernelwiki/candidates/experience/experience-test-round-001.json
```

If `--output` is omitted, write under `candidates/experience/` using the deterministic proposal ID. Refuse output paths inside the selected campaign root or any path named `state`. Refuse overwrite with `proposal-exists`. Print canonical JSON containing output path and proposal SHA-256.

- [ ] **Step 6: Prove extraction never publishes**

Snapshot `sources/`, `wiki/`, `queries/`, and `compiled/` before CLI execution. After success and every failure case, assert those trees are byte-identical. Add an AST contract test that `propose_from_campaign.py` imports neither catalog generation nor Card-writing helpers.

- [ ] **Step 7: Run tests and commit**

```bash
python3 -m unittest skills/kernelwiki/tests/test_experience.py skills/kernelwiki/tests/test_campaign_import.py -v
git add skills/kernelwiki/scripts/experience.py skills/kernelwiki/scripts/propose_from_campaign.py skills/kernelwiki/tests/test_experience.py
git commit -m "feat(kernelwiki): propose scoped campaign experience"
```

---

### Task 5: Historical Manual Capture Lane

**Files:**
- Create: `skills/kernelwiki/scripts/historical_capture.py`
- Create: `skills/kernelwiki/scripts/propose_historical_campaign.py`
- Create: `skills/kernelwiki/tests/test_historical_capture.py`

**Interfaces:**
- Produces: `load_historical_manifest`, `build_historical_proposal`, and `write_historical_proposal`.
- Writes only deterministic `source_lane: historical-manual` candidates under `skills/kernelwiki/candidates/experience/`.
- Never calls strict vNext validators, never creates Source/Card files, and always records `strict_vnext_validated: false`.

- [ ] **Step 1: Write failing historical-lane tests**

Require a historical manifest to include:

```text
source_id
historical_contract_version
repository_commit
project_path
local_locator
captured_at
repository_id=local
languages
kernel_types
techniques
hardware_features
tags
license_state
asset_mode=metadata-only|selected-files
allowed_audiences=[designer]
target_id
implementation_profile_id
profile_authority=historical-noncanonical
terminal_result
artifact refs with file role and SHA-256
measurement fingerprint or explicit missing declaration
exact observations
transfer boundaries
missing evidence
audiences=[designer]
```

Reject `audiences: [coder]`, `profile_authority: canonical`, missing hashes, transfer claims beyond scope, or a claim that vNext validation passed.

- [ ] **Step 2: Run tests and verify failure**

```bash
python3 -m unittest skills/kernelwiki/tests/test_historical_capture.py -v
```

Expected: missing `historical_capture` module.

- [ ] **Step 3: Implement historical candidate construction**

`build_historical_proposal` verifies every explicit artifact hash, copies no files, and emits one canonical JSON candidate with the exact eventual Source/provenance metadata, scoped observations, transfer boundaries, typed missing evidence, and `strict_vnext_validated: false`. Its deterministic proposal ID is `f"experience-historical-{manifest.source_id}"`; an existing different candidate at that path fails `proposal-exists`.

- [ ] **Step 4: Add the proposal-only maintenance command**

Support:

```bash
python3 skills/kernelwiki/scripts/propose_historical_campaign.py \
  --manifest /tmp/kernelwiki-historical-groupedtopk-round-001.yaml \
  --output skills/kernelwiki/candidates/experience/experience-historical-source-local-ascend-groupedtopk-round-001.json
```

Create reviewed manifests/candidates for all three development campaigns before holdout work:

```text
kernels/track1-triton/groupedtopk/ascend round 001
kernels/track1-triton/flexattention/ascend round 003
kernels/track1-triton/mhc_post_layer_mix/ascend final stop round 003, canonical candidate_001.py/report_001.md, last completed report_002.md
```

The implementation computes repository commit and artifact SHA-256 values; no test fixture is copied into production unchanged. Holdout paths `mm_encoder_attention/ascend` and `sparse_pooler/ascend` remain untouched until final evaluation.

- [ ] **Step 5: Prove historical and strict lanes do not collapse**

Tests patch `validate_campaign` and assert historical proposal construction never calls it. Strict proposal tests reject `historical_contract_version` manifests. Historical candidates list missing typed Sketch/binding/verdict evidence when absent and cannot be published without a separate review file.

- [ ] **Step 6: Run tests and commit**

```bash
python3 -m unittest skills/kernelwiki/tests/test_historical_capture.py -v
git add skills/kernelwiki/scripts/historical_capture.py skills/kernelwiki/scripts/propose_historical_campaign.py skills/kernelwiki/tests/test_historical_capture.py skills/kernelwiki/candidates/experience
git commit -m "feat(kernelwiki): propose historical campaign evidence"
```

---

### Task 6: Curator Review Validation and Manual Publication Boundary

**Files:**
- Create: `skills/kernelwiki/scripts/validate_lift.py`
- Modify: `skills/kernelwiki/scripts/validate.py`
- Modify: `skills/kernelwiki/tests/test_lift_contracts.py`
- Modify: `skills/kernelwiki/references/knowledge-lift-contract.md`

**Interfaces:**
- Produces: `validate_proposal(path)`, `validate_review(path, proposal)`, `validate_experience_tree(root)`.
- Does not produce or edit Source/Card files.

- [ ] **Step 1: Write failing proposal/review tests**

Generate review documents in the temporary test directory from the actual proposal:

```python
def build_review(proposal_path: Path, decision: str) -> dict[str, Any]:
    publication_target = (
        {"mode": "existing-card-example", "card_id": "technique-kernel-fusion"}
        if decision == "include"
        else None
    )
    return {
        "schema_version": 1,
        "proposal_id": json.loads(proposal_path.read_text(encoding="utf-8"))["proposal_id"],
        "proposal_sha256": sha256_file(proposal_path),
        "decision": decision,
        "reviewed_by": "kernelwiki-curator",
        "reviewed_at": "2026-08-21T00:00:00Z",
        "rationale": "Terminal evidence is complete and the scoped example teaches a reusable mechanism.",
        "publication_target": publication_target,
    }
```

Create `include`, `defer`, and `exclude` cases from this helper. Reject unknown fields, missing review identity/rationale, stale proposal hash, operator-named new Card targets, automatic Coder visibility, or a proposal containing forbidden next-candidate instructions.

Reject unknown fields, missing review identity/rationale, stale proposal hash, operator-named new Card targets, automatic Coder visibility, or a proposal containing forbidden next-candidate instructions.

- [ ] **Step 2: Run tests and verify failure**

```bash
python3 -m unittest skills/kernelwiki/tests/test_lift_contracts.py -v
```

Expected: validation functions are missing.

- [ ] **Step 3: Implement proposal and review validation**

`validate_proposal` checks schema, required scope, artifact hashes, transfer boundaries, missing evidence, suggested publication mode, and forbidden instruction keys recursively. A strict current-vNext proposal requires a nonnull exact `LoopContractIdentity`; a historical-manual proposal requires `loop_contract_identity: null` and `strict_vnext_validated: false`. `validate_review` requires exact proposal ID/hash and `include|defer|exclude`. An `include` target is either `existing-card-example` or `new-general-card`; `new-kernel-case-card` additionally requires `independent_teaching_value: true` and a non-operator reusable title.

- [ ] **Step 4: Wire experience candidates into whole-skill validation**

`validate.py` validates every proposal and review but does not require every proposal to have a decision. It rejects multiple decisions for one proposal and a review whose proposal is missing. It never interprets `include` as permission to mutate the corpus.

- [ ] **Step 5: Document the manual publication checklist**

For an included proposal, the Curator performs a separate change:

```text
create immutable local Source with exact hashes
add one scoped example to the reviewed existing Card by default
keep target/profile/runtime/shape/dtype/measurement/transfer fields exact
leave contradictory examples visible
default audiences to designer
run validate.py
generate indices
review Source/Card/generated diffs
commit
```

No proposal extractor or review validator performs those edits. Task 7 adds only an explicit Curator-invoked immutable Source capture command for an already included historical proposal; Card and generated-output publication remains manual Git work.

- [ ] **Step 6: Run tests and commit**

```bash
python3 -m unittest skills/kernelwiki/tests/test_lift_contracts.py skills/kernelwiki/tests/test_experience.py -v
git add skills/kernelwiki/scripts/validate_lift.py skills/kernelwiki/scripts/validate.py skills/kernelwiki/tests/test_lift_contracts.py skills/kernelwiki/references/knowledge-lift-contract.md
git commit -m "feat(kernelwiki): validate experience reviews"
```

---

### Task 7: Reviewed Local Examples, Contradiction Visibility, and Final Acceptance

**Files:**
- Create after Curator review: `skills/kernelwiki/sources/local/ascend/source-local-ascend-groupedtopk-round-001.md`
- Create after Curator review: `skills/kernelwiki/sources/local/ascend/source-local-ascend-flexattention-round-003.md`
- Create after Curator review if included: `skills/kernelwiki/sources/local/ascend/source-local-ascend-mhc-post-layer-mix.md`
- Create after Curator review: corresponding `skills/kernelwiki/artifacts/local/` bundles when license/size policy permits
- Modify: `skills/kernelwiki/scripts/historical_capture.py`
- Modify: `skills/kernelwiki/scripts/capture_source.py`
- Modify: `skills/kernelwiki/tests/test_historical_capture.py`
- Modify after Curator review: `skills/kernelwiki/wiki/techniques/kernel-fusion.md`
- Modify after Curator review: `skills/kernelwiki/wiki/patterns/device-win-wall-loss.md`
- Modify after Curator review: one general output-reuse Card only if existing evidence supports a coherent page
- Generate: `skills/kernelwiki/queries/*.md`
- Generate: `skills/kernelwiki/compiled/catalog.jsonl`
- Modify: `skills/kernelwiki/SKILL.md`
- Modify: `skills/kernelwiki/README.md`
- Modify: `skills/kernelwiki/tests/test_contracts.py`

**Interfaces:**
- Produces `materialize_reviewed_historical_source(proposal_path, review_path, skill_root)` for explicit Curator-invoked immutable Source capture; Card edits remain manual Git work and all extractors/validators remain proposal-only.

- [ ] **Step 1: Review development historical captures**

For grouped top-k Round 001, flexattention Round 003, and the selected terminal MHC round, compare candidate observations with original Decision/report/candidate artifacts and choose `include|defer|exclude`. Commit one review file for all three development candidates before publishing any Source or opening holdouts:

```bash
git add skills/kernelwiki/candidates/experience/reviews
git commit -m "docs(kernelwiki): review development experience candidates"
```

- [ ] **Step 2: Publish included Sources and scoped examples manually**

For each `include` review, run:

```bash
python3 skills/kernelwiki/scripts/capture_source.py reviewed-historical \
  --proposal skills/kernelwiki/candidates/experience/experience-historical-source-local-ascend-groupedtopk-round-001.json \
  --review skills/kernelwiki/candidates/experience/reviews/experience-historical-source-local-ascend-groupedtopk-round-001.yaml
```

`materialize_reviewed_historical_source` verifies the proposal hash/review, then builds Source frontmatter with exactly `schema_version`, `id`, `source_kind: local-campaign`, `title`, `url` computed as `f"local://{repository_commit}/{project_path}"`, `repository_id: local`, `captured_at`, `target_disposition: exact`, `implementation_profile_ids`, `runtime_fingerprints`, `languages`, `kernel_types`, `techniques`, `hardware_features`, `tags`, `license_state`, `artifact_dir`, `audiences: [designer]`, `profile_authority: historical-noncanonical`, `strict_vnext_validated: false`, and `missing_evidence` copied and sorted from the reviewed proposal. If selected files are allowed, `PROVENANCE.yaml` maps exactly to core `ProvenanceBundle`: top-level `schema_version`, `origin_url`, `upstream_repo`, `upstream_sha`, `license_state`, `retrieved_at`, `asset_mode`, `allowed_audiences`, `coder_access`, `source_ids`, and `files`; each file entry has `local_path`, `upstream_path`, `heading_path`, `role`, `mode`, and `sha256`. Tests build expected Source/provenance dictionaries from actual temporary artifact hashes, compare serialized bytes, then assert the materialized tree passes both `load_corpus(skill_root)` and `validate_provenance(load_provenance(bundle_path), skill_root)`.

Grouped top-k fusion records exact target/profile/runtime/shape/dtype, measurement fingerprint, `kernel_count_per_call`, wall improvement, comparability `historical-local`, and transfer limits. Flexattention records the exact device-time/wall mismatch as a counterexample without generalizing it to every attention shape/runtime. MHC is published only if its separate review is `include`; otherwise its `defer|exclude` review remains visible.

If any candidate lacks a required identity, choose `defer` instead of inventing it.

- [ ] **Step 3: Preserve contradiction visibility**

Add tests proving positive and counterexample cases for the same technique remain present in Card metadata, catalog counts, `by-evidence-level.md`, and queries. A new result creates a new Source/example; it never edits historical measured values.

- [ ] **Step 4: Run local-campaign holdout evaluation**

First assert committed `include|defer|exclude` review files exist for grouped top-k, flexattention, and MHC and that mapping code/generator hashes match the pre-holdout seal. Only then inspect the sealed `mm_encoder_attention/ascend` and `sparse_pooler/ascend` campaigns. Evaluate whether the existing general Cards surface materialization/output-reuse counterexamples. Record evaluation results without changing holdout membership or historical Sources.

- [ ] **Step 5: Run complete validation and generation**

```bash
python3 -m unittest discover -s skills/kernelwiki/tests -p 'test_*.py' -v
python3 skills/kernelwiki/scripts/validate.py
python3 skills/kernelwiki/scripts/generate_indices.py
python3 skills/kernelwiki/scripts/generate_indices.py --check
```

Expected: all tests pass; no active campaign files or `kernel-opt-loop` files changed; generated output is current.

- [ ] **Step 6: Add final nonintegration contract tests**

Assert no Orchestrator hook, final-stop callback, consultation record, project-side proposal path, active campaign write, extractor/review-triggered publisher, automatic Card publisher, or Coder auto-promotion exists. Allow only the explicit `reviewed-historical` Source capture command after a valid include review. Assert `propose_from_campaign.py` output paths are KernelWiki candidates or explicit noncampaign caller paths only.

- [ ] **Step 7: Document maintenance workflow and commit**

Document:

```text
explicit terminal bundle -> strict validation -> proposal -> Curator review
historical manifest -> proposal candidate -> Curator review -> explicit noncanonical Source capture
reviewed corpus edit -> validate -> generate -> diff review -> Git commit
```

```bash
git add skills/kernelwiki/sources skills/kernelwiki/artifacts skills/kernelwiki/wiki skills/kernelwiki/queries skills/kernelwiki/compiled skills/kernelwiki/SKILL.md skills/kernelwiki/README.md skills/kernelwiki/scripts/historical_capture.py skills/kernelwiki/scripts/capture_source.py skills/kernelwiki/tests/test_historical_capture.py skills/kernelwiki/tests/test_contracts.py skills/kernelwiki/candidates/experience
git commit -m "docs(kernelwiki): publish reviewed local experience"
git status --short
```

Expected: clean worktree. Phase E remains unimplemented and requires a separate reviewed integration spec and plan.

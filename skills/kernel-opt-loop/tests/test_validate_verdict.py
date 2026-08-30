import contextlib
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

from validate_binding import validate_binding
from validate_profile import load_profile
from validate_sketch import validate_sketch
from validate_verdict import (
    VerdictValidationError,
    apply_submission_promotion,
    extract_verifier_fact_pack,
    resolve_finalization_slot,
    select_final_tuning_configuration,
    validate_causal_graph,
    validate_verdict,
)
from vnext_common import compute_submission_snapshot_id, sha256_canonical_json, sha256_file

FIXTURES = Path(__file__).parent / "fixtures" / "vnext"
REPORTS = FIXTURES / "reports"
VERDICTS = FIXTURES / "verdicts"
SKETCHES = FIXTURES / "sketches"
BINDINGS = FIXTURES / "bindings"
CANDIDATES = FIXTURES / "candidates"
PROFILES = FIXTURES / "profiles"

VALID_SKETCH = validate_sketch(SKETCHES / "valid-load-store-kernel.json")
VALID_PROFILE = load_profile(PROFILES / "valid-partial" / "profile.yaml")
VALID_BINDING = validate_binding(
    BINDINGS / "valid-many-to-many.json",
    project_root=FIXTURES,
    sketch_result=VALID_SKETCH,
    profile=VALID_PROFILE,
    candidate_path=CANDIDATES / "valid_candidate.py",
)
VALID_DECISION = {
    "valid": True,
    "decision_kind": "optimization",
    "intervention": "fuse routing reduction into the target kernel",
    "optimization_intent": {"intervention": "fuse routing reduction into the target kernel"},
    "causal_graph": {
        "nodes": ["m.reduce-fusion", "o.external-kernel-count", "p.wall-time"],
        "edges": [
            ["m.reduce-fusion", "o.external-kernel-count"],
            ["o.external-kernel-count", "p.wall-time"],
        ],
    },
}


def input_hash(value) -> str:
    if isinstance(value, dict):
        value = {key: item for key, item in value.items() if not key.startswith("_")}
    return sha256_canonical_json(value)


def materialize_verdict(name: str, **replacements) -> dict:
    text = (VERDICTS / name).read_text(encoding="utf-8")
    for marker, digest in replacements.items():
        text = text.replace(marker, digest)
    return json.loads(text)


def campaign_inputs(*, facts=None, decision=None, sketch=None, binding=None) -> dict:
    return {
        "decision": decision if decision is not None else VALID_DECISION,
        "sketch": sketch if sketch is not None else VALID_SKETCH,
        "binding": binding if binding is not None else VALID_BINDING,
        "profile": VALID_PROFILE,
        "facts": facts if facts is not None else VALID_FACTS,
    }


VALID_FACTS = extract_verifier_fact_pack(REPORTS / "valid-report.md")
MISSING_OBSERVABLE_FACTS = extract_verifier_fact_pack(REPORTS / "missing-observable.md")


def campaign_hashes(inputs: dict) -> dict:
    return {
        "__DECISION_HASH__": input_hash(inputs["decision"]),
        "__SKETCH_HASH__": input_hash(inputs["sketch"]),
        "__BINDING_HASH__": input_hash(inputs["binding"]),
        "__PROFILE_HASH__": input_hash(inputs["profile"]),
        "__FACTS_HASH__": input_hash(inputs["facts"]),
    }


def write_verdict_to_temp(verdict: dict) -> Path:
    directory = tempfile.mkdtemp(prefix="verdict-")
    path = Path(directory) / "verdict.json"
    path.write_text(json.dumps(verdict), encoding="utf-8")
    return path


class CausalGraphTests(unittest.TestCase):
    def test_connected_graph_validates(self):
        validate_causal_graph(
            VALID_DECISION["causal_graph"],
            intervention=VALID_DECISION["intervention"],
            observable_names=["external-kernel-count"],
        )

    def test_disconnected_graph_is_rejected(self):
        graph = {"nodes": ["a", "b"], "edges": []}
        with self.assertRaisesRegex(VerdictValidationError, "disconnected"):
            validate_causal_graph(graph, intervention="x", observable_names=["a"])

    def test_observable_outside_graph_is_rejected(self):
        with self.assertRaisesRegex(VerdictValidationError, "observable"):
            validate_causal_graph(
                VALID_DECISION["causal_graph"],
                intervention=VALID_DECISION["intervention"],
                observable_names=["unrelated-observable"],
            )


class VerdictRuleTests(unittest.TestCase):
    def test_lowering_unknown_requires_all_static_gates_and_absent_observed_mechanism(self):
        report = extract_verifier_fact_pack(REPORTS / "valid-report.md")
        inputs = campaign_inputs(facts=report)
        verdict = materialize_verdict("lowering-unknown.json", **campaign_hashes(inputs))
        result = validate_verdict(write_verdict_to_temp(verdict), inputs=inputs)
        self.assertEqual("lowering-unknown", result["classification"])
        self.assertEqual("design-rejected", result["terminal_result"])
        self.assertEqual("unchanged", result["failed_attempt_effect"])

    def test_lowering_unknown_with_failed_static_gate_is_rejected(self):
        facts = json.loads(json.dumps(VALID_FACTS))
        facts["correctness"] = {"status": "fail", "evidence": []}
        inputs = campaign_inputs(facts=facts)
        verdict = materialize_verdict("lowering-unknown.json", **campaign_hashes(inputs))
        verdict["preconditions"] = [
            {"name": "binding_valid", "status": "pass"},
            {"name": "correctness_pass", "status": "pass"},
            {"name": "lowering_observed", "status": "pass"},
            {"name": "expected_mechanism_absent", "status": "pass"},
        ]
        with self.assertRaisesRegex(VerdictValidationError, "correctness_pass"):
            validate_verdict(write_verdict_to_temp(verdict), inputs=inputs)

    def test_design_causal_invalid_is_design_error(self):
        decision = json.loads(json.dumps(VALID_DECISION))
        decision["causal_graph"] = {"nodes": ["a", "b"], "edges": []}
        inputs = campaign_inputs(decision=decision)
        verdict = materialize_verdict("design-error.json", **campaign_hashes(inputs))
        result = validate_verdict(write_verdict_to_temp(verdict), inputs=inputs)
        self.assertEqual("design-error", result["classification"])
        self.assertEqual("design-rejected", result["terminal_result"])
        self.assertEqual("increment", result["failed_attempt_effect"])

    def test_correctness_failure_is_code_error(self):
        facts = json.loads(json.dumps(VALID_FACTS))
        facts["correctness"] = {"status": "fail", "evidence": []}
        inputs = campaign_inputs(facts=facts)
        verdict = materialize_verdict("code-error.json", **campaign_hashes(inputs))
        result = validate_verdict(write_verdict_to_temp(verdict), inputs=inputs)
        self.assertEqual("code-error", result["classification"])
        self.assertEqual("candidate-failed", result["terminal_result"])

    def test_missing_binding_routes_one_repair_before_terminal(self):
        inputs = campaign_inputs(binding={"valid": False})
        verdict = materialize_verdict(
            "code-error.json",
            **campaign_hashes(inputs),
        )
        verdict["rule_id"] = "CODE.BINDING.MISSING"
        verdict["preconditions"] = [{"name": "binding_valid", "status": "fail"}]
        verdict["route"] = "coder-repair"
        verdict["repair_exhausted"] = False
        verdict["terminal_result"] = None
        verdict["failed_attempt_effect"] = None
        result = validate_verdict(write_verdict_to_temp(verdict), inputs=inputs)
        self.assertEqual("coder-repair", result["route"])
        self.assertIsNone(result["terminal_result"])

    def test_missing_binding_after_repair_exhaustion_is_terminal(self):
        inputs = campaign_inputs(binding={"valid": False})
        verdict = materialize_verdict(
            "code-error.json",
            **campaign_hashes(inputs),
        )
        verdict["rule_id"] = "CODE.BINDING.MISSING"
        verdict["preconditions"] = [{"name": "binding_valid", "status": "fail"}]
        verdict["route"] = None
        verdict["repair_exhausted"] = True
        verdict["terminal_result"] = "candidate-failed"
        verdict["failed_attempt_effect"] = "increment"
        result = validate_verdict(write_verdict_to_temp(verdict), inputs=inputs)
        self.assertEqual("candidate-failed", result["terminal_result"])

    def test_environment_evidence_gap_maps_to_blocked(self):
        inputs = campaign_inputs(facts=MISSING_OBSERVABLE_FACTS)
        verdict = materialize_verdict("evidence-gap.json", **campaign_hashes(inputs))
        result = validate_verdict(write_verdict_to_temp(verdict), inputs=inputs)
        self.assertEqual("evidence-gap", result["classification"])
        self.assertEqual("blocked", result["terminal_result"])
        self.assertEqual("unchanged", result["failed_attempt_effect"])

    def test_decision_cause_override_maps_to_design_error(self):
        facts = json.loads(json.dumps(MISSING_OBSERVABLE_FACTS))
        facts["evidence_gap_cause"] = "decision"
        inputs = campaign_inputs(facts=facts)
        verdict = materialize_verdict("evidence-gap.json", **campaign_hashes(inputs))
        verdict["classification"] = "design-error"
        verdict["terminal_result"] = "design-rejected"
        verdict["failed_attempt_effect"] = "increment"
        result = validate_verdict(write_verdict_to_temp(verdict), inputs=inputs)
        self.assertEqual("design-rejected", result["terminal_result"])

    def test_no_improvement_with_mechanism_improvement_is_classification_none(self):
        inputs = campaign_inputs()
        verdict = materialize_verdict("code-error.json", **campaign_hashes(inputs))
        verdict["rule_id"] = None
        verdict["classification"] = "none"
        verdict["terminal_result"] = "no-improvement"
        verdict["failed_attempt_effect"] = None
        result = validate_verdict(write_verdict_to_temp(verdict), inputs=inputs)
        self.assertEqual("no-improvement", result["terminal_result"])
        self.assertEqual("none", result["classification"])

    def test_verdict_hash_mismatch_is_rejected(self):
        inputs = campaign_inputs()
        verdict = materialize_verdict("lowering-unknown.json", **campaign_hashes(inputs))
        verdict["sketch_sha256"] = "0" * 64
        with self.assertRaisesRegex(VerdictValidationError, "hash"):
            validate_verdict(write_verdict_to_temp(verdict), inputs=inputs)

    def test_rule_mismatch_with_truth_is_rejected(self):
        inputs = campaign_inputs()
        verdict = materialize_verdict("lowering-unknown.json", **campaign_hashes(inputs))
        verdict["classification"] = "code-error"
        verdict["terminal_result"] = "candidate-failed"
        with self.assertRaisesRegex(VerdictValidationError, "rule"):
            validate_verdict(write_verdict_to_temp(verdict), inputs=inputs)


class FinalTuningVerdictTests(unittest.TestCase):
    def setUp(self):
        self.decision = {
            "valid": True,
            "decision_kind": "final-autotune",
            "final_tuning_contract": {
                "artifact_index": "002",
                "submission_snapshot_id": "0" * 64,
                "configurations": [
                    {"num_warps": 1, "num_stages": 2},
                    {"num_warps": 2, "num_stages": 2},
                ],
                "fallback_configuration": {"num_warps": 1, "num_stages": 2},
                "comparison_metric": "median-wall-time-ms",
                "tie_rule": "first-in-declared-order",
                "max_trials": 4,
            },
        }
        self.pinned_sha = "e" * 64
        self.report_text = (REPORTS / "valid-final-tuning-report.md").read_text(encoding="utf-8")
        self.report_text = self.report_text.replace("__PINNED_SHA256__", self.pinned_sha)
        self.report_text = self.report_text.replace("__SNAPSHOT_ID__", "0" * 64)
        self.report_dir = tempfile.mkdtemp(prefix="final-report-")
        self.report_path = Path(self.report_dir) / "report_002.md"
        self.report_path.write_text(self.report_text, encoding="utf-8")
        self.facts = extract_verifier_fact_pack(self.report_path)
        self.facts["final_configuration_tuning"]["submission_snapshot_id"] = "0" * 64

    def tearDown(self):
        shutil.rmtree(self.report_dir, ignore_errors=True)

    def inputs(self):
        return {
            "decision": self.decision,
            "sketch": VALID_SKETCH,
            "binding": VALID_BINDING,
            "profile": VALID_PROFILE,
            "facts": self.facts,
        }

    def verdict(self, inputs):
        selected_sha = sha256_canonical_json({"num_warps": 2, "num_stages": 2})
        return {
            "schema_version": 1,
            "artifact_kind": "submission-finalization",
            "artifact_index": "002",
            "decision_sha256": input_hash(inputs["decision"]),
            "sketch_sha256": input_hash(inputs["sketch"]),
            "binding_sha256": input_hash(inputs["binding"]),
            "profile_sha256": input_hash(inputs["profile"]),
            "report_fact_pack_sha256": input_hash(inputs["facts"]),
            "route": "submission-ready",
            "submission_snapshot_id": "0" * 64,
            "selected_configuration_sha256": selected_sha,
            "final_configuration_sha256": selected_sha,
            "final_candidate_sha256": self.pinned_sha,
            "final_binding_sha256": input_hash(inputs["binding"]),
            "post_pin_gates": {"binding_valid": True, "correctness": True, "lowering": True, "promotion_evidence": True, "official_evidence": True},
            "final_candidate_path": "pinned_candidate.py",
            "final_report_path": "rounds/report_002.md",
        }

    def test_submission_ready_requires_full_post_pin_evidence(self):
        inputs = self.inputs()
        result = validate_verdict(write_verdict_to_temp(self.verdict(inputs)), inputs=inputs)
        self.assertEqual("submission-ready", result["route"])
        self.assertNotIn("classification", result)
        self.assertNotIn("terminal_result", result)
        self.assertNotIn("failed_attempt_effect", result)
        self.assertEqual("improved", result["selection_outcome"])

    def test_failed_post_pin_gate_routes_blocked(self):
        self.facts["final_configuration_tuning"]["post_pin_official"]["official_evidence"] = False
        inputs = self.inputs()
        verdict = self.verdict(inputs)
        verdict["route"] = "blocked"
        verdict["post_pin_gates"] = {**verdict["post_pin_gates"], "official_evidence": False}
        result = validate_verdict(write_verdict_to_temp(verdict), inputs=inputs)
        self.assertEqual("blocked", result["route"])

    def test_finalization_verdict_rejects_campaign_fields(self):
        inputs = self.inputs()
        verdict = self.verdict(inputs)
        verdict["classification"] = "design-error"
        with self.assertRaisesRegex(VerdictValidationError, "finalization"):
            validate_verdict(write_verdict_to_temp(verdict), inputs=inputs)

    def test_selection_selector_is_deterministic(self):
        contract = self.decision["final_tuning_contract"]
        trials = self.facts["final_configuration_tuning"]["search_trials"]
        self.assertEqual({"num_warps": 2, "num_stages": 2}, select_final_tuning_configuration(contract, trials))

    def test_selector_falls_back_when_no_trial_wins(self):
        contract = self.decision["final_tuning_contract"]
        trials = [
            {
                "configuration": {"num_warps": 1, "num_stages": 2},
                "order_index": 0,
                "compile_status": "ok",
                "correctness_status": "pass",
                "reset_status": "ok",
                "statistic": 100.0,
                "eligibility": True,
            },
            {
                "configuration": {"num_warps": 2, "num_stages": 2},
                "order_index": 1,
                "compile_status": "ok",
                "correctness_status": "pass",
                "reset_status": "ok",
                "statistic": 120.0,
                "eligibility": True,
            },
        ]
        self.assertEqual({"num_warps": 1, "num_stages": 2}, select_final_tuning_configuration(contract, trials))

    def test_selector_rejects_undeclared_trial(self):
        contract = self.decision["final_tuning_contract"]
        trials = [{"configuration": {"num_warps": 4, "num_stages": 4}, "order_index": 0, "eligibility": True}]
        with self.assertRaisesRegex(VerdictValidationError, "declared"):
            select_final_tuning_configuration(contract, trials)

    def test_apply_submission_promotion_improved_updates_pair_only(self):
        before = {"last_accepted_kernel": "accepted_candidate.py", "last_accepted_report": "rounds/report_001.md", "last_accepted_round": "001", "failed_attempt_streak": 0}
        verdict = {"route": "submission-ready", "selection_outcome": "improved", "final_candidate_path": "pinned_candidate.py", "final_report_path": "rounds/report_002.md"}
        after = apply_submission_promotion(before, verdict)
        self.assertEqual("pinned_candidate.py", after["last_accepted_kernel"])
        self.assertEqual("rounds/report_002.md", after["last_accepted_report"])
        self.assertEqual("001", after["last_accepted_round"])
        self.assertEqual(0, after["failed_attempt_streak"])

    def test_apply_submission_promotion_fallback_retained_changes_nothing(self):
        before = {"last_accepted_kernel": "accepted_candidate.py", "last_accepted_report": "rounds/report_001.md", "last_accepted_round": "001", "failed_attempt_streak": 0}
        verdict = {"route": "submission-ready", "selection_outcome": "fallback-retained"}
        after = apply_submission_promotion(before, verdict)
        self.assertEqual(before, after)

    def test_apply_submission_promotion_rejects_partial_pair(self):
        verdict = {"route": "submission-ready", "selection_outcome": "improved", "final_candidate_path": "pinned_candidate.py"}
        with self.assertRaisesRegex(VerdictValidationError, "partial"):
            apply_submission_promotion({}, verdict)


class FinalizationSlotTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix="finalization-slot-"))
        (self.root / "rounds").mkdir()
        self.snapshot_id = "a" * 64

    def tearDown(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def test_fresh_allocation_picks_max_plus_one(self):
        (self.root / "rounds" / "decision_001.md").write_text("# Decision 001\n", encoding="utf-8")
        (self.root / "rounds" / "report_001.md").write_text("# Report 001\n", encoding="utf-8")
        slot = resolve_finalization_slot(self.root, self.snapshot_id)
        self.assertEqual("002", slot.artifact_index)

    def test_same_snapshot_decision_resumes_its_index(self):
        from validate_decision import extract_sections  # noqa: F401

        decision_text = """# Decision 002

## Metadata

```json
{"schema_version": 2, "decision_kind": "final-autotune", "artifact_index": "002"}
```

## Final Configuration Tuning

```json
{"submission_snapshot_id": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}
```
"""
        (self.root / "rounds" / "decision_002.md").write_text(decision_text, encoding="utf-8")
        slot = resolve_finalization_slot(self.root, self.snapshot_id)
        self.assertEqual("002", slot.artifact_index)
        self.assertIsNotNone(slot.resume_decision)

    def test_completed_snapshot_is_rejected(self):
        verdict = {"schema_version": 1, "submission_snapshot_id": self.snapshot_id, "route": "submission-ready", "artifact_index": "002"}
        (self.root / "rounds" / "verdict_002.json").write_text(json.dumps(verdict), encoding="utf-8")
        with self.assertRaisesRegex(VerdictValidationError, "already finalized"):
            resolve_finalization_slot(self.root, self.snapshot_id)

    def test_conflicting_decision_hash_blocks(self):
        decision_text = """# Decision 002

## Metadata

```json
{"schema_version": 2, "decision_kind": "final-autotune", "artifact_index": "002"}
```

## Final Configuration Tuning

```json
{"submission_snapshot_id": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"}
```
"""
        (self.root / "rounds" / "decision_001.md").write_text("# Decision 001\n", encoding="utf-8")
        (self.root / "rounds" / "decision_002.md").write_text(decision_text, encoding="utf-8")
        with self.assertRaisesRegex(VerdictValidationError, "conflicting"):
            resolve_finalization_slot(self.root, self.snapshot_id)


if __name__ == "__main__":
    unittest.main()


INTEGRATION = FIXTURES / "integration"
CLAIMS = FIXTURES / "claims"
PROFILES = FIXTURES / "profiles"
PROBES = FIXTURES / "probes"
S60 = PROBES / "qualification" / "s60"


def _replace(text: str, **replacements) -> str:
    for marker, digest in replacements.items():
        text = text.replace(marker, digest)
    return text


@contextlib.contextmanager
def materialized_integration_campaign():
    root = Path(tempfile.mkdtemp(prefix="campaign-"))
    try:
        (root / "rounds").mkdir()
        (root / "state").mkdir()
        (root / "project.md").write_text("# Project\n\n## runtime-fingerprint\n\nfixture\n", encoding="utf-8")
        (root / "baseline_adapter.py").write_text("class ModelNew: pass\n", encoding="utf-8")
        (root / "rounds" / "report_000.md").write_text("# Report 000\n", encoding="utf-8")
        candidate = root / "candidate.py"
        candidate.write_text((CANDIDATES / "valid_candidate.py").read_text(encoding="utf-8"), encoding="utf-8")
        shutil.copyfile(INTEGRATION / "campaign" / "sketch_001.json", root / "rounds" / "sketch_001.json")
        shutil.copytree(PROFILES / "valid-partial", root / "state" / "implementation_profile_snapshot")
        shutil.copyfile(CLAIMS / "valid-claim.json", root / "state" / "project_capability_claim.json")

        text = (INTEGRATION / "campaign" / "decision_001.md").read_text(encoding="utf-8")
        text = _replace(
            text,
            __SKETCH_SHA256__=sha256_file(root / "rounds" / "sketch_001.json"),
            __PROFILE_SHA256__=sha256_file(root / "state" / "implementation_profile_snapshot" / "profile.yaml"),
            __CLAIM_SHA256__=sha256_file(root / "state" / "project_capability_claim.json"),
        )
        (root / "rounds" / "decision_001.md").write_text(text, encoding="utf-8")

        binding = json.loads((INTEGRATION / "campaign" / "binding_001.json").read_text(encoding="utf-8"))
        binding["candidate_sha256"] = sha256_file(candidate)
        binding["sketch_sha256"] = sha256_file(root / "rounds" / "sketch_001.json")
        binding["decision_sha256"] = sha256_file(root / "rounds" / "decision_001.md")
        (root / "rounds" / "binding_001.json").write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        report = (INTEGRATION / "campaign" / "report_001.md").read_text(encoding="utf-8")
        report = _replace(report, __CANDIDATE_SHA256__=sha256_file(candidate))
        (root / "rounds" / "report_001.md").write_text(report, encoding="utf-8")
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


@contextlib.contextmanager
def materialized_final_tuning():
    root = Path(tempfile.mkdtemp(prefix="final-tuning-"))
    try:
        (root / "rounds").mkdir()
        (root / "state").mkdir()
        (root / "project.md").write_text("# Project\n\n## runtime-fingerprint\n\nfixture\n", encoding="utf-8")
        (root / "base.py").write_text("class Model: pass\n", encoding="utf-8")
        (root / "auto_bench.py").write_text("def main(): pass\n", encoding="utf-8")
        shutil.copyfile(INTEGRATION / "final-tuning" / "accepted_candidate.py", root / "accepted_candidate.py")
        shutil.copyfile(INTEGRATION / "final-tuning" / "pinned_candidate.py", root / "pinned_candidate.py")
        shutil.copyfile(INTEGRATION / "final-tuning" / "sketch_001.json", root / "rounds" / "sketch_001.json")
        (root / "rounds" / "binding_001.json").write_text('{"schema_version": 1, "round": "001"}\n', encoding="utf-8")
        (root / "state" / "runtime-snapshot.json").write_text('{"interpreter": "python3", "target_id": "mlu590"}\n', encoding="utf-8")
        shutil.copytree(PROFILES / "valid-partial", root / "state" / "implementation_profile_snapshot")
        shutil.copyfile(CLAIMS / "valid-claim.json", root / "state" / "project_capability_claim.json")

        anchors = {
            "candidate_sha256": sha256_file(root / "accepted_candidate.py"),
            "binding_sha256": sha256_file(root / "rounds" / "binding_001.json"),
            "sketch_sha256": sha256_file(root / "rounds" / "sketch_001.json"),
            "profile_sha256": sha256_file(root / "state" / "implementation_profile_snapshot" / "profile.yaml"),
            "claim_sha256": sha256_file(root / "state" / "project_capability_claim.json"),
            "runtime_snapshot_sha256": sha256_file(root / "state" / "runtime-snapshot.json"),
            "measurement_fingerprint_sha256": "6" * 64,
            "harness_sha256": sha256_file(root / "auto_bench.py"),
            "base_sha256": sha256_file(root / "base.py"),
        }
        snapshot_id = compute_submission_snapshot_id(anchors)
        text = (INTEGRATION / "final-tuning" / "decision_002.md").read_text(encoding="utf-8")
        text = _replace(
            text,
            __SKETCH_SHA256__=anchors["sketch_sha256"],
            __PROFILE_SHA256__=anchors["profile_sha256"],
            __CLAIM_SHA256__=anchors["claim_sha256"],
            __CANDIDATE_SHA256__=anchors["candidate_sha256"],
            __BINDING_SHA256__=anchors["binding_sha256"],
            __RUNTIME_SNAPSHOT_SHA256__=anchors["runtime_snapshot_sha256"],
            __MEASUREMENT_SHA256__=anchors["measurement_fingerprint_sha256"],
            __HARNESS_SHA256__=anchors["harness_sha256"],
            __BASE_SHA256__=anchors["base_sha256"],
            __SNAPSHOT_ID__=snapshot_id,
        )
        (root / "rounds" / "decision_002.md").write_text(text, encoding="utf-8")

        binding = json.loads((INTEGRATION / "final-tuning" / "binding_002.json").read_text(encoding="utf-8"))
        binding["candidate_sha256"] = sha256_file(root / "pinned_candidate.py")
        binding["sketch_sha256"] = sha256_file(root / "rounds" / "sketch_001.json")
        binding["decision_sha256"] = sha256_file(root / "rounds" / "decision_002.md")
        (root / "rounds" / "binding_002.json").write_text(json.dumps(binding, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        report = (INTEGRATION / "final-tuning" / "report_002.md").read_text(encoding="utf-8")
        report = _replace(report, __PINNED_SHA256__=sha256_file(root / "pinned_candidate.py"), __SNAPSHOT_ID__=snapshot_id)
        (root / "rounds" / "report_002.md").write_text(report, encoding="utf-8")
        yield root, snapshot_id
    finally:
        shutil.rmtree(root, ignore_errors=True)


class VNextIntegrationCampaignTests(unittest.TestCase):
    RUNTIME_SNAPSHOT = {
        "target_id": "mlu590",
        "implementation_profile_id": "triton_mlu",
        "triton_version": "3.6.0",
        "device_arch": "mlu-arch",
    }
    BASE_STATE = {
        "total_rounds": 0,
        "performance_miss_streak": 0,
        "failed_attempt_streak": 0,
        "last_checkpoint_round": None,
        "max_rounds": 20,
        "valid_no_improvement_limit": 3,
    }

    def test_vnext_campaign_flow_reaches_lowering_unknown_without_failed_streak_increment(self):
        from evaluate_run_policy import evaluate_terminal
        from validate_binding import validate_binding
        from validate_decision import validate_decision
        from validate_profile import load_profile, validate_project_claim
        from validate_sketch import validate_sketch

        with materialized_integration_campaign() as project_root:
            decision = validate_decision(
                project_root / "rounds" / "decision_001.md",
                project_root=project_root,
                expected_implementation_profile="triton_mlu",
            )
            sketch = validate_sketch(project_root / decision["sketch_ref"], expected_round="001")
            profile = load_profile(project_root / decision["implementation_profile_snapshot_ref"])
            claim = validate_project_claim(
                project_root / decision["project_capability_claim_ref"],
                profile=profile,
                snapshot=self.RUNTIME_SNAPSHOT,
            )
            binding = validate_binding(
                project_root / "rounds" / "binding_001.json",
                project_root=project_root,
                sketch_result=sketch,
                profile=profile,
                candidate_path=project_root / "candidate.py",
            )
            facts = extract_verifier_fact_pack(project_root / "rounds" / "report_001.md")
            inputs = {
                "decision": decision,
                "sketch": sketch,
                "profile": profile,
                "claim": claim,
                "binding": binding,
                "facts": facts,
            }
            verdict_template = json.loads((INTEGRATION / "campaign" / "verdict_001.json").read_text(encoding="utf-8"))
            verdict_text = json.dumps(verdict_template)
            verdict_text = _replace(
                verdict_text,
                __DECISION_HASH__=input_hash(inputs["decision"]),
                __SKETCH_HASH__=input_hash(inputs["sketch"]),
                __BINDING_HASH__=input_hash(inputs["binding"]),
                __PROFILE_HASH__=input_hash(inputs["profile"]),
                __FACTS_HASH__=input_hash(inputs["facts"]),
            )
            verdict_path = project_root / "rounds" / "verdict_001.json"
            verdict_path.write_text(verdict_text, encoding="utf-8")
            verdict = validate_verdict(verdict_path, inputs=inputs)
            policy = evaluate_terminal(
                self.BASE_STATE,
                verdict["terminal_result"],
                attribution=verdict["classification"],
                failed_attempt_effect=verdict["failed_attempt_effect"],
            )
            self.assertEqual("lowering-unknown", verdict["classification"])
            self.assertEqual("design-rejected", verdict["terminal_result"])
            self.assertEqual(0, policy["failed_attempt_streak"])

    def test_final_tuning_selects_pinned_candidate_and_preserves_campaign_state(self):
        from validate_binding import validate_binding
        from validate_decision import validate_decision
        from validate_profile import load_profile
        from validate_sketch import validate_sketch

        with materialized_final_tuning() as (project_root, snapshot_id):
            decision = validate_decision(
                project_root / "rounds" / "decision_002.md",
                project_root=project_root,
                expected_implementation_profile="triton_mlu",
            )
            profile = load_profile(project_root / decision["implementation_profile_snapshot_ref"])
            facts = extract_verifier_fact_pack(project_root / "rounds" / "report_002.md")
            self.assertEqual(decision["final_tuning_contract"]["submission_snapshot_id"], facts["final_configuration_tuning"]["submission_snapshot_id"])
            selected = select_final_tuning_configuration(decision["final_tuning_contract"], facts["final_configuration_tuning"]["search_trials"])
            self.assertEqual({"num_warps": 2, "num_stages": 2}, selected)
            sketch = validate_sketch(project_root / decision["sketch_ref"])
            binding = validate_binding(
                project_root / "rounds" / "binding_002.json",
                project_root=project_root,
                sketch_result=sketch,
                profile=profile,
                candidate_path=project_root / "pinned_candidate.py",
                accepted_candidate_path=project_root / "accepted_candidate.py",
                final_tuning_contract=decision["final_tuning_contract"],
            )
            inputs = {
                "decision": decision,
                "profile": profile,
                "binding": binding,
                "facts": facts,
            }
            verdict_template = json.loads((INTEGRATION / "final-tuning" / "verdict_002.json").read_text(encoding="utf-8"))
            selected_sha = sha256_canonical_json({"num_warps": 2, "num_stages": 2})
            verdict_text = json.dumps(verdict_template)
            verdict_text = _replace(
                verdict_text,
                __DECISION_HASH__=input_hash(inputs["decision"]),
                __BINDING_HASH__=input_hash(inputs["binding"]),
                __PROFILE_HASH__=input_hash(inputs["profile"]),
                __FACTS_HASH__=input_hash(inputs["facts"]),
                __SNAPSHOT_ID__=snapshot_id,
                __SELECTED_SHA256__=selected_sha,
                __PINNED_SHA256__=sha256_file(project_root / "pinned_candidate.py"),
            )
            verdict_path = project_root / "rounds" / "verdict_002.json"
            verdict_path.write_text(verdict_text, encoding="utf-8")
            verdict = validate_verdict(verdict_path, inputs=inputs)
            self.assertEqual("submission-ready", verdict["route"])
            self.assertNotIn("classification", verdict)
            self.assertNotIn("terminal_result", verdict)
            self.assertNotIn("failed_attempt_effect", verdict)
            before = {"last_accepted_kernel": "accepted_candidate.py", "last_accepted_report": "rounds/report_001.md", "last_accepted_round": "001", "failed_attempt_streak": 0}
            after = apply_submission_promotion(before, verdict)
            self.assertEqual("pinned_candidate.py", after["last_accepted_kernel"])
            self.assertEqual("rounds/report_002.md", after["last_accepted_report"])
            self.assertEqual("001", after["last_accepted_round"])
            self.assertEqual(0, after["failed_attempt_streak"])
            self.assertFalse((project_root / "state" / "final-tuning.json").exists())
            self.assertFalse((project_root / "log" / "final-tuning").exists())

    def test_failed_s60_qualification_requires_authorized_fallback_disposition(self):
        from evaluate_run_policy import evaluate_terminal  # noqa: F401
        from run_profile_probe import run_profile_probe
        from validate_decision import DecisionValidationError, validate_decision
        from validate_profile import load_profile
        from validate_probe import select_profile_probes, validate_probe_run
        from validate_sketch import validate_sketch  # noqa: F401

        s60_root = Path(tempfile.mkdtemp(prefix="s60-fallback-"))
        output_root = Path(tempfile.mkdtemp(prefix="s60-fallback-runs-"))
        try:
            shutil.copytree(S60, s60_root / "s60")
            shutil.copyfile(PROBES / "profile" / "probes" / "fake-nonzero.py", s60_root / "s60" / "probes" / "fake-dot-success.py")
            definition_path = s60_root / "s60" / "probes" / "dot-fp16.json"
            definition = json.loads(definition_path.read_text(encoding="utf-8"))
            for artifact in definition["input_artifacts"]:
                artifact["sha256"] = sha256_file(s60_root / "s60" / artifact["path"])
            definition_path.write_text(json.dumps(definition, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            profile_path = s60_root / "s60" / "profile.yaml"
            profile_value = json.loads(profile_path.read_text(encoding="utf-8"))
            profile_value["probe_catalog"] = [
                {"probe_id": "s60-dot-fp16-001", "definition_path": "probes/dot-fp16.json", "definition_sha256": sha256_file(definition_path)}
            ]
            profile_path.write_text(json.dumps(profile_value, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            requirement_path = s60_root / "s60" / "requirements" / "attention-dot-before-fallback.json"
            runtime_path = s60_root / "s60" / "runtime-snapshot.json"
            requirement = json.loads(requirement_path.read_text(encoding="utf-8"))
            runtime_snapshot = json.loads(runtime_path.read_text(encoding="utf-8"))
            profile = load_profile(profile_path)
            plan = select_profile_probes(profile, [requirement], runtime_snapshot)
            run_dir = run_profile_probe(
                profile_path=profile_path,
                probe_id=plan.selections[0].probe_id,
                target_id="s60",
                runtime_snapshot_path=runtime_path,
                qualification_requirement_path=requirement_path,
                output_root=output_root,
                run_id="s60-fallback-001",
            )
            self.assertEqual("probe-failed", validate_probe_run(run_dir)["summary"])

            disposition = {
                "disposition_id": "s60-attention-dot-fallback-001",
                "requirement": requirement,
                "requirement_sha256": sha256_canonical_json(requirement),
                "onboarding_outcome": "probe-failed",
                "promotion_disposition": "declined",
                "fallback_authorized": True,
                "reason": "dot remains unproven; explicit sum substitution for this run epoch",
                "maintainer_confirmation": {
                    "confirmed_by": "fixture-maintainer",
                    "confirmed_at": "2026-08-19T00:00:00Z",
                    "method": "explicit-user-instruction",
                },
                "probe_id": "s60-dot-fp16-001",
                "probe_definition_sha256": sha256_file(definition_path),
                "probe_result_sha256": sha256_file(run_dir / "results" / "s60-dot-fp16-001.json"),
                "primary_remains_unknown": True,
            }
            claim = {
                "schema_version": 1,
                "implementation_profile_id": "s60_triton",
                "implementation_profile_version": 1,
                "implementation_profile_sha256": sha256_file(profile_path),
                "target_id": "s60",
                "runtime_fingerprint": "s60-runtime",
                "primary_contract": "matrix.dot",
                "primary_signature": requirement["primary_signature"],
                "fallback_contract": "reduction.sum",
                "fallback_signature": requirement["fallback_signature"],
                "fallback_kind": "algorithm-substitution",
                "probe_policy": "before-fallback",
                "qualification_dispositions": [disposition],
            }
            # Delete the entire pre-campaign run; campaign history must not depend on it.
            shutil.rmtree(output_root, ignore_errors=True)
            self.assertFalse(run_dir.exists())

            root = Path(tempfile.mkdtemp(prefix="s60-campaign-"))
            try:
                (root / "rounds").mkdir()
                (root / "state").mkdir()
                (root / "project.md").write_text("# Project\n\n## runtime-fingerprint\n\nfixture\n", encoding="utf-8")
                (root / "baseline_adapter.py").write_text("class ModelNew: pass\n", encoding="utf-8")
                (root / "rounds" / "report_000.md").write_text("# Report 000\n", encoding="utf-8")
                shutil.copytree(s60_root / "s60", root / "state" / "implementation_profile_snapshot")
                (root / "state" / "project_capability_claim.json").write_text(json.dumps(claim, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                shutil.copyfile(SKETCHES / "valid-load-store-kernel.json", root / "rounds" / "sketch_001.json")

                text = (FIXTURES / "decisions" / "valid-explicit-fallback.md").read_text(encoding="utf-8")
                original_claim_sha = sha256_file(root / "state" / "project_capability_claim.json")
                text = _replace(
                    text,
                    __SKETCH_SHA256__=sha256_file(root / "rounds" / "sketch_001.json"),
                    __PROFILE_SHA256__=sha256_file(root / "state" / "implementation_profile_snapshot" / "profile.yaml"),
                    __CLAIM_SHA256__=original_claim_sha,
                    __DISPOSITION_SHA256__=sha256_canonical_json(disposition),
                )
                decision_path = root / "rounds" / "decision_001.md"
                decision_path.write_text(text, encoding="utf-8")
                decision = validate_decision(decision_path, project_root=root, expected_implementation_profile="s60_triton")
                self.assertTrue(decision["fallback_provenance"]["primary_remains_unknown"])

                # Mutating the embedded disposition invalidates qualification_disposition_sha256.
                mutated_claim = json.loads(json.dumps(claim))
                mutated_claim["qualification_dispositions"][0]["reason"] = "mutated reason"
                (root / "state" / "project_capability_claim.json").write_text(json.dumps(mutated_claim, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                new_claim_sha = sha256_file(root / "state" / "project_capability_claim.json")
                text = (root / "rounds" / "decision_001.md").read_text(encoding="utf-8")
                text = text.replace(f'"project_capability_claim_sha256": "{original_claim_sha}"', f'"project_capability_claim_sha256": "{new_claim_sha}"')
                decision_path.write_text(text, encoding="utf-8")
                with self.assertRaisesRegex(DecisionValidationError, "disposition"):
                    validate_decision(decision_path, project_root=root, expected_implementation_profile="s60_triton")
            finally:
                shutil.rmtree(root, ignore_errors=True)
        finally:
            shutil.rmtree(s60_root, ignore_errors=True)
            shutil.rmtree(output_root, ignore_errors=True)

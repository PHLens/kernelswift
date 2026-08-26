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
from vnext_common import sha256_canonical_json

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

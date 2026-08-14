import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
REFERENCES = SKILL_ROOT / "references"


def read_reference(name: str) -> str:
    return (REFERENCES / name).read_text(encoding="utf-8")


class DurableContractTests(unittest.TestCase):
    def test_team_state_contains_canonical_manifest_fields(self):
        template = read_reference("team-state-template.md")

        initial_frontmatter = """---
schema_version: 1
skill_version: 2.0.0
runtime: unset
phase: initializing
project_started_at: null
current_round: "000"
last_completed_round: null
last_accepted_round: null
last_accepted_kernel: null
last_accepted_report: null
last_completed_decision: null
last_completed_coder_result: null
last_completed_report: null
last_result: null
performance_miss_streak: 0
failed_attempt_streak: 0
total_rounds: 0
measurement_fingerprint: null
implementation_language: triton
implementation_backend: mlu
target_profile: triton_mlu
runtime_fingerprint_ref: project.md#runtime-fingerprint
blocked_incident: null
stop_reason: null
stop_timestamp: null
resume_eligible: always
resume_constraints: []
---
"""
        self.assertTrue(template.startswith(initial_frontmatter))

        for field in (
            "last_accepted_kernel",
            "last_accepted_report",
            "last_completed_round",
            "performance_miss_streak",
            "failed_attempt_streak",
            "measurement_fingerprint",
            "target_profile",
            "runtime_fingerprint_ref",
            "blocked_incident",
        ):
            with self.subTest(field=field):
                self.assertIn(field, template)

        for phase in (
            "initializing",
            "ready",
            "designing",
            "coding",
            "verifying",
            "repairing",
            "measuring",
            "blocked",
            "stopped",
        ):
            with self.subTest(phase=phase):
                self.assertIn(phase, template)

        self.assertIn(
            "| Timestamp | Phase | Round | Result | Canonical | Incident | Commit |",
            template,
        )

    def test_project_template_records_runtime_and_round_identity(self):
        template = read_reference("project-template.md")

        for field in (
            "triton_distribution",
            "triton_version",
            "backend_target",
            "backend_version",
            "device_arch",
            "measurement_fingerprint",
        ):
            with self.subTest(field=field):
                self.assertIn(field, template)

        self.assertIn(
            "| Round | Decision | Candidate | Result | Compared against | Wall ms | Device us/call | Improvement | Hypothesis verdict | Canonical after round |",
            template,
        )

    def test_report_template_mirrors_evaluation_contract(self):
        template = read_reference("report-template.md")

        for text in (
            "evidence_for_next_round",
            "confirmed | partially-confirmed | falsified | inconclusive",
            "not-applicable: Phase 0",
            "improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100",
            "device_ratio = device_us_per_call / (candidate_median_ms * 1000)",
        ):
            with self.subTest(text=text):
                self.assertIn(text, template)

        for column in ("Observable", "Expectation", "Observation", "Verdict"):
            with self.subTest(column=column):
                self.assertIn(column, template)

    def test_templates_do_not_add_future_routing_state(self):
        combined_templates = "\n".join(
            read_reference(name)
            for name in (
                "project-template.md",
                "report-template.md",
                "team-state-template.md",
            )
        )

        self.assertNotIn("target_dsl_candidates", combined_templates)
        self.assertNotIn("capability_miss_log", combined_templates)

    def test_invariants_cover_ownership_and_attribution(self):
        invariants = read_reference("invariants.md")

        for text in (
            "base.py",
            "baseline_adapter.py",
            "last_accepted_kernel",
            "Orchestrator",
            "Designer",
            "Coder",
            "Verifier",
            "separate reference and candidate scopes",
        ):
            with self.subTest(text=text):
                self.assertIn(text, invariants)

    def test_anti_patterns_are_evidence_scoped(self):
        anti_patterns = read_reference("anti-patterns.md")

        for title in (
            "Winner tree",
            "Sort-32 plus sort-64",
            "tl.gather compaction",
            "Cumsum compaction",
        ):
            with self.subTest(title=title):
                self.assertIn(title, anti_patterns)

        self.assertEqual(4, anti_patterns.count("**Evidence revision**"))
        self.assertEqual(4, anti_patterns.count("**Preconditions**"))
        self.assertEqual(4, anti_patterns.count("**Attempt**"))
        self.assertEqual(4, anti_patterns.count("**Observed failure**"))
        self.assertEqual(4, anti_patterns.count("**Reconsider when**"))

    def test_bottleneck_guidance_uses_attributable_measurements(self):
        guidance = read_reference("bottleneck-judgment.md")

        for text in (
            "per forward call",
            "benchmark wall time",
            "profiler time",
            "separate reference and candidate scopes",
            "Level 2",
            "Evaluation Contract",
        ):
            with self.subTest(text=text):
                self.assertIn(text, guidance)

    def test_legacy_log_template_is_deleted(self):
        self.assertFalse((REFERENCES / "log-template.md").exists())


if __name__ == "__main__":
    unittest.main()

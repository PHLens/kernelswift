import unittest
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
REFERENCES = SKILL_ROOT / "references"
ADAPTERS = SKILL_ROOT / "adapters"
PROMPTS = SKILL_ROOT / "prompts"


def read_reference(name: str) -> str:
    return (REFERENCES / name).read_text(encoding="utf-8")


def read_adapter(name: str) -> str:
    return (ADAPTERS / name).read_text(encoding="utf-8")


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


class RuntimeAdapterContractTests(unittest.TestCase):
    def test_claude_code_adapter_maps_common_operations(self):
        adapter = read_adapter("claude-code.md")

        for operation in (
            "start_role",
            "continue_idle_role",
            "send_advisory",
            "wait_for_completion",
            "inspect_roles",
            "end_workflow",
        ):
            with self.subTest(operation=operation):
                self.assertIn(operation, adapter)

        self.assertIn("CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1", adapter)
        self.assertIn("2.1.178", adapter)
        for removed_syntax in ("TeamCreate(", "TeamDelete(", "team_name="):
            with self.subTest(removed_syntax=removed_syntax):
                self.assertNotIn(removed_syntax, adapter)

    def test_codex_adapter_maps_collaboration_tools(self):
        adapter = read_adapter("codex.md")

        for tool in (
            "spawn_agent",
            "followup_task",
            "send_message",
            "wait_agent",
            "list_agents",
            "interrupt_agent",
            'fork_turns="none"',
        ):
            with self.subTest(tool=tool):
                self.assertIn(tool, adapter)

        self.assertNotIn("codex exec", adapter.lower())


class RoleContractTests(unittest.TestCase):
    def test_designer_owns_decisions_but_not_runtime_or_manifest(self):
        designer = (PROMPTS / "designer.md").read_text(encoding="utf-8")

        for text in (
            "last_accepted_kernel",
            "last_accepted_report",
            "Optimization Intent",
            "Unified Sketch",
            "Host Plan",
            "Evaluation Contract",
            "Pitfalls and Anti-pattern Consultation",
            "Rationale and Evidence",
            "validate_decision.py --expected-profile triton_mlu",
            "never revise",
        ):
            with self.subTest(text=text):
                self.assertIn(text, designer)

        self.assertIn("must not invent or write runtime measurements", designer)
        self.assertIn("must not edit `team-state.md`", designer)

    def test_coder_taxonomy_and_ownership_are_explicit(self):
        coder = (PROMPTS / "coder.md").read_text(encoding="utf-8")

        for text in (
            "candidate-ready",
            "design-revision-required",
            "implementation-failed",
            "environment-blocked",
            "major-deviation",
            "capability-miss",
            "Coder never returns accepted",
            "last_accepted_kernel",
            "validate_decision.py",
            "coder_result_NNN.md",
        ):
            with self.subTest(text=text):
                self.assertIn(text, coder)

        for forbidden_write in (
            "decision_NNN.md",
            "target profile",
            "team-state.md",
            "project overview",
            "report_NNN.md",
        ):
            with self.subTest(forbidden_write=forbidden_write):
                self.assertIn(f"must not edit {forbidden_write}", coder)

    def test_triton_mlu_profile_is_complete_and_evidence_backed(self):
        profile = (PROMPTS / "coder_targets" / "triton_mlu.md").read_text(
            encoding="utf-8"
        )

        for heading in (
            "# Target Profile: triton_mlu",
            "## Identity and Match",
            "## Runtime and Launcher Conventions",
            "## Supported Primitives",
            "## Constrained Primitives",
            "## Unsupported Primitives",
            "## Unknown Primitives",
            "## Allowed Fallbacks",
            "## Target-specific Pitfalls",
            "## Evidence Ledger",
        ):
            with self.subTest(heading=heading):
                self.assertIn(heading, profile)

        for column in (
            "Primitive",
            "Status",
            "Constraint",
            "Evidence",
            "Failure classification",
        ):
            with self.subTest(column=column):
                self.assertIn(column, profile)

        for primitive in (
            "`tl.load`",
            "`tl.store`",
            "`tl.arange`",
            "`tl.program_id`",
            "`tl.dot`",
            "`tl.argmax`",
            "`tl.reshape`",
            "`tl.zeros`",
            "`tl.make_block_ptr`",
            "`vectorize`",
            "`async_copy`",
        ):
            with self.subTest(primitive=primitive):
                self.assertIn(primitive, profile)

        self.assertIn("num_warps=1", profile)
        self.assertIn("num_warps=2", profile)
        self.assertIn("num_stages=2", profile)
        self.assertIn("runtime introspection", profile)

    def test_no_inactive_target_stubs_or_fake_lowering_claims(self):
        targets = PROMPTS / "coder_targets"
        for name in (
            "triton_cuda.md",
            "triton_hip.md",
            "triton_ascend.md",
            "tilelang.md",
        ):
            with self.subTest(name=name):
                self.assertFalse((targets / name).exists())

        profile = (targets / "triton_mlu.md").read_text(encoding="utf-8")
        self.assertNotRegex(profile.lower(), r"tl\.make_block_ptr.*register tile")
        self.assertNotRegex(profile.lower(), r"tl\.zeros.*smem")


class VerifierContractTests(unittest.TestCase):
    def test_verifier_owns_runtime_evidence_and_preserves_other_artifacts(self):
        verifier = (PROMPTS / "verifier.md").read_text(encoding="utf-8")

        for text in (
            "sole authoritative runtime owner",
            "last_accepted_kernel",
            "correctness before timing",
            "exactly one same-round Coder repair",
            "implementation-repair-required",
            "measurement-incomplete",
            "Evaluation Contract",
            "normalize",
            "reference_baseline_adapter",
            "candidate_triton_operator_001",
            "evidence_for_next_round",
        ):
            with self.subTest(text=text):
                self.assertIn(text, verifier)

        for path in (
            "candidate source",
            "decision_NNN.md",
            "team-state.md",
            "project overview",
        ):
            with self.subTest(path=path):
                self.assertIn(f"must not edit {path}", verifier)

    def test_verifier_timing_and_result_rules_are_exact(self):
        verifier = (PROMPTS / "verifier.md").read_text(encoding="utf-8")

        for text in (
            "accepted reference, candidate",
            "unrounded median",
            "improvement_pct >= 5.0",
            "accepted",
            "no-improvement",
            "candidate-failed",
            "design-rejected",
            "accepted|no-improvement|candidate-failed|design-rejected",
            "Level 0",
            "Level 1",
            "Level 2",
            "Level 3",
            "device_us_per_call",
            "kernel_count_per_call",
            "device_ratio",
        ):
            with self.subTest(text=text):
                self.assertIn(text, verifier)

    def test_environment_incidents_are_nonterminal_and_counter_neutral(self):
        verifier = (PROMPTS / "verifier.md").read_text(encoding="utf-8")

        for text in (
            "incident_NNN_<UTC-timestamp>.md",
            "does not write a terminal result",
            "does not change total_rounds",
            "does not change either progress streak",
            "measurement-bound",
            "diminishing returns",
            "upbound reached",
            "resource exhausted",
            "user intervention",
        ):
            with self.subTest(text=text):
                self.assertIn(text, verifier)


if __name__ == "__main__":
    unittest.main()

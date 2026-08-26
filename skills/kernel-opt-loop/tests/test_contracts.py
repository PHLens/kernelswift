import unittest
import os
import re
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = SKILL_ROOT / "scripts"
REPO_ROOT = SKILL_ROOT.parents[1]
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
schema_version: 2
skill_version: 3.0.0
contract_version: 3
semantic_contract: typed-sketch-v1
attribution_contract: verdict-v1
runtime: unset
phase: initializing
workflow_status: running
run_epoch: 1
project_started_at: null
current_round: "000"
last_completed_round: null
last_accepted_round: null
last_accepted_kernel: null
last_accepted_report: null
last_completed_decision: null
last_completed_sketch: null
last_completed_binding: null
last_completed_verdict: null
last_attribution: null
last_completed_coder_result: null
last_completed_report: null
last_result: null
performance_miss_streak: 0
failed_attempt_streak: 0
total_rounds: 0
max_rounds: 20
valid_no_improvement_limit: 3
adoption_threshold_pct: 5
measurement_fingerprint: null
target_mode: null
target_value: null
target_measurement_fingerprint: null
target_source: null
last_checkpoint_round: null
base_branch: null
base_commit: null
run_branch: null
measurement_exclusive: false
implementation_language: triton
implementation_backend: unset
target_profile: unset
implementation_profile_snapshot_ref: null
implementation_profile_snapshot_sha256: null
project_capability_claim_ref: null
project_capability_claim_sha256: null
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
            "contract_version",
            "semantic_contract",
            "attribution_contract",
            "implementation_profile_snapshot_ref",
            "project_capability_claim_ref",
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

        for field in (
            "## Optional Target",
            "absolute_latency_ms",
            "speedup_vs_baseline",
            "wall_time_ms",
            "source: user",
            "## Git Run Identity",
            "base_branch",
            "base_commit",
            "run_branch",
        ):
            with self.subTest(field=field):
                self.assertIn(field, template)

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

        for text in (
            "screened-out",
            "verification_tier: baseline | screening | authoritative",
            "## Screening Evidence",
            "screening_pairs",
            "both pairs are at least 10% slower",
            "required | not-run: screened-out | not-run: not-needed",
            "valid-no-improvement-limit",
            "round-budget-exhausted",
            "Only authoritative timing can yield `accepted` or `no-improvement`",
        ):
            with self.subTest(text=text):
                self.assertIn(text, template)

    def test_team_state_contains_v2_workflow_policy(self):
        template = read_reference("team-state-template.md")
        for field in (
            "workflow_status: running",
            "run_epoch: 1",
            "max_rounds: 20",
            "valid_no_improvement_limit: 3",
            "adoption_threshold_pct: 5",
            "target_mode: null",
            "target_measurement_fingerprint: null",
            "last_checkpoint_round: null",
            "base_branch: null",
            "run_branch: null",
            "measurement_exclusive: false",
            "## Policy Revisions",
        ):
            with self.subTest(field=field):
                self.assertIn(field, template)

        self.assertIn("workflow_status", template)
        self.assertIn("screened-out", template)

    def test_role_context_template_has_rehydrate_fields(self):
        template = read_reference("role-context-template.md")
        for field in (
            "role_contract_sha256",
            "context_epoch",
            "last_completed_round",
            "recent_three_round_evidence",
            "open_hypotheses",
            "artifact_read_hashes",
        ):
            with self.subTest(field=field):
                self.assertIn(field, template)

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

    def test_context_naming_uses_only_context_files_and_no_state_aliases(self):
        skill_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (
                *REFERENCES.glob("*.md"),
                *PROMPTS.glob("*.md"),
                SKILL_ROOT / "SKILL.md",
            )
        )
        for alias in (
            "designer_state.md",
            "coder_state.md",
            "verifier_state.md",
        ):
            with self.subTest(alias=alias):
                self.assertNotIn(alias, skill_text)
        for canonical in (
            "state/designer_context.md",
            "state/coder_context.md",
            "state/verifier_context.md",
        ):
            self.assertIn(canonical, skill_text)

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

    def test_v2_contracts_share_terminal_and_future_scope_boundaries(self):
        report = read_reference("report-template.md")
        verifier = (PROMPTS / "verifier.md").read_text(encoding="utf-8")
        orchestrator = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
        for result in (
            "accepted",
            "no-improvement",
            "screened-out",
            "design-rejected",
            "candidate-failed",
            "aborted",
        ):
            for owner_text in (report, verifier, orchestrator):
                with self.subTest(result=result):
                    self.assertIn(result, owner_text)

        reference_text = "\n".join(
            path.read_text(encoding="utf-8") for path in REFERENCES.glob("*.md")
        )
        for future_only in ("KernelWiki API", "token-accounting telemetry", "daemon"):
            with self.subTest(future_only=future_only):
                self.assertNotIn(future_only, reference_text)

    def test_raw_profiler_traces_are_gitignored_and_markdown_is_complete(self):
        gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("*.pt.trace.json", gitignore)
        self.assertIn("**/log/", gitignore)

        tracked_evidence = "\n".join(
            read_reference(name)
            for name in (
                "project-template.md",
                "report-template.md",
                "team-state-template.md",
            )
        )
        self.assertNotIn("*.pt.trace.json", tracked_evidence)

        required_markdown = (
            SKILL_ROOT / "SKILL.md",
            REFERENCES / "role-context-template.md",
            PROMPTS / "designer.md",
            PROMPTS / "coder.md",
            PROMPTS / "verifier.md",
            ADAPTERS / "codex.md",
            ADAPTERS / "claude-code.md",
        )
        for path in required_markdown:
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file())
                self.assertEqual(0, path.read_text(encoding="utf-8").count("```") % 2)


class RuntimeAdapterContractTests(unittest.TestCase):
    def test_adapters_declare_v2_continuation_and_rehydrate_behavior(self):
        for adapter_name in ("codex.md", "claude-code.md"):
            adapter = read_adapter(adapter_name)
            for text in (
                "runtime_capabilities:",
                "persistent_role_session:",
                "effective_context_mode:",
                "autonomous_scope: one-live-orchestrator-session",
                "role-context-template.md",
                "compact bootstrap delta",
                "cold rehydrate",
                "three-round reconciliation",
                "persistent_role_session: false",
                "effective_context_mode: rehydrate",
            ):
                with self.subTest(adapter=adapter_name, text=text):
                    self.assertIn(text, adapter)
            self.assertNotIn("daemon", adapter.lower())
            self.assertNotIn("autonomous continuation across sessions", adapter)

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
    def test_v2_role_contracts_define_context_and_measurement_boundaries(self):
        designer = (PROMPTS / "designer.md").read_text(encoding="utf-8")
        coder = (PROMPTS / "coder.md").read_text(encoding="utf-8")
        verifier = (PROMPTS / "verifier.md").read_text(encoding="utf-8")

        for text in (
            "three to five",
            "change_family",
            "different change family",
            "role-context-template.md",
            "Verifier-backed observation",
            "must remain idle while Verifier owns measurement-exclusive",
        ):
            with self.subTest(role="designer", text=text):
                self.assertIn(text, designer)

        for text in (
            "warm-up / compile smoke",
            "at most twice",
            "attempt ledger",
            "must remain idle while Verifier owns measurement-exclusive",
        ):
            with self.subTest(role="coder", text=text):
                self.assertIn(text, coder)

        for text in (
            "screened-out",
            "two short interleaved",
            "10%",
            "measurement-exclusive",
            "liveness watchdog",
            "must not promote a screen result to `accepted` or `no-improvement`",
        ):
            with self.subTest(role="verifier", text=text):
                self.assertIn(text, verifier)

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
            "validate_decision.py --expected-profile <manifest target_profile>",
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

    def test_triton_gcu_profile_is_complete_and_evidence_scoped(self):
        profile = (PROMPTS / "coder_targets" / "triton_gcu.md").read_text(
            encoding="utf-8"
        )

        for heading in (
            "# Target Profile: triton_gcu",
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

        for primitive in (
            "`tl.load`",
            "`tl.store`",
            "`tl.arange`",
            "`tl.program_id`",
            "`tl.zeros`",
            "`tl.reshape`",
            "`tl.max`",
            "`tl.argmax`",
            "`tl.dot`",
            "`tl.make_block_ptr`",
            "`fast_libentry`",
        ):
            with self.subTest(primitive=primitive):
                self.assertIn(primitive, profile)

        for evidence in (
            "backend: gcu",
            "target_profile: triton_gcu",
            "s60/groupedtopk/triton_grouped_topk_001.py",
            "major=3, minor=0",
            "device=\"gcu\"",
            "torch.gcu.synchronize()",
        ):
            with self.subTest(evidence=evidence):
                self.assertIn(evidence, profile)

        self.assertIn("`tl.dot` | Unknown", profile)
        self.assertIn("`fast_libentry` | Unknown", profile)

    def test_triton_cuda_profile_is_complete_and_evidence_scoped(self):
        profile = (PROMPTS / "coder_targets" / "triton_cuda.md").read_text(
            encoding="utf-8"
        )

        for heading in (
            "# Target Profile: triton_cuda",
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

        for primitive in (
            "`tl.load`",
            "`tl.store`",
            "`tl.arange`",
            "`tl.program_id`",
            "`tl.zeros`",
            "`tl.reshape`",
            "`tl.max`",
            "`tl.argmax`",
            "`tl.sum`",
            "`tl.exp`",
            "`tl.where`",
            "`tl.dot`",
            "`tl.make_block_ptr`",
            "`fast_libentry`",
            "`num_warps`",
            "`num_stages`",
        ):
            with self.subTest(primitive=primitive):
                self.assertIn(primitive, profile)

        for evidence in (
            "backend: cuda",
            "target_profile: triton_cuda",
            "scripts/bi150_triton_smoke.py",
            "scripts/bi150_groupedtopk_probe.py",
            "docs/bi150-kernel-opt-loop-prep.md",
            "Iluvatar BI-V150",
            "device=\"cuda\"",
            "torch.cuda.synchronize()",
            "COREX_VERSION=4.4.0",
        ):
            with self.subTest(evidence=evidence):
                self.assertIn(evidence, profile)

        self.assertIn("`tl.dot` | Supported", profile)
        self.assertIn("`fast_libentry` | Unknown", profile)
        self.assertIn("`num_warps` | Unknown", profile)

    def test_no_inactive_target_stubs_or_fake_lowering_claims(self):
        targets = PROMPTS / "coder_targets"
        for name in (
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
            "accepted|no-improvement|screened-out|candidate-failed|design-rejected|aborted",
            "Level 0",
            "authoritative timing",
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
            "target-reached",
            "valid-no-improvement-limit",
            "round-budget-exhausted",
            "user-intervention",
        ):
            with self.subTest(text=text):
                self.assertIn(text, verifier)


class OrchestratorContractTests(unittest.TestCase):
    def setUp(self):
        self.skill = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")

    def test_required_runtime_neutral_sections_exist(self):
        for heading in (
            "## When to use",
            "## Required inputs",
            "## Runtime selection",
            "## Agent bootstrap contract",
            "## Phase 0",
            "## Round N",
            "## Routing and state transitions",
            "## Knowledge lift",
            "## References",
            "## Continuous run controller",
            "## Global termination policy",
            "## Measurement-exclusive phases",
            "## Run epochs and recovery",
            "## Git evidence ledger",
        ):
            with self.subTest(heading=heading):
                self.assertIn(heading, self.skill)

        self.assertNotIn("write the new kernel file", self.skill.lower())
        self.assertNotIn("log.md", self.skill)

    def test_runtime_selection_and_bootstrap_are_portable(self):
        self.assertTrue(self.skill.startswith("---\nname: kernel-opt-loop\n"))
        normalized_skill = " ".join(self.skill.split())
        selection = (
            "Codex collaboration when exposed, Claude Code agent teams when "
            "enabled, then sequential fallback"
        )
        self.assertIn(selection, self.skill)
        self.assertIn("Load exactly one runtime adapter", self.skill)

        for text in (
            "You are the <role> for kernel-opt-loop.",
            "Before taking any action, read these files completely and follow them:",
            "Role contract: <absolute-skill-root>/prompts/<role>.md",
            "Runtime adapter: <absolute-skill-root>/adapters/<runtime>.md",
            "Do not rely on parent conversation history.",
            "Do not write files outside your declared ownership.",
            "Report completion through the runtime adapter.",
        ):
            with self.subTest(text=text):
                self.assertIn(" ".join(text.split()), normalized_skill)

    def test_phase_zero_and_round_state_machine_are_deterministic(self):
        for text in (
            "baseline_adapter.py",
            "base bytes, NUL, harness bytes, NUL",
            "sort_keys=True",
            "separators=(',', ':')",
            'last_completed_round: "000"',
            'last_accepted_round: "000"',
            "last_accepted_kernel: baseline_adapter.py",
            "last_accepted_report: rounds/report_000.md",
            "Round number is `total_rounds + 1`",
            "last_accepted_kernel",
            "last_accepted_report",
            "commit, then and only then dispatch the next round",
        ):
            with self.subTest(text=text):
                self.assertIn(text, self.skill)

        for key in (
            '"shape"',
            '"dtype"',
            '"device"',
            '"warmup"',
            '"repeat"',
            '"profile_mode"',
            '"profile_warmup"',
            '"profile_iterations"',
        ):
            with self.subTest(key=key):
                self.assertIn(key, self.skill)

    def test_routing_counters_stop_and_resume_are_explicit(self):
        for text in (
            "candidate-ready",
            "major-deviation",
            "capability-miss",
            "implementation-failed",
            "environment-blocked",
            "implementation-repair-required",
            "measurement-incomplete",
            "accepted",
            "no-improvement",
            "design-rejected",
            "candidate-failed",
            "aborted",
            "performance_miss_streak",
            "failed_attempt_streak",
            "Environment incidents update neither counter nor `total_rounds`",
            "screened-out",
            "target-reached",
            "valid-no-improvement-limit",
            "round-budget-exhausted",
            "user-intervention",
            "never reopen a completed decision",
            "explicit user approval",
        ):
            with self.subTest(text=text):
                self.assertIn(text, self.skill)

    def test_v2_continuous_controller_contract_is_complete(self):
        for text in (
            "evaluate_run_policy.py",
            "max_rounds: 20",
            "valid_no_improvement_limit: 3",
            "round_result is not workflow termination",
            "last_checkpoint_round",
            "kernel-opt/<operator>-<run-epoch-or-timestamp>",
            "terminal artifact gate -> terminal commit -> evaluate_run_policy.py",
            "workflow_status=running -> optional checkpoint -> continue idle Designer",
            "workflow_status=stopped -> final summary commit -> end_workflow",
            "workflow_status=blocked -> incident commit -> blocking report -> end live run",
        ):
            with self.subTest(text=text):
                self.assertIn(text, self.skill)

        self.assertNotIn("Designer may reject another non-user stop", self.skill)
        self.assertNotIn("normalized device ratio is below 5%", self.skill)

    def test_v2_terminal_contract_and_evaluator_are_present(self):
        report = read_reference("report-template.md")
        verifier = (PROMPTS / "verifier.md").read_text(encoding="utf-8")
        for text in (report, verifier, self.skill):
            for result in (
                "accepted",
                "no-improvement",
                "screened-out",
                "design-rejected",
                "candidate-failed",
                "aborted",
            ):
                with self.subTest(result=result, source=id(text)):
                    self.assertIn(result, text)

        evaluator = SKILL_ROOT / "scripts" / "evaluate_run_policy.py"
        self.assertTrue(evaluator.is_file())
        self.assertGreater(evaluator.stat().st_size, 0)
        self.assertTrue(os.access(evaluator, os.X_OK))


class CrossFileContractTests(unittest.TestCase):
    def test_final_structure_exists_and_is_nonempty(self):
        expected = (
            "SKILL.md",
            "adapters/claude-code.md",
            "adapters/codex.md",
            "prompts/designer.md",
            "prompts/coder.md",
            "prompts/verifier.md",
            "prompts/coder_targets/triton_mlu.md",
            "prompts/coder_targets/triton_gcu.md",
            "prompts/coder_targets/triton_cuda.md",
            "references/anti-patterns.md",
            "references/bottleneck-judgment.md",
            "references/decision-template.md",
            "references/invariants.md",
            "references/project-template.md",
            "references/report-template.md",
            "references/team-state-template.md",
            "scripts/make_baseline_adapter.py",
            "scripts/summarize_trace.py",
            "scripts/validate_decision.py",
            "tests/test_validate_decision.py",
            "tests/test_helpers.py",
            "tests/test_contracts.py",
        )
        for relative in expected:
            with self.subTest(relative=relative):
                path = SKILL_ROOT / relative
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)

    def test_decision_headings_and_result_enums_are_consistent(self):
        decision_files = (
            REFERENCES / "decision-template.md",
            SKILL_ROOT / "tests/fixtures/decisions/kernel-valid.md",
            SKILL_ROOT / "tests/fixtures/decisions/host-valid.md",
            SKILL_ROOT / "tests/fixtures/decisions/mixed-valid.md",
        )
        headings = (
            "Metadata",
            "Optimization Intent",
            "Unified Sketch",
            "Host Plan",
            "Evaluation Contract",
            "Pitfalls and Anti-pattern Consultation",
            "Rationale and Evidence",
        )
        for path in decision_files:
            text = path.read_text(encoding="utf-8")
            for heading in headings:
                with self.subTest(path=path.name, heading=heading):
                    self.assertIn(f"## {heading}", text)

        coder = (PROMPTS / "coder.md").read_text(encoding="utf-8")
        for result in (
            "candidate-ready",
            "design-revision-required",
            "implementation-failed",
            "environment-blocked",
        ):
            self.assertIn(result, coder)

        verifier = (PROMPTS / "verifier.md").read_text(encoding="utf-8")
        orchestrator = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
        report = read_reference("report-template.md")
        for result in (
            "accepted",
            "no-improvement",
            "design-rejected",
            "candidate-failed",
        ):
            for owner_text in (verifier, orchestrator, report):
                with self.subTest(result=result):
                    self.assertIn(result, owner_text)

        self.assertIn("Coder never returns accepted", coder)
        self.assertNotRegex(coder, r"(?m)^Result:\s*accepted\b")

    def test_manifest_phases_counters_and_ownership_match(self):
        manifest = read_reference("team-state-template.md")
        orchestrator = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
        phases = (
            "initializing",
            "ready",
            "designing",
            "coding",
            "verifying",
            "repairing",
            "measuring",
            "blocked",
            "stopped",
        )
        for phase in phases:
            self.assertIn(phase, manifest)
            self.assertIn(phase, orchestrator)

        for counter in (
            "total_rounds",
            "performance_miss_streak",
            "failed_attempt_streak",
        ):
            self.assertIn(counter, manifest)
            self.assertIn(counter, orchestrator)

        ownership = {
            "designer.md": ("team-state.md", "report_NNN.md"),
            "coder.md": ("decision_NNN.md", "team-state.md", "report_NNN.md"),
            "verifier.md": ("candidate source", "decision_NNN.md", "team-state.md"),
        }
        for name, forbidden in ownership.items():
            text = (PROMPTS / name).read_text(encoding="utf-8")
            self.assertIn("must not edit", text)
            for target in forbidden:
                with self.subTest(role=name, target=target):
                    self.assertIn(target, text)

    def test_profiler_and_evaluation_contract_are_mirrored(self):
        verifier = (PROMPTS / "verifier.md").read_text(encoding="utf-8")
        report = read_reference("report-template.md")
        for text in (
            "Observable",
            "Expectation",
            "Observation",
            "Verdict",
            "device_us_per_call",
            "kernel_count_per_call",
            "device_ratio",
            "evidence_for_next_round",
        ):
            self.assertIn(text.lower(), verifier.lower())
            self.assertIn(text.lower(), report.lower())

        self.assertIn(
            "improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100",
            report,
        )
        self.assertIn(
            "device_ratio = device_us_per_call / (candidate_median_ms * 1000)",
            report,
        )
        self.assertIn("unrounded median", verifier)

    def test_runtime_syntax_is_adapter_local_and_future_scope_is_absent(self):
        runtime_neutral = [SKILL_ROOT / "SKILL.md"]
        runtime_neutral.extend(PROMPTS.rglob("*.md"))
        runtime_neutral.extend(REFERENCES.rglob("*.md"))
        neutral_text = "\n".join(
            path.read_text(encoding="utf-8") for path in runtime_neutral
        )
        for syntax in (
            "spawn_agent",
            "followup_task",
            "send_message",
            "wait_agent",
            "list_agents",
            "interrupt_agent",
            "TeamCreate(",
            "TeamDelete(",
            "team_name=",
        ):
            with self.subTest(syntax=syntax):
                self.assertNotIn(syntax, neutral_text)

        all_skill_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in SKILL_ROOT.rglob("*.md")
        )
        for forbidden in (
            "target_dsl_candidates",
            "capability_miss_log",
            "KernelWiki API",
            "deterministic lowering implementation",
            "deep-profiler implementation",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, all_skill_text)

        for name in (
            "triton_hip.md",
            "triton_ascend.md",
            "tilelang.md",
        ):
            self.assertFalse((PROMPTS / "coder_targets" / name).exists())

    def test_markdown_fences_close_and_validator_is_executable(self):
        for path in SKILL_ROOT.rglob("*.md"):
            with self.subTest(path=path.relative_to(SKILL_ROOT)):
                self.assertEqual(0, path.read_text(encoding="utf-8").count("```") % 2)

        validator = SKILL_ROOT / "scripts/validate_decision.py"
        self.assertTrue(os.access(validator, os.X_OK))
class VNextContractTests(unittest.TestCase):
    def test_vnext_artifact_ownership_boundaries_are_exact(self):
        skill = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
        designer = (PROMPTS / "designer.md").read_text(encoding="utf-8")
        coder = (PROMPTS / "coder.md").read_text(encoding="utf-8")
        verifier = (PROMPTS / "verifier.md").read_text(encoding="utf-8")

        import re as _re

        def normalize(value: str) -> str:
            return _re.sub(r"\s+", " ", value)

        for text, phrase in (
            (skill, "runs one bounded pre-campaign probe lifecycle"),
            (skill, "stop without entering Phase 0"),
            (skill, "unrelated Unknowns are ignored and ambiguous matches fail"),
            (skill, "stops as `promotion-pending`"),
            (skill, "materializes the project capability claim"),
            (skill, "freezes the implementation-profile snapshot"),
            (skill, "validates `verdict_NNN.json`; it may route one `code-error` repair"),
            (skill, "`lowering-unknown` terminates as `design-rejected` with unchanged failed"),
            (skill, "`resolve_finalization_slot()`"),
            (skill, "submission-ready|blocked` through the separate finalization verdict branch"),
            (skill, "never calls `evaluate_terminal()`"),
            (skill, "no runtime/online `@triton.autotune`"),
            (designer, "without writing a Decision, Sketch, or campaign file"),
            (designer, "never equates an Unknown capability with unavailable"),
            (designer, "reuses the accepted Sketch"),
            (coder, "passes the deterministic conformance checker before `candidate-ready`"),
            (coder, "at most one pinned candidate derived from the accepted source"),
            (coder, "never owns pre-campaign qualification"),
            (verifier, "without assigning design or code blame"),
            (verifier, "without a persisted selection artifact"),
            (verifier, "atomically writes the sealed report once"),
            (verifier, "Search measurements never authorize a submission"),
        ):
            with self.subTest(text=text[:40]):
                self.assertIn(phrase, normalize(text))

    def test_finalization_uses_existing_families_and_no_new_state(self):
        normalize = lambda value: re.sub(r"\s+", " ", value)
        team_state = normalize(read_reference("team-state-template.md"))
        skill = (SKILL_ROOT / "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("contract_version: 3", team_state)
        self.assertIn("No finalization-specific state field is added", team_state)
        self.assertIn("artifact_kind: submission-finalization", normalize(read_reference("report-template.md")))
        self.assertIn("last_accepted_round", skill)
        self.assertNotIn("final-tuning.json", team_state)

    def test_attribution_effects_are_explicit_in_the_evaluator(self):
        policy = (SCRIPTS / "evaluate_run_policy.py").read_text(encoding="utf-8")
        self.assertIn("ATTRIBUTIONS", policy)
        self.assertIn("FAILED_ATTEMPT_EFFECTS", policy)
        self.assertIn("lowering-unknown", policy)
        self.assertIn("_apply_failed_attempt_effect", policy)
        self.assertIn("must be supplied together", policy)



if __name__ == "__main__":
    unittest.main()

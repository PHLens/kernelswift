import contextlib
import io
import json
from pathlib import Path
import sys
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from evaluate_run_policy import (  # noqa: E402
    RunPolicyError,
    evaluate_block,
    evaluate_terminal,
    main,
)


def state(**overrides):
    value = {
        "total_rounds": 0,
        "performance_miss_streak": 0,
        "failed_attempt_streak": 0,
        "last_checkpoint_round": None,
        "max_rounds": 20,
        "valid_no_improvement_limit": 3,
    }
    value.update(overrides)
    return value


class RunPolicyTests(unittest.TestCase):
    def test_third_valid_no_improvement_stops_without_checkpoint(self):
        outcome = evaluate_terminal(
            state(total_rounds=2, performance_miss_streak=2),
            "no-improvement",
        )

        self.assertEqual(outcome["total_rounds"], 3)
        self.assertEqual(outcome["performance_miss_streak"], 3)
        self.assertEqual(outcome["workflow_status"], "stopped")
        self.assertEqual(outcome["phase"], "stopped")
        self.assertEqual(outcome["stop_reason"], "valid-no-improvement-limit")
        self.assertFalse(outcome["dispatch_next_round"])
        self.assertFalse(outcome["emit_checkpoint"])
        self.assertIsNone(outcome["last_checkpoint_round"])

    def test_accepted_resets_both_streaks(self):
        outcome = evaluate_terminal(
            state(performance_miss_streak=2, failed_attempt_streak=4), "accepted"
        )

        self.assertEqual(outcome["total_rounds"], 1)
        self.assertEqual(outcome["performance_miss_streak"], 0)
        self.assertEqual(outcome["failed_attempt_streak"], 0)
        self.assertTrue(outcome["dispatch_next_round"])

    def test_screened_out_only_consumes_a_round(self):
        outcome = evaluate_terminal(
            state(performance_miss_streak=2, failed_attempt_streak=4),
            "screened-out",
        )

        self.assertEqual(outcome["total_rounds"], 1)
        self.assertEqual(outcome["performance_miss_streak"], 2)
        self.assertEqual(outcome["failed_attempt_streak"], 4)

    def test_failed_results_increment_only_the_failed_attempt_streak(self):
        for result in ("design-rejected", "candidate-failed", "aborted"):
            with self.subTest(result=result):
                outcome = evaluate_terminal(
                    state(performance_miss_streak=2, failed_attempt_streak=4), result
                )
                self.assertEqual(outcome["total_rounds"], 1)
                self.assertEqual(outcome["performance_miss_streak"], 2)
                self.assertEqual(outcome["failed_attempt_streak"], 5)

    def test_twentieth_round_stops_for_the_round_budget(self):
        outcome = evaluate_terminal(state(total_rounds=19), "screened-out")

        self.assertEqual(outcome["total_rounds"], 20)
        self.assertEqual(outcome["workflow_status"], "stopped")
        self.assertEqual(outcome["stop_reason"], "round-budget-exhausted")
        self.assertFalse(outcome["dispatch_next_round"])
        self.assertFalse(outcome["emit_checkpoint"])

    def test_user_stop_precedes_target_and_other_stop_conditions(self):
        outcome = evaluate_terminal(
            state(total_rounds=19, performance_miss_streak=2),
            "no-improvement",
            target_reached=True,
            user_stop_requested=True,
        )

        self.assertEqual(outcome["stop_reason"], "user-intervention")

    def test_target_precedes_miss_limit_and_round_budget(self):
        outcome = evaluate_terminal(
            state(total_rounds=19, performance_miss_streak=2),
            "no-improvement",
            target_reached=True,
        )

        self.assertEqual(outcome["stop_reason"], "target-reached")

    def test_running_third_round_emits_a_checkpoint_once(self):
        outcome = evaluate_terminal(state(total_rounds=2), "screened-out")

        self.assertTrue(outcome["emit_checkpoint"])
        self.assertEqual(outcome["last_checkpoint_round"], 3)

    def test_checkpoint_is_suppressed_when_round_was_already_reported(self):
        outcome = evaluate_terminal(
            state(total_rounds=2, last_checkpoint_round=3), "screened-out"
        )

        self.assertFalse(outcome["emit_checkpoint"])
        self.assertEqual(outcome["last_checkpoint_round"], 3)

    def test_block_preserves_counters(self):
        outcome = evaluate_block(
            state(
                total_rounds=7,
                performance_miss_streak=2,
                failed_attempt_streak=4,
            ),
            "benchmark host unavailable",
        )

        self.assertEqual(outcome["total_rounds"], 7)
        self.assertEqual(outcome["performance_miss_streak"], 2)
        self.assertEqual(outcome["failed_attempt_streak"], 4)
        self.assertEqual(outcome["workflow_status"], "blocked")
        self.assertEqual(outcome["phase"], "blocked")
        self.assertEqual(outcome["blocked_incident"], "benchmark host unavailable")
        self.assertFalse(outcome["dispatch_next_round"])
        self.assertFalse(outcome["emit_checkpoint"])

    def test_unknown_result_raises_policy_error(self):
        with self.assertRaisesRegex(RunPolicyError, "unknown terminal result"):
            evaluate_terminal(state(), "not-a-result")

    def test_invalid_numeric_or_malformed_state_raises_policy_error(self):
        invalid_states = (
            None,
            state(total_rounds=True),
            state(performance_miss_streak=-1),
            state(failed_attempt_streak=1.5),
            state(max_rounds=0),
            state(valid_no_improvement_limit=False),
            {"total_rounds": 0},
        )

        for invalid_state in invalid_states:
            with self.subTest(state=invalid_state):
                with self.assertRaises(RunPolicyError):
                    evaluate_terminal(invalid_state, "accepted")

    def test_block_requires_nonempty_incident(self):
        for incident in (None, ""):
            with self.subTest(incident=incident):
                with self.assertRaisesRegex(RunPolicyError, "non-empty string"):
                    evaluate_block(state(), incident)

    def test_cli_prints_sorted_json_for_a_valid_terminal_result(self):
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = main(
                [
                    "--state-json",
                    json.dumps(state()),
                    "--result",
                    "accepted",
                ]
            )

        self.assertEqual(exit_code, 0)
        rendered = stdout.getvalue()
        self.assertEqual(list(json.loads(rendered)), sorted(json.loads(rendered)))
        self.assertEqual(json.loads(rendered)["total_rounds"], 1)

    def test_cli_reports_json_and_policy_errors_on_one_stderr_line(self):
        cases = (
            ["--state-json", "not-json", "--result", "accepted"],
            ["--state-json", "[]", "--result", "accepted"],
            [
                "--state-json",
                json.dumps(state(max_rounds=0)),
                "--result",
                "accepted",
            ],
        )
        for argv in cases:
            with self.subTest(argv=argv):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    self.assertEqual(main(argv), 2)
                self.assertTrue(stderr.getvalue().startswith("error: "))
                self.assertEqual(stderr.getvalue().count("\n"), 1)

    def test_cli_rejects_nonstandard_json_constants_and_checkpoint_values(self):
        cases = (
            [
                "--state-json",
                '{"total_rounds":0,"performance_miss_streak":0,'
                '"failed_attempt_streak":0,"last_checkpoint_round":NaN,'
                '"max_rounds":20,"valid_no_improvement_limit":3}',
                "--result",
                "accepted",
            ],
            [
                "--state-json",
                json.dumps(state(last_checkpoint_round=-1)),
                "--result",
                "accepted",
            ],
        )
        for argv in cases:
            with self.subTest(argv=argv):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    self.assertEqual(main(argv), 2)
                self.assertTrue(stderr.getvalue().startswith("error: "))
                self.assertEqual(stderr.getvalue().count("\n"), 1)

    def test_cli_reports_parser_errors_on_one_stderr_line(self):
        state_json = json.dumps(state())
        cases = (
            ["--state-json", state_json],
            [
                "--state-json",
                state_json,
                "--result",
                "accepted",
                "--block-incident",
                "benchmark host unavailable",
            ],
        )
        for argv in cases:
            with self.subTest(argv=argv):
                stderr = io.StringIO()
                with contextlib.redirect_stderr(stderr):
                    self.assertEqual(main(argv), 2)
                self.assertTrue(stderr.getvalue().startswith("error: "))
                self.assertEqual(stderr.getvalue().count("\n"), 1)


if __name__ == "__main__":
    unittest.main()

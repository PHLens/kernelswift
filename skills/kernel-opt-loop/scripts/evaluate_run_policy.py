#!/usr/bin/env python3
"""Evaluate v2 terminal-run policy from a JSON manifest projection."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import sys
from typing import Any


TERMINAL_RESULTS = frozenset(
    {
        "accepted",
        "no-improvement",
        "screened-out",
        "design-rejected",
        "candidate-failed",
        "aborted",
    }
)
FAILED_RESULTS = frozenset({"design-rejected", "candidate-failed", "aborted"})
ATTRIBUTIONS = frozenset({"design-error", "code-error", "lowering-unknown", "evidence-gap", "none"})
FAILED_ATTEMPT_EFFECTS = frozenset({"increment", "unchanged", "reset"})

__all__ = ("RunPolicyError", "evaluate_terminal", "evaluate_block", "main")


class RunPolicyError(ValueError):
    """Raised when a run-policy input is invalid."""


class _RunPolicyArgumentParser(argparse.ArgumentParser):
    """Convert argparse failures to the CLI's stable error contract."""

    def error(self, message: str) -> None:
        raise RunPolicyError(message)


def _state_mapping(state: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(state, Mapping):
        raise RunPolicyError("state must be a mapping")
    return state


def _non_negative_int(state: Mapping[str, Any], name: str) -> int:
    value = state.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RunPolicyError(f"{name} must be a non-negative integer")
    return value


def _last_checkpoint_round(state: Mapping[str, Any]) -> int | None:
    value = state.get("last_checkpoint_round")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RunPolicyError(
            "last_checkpoint_round must be null or a non-negative integer"
        )
    return value


def _reject_nonstandard_constant(constant: str) -> None:
    raise RunPolicyError(f"non-standard JSON constant: {constant}")


def _policy_state(state: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
    state = _state_mapping(state)
    total_rounds = _non_negative_int(state, "total_rounds")
    performance_miss_streak = _non_negative_int(state, "performance_miss_streak")
    failed_attempt_streak = _non_negative_int(state, "failed_attempt_streak")
    max_rounds = _non_negative_int(state, "max_rounds")
    valid_no_improvement_limit = _non_negative_int(
        state, "valid_no_improvement_limit"
    )
    if max_rounds <= 0 or valid_no_improvement_limit <= 0:
        raise RunPolicyError(
            "max_rounds and valid_no_improvement_limit must be positive"
        )
    return (
        total_rounds,
        performance_miss_streak,
        failed_attempt_streak,
        max_rounds,
        valid_no_improvement_limit,
    )


def _apply_failed_attempt_effect(current: int, effect: str) -> int:
    if effect == "increment":
        return current + 1
    if effect == "reset":
        return 0
    return current


def evaluate_terminal(
    state: Mapping[str, Any],
    result: str,
    *,
    target_reached: bool = False,
    user_stop_requested: bool = False,
    attribution: str | None = None,
    failed_attempt_effect: str | None = None,
) -> dict[str, Any]:
    """Return the state transition for one completed terminal round.

    Legacy calls omit both optional attribution arguments and keep the
    ``FAILED_RESULTS`` counter behavior. vNext campaign-terminal calls supply the
    verdict's ``failed_attempt_effect``; finalization verdicts never call this
    interface.
    """

    if not isinstance(result, str) or result not in TERMINAL_RESULTS:
        raise RunPolicyError(f"unknown terminal result: {result}")

    explicit_effect = None
    if attribution is not None or failed_attempt_effect is not None:
        if attribution is None or failed_attempt_effect is None:
            raise RunPolicyError("attribution and failed_attempt_effect must be supplied together")
        if attribution not in ATTRIBUTIONS:
            raise RunPolicyError(f"unknown attribution: {attribution}")
        if failed_attempt_effect not in FAILED_ATTEMPT_EFFECTS:
            raise RunPolicyError(f"unknown failed_attempt_effect: {failed_attempt_effect}")
        explicit_effect = failed_attempt_effect

    (
        total_rounds,
        performance_miss_streak,
        failed_attempt_streak,
        max_rounds,
        miss_limit,
    ) = _policy_state(state)
    total_rounds += 1

    if result == "accepted":
        performance_miss_streak = 0
        failed_attempt_streak = 0
    elif result == "no-improvement":
        performance_miss_streak += 1
    elif explicit_effect is not None:
        failed_attempt_streak = _apply_failed_attempt_effect(failed_attempt_streak, explicit_effect)
    elif result in FAILED_RESULTS:
        failed_attempt_streak += 1

    stop_reason = None
    if user_stop_requested:
        stop_reason = "user-intervention"
    elif target_reached:
        stop_reason = "target-reached"
    elif performance_miss_streak >= miss_limit:
        stop_reason = "valid-no-improvement-limit"
    elif total_rounds >= max_rounds:
        stop_reason = "round-budget-exhausted"

    running = stop_reason is None
    last_checkpoint_round = _last_checkpoint_round(state)
    emit_checkpoint = (
        running
        and total_rounds % 3 == 0
        and last_checkpoint_round != total_rounds
    )
    outcome: dict[str, Any] = {
        "total_rounds": total_rounds,
        "performance_miss_streak": performance_miss_streak,
        "failed_attempt_streak": failed_attempt_streak,
        "workflow_status": "running" if running else "stopped",
        "phase": "ready" if running else "stopped",
        "stop_reason": stop_reason,
        "dispatch_next_round": running,
        "emit_checkpoint": emit_checkpoint,
        "last_checkpoint_round": (
            total_rounds if emit_checkpoint else last_checkpoint_round
        ),
    }
    if explicit_effect is not None:
        outcome["attribution"] = attribution
        outcome["failed_attempt_effect"] = explicit_effect
    return outcome


def evaluate_block(state: Mapping[str, Any], incident: str) -> dict[str, Any]:
    """Return the counter-neutral transition for an environment block."""

    if not isinstance(incident, str) or not incident:
        raise RunPolicyError("incident must be a non-empty string")

    (
        total_rounds,
        performance_miss_streak,
        failed_attempt_streak,
        _max_rounds,
        _miss_limit,
    ) = _policy_state(state)
    return {
        "total_rounds": total_rounds,
        "performance_miss_streak": performance_miss_streak,
        "failed_attempt_streak": failed_attempt_streak,
        "workflow_status": "blocked",
        "phase": "blocked",
        "blocked_incident": incident,
        "dispatch_next_round": False,
        "emit_checkpoint": False,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Evaluate a JSON state projection from command-line arguments."""

    parser = _RunPolicyArgumentParser()
    parser.add_argument("--state-json", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--result", choices=sorted(TERMINAL_RESULTS))
    group.add_argument("--block-incident")
    parser.add_argument("--target-reached", action="store_true")
    parser.add_argument("--user-stop-requested", action="store_true")
    parser.add_argument("--attribution", choices=sorted(ATTRIBUTIONS))
    parser.add_argument("--failed-attempt-effect", choices=sorted(FAILED_ATTEMPT_EFFECTS))

    try:
        args = parser.parse_args(argv)
        state = json.loads(
            args.state_json, parse_constant=_reject_nonstandard_constant
        )
        if not isinstance(state, dict):
            raise RunPolicyError("state JSON must be an object")
        if (args.attribution is None) != (args.failed_attempt_effect is None):
            raise RunPolicyError("--attribution and --failed-attempt-effect must be supplied together")
        outcome = (
            evaluate_block(state, args.block_incident)
            if args.block_incident is not None
            else evaluate_terminal(
                state,
                args.result,
                target_reached=args.target_reached,
                user_stop_requested=args.user_stop_requested,
                attribution=args.attribution,
                failed_attempt_effect=args.failed_attempt_effect,
            )
        )
    except (json.JSONDecodeError, RunPolicyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(json.dumps(outcome, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

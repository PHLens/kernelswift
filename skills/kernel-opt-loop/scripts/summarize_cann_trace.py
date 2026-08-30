#!/usr/bin/env python3
"""Summarize Ascend NPU device-kernel evidence from a CANN msprof profile.

The `torch.profiler` chrome trace on Ascend exposes only host-side `cpu_op`
events; NPU AI Core kernel durations live in the CANN msprof output written by
`torch_npu.profiler`. This helper reads `ai_core_op_summary.db` (table
`task_time` joined with `ge_summary`) and produces a summary compatible with
`summarize_trace.py`, with `device_time_available=true`.

Input: the path to the `*_ascend_pt` directory (or any directory under it
containing `device_0/sqlite/ai_core_op_summary.db`). Output is one JSON object.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import sqlite3
from pathlib import Path
import sys
from typing import Any, Sequence


class CannTraceSummaryError(ValueError):
    """Raised when the CANN profile cannot be summarized unambiguously."""


def _find_db(root: Path) -> Path:
    """Locate the ai_core_op_summary.db under a CANN profile directory."""
    if root.is_file():
        candidate = root
        if candidate.name == "ai_core_op_summary.db":
            return candidate
        raise CannTraceSummaryError(
            f"expected ai_core_op_summary.db, got file {candidate.name!r}"
        )
    matches = list(root.rglob("ai_core_op_summary.db"))
    if not matches:
        raise CannTraceSummaryError(
            f"no ai_core_op_summary.db found under {root}"
        )
    if len(matches) > 1:
        raise CannTraceSummaryError(
            f"multiple ai_core_op_summary.db found under {root}: {matches}"
        )
    return matches[0]


def _query_tasks(
    db: Path,
    time_range: tuple[float, float] | None = None,
) -> list[tuple[str, float]]:
    """Return (op_name, duration_us) rows with positive duration.

    A single logical task can be split into parallel `AI_CORE` and `MIX_AIV`
    subtasks that share a `task_id` and near-identical `start_time`; the task's
    wall duration is the max of its subtasks, not their sum. `duration_time` is
    stored in nanoseconds. When `time_range` is given, only tasks whose
    `start_time` falls inside `[start_ns, end_ns]` are returned; this isolates a
    single profiler scope (reference vs candidate) because the CANN sqlite
    accumulates tasks from all scopes in one capture.
    """
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    except sqlite3.Error as error:
        raise CannTraceSummaryError(f"cannot open {db}: {error}") from error
    try:
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}
        if "task_time" not in tables:
            raise CannTraceSummaryError(f"{db} has no task_time table")
        # ge_summary is optional; fall back to task_type-only names.
        has_ge = "ge_summary" in tables
        where_clause = "t.duration_time > 0"
        params: list[Any] = []
        if time_range is not None:
            start_ns, end_ns = time_range
            where_clause += " AND t.start_time >= ? AND t.start_time <= ?"
            params.extend([int(start_ns), int(end_ns)])
        if has_ge:
            cur.execute(
                f"""
                SELECT t.task_id,
                       COALESCE(g.op_name, t.task_type) AS op_name,
                       t.duration_time
                FROM task_time t
                LEFT JOIN ge_summary g ON g.task_id = t.task_id
                WHERE {where_clause}
                """,
                params,
            )
        else:
            cur.execute(
                f"""
                SELECT task_id, task_type AS op_name, duration_time
                FROM task_time
                WHERE {where_clause.replace('t.', '')}
                """,
                params,
            )
        rows = cur.fetchall()
    except sqlite3.Error as error:
        raise CannTraceSummaryError(f"query failed on {db}: {error}") from error
    finally:
        con.close()
    if not rows:
        raise CannTraceSummaryError(f"{db} has no positive-duration tasks")

    # Deduplicate parallel subtasks: max duration per task_id, then ns -> us.
    per_task: dict[object, tuple[str, float]] = {}
    for task_id, op_name, duration_ns in rows:
        op = str(op_name)
        dur_us = float(duration_ns) / 1000.0
        if task_id not in per_task or dur_us > per_task[task_id][1]:
            per_task[task_id] = (op, dur_us)
    return list(per_task.values())


def _scope_time_ranges(trace: Path, scope: str) -> list[tuple[float, float]]:
    """Extract `[start_ns, end_ns]` spans of a `record_function` scope.

    The torch_npu chrome trace is a bare JSON list of events; the reference and
    candidate `record_function` scopes appear as `X` events whose `name` matches
    `scope`, with `ts`/`dur` in nanoseconds. Multiple spans may exist across
    repeated calls.
    """
    try:
        events = json.loads(trace.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise CannTraceSummaryError(f"cannot read trace {trace}: {error}") from error
    if not isinstance(events, list):
        raise CannTraceSummaryError(f"trace {trace} is not a list of events")
    spans: list[tuple[float, float]] = []
    for event in events:
        if event.get("ph") != "X" or event.get("name") != scope:
            continue
        ts = float(event.get("ts", 0.0))
        dur = float(event.get("dur", 0.0))
        spans.append((ts, ts + dur))
    if not spans:
        raise CannTraceSummaryError(
            f"no scope {scope!r} found in trace {trace}"
        )
    return spans


def summarize_cann(
    root: Path,
    iterations: int,
    scope: str | None,
    wall_ms: float | None,
    trace: Path | None = None,
) -> dict[str, object]:
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise CannTraceSummaryError("iterations must be positive")
    if scope is not None and (not isinstance(scope, str) or not scope):
        raise CannTraceSummaryError("scope must be a non-empty string")
    if trace is not None and scope is None:
        raise CannTraceSummaryError("trace requires a scope")
    if wall_ms is not None:
        if (
            isinstance(wall_ms, bool)
            or not isinstance(wall_ms, (int, float))
            or not math.isfinite(float(wall_ms))
            or wall_ms <= 0
        ):
            raise CannTraceSummaryError("wall_ms must be positive")

    db = _find_db(Path(root))

    # Isolate a single scope's tasks when a trace + scope are supplied. The
    # CANN sqlite accumulates tasks across reference and candidate scopes.
    time_ranges: list[tuple[float, float]] | None = None
    if trace is not None and scope is not None:
        time_ranges = _scope_time_ranges(Path(trace), scope)

    if time_ranges is None:
        tasks = _query_tasks(db)
    else:
        # A scope may repeat (e.g. warmup + measured iterations); collect tasks
        # across all spans. task_id is unique per launch so cross-span
        # duplicates cannot occur.
        tasks = []
        for start_ns, end_ns in time_ranges:
            tasks.extend(_query_tasks(db, (start_ns, end_ns)))

    totals: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"count": 0, "duration": 0.0}
    )
    for op_name, duration in tasks:
        totals[op_name]["count"] += 1
        totals[op_name]["duration"] += duration

    kernel_summaries = [
        {
            "name": name,
            "count_total": int(agg["count"]),
            "count_per_call": int(agg["count"]) / iterations,
            "total_us": float(agg["duration"]),
            "us_per_call": float(agg["duration"]) / iterations,
        }
        for name, agg in totals.items()
    ]
    kernel_summaries.sort(key=lambda item: (-item["total_us"], item["name"]))

    device_total_us = sum(float(agg["duration"]) for agg in totals.values())
    device_us_per_call = device_total_us / iterations
    kernel_count_total = len(tasks)

    summary: dict[str, object] = {
        "scope": scope,
        "iterations": iterations,
        "device_time_available": True,
        "device_total_us": device_total_us,
        "device_us_per_call": device_us_per_call,
        "kernel_count_total": kernel_count_total,
        "kernel_count_per_call": kernel_count_total / iterations,
        "kernels": kernel_summaries,
        "source": str(db),
    }
    if wall_ms is not None:
        summary["device_ratio"] = device_us_per_call / (float(wall_ms) * 1000.0)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize Ascend NPU kernel durations from a CANN msprof profile."
    )
    parser.add_argument("root", type=Path, help="CANN profile dir or ai_core_op_summary.db")
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--scope")
    parser.add_argument("--wall-ms", type=float)
    parser.add_argument(
        "--trace",
        type=Path,
        help="chrome trace (list format) whose record_function scope isolates this scope's tasks by start_time",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = summarize_cann(
            args.root, args.iterations, args.scope, args.wall_ms, args.trace
        )
    except CannTraceSummaryError as error:
        print(f"{args.root}: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

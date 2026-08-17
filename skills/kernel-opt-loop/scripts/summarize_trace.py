#!/usr/bin/env python3
"""Summarize scoped device-kernel evidence from a profiler trace."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence


class TraceSummaryError(ValueError):
    """Raised when profiler evidence cannot be summarized unambiguously."""


def _finite_number(event: dict[str, Any], field: str, description: str) -> float:
    value = event.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TraceSummaryError(f"{description} has invalid {field}")
    number = float(value)
    if not math.isfinite(number):
        raise TraceSummaryError(f"{description} has invalid {field}")
    return number


def _event_interval(
    event: dict[str, Any], description: str
) -> tuple[float, float]:
    start = _finite_number(event, "ts", description)
    duration = _finite_number(event, "dur", description)
    if duration < 0:
        raise TraceSummaryError(f"{description} has negative dur")
    return start, start + duration


def _read_events(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise TraceSummaryError(f"cannot read trace: {error}") from error
    except (UnicodeError, json.JSONDecodeError) as error:
        raise TraceSummaryError(f"invalid trace JSON: {error}") from error

    if not isinstance(payload, dict) or not isinstance(payload.get("traceEvents"), list):
        raise TraceSummaryError("trace JSON must contain a traceEvents array")
    events = payload["traceEvents"]
    if not all(isinstance(event, dict) for event in events):
        raise TraceSummaryError("every traceEvents entry must be an object")
    return events


def _scope_intervals(
    events: list[dict[str, Any]], scope: str
) -> list[tuple[float, float]]:
    intervals = [
        _event_interval(event, f"scope event {scope!r}")
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") != "kernel"
        and event.get("name") == scope
    ]
    if not intervals:
        raise TraceSummaryError(f"scope not found: {scope}")

    intervals.sort()
    for previous, current in zip(intervals, intervals[1:]):
        if current[0] < previous[1]:
            raise TraceSummaryError(f"overlapping scope events: {scope}")
    return intervals


_GCU_RUNTIME_LAUNCHES = frozenset(
    {
        "topsLaunchKernel",
        "topsLaunchCooperativeKernel",
        "topsLaunchKernelEx",
        "topsModuleLaunchKernel",
    }
)


def _event_is_in_scope(
    event: dict[str, Any], intervals: list[tuple[float, float]] | None
) -> bool:
    if intervals is None:
        return True
    start, end = _event_interval(event, f"event {event.get('name')!r}")
    return any(
        start >= scope_start and end <= scope_end
        for scope_start, scope_end in intervals
    )


def _summarize_gcu_runtime(
    events: list[dict[str, Any]],
    intervals: list[tuple[float, float]] | None,
    iterations: int,
    scope: str | None,
    wall_ms: float | None,
) -> dict[str, object] | None:
    launches: list[dict[str, Any]] = []
    for event in events:
        if (
            event.get("cat") == "gcu_runtime"
            and event.get("ph") == "X"
            and event.get("name") in _GCU_RUNTIME_LAUNCHES
            and _event_is_in_scope(event, intervals)
        ):
            start, end = _event_interval(event, f"GCU runtime event {event.get('name')!r}")
            launches.append({"name": event["name"], "duration": end - start})

    if not launches:
        return None

    totals: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"count": 0, "duration": 0.0}
    )
    for launch in launches:
        aggregate = totals[launch["name"]]
        aggregate["count"] += 1
        aggregate["duration"] += launch["duration"]

    launch_summaries = [
        {
            "name": name,
            "count_total": aggregate["count"],
            "count_per_call": aggregate["count"] / iterations,
            "total_us": aggregate["duration"],
            "us_per_call": aggregate["duration"] / iterations,
        }
        for name, aggregate in totals.items()
    ]
    launch_summaries.sort(key=lambda item: (-item["total_us"], item["name"]))
    runtime_total_us = sum(launch["duration"] for launch in launches)
    runtime_us_per_call = runtime_total_us / iterations
    summary: dict[str, object] = {
        "scope": scope,
        "iterations": iterations,
        "device_time_available": False,
        "device_time_reason": (
            "GCU trace exposes runtime launch events but no cat=kernel device durations"
        ),
        "device_total_us": None,
        "device_us_per_call": None,
        "kernel_count_total": None,
        "kernel_count_per_call": None,
        "kernels": [],
        "runtime_launch_total_us": runtime_total_us,
        "runtime_launch_us_per_call": runtime_us_per_call,
        "runtime_launch_count_total": len(launches),
        "runtime_launch_count_per_call": len(launches) / iterations,
        "runtime_launches": launch_summaries,
    }
    if wall_ms is not None:
        summary["runtime_launch_ratio"] = runtime_us_per_call / (float(wall_ms) * 1000.0)
    return summary


def summarize_trace(
    path: Path,
    iterations: int,
    scope: str | None,
    wall_ms: float | None,
) -> dict[str, object]:
    """Return per-call device totals and kernel breakdown for one scope."""
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise TraceSummaryError("iterations must be positive")
    if scope is not None and (not isinstance(scope, str) or not scope):
        raise TraceSummaryError("scope must be a non-empty string")
    if wall_ms is not None:
        if (
            isinstance(wall_ms, bool)
            or not isinstance(wall_ms, (int, float))
            or not math.isfinite(float(wall_ms))
            or wall_ms <= 0
        ):
            raise TraceSummaryError("wall_ms must be positive")

    path = Path(path)
    events = _read_events(path)
    intervals = _scope_intervals(events, scope) if scope is not None else None

    kernels: list[dict[str, Any]] = []
    for event in events:
        if event.get("cat") != "kernel":
            continue
        name = event.get("name")
        if not isinstance(name, str) or not name:
            raise TraceSummaryError("kernel event has invalid name")
        start, end = _event_interval(event, f"kernel event {name!r}")
        if intervals is None or any(
            start >= scope_start and end <= scope_end
            for scope_start, scope_end in intervals
        ):
            kernels.append({"name": name, "duration": end - start})

    if not kernels:
        runtime_summary = _summarize_gcu_runtime(
            events, intervals, iterations, scope, wall_ms
        )
        if runtime_summary is not None:
            return runtime_summary
        if scope is None:
            raise TraceSummaryError("trace has no kernel events")
        raise TraceSummaryError(f"scope has no kernel events: {scope}")

    totals: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {"count": 0, "duration": 0.0}
    )
    for kernel in kernels:
        aggregate = totals[kernel["name"]]
        aggregate["count"] += 1
        aggregate["duration"] += kernel["duration"]

    kernel_summaries = [
        {
            "name": name,
            "count_total": aggregate["count"],
            "count_per_call": aggregate["count"] / iterations,
            "total_us": aggregate["duration"],
            "us_per_call": aggregate["duration"] / iterations,
        }
        for name, aggregate in totals.items()
    ]
    kernel_summaries.sort(key=lambda item: (-item["total_us"], item["name"]))

    device_total_us = sum(kernel["duration"] for kernel in kernels)
    device_us_per_call = device_total_us / iterations
    summary: dict[str, object] = {
        "scope": scope,
        "iterations": iterations,
        "device_time_available": True,
        "device_total_us": device_total_us,
        "device_us_per_call": device_us_per_call,
        "kernel_count_total": len(kernels),
        "kernel_count_per_call": len(kernels) / iterations,
        "kernels": kernel_summaries,
    }
    if wall_ms is not None:
        summary["device_ratio"] = device_us_per_call / (float(wall_ms) * 1000.0)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Summarize profiler kernel events, normalized per forward call."
    )
    parser.add_argument("trace", type=Path)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--scope")
    parser.add_argument("--wall-ms", type=float)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = summarize_trace(
            args.trace, args.iterations, args.scope, args.wall_ms
        )
    except TraceSummaryError as error:
        print(f"{args.trace}: error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

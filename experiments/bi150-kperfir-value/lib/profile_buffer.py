"""Decode the small fixed BI150 profile buffer."""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

WORDS_PER_SLOT = 3


def unsigned_cycle_delta(start: int, end: int, bits: int = 64) -> int:
    """Return ``end - start`` in an unsigned wrapping counter domain."""
    if bits <= 0:
        raise ValueError("bits must be positive")
    mask = (1 << bits) - 1
    return (int(end) - int(start)) & mask


def selected_local_warps_for_num_warps(num_warps: int) -> tuple[int, ...]:
    """Return the local warp indices present in a Triton launch."""
    if num_warps <= 0:
        raise ValueError("num_warps must be positive")
    return tuple(range(num_warps))


def _nearest_rank(values: list[int], fraction: float) -> int:
    rank = max(1, math.ceil(fraction * len(values)))
    return values[rank - 1]


def summarize_cycles(samples: Sequence[int]) -> dict[str, int | float]:
    """Summarize raw cycle samples without converting them to time."""
    if not samples:
        raise ValueError("samples must not be empty")
    ordered = sorted(int(value) for value in samples)
    mean = statistics.fmean(ordered)
    cv = statistics.pstdev(ordered) / mean if mean else 0.0
    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "maximum": ordered[-1],
        "median": statistics.median(ordered),
        "p10": _nearest_rank(ordered, 0.10),
        "p90": _nearest_rank(ordered, 0.90),
        "coefficient_of_variation": cv,
    }


def decode_profile_buffer(
    values: Sequence[int],
    selected_pids: tuple[int, ...],
    selected_local_warps: tuple[int, ...],
    generation: int,
    *,
    counter_bits: int = 64,
) -> list[dict[str, int | str]]:
    """Decode ``[generation, start, end]`` slots in PID/warp order."""
    expected_words = len(selected_pids) * len(selected_local_warps) * WORDS_PER_SLOT
    if len(values) != expected_words:
        raise ValueError(
            f"profile buffer has {len(values)} words; expected {expected_words}"
        )

    rows: list[dict[str, int | str]] = []
    offset = 0
    for pid in selected_pids:
        for local_warp in selected_local_warps:
            observed_generation, start, end = (
                int(values[offset]),
                int(values[offset + 1]),
                int(values[offset + 2]),
            )
            offset += WORDS_PER_SLOT
            row: dict[str, int | str] = {
                "pid": pid,
                "local_warp": local_warp,
                "start_boundary": "region-start",
                "end_boundary": "region-end",
            }
            if observed_generation != generation:
                row.update(status="unavailable", cause="generation-mismatch")
            else:
                row.update(
                    status="observed",
                    cause="none",
                    raw_cycle_start=start,
                    raw_cycle_end=end,
                    raw_cycle_delta=unsigned_cycle_delta(start, end, counter_bits),
                )
            rows.append(row)
    return rows

# Decision 002

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"002","reference_implementation":"triton_rotary_001.py","reference_report":"rounds/report_001.md","language":"triton","backend":"maca","target_profile":"triton_maca","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-change"}
```

## Optimization Intent

```json
{"bottleneck_class":"host-bound","intervention":"no candidate-owned intervention clears the 5% wall threshold; the residual ~63 us of non-device wall time is dominated by harness-fixed set_seed + full sync_devices() inside the timed loop, which the candidate cannot optimize","allowed_changes":[],"invariants":["ModelNew public contract","output tuple shape dtype device","numerical semantics","non-mutation of timestamps","caller-selected device and current stream","PyTorch fallback for non-benchmark shapes"],"expected_wall_improvement_pct":0.0}
```

## Unified Sketch

N/A: aborted

## Host Plan

```json
{"applicability":"not-applicable","reason":"aborted"}
```

## Evaluation Contract

```json
{"applicability":"not-applicable","reason":"aborted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`. The four archived entries (winner-tree
  argmax, sort-32/sort-64 selection networks, dynamic `tl.gather` compaction,
  cumsum compaction) all concern reduction/selection dataflows on the MLU runtime
  and do not apply to this flat elementwise kernel. The Round 002-relevant lesson
  is not in this file but in the groupedtopk C500 archive: after fusion reached a
  low device ratio, three candidate-owned directions were each falsified with a
  `no-improvement` result — allocation coalescing `-13.71%` (Round 002), combined
  reduction `+0.05%` (Round 003), and dispatch specialization `+2.69%` (Round 004)
  — reaching `valid-no-improvement-limit`. That archive is the direct evidence
  that the remaining host-bound tail on this C500 runtime is not candidate-owned.
- Consulted `references/bottleneck-judgment.md`. device_ratio 0.211 places this in
  the `mixed` band (20%–80%), but the primary ratio says only *where* time is
  observed, not *whether it is compressible by the candidate*. The Level 2
  discipline requires separating candidate-owned host work (1 direct launch + 2
  `torch.empty` allocations) from harness-fixed work (`set_seed`, clone, recursive
  `compare_values`, and the full `sync_devices()` inside the timed loop). The
  harness-fixed work is un-optimizable by definition of the measurement regime.

## Rationale and Evidence

`report_001.md` establishes the accepted canonical: wall median 0.080036 ms
(80.036 us/call), device 16.90 us/call from a single `_rotary_embed_fused_kernel`,
device_ratio 0.211. That leaves ~63 us/call of non-device wall time.

The authoritative harness (`auto_bench.py` `time_forward`, the immutable
measurement regime) times each sample as:

```text
set_seed(seed)             # torch.manual_seed + manual_seed_all — harness-fixed
start = perf_counter()
model.forward(*inputs)     # candidate-owned: 1 launch + 2 torch.empty + 16.9us device
sync_devices()             # full torch.cuda.synchronize() — harness-fixed, inside timer
stop = perf_counter()
```

Both `set_seed(seed)` and `sync_devices()` (a complete device synchronize) are
inside the timed interval and are harness-owned. The candidate cannot remove or
shrink them without violating the immutable measurement fingerprint. The candidate
already executes exactly one Triton launch and two `torch.empty` allocations for
two 16 KB fp32 buffers; the remaining host work is the launch dispatch plus those
two allocator hits.

Evaluated candidate-owned levers against the 4.0 us (5%) threshold, none is
defensible:

1. **Output buffer coalescing** (2 × `torch.empty` → 1 alloc + 2 views): the
   groupedtopk C500 archive already falsified this exact family at `-13.71%`
   (`decision_002.md`, `final_summary.md`), even with same-dtype fp32 outputs
   there being a simpler case than the int32/fp32 reinterpretation. Here both
   outputs are identical-dtype fp32 `[4,32,128]`, so a shared flat backing is
   expressible without dtype reinterpretation, but (a) the two `torch.empty` calls
   are CUDA-caching-allocator host hits of ~16 KB each, whose inclusive CPU cost is
   on the order of single-digit us and not separately observed in `report_001.md`
   (no `aten::empty` evidence was requested or captured); (b) a fresh shared
   backing introduces aliasing/lifetime semantics that require a full Host Plan
   and carry regression risk on this runtime. No Verifier-backed observation
   supports an `aten::empty` inclusive cost of >=4 us, so the intervention is
   speculative and the archived evidence says it regresses.

2. **Single-kernel internal optimization** (device 16.90 us — vectorization /
   memory access / grid): device time is not the binding constraint at
   device_ratio 0.211. The kernel already overlaps asynchronously with host work,
   so shrinking device time does not reduce wall time while the host side is the
   bottleneck. To clear 4.0 us of wall it would need to cut device time by ~24% of
   16.9 us, and even a full elimination cannot address the ~63 us host-fixed tail.

3. **num_warps / grid layout**: `num_warps=1` is the only `Constrained`-safe value
   (target warp_size=64); all other values are `Unknown` in the target profile and
   would be capability-miss, not tuning. The kernel is a 16384-element elementwise
   map (grid=(16,), BLOCK=1024); warp-count tuning targets device time, which is
   not the wall lever.

None of the three candidate-owned directions has a falsifiable >=5% wall path, and
the groupedtopk C500 archive already shows the host-bound tail on this exact
runtime is not candidate-owned. Manufacturing a hypothesis here would predictably
produce another `no-improvement`; the honest, Level-2-disciplined outcome is to
stop. The accepted canonical `triton_rotary_001.py` retains the 55.57% fusion win.

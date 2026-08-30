# Decision 002

## Metadata

```json
{"schema_version":1,"decision":"abort","round":"002","reference_implementation":"triton_mhcc_001.py","reference_report":"rounds/report_001.md","language":"triton","backend":"maca","target_profile":"triton_maca","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"none","change_family":"no-change"}
```

## Optimization Intent

```json
{"bottleneck_class":"none","intervention":"no stable intervention clears the 5% adoption threshold: the remaining wall time is single-launch device latency plus harness-fixed host cost, neither of which the candidate can reduce without violating semantics or the measurement regime","allowed_changes":[],"invariants":["ModelNew public contract","output tuple shape/dtype/device","exact fp32 Sinkhorn semantics","input non-mutation","caller-selected device and current stream preserved"],"expected_wall_improvement_pct":0.0}
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

- Consulted `references/bottleneck-judgment.md`. The device ratio after fusion is
  ~0.37, which the heuristic would label "mixed", but the ratio says where time is
  observed, not whether it is compressible. Level 2 discipline decomposes the
  remaining time: device is 43.791 us for exactly ONE fused kernel (grid=16,
  num_warps=1), and host is ~74 us. The candidate-owned host portion (one direct
  launch + three `torch.empty` allocations of tiny `[16,4]`, `[16,4]`, `[16,4,4]`
  tensors) is already minimal and is a single-launch fixed cost, not multi-launch
  accumulation. The host remainder is harness-fixed (`torch.manual_seed` in
  `get_inputs` plus `torch.cuda.synchronize` in the measurement regime), which is
  part of the measurement fingerprint and cannot be optimized by the candidate.
- Consulted `references/anti-patterns.md`. No catalog entry's preconditions match
  the current situation; the fusion already eliminated every reducible library
  kernel. There is no dynamic gather/cumsum/sort-network lowering to regress.
- Grid 16 -> 1 merge was evaluated and rejected: launch overhead is per-kernel, not
  per-program, so one kernel is already the minimum launch count; serializing the
  16 independent (b,s) positions into one program would lengthen the 20-iteration
  Sinkhorn dependency chain and likely increase, not decrease, device time.

## Rationale and Evidence

Round 001 accepted `triton_mhcc_001.py` with a 92.9% wall improvement (1.665487 ms
-> 0.118357 ms) by fusing the entire forward into one Triton kernel. The
Evaluation Contract observables all confirmed: kernel count 133 -> 1,
`device_us_per_call` 534.014 -> 43.791 us, and all sum/div library kernels
eliminated. The host-bound multi-launch bottleneck is fully resolved.

The remaining wall is now latency-bound, not throughput-bound. Device time is
43.791 us for a single fused kernel that launches 16 programs, each handling one
(b,s) position with a 20-iteration alternating Sinkhorn normalization. The
Sinkhorn loop is a serial dependency chain: each column-sum depends on the prior
row-normalize, so the 20 iterations cannot be parallelized without changing the
exact fp32 semantics. This is an algorithmic latency floor, not compressible
work. Merging the 16 programs into 1 would serialize 16x more per-program work
into the same single launch and cannot reduce launch count (already 1); it would
at best leave device time unchanged and more likely increase it.

The ~63% host remainder (~74 us) decomposes into candidate-owned (one direct
launch + three tiny `torch.empty` allocations) and harness-fixed (`set_seed` +
`sync_devices`). The candidate-owned portion is already a single-launch fixed
cost and a handful of tiny allocations; caching output buffers would be a host
change requiring an explicit Host Plan for negligible wall gain while adding
aliasing and concurrency risk. The harness-fixed portion is part of the
measurement fingerprint and is out of scope for the candidate.

The 5% adoption threshold is 5.92 us. No falsifiable intervention offers a
causal chain of at least that magnitude: device time (43.8 us) is at its
single-launch latency floor for the required 20-step serial Sinkhorn semantics,
and host time is dominated by harness-fixed seed and synchronization costs. The
groupedtopk and rotary campaigns reached the same measurement-bound conclusion
after fusion. The evidence does not justify another stable improvement attempt,
so this round records an abort.

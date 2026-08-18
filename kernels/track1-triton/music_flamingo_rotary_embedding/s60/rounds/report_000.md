# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `base.py`
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `9fc87abbe0e6268f06c969e94f5400abea51cdf315276a4ac5cef5bd0ad8a26f`
- Accepted reference SHA256: `99829754f4cdc4bfd2808e051de549f0791414241e7fdbad7a1b8294a15be475`
- Base SHA256: `99829754f4cdc4bfd2808e051de549f0791414241e7fdbad7a1b8294a15be475`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `a1ee09ca54ab2210943bd030a6649c57d96b09d4c1beed863f4a98681ae425f2`
- verification_tier: `baseline`
- screening_pairs: `not-applicable: Phase 0`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | pass | `base.py` and `baseline_adapter.py` tuple outputs `(cos, sin)` matched at `atol=1e-2, rtol=1e-2`; shapes `[4,32,128]` fp32 | pass | S60 auto_bench smoke and formal benchmark command |
| immutable base | base bytes unchanged by adapter generation | base hash recorded as `99829754...` | pass | `make_baseline_adapter.py` and SHA-256 ledger |
| GCU runtime | selected profile matches runtime | `torch_gcu` and `triton_gcu` imported; `gcu:0` available; arch `major=3, minor=0` | pass | S60 runtime discovery and baseline harness run |

## Screening Evidence

Not applicable to Phase 0.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `base.py`, then `baseline_adapter.py` in the unchanged harness
- reference_raw_samples_ms: `[0.465926]`
- candidate_raw_samples_ms: `[0.464657]`
- reference_median_ms: `0.465926`
- candidate_median_ms: `0.464657`
- improvement_pct: `not-applicable: baseline adapter is the executable canonical baseline`

## Evaluation Contract Mirror

- Evaluation Contract applicability: `not-applicable: Phase 0`
- hypothesis_id: `not-applicable: Phase 0`
- intervention: `not-applicable: Phase 0`
- expected_causal_chain: `not-applicable: Phase 0`
- primary_metric: `not-applicable: Phase 0`
- Hypothesis verdict: `not-applicable: Phase 0`

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `summary`
- profiler_device_time: `unavailable: recorded GCU PrivateUse1 trace has no cat=kernel device-duration events`
- iterations: `50`
- normalized_fields: `device_*` and `kernels` are `null/unavailable`; `runtime_launch_*` fields available
- trace: `log/rotary_baseline_forward_50iter.pt.trace.json`
- trace_sha256: `2c5e3cede88635da18961f23cb96bd5a5f8d58cb783d9ce1349d4d39cf0f87c7`

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio | Runtime launches/call | Runtime launch us/call |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_base | unavailable | unavailable | unavailable | unavailable | 0.465926 | unavailable | 13.0 | 139.647852 |
| candidate_baseline_adapter | unavailable | unavailable | unavailable | unavailable | 0.464657 | unavailable | 13.0 | 139.647852 |

GCU trace evidence is diagnostic: the eager reference issues 13 runtime launches
per forward call (all `topsLaunchKernel`), from the pure elementwise/view ops
(arange, mul/div, repeat_interleave, broadcast, cat, cos, sin). Launch overhead
is ~139.6 us/call ≈ 30% of wall — the largest launch ratio seen among the
track1-triton s60 operators so far, indicating a clear elementwise-fusion headroom.

## Retry History

No retries; Phase 0 baseline gate passed.

## evidence_for_next_round

- The eager reference issues 13 GCU runtime launches per forward call, all from
  elementwise/view ops; this is a pure elementwise workload (no GEMM, no
  data-axis reduction), so a single fused Triton elementwise kernel is the
  natural fusion target and does not carry the segment-reduction device penalty
  that sank sparse_pooler's fusion (its kernel had a serial `range(seq_len)` loop).
- Launch overhead is ~139.6 us/call ≈ 30% of wall (0.465926 ms), so collapsing
  13 launches toward 1 has a clear mechanism for a >=5% wall win.
- Output is a tuple `(cos, sin)` of two `[4,32,128]` fp32 tensors; the kernel can
  write cos/sin to two output buffers in one launch.
- `seq_len` is a Python int (not a tensor) passed straight through `clone_value`;
  the kernel must treat it as a compile-time or runtime scalar, not a device tensor.
- GCU profiler does not provide device durations; `runtime_launch_count_per_call`
  and `runtime_launch_us_per_call` are the observable mechanism fields.

## Stop Recommendation

- recommendation: `continue`
- evidence: baseline established; 13-launch elementwise fusion headroom is clear

## Exact Reproduction Commands

```bash
cd /root/kernelswift/.worktrees/rotary-embedding-s60
python3 auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/s60/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback
```

```bash
cd /root/kernelswift/.worktrees/rotary-embedding-s60
python3 auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/s60/baseline_adapter.py --warmup 50 --repeat 100
```

```bash
cd /root/kernelswift/.worktrees/rotary-embedding-s60
python3 auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/s60/baseline_adapter.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/music_flamingo_rotary_embedding/s60/log/rotary_baseline_forward_50iter.pt.trace.json
```

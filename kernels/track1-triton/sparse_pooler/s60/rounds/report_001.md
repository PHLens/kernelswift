# Report 001

Result: no-improvement

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md`
- Candidate: `kernels/track1-triton/sparse_pooler/s60/triton_sparse_pooler_001.py`
- Accepted reference: `kernels/track1-triton/sparse_pooler/s60/baseline_adapter.py`
- Accepted reference report: `rounds/report_000.md`
- Decision SHA256: `a625d1d56257c797833f3226acc8b07474b748bf8d343a0da99d64017c3cede8`
- Candidate SHA256: `60aadce88e02776b71960092bf9df59c0adcdc8ace75319e845f7c9122a3f80e`
- Accepted reference SHA256: `359f4c808a0cf210416116322e4cc01f74ee42961b68c1fd365672af2a59bde8`
- Base SHA256: `46106baa46b9f1952b32388d99880c967438ea99becdc874e3f68ba81a727d58`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `15ffdaf1e8fcc0a9b8b5af2a429e4ddad7c4e3ac67b345a9600d6cb8aa6bd226`
- verification_tier: `authoritative`
- screening_pairs: `not-run`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | pass | `base.py` and `triton_sparse_pooler_001.py` list outputs matched (4 x `[30522]` fp32) at `atol=1e-2, rtol=1e-2` | pass | `PASS accuracy; v0=0.870069 ms, v1=1.087015 ms, speedup=0.800x` (warmup 5 / repeat 10 smoke) |
| output is a Python list of num_seq tensors each [vocab_size] fp32 | list of 4 x `[30522]` fp32 | `forward` returns `[out[i] for i in range(num_seq)]`; harness `compare_values` list-recursion passed | pass | correctness PASS (type/shape check inside `compare_values`) |
| numerical semantics log(1+relu(decoder_logits)) max-pooled per sequence | max reduction over sequence axis | kernel accumulates `tl.where(acc < x, x, acc)` over `range(seq_len)` after `relu`+`log(1+x)`; matched base at tolerance | pass | correctness PASS at atol/rtol 1e-2 |
| caller-selected device and current stream preserved | same device / stream | no explicit device context introduced; output allocated on `x.device` | pass | correctness PASS on `gcu:0` |
| dense GELU LayerNorm decoder matmul pipeline unchanged | library ops unchanged | `dense`/`act`/`layer_norm`/`decoder` remain `nn.*` library ops in `__init__` | pass | candidate source lines 64-68 |
| ModelNew public constructor and forward signature unchanged | signature preserved | `__init__(hidden_size=768, vocab_size=30522, pooling="max")`; `forward(hidden_states, seq_lens)` | pass | candidate source lines 62-70 |
| load_state_dict compatibility maintained | state_dict keys unchanged | submodule names `dense`/`act`/`layer_norm`/`decoder` unchanged | pass | correctness PASS (no silent weight-sync skip) |

## Screening Evidence

Not run. Correctness passed, so the candidate proceeded directly to authoritative timing (a
correct candidate is only screened-out when two short pairs are both at least 10% slower;
screening is optional here because the formal benchmark already shows the candidate is
slower, but per contract the authoritative timing below is the controlling evidence).

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate` (harness-run)
- reference_raw_samples_ms: `[0.861388]` (canonical baseline median from report_000; re-measured v0=0.866665)
- candidate_raw_samples_ms: `[1.092186]`
- reference_median_ms: `0.861388`
- candidate_median_ms: `1.092186`
- improvement_pct: `-26.793733`

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
               = (0.861388 - 1.092186) / 0.861388 * 100
               = -26.793733
```

The candidate is 26.79% slower than the accepted reference. The unrounded improvement is
negative, far below the +5.0% adoption threshold, so this is classified `no-improvement`.

Authoritative benchmark output (step 2): `PASS accuracy; v0=0.866665 ms, v1=1.092186 ms, speedup=0.794x`.
The harness reports a single median per side (v0 = accepted reference base, v1 = candidate).
The canonical reference median `0.861388` from report_000 is used for the improvement
computation; the step-2 re-measure of v0 (`0.866665`) is consistent within run-to-run noise.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| runtime_launch_count_per_call | decrease from 11 toward approximately 6 | 11.0 → 6.0 (550 total → 300 total over 50 iterations); 5x `topsLaunchKernel` + 1x `topsModuleLaunchKernel` | pass | `summarize_trace.py` candidate scope `candidate_triton_sparse_pooler_001` |
| host_sync_count_per_call | decrease because the seq_lens.tolist() D2H sync is eliminated | baseline: `topsStreamSynchronize` 1.0/call + `topsMemcpyAsync` 1.0/call + `aten::copy_` 1.0/call + `GCU::_copy_from` 1.0/call; candidate: 0 of all sync/copy events | pass | trace event count (in-scope gcu_runtime/cpu_op sync+copy names) |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-001`
- intervention: `fuse relu, log1p, and per-sequence max pooling into a single Triton kernel launched once per forward, eliminating the Python for-loop over seq_lens.tolist() and the D2H synchronization it triggers, and replacing six device kernels with one fused reduction kernel`
- expected_causal_chain: `[python for-loop + 4x torch.max -> 1 fused kernel launch; seq_lens.tolist() D2H sync eliminated; runtime launch count 11 -> ~6; host dispatch/sync overhead decrease; wall time decreases by at least 5%]`
- primary_metric: `wall_time`
- Hypothesis verdict: `falsified`

Both mechanism observables are confirmed (launch count 11→6; host sync eliminated).
However the final causal link — "wall time decreases by at least 5%" — is falsified: wall
time increased by 26.79%. The fused kernel's per-element work (a serialized `range(seq_len)`
loop over the sequence axis with `num_warps=1` and a small `BLOCK_V=256`, i.e. 120 vocab
tiles per program) is substantially slower than the library elementwise relu/log1p plus the
4x `chunk.max(dim=0)` reductions it replaced. The launch/sync savings (~111 us/call →
~62 us/call runtime launch time) are dwarfed by the added device-side compute.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `unavailable: recorded GCU trace has no cat=kernel device-duration events`
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels` are `null/unavailable`; `runtime_launch_*` fields available
- backend_runtime_fields: `runtime_launch_total_us`, `runtime_launch_us_per_call`, `runtime_launch_count_total`, `runtime_launch_count_per_call`, `runtime_launches`

Reference and candidate scopes are summarized independently. All totals are normalized by
`iterations = 50`. Device time is unavailable on this GCU exporter; runtime launch evidence
is retained and is NOT relabeled as device kernel time.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference (`candidate_baseline_adapter`) | unavailable | unavailable | unavailable | unavailable | 0.861388 | unavailable |
| candidate (`candidate_triton_sparse_pooler_001`) | unavailable | unavailable | unavailable | unavailable | 1.092186 | unavailable |

```text
device_ratio = unavailable (no cat=kernel device durations on GCU exporter)
```

Runtime launch comparison (per call):

| Scope | runtime launch count/call | runtime launch us/call | runtime launch ratio |
|---|---:|---:|---:|
| accepted_reference | 11.0 | 111.306377 | 0.129217 |
| candidate | 6.0 | 62.484932 | 0.057129 |

### Accepted Reference Runtime Launches

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| topsLaunchKernel | 550 | 11.0 | 5565.318848 | 111.306377 |

### Candidate Runtime Launches

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| topsLaunchKernel | 250 | 5.0 | 2585.106445 | 51.702129 |
| topsModuleLaunchKernel | 50 | 1.0 | 539.140137 | 10.782803 |

The runtime launch overhead dropped from ~111 us/call to ~62 us/call (a ~44% reduction in
launch time), and the D2H sync (`topsStreamSynchronize`/`topsMemcpyAsync`/`aten::copy_`) is
fully eliminated. Yet wall time increased, proving the fused kernel's added device-side
compute dominates the launch/sync savings.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial verification | not-applicable | `60aadce88e02776b71960092bf9df59c0adcdc8ace75319e845f7c9122a3f80e` | correct but 26.79% slower → `no-improvement` |

At most one Verifier-to-Coder repair is allowed per round; no repair was triggered because
correctness passed and the failure is a performance (not implementation) defect.

## evidence_for_next_round

- The structural fusion mechanism is confirmed at the launch level: `runtime_launch_count_per_call` fell from 11 to 6 and the `seq_lens.tolist()` D2H sync (`topsStreamSynchronize` + `topsMemcpyAsync` + `aten::copy_`) was fully eliminated (0/call vs 1/call each).
- The launch/sync savings are real but small (~49 us/call of runtime launch time, from ~111 us to ~62 us), and are entirely offset by added device-side compute.
- The fused kernel is the bottleneck now: it serializes `range(seq_len)` over the sequence axis with `num_warps=1` and uses a small `BLOCK_V=256` (120 vocab tiles, last tile only 58 live lanes), producing a slow per-element reduction relative to the library relu/log1p + 4x `chunk.max(dim=0)` it replaced.
- Candidate wall `1.092186 ms` vs reference `0.861388 ms` — the candidate is ~27% slower; the fused Triton reduction is not competitive with the PyTorch library ops on GCU for this small workload (num_seq=4, seq_len ≤ 25).
- GCU device time is unavailable (no `cat=kernel` events), so attribution rests on runtime-launch counts and wall time; a device-side duration probe would strengthen next-round attribution.

## Stop Recommendation

- recommendation: `continue`
- evidence: `Round 001 is the first optimization round (no-improvement limit not reached). The structural fusion mechanism (launch-count and D2H-sync reduction) is confirmed but the fused kernel is slower than the library baseline; the kernel-fusion direction needs revision before the 5% wall-time threshold can be met.`

Orchestrator owns the stop transition.

## Exact Reproduction Commands

```bash
cd /root/kernelswift/.worktrees/sparse-pooler-s60 && python3 auto_bench.py --v0_file kernels/track1-triton/sparse_pooler/base.py --v1_file kernels/track1-triton/sparse_pooler/s60/triton_sparse_pooler_001.py --warmup 5 --repeat 10 --full-traceback
```

```bash
cd /root/kernelswift/.worktrees/sparse-pooler-s60 && python3 auto_bench.py --v0_file kernels/track1-triton/sparse_pooler/base.py --v1_file kernels/track1-triton/sparse_pooler/s60/triton_sparse_pooler_001.py --warmup 50 --repeat 100
```

```bash
cd /root/kernelswift/.worktrees/sparse-pooler-s60 && python3 auto_bench.py --v0_file kernels/track1-triton/sparse_pooler/base.py --v1_file kernels/track1-triton/sparse_pooler/s60/triton_sparse_pooler_001.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/sparse_pooler/s60/log/sparse_pooler_round_001_forward_50iter.pt.trace.json
```

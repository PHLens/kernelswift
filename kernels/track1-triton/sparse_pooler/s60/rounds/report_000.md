# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `base.py`
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `359f4c808a0cf210416116322e4cc01f74ee42961b68c1fd365672af2a59bde8`
- Accepted reference SHA256: `46106baa46b9f1952b32388d99880c967438ea99becdc874e3f68ba81a727d58`
- Base SHA256: `46106baa46b9f1952b32388d99880c967438ea99becdc874e3f68ba81a727d58`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `15ffdaf1e8fcc0a9b8b5af2a429e4ddad7c4e3ac67b345a9600d6cb8aa6bd226`
- verification_tier: `baseline`
- screening_pairs: `not-applicable: Phase 0`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | pass | `base.py` and `baseline_adapter.py` list outputs matched (4 x `[30522]` fp32) at `atol=1e-2, rtol=1e-2` | pass | S60 auto_bench smoke and formal benchmark command |
| immutable base | base bytes unchanged by adapter generation | base hash recorded as `46106baa...` before and after adapter generation | pass | `make_baseline_adapter.py` and SHA-256 ledger |
| GCU runtime | selected profile matches runtime | `torch_gcu` and `triton_gcu` imported; `gcu:0` available; architecture `major=3, minor=0` | pass | S60 runtime discovery commands and baseline harness run |

## Screening Evidence

Not applicable to Phase 0.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `base.py`, then `baseline_adapter.py` in the unchanged harness
- reference_raw_samples_ms: `[0.862541]`
- candidate_raw_samples_ms: `[0.861388]`
- reference_median_ms: `0.862541`
- candidate_median_ms: `0.861388`
- improvement_pct: `not-applicable: baseline adapter is the executable canonical baseline`

The baseline adapter is retained as canonical because it is generated from the
immutable base, not because its small timing difference is an optimization claim.

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
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels` are `null/unavailable`; `runtime_launch_*` fields are available
- trace: `log/sparse_pooler_baseline_forward_50iter.pt.trace.json`
- trace_sha256: `5176fc8786a07bc4ac8bca1505eeb55ff560365f5b6f11dd9e74916895269c75`

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio | Runtime launches/call | Runtime launch us/call |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_base | unavailable | unavailable | unavailable | unavailable | 0.862541 | unavailable | 11.0 | 111.009502 |
| candidate_baseline_adapter | unavailable | unavailable | unavailable | unavailable | 0.861388 | unavailable | 11.0 | 111.009502 |

GCU trace evidence is diagnostic: the eager reference issues 11 runtime launches
per forward call (all `topsLaunchKernel`), driven by the MLM head library kernels
(dense GEMM, GELU, LayerNorm, decoder GEMM), the elementwise relu/log1p, and 4
per-sequence `chunk.max(dim=0)` launches plus the `seq_lens.tolist()` host-side
D2H sync loop. Runtime launch duration is not device kernel duration and is not
converted to `device_ratio`.

## Retry History

No retries; Phase 0 baseline gate passed.

## evidence_for_next_round

- The eager reference issues 11 GCU runtime launches per forward call; the 4
  per-sequence `chunk.max(dim=0)` launches and the `seq_lens.tolist()` D2H sync
  loop are the structural fusion targets (matching MLU round 001's premise).
- MLU's 1.60x path fused only relu/log1p/max-pooling + a device-side prefix scan
  (eliminating the D2H sync), and kept the MLM head GEMMs as library ops (MLU
  round 003 proved fusing the decoder matmul into `tl.dot` is a 33% regression).
- GCU profiler export does not provide `cat=kernel` device durations; runtime-launch
  evidence is the available normalized backend diagnostic.
- GELU must remain a library op to match base's device-level behavior (GCU may
  approximate `nn.GELU()` to tanh); do not hand-write erf GELU in the kernel.
- Output must stay a Python `list` of 4 `[30522]` fp32 tensors; a stacked tensor
  fails the harness `compare_values` type/shape check.
- Baseline wall median is ~0.862 ms (benchmark); launch overhead is ~111 us/call
  (~12.9% of wall), so the fusion headroom is smaller than fused_moe's.

## Stop Recommendation

- recommendation: `continue`
- evidence: baseline is established and no candidate round has been evaluated yet

## Exact Reproduction Commands

```bash
cd /root/kernelswift/.worktrees/sparse-pooler-s60
python3 auto_bench.py --v0_file kernels/track1-triton/sparse_pooler/base.py --v1_file kernels/track1-triton/sparse_pooler/s60/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback
```

```bash
cd /root/kernelswift/.worktrees/sparse-pooler-s60
python3 auto_bench.py --v0_file kernels/track1-triton/sparse_pooler/base.py --v1_file kernels/track1-triton/sparse_pooler/s60/baseline_adapter.py --warmup 50 --repeat 100
```

```bash
cd /root/kernelswift/.worktrees/sparse-pooler-s60
python3 auto_bench.py --v0_file kernels/track1-triton/sparse_pooler/base.py --v1_file kernels/track1-triton/sparse_pooler/s60/baseline_adapter.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/sparse_pooler/s60/log/sparse_pooler_baseline_forward_50iter.pt.trace.json
```

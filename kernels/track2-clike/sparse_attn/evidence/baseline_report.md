# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `../base.py`
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `5922fccb822f18d2472b49706b349033733309d2a7cfd5abe0d2054df71632c2`
- Accepted reference SHA256: `64fe0fbd270c0270ed7065dd63cd5a1aabd580fd8791f5c4b4dd7504b63c4a88`
- Base SHA256: `64fe0fbd270c0270ed7065dd63cd5a1aabd580fd8791f5c4b4dd7504b63c4a88`
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `13104b9a206d67116e524ae41a61cd5fece6b44f012c4f4a99a18d797f42ac5f`
- verification_tier: `baseline`
- screening_pairs: `not-applicable: Phase 0`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| official nominal correctness | `atol=1e-2`, `rtol=1e-2`, `equal_nan=True`, seed 42, fixed scope | harness reported `PASS accuracy` | pass | `log/baseline_nominal_wall.txt` |
| mixed invalid `-1` | invalid entries affect neither numerator nor ordinary denominator | exact output equality within official tolerance | pass | `log/baseline_guardrails.txt` |
| all-invalid | output exactly zero | `torch.count_nonzero(output)==0` | pass | `log/baseline_guardrails.txt` |
| duplicate indices | each occurrence contributes | exact output equality within official tolerance | pass | `log/baseline_guardrails.txt` |
| nonzero sink/state | current arbitrary `attn_sink` is denominator-only | linearly spaced `[-2,2]` state passed | pass | `log/baseline_guardrails.txt` |
| input immutability | `q`, `kv`, `topk_idxs`, `attn_sink` unchanged | cloned snapshots equal after forward for every case | pass | `log/baseline_guardrails.txt` |
| output non-aliasing | output pointer differs from all public inputs/state | pointer inequality passed for every case | pass | `log/baseline_guardrails.txt` |

All probes used the fixed complete shape and official `1e-2/1e-2` guardrail.

## Screening Evidence

Not applicable to Phase 0.

## Baseline Wall Timing

- warmup: `200`
- repeat: `500`
- seed: `42`
- base wall median ms: `12.773625`
- baseline_adapter wall median ms: `12.772110`
- canonical baseline wall_time_ms: `12.772110`
- raw artifact: `log/baseline_nominal_wall.txt`

The harness emits the unrounded median, not all 500 individual samples. The adapter is a semantic rename of the immutable base and its measured median is the Phase 0 baseline.

## Evaluation Contract Mirror

- Evaluation Contract applicability: `not-applicable: Phase 0`
- hypothesis_id: `not-applicable: Phase 0`
- intervention: `not-applicable: Phase 0`
- expected_causal_chain: `not-applicable: Phase 0`
- primary_metric: `wall_time_ms`
- Hypothesis verdict: `confirmed` (the eager adapter reproduces base semantics and establishes baseline evidence)

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `summary`
- profiler_device_time: `available from independently captured CANN ai_core_op_summary.db`
- scope: `candidate_baseline_adapter` (separate CANN capture)
- iterations: `100` forward calls
- wall ms used for ratio: `12.772110`
- device_total_us: `1296433.78`
- device_us_per_call: `12964.3378`
- kernel_count_total: `3400`
- kernel_count_per_call: `34.0`
- summed-device-time/wall ratio: `1.0150505907011451`

CANN durations are summed task durations; overlap/concurrency can make the summed ratio exceed 1, so the ratio is descriptive rather than a utilization percentage.

### Baseline Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `aclnnInplaceCopy_CastAiCore_Cast` | 500 | 5.0 | 412094.86 | 4120.9486 |
| `aclnnBatchMatMul_BatchMatMulNd_BatchMatMulV2` | 200 | 2.0 | 305406.48 | 3054.0648 |
| `aclnnIndex_IndexAiCore_Index` | 100 | 1.0 | 234182.64 | 2341.8264 |
| `aclnnInplaceCopy_TensorMoveAiCore_TensorMove` | 300 | 3.0 | 77006.88 | 770.0688 |
| `aclnnInplaceMaskedFillScalar_MaskedFillAiCore_MaskedFill` | 300 | 3.0 | 68463.66 | 684.6366 |
| `aclnnReduceSum_ReduceSumOpAiCore_ReduceSum` | 200 | 2.0 | 36455.90 | 364.5590 |
| `aclnnAmax_ReduceMaxAiCore_ReduceMax` | 200 | 2.0 | 35851.56 | 358.5156 |
| `aclnnSub_SubAiCore_Sub` | 200 | 2.0 | 34004.16 | 340.0416 |
| `aclnnDiv_RealDivAiCore_RealDiv` | 100 | 1.0 | 32877.30 | 328.7730 |
| `aclnnExp_ExpAiCore_Exp` | 200 | 2.0 | 27010.00 | 270.1000 |

Raw trace/profiler evidence:
- `log/baseline_forward_100iter.pt.trace.json`
- `log/profiling_data/candidate_baseline_adapter/`
- `log/baseline_forward_summary.json`
- `log/baseline_forward_profile.txt`

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | guardrail command quoting error | `5922fccb...` | unchanged | shell `SyntaxError`; no model execution; discarded |
| 2 | corrected quoting plus `pipefail` | `5922fccb...` | unchanged | complete guardrail PASS |

## evidence_for_next_round

- Eager baseline is correct under nominal and every required exact-scope semantic guardrail.
- Baseline wall median is `12.772110 ms` under warmup 200/repeat 500.
- Eager forward decomposes into `34` AI Core tasks/call; the largest summed task families are Cast (`4120.9486 us/call`), BatchMatMul (`3054.0648 us/call`), and Index (`2341.8264 us/call`).
- Current baseline bottleneck evidence is broad eager decomposition and dominant cast/matmul/index device work; profiler task durations may overlap.

## Stop Recommendation

- recommendation: `continue`
- evidence: correct comparable baseline and required profiler evidence now exist; no optional target is configured.

## Exact Reproduction Commands

```bash
cd /workspace/kernelswift-dev-4ff2094/kernels/track2-clike/sparse_attn/ascendc && /usr/local/python3.11.15/bin/python3 /workspace/kernelswift-dev-4ff2094/auto_bench.py --v0_file ../base.py --v1_file baseline_adapter.py --seed 42 --atol 1e-2 --rtol 1e-2 --warmup 200 --repeat 500
```

The exact semantic guardrail command is preserved in the shell execution history and its durable output is `log/baseline_guardrails.txt`.

```bash
cd /workspace/kernelswift-dev-4ff2094/kernels/track2-clike/sparse_attn/ascendc && /usr/local/python3.11.15/bin/python3 /workspace/kernelswift-dev-4ff2094/auto_bench.py --v0_file ../base.py --v1_file baseline_adapter.py --seed 42 --atol 1e-2 --rtol 1e-2 --warmup 200 --repeat 500 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 100 --profile-output log/baseline_forward_100iter.pt.trace.json
```

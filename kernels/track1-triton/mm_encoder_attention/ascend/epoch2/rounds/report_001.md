# Report 001

Result: accepted

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md`
- Decision SHA256: `3775c9548afc7070898ee73ead2e6ecad19225525b58052946f2ff5e3c4c0167`
- Sketch: `rounds/sketch_001.json` (sha256 `76818c21a7502a68b6ec5c6230607fa24bddf3e342e61d4d333990d16d639738`)
- Candidate: `triton_mm_encoder_attention_e2_001.py`
- Candidate SHA256: `c75ec5ffaab3883ef7c5b1e62778b39fbd5413619a625fd36a86d70390e92124`
- Accepted reference: `base.py`
- Compared against: `baseline_adapter.py` (last accepted kernel, `rounds/report_000.md`)
- Last accepted kernel SHA256: `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e`
- Base SHA256: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2`
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `1b1822d7b74a8cd41411a27fcbc18a89cb50b1cfefb9fdac2585cdd520e9a79a`
- verification_tier: candidate
- screening_pairs: `3`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | candidate output matches reference within atol=1e-2, rtol=1e-2 (equal_nan=True) | `PASS accuracy` on all three interleaved pairs | pass | `auto_bench.py --v0_file .../base.py --v1_file .../epoch2/triton_mm_encoder_attention_e2_001.py --warmup 50 --repeat 100` |
| output shape | `[2, 83, 512]` fp16 on `npu:0` | unchanged from base.py | pass | correctness run |
| output dtype | fp16 | unchanged from base.py | pass | correctness run |
| public contract | `ModelNew(num_heads=8, head_size=64, num_kv_heads=8).forward(query, key, value)` | unchanged | pass | `coder_result_001.md` |
| base.py bytes | immutable | `86ac5703…` unchanged | pass | `git status` clean on tracked base |

## Screening Evidence

Three screening pairs were run in one Verifier turn before the adoption measurement:

| Pair | Reference median ms | Candidate median ms | Speedup |
|---:|---:|---:|---:|
| 1 | 0.366920 | 0.337930 | 1.086x |
| 2 | 0.360640 | 0.327770 | 1.100x |
| 3 | 0.365400 | 0.325085 | 1.124x |

All three pairs show the candidate ahead, so the effect is stable across repeats
and is not an artifact of a single warm-up state.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate` (3 pairs in one Verifier turn)
- reference_median_ms: `0.365400`
- candidate_median_ms: `0.327770`
- improvement_pct: `10.2983`

```text
improvement_pct = (0.365400 - 0.327770) / 0.365400 * 100 = 10.2983
```

The improvement clears the 5% adoption threshold. This is the first round on this
operator and backend to do so: the epoch-1 Triton rewrite measured only +2.56% and
was recorded as `no-improvement`.

Note on comparability: the harness requires `--v0_file` to define `Model`, so the
paired measurement always runs `base.py` against the candidate rather than
`baseline_adapter.py` against the candidate. This matches the epoch-1 convention
and keeps the reference re-measured in the same turn, which controls the
+9% baseline drift recorded in `rounds/report_000.md`.

## Evaluation Contract Mirror

- Evaluation Contract applicability: `applicable`
- hypothesis_id: `H-001`
- intervention: replace the native SDPA call and its materialized q/k/v transpose chain with one fused Triton flash-attention kernel that indexes q/k/v and writes the output directly in the native `[B,S,NH*HEAD_DIM]` layout
- expected_causal_chain: one fused kernel replaces 6.96-6.98 launches per call → host launch and synchronization overhead decreases → the three materialized transposes and the inplace-copy transpose disappear → device us per call decreases → wall time decreases
- primary_metric: `wall_time`
- Hypothesis verdict: `confirmed` on the device and launch links, `partially-confirmed` on the host link

### Mechanism observables

| Observable | Expectation | Reference | Candidate | Verdict |
|---|---|---|---|---|
| `kernel_count_per_call` | decrease | 6.98 | **1.00** | confirmed |
| `device_us_per_call` | decrease | 118.892 | **13.4064** | confirmed |
| `transpose_kernel_count_per_call` | decrease | 4.00 (3 transpose + 1 inplace-copy transpose) | **0.00** | confirmed |
| `device_ratio` | decrease | 0.3314 | **0.0407** | confirmed |

Every declared mechanism moved in the predicted direction, and the primary metric
cleared its threshold.

## Profiler Evidence

- profiler_applicability: `required` (round-001 device evidence)
- profiler_level: `targeted`
- profiler_device_time: `available`
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- backend_runtime_fields: `not-applicable`

Reference and candidate scopes are collected and summarized independently. All
totals normalized by `iterations=50`. The reference scope directory accumulated a
second capture from this round, so the summary was taken from the round-001
database explicitly rather than from the shared directory.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| reference_baseline_adapter | 5944.60 | 118.8920 | 349 | 6.98 | 0.358720 | 0.3314 |
| candidate_triton_mm_encoder_attention_e2_001 | 670.32 | 13.4064 | 50 | 1.00 | 0.329365 | 0.0407 |

```text
device_ratio = device_us_per_call / (wall_ms * 1000)
```

### Reference Top Kernels (reference_baseline_adapter)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| aclnnFlashAttentionScore_TransposeAiCore_Transpose | 150 | 3.0 | 2412.04 | 48.2408 |
| EVENT_WAIT_SQE | 49 | 0.98 | 1615.76 | 32.3152 |
| aclnnFlashAttentionScore_FlashAttentionScore_FlashAttentionScore | 50 | 1.0 | 1145.38 | 22.9076 |
| aclnnInplaceCopy_TransposeAiCore_Transpose | 50 | 1.0 | 770.42 | 15.4084 |
| EVENT_RECORD_SQE | 50 | 1.0 | 1.00 | 0.0200 |

### Candidate Top Kernels (candidate_triton_mm_encoder_attention_e2_001)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| _fused_attention_kernel | 50 | 1.0 | 670.32 | 13.4064 |

### Attribution

This round is a textbook instance of the `device-win-wall-loss` pattern, and the
numbers show precisely why the pattern exists:

- Device time fell **88.7%** (`118.892` → `13.4064 us/call`) and launch count fell
  **85.7%** (`6.98` → `1.00`), yet wall time fell only **10.3%**.
- Device time went from about one third of wall time to about **4%** of it.
  The candidate is no longer device-bound in any meaningful sense.
- The remaining `~316 us/call` is host-side cost that did **not** scale with launch
  count. Removing six of seven launches recovered only about `38 us` of wall time,
  so the dominant host term is a per-call fixed cost (harness synchronization,
  dispatch, and the Ascend launch path itself), not the number of kernel launches.

The practical consequence for round 002: further kernel-side work has almost
nothing left to win. Device time is already `13.4 us` against a `327.8 us` wall.
The next round must attack the per-call fixed host cost, and it will have to do so
with a mechanism other than launch-count reduction.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial coding from decision 001 | `not-applicable` | `c75ec5ffaab3883ef7c5b1e62778b39fbd5413619a625fd36a86d70390e92124` | correctness pass on first attempt, no repair needed |

## evidence_for_next_round

- **Adoption cleared**: wall median `0.327770` ms versus reference `0.365400` ms, improvement `10.2983%`, above the 5% threshold. First accepted round for this operator on Ascend.
- Device time is now `13.4064 us/call` at `1.00` kernels per call, down from `118.892 us/call` at `6.98`.
- `device_ratio` is `0.0407`. Device-side optimization is exhausted: the kernel is already a single launch doing `13.4 us` of work.
- **The remaining bottleneck is a fixed per-call host cost of roughly `316 us`** that did not decrease proportionally with launch count. Launch-count reduction is no longer a lever.
- The winning mechanism was eliminating materialized layout conversion (`63.35 us/call` of transposes) plus collapsing launches. The attention math itself (`22.91 us/call` in the reference) now costs `13.41 us/call` in Triton, so Triton is already beating the vendor SDPA kernel on device time as well.
- Candidate is correctness-PASS and is now the canonical accepted kernel, which also satisfies the deliverable rule that the submission be a Triton implementation.
- Candidate limitation to record: the kernel requires `S <= 128`. The campaign shape is `S=83`. A larger sequence would need a row-blocked loop.

## Stop Recommendation

- recommendation: `continue`
- evidence: an accepted round just advanced the canonical kernel; the round budget (0/20 terminal so far in this epoch, 1 completed), the no-improvement streak (reset to 0), and the failed-attempt streak (0) all permit another round. The largest remaining term, the fixed per-call host cost, has not yet been attacked.

## Exact Reproduction Commands

```bash
cd /workspace/kernelswift-dev-4ff2094
python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/triton_mm_encoder_attention_e2_001.py --warmup 50 --repeat 100
```

```bash
cd /workspace/kernelswift-dev-4ff2094
python3 auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/triton_mm_encoder_attention_e2_001.py --profile --profile-reference-file kernels/track1-triton/mm_encoder_attention/ascend/epoch2/baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/mm_encoder_attention/ascend/epoch2/log/round_001_forward_50iter.pt.trace.json
```

```bash
cd /workspace/kernelswift-dev-4ff2094
python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py "kernels/track1-triton/mm_encoder_attention/ascend/epoch2/log/profiling_data/reference_baseline_adapter/profiling_data/16458e336fc3_47846_20260830053214549_ascend_pt/PROF_000001_20260830053214573_00047846RKBOOGLA/device_0/sqlite/ai_core_op_summary.db" --iterations 50 --scope reference_baseline_adapter --wall-ms 0.358720

python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py "kernels/track1-triton/mm_encoder_attention/ascend/epoch2/log/profiling_data/candidate_triton_mm_encoder_attention_e2_001/profiling_data" --iterations 50 --scope candidate_triton_mm_encoder_attention_e2_001 --wall-ms 0.329365
```

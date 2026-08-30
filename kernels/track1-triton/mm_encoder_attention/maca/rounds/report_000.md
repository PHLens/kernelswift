# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `base.py for Phase 0`
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e`
- Accepted reference SHA256: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2`
- Base SHA256: `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `29ecde127206fc1808c2d7f28951e44ee55a257aadfda78517e64d3493ce1862`
- verification_tier: baseline
- screening_pairs: `not-run (Phase 0 baseline)`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | v0 `Model` and v1 `ModelNew` outputs allclose (atol=1e-2, rtol=1e-2, equal_nan=True) | `PASS accuracy; v0=0.130614 ms, v1=0.122941 ms, speedup=1.062x` | pass | `RETURN_CODE=0` from correctness command (warmup 5 repeat 10) |
| non-mutation of inputs | forward must not mutate inputs | Harness clones inputs before each forward and both sides share cloned reference inputs | pass | `auto_bench.py` `clone_value` + `run_forward` (torch.no_grad) |
| output shape/dtype/device | `(bsz, seq_len, hidden)` fp16 on input device | compare passed (shape/dtype/device all matched, no mismatch raised) | pass | correctness command output `PASS accuracy` |
| device/stream preservation | candidate preserves caller-selected device/current stream | Both models run on `cuda:0`; harness moves inputs to detected device | pass | `_detect_target_device` returned cuda:0 |

Conformance, correctness, and every declared guardrail pass before adoption.

## Screening Evidence

Not applicable: Phase 0 baseline (no candidate vs. accepted-reference comparison).

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `sequential complete accepted-reference block, then complete candidate block` (Phase 0 baseline regime)
- reference_raw_samples_ms (v0, three authoritative runs): `[0.117305, 0.117117, 0.117557]`
- candidate_raw_samples_ms (v1, three authoritative runs): `[0.115726, 0.115881, 0.115761]`
- reference_median_ms: `0.117305`
- candidate_median_ms: `0.115761`
- improvement_pct: `1.316073818` (candidate faster than reference; not an adoption decision in Phase 0)

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
```

Note: Phase 0 measures the baseline. `baseline_adapter.py` (v1, `ModelNew`) is a byte-level functional equivalent of `base.py` (v0, `Model`); both lower `F.scaled_dot_product_attention` identically. The tiny measured delta (~1.3%) is run-to-run variance, not an intervention. The canonical Phase 0 baseline is `baseline_adapter.py`.

## Evaluation Contract Mirror

- Evaluation Contract applicability: `not-applicable: Phase 0`
- hypothesis_id: `not-applicable: Phase 0`
- intervention: `not-applicable: Phase 0`
- expected_causal_chain: `not-applicable: Phase 0`
- primary_metric: `wall_time | not-applicable: Phase 0`
- Hypothesis verdict: `not-applicable: Phase 0`

No round decision exists in Phase 0; there are no mechanism observables to mirror.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `summary`
- profiler_device_time: `available`
- iterations: `50`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- backend_runtime_fields: `not applicable (device kernel time available on C500)`

### C500 trace filtering note (documented)

The raw trace `round_000_forward_50iter.pt.trace.json` contains duplicate nested
scope markers: each scope (`baseline_base`, `candidate_baseline_adapter`) has a
CPU-side `cat=user_annotation` X event that fully contains a nested GPU-side
`cat=gpu_user_annotation` X event. `summarize_trace.py` matches any
`ph=X, cat!=kernel, name==scope` event, so both events are matched and reported
as `overlapping scope events` (the known C500 issue).

Fix applied: the raw trace was preserved unchanged; a filtered trace
`round_000_forward_50iter.filtered.pt.trace.json` was produced by dropping the
two `cat=user_annotation` X events (the outer CPU-side markers), keeping the
`cat=gpu_user_annotation` device-side markers that bound the kernel durations.
All 200 kernel events were verified to fall inside their respective
`gpu_user_annotation` intervals before and after filtering. The summaries below
are from the filtered trace; totals are unchanged by the filter because no
kernel events were touched.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| baseline_base (v0) | 753.41064453125 | 15.068212890625 | 100 | 2.0 | 0.117305 | 0.12845328750372959 |
| candidate_baseline_adapter (v1) | 749.0595703125 | 14.98119140625 | 100 | 2.0 | 0.115761 | 0.1294148409762355 |

```text
device_ratio = device_us_per_call / (candidate_median_ms * 1000)
```

### Accepted Reference (v0, baseline_base) Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `void flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<64, 64, 64, 4, true, true, mctlass::half_t, 64, Flash_kernel_traits<64, 64, 64, 4, mctlass::half_t> >, false, false, false, false, true, false, true, false, false, (Arch)1000>(mcFlashAttn::Flash_fwd_params, int, int)` | 50 | 1.0 | 436.7412109375 | 8.73482421875 |
| `void flash_fwd_splitkv_combine_kernel<Flash_fwd_kernel_traits<64, 64, 64, 4, true, true, mctlass::half_t, 64, Flash_kernel_traits<64, 64, 64, 4, mctlass::half_t> >, 16, 1, true, false>(mcFlashAttn::Flash_fwd_params)` | 50 | 1.0 | 316.66943359375 | 6.333388671875 |

### Candidate (v1, candidate_baseline_adapter) Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `void flash_fwd_splitkv_kernel<Flash_fwd_kernel_traits<64, 64, 64, 4, true, true, mctlass::half_t, 64, Flash_kernel_traits<64, 64, 64, 4, mctlass::half_t> >, false, false, false, false, true, false, true, false, false, (Arch)1000>(mcFlashAttn::Flash_fwd_params, int, int)` | 50 | 1.0 | 434.693359375 | 8.6938671875 |
| `void flash_fwd_splitkv_combine_kernel<Flash_fwd_kernel_traits<64, 64, 64, 4, true, true, mctlass::half_t, 64, Flash_kernel_traits<64, 64, 64, 4, mctlass::half_t> >, 16, 1, true, false>(mcFlashAttn::Flash_fwd_params)` | 50 | 1.0 | 314.3662109375 | 6.28732421875 |

### SDPA lowering (critical evidence for Round 1 decision)

`F.scaled_dot_product_attention` (fp16, `[2,8,83,64]`, MHA with
`num_kv_heads==num_heads==8`, no mask, `cu_seqlens=None`) lowers on C500 to
**flash attention** (not the mem-efficient path, not the math fallback, not a
fused MHA op). Exactly two device kernels per forward call, in the
`mcFlashAttn` namespace:

1. `flash_fwd_splitkv_kernel` — the main flash-attention split-KV forward kernel (dominant, ~58% of device time).
2. `flash_fwd_splitkv_combine_kernel` — the split-KV combine/reduction kernel (~42% of device time).

Kernel count per forward call is exactly 2, and the device time per call is
~15 us (out of ~117 us wall), so the dominant cost is device-side flash
attention with a split-KV combine step, plus host/launch overhead (device ratio
~12.8%).

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial Phase 0 baseline verification | `not-applicable` | `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e` | correctness pass, timing + profiler complete |

## evidence_for_next_round

- SDPA lowers to flash attention on C500: two `mcFlashAttn` kernels per forward — `flash_fwd_splitkv_kernel` (dominant, ~8.7 us/call) and `flash_fwd_splitkv_combine_kernel` (~6.3 us/call). Kernel count per call is exactly 2.
- Device time per call is ~15 us against ~117 us wall, so device ratio is only ~12.9%; the majority of wall time is host-side launch overhead / fixed per-call cost, not device kernel compute.
- The split-KV combine step (`flash_fwd_splitkv_combine_kernel`) is a substantial secondary kernel (~42% of device time); any optimization must account for it, not just the main splitkv kernel.
- No mem-efficient or math-fallback path is observed; the backend already selects flash attention for this MHA shape.

## Stop Recommendation

- recommendation: `continue`
- evidence: `Phase 0 baseline established; wall ~0.117 ms, device ~15 us/call across 2 flash-attention kernels. No stop condition met.`

## Exact Reproduction Commands

```bash
cd /root/kernelswift-mma && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/maca/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback
```

```bash
cd /root/kernelswift-mma && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/maca/baseline_adapter.py --warmup 50 --repeat 100
```

```bash
cd /root/kernelswift-mma && /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/maca/baseline_adapter.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/mm_encoder_attention/maca/log/round_000_forward_50iter.pt.trace.json
```

```bash
cd /root/kernelswift-mma && python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/mm_encoder_attention/maca/log/round_000_forward_50iter.filtered.pt.trace.json --iterations 50 --scope baseline_base --wall-ms 0.117305
cd /root/kernelswift-mma && python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/mm_encoder_attention/maca/log/round_000_forward_50iter.filtered.pt.trace.json --iterations 50 --scope candidate_baseline_adapter --wall-ms 0.115761
```

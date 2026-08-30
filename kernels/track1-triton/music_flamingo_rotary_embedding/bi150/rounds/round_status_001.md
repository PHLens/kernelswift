# Round Status 001

Result: accepted

## Phase

`verifying` (Round 001, task `music_flamingo_rotary_embedding`, BI150 backend, measurement-exclusive) — complete.

## Completed Steps

### 0. Frozen-artifact hash verification (before measurement)

| Artifact | Actual SHA-256 | Expected | Match |
|---|---|---|---|
| `base.py` | `98be7d25ad94930fd9497e6ab33e64970c509c6c3569c8c3df40ae9d22b9a341` | `98be7d25ad949...` | match |
| `baseline_adapter.py` | `433569bbac3bab158ff211a6de7ecb40ec7236d74a3eb7ab7c2b487e1b41772a` | `433569bb...` | match |
| `triton_music_flamingo_rotary_embedding_001.py` | `d91a112c4d703e140358b0e648a83187ad1ae1ab44dd67ef1d80c69097fedd46` | `d91a112c...` | match |
| `auto_bench.py` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` | `3d4fa4ee...` | match |
| `summarize_trace.py` | `f625276c05a4539f86f272f68d78dd84ba0c94bb4ce153613dc4437df432148c` | (Phase 0 recorded) | match |

### 1. Correctness (base.py vs candidate 001)

Command return code `0`; `PASS accuracy; v0=0.341751 ms, v1=0.175280 ms, speedup=1.950x`;
`Summary: 1 passed, 0 failed, 1 total.`

### 1b. Independent numerical-semantics probe

base `Model` vs candidate `ModelNew` via harness AST loader, same `timestamps`:
- cos allclose `True`, sin allclose `True` (`atol=1e-2, rtol=1e-2, equal_nan=True`)
- `max_abs_diff = 0.0` for both cos and sin (exact match)
- both shapes `(4, 32, 128)`

### 2. Authoritative interleaved wall timing (reference wrapper vs candidate)

Reference wrapper: `sed 's/^class ModelNew/class Model/' baseline_adapter.py`,
SHA256 `a7f0825841f5f11efb7d16db75479cb92744610bcd0a35e29359a6461a1e5e9d` (deleted after use).

warmup=50, repeat=100, three independent invocations:

| Invocation | Reference wall ms (v0) | Candidate wall ms (v1) | speedup | return code |
|---|---:|---:|---:|---:|
| 1 | `0.336145` | `0.175263` | `1.918x` | 0 |
| 2 | `0.342906` | `0.177024` | `1.937x` | 0 |
| 3 | `0.343957` | `0.176121` | `1.953x` | 0 |

- reference_raw_samples_ms: `[0.336145, 0.342906, 0.343957]`
- candidate_raw_samples_ms: `[0.175263, 0.177024, 0.176121]`
- reference_median_ms: `0.342906`
- candidate_median_ms: `0.176121`
- improvement_pct: `48.63869398610698` (>= 5.0 → accepted)

### 3. Targeted profiler (forward, 20/50)

Trace: `log/round_001_forward_50iter.pt.trace.json`, SHA256
`d7200beabe8bdcbb68046e8889ea00c6614f4a32747f27581d652d5a121f591a`.

| Scope | device_total_us | device_us_per_call | kernel_count_total | kernel_count_per_call | device_ratio |
|---|---:|---:|---:|---:|---:|
| `reference_baseline_adapter` | `3442.343` | `68.847` | `543` | `10.86` | `0.200775` |
| `candidate_triton_music_flamingo_rotary_embedding_001` | `1541.453` | `30.829` | `50` | `1.0` | `0.175044` |

Fusion confirmed: candidate kernel_count_per_call dropped `10.86 -> 1.0`
(exactly one `_fused_rotary_embedding_kernel` per forward call).

Candidate scope summarization note: the unmodified `summarize_trace.py` returned
`2` (`overlapping scope events`) because the fused forward's `record_function`
markers overlap in host time (a BI150 profiler artifact for very fast forwards).
Candidate device evidence was attributed directly from the 50
`_fused_rotary_embedding_kernel` events (all within the candidate scope span),
yielding `kernel_count_per_call = 1.0` and `device_us_per_call = 30.829 us`.
Seven aten kernels in the candidate time window are reference-tail leakage
(ts `4096298026513..6568`, before the first Triton kernel at `4096298026569`)
and are excluded from candidate totals.

### 4. Post-measurement hash verification

Identical to pre-measurement; no frozen artifact changed. Wrapper deleted.

## Next Safe Action

None. Verification complete; report `Result=accepted` to Orchestrator. Do not
update `last_accepted_kernel` (Orchestrator owns canonical pointer updates).

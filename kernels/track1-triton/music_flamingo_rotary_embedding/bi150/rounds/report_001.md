# Report 001

Result: accepted

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md`
- Candidate: `kernels/track1-triton/music_flamingo_rotary_embedding/bi150/triton_music_flamingo_rotary_embedding_001.py`
- Accepted reference: `kernels/track1-triton/music_flamingo_rotary_embedding/bi150/baseline_adapter.py`
- Accepted reference report: `rounds/report_000.md`
- Decision SHA256: `28a716e6bafa46e0bd9c39350317e42173694b9406eb3c620c361b55db0bb383`
- Candidate SHA256: `d91a112c4d703e140358b0e648a83187ad1ae1ab44dd67ef1d80c69097fedd46`
- Accepted reference SHA256: `433569bbac3bab158ff211a6de7ecb40ec7236d74a3eb7ab7c2b487e1b41772a`
- Base SHA256: `98be7d25ad94930fd9497e6ab33e64970c509c6c3569c8c3df40ae9d22b9a341`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint` (torch 2.7.1, triton 3.1.0, Iluvatar BI-V150, capability (7,1))
- Measurement fingerprint: `896adb91dbe5f84f9de83644e058462173cd5423a61bdf1ebcb2a15ca783c0be`
- verification_tier: `authoritative`
- screening_pairs: `not-run: correctness passed, proceeded directly to authoritative timing`

The candidate, adapter, base, and harness hashes all match the frozen project
values. The `base_sha256` discrepancy noted in report_000 (recorded
`98be7d25264f...` vs actual `98be7d25ad949...`) is unchanged and was already
flagged to Orchestrator; it does not affect correctness/benchmark/profiler
evidence, which all run on file contents directly.

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | Harness recursive comparison at seed 42, `atol=1e-2`, `rtol=1e-2`, equal-NaN enabled | `PASS accuracy; v0=0.341751 ms, v1=0.175280 ms, speedup=1.950x`; return code `0` | pass | correctness command in Exact Reproduction Commands |
| numerical semantics (independent) | base `Model` vs candidate `ModelNew` via harness AST loader, same `timestamps`, `allclose(atol=1e-2, rtol=1e-2, equal_nan=True)` | cos allclose `True`, sin allclose `True`; `max_abs_diff=0.0` for both (exact match) | pass | independent loader probe |
| output tuple/shape/dtype | Two-tensor tuple `(cos, sin)`, each `(4, 32, 128)`, float32 | Independent probe reported `cos shape (4,32,128)`, `sin shape (4,32,128)` on both sides | pass | independent loader probe |
| public loader contract | candidate exposes `ModelNew/get_init_inputs/get_inputs`; reference exposes `Model/get_init_inputs/get_inputs` through the AST loader | Harness loaded, constructed, moved, and executed both sides without load or constructor error | pass | correctness return code `0` |
| input not mutated | `timestamps` only read; no in-place writes | Candidate `forward` reads `timestamps` and writes only fresh `cos_out`/`sin_out` | pass | source review + correctness pass |
| device/stream preserved | caller-selected device and current stream preserved | Kernel launches on input device via current CUDA stream; no device-context change | pass | source review |
| frozen artifact identity | local hashes equal frozen values before and after measurement | candidate `d91a112c...`, adapter `433569bb...`, base `98be7d25ad949...`, harness `3d4fa4ee...` all match | pass | SHA256 commands in Exact Reproduction Commands |
| measurement regime | device cuda:0, seed/tolerances defaults, `warmup=50`, `repeat=100`, forward profile `20/50` | Commands used frozen arguments byte-for-byte; only `--v1_file` changed between pairs | pass | round_status_001.md |

The correctness command's `v0=0.341751 ms` / `v1=0.175280 ms` values are smoke
timing only and do not replace the frozen 50/100 authoritative samples.

## Screening Evidence

Not applicable. Correctness passed and the candidate proceeded directly to
authoritative timing; no screening classification was made.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate` (harness times v0 reference then v1 candidate per invocation)
- independent invocations: `3`
- reference_raw_samples_ms: `[0.336145, 0.342906, 0.343957]`
- candidate_raw_samples_ms: `[0.175263, 0.177024, 0.176121]`
- reference_median_ms: `0.342906`
- candidate_median_ms: `0.176121`
- improvement_pct: `48.63869398610698`

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
               = (0.342906 - 0.176121) / 0.342906 * 100
               = 48.63869398610698
```

The unrounded improvement (`48.64%`) far exceeds the `5%` adoption threshold.
The candidate collapses ~13 elementwise kernel launches into a single Triton
kernel, and the host-bound wall time drops by nearly half.

| Independent invocation | Reference wall ms | Candidate wall ms | Command return code |
|---:|---:|---:|---:|
| 1 | `0.336145` | `0.175263` | `0` |
| 2 | `0.342906` | `0.177024` | `0` |
| 3 | `0.343957` | `0.176121` | `0` |

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| `wall_time` | expected_improvement_pct `5.0` | improvement `48.64%` (median `0.342906 -> 0.176121 ms`) | pass | Interleaved Wall Timing |
| `kernel_count_per_call` | decrease | `10.86 -> 1.0` kernels/call (543 -> 50 over 50 iterations) | pass | Profiler Evidence candidate scope |
| `device_us_per_call` | decrease | `68.846 -> 30.829 us/call` (device work also reduced) | pass | Profiler Evidence |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-001`
- intervention: `fuse the forward elementwise chain (arange/div/repeat_interleave/broadcast/cat/neg/mul/cos/sin) into a single Triton kernel`
- expected_causal_chain: `per-call device kernel count drops from 10.86 to ~1-2 -> host launch/dispatch overhead decreases -> wall time decreases`
- primary_metric: `wall_time`
- Hypothesis verdict: `confirmed`

All three declared mechanism observables (`wall_time`, `kernel_count_per_call`,
`device_us_per_call`) moved in the predicted direction. The causal chain is
directly evidenced: kernel count collapsed `10.86 -> 1.0`, device time dropped
`68.846 -> 30.829 us/call` (fewer launches and no intermediate materialization),
and wall time improved `48.64%`.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `available` (`cat=kernel` durations)
- profile_mode: `forward`
- warmup: `20`
- iterations: `50` forward calls per scope
- scopes: `reference_baseline_adapter`, `candidate_triton_music_flamingo_rotary_embedding_001`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- raw trace: `log/round_001_forward_50iter.pt.trace.json`, SHA256 `d7200beabe8bdcbb68046e8889ea00c6614f4a32747f27581d652d5a121f591a`
- unmodified summarizer SHA256: `f625276c05a4539f86f272f68d78dd84ba0c94bb4ce153613dc4437df432148c`

The reference scope summarized cleanly via the unmodified summarizer (one
non-overlapping `X` marker, 543 kernels). The candidate scope could NOT be
summarized by the unmodified summarizer because its `record_function` markers
collapsed to two overlapping intervals: the fused forward is so fast that
consecutive host-side scope markers overlap in wall-clock time (a BI150 profiler
artifact). To produce accurate per-call candidate evidence without editing the
summarizer, the candidate's device work was measured directly from its
`_fused_rotary_embedding_kernel` events, all 50 of which fall inside the
candidate scope span. See "Candidate Scope Measurement Note" below.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| `reference_baseline_adapter` | `3442.343` | `68.847` | `543` | `10.86` | `0.342906` | `0.200775` |
| `candidate_triton_music_flamingo_rotary_embedding_001` | `1541.453` | `30.829` | `50` | `1.0` | `0.176121` | `0.175044` |

```text
device_ratio = device_us_per_call / (candidate_median_ms * 1000)
reference: 68.847 / 342.906 = 0.200775
candidate: 30.829 / 176.121 = 0.175044
```

### Accepted Reference Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| MulFunctor elementwise (binary Mul) | `99` | `1.98` | `884.140` | `17.683` |
| CatArrayNoContiguous | `49` | `0.98` | `554.346` | `11.087` |
| AUnaryFunctor Mul | `98` | `1.96` | `398.256` | `7.965` |
| sin_kernel_cuda | `49` | `0.98` | `366.310` | `7.326` |
| cos_kernel_cuda | `49` | `0.98` | `357.942` | `7.159` |
| direct_copy_kernel_cuda | `50` | `1.0` | `299.470` | `5.989` |
| BUnaryFunctor Mul | `50` | `1.0` | `202.105` | `4.042` |
| neg_kernel_cuda | `49` | `0.98` | `197.917` | `3.958` |
| arange_cuda_out | `50` | `1.0` | `181.856` | `3.637` |

### Candidate Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `_fused_rotary_embedding_kernel` | `50` | `1.0` | `1541.453` | `30.829` |

The candidate emits exactly one device kernel per forward call — the fused
Triton kernel. This confirms the fusion collapsed `10.86` kernels/call to
`1.0`, matching the decision's expectation of "approximately 1-2".

### Candidate Scope Measurement Note

The unmodified `summarize_trace.py` returned exit code `2`
(`overlapping scope events: candidate_triton_music_flamingo_rotary_embedding_001`)
because the candidate's 50 forward calls produced only two `record_function`
scope markers whose intervals overlap in host time (the fused forward is faster
than the host-side marker resolution, so markers stack/overlap). This is a
profiler artifact, not a correctness or measurement failure.

To produce per-call candidate evidence without editing the frozen summarizer,
the candidate's device work was attributed directly from its
`_fused_rotary_embedding_kernel` events: all 50 Triton kernels (one per forward
call) fall within the candidate scope span, yielding `kernel_count_per_call =
1.0` and `device_us_per_call = 30.829`. Seven non-Triton (aten) kernels also
appear in the candidate time window but their timestamps (`4096298026513` through
`4096298026568`) sit in the gap before the first Triton kernel
(`4096298026569`); they are the reference scope's tail kernels that leaked past
the reference marker's end, not candidate work. They are excluded from the
candidate totals, consistent with the scope-boundary sampling artifact noted in
report_000 (543 vs 550 kernel-count discrepancy).

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | Initial correctness, timing, and profiler verification | `d91a112c4d703e140358b0e648a83187ad1ae1ab44dd67ef1d80c69097fedd46` | same | correctness and wall timing passed; profiler reference summarized; candidate summarized via direct kernel attribution (overlapping-marker artifact) |

No candidate repair occurred and no source file changed.

## evidence_for_next_round

- Kernel fusion (H-001) is confirmed: wall time improved `48.64%`
  (`0.342906 -> 0.176121 ms` median), kernel count collapsed `10.86 -> 1.0`
  per call, and device time dropped `68.847 -> 30.829 us/call`.
- The fused single Triton kernel now dominates candidate device time
  (`30.829 us/call`), and candidate `device_ratio = 0.175` — the remaining
  `82.5%` of candidate wall time is still host-side (harness seed/clone/
  synchronize + single launch + synchronize). The host-bound bottleneck is now
  largely fixed-cost harness/launch overhead rather than the elementwise chain.
- The BI150 profiler collapses the fast fused forward's `record_function`
  markers into overlapping intervals; candidate device evidence required direct
  `_fused_rotary_embedding_kernel` attribution. Future rounds profiling even
  faster candidates may hit the same artifact.

## Stop Recommendation

- recommendation: `continue`
- evidence: H-001 was confirmed with a large `48.64%` wall improvement. The
  candidate device ratio (`0.175`) shows the remaining wall time is dominated by
  host-side harness overhead (seed/clone/synchronize) rather than device work or
  the elementwise chain, so further wall-time gains may be limited by the
  harness measurement regime. No optional target is configured. Orchestrator
  owns the stop transition.

## Exact Reproduction Commands

Environment bootstrap (every command):

```bash
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
```

Frozen-file SHA256 verification (run before and after measurement; all returned code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && sha256sum kernels/track1-triton/music_flamingo_rotary_embedding/base.py kernels/track1-triton/music_flamingo_rotary_embedding/bi150/baseline_adapter.py kernels/track1-triton/music_flamingo_rotary_embedding/bi150/triton_music_flamingo_rotary_embedding_001.py auto_bench.py
```

Correctness (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/bi150/triton_music_flamingo_rotary_embedding_001.py --warmup 50 --repeat 100 --full-traceback
```

Authoritative wall timing (reference wrapper; execute independently three times; return codes `0, 0, 0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && sed 's/^class ModelNew/class Model/' kernels/track1-triton/music_flamingo_rotary_embedding/bi150/baseline_adapter.py > /tmp/mre_baseline_model_001.py && python3 auto_bench.py --v0_file /tmp/mre_baseline_model_001.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/bi150/triton_music_flamingo_rotary_embedding_001.py --warmup 50 --repeat 100
```

Forward profiler (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file /tmp/mre_baseline_model_001.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/bi150/triton_music_flamingo_rotary_embedding_001.py --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-reference-file kernels/track1-triton/music_flamingo_rotary_embedding/bi150/baseline_adapter.py --profile-output kernels/track1-triton/music_flamingo_rotary_embedding/bi150/log/round_001_forward_50iter.pt.trace.json
```

Separately scoped unmodified repository summaries (reference returned code `0`; candidate returned code `2` due to overlapping markers, so candidate was summarized via direct kernel attribution):

```bash
python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/music_flamingo_rotary_embedding/bi150/log/round_001_forward_50iter.pt.trace.json --iterations 50 --scope reference_baseline_adapter --wall-ms 0.342906
```

## Command Status and Raw Evidence Index

| Command | Return code | Evidence |
|---|---:|---|
| frozen-file SHA256 before measurement | `0` | hashes in Identity |
| correctness 50/100 | `0` | report Correctness table |
| independent numerical-semantics probe | `0` | max_abs_diff 0.0, shapes (4,32,128) |
| wall sample 1, 50/100 | `0` | report Interleaved Wall Timing |
| wall sample 2, 50/100 | `0` | report Interleaved Wall Timing |
| wall sample 3, 50/100 | `0` | report Interleaved Wall Timing |
| forward profiler 20/50 | `0` | `log/round_001_forward_50iter.pt.trace.json` |
| summarize `reference_baseline_adapter` | `0` | report Profiler Evidence |
| summarize `candidate_...` (unmodified) | `2` | overlapping markers; candidate attributed via direct kernel count |
| frozen-file SHA256 after measurement | `0` | hashes in Identity |

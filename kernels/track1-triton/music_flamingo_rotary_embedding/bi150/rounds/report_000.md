# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `base.py` (Phase 0 source reference; no pre-existing canonical implementation)
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `433569bbac3bab158ff211a6de7ecb40ec7236d74a3eb7ab7c2b487e1b41772a`
- Accepted reference SHA256: `98be7d25ad94930fd9497e6ab33e64970c509c6c3569c8c3df40ae9d22b9a341`
- Base SHA256: `98be7d25ad94930fd9497e6ab33e64970c509c6c3569c8c3df40ae9d22b9a341`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint` (torch 2.7.1, triton 3.1.0, Iluvatar BI-V150, capability (7,1))
- Measurement fingerprint: `896adb91dbe5f84f9de83644e058462173cd5423a61bdf1ebcb2a15ca783c0be`
- verification_tier: `baseline`
- screening_pairs: `not-run: Phase 0`
- completed_at: `2026-08-18T12:10:00Z`

The adapter and harness hashes match the frozen project values exactly. The
`base.py` hash recorded in `project.md` (`98be7d25264f...`) does not match the
actual file SHA-256 (`98be7d25ad949...`); see `Frozen Artifact Hash Note` below.
This is a recorded-hash discrepancy, not a file mutation: `git hash-object` and
`git rev-parse HEAD:<path>` both return `f1998e7fe09f798d5562be44dbc6453263a4a871`
and `git status --short` reports no modification, so the working-tree `base.py`
bytes are identical to the committed bytes.

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | Harness recursive comparison at seed 42, `atol=1e-2`, `rtol=1e-2`, equal-NaN enabled | `PASS accuracy; v0=0.348312 ms, v1=0.348387 ms, speedup=1.000x`; return code `0` | pass | correctness command in Exact Reproduction Commands |
| public loader contract | `base.py` exposes `Model/get_init_inputs/get_inputs`; adapter exposes `ModelNew/get_init_inputs/get_inputs` through the actual AST loader | Frozen harness loaded, constructed, moved, and executed both sides without load or constructor error | pass | correctness return code `0` |
| output tuple/shape/dtype | Two-tensor tuple `(cos, sin)`, each `(4, 32, 128)`, float32 | Harness recursive comparator accepted structure, shapes, and dtypes | pass | correctness return code `0`; `project.md#semantics` |
| floating values | `torch.allclose(..., atol=1e-2, rtol=1e-2, equal_nan=True)` | No floating mismatch | pass | correctness return code `0` |
| frozen artifact identity | local hashes equal project.md before and after measurement | adapter `433569bb...`, harness `3d4fa4ee...` match; base `98be7d25ad949...` actual vs `98be7d25264f...` recorded (see note) | pass (adapter/harness); base hash record flagged | SHA256 commands in Exact Reproduction Commands |
| measurement regime | device cuda:0, seed/tolerances defaults, `warmup=50`, `repeat=100`, forward profile `20/50` | Commands used frozen arguments byte-for-byte | pass | round_status_000.md |

The correctness command's `v0=0.348312 ms` and `v1=0.348387 ms` values are smoke
timing only and do not replace the frozen 50/100 baseline samples.

## Screening Evidence

Not applicable in Phase 0. No screening classification was made.

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate` (harness times v0 then v1 per invocation)
- independent invocations: `3`
- reference_raw_samples_ms: `[0.353050, 0.355387, 0.353447]`
- candidate_raw_samples_ms: `[0.349781, 0.354048, 0.350067]`
- reference_median_ms: `0.353447`
- candidate_median_ms: `0.350067`
- improvement_pct: `0.9560616173939068`

```text
improvement_pct = (0.353447 - 0.350067) / 0.353447 * 100
               = 0.9560616173939068
```

This descriptive mechanical-adapter comparison is not an optimization-adoption
decision. Round 000 establishes `baseline_adapter.py` as the baseline; the result
is neither `accepted` nor `no-improvement`.

| Independent invocation | Reference wall ms | Candidate wall ms | Command return code |
|---:|---:|---:|---:|
| 1 | `0.353050` | `0.349781` | `0` |
| 2 | `0.355387` | `0.354048` | `0` |
| 3 | `0.353447` | `0.350067` | `0` |

## Evaluation Contract Mirror

- Evaluation Contract applicability: `not-applicable: Phase 0`
- hypothesis_id: `not-applicable: Phase 0`
- intervention: `not-applicable: Phase 0`
- expected_causal_chain: `not-applicable: Phase 0`
- primary_metric: `not-applicable: Phase 0`
- Hypothesis verdict: `inconclusive` (no Phase 0 optimization hypothesis exists)

No decision or `mechanism_observables[]` exists for Phase 0, so there are no
missing required observables.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `summary`
- profiler_device_time: `available` (`cat=kernel` durations)
- profile_mode: `forward`
- warmup: `20`
- iterations: `50` forward calls per scope
- scopes: `baseline_base`, `candidate_baseline_adapter`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- raw trace: `log/round_000_forward_50iter.pt.trace.json`, SHA256 `bb0fc81b27401060e945243f3ca9a52b0c6a40b0d9f7a7b897b46c1638892412`
- unmodified summarizer SHA256: `f625276c05a4539f86f272f68d78dd84ba0c94bb4ce153613dc4437df432148c`

The BI150 trace emitted a single non-overlapping `X` scope marker for each scope
(`baseline_base` and `candidate_baseline_adapter`), so the unmodified repository
summarizer returned code `0` for both scopes with no duplicate-marker filtering
required (unlike the C500 grouped-topk reference).

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_base` | `3431.8212890625` | `68.63642578125` | `543` | `10.86` | `0.353447` | `0.19419156417015845` |
| `candidate_baseline_adapter` | `3467.314453125` | `69.3462890625` | `550` | `11.0` | `0.350067` | `0.19809433354900635` |

The kernel-count difference (543 vs 550) is a scope-boundary sampling artifact of
the 50-iteration forward profile, not a semantic difference: the two scopes share
an identical top-kernel set and the same computation.

### Baseline Top Kernels (baseline_base scope)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| MulFunctor elementwise (binary Mul) | `99` | `1.98` | `884.717` | `17.694` |
| CatArrayNoContiguous | `49` | `0.98` | `547.706` | `10.954` |
| AUnaryFunctor Mul | `98` | `1.96` | `395.876` | `7.918` |
| sin_kernel_cuda | `49` | `0.98` | `363.833` | `7.277` |
| cos_kernel_cuda | `49` | `0.98` | `356.596` | `7.132` |
| direct_copy_kernel_cuda | `50` | `1.0` | `303.202` | `6.064` |
| BUnaryFunctor Mul | `50` | `1.0` | `201.568` | `4.031` |
| neg_kernel_cuda | `49` | `0.98` | `197.596` | `3.952` |
| arange_cuda_out | `50` | `1.0` | `180.729` | `3.615` |

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | Initial correctness, timing, and profiler verification | `433569bbac3bab158ff211a6de7ecb40ec7236d74a3eb7ab7c2b487e1b41772a` | same | correctness and wall timing passed; profiler summarized without filtering |

No candidate repair occurred and no source file changed.

## Frozen Artifact Hash Note

The `base_sha256` recorded in `project.md`
(`98be7d25264fd4b2af69c6871f6b01a2f90f8d7e6d449d4c8ee85a55dd5c1c42`) does not
match the actual `base.py` SHA-256
(`98be7d25ad94930fd9497e6ab33e64970c509c6c3569c8c3df40ae9d22b9a341`). The adapter
and harness hashes match exactly, and the base.py working-tree bytes are identical
to the committed bytes (git blob `f1998e7fe09f798d5562be44dbc6453263a4a871`),
proving no file mutation. This discrepancy affects only the recorded identity
value and the measurement-fingerprint derivation (which hashes `base.py` bytes
plus `auto_bench.py` plus settings); it does not affect the correctness, wall
timing, or profiler evidence, which all run on file contents directly. Flagged to
Orchestrator for correction.

## evidence_for_next_round

- Canonical Phase 0 baseline: `baseline_adapter.py`, wall median `0.353447 ms`
  from three independent 50/100 samples under measurement fingerprint
  `896adb91dbe5f84f9de83644e058462173cd5423a61bdf1ebcb2a15ca783c0be`.
- `baseline_base` scope measured `68.63642578125 us/device-call` and
  `10.86 kernels/call`. Device ratio ≈ `0.194`, so ~80% of wall time is host /
  launch overhead rather than device kernel time.
- The dominant device kernels are two MulFunctor elementwise kernels
  (`17.694 + 7.918 = 25.612 us/call`) and a Cat kernel (`10.954 us/call`);
  sin/cos contribute `7.277 + 7.132 = 14.409 us/call`.
- Base and adapter are semantically equivalent (adapter is a top-level class
  rename); the small wall/device differences are measurement observations, not an
  optimization mechanism.
- The `base_sha256` recorded in `project.md` differs from the actual file hash;
  Orchestrator should reconcile the recorded value before Round 001.

## Stop Recommendation

- recommendation: `continue`
- evidence: Phase 0 baseline is valid; no optional target is configured, and no
  terminal-round limit applies to baseline establishment.

Orchestrator owns canonical pointer updates and workflow transition.

## Exact Reproduction Commands

Environment bootstrap (every command):

```bash
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
```

Frozen-file SHA256 verification (run before and after measurement; all returned code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && sha256sum kernels/track1-triton/music_flamingo_rotary_embedding/base.py kernels/track1-triton/music_flamingo_rotary_embedding/bi150/baseline_adapter.py auto_bench.py
```

Correctness (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/bi150/baseline_adapter.py --warmup 50 --repeat 100 --full-traceback
```

Wall timing (execute independently three times; return codes `0, 0, 0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/bi150/baseline_adapter.py --warmup 50 --repeat 100
```

Forward profiler (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/music_flamingo_rotary_embedding/base.py --v1_file kernels/track1-triton/music_flamingo_rotary_embedding/bi150/baseline_adapter.py --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output kernels/track1-triton/music_flamingo_rotary_embedding/bi150/log/round_000_forward_50iter.pt.trace.json
```

Separately scoped unmodified repository summaries (both returned code `0`):

```bash
python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/music_flamingo_rotary_embedding/bi150/log/round_000_forward_50iter.pt.trace.json --iterations 50 --scope baseline_base --wall-ms 0.353447
```

```bash
python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/music_flamingo_rotary_embedding/bi150/log/round_000_forward_50iter.pt.trace.json --iterations 50 --scope candidate_baseline_adapter --wall-ms 0.350067
```

## Command Status and Raw Evidence Index

| Command | Return code | Evidence |
|---|---:|---|
| frozen-file SHA256 before measurement | `0` | hashes in Identity / Frozen Artifact Hash Note |
| runtime fingerprint check | `0` | torch 2.7.1, triton 3.1.0, BI-V150 (7,1) |
| correctness 50/100 | `0` | round_status_000.md; report Correctness table |
| wall sample 1, 50/100 | `0` | report Interleaved Wall Timing |
| wall sample 2, 50/100 | `0` | report Interleaved Wall Timing |
| wall sample 3, 50/100 | `0` | report Interleaved Wall Timing |
| forward profiler 20/50 | `0` | `log/round_000_forward_50iter.pt.trace.json` |
| summarize `baseline_base` | `0` | report Profiler Evidence |
| summarize `candidate_baseline_adapter` | `0` | report Profiler Evidence |
| frozen-file SHA256 after measurement | `0` | hashes in Identity |

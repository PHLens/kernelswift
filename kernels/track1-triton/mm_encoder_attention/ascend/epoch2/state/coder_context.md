# Coder Context

> Naming contract: the durable role context file is exactly
> `state/designer_context.md`, `state/coder_context.md`, or
> `state/verifier_context.md` — one `*_context.md` per role. No `*_state.md`
> alias exists and no compatibility alias may be created.

- role_contract_sha256: `26c40a94bacbbe5ac4cf12b330516b0439a823e7ca8fd648bdace3fdfcce9cba`
- context_epoch: `2`
- last_completed_round: `003`
- coder_handoff_round: `004` (handoff written, awaiting Verifier)
- accepted_kernel: `triton_mm_encoder_attention_e2_003.py`
- accepted_report: `rounds/report_003.md`
- recent_three_round_evidence: see the table below
- open_hypotheses: see below
- artifact_read_hashes: see the table below

This file holds compact ownership-safe state only. It contains neither
authoritative measurement claims nor a replacement for
`rounds/coder_result_004.md`.

## Current Bottleneck

Verifier-backed facts, from `rounds/report_003.md` Level 2 decomposition against
a `297.410 us` wall:

| Slice | us/call | Share | Reachable by a host round? |
|---|---:|---:|---|
| harness-fixed (outside `ModelNew.forward`) | 91.035 | 30.61% | **no** |
| Triton launch path (bare launch) | 183.740 | 61.78% | `launch-path-reduction` — **proven this round** |
| residual `forward` wrapper | 22.635 | 7.61% | yes, no new capability needed |
| device kernel time | 13.4224 | sub-component of the sync term | bounded at 4.09% |

- Kernel count `1.00`, `device_us_per_call` `13.4224`, `device_ratio` `0.0445`.
- Adoption budget: `0.05 * 297.410 = 14.871 us/call`.
- The `91.035 us` harness-fixed term is a hard floor: even a zero-cost
  `ModelNew.forward` leaves 30.61% of wall standing.

## Recent Three-round Evidence

| Round | Result | Candidate | Change family | Coder outcome |
|---:|---|---|---|---|
| `000` | baseline | `baseline_adapter.py` | not-applicable | Phase 0 baseline, `0.347800` ms |
| `001` | accepted | `triton_mm_encoder_attention_e2_001.py` | kernel / launch-collapse | `candidate-ready`, `+10.2983%` wall |
| `002` | aborted | none | device-only (no viable intervention) | no candidate produced; not a Coder failure |
| `003` | accepted | `triton_mm_encoder_attention_e2_003.py` | host / `allocation-reuse` | `candidate-ready`, `+17.3965%` wall vs `base.py` |
| `004` | handoff | `triton_mm_encoder_attention_e2_004.py` | host / `launch-path-reduction` | `candidate-ready`, capability **proven** |

## Capability Probe Outcome (round 004, decision-scoped)

`lifecycle.fast-launcher` was `Unknown` in the frozen snapshot. The probe
(`log/probes/`) established it on this runtime. All three candidate mechanisms
exist, drive the same compiled kernel, and are bit-identical:

| Mechanism | us/call | vs proven baseline (186.255) |
|---|---:|---:|
| M0 proven `kernel[grid](...)` (reproduced) | 186.255 | — |
| M1 `fast_libentry` — **selected** | 164.225 | −22.030 |
| M2 cached `CompiledKernel` | 66.895 | −119.360 |
| M3 `NPULauncher.launch` C entry | 46.675 | −139.580 |

Decision 004 fixes the order and says stop at the first mechanism satisfying all
four criteria; all three do, so **M1** was implemented. M2/M3 are 5-6x larger and
are recorded for a future decision — Coder does not reorder a normative
mechanism list.

**This evidence is round-local.** The frozen snapshot stays `Unknown` and
hash-pinned; a later round must re-establish legality on its own evidence.

## Open Hypotheses or Checks

- Round 004 is `candidate-ready` with the capability proven. The forward-level
  lever measured in-process is `-18.470 us/call` on a `297.410 us` wall (~6.2%),
  against a `14.871 us` threshold. **Margin is thin; adoption is Verifier's call.**
- If round 004 lands `no-improvement` on margin, the highest-value next step is a
  decision naming **M2 or M3** explicitly. The probe evidence for both already
  exists in `log/probes/`; no new probe is needed.
- Inside the current host family only `~22.6 us/call` remains, and clearing 5%
  needs `14.871 us` of it — about 66%.
- Standing implementation constraints: `import torch_npu` before any NPU
  allocation; `device="npu"` and `torch.npu.synchronize()`; direct launch
  `kernel[(grid,)](...)`; never `import triton_ascend`; never hardcode `"cuda"`.
- **Launch-count interception gotcha:** the active launcher class is the compiled
  C++ `ascend.NPULauncher`, reachable via
  `triton.runtime.driver.active.launcher_cls`. The Python
  `triton.backends.ascend.driver.NPULauncher` is shadowed and patching it yields
  a silent zero count.
- Measurement exclusivity: while Verifier owns `verifying` or `measuring`, Coder
  must stay idle — no local commands, builds, scans, or file edits.
- Coder owns Decision-scoped capability/compile probes; results live under
  campaign-local `log/probes/` and never mutate the frozen profile or the Phase 0
  project claim.

## Local Conformance Checks Completed at Round 004

| Check | Result |
|---|---|
| `validate_decision.py --expected-implementation-profile triton_ascend` | exit `0`, `"valid":true` |
| kernel definition vs `e2_003` (`diff` lines 1-76) and vs `e2_001` (1-74) | exit `0`, byte-identical |
| `ast.parse` / real harness AST loader | ok |
| smoke `--warmup 5 --repeat 10 --full-traceback` | `PASS accuracy`, exit `0` |
| `read_lints` | `totalCount: 0` |
| bit-identity vs `e2_003` | `True`, `max_abs_diff = 0.0` |
| exact device launches per call (e2_003 / e2_004) | `1.00` / `1.00` |
| `state_dict()` after forwards | `[]` — handle and buffer are not module state |
| cache hit reuses buffer; poisoned buffer leaks no NaN | `True` / `False` |
| stride change discards handle and re-proves it | `True`; restored output bit-equal |

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `../../base.py` | `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` | 004 |
| `../../auto_bench.py` | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` | 004 |
| `baseline_adapter.py` | `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e` | 000 |
| `triton_mm_encoder_attention_e2_001.py` | `c75ec5ffaab3883ef7c5b1e62778b39fbd5413619a625fd36a86d70390e92124` | 004 |
| `triton_mm_encoder_attention_e2_003.py` (canonical) | `c39142c1df7d719e9ef7680b4712b226e293f683d934228f930fa966324c6bfe` | 004 |
| `triton_mm_encoder_attention_e2_004.py` (this round's candidate, written) | `f5aa1d709e4deeb1562757d795dd4da41217238dd15c53d33c2c338da1938020` | 004 |
| `rounds/decision_004.md` | `30758ad4dd30ccb0087534e47f61ea0443bdeead40ba64d41c28dd052c397088` | 004 |
| `rounds/sketch_004.json` | `d3e52f6af032014381908e03e87a6b1c3f5694090686df2af3bfe3a6d9474dbf` | 004 |
| `state/implementation_profile_snapshot/profile.yaml` | `a2c3e2e4622fd2d9d2ffd67206912699217279238a14d66a0816cdc188d96321` | 004 |

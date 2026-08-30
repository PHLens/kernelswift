# Coder Context

> Naming contract: the durable role context file is exactly
> `state/designer_context.md`, `state/coder_context.md`, or
> `state/verifier_context.md` — one `*_context.md` per role. No `*_state.md`
> alias exists and no compatibility alias may be created.

- role_contract_sha256: `26c40a94bacbbe5ac4cf12b330516b0439a823e7ca8fd648bdace3fdfcce9cba`
- context_epoch: `2`
- last_completed_round: `004`
- coder_handoff_round: `005` (handoff written, awaiting Verifier)
- accepted_kernel: `triton_mm_encoder_attention_e2_003.py`
- accepted_report: `rounds/report_003.md`
- recent_three_round_evidence: see the table below
- open_hypotheses: see below
- artifact_read_hashes: see the table below

This file holds compact ownership-safe state only. It contains neither
authoritative measurement claims nor a replacement for
`rounds/coder_result_005.md`.

## Current Bottleneck

Verifier-backed facts, from `rounds/report_003.md` Level 2 against a
`297.410 us` wall:

| Slice | us/call | Share | Reachable by a host round? |
|---|---:|---:|---|
| harness-fixed (outside `ModelNew.forward`) | 91.035 | 30.61% | **no** — hard floor |
| Triton launch path (bare launch) | 183.740 | 61.78% | `launch-path-reduction` — **proven** |
| residual `forward` wrapper | 22.635 | 7.61% | yes, no new capability needed |
| device kernel time | 13.4224 | sub-component of the sync term | bounded at 4.09% |

- Kernel count `1.00`, `device_us_per_call` `13.4224`, `device_ratio` `0.0445`.
- Adoption budget: `0.05 * 297.410 = 14.871 us/call`.
- **The wall conversion is lossy at roughly 75%.** Verifier's round-004
  finding: forward time fell `-18.965 us` while wall fell `-14.330 us`. Predict
  wall gain from forward gain at ~75%, not 100%.

## Recent Three-round Evidence

| Round | Result | Candidate | Change family | Coder outcome |
|---:|---|---|---|---|
| `001` | accepted | `triton_mm_encoder_attention_e2_001.py` | kernel / launch-collapse | `candidate-ready`, `+10.2983%` wall |
| `002` | aborted | none | device-only (no viable intervention) | no candidate; device ceiling 4.09% < 5% |
| `003` | accepted | `triton_mm_encoder_attention_e2_003.py` | host / `allocation-reuse` | `candidate-ready`, `+17.3965%` wall vs `base.py` |
| `004` | no-improvement | `triton_mm_encoder_attention_e2_004.py` | host / `launch-path-reduction` (M1) | `candidate-ready`; capability **proven**; magnitude `+2.8874%` vs `+5%` bar |
| `005` | handoff | `triton_mm_encoder_attention_e2_005.py` | host / `launch-path-reduction` (M2) | `candidate-ready`, capability proven by citation |

## Mechanism Table (round-004 probe, round-005 refinement)

`lifecycle.fast-launcher` is proven on this runtime. All mechanisms drive the
same `CompiledKernel` (hash `18db9f0320830a397f740d02078551aeea898355fd7e06d59bb3a7bca2e1c903`),
are bit-identical, and launch exactly once per call.

| Mechanism | us/call | saving vs M0 | status |
|---|---:|---:|---|
| M0 proven `kernel[grid](...)` | 173.5-186.3 (in-process) | — | control |
| M1 `fast_libentry` | 164.225 | 22.030 | tried in round 004, sub-threshold |
| M2a cached `CompiledKernel`, stream cached | 63.065-64.670 | 110-117 | faster, but contradicts the Host Plan stream clause |
| **M2b cached `CompiledKernel`, stream per call — SHIPPED** | **85.770-88.720** | **87.8-93.4** | round 005 |
| M3 `NPULauncher.launch` C entry | 46.675 | 139.580 | rejected as dominated (decision 005 §3) |

**Per-call stream resolution costs 22-24 us** — more than M1's entire lever.
It is paid for Host Plan conformance.

## Open Hypotheses or Checks

- Round 005's forward lever is **`-90.955 us/call` (-41.42%)**. Even at the
  lossy 75% wall conversion this implies far more than the `14.871 us`
  threshold. The risk is not magnitude but drift: this machine moved `5-7%`
  within a turn in rounds 003 and 004.
- **Unclaimed magnitude on the table:** recovering the `22-24 us` stream cost
  needs a decision amending `device_stream_behavior`; M3's further `20 us`
  needs a decision that answers decision 005 §3's coupling objection. Both
  measurements already exist under `log/probes/`.
- Inside the current host family only `~22.6 us/call` of wrapper remains.
- **Adoption bar is now stated as:** `speedup(candidate) / speedup(last accepted) - 1 >= 5%`,
  measured by strict pair-by-pair alternation in one window. Only that form
  cancels drift on this machine; cross-window speedup comparisons moved
  `4.31%` on their own in round 004.
- Standing constraints: `import torch_npu` before any NPU allocation;
  `device="npu"` and `torch.npu.synchronize()`; direct launch
  `kernel[(grid,)](...)`; never `import triton_ascend`; never hardcode `"cuda"`.
- **Launch-count interception gotcha:** the active launcher class is the
  compiled C++ `ascend.NPULauncher`, reachable via
  `triton.runtime.driver.active.launcher_cls`. The Python
  `triton.backends.ascend.driver.NPULauncher` is shadowed; patching it counts
  zero silently.
- Measurement exclusivity: while Verifier owns `verifying` or `measuring`,
  Coder must stay idle — no local commands, builds, scans, or file edits.
- Coder owns Decision-scoped capability/compile probes; results live under
  campaign-local `log/probes/` and never mutate the frozen profile or the
  Phase 0 project claim.

## Local Conformance Checks Completed at Round 005

| Check | Result |
|---|---|
| `validate_decision.py --expected-implementation-profile triton_ascend` | exit `0`, `"valid":true` |
| kernel definition vs `e2_003` (1-76) and `e2_001` (1-74) | exit `0`, byte-identical |
| no `LibEntry` / `libentry` anywhere in the candidate | confirmed by `grep` |
| `ast.parse` / real harness AST loader / public signatures | ok, identical to `e2_003` |
| smoke `--warmup 5 --repeat 10 --full-traceback` | `PASS accuracy`, exit `0` (twice) |
| `read_lints` | `totalCount: 0` |
| bit-identity vs `e2_003` | `True`, `max_abs_diff = 0.0` |
| exact device launches per call (e2_003 / e2_005) | `1.00` / `1.00` |
| `state_dict()` after forwards | `[]` |
| cache hit reuses buffer; poisoned buffer leaks no NaN | `True` / `False` |
| `S` change and stride change each re-prove, not disabled | `True` / `True` |
| identity mismatch → disabled, correct output, handle cleared | `True` ×3 |
| forced exception → disabled, correct output, handle cleared | `True` ×3 |

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `../../base.py` | `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` | 005 |
| `../../auto_bench.py` | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` | 005 |
| `baseline_adapter.py` | `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e` | 000 |
| `triton_mm_encoder_attention_e2_001.py` | `c75ec5ffaab3883ef7c5b1e62778b39fbd5413619a625fd36a86d70390e92124` | 005 |
| `triton_mm_encoder_attention_e2_003.py` (canonical) | `c39142c1df7d719e9ef7680b4712b226e293f683d934228f930fa966324c6bfe` | 005 |
| `triton_mm_encoder_attention_e2_004.py` | `f5aa1d709e4deeb1562757d795dd4da41217238dd15c53d33c2c338da1938020` | 005 |
| `triton_mm_encoder_attention_e2_005.py` (this round's candidate, written) | `bf54cea2a1fcdafd8916c2e0bf607766a6e7ffc2981fd956e18e92bf51b88b26` | 005 |
| `rounds/decision_005.md` | `1fdd16d7ddca961760260b9e6130c7e6d2fb17b689728474ee9e5bea9b8ce551` | 005 |
| `rounds/sketch_005.json` | `f44ed2bfbef80e9dc603494221bbc2cd47db40a9d8d48d85ee2ae344cd11c4ee` | 005 |
| `state/implementation_profile_snapshot/profile.yaml` | `a2c3e2e4622fd2d9d2ffd67206912699217279238a14d66a0816cdc188d96321` | 005 |

# Coder Context

> Naming contract: the durable role context file is exactly
> `state/designer_context.md`, `state/coder_context.md`, or
> `state/verifier_context.md` — one `*_context.md` per role. No `*_state.md`
> alias exists and no compatibility alias may be created.

- role_contract_sha256: `26c40a94bacbbe5ac4cf12b330516b0439a823e7ca8fd648bdace3fdfcce9cba`
- context_epoch: `2`
- last_completed_round: `002`
- coder_handoff_round: `003` (handoff written, awaiting Verifier)
- accepted_kernel: `triton_mm_encoder_attention_e2_001.py`
- accepted_report: `rounds/report_001.md`
- recent_three_round_evidence: see the table below
- open_hypotheses: see below
- artifact_read_hashes: see the table below

This file holds compact ownership-safe state only. It contains neither
authoritative measurement claims nor a replacement for
`rounds/coder_result_003.md`.

## Current Bottleneck

Verifier-backed facts only, from `rounds/report_001.md`:

- `device_us_per_call` is `13.4064` at `kernel_count_per_call` `1.00`;
  `device_ratio` is `0.0407`. The candidate is host-bound.
- The non-device residual is about `316 us/call`
  (`329.365 - 13.4064`). Its split between the harness synchronize, the Triton
  launch path, and the output allocation is **not measured**; no round so far
  has decomposed it.
- Round 002 closed the device side by arithmetic: a `5%` adoption budget of
  `16.3885 us/call` against a complete device budget of `13.4064 us/call` caps
  device-only improvement at `4.0902%`.
- Round 003 therefore attacks host work only, under the maintainer's
  `host_code` policy revision (commit `de1b9b7`).

## Recent Three-round Evidence

| Round | Result | Candidate | Change family | Coder outcome |
|---:|---|---|---|---|
| `000` | baseline | `baseline_adapter.py` | not-applicable | Phase 0 baseline, `0.347800` ms |
| `001` | accepted | `triton_mm_encoder_attention_e2_001.py` | kernel / launch-collapse | `candidate-ready`, `+10.2983%` wall |
| `002` | aborted | none | device-only (no viable intervention) | no candidate produced; not a Coder failure |
| `003` | handoff | `triton_mm_encoder_attention_e2_003.py` | host / `allocation-reuse` | `candidate-ready` (this handoff) |

## Open Hypotheses or Checks

- Round 003 is `candidate-ready`. Its single attributable cause is the per-call
  output allocation; the fused kernel body is byte-identical to the accepted
  source, so `device_us_per_call` and `kernel_count_per_call` must stay at
  `13.4064` and `1.00`. Any device movement breaks attribution.
- If round 003 returns `no-improvement`, the next authorized family is
  `launch-path-reduction`. Per the decision, that must be preceded by an Ascend
  probe of the launch ABI: `lifecycle.fast-launcher` is `unknown` and declaring
  it normative would turn a performance round into a `capability-miss`. Coder
  must not attempt it as a code experiment.
- Standing implementation constraints on this runtime: `import torch_npu`
  before any NPU allocation; `device="npu"` and `torch.npu.synchronize()`;
  direct launch `kernel[(grid,)](...)`; never `import triton_ascend`; never
  hardcode `"cuda"` in `get_inputs`.
- Measurement exclusivity: while Verifier owns `verifying` or `measuring`,
  Coder must stay idle — no local commands, builds, scans, or file edits.
- Coder owns the compile/capability probes required to establish source
  conformance; results live under campaign-local `log/probes/` and never mutate
  the frozen profile or the Phase 0 project claim.

## Local Conformance Checks Completed at Round 003

| Check | Result |
|---|---|
| `validate_decision.py --expected-implementation-profile triton_ascend` | exit `0`, `"valid":true` |
| kernel body vs canonical, `diff` lines 1-74 | exit `0`, byte-identical |
| launch sites in candidate | exactly 1 |
| `ast.parse` | ok |
| real harness AST loader | ok, `get_init_inputs() == [8, 64, 8]` |
| smoke `--warmup 5 --repeat 10 --full-traceback` | `PASS accuracy`, exit `0` |
| `state_dict()` after forwards | `{}` — buffer is not module state |
| cache hit reuses buffer / miss reallocates | both observed |
| poisoned-buffer overwrite test | no stale value escapes |
| aliasing of query / key / value | none |

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `../../base.py` | `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` | 003 |
| `../../auto_bench.py` | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` | 003 |
| `baseline_adapter.py` | `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e` | 000 |
| `triton_mm_encoder_attention_e2_001.py` (canonical) | `c75ec5ffaab3883ef7c5b1e62778b39fbd5413619a625fd36a86d70390e92124` | 003 |
| `triton_mm_encoder_attention_e2_003.py` (this round's candidate, written) | `c39142c1df7d719e9ef7680b4712b226e293f683d934228f930fa966324c6bfe` | 003 |
| `rounds/decision_003.md` | `a4956891de5fef4b9bd629fb3cceb270db5a247ba18b591aecee9480d96c5455` | 003 |
| `rounds/sketch_003.json` | `51ebe3a735c7659309e781fd2f35286fd4e67acc86b5d0a9f6676f08f08af69c` | 003 |
| `state/implementation_profile_snapshot/profile.yaml` | `a2c3e2e4622fd2d9d2ffd67206912699217279238a14d66a0816cdc188d96321` | 003 |

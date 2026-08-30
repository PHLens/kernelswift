# Coder Context

> Naming contract: the durable role context file is exactly
> `state/designer_context.md`, `state/coder_context.md`, or
> `state/verifier_context.md` — one `*_context.md` per role. No `*_state.md`
> alias exists and no compatibility alias may be created.

- role_contract_sha256: `26c40a94bacbbe5ac4cf12b330516b0439a823e7ca8fd648bdace3fdfcce9cba`
- context_epoch: `2`
- last_completed_round: `000`
- accepted_kernel: `baseline_adapter.py`
- accepted_report: `rounds/report_000.md`
- recent_three_round_evidence: `Phase 0 baseline only; no candidate has been coded in this epoch.`
- open_hypotheses: `awaiting the round-001 decision from Designer.`
- artifact_read_hashes: `see the table below`

## Current Bottleneck

Verifier-backed facts only, from `rounds/report_000.md`:

- Device time is about one third of wall time; host launch and synchronization
  dominate, with `6.96-6.98` kernels per call.
- Materialized layout conversions cost `63.35 us/call` of device time versus
  `23.0232 us/call` for the FlashAttentionScore kernel itself.

## Recent Three-round Evidence

- `000` / `baseline` / `rounds/report_000.md` / change family `not-applicable`:
  baseline established at `0.349625` ms reference and `0.347800` ms candidate.

## Open Hypotheses or Checks

- No decision exists yet. Coder must not act until Orchestrator supplies
  `rounds/decision_001.md` with a validated schema-v2 Decision.
- Implementation constraints discovered at Phase 0: `import torch_npu` before any
  NPU allocation; use `device="npu"` and `torch.npu.synchronize()`; direct launch
  `kernel[(grid,)](...)`; never `import triton_ascend`; never hardcode `"cuda"` in
  `get_inputs`.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `../../base.py` | `86ac570376eda42cea73e0e7683454deeff43c11e5e85f16e1b3eb63395d6ed2` | 000 |
| `baseline_adapter.py` | `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e` | 000 |
| `state/implementation_profile_snapshot/profile.yaml` | `a2c3e2e4622fd2d9d2ffd67206912699217279238a14d66a0816cdc188d96321` | 000 |

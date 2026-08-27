# Verifier Context

- role_contract_sha256: `62f10a0940ca3665260226a7891f5d34e1b571e70937862bb02ad68aa2bbc82f`
- context_epoch: `1`
- last_completed_round: `000`
- accepted_kernel: `baseline_adapter.py` (sha256 `ecce4dacee211a86ba38584b6b78fc2f575ba60cedccdc6f79ac4f6fb0139fa5`)
- accepted_report: `rounds/report_000.md` (Result: baseline, written 2026-08-27)
- recent_three_round_evidence: `Round 000 Phase-0 baseline only — no candidate rounds yet; run epoch 2 of the groupedtopk@bi150 lineage (v2 campaign at ../bi150 is historical read-only)`
- open_hypotheses: `none yet`
- artifact_read_hashes:
  - `../base.py`: `12f3324896d6b72bd4eac556839db53fb7045b2965b7f8caf2734551daad0f58` (3541 bytes, immutable, unchanged)
  - harness `/root/CodeBuddy/20260818191200/kernelswift/auto_bench.py`: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (29428 bytes, unchanged)
  - `baseline_adapter.py`: `ecce4dacee211a86ba38584b6b78fc2f575ba60cedccdc6f79ac4f6fb0139fa5`
  - `project.md`: `9b2483b34789463aa0afe7f86ef2475a2d7926cd654a90b41cbfbf58d3a66821`
  - `team-state.md`: read-only (Orchestrator-owned), manifest epoch 2, phase initializing→(awaiting transition on this report)
  - `state/designer_context.md`, `state/coder_context.md`: read-only, untouched
  - trace `log/groupedtopk_baseline_forward_100iter.pt.trace.json`: `666c9d2fb8db86eb0cab7f39f52020107fb7f597cccd3e0e40c7542599275228`

## Round 000 Measurement Facts (authoritative)

- Correctness: base.py vs baseline_adapter.py PASS ×4 invocations (harness comparator: fp allclose atol/rtol 1e-2, int exact, seed 42).
- Wall medians (--warmup 50 --repeat 100): reference `0.483530 ms`, candidate `0.481109 ms` (~1.00x identity; three ordered pairs `0.484525/0.481109`, `0.483530/0.482140`, `0.452363/0.451582`).
- Device time (forward-scope traces, profile_warmup 20 / iterations 100): reference `180.114755859375 us/call`, candidate `178.84361328125 us/call`; kernels `14.94/call` both; device_ratio ≈ `0.372/0.372`.
- Wall is host-dominated: ≥62% of wall per call is outside kernel execution (device_ratio ~0.37) at shape [83,256]/[83,7168].
- Kernel structure (both scopes identical, 13 distinct kernels): `sbtopk::gatherTopK` ~49.3 us/call + `bitonicSortKVInPlace` ~37.0–37.2 us/call dominate; two reduce kernels (MaxOps group-max ~17.9–18.1, sum ~15.2–15.3); softmax warp, scatter fill, masked_fill, bitwise_not, div, copies, fill make up the rest.
- Environment verified live: CoreX 4.4.0 bootstrap required per shell; runtime fingerprint matches project.md exactly (triton 3.1.0, torch 2.7.1, nvcc V10.2.89, BI-V150 sm_71 16 SMs).

## Current Bottleneck

- `Wall time on bi150 small-shape grouped-topk is dominated by fixed host overhead (~303 us/call outside kernels: ~66 µs/device sync pair lineage fact + launch gaps); device time floor for the unfused torch pipeline is ~180 us/call dominated by dual top-k machinery (gatherTopK + bitonic sort ≈ 86.5 us/call).`

## Recent Three-round Evidence

- `historical ../bi150/rounds/report_000..009 — final accepted torch.compile(reduce-overhead) host-launch path; device fused path never attempted`
- `epoch-2 round 000 — baseline evidence refreshed under contract v3 regime (this campaign's canonical fingerprints verified)`

## Open Hypotheses or Checks

- `(empty at Phase 0)`
- Check carried forward: `profile_mode=kernel requires ModelNew.run_out on candidates; forward-mode scoping is the only available dual-scope profiler path for torch-shaped adapters (exact KsCompareError recorded in rounds/report_000.md)`

# Grouped TopK S60 Campaign Final Summary

- stop_reason: `user-intervention`
- stopped_at: `2026-08-18T05:21:33Z`
- run_branch: `kernel-opt/groupedtopk-s60-continue`
- base_commit: `6a970c9`
- total_terminal_rounds: `8`
- accepted_round: `003`
- canonical: `triton_grouped_topk_003.py`
- accepted_report: `rounds/report_003.md`
- measurement_fingerprint: `3942e25aebbe7690a55cf27768a3bc3fd552cc8106f6bd2dd7416cea2d274bf3`

## Accepted Progress

- Round 001: kernel fusion, `39.08693002628853%` wall improvement; runtime launches reduced from `12.0` to `1.0` per call.
- Round 002: instance-private output-pool reuse, `9.02136875254568%` wall improvement.
- Round 003: exact-key host metadata specialization, `6.464721724746064%` wall improvement; canonical advanced to `triton_grouped_topk_003.py`.

## Continued Evidence

- Round 004: launcher-context specialization was correct and preserved all lifecycle/device/stream guardrails, but improved wall median only `2.058982586436897%`; canonical was unchanged.
- Rounds 005-007: aborted because no distinct candidate-owned intervention had a defensible >=5% path under the available evidence.
- Round 008 named S60 probe: unchanged correctness passed with reference/canonical raw pair `0.282114/0.282032 ms`; both profile scopes emitted `1.0` `topsModuleLaunchKernel` per call.
- Round 008 trace SHA-256: `1c04a827a50cbb065c1c9943e7c0f5ddf961aeca7f27c06aa2e912f5d2b1a7ec`.
- GCU device duration remains unavailable: the trace contains runtime launch events but no `cat=kernel` events. Runtime-launch duration was retained as diagnostic evidence and never treated as device time.

## Final State

No candidate source was promoted after Round 003. The accepted canonical remains unchanged. The remote S60 execution copy was verified against local `base.py`, `auto_bench.py`, canonical Round 003, and Round 004 reference/candidate hashes. Raw profiler logs remain gitignored; reports retain their hashes and reproduction commands.

## Reconsideration Conditions

A future run should first obtain a matched GCU device-duration exporter or same-runtime microbenchmark that identifies a candidate-owned bottleneck and proves supported lowering/tie behavior. A new launcher/context experiment requires a distinct candidate-owned mechanism from Round 004. No changes were made to shared anti-pattern references.

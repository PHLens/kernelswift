# Coder Context

- role_contract_sha256: `26c40a94bacbbe5ac4cf12b330516b0439a823e7ca8fd648bdace3fdfcce9cba`
- context_epoch: `2`
- last_completed_round: `001` (coding phase; result returned, canonical pointers unchanged pending Verifier)
- accepted_kernel: `baseline_adapter.py`
- accepted_report: `rounds/report_000.md`
- recent_three_round_evidence: `Phase 0 baseline (report_000): wall v0=0.483530ms v1=0.481109ms ~1.00x identity; 14.94 kernels/call; device_ratio 0.372 host-dominated; kernel-mode profiling requires ModelNew.run_out`
- open_hypotheses: `H-001 preprocess-fusion-triton-stages implemented in candidate triton_grouped_topk_r2_001.py (sha256 4ae64cad913267f2198fec735e08f1b9490cafa1139d3a48ee11400aacb80de3); awaiting Verifier measurement under fingerprint 8deb1b01...`
- artifact_read_hashes: decision_001=93783baa…3532b; sketch_001=637917e0…f6985; baseline_adapter=ecce4dac…39fa5; profile_snapshot/triton_cuda.yaml=dc8fa4c0…b7ae; base=12f33248…d0f58; harness=71fb3ad0…fe29

## Current Bottleneck

- `Verifier-backed facts carried from historical epoch 1: wall time on bi150 small-shape operators is dominated by fixed host overhead (~66 µs/device sync pair); device time after full fusion reaches single-kernel floor.`

## Recent Three-round Evidence

- `historical ../bi150/rounds/report_000..009 — final accepted torch.compile(reduce-overhead) host-launch path; device fused path never attempted`

## Round 001 Coding State

- classification written: `candidate-ready` → `rounds/coder_result_001.md` (no major deviation, no capability miss)
- candidate implements Decision-001 family `preprocess-fusion-triton-stages`: stage-A softmax+group-max kernel, stage-B arithmetic group-membership mask via tl.where, stage-C renorm+scale+kernel-side int64→int32 narrowing, BOTH retained exact torch.topk sites verified by AST (`binding_statement_report.json` @5fbddd0d…11d0), run_out preallocated-output surface present and live-checked
- probes (all under log/probes/, no timing/profiler usage):
  - cast_narrow probe result @8f225a3eb8c881faee8bad5f2e6ad4232c31c2bfe48a694bfc136d332af07f50 — evidence-ready; cast.narrow.int64-to-int32-kernel-side OBSERVED (host-side narrowing fallback NOT needed); memory.load.int64-scalar-membermask OBSERVED
  - coder smoke result @09cd56ba499dbff02acd91687cf6cbf7f258b6657e71bd2807e731c2b5daf066 — all_pass=true: ast-parse, real-harness-loader, warmup compile smoke, correctness vs base incl. all-equal/two-expert-tie/structured-group/duplicate-max suites (ids exact), run_out byte-equality + no-cross-call-caching
- open local checks: none outstanding; num_warps/num_stages deliberately unset profile-wide; tl.argmax unused
- same-round repair budget: untouched (0 of 1 used)

## Open Hypotheses or Checks

- H-001 awaiting authoritative Verifier wall comparison (expected ≥5% unrounded paired median improvement; expected-gain prior 8%)

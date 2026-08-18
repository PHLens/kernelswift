# Round Status 002

- round: `002`
- phase: `complete`
- result: `accepted`
- ended_at: `2026-08-18`

## Progress

- [x] Role bootstrap: read verifier contract, runtime adapter, decision_002, coder_result_002, project.md, report_001, report_000, team-state, verifier_context
- [x] Artifact hash verification: candidate `9315412c...` matches coder_result_002; reference `b7b47d1f...`, harness `71fb3ad0...`, base `12f33248...`, decision `a3b8aebf...` all match records
- [x] Runtime fingerprint confirmed: Python 3.11.15, torch 2.7.1+cpu, torch_npu 2.7.1.post4, triton 3.2.0, npu available
- [x] Correctness gate (auto_bench, warmup 5 / repeat 10): `PASS accuracy`
- [x] Interleaved wall timing (warmup 50, repeat 100, 3 ordered pairs): reference median `0.326705` ms, candidate median `0.267220` ms, improvement `+18.21%`
- [x] Profiler evidence (targeted, forward mode, 50 iterations, dual CANN scope): device `34.962` -> `35.134` us/call (flat, 1.0 kernel/call); host time `291.743` -> `232.086` us/call (-20.45%)
- [x] H-002 Evaluation Contract mirror: confirmed (wall_time +18.21% >= 5.0; output_allocations_per_call 2.0 -> 0.0 steady state; host_us_per_call -20.45%)
- [x] report_002.md written (`accepted`)
- [x] round_status_002.md final

## Commands Completed

```bash
cd /workspace/kernelswift/.worktrees/groupedtopk-ascend
sha256sum kernels/track1-triton/groupedtopk/ascend/triton_grouped_topk_002.py kernels/track1-triton/groupedtopk/ascend/triton_grouped_topk_001.py auto_bench.py kernels/track1-triton/groupedtopk/base.py kernels/track1-triton/groupedtopk/ascend/baseline_adapter.py
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/ascend/triton_grouped_topk_002.py --warmup 5 --repeat 10 --full-traceback
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/ascend/triton_grouped_topk_001.py --warmup 50 --repeat 100  # x3 interleaved
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/ascend/triton_grouped_topk_002.py --warmup 50 --repeat 100  # x3 interleaved
python3 auto_bench.py --v0_file kernels/track1-triton/groupedtopk/base.py --v1_file kernels/track1-triton/groupedtopk/ascend/triton_grouped_topk_002.py --warmup 50 --repeat 100 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-reference-file kernels/track1-triton/groupedtopk/ascend/triton_grouped_topk_001.py --profile-output kernels/track1-triton/groupedtopk/ascend/log/groupedtopk_round_002_forward_50iter.pt.trace.json
python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py .../reference_triton_grouped_topk_001/profiling_data --iterations 50 --scope reference_triton_grouped_topk_001 --wall-ms 0.326705
python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py .../candidate_triton_grouped_topk_002/profiling_data --iterations 50 --scope candidate_triton_grouped_topk_002 --wall-ms 0.267220
```

## Key Evidence

- Correctness: `PASS accuracy` (both gates)
- Wall medians: reference `0.326705` ms vs candidate `0.267220` ms
- improvement_pct: `18.2076` (>= 5.0 threshold)
- Device: `34.962` -> `35.134` us/call, 1.0 kernel/call (host-only change, device flat)
- Host time: `291.743` -> `232.086` us/call (-20.45%)

## Next Safe Action

None required. Report delivered to team lead via runtime adapter.

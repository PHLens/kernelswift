# Verifier Context — sparse_pooler (BI150)

## Role

Verifier: sole authoritative runtime owner for this campaign. Executes the
project and records attributable correctness, benchmark, and profiler evidence.
Classifies outcomes; Orchestrator alone applies state transitions.

## Current State (Round 001 complete)

- Result: `accepted`
- Candidate: `triton_sparse_pooler_001.py`
- Candidate SHA256: `f3fd85a2c913d477e2cac7f65ed1f79dd5e1b9a3a60481782dbb4acaa43d2d98`
- Decision: `rounds/decision_001.md` SHA256 `0fbbdb6929e1b75f939fc2d513c28878b7a53587f33e8fcaf66401f1269256f1`
- improvement_pct: `16.990` (above 5% threshold)
- Hypothesis verdict: `confirmed`

## Round 001 Timings

- reference_median_ms: `1.060573` (baseline_adapter)
- candidate_median_ms: `0.880377`
- reference device_us_per_call: `743.797`; kernel_count_per_call `11.92`
- candidate device_us_per_call: `609.397`; kernel_count_per_call `6.88`
- candidate device_ratio: `0.692`

## Key Findings (evidence_for_next_round)

- Activation + pooling fusion confirmed: 6 tail kernels (clamp_scalar + log1p + 4× reduce_kernel<MaxOps>) → 1 `_sparse_pooler_fused_kernel` at 28.34 us/call.
- Additional win: candidate removed baseline's `seq_lens.tolist()` D2H sync (50× Memcpy DtoH + cudaStreamSynchronize per profile run → 0).
- Remaining bottleneck: `gemm_tcu_h` + `GEMM_Epilogue` ≈ 563.8 us/call ≈ 92.5% of device time (dense 768×768 + decoder 768×30522 on vendor TCU).
- fp32 large-N `tl.dot` rewrite of the GEMMs remains unproven (only (32,32)@(32,32) recorded) — high capability-miss risk, needs matched local probe first.

## Profiler Tooling Note

Triton's `cuLaunchKernel` instrumentation emits a nested duplicate
`record_function` for the candidate scope, tripping `summarize_trace.py`'s
overlap guard. Summarize the candidate scope against the outer enclosing
interval (the event without `finished: True`). This is a measurement quirk, not
a kernel defect.

## Key Frozen Hashes

- base_sha256: `46106baa46b9f1952b32388d99880c967438ea99becdc874e3f68ba81a727d58`
- baseline_adapter_sha256: `359f4c808a0cf210416116322e4cc01f74ee42961b68c1fd365672af2a59bde8`
- harness_sha256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- measurement_fingerprint: `72be9562432197795bf6a24300483ccb2c3219b804b73258611048014cd804a9`

## Operator Semantics

- `sparse_pooler` (SPLADE): dense(768×768) → GELU → LayerNorm → decoder(768×30522) → log1p(relu) → per-sequence max pooling (4× loop).
- inputs: `hidden_states[83,768]` fp32, `seq_lens[4]` int32 `[20,25,18,20]`.
- output: list of 4 tensors each `[30522]` fp32.
- compare: `torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True)` per list element.

## Environment Bootstrap

Every command must run:
```bash
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
```

## Ownership Notes

Verifier may write: `rounds/report_NNN.md`, `rounds/round_status_NNN.md`,
`rounds/incident_*.md`, `state/verifier_context.md`, `log/` raw traces.
Verifier must NOT edit: candidate source, decision_NNN.md, team-state.md,
base.py, harness, coder results, canonical pointers, counters.

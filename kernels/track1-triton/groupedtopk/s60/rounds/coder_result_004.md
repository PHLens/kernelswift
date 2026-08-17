# Coder Result 004

Result: `candidate-ready`

- round: `004`
- source_canonical: `triton_grouped_topk_003.py`
- source_canonical_sha256: `3aad6be6422ff08aeb0c6e6be0c92d0645588c7b93429809c525f55c6a6b3e37`
- reference_adapter: `reference_triton_grouped_topk_003.py`
- reference_adapter_sha256: `9977aaf9ec96c851be33f2582e6284451fd41686a1acc4607deb4e104dca5ea7`
- decision: `rounds/decision_004.md`
- decision_sha256: `a126c9abc86da11734be828bc6c5900e0b1107ba07ecbfa079fc4f74d1416713`
- candidate: `triton_grouped_topk_004.py`
- candidate_sha256: `ea9be7896731f7f371f9ba087c8d01daca6556c66a3e50b2c6146fe6de118bb0`
- selected_profile: `triton_gcu`
- runtime_fingerprint: `project.md#runtime-fingerprint`
- measurement_fingerprint: `3942e25aebbe7690a55cf27768a3bc3fd552cc8106f6bd2dd7416cea2d274bf3`
- adoption: `not-adopted; wall improvement below threshold`

## Implementation Conformance

- The candidate samples the caller-selected GCU device/current-stream identity
  once in its own `_stream_snapshot` path and passes that immutable snapshot to
  both metadata lookup and output-pool lookup.
- A stack trace on S60 confirmed the second observed `current_stream` call is
  inside `triton_gcu` direct-launch backend internals, not candidate code.
- Kernel body, grid, constexprs, `num_warps=1`, metadata cache, output pool,
  device, and stream semantics remain unchanged.

## Guardrail Evidence

| Check | Result | Evidence |
|---|---|---|
| candidate explicit stream snapshot | PASS | S60 stack trace: one candidate `_stream_snapshot` call per forward |
| backend internal stream lookup | observed | S60 stack trace at `triton_gcu/triton/backend.py`; outside candidate change scope |
| exact-key metadata cache | PASS | Same-key hit, shape miss, and original-key hit checks |
| retained output lifetime | PASS | retained output distinct and stable |
| concurrent output-pool safety | PASS | concurrent calls received distinct storage |
| correctness against accepted reference | PASS | S60 smoke, three formal paired runs, and profile run |

## Attempt Ledger

| Attempt | Command / change | Exit status | Defect | Candidate before SHA256 | Candidate after SHA256 |
|---:|---|---:|---|---|---|
| 1 | `python3 -m py_compile s60/groupedtopk/triton_grouped_topk_004.py` | 0 | none | `ea9be7896731f7f371f9ba087c8d01daca6556c66a3e50b2c6146fe6de118bb0` | `ea9be7896731f7f371f9ba087c8d01daca6556c66a3e50b2c6146fe6de118bb0` |
| 2 | S60 stream snapshot, metadata, lifecycle, and concurrency checks | 0 | none; candidate guardrails PASS | `ea9be7896731f7f371f9ba087c8d01daca6556c66a3e50b2c6146fe6de118bb0` | `ea9be7896731f7f371f9ba087c8d01daca6556c66a3e50b2c6146fe6de118bb0` |
| 3 | S60 smoke, three paired benchmarks, and paired profile | 0 | none; correctness PASS, performance below threshold | `ea9be7896731f7f371f9ba087c8d01daca6556c66a3e50b2c6146fe6de118bb0` | `ea9be7896731f7f371f9ba087c8d01daca6556c66a3e50b2c6146fe6de118bb0` |

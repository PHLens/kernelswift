# Round Status 004

## Verification start

- phase: `verifying`
- verifier: sole runtime owner (`measurement_exclusive=true`)
- candidate: `triton_sparse_pooler_004.py` (SHA-256 `81cdea2b958c288e1382aef0b30cfc6dffb544c55a0e44825fab51b53cac7842`)
- accepted reference: `triton_sparse_pooler_001.py` (SHA-256 `182f2ebb32b9762f5f19dae00a30e25618bcdb0b8563f7acbc3d9ea9ca348dcd`)
- decision: `rounds/decision_004.md` (SHA-256 `dc33e45ee2c95319608bc08f9ed8a5a3e3ae0882305f52eb07f0a449ea33f111`)
- base SHA-256: `ccccbbefadf1d697341451b542f17392acc8a2b9e4a3a41e50b2f9d58dbf61de`
- harness SHA-256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- measurement fingerprint: `a0208c7da7e371d45c88f82ebddd3850d01669aa5d912f31db9234a7a56ebab7`
- runtime fingerprint: triton 3.2.0, torch_mlu 1.32.0+torch2.11.0, MLU driver 6.5.49, MLU590-H8 capability (5, 0)
- change_scope: `host`
- change_family: `host-allocation-reuse`
- coder_result: `rounds/coder_result_004.md` (candidate-ready; smoke v0=0.900ms v1=0.569ms speedup=1.580x)

## Pre-flight checks

- artifact hashes match `coder_result_004.md` and `project.md`
  - candidate SHA-256: `81cdea2b958c288e1382aef0b30cfc6dffb544c55a0e44825fab51b53cac7842` (matches coder_result_004.md)
  - accepted reference SHA-256: `182f2ebb32b9762f5f19dae00a30e25618bcdb0b8563f7acbc3d9ea9ca348dcd` (matches team-state.md last_accepted_kernel)
  - decision SHA-256: `dc33e45ee2c95319608bc08f9ed8a5a3e3ae0882305f52eb07f0a449ea33f111` (matches coder_result_004.md)
- runtime fingerprint matches `project.md#runtime-fingerprint`:
  - triton 3.2.0, torch_mlu 1.32.0+torch2.11.0, MLU driver 6.5.49, MLU590-H8 capability (5, 0)
- interpreter `/projs/framework/lipenghui/venv/pytorch_main/bin/python3` (python3 on PATH) has triton 3.2.0, torch_mlu 1.32.0+torch2.11.0, MLU590-H8 cap 5.0
- accepted reference is `triton_sparse_pooler_001.py` (Round 001 canonical); Rounds 002 and 003 rejected candidates are never the comparison source
- Verifier owns: `rounds/report_004.md`, `rounds/round_status_004.md`, `state/verifier_state.md`, `rounds/incident_004_<ts>.md` if needed, raw profiler output under `log/`
- Verifier does NOT edit: candidate source, decision_004.md, team-state.md, project.md, base.py, the harness, coder_result_004.md, canonical pointers

## Safe-step plan

1. Correctness gate: `python3 auto_bench.py --v0_file sparse_pooler/base.py --v1_file sparse_pooler/triton_sparse_pooler_004.py --warmup 50 --repeat 100`. Required: `PASS accuracy` and all guardrails pass. On a local implementation defect, return `implementation-repair-required` with the failing guardrail and minimal diagnosis.
2. v2 screening: two short interleaved pairs (warmup 5, repeat 5) of accepted reference vs candidate. `screened-out` only if BOTH pairs are >=10% SLOWER than the accepted reference (candidate worse than reference); a faster candidate is NOT screened-out. Otherwise proceed to authoritative timing.
3. Authoritative 3-pair wall timing (warmup 50, repeat 100, interleaved accepted/candidate). Persist all 6 raw samples; compare UNROUNDED medians of the 3 reference samples vs the 3 candidate samples.
4. Level 1 profiler: `--v1_file sparse_pooler/triton_sparse_pooler_004.py --profile-reference-file sparse_pooler/triton_sparse_pooler_001.py --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output sparse_pooler/log/round_004_forward_50iter.pt.trace.json`. Use `gpu_user_annotation`-only fallback scoping (summarize_trace.py hits `overlapping scope events` on this harness).
5. Write `rounds/report_004.md` (Evaluation Contract mirror with 4 mechanism observables: fused_kernel_us_per_call, device_us_per_call, kernel_count_per_call, output_allocations_per_call; hypothesis verdict; improvement_pct; stop recommendation; exact reproduction commands) and update `state/verifier_state.md`.

## Next safe action

Run the correctness gate (step 1).

## After correctness gate

- command: `python3 /projs/framework/lipenghui/projects/kernelswift/auto_bench.py --v0_file sparse_pooler/base.py --v1_file sparse_pooler/triton_sparse_pooler_004.py --warmup 50 --repeat 100`
- exit code: 0
- stdout: `PASS accuracy; v0=0.888383 ms, v1=0.563431 ms, speedup=1.577x`
- correctness verdict: pass (all 4 outputs match within atol=1e-2 rtol=1e-2 equal_nan=True)
- guardrails: pass
  - output structure: list of 4 tensors each [30522] fp32 mlu:0 (returned from cached [num_seq, vocab_size] buffer)
  - numerical semantics: `log(1+relu(decoder(LayerNorm(GELU(Dense(hidden)))))` max-pooled per sequence, fused kernel body byte-identical to accepted reference
  - caller-selected device and current stream preserved (no `torch.mlu.device()` introduced)
  - dense GELU LayerNorm decoder matmul library pipeline unchanged
  - ModelNew public constructor and forward signature unchanged
  - load_state_dict compatibility confirmed (PASS accuracy; `_out_cache` not in state_dict)
  - kernel_count_per_call expected to remain 5 (profiler will confirm)
  - num_warps=1 used (fast_libentry wrapper passes through)
- candidate SHA-256 after correctness: `81cdea2b958c288e1382aef0b30cfc6dffb544c55a0e44825fab51b53cac7842` (unchanged; Verifier does not edit the candidate)

## Next safe action

Run v2 screening (step 2): two short interleaved accepted-reference/candidate pairs (warmup 5, repeat 5).

## After v2 screening

- protocol: two short interleaved accepted-reference/candidate pairs (warmup 5, repeat 5). `screened-out` only if BOTH pairs are >=10% SLOWER than the accepted reference; a faster candidate is NOT screened-out.
- pair 1 reference: `python3 auto_bench.py --v0_file sparse_pooler/base.py --v1_file sparse_pooler/triton_sparse_pooler_001.py --warmup 5 --repeat 5` → exit 0, `PASS accuracy; v0=0.908393 ms, v1=0.611090 ms, speedup=1.487x`
- pair 1 candidate: `python3 auto_bench.py --v0_file sparse_pooler/base.py --v1_file sparse_pooler/triton_sparse_pooler_004.py --warmup 5 --repeat 5` → exit 0, `PASS accuracy; v0=0.890209 ms, v1=0.554890 ms, speedup=1.604x`
- pair 2 reference: `python3 auto_bench.py --v0_file sparse_pooler/base.py --v1_file sparse_pooler/triton_sparse_pooler_001.py --warmup 5 --repeat 5` → exit 0, `PASS accuracy; v0=0.917981 ms, v1=0.622712 ms, speedup=1.474x`
- pair 2 candidate: `python3 auto_bench.py --v0_file sparse_pooler/base.py --v1_file sparse_pooler/triton_sparse_pooler_004.py --warmup 5 --repeat 5` → exit 0, `PASS accuracy; v0=0.910517 ms, v1=0.568455 ms, speedup=1.602x`

### Screening computation

- pair 1: candidate is `(0.554890 - 0.611090) / 0.611090 * 100 = -9.19%` (9.19% FASTER than accepted reference)
- pair 2: candidate is `(0.568455 - 0.622712) / 0.622712 * 100 = -8.71%` (8.71% FASTER than accepted reference)
- v2 screening verdict: BOTH pairs are FASTER than the accepted reference (NOT >=10% slower) → `NOT screened-out`
- proceed to authoritative 3-pair timing per the v2 protocol

## Next safe action

Run authoritative 3-pair wall timing (step 3): 3 interleaved pairs (warmup 50, repeat 100) of accepted reference vs candidate.

## After authoritative pair 1

- pair 1 reference: `--v1_file sparse_pooler/triton_sparse_pooler_001.py --warmup 50 --repeat 100` → exit 0, `PASS accuracy; v0=0.887205 ms, v1=0.596537 ms, speedup=1.487x`
- pair 1 candidate: `--v1_file sparse_pooler/triton_sparse_pooler_004.py --warmup 50 --repeat 100` → exit 0, `PASS accuracy; v0=0.901875 ms, v1=0.567125 ms, speedup=1.590x`
- pair 1 reference v1 ms: 0.596537
- pair 1 candidate v1 ms: 0.567125
- pair 1 delta: candidate is (0.567125 - 0.596537)/0.596537 * 100 = -4.93% (faster)

## Next safe action

Run authoritative pair 2 (accepted reference then candidate).

## After authoritative pair 2

- pair 2 reference: `--v1_file sparse_pooler/triton_sparse_pooler_001.py --warmup 50 --repeat 100` → exit 0, `PASS accuracy; v0=0.892008 ms, v1=0.601970 ms, speedup=1.482x`
- pair 2 candidate: `--v1_file sparse_pooler/triton_sparse_pooler_004.py --warmup 50 --repeat 100` → exit 0, `PASS accuracy; v0=0.908380 ms, v1=0.570159 ms, speedup=1.593x`
- pair 2 reference v1 ms: 0.601970
- pair 2 candidate v1 ms: 0.570159
- pair 2 delta: candidate is (0.570159 - 0.601970)/0.601970 * 100 = -5.28% (faster)

## Next safe action

Run authoritative pair 3 (accepted reference then candidate).

## After authoritative pair 3

- pair 3 reference: `--v1_file sparse_pooler/triton_sparse_pooler_001.py --warmup 50 --repeat 100` → exit 0, `PASS accuracy; v0=0.896377 ms, v1=0.603689 ms, speedup=1.485x`
- pair 3 candidate: `--v1_file sparse_pooler/triton_sparse_pooler_004.py --warmup 50 --repeat 100` → exit 0, `PASS accuracy; v0=0.884404 ms, v1=0.561303 ms, speedup=1.576x`
- pair 3 reference v1 ms: 0.603689
- pair 3 candidate v1 ms: 0.561303
- pair 3 delta: candidate is (0.561303 - 0.603689)/0.603689 * 100 = -7.02% (faster)

## Authoritative 3-pair summary

- reference_raw_samples_ms: `[0.596537, 0.601970, 0.603689]`
- candidate_raw_samples_ms: `[0.567125, 0.570159, 0.561303]`
- reference_median_ms (unrounded): `0.601970` (middle of sorted [0.596537, 0.601970, 0.603689])
- candidate_median_ms (unrounded): `0.567125` (middle of sorted [0.561303, 0.567125, 0.570159])
- improvement_pct (unrounded): `5.788494443244682`
- improvement_pct >= 5.0 adoption threshold: `True`

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
                = (0.601970 - 0.567125) / 0.601970 * 100
                = 5.788494443244682
```

Raw per-run harness output (v0 = base.py Model, v1 = accepted reference or candidate):

| Pair | Role | v0 ms | v1 ms | speedup | exit |
|---:|---|---:|---:|---:|---:|
| 1 | reference (triton_sparse_pooler_001) | 0.887205 | 0.596537 | 1.487x | 0 |
| 1 | candidate (triton_sparse_pooler_004) | 0.901875 | 0.567125 | 1.590x | 0 |
| 2 | reference (triton_sparse_pooler_001) | 0.892008 | 0.601970 | 1.482x | 0 |
| 2 | candidate (triton_sparse_pooler_004) | 0.908380 | 0.570159 | 1.593x | 0 |
| 3 | reference (triton_sparse_pooler_001) | 0.896377 | 0.603689 | 1.485x | 0 |
| 3 | candidate (triton_sparse_pooler_004) | 0.884404 | 0.561303 | 1.576x | 0 |

## Next safe action

Run Level 1 profiler (step 4): `--v1_file sparse_pooler/triton_sparse_pooler_004.py --profile-reference-file sparse_pooler/triton_sparse_pooler_001.py --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output sparse_pooler/log/round_004_forward_50iter.pt.trace.json`.

## After Level 1 profiler

- command: `python3 /projs/framework/lipenghui/projects/kernelswift/auto_bench.py --v0_file sparse_pooler/base.py --v1_file sparse_pooler/triton_sparse_pooler_004.py --profile --profile-reference-file sparse_pooler/triton_sparse_pooler_001.py --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-output sparse_pooler/log/round_004_forward_50iter.pt.trace.json`
- exit code: 0
- stdout: `PASS accuracy; v0=0.891202 ms, v1=0.559319 ms, speedup=1.593x` (diagnostic only; not the authoritative 3-pair median)
- trace: `/projs/framework/lipenghui/projects/kernelswift/sparse_pooler/log/round_004_forward_50iter.pt.trace.json` (2140211 bytes)
- `summarize_trace.py` hits `overlapping scope events` on both `reference_triton_sparse_pooler_001` and `candidate_triton_sparse_pooler_004` scopes (each scope has 2 overlapping events: `user_annotation` on CPU pid=3661212 + `gpu_user_annotation` on GPU pid=0/tid=1). Used the gpu_user_annotation-only fallback scoping documented in `state/verifier_context.md`, `rounds/report_001.md`, `rounds/report_002.md`, and `rounds/report_003.md`. Device kernel events attributed by full containment in the pid=0/tid=1 `gpu_user_annotation` interval.

### Reference scope (triton_sparse_pooler_001, 50 iterations)

- device_total_us: 10400.96
- device_us_per_call: 208.02
- kernel_count_total: 250
- kernel_count_per_call: 5.0
- top kernels (us/call):
  - `_sparse_pooler_max_kernel`: 98.76
  - `MLUFusedMatMulGepm` (decoder matmul 768->30522): 89.37
  - `MLUFusedMatMulGepdot` (dense matmul 768->768): 8.32
  - `layerNormForwardKernel`: 7.19
  - `MLUBlockKernel3StagePipelineGeluHighAccCubic` (GELU): 4.38

### Candidate scope (triton_sparse_pooler_004, 50 iterations)

- device_total_us: 10416.12
- device_us_per_call: 208.32
- kernel_count_total: 250
- kernel_count_per_call: 5.0
- top kernels (us/call):
  - `_sparse_pooler_max_kernel`: 98.75 (reference 98.76; unchanged within noise)
  - `MLUFusedMatMulGepm` (decoder matmul 768->30522): 89.63 (reference 89.37; unchanged within noise)
  - `MLUFusedMatMulGepdot` (dense matmul 768->768): 8.37 (reference 8.32; unchanged within noise)
  - `layerNormForwardKernel`: 7.19 (reference 7.19; identical)
  - `MLUBlockKernel3StagePipelineGeluHighAccCubic` (GELU): 4.39 (reference 4.38; unchanged within noise)

### Host-side allocation observable (aten::empty per scope, 50 iterations)

- reference_triton_sparse_pooler_001: 200 total `aten::empty` -> 4.00 per call (steady state; 49 of 50 forwards have 4 allocations: GELU output, LayerNorm output, decoder matmul logits, output buffer)
- candidate_triton_sparse_pooler_004: 150 total `aten::empty` -> 3.00 per call (all 50 forwards have exactly 3 allocations: GELU output, LayerNorm output, decoder matmul logits; the output buffer is NOT allocated — cache hit on every measured forward because warmup=20 populated `_out_cache` before the 50-iteration measured window)
- output_allocations_per_call: reference 1 -> candidate 0 on steady-state cache hits (confirmed)

### Direct cache-hit probe (additional Level 2 evidence)

- after forward 1: `_out_cache.data_ptr() = P1`
- after forward 2: `_out_cache.data_ptr() == P1` (cache hit, same buffer)
- after forward 3: `_out_cache.data_ptr() == P1` (cache hit, same buffer)
- `out2[0].data_ptr() == P1` (returned slices share storage with cached buffer)
- `_out_cache` is NOT in `state_dict()` (load_state_dict compatibility maintained; state_dict keys: `dense.weight`, `dense.bias`, `layer_norm.weight`, `layer_norm.bias`, `decoder.weight`, `decoder.bias`)

### Observable summary

- `fused_kernel_us_per_call`: reference 98.76 -> candidate 98.75 (unchanged within noise; the fused kernel body is byte-identical and the fast_libentry wrapper changes only the launcher path, not the kernel) — confirmed
- `device_us_per_call`: reference 208.02 -> candidate 208.32 (+0.30 us/call; unchanged within noise; no device-side change) — confirmed
- `kernel_count_per_call`: reference 5.0 -> candidate 5.0 (exactly; no kernel added or removed) — confirmed
- `output_allocations_per_call`: reference 1 -> candidate 0 on steady-state cache hits (all 50 measured forwards reuse the cached buffer) — confirmed

## Next safe action

Write `rounds/report_004.md` (final report with correctness/guardrail matrix, 6 raw wall samples and unrounded medians, improvement_pct, Evaluation Contract mirror with 4 mechanism observables + hypothesis verdict, profiler data, retry history, evidence_for_next_round, stop recommendation, exact reproduction commands) and update `state/verifier_state.md`.

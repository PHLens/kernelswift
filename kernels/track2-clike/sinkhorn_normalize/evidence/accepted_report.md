# Report 001

Result: accepted

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md`
- Candidate: `candidate_001.py` with `sinkhorn_normalize.cpp` and `CMakeLists.txt`
- Accepted reference: `baseline_adapter.py`
- Accepted reference report: `rounds/report_000.md`
- Decision SHA256: `64902ecc3ce87d8751578c2d033d57d872f0f7eabe4e364ca72081d12d09d184`
- Sketch SHA256: `8325f8cb80b82cafd4291772748d3e372de2bf729e0a989084fad6e2a6b13b0a`
- Candidate SHA256: `db4c3034c1d5b22b0d348aedb6ee6275e8a4597373c4d9a0bad3ac2c58df2921`
- Device source SHA256: `4ad28aadb0f8abe078d56c0826785870b1f4e42678fdb38c4d382cf727d1d209`
- CMake SHA256: `94c78637018e727b65432ee3f3318c64a6d8da0f1b6b28e039a17b6b31cb9179`
- Binding SHA256: `f1c1246859d504d56ddbe30333b4f453d331f8af5e684cdfe70cd5324803f44b`
- Accepted reference SHA256: `eea89997f9ab0c8849dea7f944ee73f084f9eb71b42049a021dc49b1f0d96f44`
- Base SHA256: `71c7444365a0187568baa0b486309ec465284d32444db66e56db85b6a395650a`
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Implementation profile snapshot SHA256: `4889e5c58279d37052dbcf148e1353581d353f852103f8921e4154fee2deea05`
- Capability claim SHA256: `30aafe19d7c7d91f9906369289af9d48e514df35792523c12e49959a0eabe76b`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `7a93739747647c249418c46c1ef711ee63542469eadeb1555b0aeabc67b2262f`
- verification_tier: `authoritative`
- screening_pairs: `[(1.586930,0.480575),(1.465880,0.481400)] ms`

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | `torch.allclose(atol=1e-2, rtol=1e-2, equal_nan=True)`, seed 42 | All five benchmark invocations printed `PASS accuracy`; three full-regime invocations exited 0 | pass | Exact commands below |
| harness loader | Real filtered AST loader and `cuda` placeholder rewrite | `auto_bench.build_case` loaded and executed candidate on NPU | pass | Guardrail command |
| output contract | `[1,1024,4,4]`, fp32, contiguous, NPU | Exact shape/dtype/layout/device assertions passed | pass | Guardrail command |
| input immutability | Input unchanged | Zero-tolerance comparison with pre-forward clone passed | pass | Guardrail command |
| cross-forward stability | Repeated outputs stable and independently allocated | Zero-tolerance equality and distinct output pointers passed | pass | Guardrail command |
| per-matrix independence | No statistics mixed across matrices | Modifying matrix 0 left matrices 1..1023 bitwise unchanged | pass | Guardrail command |
| non-aliasing | Fresh output distinct from input and prior output | Pointer inequality assertions passed | pass | Guardrail command |
| fixed public parameters | repeat 10 and eps 1e-6 exactly | Constructor rejects other values; official default path passed | pass | Candidate source and official commands |
| exact Sinkhorn sequence | eps after stable softmax; initial column normalization; nine row/column pairs; eps in 19 denominators | Source/binding observation matches Decision and numerical correctness passes | pass | `sinkhorn_normalize.cpp`, `rounds/binding_001.json` |
| current stream | One launch on caller current stream without candidate synchronization | Adapter passes `torch.npu.current_stream().npu_stream`; profiler observes one device kernel/call | pass | Candidate source and profiler |
| source immutability | Base, harness and canonical accepted source frozen | Independent hashes equal Phase 0 values | pass | Fingerprint command |

The first custom guardrail invocation exited 1 solely because the Verifier probe treated the harness tuple output as a tensor. The corrected probe unwrapped the tuple and passed without candidate modification; this is verifier retry history, not a candidate repair.

## Screening Evidence

| Pair | Reference short wall ms | Candidate short wall ms | Candidate slower pct | Evidence |
|---:|---:|---:|---:|---|
| 1 | 1.586930 | 0.480575 | -69.716660 | short command, invocation 1 |
| 2 | 1.465880 | 0.481400 | -67.159630 | short command, invocation 2 |

Neither pair is slower, so the candidate proceeded to authoritative timing.

## Interleaved Wall Timing

- seed: `42`
- atol/rtol: `1e-2 / 1e-2`
- warmup: `200`
- repeat: `500`
- order within each invocation: immutable base then candidate
- reference_raw_samples_ms: `[1.481490, 1.672045, 1.524825]`
- candidate_raw_samples_ms: `[0.478665, 0.485885, 0.484020]`
- reference_median_ms: `1.524825`
- candidate_median_ms: `0.484020`
- improvement_pct: `68.25734026854723`
- speedup_vs_reference_median: `3.1503450270657824x`

```text
improvement_pct = (1.524825 - 0.484020) / 1.524825 * 100
```

The unrounded improvement exceeds the frozen 5% adoption threshold.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| `kernel_count_per_call` | candidate one launch instead of accepted 59.0 kernels/call | `59.0 -> 1.0` kernels/call | pass | Independent CANN summaries |
| `accepted_eager_component_kernel_count_per_call` | zero candidate aclnnAdds, aclnnReduceSum, aclnnRealDiv, aclnnSoftmax launches | Candidate scope contains only `sinkhorn_normalize_0_mix_aic`; component count `0.0/call` | pass | Candidate CANN summary |
| `device_us_per_call` | decrease from accepted 498.1568 us/call | contemporaneous accepted `481.5474`, candidate `320.7216 us/call`; decrease `33.3967%` | pass | Independent CANN summaries |
| `sinkhorn_launch_count_per_call` | exactly one device kernel per forward | `sinkhorn_normalize_0_mix_aic` count `100/100 = 1.0/call` | pass | Candidate CANN summary |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-001`
- intervention: replace the eager 59-kernel decomposition with one current-stream Ascend C launch
- expected_causal_chain: eager component launches eliminated; kernel count and device work decrease; on-chip reuse avoids global intermediates; wall time improves at least 5%
- primary_metric: `wall_time`
- Hypothesis verdict: `confirmed`

All named observables are present. Correctness and semantic guardrails pass, the expected lowering is observed, and wall improvement is 68.25734026854723%.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted (Level 2 as declared)`
- profiler_device_time: `available`
- profile_mode: `forward`
- profile_warmup: `20`
- iterations: `100`
- trace SHA256: `3f8aa6b33aa2202c4730058fb1b076eb230a5f752ede12d3e7e83ca531f2c77c`
- reference database SHA256: `2be579e49892890510fb1167ed204cf11df39bed3611fd4033b84e9f43d66e8c`
- candidate database SHA256: `074e3eaf4e750a6fb4202fe1f81646a0363cb68849f6cc66fc2883700d7ceca4`

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| accepted_reference (`reference_baseline_adapter`) | 48154.7400 | 481.5474 | 5900 | 59.0 | 1.524825 | 0.31580502680635464 |
| candidate (`candidate_candidate_001`) | 32072.1600 | 320.7216 | 100 | 1.0 | 0.484020 | 0.6626205528697162 |

### Accepted Reference Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `aclnnReduceSum_ReduceSumOpAiCore_ReduceSum` | 1900 | 19.0 | 25448.66 | 254.4866 |
| `aclnnDiv_RealDivAiCore_RealDiv` | 1900 | 19.0 | 17664.84 | 176.6484 |
| `aclnnAdds_AddAiCore_Add` | 2000 | 20.0 | 3193.70 | 31.9370 |
| `aclnnSoftmax_SoftmaxAiCore_SoftmaxV2` | 100 | 1.0 | 1847.54 | 18.4754 |

### Candidate Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `sinkhorn_normalize_0_mix_aic` | 100 | 1.0 | 32072.16 | 320.7216 |

Observed lowering is exactly one fused Ascend C device kernel per forward and zero eager component kernels. Device time falls by about 33.40%, while the larger 68.26% wall improvement also supports eliminated launch/runtime overhead.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | initial full correctness | `db4c3034...` | `db4c3034...` | pass |
| 2 | verifier guardrail probe tuple-unwrapping error | `db4c3034...` | `db4c3034...` | corrected probe passed; no candidate repair |
| 3 | profiler database hash used guessed path | `db4c3034...` | `db4c3034...` | exact summarizer-returned paths hashed; no measurement rerun |

No Verifier-to-Coder repair was requested.

## evidence_for_next_round

- The single-launch Ascend C boundary is confirmed: `59.0 -> 1.0` device kernels/call and all eager Add/ReduceSum/RealDiv/Softmax component launches disappear.
- Candidate device time is `320.7216 us/call`, down approximately `33.40%` from the contemporaneous accepted reference.
- Candidate wall median is `0.484020 ms`, a `68.25734026854723%` improvement over the interleaved reference median.
- The remaining measured bottleneck is the single fused kernel itself at `320.7216 us/call`; about 33.7% of candidate wall time remains outside summed AI Core duration.
- Correctness, input immutability, non-aliasing, repeated-forward stability, per-matrix independence, loader, output contract, current-stream behavior, and frozen source hashes all pass.

## Stop Recommendation

- recommendation: `continue`
- evidence: no optional target is configured, this is round 1 of 20, and no global stop criterion applies.

## vNext Fact Pack

```json
{
  "schema_version": 1,
  "candidate_sha256": "db4c3034c1d5b22b0d348aedb6ee6275e8a4597373c4d9a0bad3ac2c58df2921",
  "correctness": {
    "status": "pass",
    "evidence": [
      "three exact-regime auto_bench invocations printed PASS accuracy",
      "independent loader and semantic guardrail probe passed"
    ]
  },
  "observables": [
    {
      "name": "kernel_count_per_call",
      "status": "observed",
      "value": "59.0 -> 1.0",
      "confidence": "high",
      "evidence": ["separately scoped CANN ai_core_op_summary.db summaries"]
    },
    {
      "name": "accepted_eager_component_kernel_count_per_call",
      "status": "observed",
      "value": "candidate 0.0/call",
      "confidence": "high",
      "evidence": ["candidate scope contains only sinkhorn_normalize_0_mix_aic"]
    },
    {
      "name": "device_us_per_call",
      "status": "observed",
      "value": "481.5474 -> 320.7216 us/call",
      "confidence": "high",
      "evidence": ["separately scoped CANN ai_core_op_summary.db summaries"]
    },
    {
      "name": "sinkhorn_launch_count_per_call",
      "status": "observed",
      "value": "1.0/call",
      "confidence": "high",
      "evidence": ["100 sinkhorn_normalize_0_mix_aic events / 100 forwards"]
    }
  ],
  "lowering": {
    "status": "observed",
    "expected_mechanism": "present",
    "evidence_contract": "cann-ai-core-summary-v1",
    "evidence": [
      "candidate scope: sinkhorn_normalize_0_mix_aic only",
      "candidate kernel_count_per_call: 1.0"
    ]
  },
  "evidence_gap_cause": "none"
}
```

## Exact Reproduction Commands

Fingerprint:

```bash
cd /workspace/kernelswift-dev-4ff2094 && sha256sum kernels/track2-clike/sinkhorn_normalize/base.py auto_bench.py kernels/track2-clike/sinkhorn_normalize/ascendc/baseline_adapter.py kernels/track2-clike/sinkhorn_normalize/ascendc/candidate_001.py kernels/track2-clike/sinkhorn_normalize/ascendc/sinkhorn_normalize.cpp kernels/track2-clike/sinkhorn_normalize/ascendc/CMakeLists.txt
```

Correctness and each authoritative pair (run exactly three times):

```bash
cd /workspace/kernelswift-dev-4ff2094 && /usr/local/python3.11.15/bin/python3 auto_bench.py --v0_file kernels/track2-clike/sinkhorn_normalize/base.py --v1_file kernels/track2-clike/sinkhorn_normalize/ascendc/candidate_001.py --seed 42 --atol 1e-2 --rtol 1e-2 --warmup 200 --repeat 500 --full-traceback
```

Each screening pair (run exactly twice):

```bash
cd /workspace/kernelswift-dev-4ff2094 && /usr/local/python3.11.15/bin/python3 auto_bench.py --v0_file kernels/track2-clike/sinkhorn_normalize/base.py --v1_file kernels/track2-clike/sinkhorn_normalize/ascendc/candidate_001.py --seed 42 --atol 1e-2 --rtol 1e-2 --warmup 50 --repeat 100 --full-traceback
```

Profiler:

```bash
cd /workspace/kernelswift-dev-4ff2094 && /usr/local/python3.11.15/bin/python3 auto_bench.py --v0_file kernels/track2-clike/sinkhorn_normalize/base.py --v1_file kernels/track2-clike/sinkhorn_normalize/ascendc/candidate_001.py --seed 42 --atol 1e-2 --rtol 1e-2 --warmup 200 --repeat 500 --profile --profile-reference-file kernels/track2-clike/sinkhorn_normalize/ascendc/baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 100 --profile-output kernels/track2-clike/sinkhorn_normalize/ascendc/log/round_001_forward_100iter.pt.trace.json --full-traceback
```

Profiler summaries:

```bash
cd /workspace/kernelswift-dev-4ff2094 && /usr/local/python3.11.15/bin/python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py kernels/track2-clike/sinkhorn_normalize/ascendc/log/profiling_data/reference_baseline_adapter/profiling_data/16458e336fc3_368063_20260830154920186_ascend_pt --iterations 100 --wall-ms 1.524825
cd /workspace/kernelswift-dev-4ff2094 && /usr/local/python3.11.15/bin/python3 skills/kernel-opt-loop/scripts/summarize_cann_trace.py kernels/track2-clike/sinkhorn_normalize/ascendc/log/profiling_data/candidate_candidate_001/profiling_data/16458e336fc3_368063_20260830154927257_ascend_pt --iterations 100 --wall-ms 0.484020
```

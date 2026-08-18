# Report 000

Result: baseline

## Identity

- Round: `000`
- Decision: `not-applicable: Phase 0`
- Candidate: `baseline_adapter.py`
- Accepted reference: `base.py` (Phase 0 source reference; no pre-existing canonical implementation)
- Accepted reference report: `not-applicable: Phase 0`
- Decision SHA256: `not-applicable: Phase 0`
- Candidate SHA256: `d92a1be10bf9ad036287ed0f769b195262132893fa78d88fa30f992fbe757827`
- Accepted reference SHA256: `49ec0cf7a48679c23c5187eb7ba546fd41025019a9d6e9b59e1273c48a31dfbb`
- Base SHA256: `49ec0cf7a48679c23c5187eb7ba546fd41025019a9d6e9b59e1273c48a31dfbb`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `3fe7d50260b3670756d26f003427faccb76e1e79c204b0a500a62d2eb481c809`
- verification_tier: `baseline`
- screening_pairs: `not-run: Phase 0`
- completed_at: `2026-08-18T05:48:23Z`

The local and remote copies of `base.py`, `baseline_adapter.py`, and `auto_bench.py` matched the frozen project hashes both before measurement and at the final post-measurement check.

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | Harness recursive comparison passes at seed 42, `atol=1e-2`, `rtol=1e-2`, equal-NaN enabled | `PASS accuracy`; `1 passed, 0 failed, 1 total`; command return code `0` | pass | `log/correctness_000.log` |
| public loader contract | `base.py` exposes `Model/get_init_inputs/get_inputs`; adapter exposes `ModelNew/get_init_inputs/get_inputs` through the actual AST loader | Frozen harness loaded, constructed, moved, and executed both sides without load or constructor error | pass | `log/correctness_000.log`; frozen harness SHA above |
| output tuple/shape/dtype | Same nested structure; tensors `(83, 8)`, weights `float32`, IDs `int32` | Harness recursive comparator accepted structure, shapes, and dtypes | pass | `log/correctness_000.log`; `project.md#semantics` |
| floating values | `torch.allclose(..., atol=1e-2, rtol=1e-2, equal_nan=True)` | No floating mismatch | pass | `log/correctness_000.log` |
| integer IDs | Exact `torch.equal` | No integer mismatch | pass | `log/correctness_000.log` |
| source semantic equivalence | After renaming the one top-level `Model` to `ModelNew`, normalized ASTs are equal | `normalized_ast_equal=True`; return code `0` | pass | `log/adapter_ast_equivalence_000.log` |
| frozen artifact identity | Local and remote hashes equal project.md before measurement and remain equal afterward | base `49ec0cf7...`, adapter `d92a1be1...`, harness `3d4fa4ee...` | pass | SHA256 commands in Exact Reproduction Commands |
| measurement regime | Exact C500 device, seed/tolerances, `warmup=200`, `repeat=500`, forward profile `20/100` | Commands used frozen arguments byte-for-byte | pass | `log/wall_000_sample_*.log`; `log/profile_000.log` |

The correctness command's `v0=0.254451 ms` and `v1=0.252472 ms` values are smoke timing only and do not replace the frozen 200/500 baseline samples.

## Screening Evidence

Not applicable in Phase 0. No screening classification was made.

## Sequential Block Wall Timing

- warmup: `200`
- repeat: `500`
- independent invocations: `3`
- actual harness timing order: `sequential complete accepted-reference block, then complete candidate block`
- interleaving: `not used by auto_bench.py in this run`
- base_raw_samples_ms: `[0.233009, 0.233050, 0.227752]`
- baseline_adapter_raw_samples_ms: `[0.231739, 0.232012, 0.226243]`
- base_median_ms: `0.233009`
- baseline_adapter_median_ms: `0.231739`
- canonical Round 000 baseline wall median: `0.231739 ms` (`baseline_adapter.py`)
- descriptive adapter-vs-base improvement_pct: `0.5450433245067758`

```text
descriptive improvement_pct = (0.233009 - 0.231739) / 0.233009 * 100
                           = 0.5450433245067758
```

This descriptive mechanical-adapter comparison is not an optimization-adoption decision. Round 000 establishes `baseline_adapter.py` as the baseline; the result is neither `accepted` nor `no-improvement`.

| Independent invocation | Base wall ms | Baseline adapter wall ms | Command return code | Evidence |
|---:|---:|---:|---:|---|
| 1 | `0.233009` | `0.231739` | `0` | `log/wall_000_sample_1.log` |
| 2 | `0.233050` | `0.232012` | `0` | `log/wall_000_sample_2.log` |
| 3 | `0.227752` | `0.226243` | `0` | `log/wall_000_sample_3.log` |

## Evaluation Contract Mirror

- Evaluation Contract applicability: `not-applicable: Phase 0`
- hypothesis_id: `not-applicable: Phase 0`
- intervention: `not-applicable: Phase 0`
- expected_causal_chain: `not-applicable: Phase 0`
- primary_metric: `not-applicable: Phase 0`
- Hypothesis verdict: `inconclusive` (no Phase 0 optimization hypothesis exists)

No decision or `mechanism_observables[]` exists for Phase 0, so there are no missing required observables.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `summary`
- profiler_device_time: `available` (`cat=kernel` durations)
- profile_mode: `forward`
- warmup: `20`
- iterations: `100` forward calls per scope
- scopes: `baseline_base`, `candidate_baseline_adapter`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- raw trace: `log/round_000_forward_100iter.pt.trace.json`, SHA256 `d5fa6827a1c1e8b92f210fd06d0ccfe17b5a14098c2bf9b218e9f91b8e287da3`
- attributable derived trace: `log/round_000_forward_100iter.dedup.pt.trace.json`, SHA256 `d712820f7adae8a164ddd996f18a37726c0abdf0bb2367719b15678be419e7e3`
- trace processing audit: `log/profile_processing_000.log`
- unmodified summarizer SHA256: `f625276c05a4539f86f272f68d78dd84ba0c94bb4ce153613dc4437df432148c`

The raw C500 trace emitted one CPU `user_annotation` and one fully nested `gpu_user_annotation` for each named scope. The unmodified repository summarizer therefore first returned `overlapping scope events` for both scopes. The raw trace was preserved. A documented jq filter removed exactly the two duplicate GPU scope-marker events and preserved all `3,000` kernel and `5,202` `cuda_runtime` events. Running the unmodified repository `summarize_trace.py` on that attributable derived trace returned code `0` for both scopes. Full demangled kernel names and every kernel aggregate are in the two summary JSON files below.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio | Summary |
|---|---:|---:|---:|---:|---:|---:|---|
| `baseline_base` | `14734.81103515625` | `147.3481103515625` | `1500` | `15.0` | `0.233009` | `0.6323708970536009` | `log/round_000_baseline_base_summary.json` |
| `candidate_baseline_adapter` | `14775.26708984375` | `147.7526708984375` | `1500` | `15.0` | `0.231739` | `0.6375822407900158` | `log/round_000_candidate_baseline_adapter_summary.json` |

The adapter device time is descriptively `0.2745610689609361%` higher than base in this single separately scoped profiler trace; profiler time is diagnostic and does not replace wall timing.

### Exact Names of the Top Three Kernels

- K1: `void at::native::gatherTopK_opt<float, unsigned int, 2>(at::cuda::detail::TensorInfo<float const, unsigned int>, unsigned int, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, float*)`
- K2: `void at::native::bitonicSortKVInPlace<2, -1, 16, 16, float, long, at::native::GTOp<float, true>, unsigned int>(at::cuda::detail::TensorInfo<float, unsigned int>, unsigned int, unsigned int, unsigned int, at::cuda::detail::TensorInfo<long, unsigned int>, unsigned int, at::native::GTOp<float, true>)`
- K3: `std::enable_if<((std::is_same<thrust::pair<float, long>, thrust::pair<float, long> >::value||std::is_same<thrust::pair<float, long>, thrust::pair<float, long> >::value)||std::is_same<thrust::pair<float, long>, thrust::pair<float, long> >::value)&&(!std::is_same<thrust::pair<float, long>, thrust::pair<long, long> >::value), void>::type at::native::InputPerOutputContinuousReduceKernel<float, float, thrust::pair<float, long>, 1, false, false, at::native::MaxOps<float>, at::native::reduce::OffsetCalculator<1, unsigned int, false>, at::native::reduce::OffsetCalculator<2, unsigned int, false> >(float const*, at::native::MaxOps<float>, thrust::pair<float, long>, int, int, at::native::reduce::OffsetCalculator<1, unsigned int, false>, at::native::reduce::OffsetCalculator<2, unsigned int, false>, int, bool, char const*, char const*, void*, int*, void*, long, bool, bool)`

### Base Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| K1 | `200` | `2.0` | `5080.0654296875` | `50.800654296875` |
| K2 | `200` | `2.0` | `3867.6455078125` | `38.676455078125` |
| K3 | `100` | `1.0` | `896.00048828125` | `8.9600048828125` |

### Baseline Adapter Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| K1 | `200` | `2.0` | `5049.0888671875` | `50.490888671875` |
| K2 | `200` | `2.0` | `3918.33056640625` | `39.1833056640625` |
| K3 | `100` | `1.0` | `887.04248046875` | `8.8704248046875` |

For the canonical adapter baseline, K1 and K2 together account for `89.6741943359375 us/call`, or about `60.69%` of measured device time, across `4.0` launches per call.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | Initial correctness and timing verification | `d92a1be10bf9ad036287ed0f769b195262132893fa78d88fa30f992fbe757827` | same | correctness and wall timing passed |
| 2 | Raw trace had duplicated nested GPU scope markers unsupported by summarizer interval rules | same | same | preserved raw trace; removed exactly two duplicate marker events in an auditable derived trace; both independent summaries passed |

No candidate repair occurred and no source file changed.

## evidence_for_next_round

- Canonical Phase 0 baseline: `baseline_adapter.py`, wall median `0.231739 ms` from three independent 200/500 samples under measurement fingerprint `3fe7d502...`.
- The canonical adapter scope measured `147.7526708984375 us/device-call` and `15.0 kernels/call`; K1 gatherTopK and K2 bitonic sort together used `89.6741943359375 us/call` (`~60.69%` of device time) and `4.0` launches/call.
- Base and adapter have equal normalized ASTs after the mandated top-level class rename; the small wall/device differences are measurement observations, not an optimization mechanism.
- The C500 trace format duplicates record-function scopes in CPU and GPU annotation categories; future summaries using this profiler output must preserve raw traces and account for the duplicate marker behavior without altering kernel/runtime events.

## Stop Recommendation

- recommendation: `continue`
- evidence: Phase 0 baseline is valid; no optional target is configured, and no terminal-round limit applies to baseline establishment.

Orchestrator owns canonical pointer updates and workflow transition.

## Exact Reproduction Commands

Remote SHA256 verification (run before and after measurement; both returned code `0`):

```bash
ssh -S /tmp/kernelswift-c500.sock -o BatchMode=yes -p 32222 root+vm-LmwqjLhYIUQymN0v@140.207.205.81 sha256sum /data/kernelswift-c500/maca/groupedtopk/base.py /data/kernelswift-c500/maca/groupedtopk/baseline_adapter.py /data/kernelswift-c500/auto_bench.py
```

Correctness (return code `0`):

```bash
cd /data/kernelswift-c500 && /data/kernelswift-c500/c500_run.sh /opt/conda/bin/python auto_bench.py --v0_file maca/groupedtopk/base.py --v1_file maca/groupedtopk/baseline_adapter.py --warmup 5 --repeat 10 --full-traceback
```

Wall timing (execute independently three times; return codes `0, 0, 0`):

```bash
cd /data/kernelswift-c500 && /data/kernelswift-c500/c500_run.sh /opt/conda/bin/python auto_bench.py --v0_file maca/groupedtopk/base.py --v1_file maca/groupedtopk/baseline_adapter.py --warmup 200 --repeat 500
```

Forward profiler (return code `0`):

```bash
cd /data/kernelswift-c500 && /data/kernelswift-c500/c500_run.sh /opt/conda/bin/python auto_bench.py --v0_file maca/groupedtopk/base.py --v1_file maca/groupedtopk/baseline_adapter.py --warmup 200 --repeat 500 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 100 --profile-output maca/groupedtopk/log/round_000_forward_100iter.pt.trace.json
```

Trace filter after the documented initial summarizer failures (filter returned code `0`):

```bash
jq -f maca/groupedtopk/log/profile_scope_filter_000.jq maca/groupedtopk/log/round_000_forward_100iter.pt.trace.json > maca/groupedtopk/log/round_000_forward_100iter.dedup.pt.trace.json
```

Separately scoped unmodified repository summaries (both returned code `0`):

```bash
/usr/bin/python3 skills/kernel-opt-loop/scripts/summarize_trace.py maca/groupedtopk/log/round_000_forward_100iter.dedup.pt.trace.json --iterations 100 --scope baseline_base --wall-ms 0.233009
```

```bash
/usr/bin/python3 skills/kernel-opt-loop/scripts/summarize_trace.py maca/groupedtopk/log/round_000_forward_100iter.dedup.pt.trace.json --iterations 100 --scope candidate_baseline_adapter --wall-ms 0.231739
```

## Command Status and Raw Evidence Index

| Command | Return code | Evidence |
|---|---:|---|
| local frozen-file SHA256 | `0` | hashes in Identity |
| remote frozen-file SHA256 before measurement | `0` | hashes in Identity |
| correctness 5/10 | `0` | `log/correctness_000.log` |
| wall sample 1, 200/500 | `0` | `log/wall_000_sample_1.log` |
| wall sample 2, 200/500 | `0` | `log/wall_000_sample_2.log` |
| wall sample 3, 200/500 | `0` | `log/wall_000_sample_3.log` |
| forward profiler 20/100 | `0` | `log/profile_000.log`; raw trace |
| copy remote trace with existing control connection | `0` | raw trace SHA above |
| summarize raw `baseline_base` | `1` | `log/profile_processing_000.log` |
| summarize raw `candidate_baseline_adapter` | `1` | `log/profile_processing_000.log` |
| audited duplicate-marker filter | `0` | `log/profile_scope_filter_000.jq`; processing log |
| derived trace integrity statistics | `0` | `log/profile_processing_000.log` |
| summarize derived `baseline_base` | `0` | `log/round_000_baseline_base_summary.json` |
| summarize derived `candidate_baseline_adapter` | `0` | `log/round_000_candidate_baseline_adapter_summary.json` |
| normalized-AST equivalence | `0` | `log/adapter_ast_equivalence_000.log` |
| remote frozen-file SHA256 after measurement | `0` | hashes in Identity |

# Report 002

Result: no-improvement

## Identity

- Round: `002`
- Decision: `rounds/decision_002.md`
- Candidate: `triton_grouped_topk_002.py`
- Accepted reference: `triton_grouped_topk_001.py`
- Harness reference adapter: `reference_triton_grouped_topk_001.py`
- Accepted reference report: `rounds/report_001.md`
- Decision SHA256: `96b175002ab35ebbdeab2e647e1f0acfb150d08ca30792db1c6657a3afea7c55`
- Coder result SHA256: `93eafe2b03f0a83fe65cb86b8453c787be60c753b3e4a5d42d834d6192ecfac7`
- Candidate SHA256: `1cbfddc1fd91ef4d73e388758467962cb471fc2a5f508c0af0749dcce53080d1`
- Accepted reference SHA256: `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384`
- Reference adapter SHA256: `70258939973f728858383b832e37069ec6b3d4681200cdb4e70daac42229b2f9`
- Base SHA256: `49ec0cf7a48679c23c5187eb7ba546fd41025019a9d6e9b59e1273c48a31dfbb`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `3fe7d50260b3670756d26f003427faccb76e1e79c204b0a500a62d2eb481c809`
- verification_tier: `authoritative`
- screening_pairs: `not-run: Round 002 delta requires its targeted storage/lifetime gate followed directly by three independent formal wall samples`
- completed_at: `2026-08-18T08:07:57Z`

Local and remote hashes matched for the accepted canonical, its one-line harness adapter, the candidate, and the harness before execution. The adapter differs from the accepted canonical by exactly `class ModelNew` renamed to `class Model`; it is the same accepted implementation in harness-loadable form. Verifier did not edit source, decision, project, harness, or team-state.

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| standard correctness | Actual harness recursive comparison passes at seed 42, `atol=1e-2`, `rtol=1e-2`, equal-NaN enabled | `PASS accuracy`; `1 passed, 0 failed, 1 total`; return code `0` | pass | `log/correctness_002.log` |
| reference identity | Adapter is only `ModelNew` to `Model`, at frozen SHA | Unified diff contains exactly that one source-line replacement; local and remote SHA match | pass | `log/source_equivalence_002.log` |
| frozen kernel and launch | Kernel body and `(83,)`, `T=83`, `BLOCK_E=256`, `num_warps=1` launch are byte-equivalent to Round 001 | Exact source comparison returned all true | pass | `log/source_equivalence_002.log` |
| guard/fallback/public behavior | Fast guard, canonical fallback, constructor, inputs and init inputs unchanged | Exact source comparison returned all true; actual harness loaded and executed both sides | pass | `log/source_equivalence_002.log`; `log/correctness_002.log` |
| dtype-view capability | CUDA-compatible int32 backing can be viewed as fp32 with no copy | Candidate executed; weights and IDs share the reported storage identity; exactly one intercepted `torch.empty` call | pass | `log/storage_lifetime_gate_002.log` |
| output storage contract | Contiguous `[83,8]` fp32/int32 outputs occupy strict non-overlapping backing byte spans | Weights `[0,2656)`, IDs `[2656,5312)` in one 5312-byte backing; all dtype/shape/device/stride checks true | pass | `log/storage_lifetime_gate_002.log` |
| output lifetime and isolation | Retained call-A output survives call B; calls and instances have distinct backing; mutation does not cross output views | All corresponding targeted checks true | pass | `log/storage_lifetime_gate_002.log` |
| allocation/state contract | One fresh allocation per fast call; no cache/model tensor state | One `(1328,)` int32 `torch.empty` per call; call and instance backing identities differ; no model tensor state | pass | `log/storage_lifetime_gate_002.log` |
| values, IDs, and inputs | Candidate matches reference; IDs exact; inputs unchanged | Max weight diff `0.0`; token-0 IDs both `[220,171,144,105,200,164,177,141]`; input check true | pass | `log/storage_lifetime_gate_002.log` |
| actual fast-path execution | Exact fixed contract under no-grad, with no silent fallback | Corrected aggregate no-grad check true; fallback `torch.softmax` trap not triggered | pass | `log/storage_lifetime_gate_002.log` |
| authoritative wall threshold | Unrounded median improvement at least `5.0%` | `-13.711567434852972%` | fail | `log/wall_002_sample_*.log` |

The correctness command's `v0=0.074996 ms` and `v1=0.079958 ms` are smoke values only. They do not contribute to the adoption decision.

## Storage/Lifetime Gate Retry

Attempt 1 returned code `1` after all substantive storage, lifetime, isolation, and correctness checks passed. Its final aggregate fast-contract flag was accidentally recomputed after leaving `torch.no_grad`, so only that probe bookkeeping field was false. Formal timing had not started. The original script SHA was `cdf00e592959749400b230cc0e42405be9e64f39e1ddfb6a5b5b9e2a13f29be4`; evidence is preserved in `log/storage_lifetime_gate_002_attempt1.log`.

Orchestrator approved the probe-only correction: compute the summary fast-contract value inside `torch.no_grad` and use that captured value in the checks dictionary. No candidate or other check changed. Corrected local and remote script SHA was `dd91be58a4450412bee285e11ac323a3fa7103e4d8d8bd1e412691d95a73cd75`; the one corrected rerun returned code `0` in `15.2981375 s`, with all 20 checks true and `overall=PASS`. Evidence is `log/storage_lifetime_gate_002.py` and `log/storage_lifetime_gate_002.log`.

## Screening Evidence

Not run under the explicit Round 002 verification delta. Classification uses the mandatory targeted capability/lifetime gate and three independent formal 200/500 wall invocations.

## Sequential Block Authoritative Wall Timing

- warmup: `200`
- repeat: `500`
- independent invocations: `3`
- actual harness order: `sequential complete accepted-reference-adapter block, then complete candidate block`
- interleaving: `not used by auto_bench.py`
- accepted_reference_raw_samples_ms: `[0.072343, 0.067703, 0.071684]`
- candidate_raw_samples_ms: `[0.082707, 0.076745, 0.081513]`
- accepted_reference_median_ms: `0.071684`
- candidate_median_ms: `0.081513`
- improvement_pct: `-13.711567434852972`
- speedup_from_medians: `0.8794180069436777x`

```text
improvement_pct = (0.071684 - 0.081513) / 0.071684 * 100
                = -13.711567434852972
```

| Invocation | Accepted reference ms | Candidate ms | Improvement pct | SSH elapsed s | Return code | Evidence |
|---:|---:|---:|---:|---:|---:|---|
| 1 | `0.072343` | `0.082707` | `-14.326196038317457` | `15.2866939` | `0` | `log/wall_002_sample_1.log` |
| 2 | `0.067703` | `0.076745` | `-13.355390455371246` | `14.9358120` | `0` | `log/wall_002_sample_2.log` |
| 3 | `0.071684` | `0.081513` | `-13.711567434852972` | `15.1908524` | `0` | `log/wall_002_sample_3.log` |

The unrounded cross-invocation median improvement controls adoption. It fails the `+5.0%` threshold and is a regression, so the result is `no-improvement` and `triton_grouped_topk_001.py` remains canonical. Profiler collection was contractually gated on passing this wall threshold and was therefore not run.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| `aten_empty_count_per_call` | decrease from 2.0 to exactly 1.0 in the fixed-fast-path CPU scope | Targeted interception observed exactly one fresh `(1328,)` int32 `torch.empty` for each candidate call; scoped profiler was not reached | pass for capability gate | `log/storage_lifetime_gate_002.log` |
| `aten_empty_inclusive_us_per_call` | decrease from accepted `10.03988 us/call`; inclusive diagnostic only | not collected because formal wall failed the mandatory profiler precondition | not-run: wall-gated | formal wall section |
| `triton_kernel_count_per_call` | remain exactly 1.0 | Kernel and direct-launch text are unchanged and actual fast path executed; trace count not recollected after wall rejection | not-run: wall-gated | `log/source_equivalence_002.log`; storage gate |
| `candidate_device_us_per_call` | no more than 1.05 times concurrently profiled accepted reference device time | not collected because formal wall failed the mandatory profiler precondition | not-run: wall-gated | formal wall section |
| `mcModuleLaunchKernel_count_per_call` | remain exactly 1.0 | Direct launch is source-equivalent; trace count not recollected after wall rejection | not-run: wall-gated | `log/source_equivalence_002.log` |
| `mcModuleLaunchKernel_inclusive_us_per_call` | no material regression from accepted `4.88562 us/call` inclusive diagnostic | not collected because formal wall failed the mandatory profiler precondition | not-run: wall-gated | formal wall section |
| `kernel_body_grid_num_warps_byte_equivalence` | kernel source and `(83,)`, `T=83`, `BLOCK_E=256`, `num_warps=1` launch byte-equivalent | exact text comparison true | pass | `log/source_equivalence_002.log` |
| `reference_adapter_class_rename_only` | frozen adapter SHA and only `ModelNew` to `Model` diff | SHA `70258939973f728858383b832e37069ec6b3d4681200cdb4e70daac42229b2f9`; one-line diff only | pass | `log/source_equivalence_002.log` |
| `maca_cuda_dtype_view_capability` | int32-to-fp32 CUDA-compatible view executes without copy, extra allocation, or kernel | same backing identity, exact typed views, one intercepted allocation, actual fast path | pass | `log/storage_lifetime_gate_002.log` |
| `output_storage_spans_disjoint` | same backing, strict disjoint intervals, contiguous typed `[83,8]` views | weights `[0,2656)`, IDs `[2656,5312)`; all checks true | pass | `log/storage_lifetime_gate_002.log` |
| `cross_call_live_outputs_distinct_backing` | retained call A, call B, and another model instance use distinct backing | all identity and retained-output checks true | pass | `log/storage_lifetime_gate_002.log` |
| `output_alias_mutation_isolation` | in-bounds mutation of one output cannot alter the other | bidirectional mutation-isolation checks true | pass | `log/storage_lifetime_gate_002.log` |

- Evaluation Contract applicability: `required; profiler-dependent observables conditionally not run after mandatory primary-metric rejection`
- hypothesis_id: `H-002`
- intervention: `replace two fixed-fast-path torch.empty calls with one fresh int32 backing and two non-overlapping typed views, without cache, reuse, pooling, or kernel changes`
- expected_causal_chain: `one allocation instead of two; allocation inclusive time decreases; kernel/launch/device behavior remains unchanged; median wall decreases at least 5%`
- primary_metric: `wall_time`
- Hypothesis verdict: `partially-confirmed: the one-allocation storage/lifetime intervention and frozen source invariants passed, but the primary wall-time claim was falsified by a 13.711567434852972% regression; conditional profile consequences were not evaluated`

The profiler-dependent rows are neither silently treated as passes nor classified as an infrastructure measurement failure: the prescribed sequence explicitly stops profiling when the primary wall gate is below `+5%`. They are unnecessary to the non-adoption decision.

## Profiler Evidence

- profiler_applicability: `not-run: formal wall median failed the mandatory +5% profiler precondition`
- profiler_level: `targeted planned`
- profile_mode: `forward planned`
- warmup/iterations: `20/100 planned, not executed`
- raw trace: `not-created`
- derived trace: `not-created`
- CPU/runtime inclusive diagnostics: `not collected`

The accepted Round 001 durable values remain evidence only: candidate scope `41.58952 us/call`, `aten::empty 2.0/call` at `10.03988 us/call`, `mcModuleLaunchKernel 1.0/call` at `4.88562 us/call`, and device time `10.7442822265625 us/call`. Inclusive CPU/runtime events may nest or overlap and cannot be added, subtracted, or used to reconstruct wall time. No Round 002 profile comparison is claimed.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | Initial storage/lifetime probe | `1cbfddc1fd91ef4d73e388758467962cb471fc2a5f508c0af0749dcce53080d1` | same | Probe returned `1` solely because its aggregate fast-contract value was recomputed outside no-grad; all substantive checks passed; no formal timing had begun |
| 2 | Orchestrator-approved probe-only correction | same | same | Corrected script SHA matched remotely; all 20 checks passed; formal wall proceeded |

No candidate repair occurred.

## evidence_for_next_round

- The allocation coalescing is feasible on the matched MACA runtime: one fresh 5312-byte int32 backing safely supports the two disjoint typed output views with correct lifetime and values.
- Despite that mechanism, the formal candidate median was `0.081513 ms` versus `0.071684 ms` for the concurrent accepted adapter, a `-13.711567434852972%` improvement (regression).
- Round 002 generated no profile trace because it failed the pre-profile wall threshold. Do not infer an allocation-event or launch-time cause for the regression from Round 001's inclusive diagnostics.

## Stop Recommendation

- recommendation: `continue`
- evidence: This is one valid no-improvement after Round 001 acceptance; the configured limit is three, the round budget is not exhausted, and no target stop is configured.

Orchestrator owns canonical pointers, counters, state transitions, and release of measurement exclusivity.

## Exact Reproduction Commands

Local and remote frozen-file hashes:

```bash
sha256sum maca/groupedtopk/triton_grouped_topk_001.py maca/groupedtopk/reference_triton_grouped_topk_001.py maca/groupedtopk/triton_grouped_topk_002.py auto_bench.py
ssh -S /tmp/kernelswift-c500.sock -o BatchMode=yes -p 32222 root+vm-LmwqjLhYIUQymN0v@140.207.205.81 sha256sum /data/kernelswift-c500/maca/groupedtopk/triton_grouped_topk_001.py /data/kernelswift-c500/maca/groupedtopk/reference_triton_grouped_topk_001.py /data/kernelswift-c500/maca/groupedtopk/triton_grouped_topk_002.py /data/kernelswift-c500/auto_bench.py
```

Adapter/source equivalence:

```bash
diff -u maca/groupedtopk/triton_grouped_topk_001.py maca/groupedtopk/reference_triton_grouped_topk_001.py
```

The exact local source-region comparison output and launch text are preserved in `log/source_equivalence_002.log`.

Standard correctness:

```bash
cd /data/kernelswift-c500 && /data/kernelswift-c500/c500_run.sh /opt/conda/bin/python auto_bench.py --v0_file maca/groupedtopk/reference_triton_grouped_topk_001.py --v1_file maca/groupedtopk/triton_grouped_topk_002.py --warmup 5 --repeat 10 --full-traceback
```

Corrected targeted storage/lifetime gate:

```bash
cd /data/kernelswift-c500 && /data/kernelswift-c500/c500_run.sh /opt/conda/bin/python maca/groupedtopk/log/storage_lifetime_gate_002.py
```

Formal wall timing, executed independently three times:

```bash
cd /data/kernelswift-c500 && /data/kernelswift-c500/c500_run.sh /opt/conda/bin/python auto_bench.py --v0_file maca/groupedtopk/reference_triton_grouped_topk_001.py --v1_file maca/groupedtopk/triton_grouped_topk_002.py --warmup 200 --repeat 500
```

The planned profiler command was not executed:

```bash
cd /data/kernelswift-c500 && /data/kernelswift-c500/c500_run.sh /opt/conda/bin/python auto_bench.py --v0_file maca/groupedtopk/reference_triton_grouped_topk_001.py --v1_file maca/groupedtopk/triton_grouped_topk_002.py --warmup 200 --repeat 500 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 100 --profile-output maca/groupedtopk/log/round_002_forward_100iter.pt.trace.json
```

## Command Status and Raw Evidence Index

| Command | Return code | Evidence |
|---|---:|---|
| local artifact SHA256 | `0` | Identity; `log/source_equivalence_002.log` |
| remote executable SHA256 | `0` | Identity; `log/source_equivalence_002.log` |
| adapter unified diff | `1` expected | `log/source_equivalence_002.log` |
| source-region equivalence | `0` | `log/source_equivalence_002.log` |
| standard correctness 5/10 | `0` | `log/correctness_002.log` |
| storage gate attempt 1 | `1` measurement-probe defect | `log/storage_lifetime_gate_002_attempt1.log` |
| corrected gate remote script SHA256 | `0` | `log/storage_lifetime_gate_002.log` |
| corrected storage/lifetime gate | `0` | `log/storage_lifetime_gate_002.log` |
| formal wall samples 1/2/3 | `0 / 0 / 0` | `log/wall_002_sample_*.log` |
| targeted forward profiler | `not run: wall gate failed` | no trace created |

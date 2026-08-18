# Report 004

Result: no-improvement

## Identity

- Round: `004`
- Decision: `rounds/decision_004.md`
- Candidate: `triton_grouped_topk_004.py`
- Accepted reference: `triton_grouped_topk_001.py`
- Harness reference adapter: `reference_triton_grouped_topk_001.py`
- Accepted reference report: `rounds/report_001.md`
- Decision SHA256: `5aab9a8df7ab5664cf6a2374d945b06a3a76d60a5fff9e4134a229a75bf0f587`
- Final Coder result SHA256: `c4ca2fdb07cfa49ba8ce2363f1e9238362d8ba2463aab16ee2ea00f2707a1551`
- Candidate SHA256: `50da7d1b6483822273b0dda404b33fb79fd96fbd95e3f0a9a48d8c8a3e315683`
- Accepted reference SHA256: `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384`
- Reference adapter SHA256: `70258939973f728858383b832e37069ec6b3d4681200cdb4e70daac42229b2f9`
- Base SHA256: `49ec0cf7a48679c23c5187eb7ba546fd41025019a9d6e9b59e1273c48a31dfbb`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `3fe7d50260b3670756d26f003427faccb76e1e79c204b0a500a62d2eb481c809`
- verification_tier: `authoritative`
- screening_pairs: `not-run: Round 004 required standard correctness, targeted semantic/tie guards, then three independent formal wall invocations`
- completed_at: `2026-08-18T09:56:48Z`

The dispatch initially named obsolete Coder-result SHA `eef4da55aec01136b6a3006475a3931dc84e5366de49a556f888d108845433ec`. The Orchestrator corrected this to the final record-only SHA above; the candidate was unchanged. Local and remote hashes matched for the accepted canonical, one-line harness adapter, candidate, base, and harness before execution. The adapter differs from the accepted canonical only by renaming `ModelNew` to `Model`. Verifier did not edit source, decision, project, harness, or team-state.

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| standard correctness | Frozen harness recursive comparison passes at seed 42 and configured tolerances | `PASS accuracy`; `1 passed, 0 failed`; return code `0` | pass | `log/correctness_004.log` |
| exact authorized source delta | Only fixed-fast-path host predicate/device-local specialization changes | Exact canonical-to-candidate normalization passed; no extra delta | pass | `log/source_equivalence_004.py`; `log/source_equivalence_004.log` |
| kernel/launch/config freeze | Complete Triton kernel, grid `(83,)`, arguments, `T=83`, `BLOCK_E=256`, `num_warps=1`, and direct launch unchanged | Exact text checks true | pass | `log/source_equivalence_004.log` |
| shape/materialization source count | Fast guard tuple materializations `2 -> 0` | Canonical `2`, candidate `0` | pass | `log/source_equivalence_004.log` |
| hidden metadata source count | Hidden eligibility metadata queries `4 -> 0` while preserving leading-token assertion | Canonical `4`, candidate `0`; token assertion exact and first | pass | source and semantic logs |
| device-property source count | Tensor device reads `5 -> 1` via invocation-local `gating_device` | Canonical `5`, candidate `1` | pass | `log/source_equivalence_004.log` |
| fresh allocation source/lifetime | Exactly two independent fresh `torch.empty` outputs and no retained model tensor state | Source count remains `2`; same-call, cross-call, cross-instance, and retained-output checks all true | pass | source and semantic logs |
| newly admitted hidden semantics | Eligible gating with hidden nonleading width, dtype, contiguity, or device variation matches base and demonstrably takes fast path | Four cases passed; candidate softmax calls `0`; exact IDs and tolerated values | pass | `log/semantic_guard_004.log` |
| group-cutoff tie parity | Group-rank 4/5 maxima tie and candidate IDs exactly match base | Both IDs `[0,32,64,96,1,2,3,4]`; max weight diff `0.0` | pass | `log/semantic_guard_004.log` |
| expert-cutoff tie parity | Eligible expert-rank 8/9 logits tie and candidate IDs exactly match base | Both IDs `[0,32,64,96,1,33,65,2]`; max weight diff `4.656612873077393e-10` | pass | `log/semantic_guard_004.log` |
| retained fallback/config/grad behavior | Retained gating/config conditions, sigmoid path, exact grad predicate, canonical fallback, unsupported scoring error, and token assertion preserve behavior | All targeted cases passed; fallback path demonstrated; errors matched | pass | `log/semantic_guard_004.log` |
| output/input contract | Values/IDs/shapes/dtypes/device/contiguity match and inputs remain unchanged | All value cases reported output contracts and input non-mutation true | pass | `log/semantic_guard_004.log` |
| authoritative wall threshold | Unrounded median improvement at least `5.0%` | `2.6856032379199086%` | fail | `log/wall_004_sample_*.log` |

The semantic guard executed once remotely after local and remote probe SHA256 matched `f39dcdc58d98ff37ddfc7b24e8e9776f0e2d8dbd4b4dd5647500e87022029dc8`. It returned code `0` with `18` cases and overall `PASS`. The standard correctness values `reference=0.083263 ms`, `candidate=0.073877 ms` are smoke timing only and are not used for adoption.

## Screening Evidence

Not run under the explicit Round 004 verification sequence. Classification uses the required standard correctness/semantic guards and three independent formal `200/500` wall invocations.

## Sequential Block Authoritative Wall Timing

- warmup: `200`
- repeat: `500`
- independent invocations: `3`
- actual harness order: `sequential complete accepted-reference-adapter block, then complete candidate block`
- interleaving: `not used by auto_bench.py`
- accepted_reference_raw_samples_ms: `[0.067650, 0.072364, 0.068439]`
- candidate_raw_samples_ms: `[0.065375, 0.070434, 0.066601]`
- accepted_reference_median_ms: `0.068439`
- candidate_median_ms: `0.066601`
- improvement_pct: `2.6856032379199086`
- speedup_from_medians: `1.027597183225477x`

```text
improvement_pct = (0.068439 - 0.066601) / 0.068439 * 100
                = 2.6856032379199086
```

| Invocation | Accepted reference ms | Candidate ms | Improvement pct | SSH elapsed s | Return code | Evidence |
|---:|---:|---:|---:|---:|---:|---|
| 1 | `0.067650` | `0.065375` | `3.3628972653362883` | `15.4091149` | `0` | `log/wall_004_sample_1.log` |
| 2 | `0.072364` | `0.070434` | `2.6670720247636965` | `15.5491880` | `0` | `log/wall_004_sample_2.log` |
| 3 | `0.068439` | `0.066601` | `2.6856032379199086` | `15.3127396` | `0` | `log/wall_004_sample_3.log` |

The unrounded cross-invocation median controls adoption. It is below `+5.0%`, so the result is `no-improvement` and `triton_grouped_topk_001.py` remains canonical. The Round 004 sequence conditioned profiling on first passing this wall threshold; no profiler was run.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| `fast_guard_shape_tuple_materializations_per_call` | decrease exactly `2 -> 0` by source audit | Canonical `2`, candidate `0` | pass | `log/source_equivalence_004.log` |
| `fast_guard_hidden_metadata_eligibility_queries_per_call` | decrease exactly `4 -> 0`, excluding assertion and conditional grad read | Canonical `4`, candidate `0`; assertion/grad guard retained | pass | source and semantic logs |
| `tensor_device_property_reads_per_fixed_call_source` | decrease exactly `5 -> 1` | Canonical `5`, candidate `1` local binding | pass | `log/source_equivalence_004.log` |
| `forward_cpu_scope_inclusive_us_per_call` | decrease by at least `4.1 us/call` from `41.58952 us/call` in comparable scopes | Not collected because formal wall failed mandatory profiler precondition | not-run: wall-gated | formal wall section |
| `newly_admitted_hidden_metadata_semantic_parity` | Four hidden metadata variations match base and take fast path | All four exact-ID/tolerated-value cases passed with candidate softmax count `0` | pass | `log/semantic_guard_004.log` |
| `retained_guard_and_fallback_equivalence` | Gating/config/grad conditions retained; fallback and errors match | Source exact; targeted shape/dtype/contiguity/device/config/sigmoid/grad/error/assertion cases passed | pass | source and semantic logs |
| `aten_empty_count_per_call` | remain exactly `2.0`, independent and fresh | Source count `2`; storage/lifetime checks passed; profiler count not recollected | pass at source/safety; profiler count wall-gated | source and semantic logs |
| `candidate_kernel_count_per_call` | remain exactly `1.0` | One unchanged direct launch in source; targeted trace count not recollected | not-run: wall-gated profile | `log/source_equivalence_004.log` |
| `candidate_device_us_per_call` | at most `1.05x` concurrently scoped reference | Not collected because formal wall failed mandatory profiler precondition | not-run: wall-gated | formal wall section |
| `mcModuleLaunchKernel_count_per_call` | remain exactly `1.0` with unchanged direct launch | Direct launch source unchanged; runtime trace count not recollected | not-run: wall-gated profile | `log/source_equivalence_004.log` |
| `kernel_launch_byte_equivalence` | Kernel and launch grid/arguments/config byte-equivalent | Exact source check true | pass | `log/source_equivalence_004.log` |
| `reference_adapter_class_rename_only` | Frozen SHA and only `ModelNew -> Model` | SHA `70258939973f728858383b832e37069ec6b3d4681200cdb4e70daac42229b2f9`; exact rename-only check true | pass | `log/source_equivalence_004.log` |

- Evaluation Contract applicability: `required; profiler-dependent observables conditionally not run after mandatory primary-metric rejection`
- hypothesis_id: `H-004`
- intervention: `specialize only fixed fast-path Python dispatch by removing semantically unnecessary hidden metadata checks, avoiding shape tuple materialization, and reusing one invocation-local gating device value`
- expected_causal_chain: `source host queries decrease; scoped forward CPU falls at least 4.1 us/call with allocations/launch/kernel unchanged; wall falls at least 5%`
- primary_metric: `wall_time`
- Hypothesis verdict: `partially-confirmed: exact source reductions, semantics, tie parity, guard/fallback behavior, and storage safety passed; the primary wall-time claim was falsified at 2.6856032379199086%; wall-gated profiler evidence was not collected`

Profiler-dependent rows are not inferred as passes. The prescribed sequence stops profiling after a sub-threshold wall result, and the missing trace is not needed to reject adoption. CPU profiler events, if collected, would be inclusive diagnostics and could not be added or subtracted to reconstruct wall.

## Profiler Evidence

- profiler_applicability: `not-run: formal wall median failed the mandatory +5% profiler precondition`
- profiler_level: `targeted planned`
- profile_mode: `forward planned`
- warmup/iterations: `20/100 planned, not executed`
- raw trace: `not-created`
- derived trace: `not-created`
- candidate CPU scope, empty count, launch count, kernel count, and device time: `not collected`

The durable accepted Round 001 values remain `41.58952 us/call` inclusive forward CPU, `10.7442822265625 us/device-call`, `1.0 kernel/call`, and accepted benchmark wall `0.068280 ms`. No Round 004 comparison to those profiler values is claimed.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | Combined semantic probe local precheck wrapper | `50da7d1b6483822273b0dda404b33fb79fd96fbd95e3f0a9a48d8c8a3e315683` | same | Wrapper quoting failed locally before importing or executing the probe; RC `1`; evidence retained |
| 2 | Corrected AST-only local precheck | same | same | RC `0`; no probe semantics executed locally; Orchestrator proxy-uploaded identical SHA |
| 3 | Single authorized remote semantic execution | same | same | Remote SHA matched; RC `0`; all 18 cases passed |

No candidate repair occurred, no remote semantic retry occurred, and all three formal wall commands were separate planned invocations.

## Raw Evidence Hashes

| Evidence | SHA256 |
|---|---|
| `log/source_equivalence_004.py` | `683e18ca974b5b933be2769ae7b058c62a98f44beb717872d3b8e67893833235` |
| `log/source_equivalence_004.log` | `7832f91f340c63b17efc03fcc3bd0b17ba687e35aadc56186966c0a831f61452` |
| `log/correctness_004.log` | `ef4a5f831d309e8effcb4a412c900f8e3238ba7adb523cca170753b35ef30543` |
| `log/semantic_probe_local_precheck_004.log` | `1873dcddd9a3ac5a79e5b0f5e9493257ba843eff31bc3093de3cb09024198897` |
| `log/semantic_guard_004.py` | `f39dcdc58d98ff37ddfc7b24e8e9776f0e2d8dbd4b4dd5647500e87022029dc8` |
| `log/semantic_guard_004.log` | `581d39890729c73c30ee0c74236ccdb744142985fa5a5954013f212691fe71bd` |
| `log/wall_004_sample_1.log` | `ba849f8291b871edb96148d87b2c29ad1766bdccd35ff02da6c2c638d09c7902` |
| `log/wall_004_sample_2.log` | `e16f1d1c5174c7bce476dac488f114433bfca233a02b3260271ca4c21b9d86b9` |
| `log/wall_004_sample_3.log` | `3ec5fa7176d0f6388b251cec5170007506d3f6e1030d9ccef5eb147a530214ac` |

## evidence_for_next_round

- The host-only specialization is semantically valid on the matched C500 runtime: all exact source reductions, four newly admitted hidden variants, two targeted ties, retained fallbacks/grad/errors, and storage/lifetime checks passed.
- The change is not adoptable: candidate median `0.066601 ms` versus concurrent accepted-reference median `0.068439 ms` is only `2.6856032379199086%` improvement.
- Round 004 generated no profile trace because it failed the pre-profile wall threshold; no causal claim is made about CPU scope, launch inclusive time, or device time.

## Stop Recommendation

- recommendation: `valid-no-improvement-limit`
- evidence: Round 004 is the third consecutive valid performance miss after the Round 001 acceptance (Rounds 002, 003, and 004), reaching the configured limit of three. Retain Round 001 canonical pointers and stop the optimization loop under policy.

Orchestrator owns canonical pointers, counters, state transitions, and release of measurement exclusivity.

## Exact Reproduction Commands

Frozen hashes:

```bash
sha256sum maca/groupedtopk/triton_grouped_topk_001.py maca/groupedtopk/reference_triton_grouped_topk_001.py maca/groupedtopk/triton_grouped_topk_004.py auto_bench.py maca/groupedtopk/base.py
ssh -S /tmp/kernelswift-c500.sock -o BatchMode=yes -p 32222 root+vm-LmwqjLhYIUQymN0v@140.207.205.81 sha256sum /data/kernelswift-c500/maca/groupedtopk/triton_grouped_topk_001.py /data/kernelswift-c500/maca/groupedtopk/reference_triton_grouped_topk_001.py /data/kernelswift-c500/maca/groupedtopk/triton_grouped_topk_004.py /data/kernelswift-c500/auto_bench.py /data/kernelswift-c500/maca/groupedtopk/base.py
```

Exact source delta:

```bash
/usr/bin/python3 maca/groupedtopk/log/source_equivalence_004.py
```

Standard correctness:

```bash
cd /data/kernelswift-c500 && /data/kernelswift-c500/c500_run.sh /opt/conda/bin/python auto_bench.py --v0_file maca/groupedtopk/reference_triton_grouped_topk_001.py --v1_file maca/groupedtopk/triton_grouped_topk_004.py --warmup 5 --repeat 10 --full-traceback
```

Targeted semantic/tie/fallback/storage guard:

```bash
cd /data/kernelswift-c500 && /data/kernelswift-c500/c500_run.sh /opt/conda/bin/python maca/groupedtopk/log/semantic_guard_004.py
```

Formal wall timing, executed independently three times:

```bash
cd /data/kernelswift-c500 && /data/kernelswift-c500/c500_run.sh /opt/conda/bin/python auto_bench.py --v0_file maca/groupedtopk/reference_triton_grouped_topk_001.py --v1_file maca/groupedtopk/triton_grouped_topk_004.py --warmup 200 --repeat 500
```

The planned profiler command was not executed:

```bash
cd /data/kernelswift-c500 && /data/kernelswift-c500/c500_run.sh /opt/conda/bin/python auto_bench.py --v0_file maca/groupedtopk/reference_triton_grouped_topk_001.py --v1_file maca/groupedtopk/triton_grouped_topk_004.py --warmup 200 --repeat 500 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 100 --profile-output maca/groupedtopk/log/round_004_forward_100iter.pt.trace.json
```

## Command Status and Raw Evidence Index

| Command | Return code | Evidence |
|---|---:|---|
| local/remote artifact SHA256 | `0 / 0` | Identity |
| exact source-delta probe | `0` | `log/source_equivalence_004.py`; `log/source_equivalence_004.log` |
| standard correctness 5/10 | `0` | `log/correctness_004.log` |
| semantic local wrapper precheck attempt | `1`, local wrapper quoting only; no candidate execution | `log/semantic_probe_local_precheck_004.log` |
| corrected AST-only local precheck | `0` | `log/semantic_probe_local_precheck_004.log` |
| remote semantic-probe SHA256 | `0` | `log/semantic_guard_004.log` |
| single targeted semantic/tie/fallback/storage run | `0` | `log/semantic_guard_004.log` |
| formal wall samples 1/2/3 | `0 / 0 / 0` | `log/wall_004_sample_*.log` |
| targeted forward profiler | `not run: wall gate failed` | no trace created |

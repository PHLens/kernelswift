# Report 003

Result: no-improvement

## Identity

- Round: `003`
- Decision: `rounds/decision_003.md`
- Candidate: `triton_grouped_topk_003.py`
- Accepted reference: `triton_grouped_topk_001.py`
- Harness reference adapter: `reference_triton_grouped_topk_001.py`
- Accepted reference report: `rounds/report_001.md`
- Decision SHA256: `cfcee8a61b91536da0aa302504b8bc4119c9c2deac5150878b6371870791f6b7`
- Coder result SHA256: `82372f63ad9632fa7d430f765d5f26d73afcc1d4a6688ead2cee33fec875310e`
- Candidate SHA256: `9409bd85da798b083e785774525a076ec781b6df13cd1129843fe7e9c9ead9f6`
- Accepted reference SHA256: `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384`
- Reference adapter SHA256: `70258939973f728858383b832e37069ec6b3d4681200cdb4e70daac42229b2f9`
- Base SHA256: `49ec0cf7a48679c23c5187eb7ba546fd41025019a9d6e9b59e1273c48a31dfbb`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `3fe7d50260b3670756d26f003427faccb76e1e79c204b0a500a62d2eb481c809`
- verification_tier: `authoritative`
- screening_pairs: `not-run: Round 003 delta explicitly required standard correctness and targeted ties followed by three independent formal wall invocations`
- completed_at: `2026-08-18T09:08:39Z`

Local and remote hashes matched for the accepted canonical, one-line harness adapter, candidate, base, and harness before execution. The adapter differs from the accepted canonical only by `class ModelNew` renamed to `class Model`. Verifier did not edit source, decision, project, harness, or team-state.

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| standard correctness | Frozen harness recursive comparison passes at seed 42 and configured tolerances | `PASS accuracy`; `1 passed, 0 failed`; return code `0` | pass | `log/correctness_003.log` |
| exact source delta | Exactly eight authorized expert argmax/value-sum pairs become combined max-with-index/left-tie reductions | Whole candidate equals canonical after exactly eight replacements; combined reductions `8`, expert argmax `0`, expert selected-value sum `0` | pass | `log/source_equivalence_003.py`; `log/source_equivalence_003.log` |
| group/post-selection kernel freeze | Group scoring/selection and all code after expert selection remain byte-equivalent | Exact source checks true; four group argmax calls retained | pass | `log/source_equivalence_003.log` |
| host/guard/fallback/launch freeze | Two independent fresh outputs, constructor/signature, fixed guard, fallback, direct launch and entry points unchanged | Host wrapper exact; two `torch.empty`; grid `(83,)`, `T=83`, `BLOCK_E=256`, `num_warps=1` | pass | `log/source_equivalence_003.log` |
| combined-reduction capability | Pinned MACA frontend/backend compiles and executes 256-lane max-with-index and explicit left tie | Standard harness and both targeted exact fast-path cases returned code `0` | pass | correctness and tie logs |
| group-cutoff tie parity | Only group-rank 4/5 maxima tie; exact candidate IDs match base | IDs both `[0,32,64,96,1,2,3,4]`; max weight diff `0.0` | pass | `log/tie_id_parity_003.log` |
| expert-cutoff tie parity | Only eligible expert-rank 8/9 logits tie; exact candidate IDs match base | IDs both `[0,32,64,96,1,33,65,2]`; max weight diff `4.656612873077393e-10` | pass | `log/tie_id_parity_003.log` |
| actual fast-path execution | Targeted cases take exact no-grad fixed path and do not silently fall back | Fast-contract checks true; fallback softmax trap not triggered | pass | `log/tie_id_parity_003.log` |
| output/input contract | Tuple shapes/dtypes/device/contiguity unchanged; inputs not mutated | Reference and candidate contracts true; inputs unchanged in both cases | pass | `log/tie_id_parity_003.log`; correctness log |
| authoritative wall threshold | Unrounded median improvement at least `5.0%` | `0.04903708987159917%` | fail | `log/wall_003_sample_*.log` |

The standard correctness values `reference=0.073956 ms`, `candidate=0.069987 ms` are smoke timing only and are not used for adoption.

## Screening Evidence

Not run under the explicit Round 003 verification delta. Classification uses the required correctness/tie guards and three independent formal 200/500 wall invocations.

## Sequential Block Authoritative Wall Timing

- warmup: `200`
- repeat: `500`
- independent invocations: `3`
- actual harness order: `sequential complete accepted-reference-adapter block, then complete candidate block`
- interleaving: `not used by auto_bench.py`
- accepted_reference_raw_samples_ms: `[0.067296, 0.067085, 0.072567]`
- candidate_raw_samples_ms: `[0.067263, 0.067139, 0.068747]`
- accepted_reference_median_ms: `0.067296`
- candidate_median_ms: `0.067263`
- improvement_pct: `0.04903708987159917`
- speedup_from_medians: `1.0004906114803085x`

```text
improvement_pct = (0.067296 - 0.067263) / 0.067296 * 100
                = 0.04903708987159917
```

| Invocation | Accepted reference ms | Candidate ms | Improvement pct | SSH elapsed s | Return code | Evidence |
|---:|---:|---:|---:|---:|---:|---|
| 1 | `0.067296` | `0.067263` | `0.04903708987159917` | `15.3001230` | `0` | `log/wall_003_sample_1.log` |
| 2 | `0.067085` | `0.067139` | `-0.08049489453677945` | `15.2061405` | `0` | `log/wall_003_sample_2.log` |
| 3 | `0.072567` | `0.068747` | `5.264100762054382` | `14.8943377` | `0` | `log/wall_003_sample_3.log` |

The unrounded cross-invocation median improvement controls adoption. It is far below `+5.0%`, so the result is `no-improvement` and `triton_grouped_topk_001.py` remains canonical. The Round 003 instructions condition profiler collection on first passing this wall threshold; no profiler was run.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| `expert_full_width_reductions_per_token` | decrease from 16 expert-selection reductions to exactly 8 combined max-with-index reductions | Exact source delta has eight combined reductions and no separate expert argmax/value-sum reductions | pass | `log/source_equivalence_003.log` |
| `expert_tl_sum_reductions_per_token` | decrease from exactly 8 to exactly 0, excluding final normalization | Exact source count is `0`; post-selection normalization source is unchanged | pass | `log/source_equivalence_003.log` |
| `max_with_indices_capability` | Pinned frontend/backend compiles and correctly executes 256-lane value-plus-index reduction with explicit left tie | Eight calls with explicit left tie compiled and executed in standard and targeted fast-path cases | pass | source, correctness, and tie logs |
| `tie_id_parity` | Exact IDs and permitted values match fixed seed plus targeted group/expert cutoff ties | Standard correctness passed; both targeted cases have exact full-output IDs and tolerated weights | pass | correctness and tie logs |
| `candidate_kernel_count_per_call` | remain exactly `1.0` | One direct launch is source-equivalent and executed, but targeted trace count was not recollected after wall rejection | not-run: wall-gated | `log/source_equivalence_003.log` |
| `candidate_device_us_per_call` | no more than `6.983783447265625 us/call` | Not collected because formal wall failed the mandatory profiler precondition | not-run: wall-gated | formal wall section |
| `launch_grid_num_warps_equivalence` | one direct launch with `(83,)`, `T=83`, `BLOCK_E=256`, `num_warps=1` | Exact host-wrapper/source check true | pass | `log/source_equivalence_003.log` |
| `host_allocation_source_equivalence` | retain two independent Round 001 `torch.empty` outputs; no Round 002 backing/view | Host wrapper is byte-equivalent, contains two allocations and no backing variable | pass | `log/source_equivalence_003.log` |
| `fixed_host_guard_fallback_equivalence` | constructor, signature, exact fixed guard, and canonical fallback unchanged | Whole host wrapper exact; actual loader and both paths validated as applicable | pass | source, correctness, and tie logs |
| `reference_adapter_class_rename_only` | Frozen adapter SHA and only `ModelNew` to `Model` diff | SHA `70258939973f728858383b832e37069ec6b3d4681200cdb4e70daac42229b2f9`; exact one-line rename check true | pass | `log/source_equivalence_003.log` |

- Evaluation Contract applicability: `required; profiler-dependent observables conditionally not run after mandatory primary-metric rejection`
- hypothesis_id: `H-003`
- intervention: `replace each of eight separate expert argmax plus selected-value sum pairs with one max returning value/index and explicit left tie`
- expected_causal_chain: `16 expert reductions become 8; sole device kernel falls at least 35%; wall median falls at least 5%`
- primary_metric: `wall_time`
- Hypothesis verdict: `partially-confirmed: source reduction, runtime capability, correctness, and ties passed; the primary wall-time claim was falsified because improvement was only 0.04903708987159917%; conditional device-time evidence was not collected`

Profiler-dependent rows are not inferred as passes. The prescribed sequence stops profiling after a sub-threshold wall result, and the missing trace is not needed to reject adoption.

## Profiler Evidence

- profiler_applicability: `not-run: formal wall median failed the mandatory +5% profiler precondition`
- profiler_level: `targeted planned`
- profile_mode: `forward planned`
- warmup/iterations: `20/100 planned, not executed`
- raw trace: `not-created`
- derived trace: `not-created`
- candidate device time and kernel count: `not collected`

The durable accepted Round 001 values remain `10.7442822265625 us/device-call` and `1.0 kernel/call`. No Round 003 comparison to the `6.983783447265625 us/call` device threshold is claimed.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | Initial local source-equivalence diagnostic | `9409bd85da798b083e785774525a076ec781b6df13cd1129843fe7e9c9ead9f6` | same | Exact replacement/frozen checks passed, but doubled raw-regex escaping made the non-gating group-argmax diagnostic report 0 |
| 2 | Regex-only local probe correction | same | same | Correct diagnostic reported four group argmax calls; all source gates passed |
| 3 | New Round 003 tie probe required remote placement | same | same | Verifier created local probe and paused; Orchestrator uploaded it; local/remote SHA matched and both cases passed |

No candidate repair occurred and there was no silent remote retry.

## evidence_for_next_round

- Eight combined max-with-index/left-tie reductions are supported and correct on the matched runtime; exact targeted group and expert cutoff ties match base IDs.
- Removing the eight selected-value full-width reductions did not provide an adoptable wall gain: median moved from `0.067296 ms` to `0.067263 ms`, only `0.04903708987159917%`.
- Round 003 generated no profile trace because it failed the pre-profile wall threshold; no claim is made about its kernel device-time reduction.

## Stop Recommendation

- recommendation: `continue`
- evidence: This would be the second consecutive valid no-improvement after Round 001 acceptance, below the configured limit of three; the round budget is not exhausted and no target stop is configured.

Orchestrator owns canonical pointers, counters, state transitions, and release of measurement exclusivity.

## Exact Reproduction Commands

Frozen hashes:

```bash
sha256sum maca/groupedtopk/triton_grouped_topk_001.py maca/groupedtopk/reference_triton_grouped_topk_001.py maca/groupedtopk/triton_grouped_topk_003.py auto_bench.py maca/groupedtopk/base.py
ssh -S /tmp/kernelswift-c500.sock -o BatchMode=yes -p 32222 root+vm-LmwqjLhYIUQymN0v@140.207.205.81 sha256sum /data/kernelswift-c500/maca/groupedtopk/triton_grouped_topk_001.py /data/kernelswift-c500/maca/groupedtopk/reference_triton_grouped_topk_001.py /data/kernelswift-c500/maca/groupedtopk/triton_grouped_topk_003.py /data/kernelswift-c500/auto_bench.py /data/kernelswift-c500/maca/groupedtopk/base.py
```

Exact source delta:

```bash
/usr/bin/python3 maca/groupedtopk/log/source_equivalence_003.py
```

Standard correctness:

```bash
cd /data/kernelswift-c500 && /data/kernelswift-c500/c500_run.sh /opt/conda/bin/python auto_bench.py --v0_file maca/groupedtopk/reference_triton_grouped_topk_001.py --v1_file maca/groupedtopk/triton_grouped_topk_003.py --warmup 5 --repeat 10 --full-traceback
```

Targeted tie parity:

```bash
cd /data/kernelswift-c500 && /data/kernelswift-c500/c500_run.sh /opt/conda/bin/python maca/groupedtopk/log/tie_id_parity_003.py
```

Formal wall timing, executed independently three times:

```bash
cd /data/kernelswift-c500 && /data/kernelswift-c500/c500_run.sh /opt/conda/bin/python auto_bench.py --v0_file maca/groupedtopk/reference_triton_grouped_topk_001.py --v1_file maca/groupedtopk/triton_grouped_topk_003.py --warmup 200 --repeat 500
```

The planned profiler command was not executed:

```bash
cd /data/kernelswift-c500 && /data/kernelswift-c500/c500_run.sh /opt/conda/bin/python auto_bench.py --v0_file maca/groupedtopk/reference_triton_grouped_topk_001.py --v1_file maca/groupedtopk/triton_grouped_topk_003.py --warmup 200 --repeat 500 --profile --profile-mode forward --profile-warmup 20 --profile-iterations 100 --profile-output maca/groupedtopk/log/round_003_forward_100iter.pt.trace.json
```

## Command Status and Raw Evidence Index

| Command | Return code | Evidence |
|---|---:|---|
| local/remote artifact SHA256 | `0 / 0` | Identity |
| local source probe attempt 1 | `0`, diagnostic defect | `log/source_equivalence_003_attempt1.log` |
| corrected exact source probe | `0` | `log/source_equivalence_003.py`; `log/source_equivalence_003.log` |
| standard correctness 5/10 | `0` | `log/correctness_003.log` |
| remote tie-probe SHA256 | `0` | `log/tie_id_parity_003.log` |
| targeted tie parity | `0` | `log/tie_id_parity_003.log` |
| formal wall samples 1/2/3 | `0 / 0 / 0` | `log/wall_003_sample_*.log` |
| targeted forward profiler | `not run: wall gate failed` | no trace created |

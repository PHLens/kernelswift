# Report 001

Result: accepted

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md`
- Candidate: `triton_grouped_topk_001.py`
- Accepted reference: `baseline_adapter.py`
- Accepted reference report: `rounds/report_000.md`
- Decision SHA256: `c71c970e3bcf6d7984272611627d711ce64b6f3c18d1a057b2aab440c50c173f`
- Coder result SHA256: `c8d583749759c718429a0ed118d695908de714f89edd12c8121ed44f67d03f65`
- Candidate SHA256: `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384`
- Accepted reference SHA256: `d92a1be10bf9ad036287ed0f769b195262132893fa78d88fa30f992fbe757827`
- Base SHA256: `49ec0cf7a48679c23c5187eb7ba546fd41025019a9d6e9b59e1273c48a31dfbb`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint`
- Measurement fingerprint: `3fe7d50260b3670756d26f003427faccb76e1e79c204b0a500a62d2eb481c809`
- verification_tier: `authoritative`
- screening_pairs: `not-run: Round 001 contract delta required targeted tie parity followed directly by three independent formal wall samples`
- completed_at: `2026-08-18T07:07:20Z`

Local and remote hashes matched for `base.py`, `baseline_adapter.py`, `triton_grouped_topk_001.py`, and `auto_bench.py` before execution. Verifier did not edit any source, decision, harness, project, or team-state file.

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| standard correctness | Actual harness recursive comparison passes at seed 42, `atol=1e-2`, `rtol=1e-2`, equal-NaN enabled | `PASS accuracy`; `1 passed, 0 failed, 1 total`; return code `0` | pass | `log/correctness_001.log` |
| group-cutoff tie ID parity | Fixed fast path; only group-rank 4/5 maxima tie; exact IDs match base | Duplicate value only `5.0 x2`; expert cutoff untied; ref/candidate token-0 IDs both `[0,32,64,96,1,2,3,4]`; max weight diff `0.0` | pass | `log/tie_id_parity_001.py`; `log/tie_id_parity_001.log` |
| expert-cutoff tie ID parity | Fixed fast path; group cutoff unique; only eligible expert-rank 8/9 logits tie; exact IDs match base | Duplicate value only `3.5 x2`; ref/candidate token-0 IDs both `[0,32,64,96,1,33,65,2]`; max weight diff `4.656612873077393e-10` | pass | `log/tie_id_parity_001.py`; `log/tie_id_parity_001.log` |
| actual fast-path execution | Both targeted cases satisfy exact shape/dtype/layout/device/no-grad guard and must not silently fall back | `fast_contract_satisfied_under_no_grad=true`; a temporary `torch.softmax` fallback trap was not triggered in either case | pass | `log/tie_id_parity_001.log` |
| output contract | Tuple of contiguous `[83,8]` fp32 weights and int32 IDs on `cuda:0` | Reference and candidate output-contract probes true in both targeted cases; harness recursive contract passed | pass | `log/tie_id_parity_001.log`; `log/correctness_001.log` |
| input non-mutation | Neither input is modified | Both targeted cases report `inputs_unchanged=true` | pass | `log/tie_id_parity_001.log` |
| fixed seeded IDs/weights | Floating values within tolerance and IDs exact | Standard harness passed | pass | `log/correctness_001.log` |
| wide argmax capability | Repeated 256-lane argmax/extract compiles and executes through the actual harness | Standard harness and both exact fast-path targeted cases executed successfully | pass | `log/correctness_001.log`; `log/tie_id_parity_001.log` |
| fallback/public behavior | Outside exact guard, unchanged accepted PyTorch path remains present; public constructor/forward and entry points remain compatible | Coder gate reports fallback AST identical to accepted adapter; actual harness loader constructed/executed candidate | pass | `rounds/coder_result_001.md`; `log/correctness_001.log` |
| device/current stream/lifecycle | Caller device/current stream preserved; no cache, global mutable state, output reuse, or context switch | Direct launch on input device; per-forward output allocation; no forbidden state/context operation in candidate | pass | candidate SHA above; `rounds/coder_result_001.md` |
| frozen measurement regime | Hashes, device, seed/tolerances, 200/500 wall, and forward 20/100 profile unchanged | Exact frozen commands succeeded | pass | command index below |
| authoritative wall threshold | Unrounded median improvement at least `5.0%` | `69.59021613749428%` | pass | `log/wall_001_sample_*.log` |

The correctness command's `v0=0.230905 ms` and `v1=0.070167 ms` are smoke values only; they are not used for adoption.

## Screening Evidence

Not run under the explicit Round 001 verification delta. Classification uses the required targeted correctness gates and three independent formal 200/500 wall samples below.

## Sequential Block Authoritative Wall Timing

- warmup: `200`
- repeat: `500`
- independent invocations: `3`
- actual harness order: `sequential complete base-reference block, then complete candidate block`
- interleaving: `not used by auto_bench.py`
- accepted_reference_raw_samples_ms: `[0.223698, 0.224533, 0.225974]`
- candidate_raw_samples_ms: `[0.068280, 0.068671, 0.067233]`
- accepted_reference_median_ms: `0.224533`
- candidate_median_ms: `0.068280`
- improvement_pct: `69.59021613749428`
- speedup_from_medians: `3.288415348564734x`

`auto_bench.py` requires its v0 file to expose `Model`, while the accepted canonical adapter exposes `ModelNew`. Therefore formal wall commands use `base.py` for the reference block. Round 000 proved that, after the required top-level class rename, `base.py` and `baseline_adapter.py` have equal normalized ASTs; the current profiler separately scopes the actual canonical `baseline_adapter.py`. The reference samples above are thus the harness-compatible proxy for the accepted canonical implementation, not a different algorithm.

```text
improvement_pct = (0.224533 - 0.068280) / 0.224533 * 100
                = 69.59021613749428
```

| Independent invocation | Accepted-reference proxy ms | Candidate ms | Improvement pct for invocation | SSH elapsed s | Return code | Evidence |
|---:|---:|---:|---:|---:|---:|---|
| 1 | `0.223698` | `0.068280` | `69.47670520076174` | `15.1700544` | `0` | `log/wall_001_sample_1.log` |
| 2 | `0.224533` | `0.068671` | `69.41604552671545` | `15.4756862` | `0` | `log/wall_001_sample_2.log` |
| 3 | `0.225974` | `0.067233` | `70.24746165417703` | `15.0783929` | `0` | `log/wall_001_sample_3.log` |

The unrounded cross-invocation median improvement controls adoption and exceeds the `5.0%` requirement.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| `candidate_kernel_count_per_call` | decrease from `15.0` to `1.0` on the fixed fast path | `1.0`; reference scope `15.0`; `93.33333333333333%` reduction | pass | candidate/reference summary JSON |
| `gatherTopK_plus_bitonicSort_us_per_call` | decrease from `89.6741943359375` to `0` because baseline library kernels disappear | Candidate scope contains neither kernel: `0`; concurrent reference scope reports `50.66498046875 + 38.438359375 = 89.10333984375 us/call` | pass | summary JSON and kernel tables |
| `candidate_device_us_per_call` | decrease from canonical `147.7526708984375 us/call` | `10.7442822265625 us/call`; `92.72819762835427%` lower than canonical Round 000, `92.72025166609224%` lower than concurrent reference scope | pass | candidate summary JSON; `rounds/report_000.md` |
| `fused_triton_kernel_count_per_call` | equal `1.0` in separately scoped candidate profile | `_grouped_topk_fixed_kernel`: `100` total, `1.0/call` | pass | candidate summary JSON |
| `wide_argmax_capability` | 256-lane repeated argmax/extract compiles and executes through actual harness | Standard harness and two targeted exact fast-path cases returned code `0` | pass | correctness and tie logs |
| `tie_id_parity` | exact IDs match base for fixed seeded input and targeted group/expert cutoff ties | Seeded harness passed; both targeted cases have exact full-output ID equality and the token-0 IDs recorded above | pass | correctness and tie logs |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-001`
- intervention: `replace the fixed benchmark softmax, group-max/group-top4, masked expert-top8, and renormalization chain with one direct-launch Triton-MACA program per token, ranking raw logits and normalizing only the selected eight logits after exact softmax-denominator cancellation`
- expected_causal_chain: `15 kernels and four gather/sort launches -> one fused Triton launch; full-softmax/intermediate work disappears; device/kernel counts decrease; wall median decreases >=5%`
- primary_metric: `wall_time`
- Hypothesis verdict: `confirmed`

All six exact-name required observables were collected and passed.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `available` (`cat=kernel` durations)
- profile_mode: `forward`
- warmup: `20`
- iterations: `100` forward calls per scope
- scopes: `reference_baseline_adapter`, `candidate_triton_grouped_topk_001`
- raw trace: `log/round_001_forward_100iter.pt.trace.json`, SHA256 `c3db406b8bd6213a56cc4fe92977e6ed24c378702f2dc2a139d69536dbbea1e8`
- attributable derived trace: `log/round_001_forward_100iter.dedup.pt.trace.json`, SHA256 `67a675bdd50280c165d46bbec5bb06af9e8b693f19c807cbdfc5efdf3d744b36`
- trace processing audit: `log/profile_processing_001.log`
- unmodified summarizer SHA256: `f625276c05a4539f86f272f68d78dd84ba0c94bb4ce153613dc4437df432148c`

The raw trace reproduced Round 000's known duplicate-marker behavior. Initial raw summaries returned `1/1` with `overlapping scope events`. The raw trace was preserved. The audited filter removed exactly two fully nested `gpu_user_annotation` scope markers: total events `13969 -> 13967`, while kernel events stayed `1600`, `cuda_runtime` events stayed `3002`, and CPU scope events stayed `2`. The unmodified repository summarizer then returned `0/0` on the derived trace.

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio | Summary |
|---|---:|---:|---:|---:|---:|---:|---|
| `reference_baseline_adapter` | `14759.1396484375` | `147.591396484375` | `1500` | `15.0` | `0.224533` | `0.6573260789477493` | `log/round_001_reference_baseline_adapter_summary.json` |
| `candidate_triton_grouped_topk_001` | `1074.42822265625` | `10.7442822265625` | `100` | `1.0` | `0.068280` | `0.15735621304280173` | `log/round_001_candidate_triton_grouped_topk_001_summary.json` |

### Accepted Reference Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `void at::native::gatherTopK_opt<float, unsigned int, 2>(...)` | `200` | `2.0` | `5066.498046875` | `50.66498046875` |
| `void at::native::bitonicSortKVInPlace<2, -1, 16, 16, float, long, at::native::GTOp<float, true>, unsigned int>(...)` | `200` | `2.0` | `3843.8359375` | `38.438359375` |
| `at::native::InputPerOutputContinuousReduceKernel<...MaxOps<float>...>` | `100` | `1.0` | `899.32568359375` | `8.9932568359375` |

Exact full demangled names and all 13 reference kernel aggregates are preserved in `log/round_001_reference_baseline_adapter_summary.json`.

### Candidate Top Kernels

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `_grouped_topk_fixed_kernel` | `100` | `1.0` | `1074.42822265625` | `10.7442822265625` |

Candidate `gatherTopK` and bitonic-sort counts and device time are both exactly zero.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | Initial authoritative correctness/timing verification | `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384` | same | all correctness/tie/wall gates passed |
| 2 | Verifier SCP of targeted script was rejected before process creation | same | same | no remote side effect; Orchestrator uploaded audited script; local/remote script SHA matched |
| 3 | Raw profiler trace had known duplicate nested GPU scope markers | same | same | raw evidence retained; exact two-marker filter produced attributable scopes; both summaries passed |

No candidate repair occurred.

## evidence_for_next_round

- Candidate wall median is `0.068280 ms`, a `69.59021613749428%` reduction versus the current-run accepted-reference proxy under the frozen measurement fingerprint.
- Fusing the 15-kernel chain into `_grouped_topk_fixed_kernel` reduced kernel count to `1.0/call` and device time to `10.7442822265625 us/call`; the prior gatherTopK/bitonicSort kernels are absent.
- Targeted group-cutoff and expert-cutoff tie cases exactly matched reference IDs while demonstrably taking the fixed fast path.
- Candidate device ratio is `0.15735621304280173`; therefore about `84.26%` of measured candidate wall time is outside attributed device-kernel duration in this trace. This is an observation, not a prescribed next implementation.

## Stop Recommendation

- recommendation: `continue`
- evidence: Candidate satisfies acceptance; no optional target is configured, total round budget is not exhausted, and no valid-no-improvement stop applies.

Orchestrator owns canonical-pointer/state transitions.

## Exact Reproduction Commands

Frozen hashes:

```bash
ssh -S /tmp/kernelswift-c500.sock -o BatchMode=yes -p 32222 root+vm-LmwqjLhYIUQymN0v@140.207.205.81 sha256sum /data/kernelswift-c500/maca/groupedtopk/base.py /data/kernelswift-c500/maca/groupedtopk/baseline_adapter.py /data/kernelswift-c500/maca/groupedtopk/triton_grouped_topk_001.py /data/kernelswift-c500/auto_bench.py
```

Standard correctness:

```bash
cd /data/kernelswift-c500 && /data/kernelswift-c500/c500_run.sh /opt/conda/bin/python auto_bench.py --v0_file maca/groupedtopk/base.py --v1_file maca/groupedtopk/triton_grouped_topk_001.py --warmup 5 --repeat 10 --full-traceback
```

Targeted tie parity:

```bash
cd /data/kernelswift-c500 && /data/kernelswift-c500/c500_run.sh /opt/conda/bin/python maca/groupedtopk/log/tie_id_parity_001.py
```

Formal wall timing, executed independently three times:

```bash
cd /data/kernelswift-c500 && /data/kernelswift-c500/c500_run.sh /opt/conda/bin/python auto_bench.py --v0_file maca/groupedtopk/base.py --v1_file maca/groupedtopk/triton_grouped_topk_001.py --warmup 200 --repeat 500
```

Separately scoped forward profiler:

```bash
cd /data/kernelswift-c500 && /data/kernelswift-c500/c500_run.sh /opt/conda/bin/python auto_bench.py --v0_file maca/groupedtopk/base.py --v1_file maca/groupedtopk/triton_grouped_topk_001.py --warmup 200 --repeat 500 --profile --profile-reference-file maca/groupedtopk/baseline_adapter.py --profile-mode forward --profile-warmup 20 --profile-iterations 100 --profile-output maca/groupedtopk/log/round_001_forward_100iter.pt.trace.json
```

Audited duplicate-marker filter:

```bash
jq -f maca/groupedtopk/log/profile_scope_filter_001.jq maca/groupedtopk/log/round_001_forward_100iter.pt.trace.json > maca/groupedtopk/log/round_001_forward_100iter.dedup.pt.trace.json
```

Unmodified separately scoped summaries:

```bash
/usr/bin/python3 skills/kernel-opt-loop/scripts/summarize_trace.py maca/groupedtopk/log/round_001_forward_100iter.dedup.pt.trace.json --iterations 100 --scope reference_baseline_adapter --wall-ms 0.224533
```

```bash
/usr/bin/python3 skills/kernel-opt-loop/scripts/summarize_trace.py maca/groupedtopk/log/round_001_forward_100iter.dedup.pt.trace.json --iterations 100 --scope candidate_triton_grouped_topk_001 --wall-ms 0.068280
```

## Command Status and Raw Evidence Index

| Command | Return code | Evidence |
|---|---:|---|
| local artifact SHA256 | `0` | Identity |
| remote executable SHA256 | `0` | Identity |
| standard correctness 5/10 | `0` | `log/correctness_001.log` |
| initial Verifier tie-script SCP | not started (tool-policy rejection) | `log/tie_script_transfer_001.log` |
| remote tie-script SHA256 after approved upload | `0` | `log/tie_id_parity_001.log` |
| targeted tie parity | `0` | `log/tie_id_parity_001.log` |
| formal wall samples 1/2/3 | `0 / 0 / 0` | `log/wall_001_sample_*.log` |
| separately scoped forward profiler | `0` | `log/profile_001.log` |
| raw trace copy | `0` | raw trace SHA above |
| raw trace summaries | `1 / 1` | `log/profile_processing_001.log` |
| exact duplicate-marker filter | `0` | filter and processing log |
| derived trace integrity counts | `0` | processing log |
| derived trace summaries | `0 / 0` | two summary JSON files |

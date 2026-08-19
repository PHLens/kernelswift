# Report 001

Result: accepted

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md`
- Candidate: `triton_mhc_post_layer_mix_001.py`
- Accepted reference: `baseline_adapter.py`
- Accepted reference report: `rounds/report_000.md`
- Decision SHA256: `335389df2498f37fb9f2c5c7ebc10986ab4edf555d939525413900e0e885ecfc`
- Candidate SHA256: `08a9d59f17ffa80224943b19bdcce390d908ca8ba15bf2e06ae469f45787d9fb`
- Accepted reference SHA256: `66a3a2c31863d18c725a52ab57fd1b9f89fe655dd7bab7cb4da158b8130b5d07`
- Base SHA256: `e392799f72edcf7210373ac8f2522bcf3d4f740c8f9dd0d1f117bc627939ebf3`
- Harness SHA256: `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2`
- Runtime fingerprint: `project.md#runtime-fingerprint` (torch 2.7.1, triton 3.1.0, Iluvatar BI-V150, capability (7,1))
- Measurement fingerprint: `c17c7f45d44b1bec047c7c3a315275d33b21af3226dd34e884d483899ef039b6`
- verification_tier: `authoritative`
- screening_pairs: `not-run` (correct candidate proceeded directly to authoritative timing)

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | Harness recursive comparison at seed 42, `atol=1e-2`, `rtol=1e-2`, equal-NaN enabled | `PASS accuracy; v0=8.041220 ms, v1=6.469838 ms, speedup=1.243x`; return code `0` | pass | correctness command in Exact Reproduction Commands |
| output dtype and shape unchanged | single tensor `[2,4096,4,1280]` `bfloat16` | independent probe: `(2, 4096, 4, 1280) torch.bfloat16`, `allclose(base,candidate)=True` | pass | `/tmp/mplm_probe_001.py` (deleted) |
| fp32 intermediate precision preserved | arithmetic in fp32, only final `.bfloat16()` rounds | `allclose(base, direct_fp32_ref)=True`, `max_abs_diff=0.03125`, `mean_abs_diff=1.73e-8` | pass | independent probe |
| einsum term2 result unchanged | `torch.einsum('abmn,abmc->abnc', comb_res_mix, residual.float())` left unchanged | candidate forward still calls `torch.einsum` for term2; GEMM kernel identical in both scopes | pass | candidate source; profiler top kernels |
| dim=-2 broadcast | `x.float().unsqueeze(-2) * post_layer_mix` broadcast to `[2,4096,4,1280]` | fused kernel decomposes flat index into `(a,b,n,c)` and loads `x[a,b,c]`, `post_layer_mix[a,b,n,0]`, reproducing the broadcast | pass | candidate source; independent probe |
| frozen artifact identity | local hashes equal project.md/decision before and after | candidate `08a9d59f...`, reference `66a3a2c3...`, decision `335389df...` all match | pass | SHA256 commands in Exact Reproduction Commands |
| measurement regime | device cuda:0, `warmup=50`, `repeat=100`, forward profile `20/50` | commands used frozen arguments; reference and candidate pairs byte-for-byte identical except `--v1_file` | pass | round_status_001.md |

## Screening Evidence

Not run: the candidate passed correctness and was taken directly to authoritative
timing (per contract, screening applies only when a candidate might be rejected
on two short pairs).

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate` (harness times v0 then v1 per invocation)
- independent invocations: `3`
- reference_raw_samples_ms: `[8.022035, 8.152739, 8.043548]`
- candidate_raw_samples_ms: `[6.423922, 6.427432, 6.513436]`
- reference_median_ms: `8.043548`
- candidate_median_ms: `6.427432`
- improvement_pct: `20.09`

```text
improvement_pct = (8.043548 - 6.427432) / 8.043548 * 100
               = 20.09
```

The unrounded improvement `20.09%` exceeds the 5% adoption threshold. Wall time
controls adoption; this is an `accepted` classification.

## Evaluation Contract Mirror

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| kernel_count_per_call | decrease | `5.66` (reference) → `2.96` (candidate): the four post-GEMM elementwise/cast kernels (`add`, `MulFunctor`, `direct_copy` for x, `bfloat16_copy`) collapse to one `_fused_tail_kernel` | pass | profiler scopes |
| device_us_per_call | decrease | `7516.836 us/call` (reference) → `6122.542 us/call` (candidate), −18.5% | pass | profiler scopes |

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-001`
- intervention: `fuse the post-GEMM elementwise tail (x bf16->fp32 cast, multiply x*post_layer_mix, add term2, and fp32->bf16 output cast) into a single Triton kernel`
- expected_causal_chain: `four separate elementwise/cast kernels collapse into one → kernel_count_per_call decreases → intermediate fp32 materialization and launch overhead decrease → device_us_per_call decreases → wall time decreases`
- primary_metric: `wall_time`
- Hypothesis verdict: `confirmed`

The causal chain is confirmed: kernel count dropped from 5.66 to 2.96/call
(the multiply, add, and output cast fused into `_fused_tail_kernel`), device time
dropped 18.5%, and wall time dropped 20.09%, exceeding the 5% threshold.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `available` (`cat=kernel` durations)
- profile_mode: `forward`
- warmup: `20`
- iterations: `50` forward calls per scope
- scopes: `reference_baseline_adapter`, `candidate_triton_mhc_post_layer_mix_001`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- raw trace: `log/round_001_forward_50iter.pt.trace.json`, SHA256 `8543b6ea4418b53292632e5e4321d07e96e881c320072edfde1d53a79115720a`
- unmodified summarizer SHA256: `f625276c05a4539f86f272f68d78dd84ba0c94bb4ce153613dc4437df432148c`

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| `reference_baseline_adapter` | `375841.812` | `7516.836` | `283` | `5.66` | `8.043548` | `0.9345174841045736` |
| `candidate_triton_mhc_post_layer_mix_001` | `306127.103` | `6122.542` | `148` | `2.96` | `6.427432` | `0.952564...` |

Note on candidate scope summarization: the unmodified summarizer returned
`overlapping scope events` (code `2`) for the candidate scope because the Triton
direct launch inside `forward` produced two nested same-name `record_function`
markers with overlapping intervals. The candidate scope numbers above are from a
manual extraction over the innermost scope interval; the reference scope was
summarized by the unmodified summarizer (code `0`). This is a profiler-marker
artifact, not a measurement error: the kernel set and counts are unambiguous.

### Accepted Reference Top Kernels (reference_baseline_adapter scope)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `gemm_tcu_h` (TCU batched GEMM) | `47` | `0.94` | `248588.288` | `4971.766` |
| `CUDAFunctor_add<float>` | `47` | `0.94` | `40598.945` | `811.979` |
| `MulFunctor<float>` | `47` | `0.94` | `31924.133` | `638.483` |
| `bfloat16_copy_kernel_cuda` | `47` | `0.94` | `27520.130` | `550.403` |
| `direct_copy_kernel_cuda` | `95` | `1.90` | `27210.316` | `544.206` |

### Candidate Top Kernels (candidate_triton_mhc_post_layer_mix_001 scope)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `gemm_tcu_h` (TCU batched GEMM, unchanged) | `49` | `0.98` | `259174.67` | `5183.49` |
| `_fused_tail_kernel` (Triton fused tail) | `50` | `1.00` | `24809.22` | `496.18` |
| `direct_copy_kernel_cuda` (residual.float() cast) | `49` | `0.98` | `22143.21` | `442.86` |

The candidate forward kernel sequence is exactly three kernels: the unchanged
TCU batched GEMM, the fused Triton tail (`_fused_tail_kernel`), and the
`residual.float()` bf16→fp32 cast that feeds the GEMM (a GEMM-input cast, outside
the fusion boundary and correctly preserved). The three post-GEMM kernels that
the decision targeted (`CUDAFunctor_add` for `+term2`, `MulFunctor` for
`x*post_layer_mix`, `bfloat16_copy` for the output cast) are gone, collapsed into
`_fused_tail_kernel`.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | Initial verification | `08a9d59f17ffa80224943b19bdcce390d908ca8ba15bf2e06ae469f45787d9fb` | same | correctness, wall timing, and profiler all passed |

No candidate repair occurred and no source file changed.

## evidence_for_next_round

- The fused tail reduced wall time from `8.043548 ms` to `6.427432 ms`
  (20.09%) and device time from `7516.836 us/call` to `6122.542 us/call`
  (−18.5%), confirming H-001.
- The remaining bottleneck is now the unchanged TCU batched GEMM at
  `5183.49 us/call` (≈85% of remaining device time), plus the residual
  `residual.float()` cast (`442.86 us/call`) and the fused tail
  (`496.18 us/call`). The GEMM (`[4,4]@[4,1280]`, contraction dim 4) is the
  dominant cost; its `tl.dot` rewrite is still a capability risk on triton_cuda.

## Stop Recommendation

- recommendation: `continue`
- evidence: H-001 confirmed with 20.09% wall improvement; the GEMM remains the
  dominant bottleneck and is a candidate for a later, higher-risk round.

Orchestrator owns canonical pointer updates and workflow transition.

## Exact Reproduction Commands

Environment bootstrap (every command):

```bash
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
```

Frozen-file SHA256 verification:

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && sha256sum kernels/track1-triton/mhc_post_layer_mix/base.py kernels/track1-triton/mhc_post_layer_mix/bi150/baseline_adapter.py kernels/track1-triton/mhc_post_layer_mix/bi150/triton_mhc_post_layer_mix_001.py kernels/track1-triton/mhc_post_layer_mix/bi150/rounds/decision_001.md auto_bench.py
```

Correctness (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/mhc_post_layer_mix/base.py --v1_file kernels/track1-triton/mhc_post_layer_mix/bi150/triton_mhc_post_layer_mix_001.py --warmup 50 --repeat 100 --full-traceback
```

Authoritative wall timing (baseline wrapper, then three interleaved pairs):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && sed -e 's/^class ModelNew/class Model/' -e 's/super(ModelNew, self)/super(Model, self)/' kernels/track1-triton/mhc_post_layer_mix/bi150/baseline_adapter.py > /tmp/mplm_baseline_model_001.py
```

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file /tmp/mplm_baseline_model_001.py --v1_file kernels/track1-triton/mhc_post_layer_mix/bi150/triton_mhc_post_layer_mix_001.py --warmup 50 --repeat 100
```

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file /tmp/mplm_baseline_model_001.py --v1_file kernels/track1-triton/mhc_post_layer_mix/bi150/triton_mhc_post_layer_mix_001.py --warmup 50 --repeat 100
```

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file /tmp/mplm_baseline_model_001.py --v1_file kernels/track1-triton/mhc_post_layer_mix/bi150/triton_mhc_post_layer_mix_001.py --warmup 50 --repeat 100
```

Targeted profiler (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/mhc_post_layer_mix/base.py --v1_file kernels/track1-triton/mhc_post_layer_mix/bi150/triton_mhc_post_layer_mix_001.py --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-reference-file kernels/track1-triton/mhc_post_layer_mix/bi150/baseline_adapter.py --profile-output kernels/track1-triton/mhc_post_layer_mix/bi150/log/round_001_forward_50iter.pt.trace.json
```

Separately scoped unmodified repository summaries:

```bash
python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/mhc_post_layer_mix/bi150/log/round_001_forward_50iter.pt.trace.json --iterations 50 --scope reference_baseline_adapter --wall-ms 8.043548
```

## Command Status and Raw Evidence Index

| Command | Return code | Evidence |
|---|---:|---|
| frozen-file SHA256 before measurement | `0` | hashes in Identity |
| correctness 50/100 (vs base.py) | `0` | report Correctness table |
| independent numerical probe | `0` | report Correctness table |
| wall pair 1, 50/100 | `0` | report Interleaved Wall Timing |
| wall pair 2, 50/100 | `0` | report Interleaved Wall Timing |
| wall pair 3, 50/100 | `0` | report Interleaved Wall Timing |
| targeted profiler 20/50 | `0` | `log/round_001_forward_50iter.pt.trace.json` |
| summarize `reference_baseline_adapter` | `0` | report Profiler Evidence |
| summarize `candidate_triton_mhc_post_layer_mix_001` (unmodified) | `2` (overlapping scope) | manual extraction in report |
| frozen-file SHA256 after measurement | `0` | hashes in Identity |

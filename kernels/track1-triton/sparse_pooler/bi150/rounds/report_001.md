# Report 001

Result: accepted

## Identity

- Round: `001`
- Decision: `rounds/decision_001.md`
- Candidate: `triton_sparse_pooler_001.py`
- Accepted reference: `baseline_adapter.py`
- Accepted reference report: `rounds/report_000.md`
- Decision SHA256: `0fbbdb6929e1b75f939fc2d513c28878b7a53587f33e8fcaf66401f1269256f1`
- Candidate SHA256: `f3fd85a2c913d477e2cac7f65ed1f79dd5e1b9a3a60481782dbb4acaa43d2d98`
- Accepted reference SHA256: `359f4c808a0cf210416116322e4cc01f74ee42961b68c1fd365672af2a59bde8`
- Base SHA256: `46106baa46b9f1952b32388d99880c967438ea99becdc874e3f68ba81a727d58`
- Harness SHA256: `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29`
- Runtime fingerprint: `project.md#runtime-fingerprint` (torch 2.7.1, triton 3.1.0, Iluvatar BI-V150, capability (7,1), 16 SM, 16 GiB)
- Measurement fingerprint: `72be9562432197795bf6a24300483ccb2c3219b804b73258611048014cd804a9`
- verification_tier: `authoritative`
- screening_pairs: `not-run` (correctness passed; proceeded directly to authoritative timing)

Candidate, adapter, and decision hashes all match the values provided by the
Orchestrator. The candidate's correctness was verified against `base.py` and its
authoritative timing against a `Model`-renamed wrapper of `baseline_adapter.py`.

## Correctness and Guardrails

| Check | Requirement | Observation | Verdict | Evidence |
|---|---|---|---|---|
| correctness | Harness recursive list comparison at seed 42, `atol=1e-2`, `rtol=1e-2`, equal-NaN enabled | `PASS accuracy; v0=1.057676 ms, v1=0.881433 ms, speedup=1.200x`; return code `0` | pass | correctness command in Exact Reproduction Commands |
| output list structure | list of exactly 4 fp32 tensors each `[30522]`, in `seq_lens` order | Independent probe: `len=4`, each `[30522]` fp32, order preserved | pass | independent probe (max_abs ~1.19e-07) |
| log1p(relu(x)) semantics | `log(1+max(x,0))` per element | All 4 outputs allclose vs manual `torch.log1p(torch.relu(x))` reference | pass | independent probe |
| per-sequence max-pool segmentation | column-wise max over `[offset:offset+L]` for each of 4 seq lens `[20,25,18,20]` | On-device prefix-scan offsets correct; all 4 outputs match reference | pass | independent probe |
| GEMM/GELU/LayerNorm unchanged | dense/decoder GEMM on TCU, GELU/LayerNorm library ops | Kernel profile confirms `gemm_tcu_h` + `GEMM_Epilogue` + `vectorized_layer_norm_kernel` + `GeluCUDAKernelImpl` still present, unchanged | pass | profiler candidate scope |
| input not mutated | `forward` must not modify `hidden_states`/`seq_lens` | Candidate reads `logits` and `seq_lens` read-only; no in-place op | pass | candidate source review |
| frozen artifact identity | candidate hash equals Orchestrator value | `f3fd85a2...` matches | pass | SHA256 in round_status_001.md |

The independent numerical probe loaded both `base.Model` and the candidate
`ModelNew` through the harness AST loader, fed the same input, and compared each
of the 4 list outputs. All 4 were `allclose(atol=1e-2, rtol=1e-2, equal_nan=True)`
with `max_abs ≈ 1.19e-07` (near bit-exact; the `tl.log(1+x)` lowering matches
`torch.log1p` to float32 precision). The per-sequence max-pool segmentation and
output list structure are exact.

## Screening Evidence

Not run: correctness passed, so the candidate proceeded directly to
authoritative timing (per contract, screening is only a pre-filter for
correctness-passing candidates; it does not gate an otherwise-correct candidate).

## Interleaved Wall Timing

- warmup: `50`
- repeat: `100`
- order: `interleaved accepted-reference/candidate`
- reference_raw_samples_ms: `[1.055067, 1.060573, 1.060911]`
- candidate_raw_samples_ms: `[0.880377, 0.879838, 0.885816]`
- reference_median_ms: `1.060573`
- candidate_median_ms: `0.880377`
- improvement_pct: `16.99043818765894`

```text
improvement_pct = (reference_median_ms - candidate_median_ms) / reference_median_ms * 100
               = (1.060573 - 0.880377) / 1.060573 * 100 = 16.99%
```

| Independent invocation | Reference wall ms | Candidate wall ms | Command return code |
|---:|---:|---:|---:|
| 1 | `1.055067` | `0.880377` | `0` |
| 2 | `1.060573` | `0.879838` | `0` |
| 3 | `1.060911` | `0.885816` | `0` |

`improvement_pct = 16.99% >= 5.0%` threshold, correctness PASS, all guardrails
pass → `accepted`.

Note on the reference: `baseline_adapter.py` defines `ModelNew` while the harness
v0 slot requires `Model`. A byte-identical `ModelNew → Model` rename wrapper
(`/tmp/sp_baseline_model_001.py`, SHA `1edaf2ad...`) served as the v0 reference;
it is semantically identical to `baseline_adapter.py` and was deleted after
measurement.

## Evaluation Contract Mirror

- Evaluation Contract applicability: `required`
- hypothesis_id: `H-001`
- intervention: `fuse the post-decoder SPLADE activation chain (ReLU via clamp + log1p) and the per-sequence max-pooling into a single fused Triton kernel, eliminating the intermediate [83,30522] activations and their full-tensor read/write traffic, while leaving the dense and decoder GEMMs on the vendor TCU`
- expected_causal_chain: `clamp_scalar + log1p disappear; four max-reduce kernels collapse into in-kernel column-wise max; kernel count decreases 11.92 → ~6; device_us_per_call decreases; wall time decreases`
- primary_metric: `wall_time`
- Hypothesis verdict: `confirmed`

| Observable | Expectation | Observation | Verdict | Evidence |
|---|---|---|---|---|
| `kernel_count_per_call` | decrease | `11.92 → 6.88` (6 tail kernels collapsed to 1 fused kernel) | pass | profiler candidate scope |
| `device_us_per_call` | decrease | `743.80 → 609.40` us/call (~134 us removed) | pass | profiler candidate scope |

Both declared mechanism observables are confirmed. The intervention's causal
chain held: the `clamp_scalar` (21.3 us), `log1p` (33.8 us), and 4×
`reduce_kernel<MaxOps>` (88.8 us) kernels — ~144 us/call of tail work — collapsed
into a single `_sparse_pooler_fused_kernel` at ~28.3 us/call. Additionally, the
candidate removed the baseline's per-call `seq_lens.tolist()` device-to-host sync
(50× `Memcpy DtoH` + `cudaStreamSynchronize` in the reference scope, 0 in the
candidate scope), which explains why the 16.99% wall improvement exceeds the
~18% device-time improvement alone would suggest.

## Profiler Evidence

- profiler_applicability: `required`
- profiler_level: `targeted`
- profiler_device_time: `available` (`cat=kernel` durations)
- profile_mode: `forward`
- warmup: `20`
- iterations: `50` forward calls per scope
- scopes: `reference_baseline_adapter`, `candidate_triton_sparse_pooler_001`
- normalized_fields: `device_total_us`, `device_us_per_call`, `kernel_count_total`, `kernel_count_per_call`, `device_ratio`, `kernels`
- raw trace: `log/round_001_forward_50iter.pt.trace.json`, SHA256 `fda3dc194770f2439988967bc58edcea9b9bb8eaa235e6d14e07e76933f99754`

| Scope | Device total us | Device us/call | Kernel count total | Kernel count/call | Wall ms | Device ratio |
|---|---:|---:|---:|---:|---:|---:|
| `reference_baseline_adapter` | `37189.832` | `743.797` | `596` | `11.92` | `1.060573` | `0.70132` |
| `candidate_triton_sparse_pooler_001` | `30469.863` | `609.397` | `344` | `6.88` | `0.880377` | `0.69224` |

```text
device_ratio = device_us_per_call / (wall_median_ms * 1000)
reference  = 743.797 / 1060.573 ≈ 0.7013
candidate  = 609.397 /  880.377 ≈ 0.6922
```

### Reference Top Kernels (reference_baseline_adapter scope)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `Gemm_tcu_mr_kernel::gemm_tcu_h<64u,64u,64u,16u,16u,2u,...>` (dense + decoder GEMM) | `100` | `2.0` | `24965.437` | `499.309` |
| `reduce_kernel<1024,1,ReduceOp<float,MaxOps>>` (per-sequence max-pool) | `196` | `3.92` | `4421.632` | `88.433` |
| `GEMM_Epilogue<float,...>` (Linear bias-add) | `100` | `2.0` | `4168.120` | `83.362` |
| `elementwise_kernel<log1p_kernel_cuda...>` (SPLADE log1p) | `50` | `1.0` | `1679.175` | `33.584` |
| `elementwise_kernel<launch_clamp_scalar...>` (ReLU) | `50` | `1.0` | `1069.637` | `21.393` |
| `vectorized_layer_norm_kernel<float,float>` (LayerNorm) | `50` | `1.0` | `509.167` | `10.183` |
| `elementwise_kernel<GeluCUDAKernelImpl...>` (GELU) | `50` | `1.0` | `376.664` | `7.533` |

### Candidate Top Kernels (candidate_triton_sparse_pooler_001 scope)

| Kernel | Count total | Count/call | Total us | Us/call |
|---|---:|---:|---:|---:|
| `Gemm_tcu_mr_kernel::gemm_tcu_h<64u,64u,64u,16u,16u,2u,...>` (dense + decoder GEMM) | `98` | `1.96` | `24126.7` | `482.53` |
| `GEMM_Epilogue<float,...>` (Linear bias-add) | `98` | `1.96` | `4064.5` | `81.29` |
| `_sparse_pooler_fused_kernel` (fused log1p(relu) + per-seq max-pool) | `50` | `1.0` | `1417.0` | `28.34` |
| `vectorized_layer_norm_kernel<float,float>` (LayerNorm) | `49` | `0.98` | `495.7` | `9.91` |
| `elementwise_kernel<GeluCUDAKernelImpl...>` (GELU) | `49` | `0.98` | `365.9` | `7.32` |

### Fusion Observation

The six tail kernels in the reference — `clamp_scalar` (21.3 us), `log1p` (33.8
us), and 4× `reduce_kernel<MaxOps>` (88.8 us, ~144 us/call combined) — are
replaced by a single `_sparse_pooler_fused_kernel` at `1.0`/call and `28.34`
us/call. Kernel count drops from `11.92` to `6.88` per call (5 distinct kernels
vs 7). The GEMM (`gemm_tcu_h`) and its `GEMM_Epilogue` remain on the vendor TCU,
unchanged, and now dominate the profile (482.53 + 81.29 = ~563.8 us/call ≈ 92.5%
of candidate device time).

### D2H Sync Elimination Observation (additional mechanism)

The reference `baseline_adapter.py` calls `seq_lens.tolist()` in its forward
path, which forces a device-to-host copy and a stream synchronization on every
forward call: the reference scope shows `50× Memcpy DtoH (Device -> Pageable)`,
`50× cudaMemcpyAsync`, and `50× cudaStreamSynchronize` (one per forward call).
The candidate replaces this with an on-device bounded prefix scan over
`seq_lens`, so the candidate scope has **zero** D2H memcpy and **zero** stream
sync (only the profiler's own terminal `cudaDeviceSynchronize`). This host-side
sync removal is an additional, attributable wall-time win beyond the device-time
saving.

## Retry History

| Attempt | Trigger | Candidate before SHA256 | Candidate after SHA256 | Outcome |
|---:|---|---|---|---|
| 1 | Initial correctness, timing, and profiler verification | `f3fd85a2c913d477e2cac7f65ed1f79dd5e1b9a3a60481782dbb4acaa43d2d98` | same | correctness and wall timing passed; profiler summarized (candidate scope via outer interval due to overlapping nested record_function) |

No candidate repair occurred and no source file changed.

## evidence_for_next_round

- Canonical accepted kernel is now `triton_sparse_pooler_001.py` (wall median `0.880377 ms`, improvement `16.99%` over `baseline_adapter.py`), pending Orchestrator's canonical-pointer update.
- Activation + pooling fusion is `confirmed`: 6 tail kernels collapsed to 1 `_sparse_pooler_fused_kernel`, kernel count `11.92 → 6.88`, device time `743.80 → 609.40 us/call`.
- Removing `seq_lens.tolist()` (D2H sync) was an additional attributable win: reference had `50× Memcpy DtoH` + `50× cudaStreamSynchronize` per profile run, candidate had zero.
- The GEMM is now even more dominant: `gemm_tcu_h` + `GEMM_Epilogue` ≈ 563.8 us/call ≈ 92.5% of remaining device time. The remaining optimization target is the dense (768×768) and decoder (768×30522) GEMMs, currently on the vendor TCU. A fp32 large-N `tl.dot` rewrite is still unproven on this profile (only `(32,32)@(32,32)` recorded) and remains a high capability-miss risk; it would require a matched local probe before any decision.
- The `_sparse_pooler_fused_kernel` at 28.34 us/call is already efficient; further tail-side gains are negligible. The `GEMM_Epilogue` bias-add (81.29 us/call) is a separable epilogue that could be fused into a future GEMM rewrite but is coupled to the GEMM itself.
- Profiler note: Triton's `cuLaunchKernel` instrumentation emits a nested duplicate `record_function` for the candidate scope, which trips `summarize_trace.py`'s overlap guard; the candidate scope must be summarized against the outer enclosing interval. This is a measurement-tooling quirk, not a kernel defect.

## Stop Recommendation

- recommendation: `continue`
- evidence: Round 001 accepted at `16.99%` wall improvement (above 5% threshold). No optional target is configured. The remaining dominant bottleneck is the two TCU GEMMs (~92.5% of device time), which is a future, separately-probed hypothesis. `total_rounds=1` is far below `max_rounds=20`.

Orchestrator owns canonical pointer updates and workflow transition.

## Exact Reproduction Commands

Environment bootstrap (every command):

```bash
export COREX_VERSION=4.4.0
. /usr/local/corex/enable
```

Frozen-file SHA256 verification (all returned code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && sha256sum kernels/track1-triton/sparse_pooler/bi150/triton_sparse_pooler_001.py kernels/track1-triton/sparse_pooler/bi150/baseline_adapter.py kernels/track1-triton/sparse_pooler/bi150/rounds/decision_001.md
```

Correctness (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file kernels/track1-triton/sparse_pooler/base.py --v1_file kernels/track1-triton/sparse_pooler/bi150/triton_sparse_pooler_001.py --warmup 50 --repeat 100 --full-traceback
```

Baseline reference wrapper (byte-identical `ModelNew → Model` rename; SHA `1edaf2ad...`):

```bash
sed 's/^class ModelNew/class Model/' kernels/track1-triton/sparse_pooler/bi150/baseline_adapter.py > /tmp/sp_baseline_model_001.py
```

Wall timing (execute independently three times; return codes `0, 0, 0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file /tmp/sp_baseline_model_001.py --v1_file kernels/track1-triton/sparse_pooler/bi150/triton_sparse_pooler_001.py --warmup 50 --repeat 100
```

Targeted profiler (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 auto_bench.py --v0_file /tmp/sp_baseline_model_001.py --v1_file kernels/track1-triton/sparse_pooler/bi150/triton_sparse_pooler_001.py --profile --profile-mode forward --profile-warmup 20 --profile-iterations 50 --profile-reference-file kernels/track1-triton/sparse_pooler/bi150/baseline_adapter.py --profile-output kernels/track1-triton/sparse_pooler/bi150/log/round_001_forward_50iter.pt.trace.json
```

Reference scope summary (return code `0`):

```bash
cd /root/CodeBuddy/20260818191200/kernelswift && python3 skills/kernel-opt-loop/scripts/summarize_trace.py kernels/track1-triton/sparse_pooler/bi150/log/round_001_forward_50iter.pt.trace.json --iterations 50 --scope reference_baseline_adapter --wall-ms 1.060573
```

## Command Status and Raw Evidence Index

| Command | Return code | Evidence |
|---|---:|---|
| frozen-file SHA256 | `0` | hashes in Identity |
| correctness (base vs candidate) | `0` | report Correctness table |
| independent numerical probe | `0` | report Correctness table (max_abs ~1.19e-07) |
| baseline wrapper creation | `0` | `/tmp/sp_baseline_model_001.py` SHA `1edaf2ad...` |
| wall sample 1, 50/100 | `0` | report Interleaved Wall Timing |
| wall sample 2, 50/100 | `0` | report Interleaved Wall Timing |
| wall sample 3, 50/100 | `0` | report Interleaved Wall Timing |
| targeted profiler 20/50 | `0` | `log/round_001_forward_50iter.pt.trace.json` |
| summarize reference scope | `0` | report Profiler Evidence |
| summarize candidate scope (outer interval) | `0` | report Profiler Evidence |

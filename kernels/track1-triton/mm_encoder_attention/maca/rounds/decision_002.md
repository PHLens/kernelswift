# Decision 002

## Metadata

```json
{"schema_version":1,"decision":"proceed","round":"002","reference_implementation":"triton_mha_001.py","reference_report":"rounds/report_001.md","language":"triton","backend":"maca","target_profile":"triton_maca","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"remove-transpose-copy"}
```

## Optimization Intent

```json
{"bottleneck_class":"device-bound","intervention":"eliminate the four transpose12_copy_64 copy kernels by removing the .contiguous() materialization of q/k/v in the benchmark path and instead having the Triton kernel load directly from the original [bsz, seq_len, hidden] (equivalently [bsz, seq_len, heads, head_size]) contiguous layout using explicit per-tensor strides computed on the host","allowed_changes":["_mha_fwd_kernel address computation","ModelNew.forward benchmark-path layout handling"],"invariants":["ModelNew public contract","output dtype and shape","no input mutation","caller-selected device and current stream preserved","fp16 SDPA fallback preserved for non-benchmark shapes","fp16 in/out and fp32 accumulation numerics unchanged"],"expected_wall_improvement_pct":7.0}
```

## Unified Sketch

```sketch
# D Declarations
tensor query shape=[bsz,seq,heads,head_size] dtype=fp16 layout=contiguous memory=global
tensor key shape=[bsz,seq,heads,head_size] dtype=fp16 layout=contiguous memory=global
tensor value shape=[bsz,seq,heads,head_size] dtype=fp16 layout=contiguous memory=global
tensor output shape=[bsz,heads,seq,head_size] dtype=fp16 layout=contiguous memory=global
tile q_row shape=[head_size] dtype=fp32 memory=register
tile acc shape=[head_size] dtype=fp32 memory=register
scalar m_i dtype=fp32 memory=register
scalar l_i dtype=fp32 memory=register

# O Operations
load q_row <- query[b,q_row_index,h,0:head_size]
alloc acc shape=[head_size] dtype=fp32 init=0
compute m_i = -inf
compute l_i = 0
load k_j <- key[b,j,h,0:head_size]
compute s_j = sum(q_row * k_j) * scale
compute m_new = max(m_i, s_j)
compute l_i = l_i * exp(m_i - m_new) + exp(s_j - m_new)
load v_j <- value[b,j,h,0:head_size]
compute acc = acc * exp(m_i - m_new) + exp(s_j - m_new) * v_j
compute m_i = m_new
compute acc = acc / l_i
store output[b,h,q_row_index,0:head_size] <- acc

# C Control
parallel program over (bsz * heads * seq)
guard program < (bsz * heads * seq)
for j over seq
guard j < seq
end

# H Target Hints
target=triton_maca
num_warps=1
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only change; the .contiguous() removal changes layout handling inside the benchmark path but introduces no host-side cache, reuse, or state mutation"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-002","intervention":"eliminate the four transpose12_copy_64 copy kernels by removing the .contiguous() materialization of q/k/v in the benchmark path and instead having the Triton kernel load directly from the original [bsz, seq_len, hidden] (equivalently [bsz, seq_len, heads, head_size]) contiguous layout using explicit per-tensor strides computed on the host","expected_causal_chain":["the four transpose12_copy_64 copy kernels disappear because q/k/v are no longer materialized contiguous","total device kernel count per call drops from 5.0 to 1.0","device time per call drops from ~79.7 us to ~67.1 us (the four copies total ~12.6 us/call)","wall time decreases accordingly"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"candidate_kernel_count_per_call","expectation":"drops from 5.0 to 1.0 (transpose12_copy_64 count/call goes 4.0 -> 0.0; _mha_fwd_kernel stays 1.0)"},{"name":"candidate_device_us_per_call","expectation":"drops from ~79.7 us to ~67.1 us (transpose12_copy_64 ~12.6 us/call eliminated)"},{"name":"correctness_parity","expectation":"candidate output allclose reference (atol=1e-2, rtol=1e-2, equal_nan=True) = pass"}],"guardrails":["correctness:pass","output dtype and shape unchanged","no input mutation","caller-selected device and current stream preserved","SDPA fallback preserved for non-benchmark shapes"],"profiling_level":"targeted"}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`. The four catalogued failures (winner-tree, sort-32/sort-64 networks, dynamic gather compaction, cumsum compaction) are all MLU grouped-top-k selection failures and do not bear on this fp16 MHA layout change. No matching entry invalidates this path.
- Consulted `references/bottleneck-judgment.md`. Round 001 device ratio is ~0.485 (device-bound), and the dominant device costs are the fused `_mha_fwd_kernel` (~67.1 us/call) plus four `transpose12_copy_64` copy kernels (~12.6 us/call). The copy kernels are candidate-owned device work with a clean, low-risk removal path; this is exactly the kind of "redundant work / fusion" lever the bottleneck judgment identifies for device-bound kernels.
- `tl.dot` remains Unknown on triton_maca and MUST NOT be used; the 64-element dot stays an explicit `tl.sum` reduction over the elementwise product in fp32. This decision does not change the dot primitive.
- The address-computation change must preserve fp32 accumulation numerics exactly (same manual-dot + two-pass max-subtracted softmax as `triton_mha_001.py`); only the source layout and offsets change, not the math. This keeps correctness parity within the loose 1e-2 tolerance.
- `tl.zeros`/`tl.full` remain Unknown: the running-softmax seed must continue to be derived from the first key position (as in `triton_mha_001.py`) rather than from a zero/constant-init primitive.
- Target warp size is 64; `num_warps=1` remains the only proven launch configuration.

## Rationale and Evidence

The accepted report `rounds/report_001.md` establishes that the Round 001 candidate `triton_mha_001.py` achieves correctness parity (allclose atol=1e-2, rtol=1e-2) but emits **5.0 device kernels per call** instead of the intended 1.0: the fused `_mha_fwd_kernel` (1.0/call, ~67.1 us/call) plus **four `transpose12_copy_64` copy kernels** (4.0/call, ~12.6 us/call total) arising from `.contiguous()` on the reshaped/transposed q, k, v tensors.

The four copy kernels are pure redundant work. The root cause is in `triton_mha_001.py` forward benchmark path:

```python
q = query.view(bsz, q_len, self.num_heads, self.head_size).transpose(1, 2).contiguous()
```

`query` is `[bsz, seq_len, hidden]` contiguous with `hidden = heads * head_size`. The `.view(...)` to `[bsz, seq_len, heads, head_size]` is free (no copy), but `.transpose(1, 2)` makes `[bsz, heads, seq_len, head_size]` non-contiguous, and `.contiguous()` then materializes a copy — one `transpose12_copy_64` kernel each for q, k, and v (and a fourth from the output's `.transpose(1,2).reshape(...)` back-path, or from the value tensor's dual role; report_001 attributes 4.0/call).

**Intervention (Direction A, the only change this round):** remove `.contiguous()` and have `_mha_fwd_kernel` load directly from the original contiguous `[bsz, seq_len, heads, head_size]` layout by computing per-tensor offsets with the host-supplied stride. Specifically, the host passes the original contiguous q/k/v (no transpose, no contiguous), and the kernel computes:

```text
q_off  = b * (seq_len * heads * head_size) + row * (heads * head_size) + h * head_size
kv_off = b * (seq_len * heads * head_size) + j  * (heads * head_size) + h * head_size
```

(with `seq_len` and `head_size` as constexpr and `heads` from the pid decomposition), so `q_row = Q[q_off + offs_d]` and `k_j = K[kv_off + offs_d]` directly index the original layout. The output is written to a freshly allocated contiguous `[bsz, heads, seq_len, head_size]` buffer and reshaped/transposed on the host back to `[bsz, seq_len, hidden]` (the output materialization is a single unavoidable reshape, matching base.py's own output path). `tl.load` is Supported and accepts these non-power-of-2 strided offsets (the `seq_len=83` and `head_size=64` are constexpr; no masking needed because every position in the static loop is valid).

Expected result: kernel count 5.0 → 1.0, device time ~79.7 → ~67.1 us/call, wall time correspondingly reduced. The ~12.6 us/call is ~7.7% of the current 0.164 ms wall; eliminating it is a concrete, attributable, low-risk win.

**Direction B (single-pass online softmax / deeper kernel optimization) is explicitly deferred** to a future round. Reasons: (1) the invariants require each round to change exactly one attributable intervention, and Direction A is the cleanest single lever; (2) Direction B is higher-risk and its benefit is speculative — the dominant `_mha_fwd_kernel` cost is the manual `tl.sum` dot over `head_size=64` (repeated 83 positions, and again in the second pass), which cannot be improved without `tl.dot` (Unknown on C500); converting the two-pass max-subtracted softmax to a single-pass online softmax would remove the second K re-load but introduces register-pressure and rescaling complexity for an uncertain gain, and would not address the fundamental manual-dot cost. It is recorded as a candidate future family, not part of this round.

Correctness constraints the Coder must honor (normative in the Unified Sketch and guardrails):

1. **Layout-only change, math unchanged.** The manual-dot + two-pass max-subtracted softmax numerics are byte-for-byte the same as `triton_mha_001.py`; only the source offset computation changes. fp16 in/out, fp32 accumulation, final fp16 cast are all preserved.
2. **Direct strided load, no `.contiguous()`.** q/k/v are loaded from the original `[bsz, seq_len, hidden]` contiguous tensor via `tl.load` with the stride-based offsets above; no transpose/contiguous materialization may be introduced in the benchmark path.
3. **No `tl.dot`.** The 64-element dot remains `scale * tl.sum(q_row * k_j)` in fp32.
4. **No zero/constant init.** Running-softmax `m` and `l` are still seeded from the first key position, not from `tl.zeros`/`tl.full`.
5. **No input mutation.** The kernel only reads q/k/v; output goes to a fresh buffer.
6. **Fallback preserved.** Non-benchmark shapes keep the verbatim `F.scaled_dot_product_attention` path.
7. **Public contract.** `ModelNew.__init__` and `forward` signatures unchanged; caller-selected device/stream preserved; output `[bsz, seq_len, hidden]` fp16 contiguous.

Best-effort performance expectation: this round removes ~12.6 us/call of redundant copy work and reduces kernel count from 5 to 1, a concrete wall-time improvement of roughly 7% that is attributable and verifiable. It is not expected to beat flash attention (the fused kernel remains ~67 us vs flash ~15 us), which is acceptable under the epoch-2 policy — the goal is "optimize as much as possible" on the Triton deliverable, and each removed kernel / reduced device time is valuable.

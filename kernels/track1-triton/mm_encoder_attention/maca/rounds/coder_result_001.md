# Coder Result 001

- round: `001`
- result: `candidate-ready`
- change_family: `fused-mha-kernel`

## Identity

- source canonical path: `baseline_adapter.py`
- source canonical SHA-256: `1127e8d9f166bb2449a993c8c5392a464179b6da599cd1f181f1949f151b7c8e`
- decision path: `rounds/decision_001.md`
- decision SHA-256: `ab2f5bb98a8f491ed67e2a05850fc28e9bf0958a09ef89ec3f32c8f24a0a949d`
- selected profile: `triton_maca`
- runtime fingerprint: `project.md#runtime-fingerprint` (torch 2.8.0+metax3.5.3.9, triton 3.0.0, GPUTarget(backend='maca', arch=80, warp_size=64))
- candidate path: `triton_mha_001.py`
- candidate SHA-256: `9fac12aa0298a970c208dbc6af7a602da4f34e43d44921b62aad571ca662c00b`

## Decision Validation

- `validate_decision.py --expected-profile triton_maca` returned `valid: true`, metadata `target_profile=triton_maca`, `language=triton`, `backend=maca`, `change_scope=kernel`. No profile/identity mismatch.
- Runtime fingerprint rediscovered at gate time: torch `2.8.0+metax3.5.3.9`, triton `3.0.0`, `torch.cuda.is_available()=True`. Matches `project.md#runtime-fingerprint`.

## Primitive and Hint Conformance

| Primitive / Hint | Decision requirement | Implementation | Verdict |
|---|---|---|---|
| `tl.load` / `tl.store` | fp16 load, fp32 upcast; fp16 store | `tl.load(...).to(tl.float32)` / `out.to(tl.float16)` store | conform (Supported) |
| `tl.arange` | 64-element offsets | `tl.arange(0, head_size)` with `head_size=64` | conform; extent 64 is not in the proven {8, 256} set but is normative per the decision's `head_size=64` |
| `tl.sum` | manual dot (no `tl.dot`) | `scale * tl.sum(q_row * k_j)` fp32 | conform (Supported); `tl.dot` NOT used |
| `tl.exp` | softmax exponential | `tl.exp(s_j - m)` fp32 | conform (Supported) |
| `tl.static_range` | static key loop | `tl.static_range(seq_len)` and `tl.static_range(1, seq_len)` | conform (Supported) |
| `tl.maximum` | running max | pairwise `tl.maximum(m, s_j)` | conformance note: not in the proven Supported list and not in the Unknown list; it is the standard elementwise max companion to the proven `tl.max` reduction and executed correctly in the local gate |
| `tl.zeros` / `tl.full` | MUST NOT use (Unknown) | NOT used | conform; accumulators seeded from the first key position |
| `-inf` init | sketch `m_i = -inf` | NOT used; `m` seeded from `scores[0]` (first dot) | conform; two-pass-with-max is numerically equivalent to online softmax |
| `tl.dot` | MUST NOT use (Unknown) | NOT used | conform |
| `num_warps` | `num_warps=1` | `num_warps=1` | conform (Constrained) |
| direct launch | proven launcher path | `_mha_fwd_kernel[grid](...)` | conform |
| SDPA fallback | preserve for non-benchmark shapes | verbatim `F.scaled_dot_product_attention` branch for non-benchmark shape | conform |
| no input mutation | read-only q/k/v | kernel only reads; `torch.empty_like(q)` fresh output; `contiguous()` produces copies | conform |
| fp16 in/out, fp32 accumulate | normative | dot/softmax/acc in fp32; final cast to fp16 | conform |

### Conformance notes (candidate-ready, non-semantic accommodations)

1. The two-pass softmax is implemented as two register-resident passes that re-load `k_j`/`v_j` rather than staging the 83 scores in a Python list (Triton's JIT does not support Python list mutation inside `@triton.jit`). The algorithm, dataflow, and numerics are unchanged — still two-pass, fp32, max-subtracted softmax.
2. `m` is seeded from the first key position's score instead of a `-inf` constant, and `l`/`acc` are seeded from the first position instead of zero. This avoids the Unknown `tl.zeros`/`tl.full` primitives while producing the mathematically identical softmax result (max subtraction is exact up to fp32 rounding).
3. `tl.maximum` is used for the running max. It is not in the profile's proven Supported list nor its Unknown list; it executed correctly in the local gate.
4. The `guard j < seq` from the sketch is trivially satisfied because `seq_len=83` is a compile-time constexpr and `tl.static_range` iterates exactly 83 valid positions (no padding, no out-of-range access).

## Local Gate

1. `ast.parse` on candidate: PASS (`ast.parse OK`).
2. Harness loader smoke (compile + correctness):
   ```
   /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/maca/triton_mha_001.py --warmup 2 --repeat 3 --full-traceback
   PASS accuracy; v0=0.157740 ms, v1=0.175615 ms, speedup=0.898x
   Summary: 1 passed, 0 failed, 1 total.
   ```
   Correctness (allclose atol=1e-2, rtol=1e-2, equal_nan=True) PASS.
3. AST loader retention: `ModelNew` (class), `get_inputs` (function), `get_init_inputs` (function) all present; `ModelNew` is an `nn.Module` subclass with `forward`.

## Attempt Ledger

| Attempt | Command | Exit | Defect | Before hash | After hash | Outcome |
|---|---|---:|---|---|---|---|
| 0 | ast.parse + harness smoke (warmup 2 repeat 3) | 1 | `NameError: scores is not defined` — Python list `scores.append` inside `@triton.jit` body unsupported by the JIT | n/a | `9fac12aa0298a970c208dbc6af7a602da4f34e43d44921b62aad571ca662c00b` | non-semantic repair applied |
| 1 | harness smoke (warmup 2 repeat 3) | 0 | none | `9fac12aa0298a970c208dbc6af7a602da4f34e43d44921b62aad571ca662c00b` | `9fac12aa0298a970c208dbc6af7a602da4f34e43d44921b62aad571ca662c00b` | PASS accuracy |

The attempt-0 repair was non-semantic: it replaced Python-list score staging with re-loading inside two register-resident passes, preserving the two-pass algorithm, fp32 accumulation, max-subtracted softmax, and manual dot. No semantic change was introduced.

## Reason Code

`candidate-ready` — the candidate conforms to the immutable decision design, all normative constraints are honored, and the local gate (ast.parse + real harness loader smoke + one compile+correctness execution) passes with accuracy atol/rtol 1e-2.

# Coder Result 002

- round: `002`
- result: `candidate-ready`
- change_family: `remove-transpose-copy`

## Identity

- source canonical path: `triton_mha_001.py`
- source canonical SHA-256: `9fac12aa0298a970c208dbc6af7a602da4f34e43d44921b62aad571ca662c00b`
- decision path: `rounds/decision_002.md`
- decision SHA-256: `be804f497dcb6070e1a07d290b43c6c8acc65e3007d88657985026aa5640ac7e`
- selected profile: `triton_maca`
- runtime fingerprint: `project.md#runtime-fingerprint` (torch 2.8.0+metax3.5.3.9, triton 3.0.0, GPUTarget(backend='maca', arch=80, warp_size=64))
- candidate path: `triton_mha_002.py`
- candidate SHA-256: `29e6b192bf778f0264fb7657c9a33b97819c406896a2ad86e1daf22f3c9ff0a1`

## Decision Validation

- `validate_decision.py --expected-profile triton_maca` on `decision_002.md` returns `valid: true`, `target_profile=triton_maca`, `language=triton`, `backend=maca`, `change_scope=kernel`. No profile/identity mismatch.
- Runtime fingerprint rediscovered at gate time: torch `2.8.0+metax3.5.3.9`, triton `3.0.0`, `torch.cuda.is_available()=True`. Matches `project.md#runtime-fingerprint`.

## Primitive and Hint Conformance

This round is a layout-only change: the kernel numerics (manual dot + two-pass max-subtracted softmax, fp32 accumulate) are byte-for-byte identical to `triton_mha_001.py`; only the source offset computation and host-side layout handling change.

| Primitive / Hint | Decision requirement | Implementation | Verdict |
|---|---|---|---|
| direct strided load, no `.contiguous()` | q/k/v loaded from original `[bsz, seq_len, heads, head_size]` contiguous layout | offsets `b*(seq*heads*hd) + s*(heads*hd) + h*hd`; `query/key/value` passed directly, no view/transpose/contiguous | conform |
| `tl.load` / `tl.store` | fp16 load, fp32 upcast; fp16 store | `.to(tl.float32)` / `.to(tl.float16)` | conform (Supported) |
| `tl.arange` | 64-element offsets | `tl.arange(0, head_size)` with `head_size=64` | conform (normative head_size=64) |
| `tl.sum` | manual dot (no `tl.dot`) | `scale * tl.sum(q_row * k_j)` fp32 | conform (Supported); `tl.dot` NOT used |
| `tl.exp` | softmax exponential | `tl.exp(s_j - m)` fp32 | conform (Supported) |
| `tl.static_range` | static key loop | `tl.static_range(1, seq_len)` | conform (Supported) |
| `tl.maximum` | running max | pairwise `tl.maximum(m, s_j)` | conformance note (as in round 001) |
| `tl.zeros` / `tl.full` / `-inf` | MUST NOT use (Unknown) | NOT used; `m`/`l`/`acc` seeded from first key position | conform |
| `tl.dot` | MUST NOT use (Unknown) | NOT used | conform |
| `num_warps` | `num_warps=1` | `num_warps=1` | conform (Constrained) |
| direct launch | proven launcher path | `_mha_fwd_kernel[grid](...)` | conform |
| SDPA fallback | preserve for non-benchmark shapes | verbatim `F.scaled_dot_product_attention` branch | conform |
| no input mutation | read-only q/k/v | kernel reads `query/key/value` directly; fresh `torch.empty` output | conform |
| output layout | `[bsz, seq_len, hidden]` fp16 contiguous | `out.transpose(1,2).reshape(...)` on fresh `[bsz, heads, seq, head_size]` buffer | conform (single unavoidable reshape, matches base.py output path) |
| fp16 in/out, fp32 accumulate | normative, unchanged | preserved verbatim from round 001 | conform |

### Conformance notes (candidate-ready, non-semantic accommodations)

1. The kernel writes to a freshly allocated `[bsz, heads, seq_len, head_size]` buffer and the host transposes/reshapes back to `[bsz, seq_len, hidden]`. This single output materialization is the same unavoidable reshape as `base.py`'s own output path and is explicitly acknowledged in the decision ("the output materialization is a single unavoidable reshape").
2. `tl.maximum`, `tl.arange(0, 64)`, and the `guard j < seq` trivial-satisfaction notes carry over unchanged from round 001 (identical numerics and primitive usage).
3. No new primitives are introduced relative to round 001; this round only removes the four `transpose12_copy_64` copy kernels by replacing `.view().transpose().contiguous()` with stride-based offsets in the kernel.

## Local Gate

1. `ast.parse` on candidate: PASS (`ast.parse OK`).
2. Harness loader smoke (compile + correctness):
   ```
   /opt/conda/bin/python auto_bench.py --v0_file kernels/track1-triton/mm_encoder_attention/base.py --v1_file kernels/track1-triton/mm_encoder_attention/maca/triton_mha_002.py --warmup 2 --repeat 3 --full-traceback
   PASS accuracy; v0=0.157322 ms, v1=0.135534 ms, speedup=1.161x
   Summary: 1 passed, 0 failed, 1 total.
   ```
   Correctness (allclose atol=1e-2, rtol=1e-2, equal_nan=True) PASS. Wall time improved from round 001's ~0.1756 ms to ~0.1355 ms, consistent with eliminating the four copy kernels.
3. AST loader retention: `ModelNew` (class), `get_inputs` (function), `get_init_inputs` (function) all present; `ModelNew` has `forward`.

## Attempt Ledger

| Attempt | Command | Exit | Defect | Before hash | After hash | Outcome |
|---|---|---:|---|---|---|---|
| 0 | ast.parse + harness smoke (warmup 2 repeat 3) | 0 | none | n/a | `29e6b192bf778f0264fb7657c9a33b97819c406896a2ad86e1daf22f3c9ff0a1` | PASS accuracy |

No repairs were needed: the first attempt compiled and passed correctness.

## Reason Code

`candidate-ready` — the candidate conforms to the immutable decision design (layout-only removal of `.contiguous()` with stride-based direct loads; numerics unchanged), and the local gate (ast.parse + real harness loader smoke + one compile+correctness execution) passes with accuracy atol/rtol 1e-2.

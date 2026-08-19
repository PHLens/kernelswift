# Coder Result 001

## Identity

- round: `001`
- task: `mhc_post_layer_mix` (BI150 backend, track1-triton)
- decision path: `kernels/track1-triton/mhc_post_layer_mix/bi150/rounds/decision_001.md`
- decision SHA-256: `335389df2498f37fb9f2c5c7ebc10986ab4edf555d939525413900e0e885ecfc`
- canonical source (reference implementation): `baseline_adapter.py`
- canonical SHA-256: `66a3a2c31863d18c725a52ab57fd1b9f89fe655dd7bab7cb4da158b8130b5d07`
- candidate path: `kernels/track1-triton/mhc_post_layer_mix/bi150/triton_mhc_post_layer_mix_001.py`
- candidate SHA-256: `08a9d59f17ffa80224943b19bdcce390d908ca8ba15bf2e06ae469f45787d9fb`
- target profile: `triton_cuda`
- language: `triton`; backend: `cuda`

## Runtime Fingerprint

- triton_distribution: `corex`; triton_version: `3.1.0`
- backend_target: `cuda`; backend_version: `2.7.1` (torch 2.7.1)
- device_arch: `cuda:0 (Iluvatar BI-V150)`, capability `(7, 1)`, 16 SM, 16 GiB
- probe: `import torch,triton -> (2.7.1, 3.1.0, 'Iluvatar BI-V150', (7,1))`
- match: `pass` (matches project.md#runtime-fingerprint)

## Result

`candidate-ready`

## Implementation

Fused the post-GEMM elementwise tail into a single `@triton.jit` kernel
`_fused_tail_kernel`, preserving the unchanged `torch.einsum` GEMM for `term2`.

`ModelNew.forward` structure:

```python
def forward(self, x, residual, post_layer_mix, comb_res_mix):
    term2 = torch.einsum('abmn,abmc->abnc', comb_res_mix, residual.float())  # unchanged
    C = x.shape[2]
    total = 2 * 4096 * 4 * C
    BLOCK = 1024
    grid = (triton.cdiv(total, BLOCK),)
    out = torch.empty((2, 4096, 4, C), dtype=torch.bfloat16, device=x.device)
    _fused_tail_kernel[grid](x, post_layer_mix, term2, out, C, BLOCK)
    return out
```

Kernel semantics (one program processes a `BLOCK`-sized batch of the flattened
`[a,b,n,c]` index space, 41943040 total elements):

1. Load `x[a,b,c]` (bf16) → cast to fp32.
2. Load `post_layer_mix[a,b,n,0]` (fp32).
3. Load `term2[a,b,n,c]` (fp32).
4. Compute `acc = x_f32 * pm + t2` in fp32.
5. Cast `acc` to bf16 and store to `out[a,b,n,c]`.

Broadcast semantics: `x` is `[2,4096,1280]` and is indexed by `(a,b,c)` only,
so the same `x[a,b,c]` is reused for every head `n` in `0..3` (broadcast along
dim -2), exactly matching `x.float().unsqueeze(-2) * post_layer_mix`.

### Index mapping

- `a = idx // (4096*4*C)`
- `b = (idx % (4096*4*C)) // (4*C)`
- `n = (idx % (4*C)) // C`
- `c = idx % C`
- `x_off = a*(4096*C) + b*C + c` (contiguous `[2,4096,1280]`)
- `pm_off = a*(4096*4*1) + b*(4*1) + n*1` (contiguous `[2,4096,4,1]`)
- `t2_off = idx` (contiguous `[2,4096,4,1280]`)

### Conformance notes

- `num_warps` / `num_stages` are left unset: the target profile marks them
  `Unknown` and the decision leaves them non-normative. This is a conformance
  note, not a design deviation.
- The kernel launches through the direct Triton launch syntax
  `kernel[(grid,)](...)`, which is the observed launcher convention for this
  profile (not the unproven `fast_libentry` path).
- All arithmetic is fp32; only the final store rounds to bf16, preserving the
  required fp32 intermediate precision.
- `tl.load`/`tl.store` with contiguous bf16/fp32 data and `tl.arange`
  (extent 1024) fall under the Supported primitives. No `tl.dot`, block
  pointers, or `num_warps`/`num_stages` hints are required.

## Gate Evidence

| Gate | Command | Exit | Result |
|---|---|---|---|
| Decision validation | `validate_decision.py decision_001.md --expected-profile triton_cuda` | 0 | `valid: true` |
| AST syntax | `python3 -m py_compile ...` | 0 | OK |
| Harness smoke (accuracy + timing) | `auto_bench.py --v0_file base.py --v1_file triton_mhc_post_layer_mix_001.py --warmup 50 --repeat 100 --full-traceback` | 0 | `PASS accuracy` |
| Runtime fingerprint | `import torch,triton` probe | 0 | matches (2.7.1, 3.1.0, BI-V150 (7,1)) |

Harness smoke details: `v0=8.023760 ms, v1=6.446620 ms, speedup=1.245x`.
(24.5% wall improvement vs baseline, exceeding the 5% adoption threshold.)

## Conformance

- `ModelNew` public contract preserved: same constructor (no args), same
  `forward(x, residual, post_layer_mix, comb_res_mix)` signature, same output
  `[2,4096,4,1280]` bf16 shape/dtype.
- `get_inputs()` / `get_init_inputs()` replicate the base contract.
- `term2` GEMM (`torch.einsum`) is left unchanged, as required.
- fp32 intermediate precision preserved (multiply-add in fp32, cast to bf16
  only at store).
- The harness AST loader (`_filter_module_ast`) retains `@triton.jit` function
  defs, `ModelNew`, `get_inputs`, `get_init_inputs`, and the literal
  `n0/n1/h/mhc_mult` assignments; the smoke run confirms the loaded module
  exposes all required entry points.

## Attempt Ledger

| # | Command | Exit | Defect | Candidate hash (before/after) |
|---|---|---|---|---|
| 1 | write candidate + `py_compile` + harness smoke | 0 | none | n/a → `08a9d59f...` |

## Handoff

- Candidate is ready for Verifier measurement. No semantic deviation observed.
- Coder does not return `accepted`; adoption is Verifier/Orchestrator's decision.

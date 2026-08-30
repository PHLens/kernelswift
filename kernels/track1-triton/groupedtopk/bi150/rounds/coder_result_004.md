# Coder Result 004

Result: candidate-ready

## Identity

- Round: `004`
- Decision: `rounds/decision_004.md`
- Decision SHA-256: `307f4a03c15b08daca8bb571f0391418997a07d864ff357b9f2d113cf2fb8f65`
- Canonical source: `baseline_adapter.py`
- Canonical source SHA-256: `689d458c7abe07323508fc054bfef609dc4bd1cd9c94e3bb706d6f2d2cd00016`
- Candidate: `triton_grouped_topk_004.py`
- Candidate SHA-256: `881a549cf95746dda93ee4c898e7ab0e67e3133a526088553091f8b8d7431d83`
- Language: `triton`
- Backend: `cuda`
- Target profile: `triton_cuda`
- Runtime fingerprint: `project.md#runtime-fingerprint`

## Implementation

- Stage one directly computes contiguous fp32 row softmax and eight group maxima
  for each fixed 256-expert row, writing `scores` and `group_scores`.
- The unchanged library `torch.topk(group_scores, 4)` produces the exact group
  selection and its original tie behavior.
- Stage two consumes those returned group IDs and writes `masked_scores`.
- The unchanged library `torch.topk(masked_scores, 8)` produces final IDs and
  weights; host-side renormalization and routed scaling match the baseline.
- Each forward creates distinct temporary tensors on `gating_output.device` and
  relies on the caller's current stream. No cache, global state, or launch hint
  is used. Non-target inputs follow the eager baseline-equivalent path.

## Profile Conformance

- Uses direct launch, `tl.program_id`, `tl.arange`, contiguous load/store,
  reshape, max/sum reductions, exp, where, broadcast-style vector operations,
  and static-range loop control within the established fixed-shape envelope.
- The stage-two contiguous four-element integer group-index load and
  `offsets // 32` group mapping compiled and executed on BI150 during the real
  harness and tie gates. They are recorded as a runtime conformance note rather
  than an untested assumption.
- Does not use `tl.dot`, block pointers, `fast_libentry`, `num_warps`,
  `num_stages`, mixed precision, or a custom selector.

## Attempt Ledger

| Attempt | Candidate SHA-256 | Commands and gate | Observation | Result |
|---|---|---|---|---|
| 1 | `881a549cf95746dda93ee4c898e7ab0e67e3133a526088553091f8b8d7431d83` | `python3 -m py_compile kernels/track1-triton/groupedtopk/bi150/triton_grouped_topk_004.py`; remote `auto_bench.py --v0_file kernels/track1-triton/groupedtopk/bi150/base.py --v1_file kernels/track1-triton/groupedtopk/bi150/triton_grouped_topk_004.py --warmup 2 --repeat 3 --full-traceback`; remote tie checks loaded through `auto_bench.load_ks_module` | Static parse passed. Real harness compiled and passed seeded accuracy (`v1=0.449993 ms`). All-equal, two-expert-tie, and structured group-tie cases passed exact integer IDs and floating tolerance. | candidate-ready |

## Gate Evidence

- Local AST gate: pass.
- Real harness AST loader and BI150 compile smoke: pass.
- Seeded correctness: pass.
- Exact integer IDs and floating weights: pass for all-equal, two-expert-tie,
  and structured group-tie cases.
- Remote smoke is a Coder gate only. No adoption claim or Verifier timing claim
  is made here.

## Handoff

Route `triton_grouped_topk_004.py` to Verifier for independent BI150
correctness, screening, paired timing, and targeted profiler evidence.

# Coder Context State

- role_contract_sha256: `8501d60fd684b5e625dc9b213046fd09f7e1512f2228ee0ff1425263dfb84196`
- context_epoch: `7`
- last_completed_round: `009`
- accepted_kernel: `triton_grouped_topk_008.py`
- accepted_report: `rounds/report_008.md`
- recent_three_round_evidence: `Round 004 two-stage candidate accepted at 7.455430192%. Rounds 005-007 rejected the exhausted direct-kernel path. Round 008 added constrained torch.compile dispatch and was accepted at 19.987917795%. Round 009 applied torch.compile mode=reduce-overhead (host-only) and passed the Coder gate as candidate-ready.`
- open_hypotheses: `Round 009 candidate triton_grouped_topk_009.py is candidate-ready; awaiting Verifier for correctness, paired timing (5% threshold), and targeted profiling.`
- artifact_read_hashes: `decision_009.md, project.md, team-state.md, base.py, triton_grouped_topk_008.py, triton_grouped_topk_009.py, coder_result_008.md, report_008.md, coder_result_002.md, coder_result_004.md, report_004.md read for Round 009.`

## Current Bottleneck

- `Retained library top-k gather and bitonic sort remain dominant (49.37 and 37.02 us/call in Round 008). The Round 009 intervention targets the substantial wall component outside attributed device time via reduce-overhead mode.`

## Recent Three-round Evidence

- Round 007, `rounds/coder_result_007.md`: direct-kernel path had no falsifiable intervention; aborted.
- Round 008, `rounds/report_008.md`: compiled two-stage candidate accepted; device time `111.120595703125 us/call`, kernel count `8.96/call`, wall improvement `19.987917795%`.
- Round 009, `rounds/coder_result_009.md`: host-only reduce-overhead mode change, candidate-ready; smoke `PASS accuracy` (v1 `0.287010 ms`, speedup `1.699x` vs base), all three tie cases exact.

## Open Hypotheses or Checks

- Preserve exact library torch.topk group/final ordering, per-forward buffer ownership, current device/stream, and fallback behavior.
- Any future candidate starts from `triton_grouped_topk_008.py` (accepted canonical after Round 008).
- Round 009 adoption requires Verifier unrounded paired median improvement of at least 5% vs `triton_grouped_topk_008.py`.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `baseline_adapter.py` | `689d458c7abe07323508fc054bfef609dc4bd1cd9c94e3bb706d6f2d2cd00016` | 009 |
| `rounds/decision_008.md` | `bec59b81693001fd27302a610ab48123e38a4a81c44b65cedfff9530b059e5d1` | 009 |
| `triton_grouped_topk_008.py` | `d1fb6b03d3be92cdd6423f1f44f33ea81d13f0e4df18227fe2d5f7dceb582535` | 009 |
| `rounds/report_008.md` | `f1fa38cef46804c96be6c4eb3f5eddeaa7ec509830dae6f2bea58f8be0e2b3b7` | 009 |
| `rounds/decision_009.md` | `066045e737fa1aedcc283c4058d2eceb28b8630013c7b93342abdb516af908b8` | 009 |
| `triton_grouped_topk_009.py` | `9b58f861ef6c3de86577dfe819327895311298cc4edf4b3f514f7fe9f4bff194` | 009 |
| `rounds/coder_result_009.md` | `ad9705c4cb2c7120f593cf0a6e5bd4b4b71438d4066360812755f3dd17e4dd8b` | 009 |
| `base.py` | `d57ace7d9196e2e44bdcfd17d1738482e7fd1bbb2d86fc6c9449c43938953eb5` | 009 |

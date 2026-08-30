# Decision 001

## Metadata

```json
{"schema_version":2,"decision":"proceed","decision_kind":"optimization","round":"001","reference_implementation":"baseline_adapter.py","reference_report":"rounds/report_000.md","language":"triton","backend":"ascend","target_profile":"triton_ascend","runtime_fingerprint_ref":"project.md#runtime-fingerprint","change_scope":"kernel","change_family":"kernel-fusion","sketch_ref":"rounds/sketch_001.json","sketch_sha256":"76818c21a7502a68b6ec5c6230607fa24bddf3e342e61d4d333990d16d639738","implementation_profile_snapshot_ref":"state/implementation_profile_snapshot/profile.yaml","implementation_profile_snapshot_sha256":"a2c3e2e4622fd2d9d2ffd67206912699217279238a14d66a0816cdc188d96321","project_capability_claim_ref":"state/project_capability_claim.json","project_capability_claim_sha256":"a46ce09de93c671865f2c1b661a335a8b7f5714db475a27bae763f9d84a56b9d"}
```

## Optimization Intent

```json
{"bottleneck_class":"host-bound","intervention":"replace the native SDPA call and its materialized q/k/v transpose chain with one fused Triton flash-attention kernel that indexes q/k/v and writes the output directly in the native [B,S,NH*HEAD_DIM] layout","allowed_changes":["kernel dataflow","kernel launch structure"],"invariants":["ModelNew public contract","output shape dtype and device","numerical tolerance atol=1e-2 rtol=1e-2","base.py bytes unchanged","public constructor and forward signature"],"expected_wall_improvement_pct":30.0}
```

## Unified Sketch

```json
{"artifact":"rounds/sketch_001.json","sha256":"76818c21a7502a68b6ec5c6230607fa24bddf3e342e61d4d333990d16d639738","rendering":"# D Declarations\ntensor query shape=[B,S,NH*HEAD_DIM] dtype=fp16 layout=contiguous memory=global\ntensor key shape=[B,S,NH*HEAD_DIM] dtype=fp16 layout=contiguous memory=global\ntensor value shape=[B,S,NH*HEAD_DIM] dtype=fp16 layout=contiguous memory=global\ntensor out shape=[B,S,NH*HEAD_DIM] dtype=fp16 layout=contiguous memory=global\ntile q_tile shape=[BLOCK_M,HEAD_DIM] dtype=fp16 layout=blocked memory=register\ntile k_tile shape=[BLOCK_N,HEAD_DIM] dtype=fp16 layout=blocked memory=register\ntile v_tile shape=[BLOCK_N,HEAD_DIM] dtype=fp16 layout=blocked memory=register\nscalar scale shape=[1] dtype=fp32 layout=scalar memory=register\n\n# O Operations\nload q_tile <- query[b, 0:BLOCK_M, h*HEAD_DIM:h*HEAD_DIM+HEAD_DIM] mask row_idx < S\nload k_tile <- key[b, 0:BLOCK_N, h*HEAD_DIM:h*HEAD_DIM+HEAD_DIM] mask row_idx < S\nload v_tile <- value[b, 0:BLOCK_N, h*HEAD_DIM:h*HEAD_DIM+HEAD_DIM] mask row_idx < S\ncompute qk = dot(q_tile, trans(k_tile)) * scale  # fp32 accumulate, conversion declared\ncompute p = masked_softmax(qk)\ncompute acc = dot(p.to(fp16), v_tile)\ncompute acc_norm = acc / rowsum(p)\nstore out[b, 0:BLOCK_M, h*HEAD_DIM:h*HEAD_DIM+HEAD_DIM] <- acc_norm mask row_idx < S\n\n# C Control\nparallel bh over B*NH\nguard row_idx < S\n\n# H Target Hints\ntarget=triton_ascend\nBLOCK_M=128\nBLOCK_N=128\nHEAD_DIM=64\naccumulator_dtype=fp32\nnum_warps=4\nnum_stages=1\n"}
```

## Host Plan

```json
{"applicability":"not-applicable","reason":"kernel-only change: no host state, allocation reuse, output cache, or stream behavior is introduced; host gain is expected only indirectly from collapsing 6.96-6.98 launches per call to one"}
```

## Evaluation Contract

```json
{"hypothesis_id":"H-001","intervention":"replace the native SDPA call and its materialized q/k/v transpose chain with one fused Triton flash-attention kernel that indexes q/k/v and writes the output directly in the native [B,S,NH*HEAD_DIM] layout","expected_causal_chain":["one fused kernel replaces 6.96-6.98 launches per call","host launch and synchronization overhead decreases","the three materialized transposes and the inplace-copy transpose disappear","device us per call decreases","wall time decreases"],"primary_metric":{"name":"wall_time","expected_improvement_pct":5.0},"mechanism_observables":[{"name":"kernel_count_per_call","expectation":"decrease"},{"name":"device_us_per_call","expectation":"decrease"},{"name":"transpose_kernel_count_per_call","expectation":"decrease"},{"name":"device_ratio","expectation":"decrease"}],"guardrails":["correctness:pass","output shape dtype and device unchanged","numerical tolerance atol=1e-2 rtol=1e-2","public constructor and forward signature unchanged"],"profiling_level":"targeted","causal_graph":{"nodes":["n_launch","n_device_layout","n_device_time","n_wall"],"edges":[["n_launch","n_wall"],["n_device_layout","n_device_time"],["n_device_time","n_wall"]]}}
```

## Pitfalls and Anti-pattern Consultation

- Consulted `references/anti-patterns.md`; no matching failure invalidates this path.
- Consulted `references/bottleneck-judgment.md`: the accepted report attributes only about one third of wall time to device work, so the declared bottleneck class is `host-bound` even though the intervention also removes device layout work. The device-side gain is a mechanism, not the adoption claim.
- KernelWiki `pattern-device-win-wall-loss` is the governing counterexample for this operator family: the retained Ascend flexattention Round 003 improved device time by 55.8 percent while wall time regressed 8.3 percent. This decision is explicitly designed to avoid that trap by making launch collapse the primary mechanism rather than a side effect.
- Unlike S60, this target places no power-of-two constraint on fp16 `tl.dot`, and the frozen profile records non-multiple-of-16 tiles including `(83,64,64)` as numerically correct. `S=83` is therefore handled with `BLOCK=128` plus a row mask rather than by padding the contraction dimension, which would waste arithmetic.
- `tl.arange` extents are recorded only for 64/128/256, so `BLOCK_M=128` and `BLOCK_N=128` stay inside the proven extent set; no other extent is declared.
- `num_warps=4` and `num_stages=1` are `preferred` hints inside the profile-legal sets (1/2/4/8 and 1/2/3/4). They are not normative requirements and may be revisited by a later configuration-only round.

## Rationale and Evidence

The accepted `rounds/report_000.md` establishes four facts that make this the
highest-value falsifiable intervention:

1. Device time is `104.1264 us/call` (candidate scope) against a benchmark median
   of `0.347800 ms`, so roughly two thirds of wall time is host-side launch and
   synchronization, not device compute.
2. The call issues `6.96-6.98` kernels per call. Collapsing them to one attacks
   the dominant term directly.
3. Within device time, layout conversion dominates the actual attention math:
   three `aclnnFlashAttentionScore_TransposeAiCore_Transpose` kernels cost
   `48.0884 us/call` and one `aclnnInplaceCopy_TransposeAiCore_Transpose` costs
   `15.2628 us/call`, together `63.35 us/call` (54.5% of device), while
   `FlashAttentionScore` itself costs only `23.0232 us/call`. The transposes exist
   only because the native backend materializes the `view`/`transpose` around
   SDPA as contiguous copies.
4. The frozen profile makes the path legal: fp16 `tl.dot` with fp32 accumulation
   is `constrained` and numerically correct on every probed tile, including
   `(128,128,64)` which covers `S=83` under a row mask.

The kernel reads and writes the native `[B,S,NH*HEAD_DIM]` layout with a computed
stride, so no transpose is materialized, and one launch replaces the SDPA call
and its wrapping copies. Because `S=83` and `NH*HEAD_DIM=512` fit inside
`BLOCK_M=128` and `BLOCK_N=128`, a single program per `(batch, head)` covers the
whole sequence with no inner loop over KV blocks, keeping the kernel simple and
the launch count at exactly one.

The expected 30 percent wall improvement is a judgment, not a measurement. The
adoption test remains the harness median wall time against the accepted
`baseline_adapter.py`, and the round is `no-improvement` unless wall time
improves by at least 5 percent even if every mechanism observable moves in the
predicted direction.

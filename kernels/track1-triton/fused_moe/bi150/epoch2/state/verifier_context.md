# Verifier Context

- role_contract_sha256: `62f10a0940ca3665260226a7891f5d34e1b571e70937862bb02ad68aa2bbc82f`
- context_epoch: `1`
- last_completed_round: `002`
- accepted_kernel: `triton_fused_moe_e2_001.py` @ `da623fa9…` — round 001 remains canonical; round 002 was `no-improvement`
- accepted_report: `rounds/report_001.md`
- recent_three_round_evidence: `Round 002 / no-improvement / -0.175% vs round-001 canon 0.219792 ms (paired A/B +0.093 us, mixed signs = cost-neutral, NOT regression). Correctness-hardening by design: retention guarantee made explicit and independently tested (byte-identical at 50/150/300 further calls with changing data; negative control detects corruption at call 8). FR-1 and FR-5 fire; FR-2/FR-3/FR-4 pass. G1 closed as a line of work. Round 001 / accepted / +93.248% (0.219792 ms vs canon 3.255288 ms); host lever confirmed far beyond model (423 us replay-vs-eager vs 170 us modeled), device lever did NOT land (58.231 vs 55.954 us, device-neutral). Round 000 / baseline / canon 3.255288 ms, 123.95 launches/call, device 967.852 us/call, 29.7% device / 70.3% host.`
- open_hypotheses: `G1 is CLOSED by direct measurement: no-per-call-allocation and never-alias-across-calls are mutually exclusive below ~150 forwards. Remaining addressable item is G2 (routing prelude ~34-42 us/call, topk ~39.9 us alone and frozen by tie semantics, so realistic reclaim ~20 us against a 10.99 us gate). Harness cudaDeviceSynchronize ~122 us/call is non-addressable and sets a practical floor near 214 us.`
- artifact_read_hashes: `base 21e75853… (3598 B), harness 71fb3ad0… (29428 B), r001 canonical da623fa9…, r002 candidate 781d341c…, decision_002 dc782254…, sketch_002 015da345…, binding_002 8be91cca…, profile dc8fa4c0… — all re-verified at round 002.`

## Current Bottleneck

- `Canonical wall is 0.219792 ms (round 001). Round 002 changed nothing measurable. Of the ~220 us: harness cudaDeviceSynchronize ~122 us/call is NOT addressable; routing prelude ~34-42 us/call (topk ~39.9 us alone, frozen by tie semantics) is the largest addressable item at ~20 us realistic reclaim; the copy-out residual ~22.5 us/call is the deliberate price of non-aliasing and is NOT to be optimized away. Practical floor ~214 us.`

## Recent Three-round Evidence

- `Round 002 | no-improvement | rounds/report_002.md | G1 / C3 boundary. v0=3.217895 ms, v1=0.220177 ms; +93.158% vs paired v0, +93.236% vs r000 canon, -0.175% vs r001 canon (+0.385 us vs 10.99 us gate). Paired A/B vs r001: 0.209003 vs 0.209106 ms, delta median +0.093 us, MIXED SIGNS, bitwise-equal => cost-neutral, not regression. Retention PASS at 50/150/300 (drift 0.0); out_dest 1 ptr, still all-zero (never written under C3), never returned; negative control detects corruption at call 8. FR-1 fires (empty_like/empty_strided still 1.00/call; alloc CPU 18.45 -> 14.49 us). FR-2 pass. FR-3 pass at 2.0. FR-4 pass at +0.837 us with pre-binds asserted; kernel bodies byte-identical (61d16bde3d12fb12). 13/13 correctness suites pass, all bitwise-equal to r001.`
- `Round 001 | accepted | rounds/report_001.md | manual-graph-replay-fused. v0=3.193262 ms, v1=0.219792 ms, +93.248% vs canon, 14.81x. Host lever confirmed (0 launcher executions/call, 2.0 submissions/call, 423 us replay-vs-eager vs 170 us modeled). Device lever did not land (58.231 vs 55.954 us, device-neutral). FR-2 and FR-4 fired as recorded; FR-5 did not.`
- `Round 000 | baseline | rounds/report_000.md | Phase 0 canonization. v0=3.255288 ms. 123.95 kernels/call, device 967.852 us/call, device_ratio 0.297317 (29.7% device / 70.3% host). Dispatch/indexing 635.313 us/call (65.6% of device), GEMMs 118.831 us/call (12.27%). Launch count data-dependent: ~14 launches per active expert (148 ops/call at 8 active, 134 at 7, 64 at 2).`

## Open Hypotheses or Checks

- `G1 CLOSED — do not revisit. "No per-call allocation" and "the returned tensor never aliases across calls" are mutually exclusive below ~150 forwards: forward() must produce a fresh tensor, so torch.empty_like survives any re-targeting of the copy destination. The V3 rotating pool is the only allocation-free shape and it fails retention exactly at the pool size.`
- `Host-side allocator cost at this size does NOT convert to wall: alloc CPU improved 18.45 -> 14.49 us/call (~-4 us) with ZERO wall movement. Treat sub-5-us host allocator savings as unmonetizable on this rig.`
- `METHODOLOGY — count submissions, not raw profiler records. The copy-out is recorded TWICE: once as host API cudaMemcpyAsync and once as device activity "Memcpy DtoD (Device -> Device)". Summing raw memcpy names yields a spurious 3.00 and would falsely fire FR-3. True count is 1 cudaGraphLaunch + 1 copy = 2.0, corroborated by aten::copy_ at 1.00/call. (Same double-recording existed in round 001's raw census.)`
- `METHODOLOGY — always pre-bind workspaces before an eager device control. Disabling the tier guards also bypasses _alloc_workspace, so _pipeline re-allocates sort buffers every call (~49 us of aten::fill_ churn). Pre-bind via one real tier-1 serve and ASSERT all five buffers non-None before disabling the guards.`
- `METHODOLOGY — fp16-extreme operand tier must be capped at 32, not 1024. At 1024, silu(gate)*up reaches ~1.5e5 and overflows fp16 in base.py itself, making allclose vacuous (NaN vs NaN). Always assert a finite BASE before reading a candidate FAIL.`
- `METHODOLOGY — retention tests need CHANGING data. With constant data a rotating-pool wrap-around writes identical bytes and torch.equal still returns True (false pass). Verified by negative control: broken pool-8 model is detected at exactly call 8 with changing data.`
- `ACTIVATION LADDER is 8/7/2/2/2, not 8/7/2/2/1. torch.topk with k=2 always returns two distinct indices, so a single active expert is structurally unreachable. coder_result_002.md records "1 active" where the true count is 2.`
- `best_num_warps = 1 (round 001, 24.4% margin; FR-4 fired, sibling nw2 prior does not transfer). best_BLOCK_M = 16 (round 002 p16, 9.829 us margin). num_stages NOT recorded (0.031 us margin, inside the 0.5 us tie band).`
- `Kernel-mode profiling remains unavailable (run_out arity: harness passes 2 args, signature takes 3). Forward-mode dual-scope is canonical; no accommodation added.`
- `fp16-dot exactness is POSITIVE at these contractions (2.441e-04 vs 1e-2 tolerance, ~40x margin) per round-001 p01, unlike the mm_encoder negative.`

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `../../base.py` | `21e758535af58c21684d6e657ae44f91a0248d246eb424b6923a3c180bc32a3d` (3598 B) | `002` |
| `/root/CodeBuddy/20260818191200/kernelswift/auto_bench.py` | `71fb3ad0c3ad23c5c156c898f85abcee3d42a15800f75ff97769cfca9152fe29` (29428 B) | `002` |
| `baseline_adapter.py` | `752a25033b7629459c6eb128c60a4bdc3ab77b9c7cc97f5d3592bdff4cd45a47` | `002` |
| `triton_fused_moe_e2_001.py` (CANONICAL) | `da623fa92819185a1e20a8a7cbaca40acd9bfb4a3147f8e1e7b1e757c6b24cb7` | `002` |
| `triton_fused_moe_e2_002.py` (round-002 candidate) | `781d341cae2236917da988988fbe2754fc808ea0f016d7dff82fd142822d1b2d` | `002` |
| `rounds/decision_002.md` (file bytes) | `dc782254a54331454290fac6791b7f583fff81d8de9699f03f5d06722fd7637e` | `002` |
| `rounds/sketch_002.json` (file bytes) | `015da3456f18582ad6114d3f5a0bfd14c5122a365bfbdd8031b1e543ecfe7ebe` | `002` |
| `rounds/binding_002.json` (file bytes) | `8be91ccae9c3887c480451698d6bd02f1d1eb2b5c8c0d8ea08c55570f6b4e876` | `002` |
| `profile_snapshot/triton_cuda.yaml` (file bytes) | `dc8fa4c0c73caecc78c6a886e5ffb18cb97371f3add7cb382dbe30887743b7ae` | `002` |
| `../triton_fused_moe_002.py` (epoch-1 archive, read-only) | `6ac1f44b111285f5bf746110c51f6486868b12beb2deae3390663d74233f8ae5` | `001` |

### Verdict-chain hashes (validator convention: sha256 of PARSED canonical JSON)

| Round | decision | sketch | binding | profile | report_fact_pack |
|---|---|---|---|---|---|
| 002 | `34ae15121297881d64eadb4900ef87cf2d9cb8f29da09e90d9b8901b7ff4724f` | `780716027a8a02f3045833f56351ca13c20f5be73f6f34d3f91517418046b2c2` | `86c9ce54e9f210c08fb46d41d03cc29979e47131cc0dac11a17d75c55b61ba8b` | `55bc084448707b2be28ecf5d4e17ce914ec19ea3dba18fcb42324527cdd3ddae` | `4aa405942f91a7e73dcaf1267c76743ad3b8436bb8e7197c226a3652627aeca3` |
| 001 | `6da2b5536dabef6d051af24446ae87d17274960325d99da480cc62d5dac56259` | `0c9f745002349ca31a3fe32a585877413bc0c3ce9da8aef68350b0d37e52f613` | `0000…0000` (**absent at the time; later supplied**) | `55bc0844…` | `6bf4d38582a53eb4daa31203a9d12f84783ee155ea5c5760cd798c4f2ec6d599` |

Round 002's binding ledger was present and validated; round 001's was not (Coder-owned artifact, absent at verification time). `rounds/binding_001.json` now exists in the tree.

### Artifact file hashes (sha256 of bytes)

| Artifact | SHA-256 |
|---|---|
| `rounds/report_002.md` | `d67da1bbe1c728add6f2192e91380ed88e39c95e85a2bf03b5b32266824da17c` |
| `rounds/verdict_002.json` | `see below` |
| `rounds/round_status_002.md` | `see below` |
| `rounds/report_001.md` | `532fe3ea8f461c608bf15efd96c8c5d527ac4a0098d0eb4009b26d21c1fbb8a5` |
| `rounds/verdict_001.json` | `ac495b12fd56f96a8ef36daba078186bb5cc7d44341f52db1f8cb503915bb56a` |

### Round 002 raw evidence index

| Artifact | Path |
|---|---|
| three ordered wall pairs | `log/round_002_wall_pairs.txt` |
| paired A/B vs r001 (script) | `log/round_002_paired_ab.py` |
| paired A/B vs r001 (result) | `log/round_002_paired_ab.json` |
| retention test (script) | `log/round_002_retention.py` |
| retention test (result) | `log/round_002_retention.json` |
| correctness suite (script) | `log/round_002_correctness_suite.py` |
| correctness suite (result) | `log/round_002_correctness_suite.json` |
| census (script) | `log/round_002_census.py` |
| census (result) | `log/diagnostic_scope_census_002.json` |
| profiler run log | `log/round_002_profile.txt` |
| dual-scope chrome trace | `log/round_002_forward_100iter.pt.trace.json` |
| lifecycle | `rounds/round_status_002.md` |
| report | `rounds/report_002.md` |
| verdict | `rounds/verdict_002.json` |

### Earlier-round raw evidence

- Round 001: `log/round_001_wall_pairs.txt`, `log/round_001_correctness_suite.json`, `log/diagnostic_scope_census_001.json`, `log/round_001_forward_100iter.pt.trace.json`, `rounds/round_status_001.md`, `rounds/report_001.md`, `rounds/verdict_001.json`
- Round 000: `log/round_000_correctness.txt`, `log/round_000_wall_pairs.txt`, `log/round_000_summary_{reference,candidate}.json`, `log/round_000_forward_100iter.pt.trace.json`, `rounds/round_status_000.md`, `rounds/report_000.md`

### Positive-control fingerprint ledger

| Campaign | base SHA | bytes | measurement fingerprint | verdict |
|---|---|---:|---|---|
| fused_moe @ bi150 epoch2 (this run) | `21e75853…` | 3598 | `fe73bc58146d8c16f524be2a00fe99b31e1b9678bca6b3702f4284a3ac0a5bef` | reproduced (round 000) |
| flexattention @ bi150 epoch2 | `dd1359ad…` | 2479 | `6dc07009177b649f7c2cad8f7be5e9aad74235bd9f50abfebc88bdb273e32af4` | reproduced (round 000) |
| mm_encoder_attention @ bi150 epoch2 | `86ac5703…` | 2284 | `0c4c7d664c85e65d0580091ca5e3a77ff769a0d28f7e679f5bdf78fe5d0d966e` | reproduced (round 000) |

## G2 Pre-Measurement (post-round-002 census, verifier2 cold-rehydrate)

- `pre-G2 census: rounds/pre_g2_measurement.md + log/pre_g2_prelude_timing.{py,json}. Four eager prelude numbers CONFIRMED (topk ~41.5, sum ~14.7, div ~6.8, softmax ~5.2 µs/call; carried ~39.9/13.6/6.7/5.0 within noise). Clean graph-assisted isolated device numbers: softmax 1.925, topk 22.268, sum 3.750, div 2.459, cast 1.597, full prelude 32.865 µs/call (sum-of-parts reconciles, bitwise-equal to aten).`
- `G2 fold net: ~10 µs device / ~0 µs wall. Prelude launches INSIDE the captured graph, so folding removes no submission (submission stays 2.0) and saves ~0 host; only non-topk aten device math (~9.8 µs: softmax+sum+div+cast) is foldable, topk (~22-41 µs) frozen by tie semantics.`
- `GATE FLAG: softmax-fold TRIPS the NOT-granted reduction.sum waiver (capability_claim fallback_contract=reduction.sum, fallback_signature={axis:k, dtype:fp32}; decision_001 line 414 "waiver not granted"). Renorm-sum fold trips the SAME waiver (fp32 reduce over k). Only the fp16 cast (~1.6 µs) is waiver-clean. => G2 dead in softmax-fold form; no waiver-clean G2 fold of meaningful size exists.`

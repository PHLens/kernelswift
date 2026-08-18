# Round Status 002

- phase: `verification-complete`
- result: `no-improvement`
- decision: `rounds/decision_002.md`
- candidate: `triton_grouped_topk_002.py`
- accepted_reference: `triton_grouped_topk_001.py`
- harness_reference_adapter: `reference_triton_grouped_topk_001.py`
- measurement_fingerprint: `3fe7d50260b3670756d26f003427faccb76e1e79c204b0a500a62d2eb481c809`
- timing_order: `sequential complete accepted-reference-adapter block, then complete candidate block`
- required_pre_timing_targeted_gate: `MACA dtype-view, backing identity/spans, retained lifetime, cross-call/instance isolation, mutation isolation, input non-mutation, reference correctness`

## Artifact Hashes

| Artifact | Local SHA256 | Remote SHA256 | Verdict |
|---|---|---|---|
| `triton_grouped_topk_001.py` | `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384` | same | match |
| `reference_triton_grouped_topk_001.py` | `70258939973f728858383b832e37069ec6b3d4681200cdb4e70daac42229b2f9` | same | match |
| `triton_grouped_topk_002.py` | `1cbfddc1fd91ef4d73e388758467962cb471fc2a5f508c0af0749dcce53080d1` | same | match |
| `auto_bench.py` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` | same | match |
| `rounds/decision_002.md` | `96b175002ab35ebbdeab2e647e1f0acfb150d08ca30792db1c6657a3afea7c55` | not-applicable | match Coder result |
| `rounds/coder_result_002.md` | `93eafe2b03f0a83fe65cb86b8453c787be60c753b3e4a5d42d834d6192ecfac7` | not-applicable | read |
| `rounds/report_001.md` | `f2866692d8d1c4519e9a2028c7b1d707fcb4f9f945fd856f78139f2dbe2aec4a` | not-applicable | accepted evidence |
| `log/source_equivalence_002.log` | `30ffbc3ed4a5ec7a616001f1276df503531b97c183e2c28d4a7338cb55a0b71c` | not-applicable | retained evidence |
| `log/correctness_002.log` | `700fae3672824cd6a7d3b79ef96e8482ba0910ebca5d138b0fcc009124ba69a7` | not-applicable | retained evidence |
| `log/storage_lifetime_gate_002_attempt1.log` | `733dc650c80d24f27f41ea0349c75b8fa5b34a31ba2b182a9adeec8cc8b24424` | not-applicable | retained failed-probe evidence |
| `log/storage_lifetime_gate_002.py` | `dd91be58a4450412bee285e11ac323a3fa7103e4d8d8bd1e412691d95a73cd75` | same | matched corrected probe |
| `log/storage_lifetime_gate_002.log` | `4c4f107da800fd485e7c21ae3b1b34b89837db7bb958670220862c6d9b06b668` | not-applicable | retained corrected-gate evidence |
| `log/wall_002_sample_1.log` | `84e02d26b2c9a624095a546058db6ff3d91d98091274c47f255edcc6d22e4887` | not-applicable | retained evidence |
| `log/wall_002_sample_2.log` | `7b743d76193189745fbb39294bcb29340a4532aa19afc4c5457c5913f7398ed0` | not-applicable | retained evidence |
| `log/wall_002_sample_3.log` | `1cf47a390b7c68147f60be061dc390ba16e4a77b561486d8a033dea26dc84c89` | not-applicable | retained evidence |
| `rounds/report_002.md` | `a5ad9cfe8ead4e1e3cf06ef990ea0817537af4c088219f1eed9a551055426365` | not-applicable | final report |
| `state/verifier_context.md` | `c270638dc54852a64e6d931ac625940d4e019422a5cf4d924728a79f4f1f6c75` | not-applicable | updated durable context |

## Completed Commands

1. Read latest Round 002 durable inputs/contracts: exit `0`.
2. Local SHA256 verification: exit `0`.
3. Remote executable-file SHA256 verification: exit `0`.
4. Reference adapter unified diff: exit `1` as expected; exactly `class ModelNew` -> `class Model`, no other byte diff.
5. Candidate-vs-canonical kernel/launch/guard/fallback/constructor/entrypoint exact-source comparison: exit `0`; every frozen region equal.
6. Standard correctness (`warmup=5`, `repeat=10`, `--full-traceback`): exit `0`; `PASS accuracy`; `1 passed, 0 failed`; evidence `log/correctness_002.log`.
7. Storage/lifetime gate attempt 1: exit `1`; 18 substantive candidate checks passed, but the aggregate fast-contract flag was recomputed outside `torch.no_grad` and false; classified measurement-probe defect; formal wall not started; evidence `log/storage_lifetime_gate_002_attempt1.log`.
8. Corrected storage/lifetime gate script remote SHA256 check: exit `0`; remote `dd91be58a4450412bee285e11ac323a3fa7103e4d8d8bd1e412691d95a73cd75`, exactly matching local.
9. Corrected storage/lifetime gate: exit `0`; elapsed `15.2981375 s`; all 20 checks true and `overall=PASS`; evidence `log/storage_lifetime_gate_002.log`.
10. Formal wall sample 1 (`warmup=200`, `repeat=500`): exit `0`; elapsed `15.2866939 s`; `reference=0.072343 ms`, `candidate=0.082707 ms`; evidence `log/wall_002_sample_1.log`.
11. Formal wall sample 2 (`warmup=200`, `repeat=500`): exit `0`; elapsed `14.9358120 s`; `reference=0.067703 ms`, `candidate=0.076745 ms`; evidence `log/wall_002_sample_2.log`.
12. Formal wall sample 3 (`warmup=200`, `repeat=500`): exit `0`; elapsed `15.1908524 s`; `reference=0.071684 ms`, `candidate=0.081513 ms`; evidence `log/wall_002_sample_3.log`.
13. Independent median/adoption calculation: exit `0`; reference median `0.071684 ms`, candidate median `0.081513 ms`, unrounded improvement `-13.711567434852972%`, median speedup `0.8794180069436777x`; failed required `+5%` threshold.
14. Targeted forward profiler: not run because the strict Round 002 sequence permits it only after the formal wall gate passes; no Round 002 trace was created.

## Raw Samples

- standard correctness: `pass`; smoke-only `reference=0.074996 ms`, `candidate=0.079958 ms`; evidence `log/correctness_002.log`
- targeted storage/lifetime gate: `PASS` on the approved probe-only corrected rerun; dtype-view no-copy, exact disjoint spans `[0,2656)`/`[2656,5312)`, same-call backing identity, cross-call/cross-instance distinct backing, retained-output stability, bidirectional mutation isolation, input immutability, and reference correctness all passed; attempt 1 retained separately
- reference wall samples ms: `[0.072343, 0.067703, 0.071684]`; median `0.071684`
- candidate wall samples ms: `[0.082707, 0.076745, 0.081513]`; median `0.081513`
- unrounded improvement: `-13.711567434852972%`
- median speedup: `0.8794180069436777x`
- profile: `not-run: formal wall failed the mandatory +5% precondition`
- hypothesis verdict: `partially-confirmed: allocation/storage mechanism and guardrails passed; primary wall-time claim falsified`

## Next Safe Action

Orchestrator validates `rounds/report_002.md`, this status, and `state/verifier_context.md`; records `no-improvement`, retains `triton_grouped_topk_001.py` / `rounds/report_001.md` as canonical, updates counters, and releases measurement exclusivity. No source or team-state change is Verifier-owned.

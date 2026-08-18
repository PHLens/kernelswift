# Coder Context State

- role_contract_sha256: `8501d60fd684b5e625dc9b213046fd09f7e1512f2228ee0ff1425263dfb84196`
- context_epoch: `2`
- last_completed_round: `002`
- accepted_kernel: `triton_grouped_topk_001.py`
- accepted_report: `rounds/report_001.md`
- recent_three_round_evidence: `Round 001 accepted the fused Triton-MACA kernel; Round 002 is candidate-ready after the exact host-only allocation change passed the true loader, dtype-view, launch, and fixed-seed smoke gate.`
- open_hypotheses: `Verifier must establish one allocation per call, same-call disjoint shared storage, cross-call/model backing independence, mutation isolation, authoritative correctness, targeted profiling, and paired wall timing.`
- artifact_read_hashes: `Round 002 decision, accepted report/canonical/reference adapter, candidate, project, team-state, and profile hashes are recorded below.`

## Current Bottleneck

- Round 001 authoritatively reports the accepted candidate at 68.280 us/call,
  one 10.7442822265625 us device kernel per call, and a 41.58952 us/call
  inclusive CPU scope. Two inclusive `aten::empty` events cost
  10.03988 us/call.

## Recent Three-round Evidence

- Round 001, accepted, `rounds/report_001.md`, kernel-fusion; correctness,
  targeted tie parity, 1.0 kernel/call, and 69.59021613749428% authoritative
  wall improvement passed.
- Round 002, candidate-ready, `rounds/coder_result_002.md`,
  fresh-allocation-coalescing; real dtype-view/compile/correctness smoke passed
  without changing the accepted kernel.

## Open Hypotheses or Checks

- Verify exactly one `aten::empty` per fixed-path call and no added device
  kernel or launch.
- Verify same-call output backing identity with disjoint byte intervals,
  retained-output lifetime, mutation isolation, and distinct backing across
  calls and model instances.
- Run authoritative correctness, paired wall timing against the accepted
  class-rename reference, and targeted CPU/device profiling. Exclude 1/1 smoke
  timing from adoption.

## Artifact Read Hashes

| Artifact | SHA-256 | Last read round |
|---|---|---:|
| `skills/kernel-opt-loop/prompts/coder.md` | `8501d60fd684b5e625dc9b213046fd09f7e1512f2228ee0ff1425263dfb84196` | 001 |
| `skills/kernel-opt-loop/prompts/coder_targets/triton_maca.md` | `2cfa08c2664f01e70bb43eec7bb998be836a6a719b17535268a8d6ca18c85540` | 001 |
| `project.md` | `503ca8ca604c4cacb2cc9c52c9efedefdf36bad4d59df190da0e453da42ae310` | 002 |
| `team-state.md` | `5c63ec018525dfe50dc00ee1acc7e4600bb347570a338e6aeccb949c9e1ad306` | 002 |
| `rounds/report_001.md` | `f2866692d8d1c4519e9a2028c7b1d707fcb4f9f945fd856f78139f2dbe2aec4a` | 002 |
| `rounds/decision_002.md` | `96b175002ab35ebbdeab2e647e1f0acfb150d08ca30792db1c6657a3afea7c55` | 002 |
| `triton_grouped_topk_001.py` | `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384` | 002 |
| `reference_triton_grouped_topk_001.py` | `70258939973f728858383b832e37069ec6b3d4681200cdb4e70daac42229b2f9` | 002 |
| `triton_grouped_topk_002.py` | `1cbfddc1fd91ef4d73e388758467962cb471fc2a5f508c0af0749dcce53080d1` | 002 |

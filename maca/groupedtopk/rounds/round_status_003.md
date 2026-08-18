# Round Status 003

- phase: `verification-complete`
- result: `no-improvement`
- decision: `rounds/decision_003.md`
- candidate: `triton_grouped_topk_003.py`
- accepted_reference: `triton_grouped_topk_001.py`
- harness_reference_adapter: `reference_triton_grouped_topk_001.py`
- measurement_fingerprint: `3fe7d50260b3670756d26f003427faccb76e1e79c204b0a500a62d2eb481c809`
- timing_order: `sequential complete accepted-reference-adapter block, then complete candidate block`
- required_tie_gate: `fixed fast path group top-4 cutoff tie plus eligible expert top-8 cutoff tie, weights tolerance and exact IDs`

## Artifact Hashes

| Artifact | Local SHA256 | Remote SHA256 | Verdict |
|---|---|---|---|
| `triton_grouped_topk_001.py` | `9ba99fcfaa3515e9252f18373d3dfb6980b5ba80a21169923b0c2e5b56bef384` | same | match |
| `reference_triton_grouped_topk_001.py` | `70258939973f728858383b832e37069ec6b3d4681200cdb4e70daac42229b2f9` | same | match |
| `triton_grouped_topk_003.py` | `9409bd85da798b083e785774525a076ec781b6df13cd1129843fe7e9c9ead9f6` | same | match |
| `auto_bench.py` | `3d4fa4eefedb9fd89439c54572a1541560a9d0301ddbf48a8e1f714d00465bf2` | same | match |
| `base.py` | `49ec0cf7a48679c23c5187eb7ba546fd41025019a9d6e9b59e1273c48a31dfbb` | same | match |
| `rounds/decision_003.md` | `cfcee8a61b91536da0aa302504b8bc4119c9c2deac5150878b6371870791f6b7` | not-applicable | match Coder result |
| `rounds/coder_result_003.md` | `82372f63ad9632fa7d430f765d5f26d73afcc1d4a6688ead2cee33fec875310e` | not-applicable | read |
| `rounds/report_001.md` | `f2866692d8d1c4519e9a2028c7b1d707fcb4f9f945fd856f78139f2dbe2aec4a` | not-applicable | accepted evidence |
| `rounds/report_002.md` | `a5ad9cfe8ead4e1e3cf06ef990ea0817537af4c088219f1eed9a551055426365` | not-applicable | prior no-improvement evidence |
| `log/source_equivalence_003.py` | `6e9e7a913a1a1da85a2af58fbcd13e43f019c8eb838a2a2293e631542b8d0612` | not-applicable | local verifier probe |
| `log/source_equivalence_003_attempt1.log` | `cd12eb953e51d25384ff8b8a1f128ef00f12f4b524ae6e77f4e61d5ca64ae5ca` | not-applicable | retained diagnostic-defect evidence |
| `log/source_equivalence_003.log` | `76ac6670f4d55792677d1ca5c8df093118603779a5529443e39534cc2868a351` | not-applicable | corrected source evidence |
| `log/correctness_003.log` | `beff240e9b883d781eb486651c10076ed83ae05cab17a8ab8434f57ebd2e797b` | not-applicable | correctness evidence |
| `log/tie_id_parity_003.py` | `2c615c7cf3542ed4ff6cd187baa57ad81de12ef1b087f2fbd98e894f6eac782e` | same | matched targeted probe |
| `log/tie_id_parity_003.log` | `08819fb8f8490582ab0b535a2d78ca7d19790602aaae3db805d4a24814faf7f8` | not-applicable | tie evidence |
| `log/wall_003_sample_1.log` | `9bb17179b7c72c19c104da05eed4b2ec8bef6ecb33977e000f3cb834aba7bf28` | not-applicable | wall evidence |
| `log/wall_003_sample_2.log` | `35ca875aa999badc0542cfb81909d7a7f3e13e1c5787c534133942364399e3c8` | not-applicable | wall evidence |
| `log/wall_003_sample_3.log` | `d183b8f6b7fb9f90ec5c579dccabf293c717f76021cca19bd468b5097c375615` | not-applicable | wall evidence |
| `rounds/report_003.md` | `6c645acf858745585d4f668546609dc9d3dbc3f7c1b8110a013193f6c89c2fdd` | not-applicable | final report |
| `state/verifier_context.md` | `0ed07c489c6f1445cac17bb876bc60c119324c8d8d741dd7fc3bc9ed87ddd545` | not-applicable | updated durable context |

## Completed Commands

1. Re-read mandatory Verifier contract, Codex adapter, report template, invariants, Triton-MACA target profile, project, team-state, Round 003 decision/Coder result/candidate, accepted canonical/adapter, reports 001/002, and Verifier context: exit `0`.
2. Local SHA256 verification: exit `0`.
3. Remote executable-file SHA256 verification: exit `0`.
4. Local source-equivalence probe attempt 1: exit `0`, but a non-gating diagnostic regex escaped `\\d` literally and reported `group_argmax=0`; exact whole-file and frozen-region checks passed; evidence preserved in `log/source_equivalence_003_attempt1.log`.
5. Regex-only corrected source-equivalence probe: exit `0`; exact eight authorized replacements only, combined max-with-indices `8`, explicit-left-tie `8`, expert argmax `0`, expert selected-value sum `0`, group argmax `4`, two independent host allocations, frozen group/post-selection/host/launch, and one-line adapter identity all passed; evidence `log/source_equivalence_003.log`.
6. Standard correctness (`warmup=5`, `repeat=10`, `--full-traceback`): exit `0`; elapsed `15.2843468 s`; `PASS accuracy`; `1 passed, 0 failed`; smoke-only `reference=0.073956 ms`, `candidate=0.069987 ms`; evidence `log/correctness_003.log`.
7. Remote Round 003 tie-probe SHA256 check: exit `0`; remote `2c615c7cf3542ed4ff6cd187baa57ad81de12ef1b087f2fbd98e894f6eac782e`, exactly matching local.
8. Targeted group/expert cutoff tie parity: exit `0`; elapsed `14.9766266 s`; both cases passed exact IDs, weights tolerance, true fixed fast path, fallback trap, output contracts, and input non-mutation; evidence `log/tie_id_parity_003.log`.
9. Formal wall sample 1 (`warmup=200`, `repeat=500`): exit `0`; elapsed `15.3001230 s`; `reference=0.067296 ms`, `candidate=0.067263 ms`; evidence `log/wall_003_sample_1.log`.
10. Formal wall sample 2 (`warmup=200`, `repeat=500`): exit `0`; elapsed `15.2061405 s`; `reference=0.067085 ms`, `candidate=0.067139 ms`; evidence `log/wall_003_sample_2.log`.
11. Formal wall sample 3 (`warmup=200`, `repeat=500`): exit `0`; elapsed `14.8943377 s`; `reference=0.072567 ms`, `candidate=0.068747 ms`; evidence `log/wall_003_sample_3.log`.
12. Independent median/adoption calculation: exit `0`; reference median `0.067296 ms`, candidate median `0.067263 ms`, unrounded improvement `0.04903708987159917%`, median speedup `1.0004906114803085x`; failed required `+5%` threshold.
13. Targeted forward profiler: not run because the strict Round 003 sequence permits it only after the formal wall gate passes; no Round 003 trace was created.

## Raw Samples

- standard correctness: `pass`; smoke-only `reference=0.073956 ms`, `candidate=0.069987 ms`; evidence `log/correctness_003.log`
- targeted tie parity: `PASS`; group-cutoff IDs `[0,32,64,96,1,2,3,4]`, max weight diff `0.0`; expert-cutoff IDs `[0,32,64,96,1,33,65,2]`, max weight diff `4.656612873077393e-10`; evidence `log/tie_id_parity_003.log`
- reference wall samples ms: `[0.067296, 0.067085, 0.072567]`; median `0.067296`
- candidate wall samples ms: `[0.067263, 0.067139, 0.068747]`; median `0.067263`
- unrounded improvement: `0.04903708987159917%`
- median speedup: `1.0004906114803085x`
- profile: `not-run: formal wall failed the mandatory +5% precondition`
- hypothesis verdict: `partially-confirmed: reduction/capability/tie mechanisms passed; primary wall-time claim falsified`

## Next Safe Action

Orchestrator validates `rounds/report_003.md`, this status, and `state/verifier_context.md`; records `no-improvement`, retains `triton_grouped_topk_001.py` / `rounds/report_001.md` as canonical, updates counters, and releases measurement exclusivity. No source or team-state change is Verifier-owned.

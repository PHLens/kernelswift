# Report 001

## Metadata

```json
{"schema_version": 1, "round": "001"}
```

## vNext Fact Pack

```json
{
  "candidate_sha256": "__CANDIDATE_SHA256__",
  "correctness": {
    "evidence": [
      "python3 auto_bench.py --check-correctness --v1_file candidate.py"
    ],
    "status": "pass"
  },
  "evidence_gap_cause": "none",
  "lowering": {
    "evidence": [
      "log/profiler_candidate_summary.json"
    ],
    "evidence_contract": "mlu-kernel-summary-v1",
    "expected_mechanism": "absent",
    "status": "observed"
  },
  "observables": [
    {
      "confidence": "high",
      "evidence": [
        "log/profiler_candidate_summary.json"
      ],
      "name": "external-kernel-count",
      "status": "observed",
      "value": "3 -> 2"
    }
  ],
  "schema_version": 1
}
```

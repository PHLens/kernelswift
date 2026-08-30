# Report 001

## Metadata

```json
{"schema_version": 1, "round": "001"}
```

## vNext Fact Pack

```json
{
  "candidate_sha256": "88c41c1f1d6ee5fb35a55f1f8638f3dd3f4b27c63a4a2d91b54f5b9a6d8c7e31",
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

# Report 001

## Metadata

```json
{"schema_version": 1, "round": "001"}
```

## vNext Fact Pack

```json
{
  "candidate_sha256": "__PINNED_SHA256__",
  "correctness": {
    "evidence": [],
    "status": "pass"
  },
  "evidence_gap_cause": "none",
  "final_configuration_tuning": {
    "post_pin_official": {
      "candidate_sha256": "__PINNED_SHA256__",
      "correctness": {
        "evidence": [],
        "status": "pass"
      },
      "lowering": {
        "evidence": [],
        "evidence_contract": "fixture-summary-v1",
        "expected_mechanism": "present",
        "status": "observed"
      },
      "official_evidence": true,
      "promotion_evidence": true
    },
    "search_trials": [
      {
        "comparable": true,
        "compile_status": "ok",
        "configuration": {
          "num_stages": 2,
          "num_warps": 1
        },
        "correctness_status": "pass",
        "eligibility": true,
        "measurement_count": 20,
        "order_index": 0,
        "rejection_code": null,
        "reset_status": "ok",
        "statistic": 100.0
      },
      {
        "comparable": true,
        "compile_status": "ok",
        "configuration": {
          "num_stages": 2,
          "num_warps": 2
        },
        "correctness_status": "pass",
        "eligibility": true,
        "measurement_count": 20,
        "order_index": 1,
        "rejection_code": null,
        "reset_status": "ok",
        "statistic": 80.0
      }
    ],
    "selected_configuration": {
      "num_stages": 2,
      "num_warps": 2
    },
    "selection_outcome": "improved",
    "submission_snapshot_id": "__SNAPSHOT_ID__",
    "temporary_storage_clean": true
  },
  "lowering": {
    "evidence": [],
    "evidence_contract": "fixture-summary-v1",
    "expected_mechanism": "present",
    "status": "observed"
  },
  "observables": [
    {
      "confidence": "high",
      "evidence": [],
      "name": "external-kernel-count",
      "status": "observed",
      "value": "1"
    }
  ],
  "schema_version": 1
}
```

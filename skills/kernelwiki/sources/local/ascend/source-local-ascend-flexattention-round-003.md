---
schema_version: 1
id: source-local-ascend-flexattention-round-003
source_kind: local-campaign
title: 'Reviewed historical local campaign evidence: kernels/track1-triton/flexattention/ascend#round-003'
url: local://da0304f5d6f15b505bca8472cb1264ed9437ef7f/kernels/track1-triton/flexattention/ascend
repository_id: local
captured_at: '2026-08-18T23:18:02Z'
target_disposition: exact
target_ids:
- ascend910b4
implementation_profile_ids:
- triton_ascend
runtime_fingerprints:
- triton-3.2.0 torch-npu-2.7.1.post4
languages:
- triton
kernel_types:
- attention
techniques:
- tiling
- work-partitioning
hardware_features:
- cube
- vector
tags:
- ascend
- device-win-wall-loss
- host-bound
- tiling
- triton-ascend
- work-partitioning
license_state: unknown
audiences:
- designer
profile_authority: historical-noncanonical
strict_vnext_validated: false
missing_evidence:
- binding:missing-vnext-artifact
- sketch:missing-vnext-artifact
- verdict:missing-vnext-artifact
---
# Reviewed historical local campaign evidence: kernels/track1-triton/flexattention/ascend#round-003

Immutable reviewed historical campaign evidence. This Source is metadata-only, Designer-only, and not validated against the current vNext contract.

## Curator review

```json
{
  "decision": "include",
  "proposal_id": "experience-historical-source-local-ascend-flexattention-round-003",
  "proposal_sha256": "e718b0fc479966aa050d74230a20b3c2faacee6cd972a67d8859d49ec31c0472",
  "publication_target": {
    "card_id": "pattern-device-win-wall-loss",
    "mode": "existing-card-example"
  },
  "rationale": "Include as a Designer-only existing-Card counterexample. At committed development identity da0304f5d6f15b505bca8472cb1264ed9437ef7f, the hash-bound Round 003 Decision, candidate, Coder result, report, round status, project, and team state explicitly show correctness pass for exactly Ascend910B4, triton_ascend, Triton 3.2.0 / torch_npu 2.7.1.post4, tokens=83, heads=8, head_size=64, fp16 inputs/output: tl.dot reduced device time from 54.4268 to 24.0532 us/call while kernel count stayed 1, but synchronized wall median regressed from 0.296535 to 0.321280 ms (-8.3447%). This is a historical-noncanonical, metadata-only device-win/wall-loss example only; attribution and transfer are limited to the exact runtime, synchronization policy, host path, shape, dtype, and harness, with no Coder visibility.",
  "reviewed_at": "2026-08-21T00:00:00Z",
  "reviewed_by": "kernelwiki-curator"
}
```

## Reviewed proposal evidence

```json
{
  "artifact_hashes": {
    "candidate": "4faadac6cd0e3bb5d1faeaddafd899f0fd64c275632d2635f1612bf182686546",
    "coder-result": "ed8b7de8dc445007d2a25924848cfbe93f613b5be434b167e3d1408cf406f5f9",
    "decision": "c2d0d068f7595bed4aec4e2497b9b390ae875f67dcbcf9de551b448383991b37",
    "project": "40249adadf90b99b8b77c1a53247b4bde26515c12265ad95b708b330c7818e40",
    "report": "f1db9f584f065974490c42ba2ef49e681360afb3ff25d0c3cdd3a4f75e2b66a3",
    "round-status": "9a338e3226d42289c2ef40e634f386f60ae18160dbc489b3e7b38dd18bbe5a84",
    "team-state": "b7f37dcf012924c4c9e09eac7cc5fd178ce184fe308c4dbe51a1e1aa62e88984"
  },
  "missing_evidence": [
    "sketch:missing-vnext-artifact",
    "binding:missing-vnext-artifact",
    "verdict:missing-vnext-artifact"
  ],
  "observed": [
    {
      "evidence_ref": "kernels/track1-triton/flexattention/ascend/rounds/report_003.md",
      "metric": "correctness_pass",
      "statistic": "exact",
      "unit": "boolean",
      "value": true
    },
    {
      "evidence_ref": "kernels/track1-triton/flexattention/ascend/rounds/report_003.md",
      "metric": "reference_wall_time_ms",
      "statistic": "median",
      "unit": "milliseconds",
      "value": 0.296535
    },
    {
      "evidence_ref": "kernels/track1-triton/flexattention/ascend/rounds/report_003.md",
      "metric": "candidate_wall_time_ms",
      "statistic": "median",
      "unit": "milliseconds",
      "value": 0.32128
    },
    {
      "evidence_ref": "kernels/track1-triton/flexattention/ascend/rounds/report_003.md",
      "metric": "wall_improvement_pct",
      "statistic": "exact",
      "unit": "percent",
      "value": -8.344714789147998
    },
    {
      "evidence_ref": "kernels/track1-triton/flexattention/ascend/rounds/report_003.md",
      "metric": "reference_device_us_per_call",
      "statistic": "exact",
      "unit": "microseconds",
      "value": 54.4268
    },
    {
      "evidence_ref": "kernels/track1-triton/flexattention/ascend/rounds/report_003.md",
      "metric": "candidate_device_us_per_call",
      "statistic": "exact",
      "unit": "microseconds",
      "value": 24.0532
    }
  ],
  "scope": {
    "allowed_audiences": [
      "designer"
    ],
    "asset_mode": "metadata-only",
    "audiences": [
      "designer"
    ],
    "captured_at": "2026-08-18T23:18:02Z",
    "comparability": "historical-local",
    "hardware_features": [
      "cube",
      "vector"
    ],
    "implementation_profile_id": "triton_ascend",
    "kernel_types": [
      "attention"
    ],
    "languages": [
      "triton"
    ],
    "license_state": "unknown",
    "measurement_fingerprint": "c1359d456700562802630e66368ce04856d871a993562ce1437e037af82581b8",
    "profile_authority": "historical-noncanonical",
    "repository_id": "local",
    "source_id": "source-local-ascend-flexattention-round-003",
    "tags": [
      "ascend",
      "device-win-wall-loss",
      "host-bound",
      "tiling",
      "triton-ascend",
      "work-partitioning"
    ],
    "target_id": "ascend910b4",
    "techniques": [
      "tiling",
      "work-partitioning"
    ]
  },
  "terminal": {
    "commit": "da0304f5d6f15b505bca8472cb1264ed9437ef7f",
    "local_locator": "kernels/track1-triton/flexattention/ascend#round-003",
    "project_path": "kernels/track1-triton/flexattention/ascend",
    "result": "no-improvement",
    "strict_vnext_validated": false
  },
  "transfer_boundaries": [
    "target=ascend910b4",
    "profile=triton_ascend historical-noncanonical",
    "runtime=triton-3.2.0 torch-npu-2.7.1.post4",
    "shape=tokens83-heads8-headsize64-kvheads8",
    "dtype=query-key-value-fp16 output-fp16 accumulation-fp32",
    "round=003",
    "measurement=c1359d456700562802630e66368ce04856d871a993562ce1437e037af82581b8"
  ]
}
```

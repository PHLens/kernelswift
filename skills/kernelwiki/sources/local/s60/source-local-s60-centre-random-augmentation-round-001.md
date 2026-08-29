---
schema_version: 1
id: source-local-s60-centre-random-augmentation-round-001
source_kind: local-campaign
title: 'Reviewed historical local campaign evidence: kernels/track1-triton/centre_random_augmentation/s60/epoch2#round-001'
url: local://4ff2094d96c66fde22192f8283113228123f7397/kernels/track1-triton/centre_random_augmentation/s60/epoch2
repository_id: local
captured_at: '2026-08-29T16:00:00Z'
target_disposition: exact
target_ids:
- s60
implementation_profile_ids:
- triton_gcu
runtime_fingerprints:
- triton 3.6.0 / triton_gcu 3.6.0+1.0.20260722 / torch 2.10.0+cpu / torch_gcu 2.10.0+3.8.0.2
languages:
- python
- triton
kernel_types:
- data-preparation
techniques:
- kernel-fusion
- launch-collapse
hardware_features:
- vector
tags:
- gcu
- kernel-fusion
- launch-bound
- launch-collapse
- s60
- triton-gcu
license_state: unknown
audiences:
- designer
profile_authority: historical-noncanonical
strict_vnext_validated: false
missing_evidence:
- verdict-round-profile-target-not-modeled
---
# Reviewed historical local campaign evidence: kernels/track1-triton/centre_random_augmentation/s60/epoch2#round-001

Immutable reviewed historical campaign evidence. This Source is metadata-only, Designer-only, and not validated against the current vNext contract.

## Curator review

```json
{
  "decision": "include",
  "proposal_id": "experience-historical-source-local-s60-centre-random-augmentation-round-001",
  "proposal_sha256": "c1c65636184d1f37324746afb399fec100cc0a708af35a6f9889fb8922de72d7",
  "publication_target": {
    "card_id": "pattern-launch-bound-materialization",
    "mode": "existing-card-example"
  },
  "rationale": "Include as a Designer-only positive example of the launch-bound materialization pattern. At committed identity 4ff2094, the hash-bound report and verdict show correctness pass for S60 (Enflame GCU), triton_gcu, triton 3.6.0, n_sample=4/n_atom=256 fp32: fusing quaternion->rotation-matrix + 3x3 matvec + translation + masking into a single Triton kernel collapsed topsLaunchKernel from 96 to 10 per call and improved wall median from 3.025109 to 1.585115 ms (+47.6%, accepted). This is historical-noncanonical, metadata-only, Designer-only evidence; attribution and transfer are limited to the exact runtime, shape, dtype, and harness, with no Coder visibility.",
  "reviewed_at": "2026-08-29T16:00:00Z",
  "reviewed_by": "kernelwiki-curator"
}
```

## Reviewed proposal evidence

```json
{
  "artifact_hashes": {
    "base": "02e7020fef34db401cfc2cf8031700262493ab96a89449d4e70d333978e78553",
    "candidate": "542293c0ed3488b4f30c6c3758780115325593a592b11bc656cfa605f9d79522",
    "decision": "459a1f9b36105b33966c53b3e7740313094ba96874ffae1be358171066948c40",
    "project": "b55baedc8fc59fe6a33cdfad0c32b95228c1a485a7f2f67046a4557d8880749d",
    "report": "5154e09de23fc7f90829f03ade073b5979685da80e53903fb0d5a98730c71f1a",
    "sketch": "017b423b96d88ba28fde6f1d4d6a7534b9f0fcf486a540d78c7c59f149c4429f",
    "team-state": "c81fb6b4b337c5c81e7b3f2f5899686cca52ba3713c373468a85c1329849307a",
    "verdict": "9b9ce67031ecc205287aa6eb6920e0b5f6db48d76a708f1a83f381785cc8edd8"
  },
  "missing_evidence": [
    "verdict-round-profile-target-not-modeled"
  ],
  "observed": [
    {
      "evidence_ref": "kernels/track1-triton/centre_random_augmentation/s60/epoch2/rounds/report_001.md",
      "metric": "correctness_pass",
      "statistic": "exact",
      "unit": "boolean",
      "value": true
    },
    {
      "evidence_ref": "kernels/track1-triton/centre_random_augmentation/s60/epoch2/rounds/report_001.md",
      "metric": "reference_wall_time_ms",
      "statistic": "median",
      "unit": "milliseconds",
      "value": 3.025109
    },
    {
      "evidence_ref": "kernels/track1-triton/centre_random_augmentation/s60/epoch2/rounds/report_001.md",
      "metric": "candidate_wall_time_ms",
      "statistic": "median",
      "unit": "milliseconds",
      "value": 1.585115
    },
    {
      "evidence_ref": "kernels/track1-triton/centre_random_augmentation/s60/epoch2/rounds/report_001.md",
      "metric": "wall_improvement_pct",
      "statistic": "exact",
      "unit": "percent",
      "value": 47.6
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
    "captured_at": "2026-08-29T16:00:00Z",
    "comparability": "historical-local",
    "hardware_features": [
      "vector"
    ],
    "implementation_profile_id": "triton_gcu",
    "kernel_types": [
      "data-preparation"
    ],
    "languages": [
      "python",
      "triton"
    ],
    "license_state": "unknown",
    "measurement_fingerprint": null,
    "profile_authority": "historical-noncanonical",
    "repository_id": "local",
    "source_id": "source-local-s60-centre-random-augmentation-round-001",
    "tags": [
      "gcu",
      "kernel-fusion",
      "launch-bound",
      "launch-collapse",
      "s60",
      "triton-gcu"
    ],
    "target_id": "s60",
    "techniques": [
      "kernel-fusion",
      "launch-collapse"
    ]
  },
  "terminal": {
    "commit": "4ff2094d96c66fde22192f8283113228123f7397",
    "local_locator": "kernels/track1-triton/centre_random_augmentation/s60/epoch2#round-001",
    "project_path": "kernels/track1-triton/centre_random_augmentation/s60/epoch2",
    "result": "accepted",
    "strict_vnext_validated": false
  },
  "transfer_boundaries": [
    "target=s60",
    "profile=triton_gcu historical-noncanonical",
    "runtime=triton 3.6.0 / triton_gcu 3.6.0+1.0.20260722 / torch 2.10.0+cpu / torch_gcu 2.10.0+3.8.0.2",
    "shape=n-sample4-n-atom256",
    "dtype=fp32",
    "round=001"
  ]
}
```

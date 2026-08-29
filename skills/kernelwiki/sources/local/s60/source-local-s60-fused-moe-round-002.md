---
schema_version: 1
id: source-local-s60-fused-moe-round-002
source_kind: local-campaign
title: 'Reviewed historical local campaign evidence: kernels/track1-triton/fused_moe/s60#round-002'
url: local://4ff2094d96c66fde22192f8283113228123f7397/kernels/track1-triton/fused_moe/s60
repository_id: local
captured_at: '2026-08-29T18:00:00Z'
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
- moe
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
- sketch:missing
- verdict:missing
---
# Reviewed historical local campaign evidence: kernels/track1-triton/fused_moe/s60#round-002

Immutable reviewed historical campaign evidence. This Source is metadata-only, Designer-only, and not validated against the current vNext contract.

## Curator review

```json
{
  "decision": "include",
  "proposal_id": "experience-historical-source-local-s60-fused-moe-round-002",
  "proposal_sha256": "7635d7d18e43c623a43286378d555cc72c243d681174dbd32eb05c63217451b1",
  "publication_target": {
    "card_id": "pattern-launch-bound-materialization",
    "mode": "existing-card-example"
  },
  "rationale": "Include as Designer-only positive example of launch-bound materialization. S60 (Enflame GCU), triton_gcu, round-002: per-token routing + selection fusion collapsed 147 launches to a single kernel, 5.112406 -> 0.390289 ms (+92.37%, accepted).",
  "reviewed_at": "2026-08-29T18:00:00Z",
  "reviewed_by": "kernelwiki-curator"
}
```

## Reviewed proposal evidence

```json
{
  "artifact_hashes": {
    "base": "21e758535af58c21684d6e657ae44f91a0248d246eb424b6923a3c180bc32a3d",
    "candidate": "e5d0058d6fb6f23f94e2623ae374d2776c4b2c6d4eb235b2c6c75524fb44eb73",
    "decision": "a23045c57f2be98edb6ba8e83f9575f877b5c99ef7b2599d3f0d4a76332c6387",
    "report": "7c353508debd9b75b62e4cca693840c2ffccad8c027a7b2498f596c7f1aea35a"
  },
  "missing_evidence": [
    "verdict:missing",
    "sketch:missing"
  ],
  "observed": [
    {
      "evidence_ref": "kernels/track1-triton/fused_moe/s60/rounds/report_002.md",
      "metric": "correctness_pass",
      "statistic": "exact",
      "unit": "boolean",
      "value": true
    },
    {
      "evidence_ref": "kernels/track1-triton/fused_moe/s60/rounds/report_002.md",
      "metric": "wall_improvement_pct",
      "statistic": "exact",
      "unit": "percent",
      "value": 92.37
    },
    {
      "evidence_ref": "kernels/track1-triton/fused_moe/s60/rounds/report_002.md",
      "metric": "wall_time_ms",
      "statistic": "median",
      "unit": "milliseconds",
      "value": 0.390289
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
    "captured_at": "2026-08-29T18:00:00Z",
    "comparability": "historical-local",
    "hardware_features": [
      "vector"
    ],
    "implementation_profile_id": "triton_gcu",
    "kernel_types": [
      "moe"
    ],
    "languages": [
      "python",
      "triton"
    ],
    "license_state": "unknown",
    "measurement_fingerprint": null,
    "profile_authority": "historical-noncanonical",
    "repository_id": "local",
    "source_id": "source-local-s60-fused-moe-round-002",
    "tags": [
      "gcu",
      "s60",
      "launch-bound",
      "kernel-fusion",
      "launch-collapse",
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
    "local_locator": "kernels/track1-triton/fused_moe/s60#round-002",
    "project_path": "kernels/track1-triton/fused_moe/s60",
    "result": "accepted",
    "strict_vnext_validated": false
  },
  "transfer_boundaries": [
    "target=s60",
    "profile=triton_gcu historical-noncanonical",
    "runtime=triton 3.6.0 / triton_gcu 3.6.0+1.0.20260722 / torch 2.10.0+cpu / torch_gcu 2.10.0+3.8.0.2",
    "shape=tokens83-h128-e8-topk2-i64",
    "dtype=fp16",
    "round=002"
  ]
}
```

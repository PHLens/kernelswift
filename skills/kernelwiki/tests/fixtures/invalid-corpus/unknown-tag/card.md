---
schema_version: 1
id: invalid-unknown-tag
title: Invalid unknown tag fixture
type: technique
audiences: [designer]
authority: advisory
summary: This isolated fixture intentionally uses an undeclared tag.
targets: [ascend]
target_match: backend
languages: [triton]
kernel_types: [reduction]
techniques: [kernel-fusion]
hardware_features: [memory-hierarchy]
tags: [not-in-taxonomy]
symptoms: [launch-bound]
sources: [source-valid-manual]
related: []
prerequisites: []
version_sensitive: []
observations: []
examples: []
---
# Invalid unknown tag fixture

This file is intentionally not a complete corpus root. Tests construct an isolated valid tree and apply this single invalid condition, expecting `taxonomy-unknown`.

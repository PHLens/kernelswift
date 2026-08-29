# KernelWiki Evaluation Protocol

## Purpose

Evaluation boundaries are sealed before taxonomy, aliases, ranking, or seed Cards are authored. This prevents the repository or Track 2 holdout answers from becoming tuning input.

The authoritative seal is [`../data/evaluation-holdouts.yaml`](../data/evaluation-holdouts.yaml). The Track 2 gold fixture is [`../tests/fixtures/holdout/track2-sinkhorn-gold.yaml`](../tests/fixtures/holdout/track2-sinkhorn-gold.yaml).

## Sealed lanes

- `triton-ascend-kernels` is a reviewed manual repository lane used only for final evaluation. No Task 1 crawler or capture command reads it.
- `sparse_attn` and `index_topk` are development query contexts, not Card ontologies or operator-specific pages.
- `sinkhorn_normalize` is the final Track 2 holdout context.
- Track 2 paths and operator names must never become Card IDs or Card paths.

## Integrity check

The SHA-256 of `track2-sinkhorn-gold.yaml`, including its final newline, is:

```text
a7ea16d878f10060aff1cf7f5a2b4d99db7f18b297ef523d9d5fda327f4b2c13
```

The value must equal `track2.gold_fixture_sha256` in `evaluation-holdouts.yaml` before final evaluation. A mismatch is an input-validation failure; the evaluator must not run against modified judgments.

## Metrics

For the sealed Designer context:

- `top5_relevant_card_recall` is the number of IDs from `gold.relevant_card_ids` returned among the first five admitted Designer Cards, divided by `metrics.top5_relevant_denominator` (`4`).
- `counterexample_recall` is the number of returned IDs from `gold.counterexample_card_ids`, divided by `metrics.counterexample_denominator` (`1`).
- `capability_gap_recall` is the number of returned IDs from `gold.capability_gap_card_ids`, divided by `metrics.capability_gap_denominator` (`1`).

For the missing-profile Coder context, the evaluator counts:

- unsafe Coder admissions, expected `0`;
- cross-target recipe leaks, expected `0`;
- Unknown capability promotions, expected `0`.

All denominators and expected safety counts come from the sealed fixture. They are not inferred from required concepts or result text.

## No-tuning rule

Later tasks may read the gold fixture only during final evaluation. If a safety metric fails, fix admission logic. If a retrieval metric fails, do not edit the holdout, taxonomy, aliases, or rankings based on hidden answer details. Record the failure and open a separately reviewed source-backed curation change before any rerun.

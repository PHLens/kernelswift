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

The evaluator reads the requested manifest and gold files once in the normal CLI path. It hashes the gold bytes before parsing and requires equality with `track2.gold_fixture_sha256` from the parsed manifest. A mismatch or malformed input exits `2`; no alternate sealing, proxy, or filesystem-authority layer is used.

## Development and adversarial cases

`data/track2-development-queries.yaml` and the two development JSON fixtures contain structured `sparse_attn` and `index_topk` contexts only: target, language, kernel types, semantic features, dtypes, shape, and expected general knowledge categories. They contain no source code, recipe, Card body, or operator Card.

The five fixed adversarial contracts are:

1. generic `tl.dot` evidence does not satisfy dtype/shape-specific capability;
2. positive output reuse does not hide a conflicting counterexample;
3. device-time improvement does not imply wall-time improvement;
4. grouped-top-k evidence remains bounded when querying index-top-k;
5. raw profiler evidence does not become CANN device attribution.

Each case runs the existing admission-first role search once for Designer and once for the real missing-profile AscendC Coder context. The evaluator does not validate caller-forged internal role-result dictionaries; it consumes the existing validated search result directly.

## Metrics

For the sealed Designer context:

- `top5_relevant_card_recall` counts sealed relevant Card IDs among the first five admitted Designer Card IDs.
- `counterexample_recall` counts sealed counterexample IDs only from the Designer `counterexamples` group.
- `capability_gap_recall` counts sealed gap IDs only from `capability_gaps[].card_id` in the Designer `capability_gaps` group.

For each missing-profile Coder context, the evaluator counts:

- unsafe Coder admissions or admitted guidance/example/asset IDs;
- Unknown capability records promoted to admitted results;
- forbidden cross-target recipe terms exposed through admitted guidance/example/asset IDs.

Numerators are computed from the role-search result. Denominators and expected safety counts come from the sealed gold fixture.

## No-tuning rule

If a safety metric fails, fix admission logic. If a retrieval metric fails, do not edit the holdout, taxonomy, aliases, rankings, or Cards based on hidden answer details. Record the failure as `recorded-no-tuning`; any retrieval improvement requires a separate source-backed curation change.

## Final command

```bash
python3 skills/kernelwiki/scripts/evaluate_holdout.py \
  --manifest skills/kernelwiki/data/evaluation-holdouts.yaml \
  --gold skills/kernelwiki/tests/fixtures/holdout/track2-sinkhorn-gold.yaml
```

Successful evaluation prints deterministic canonical JSON containing the five adversarial results, sealed holdout group IDs, numerators, denominators, safety metrics, and retrieval-gate status.

## Local-campaign knowledge-lift holdout

[`../data/local-campaign-holdout.yaml`](../data/local-campaign-holdout.yaml) is sealed before any campaign-to-proposal mapping code is written. Its three development paths may be used only after Task 1; its two holdout paths are not inspected until both strict and historical lanes pass their focused tests.

The local holdout does not influence outcome mapping, publication defaults, Card naming, or Coder eligibility. Historical campaign evidence remains noncanonical and Designer-only. Evaluation records include/defer/exclude outcomes without changing holdout membership or rewriting historical evidence.

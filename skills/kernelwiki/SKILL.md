---
name: kernelwiki
description: Curate and query an offline, provenance-pinned, Ascend-first kernel engineering wiki without changing campaign authority.
---

# KernelWiki

KernelWiki is a standalone, Git-versioned knowledge skill organized as:

```text
sources/ -> wiki/ -> queries/
```

Reviewed Source records preserve provenance. Generic Wiki Cards explain reusable hardware, language, runtime, measurement, technique, and failure-pattern knowledge. Generated query views and the compiled Card catalog support deterministic offline navigation.

## Core entry points

The standalone core exposes only these command families:

- `scripts/validate.py`: validate the complete local corpus and generated outputs.
- `scripts/capture_source.py`: perform explicit maintenance-time source discovery or immutable capture.
- `scripts/generate_indices.py`: generate the Card-only catalog and checked-in query views.
- `scripts/query.py`: search Cards and Source metadata offline, optionally with a versioned Designer/Coder `--context`.
- `scripts/get_page.py`: retrieve one Card or Source page and optionally follow citations.
- `scripts/grep_wiki.py`: run bounded regex investigation over local Cards and Sources.

The complete standalone Core implements these commands with checked-in schemas, immutable provenance, deterministic generated views, active offline search, and hardware-free contract tests.

## Authority and safety boundaries

- Query, page retrieval, validation, and index generation are offline and never use a network fallback.
- KernelWiki never edits `skills/kernel-opt-loop/`, active campaign state, profiles, base implementations, or benchmark harnesses.
- Wiki evidence is advisory and observation-scoped. It never controls campaign acceptance.
- Unknown or incompatible licenses permit metadata-only records and deny code exposure.
- Repository and Track 2 holdouts are sealed before taxonomy, alias, ranking, or seed-Card work.
- No `claims/` lifecycle, `KnowledgePacket`, required dossier, campaign adapter, or loop write path exists. Phase C uses only a read-only validator bridge.

## Role-aware query

Use a Designer context for broad classified evidence or an exact Coder context for implementation guidance:

```bash
python3 skills/kernelwiki/scripts/query.py "ascend launch" \
  --context skills/kernelwiki/tests/fixtures/role/designer-context.json
python3 skills/kernelwiki/scripts/query.py "ascendc implementation" \
  --context skills/kernelwiki/tests/fixtures/role/coder-missing-profile.json
```

Admission happens before ranking. Results keep positive, counterexample, capability-gap, analogy, and excluded evidence in separate stable groups. Coder guidance and assets require exact profile/runtime/Sketch binding; there is no fallback. The missing AscendC fixture intentionally returns empty implementation guidance.

A caller may add `--output receipt.json`, but that receipt is advisory and never mutates a prompt, campaign, project state, or loop artifact. See [`references/role-query-contract.md`](references/role-query-contract.md).

## Reviewed maintenance order

```text
explicit terminal bundle -> strict validation -> proposal -> Curator review
historical manifest -> proposal candidate -> Curator review -> explicit noncanonical Source capture
reviewed Source -> explicit generic Card edit -> validate -> generate -> diff review -> Git commit
```

For an included historical proposal, the Curator must invoke the Source-only command explicitly:

```bash
python3 skills/kernelwiki/scripts/capture_source.py reviewed-historical \
  --proposal skills/kernelwiki/candidates/experience/<proposal-id>.json \
  --review skills/kernelwiki/candidates/experience/reviews/<proposal-id>.yaml
```

The command verifies proposal/review identity and writes only an immutable, Designer-only local Source. Metadata-only proposals create no artifact bundle. Card edits remain explicit Git edits; extractors and validators never publish a Card or mutate a campaign.

Phase C role-aware query admission and Phase D offline knowledge lift are available only through explicit standalone commands. No Phase E loop adapter exists.

## Development

Install the pinned dependency and run the complete hardware-free acceptance suite:

```bash
python3 -m pip install -r skills/kernelwiki/requirements.txt
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s skills/kernelwiki/tests -p 'test_*.py' -v
python3 skills/kernelwiki/scripts/validate.py
python3 skills/kernelwiki/scripts/generate_indices.py --check
```

See [`README.md`](README.md) for maintenance and smoke commands, [`references/role-query-contract.md`](references/role-query-contract.md) for role admission/receipt semantics, and [`references/evaluation-protocol.md`](references/evaluation-protocol.md) for sealed evaluation rules.

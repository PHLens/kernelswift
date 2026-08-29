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
discover candidates -> curator edits reviewed ledger -> capture immutable Source -> author/review generic Card -> validate -> generate views -> review diff -> commit
```

Source extraction never publishes a Card. Generated files are reviewed artifacts and must remain current with the authored Source/Card corpus.

Phase C role-aware query admission is available through explicit contexts and the read-only bridge. Phase D offline knowledge lift remains separate, and no Phase E loop adapter exists.

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

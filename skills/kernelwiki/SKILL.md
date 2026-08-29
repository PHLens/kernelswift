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
- `scripts/query.py`: search Cards and Source metadata offline.
- `scripts/get_page.py`: retrieve one Card or Source page and optionally follow citations.
- `scripts/grep_wiki.py`: run bounded regex investigation over local Cards and Sources.

The complete standalone Core implements these commands with checked-in schemas, immutable provenance, deterministic generated views, active offline search, and hardware-free contract tests.

## Authority and safety boundaries

- Query, page retrieval, validation, and index generation are offline and never use a network fallback.
- KernelWiki never edits `skills/kernel-opt-loop/`, active campaign state, profiles, base implementations, or benchmark harnesses.
- Wiki evidence is advisory and observation-scoped. It never controls campaign acceptance.
- Unknown or incompatible licenses permit metadata-only records and deny code exposure.
- Repository and Track 2 holdouts are sealed before taxonomy, alias, ranking, or seed-Card work.
- No `claims/` lifecycle, `KnowledgePacket`, required dossier, or loop adapter is part of the standalone core.

## Reviewed maintenance order

```text
discover candidates -> curator edits reviewed ledger -> capture immutable Source -> author/review generic Card -> validate -> generate views -> review diff -> commit
```

Source extraction never publishes a Card. Generated files are reviewed artifacts and must remain current with the authored Source/Card corpus.

Phase C role-aware query admission and Phase D offline knowledge lift have separate approved plans. The standalone Core does not provide those behaviors, and no Phase E loop adapter exists.

## Development

Install the pinned dependency and run the complete hardware-free acceptance suite:

```bash
python3 -m pip install -r skills/kernelwiki/requirements.txt
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s skills/kernelwiki/tests -p 'test_*.py' -v
python3 skills/kernelwiki/scripts/validate.py
python3 skills/kernelwiki/scripts/generate_indices.py --check
```

See [`README.md`](README.md) for maintenance and smoke commands and [`references/evaluation-protocol.md`](references/evaluation-protocol.md) for sealed evaluation rules.

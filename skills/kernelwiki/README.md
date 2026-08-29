# KernelWiki

KernelWiki is an Ascend-first, provenance-pinned, offline kernel engineering wiki. Its authored data flow is `sources/ -> wiki/ -> queries/`; generated query pages and `compiled/catalog.jsonl` are checked into Git for deterministic review.

## Install

KernelWiki pins its only non-standard dependency:

```bash
python3 -m pip install -r skills/kernelwiki/requirements.txt
```

## Standalone Core acceptance

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover \
  -s skills/kernelwiki/tests -p 'test_*.py' -v
python3 skills/kernelwiki/scripts/validate.py
python3 skills/kernelwiki/scripts/generate_indices.py --check
python3 skills/kernelwiki/scripts/query.py "ascend launch overhead" --limit 5
python3 skills/kernelwiki/scripts/get_page.py technique-kernel-fusion --follow-sources
python3 skills/kernelwiki/scripts/grep_wiki.py "device.*wall" --scope wiki
```

The complete suite requires no accelerator, active campaign, network connection, or `kernel-opt-loop` mutation. Contract tests cover generated-file freshness, provenance, deterministic offline output, sealed Track 2 boundaries, role-neutral import isolation, and the core Designer/Coder receipt contract.

## Standalone commands

```bash
python3 skills/kernelwiki/scripts/validate.py
python3 skills/kernelwiki/scripts/capture_source.py --help
python3 skills/kernelwiki/scripts/generate_indices.py --help
python3 skills/kernelwiki/scripts/query.py "kernel fusion" --scope both --limit 20
python3 skills/kernelwiki/scripts/query.py --tag double-buffering --type technique
python3 skills/kernelwiki/scripts/query.py --repo vllm-ascend --format markdown
python3 skills/kernelwiki/scripts/get_page.py technique-kernel-fusion --follow-sources
python3 skills/kernelwiki/scripts/get_page.py sources/docs/source-valid-manual.md --root skills/kernelwiki/tests/fixtures/valid-corpus --frontmatter
python3 skills/kernelwiki/scripts/grep_wiki.py "launch|materialization" --scope both
```

Query, page retrieval, regex investigation, validation, and generation are strictly offline. Only explicit source discovery and capture maintenance may use the network. `--profile-snapshot` is not a role context and fails with `error[phase-c-required]`; use the versioned `--context` interface below.

Search filters are exact, OR within one filter and AND across filters: `--type`, `--tag`, `--repo`/`--repository`, `--language`, `--target`, `--target-match`, `--symptom`, `--kernel-type`, `--evidence-level`, `--reproduction`, `--audience`, and `--has-code true|false`. Default output is canonical JSON; `--format markdown` is a navigation view.

`get_page.py --include-code` requests Designer/general-navigation access to approved local assets. It never supplies Phase C Coder admission. Metadata-only, unapproved, missing, or audience-denied assets remain hidden.

## Role-aware queries

Designer queries admit broad evidence and label its match class:

```bash
python3 skills/kernelwiki/scripts/query.py "ascend launch" \
  --context skills/kernelwiki/tests/fixtures/role/designer-context.json \
  --group-limit admitted=5
```

Coder queries require an exact implementation profile, runtime, authority hashes, and guidance-to-Sketch binding. The real missing AscendC context intentionally returns no guidance or code and does not fall back to Triton/CUDA:

```bash
python3 skills/kernelwiki/scripts/query.py "ascendc implementation" \
  --context skills/kernelwiki/tests/fixtures/role/coder-missing-profile.json
```

Use `--limit` for the default group limit, `--group-limit NAME=N` for one group, and `--show-excluded` to inspect stable denial reasons. `--output /path/to/receipt.json` saves the same canonical JSON result outside active project state. Receipts are advisory: they are not Decisions, Coder results, required dossiers, KnowledgePackets, consultation records, campaign artifacts, or prompt mutations.

Card admission and item admission are separate; readable metadata does not automatically expose examples, guidance, snippets, or full assets. See [`references/role-query-contract.md`](references/role-query-contract.md) for context fields, groups, reasons, no-fallback behavior, and the read-only loop bridge.

## Maintenance order

Always use this reviewed sequence:

```text
explicit terminal bundle -> strict validation -> proposal -> Curator review
historical manifest -> proposal candidate -> Curator review -> explicit noncanonical Source capture
reviewed Source -> explicit generic Card edit -> validate -> generate -> diff review -> Git commit
```

An included historical proposal is materialized only through the explicit Source-only command:

```bash
python3 skills/kernelwiki/scripts/capture_source.py reviewed-historical \
  --proposal skills/kernelwiki/candidates/experience/experience-historical-source-local-ascend-groupedtopk-round-001.json \
  --review skills/kernelwiki/candidates/experience/reviews/experience-historical-source-local-ascend-groupedtopk-round-001.yaml
```

The command verifies the proposal ID/SHA and include review, then writes an immutable Designer-only `local-campaign` Source. Metadata-only candidates copy no code and have no artifact directory. It never edits a Card, generated output, active campaign, profile, prompt, or loop state. Card publication remains an explicit Git edit followed by validation, generation, and diff review.

## Deferred phases

Phase C role-aware query admission and Phase D offline knowledge lift are implemented as standalone read-only/proposal-review maintenance paths. Only the Phase E `kernel-opt-loop` adapter remains excluded.

## Boundaries

KernelWiki does not edit `skills/kernel-opt-loop/`, campaigns, implementation profiles, base files, or harnesses. It does not create operator-named Track 2 Cards, fake AscendC authority, atomic claim records, or persisted KnowledgePackets.

The reviewed repository and Track 2 holdout boundary is stored in [`data/evaluation-holdouts.yaml`](data/evaluation-holdouts.yaml). Its scoring and no-tuning rules are documented in [`references/evaluation-protocol.md`](references/evaluation-protocol.md).

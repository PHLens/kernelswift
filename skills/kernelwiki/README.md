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

The complete suite requires no accelerator, active campaign, network connection, or `kernel-opt-loop` mutation. Contract tests enforce generated-file freshness, provenance and size policies, deterministic offline output, sealed Track 2 boundaries, role-neutral import isolation, and a two-second median latency budget on the checked-in seed corpus.

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

Query, page retrieval, regex investigation, validation, and generation are strictly offline. Only explicit source discovery and capture maintenance may use the network. `query.py --profile-snapshot` fails with `error[phase-c-required]` until the Phase C exact-profile admission plan is installed.

Search filters are exact, OR within one filter and AND across filters: `--type`, `--tag`, `--repo`/`--repository`, `--language`, `--target`, `--target-match`, `--symptom`, `--kernel-type`, `--evidence-level`, `--reproduction`, `--audience`, and `--has-code true|false`. Default output is canonical JSON; `--format markdown` is a navigation view.

`get_page.py --include-code` requests Designer/general-navigation access to approved local assets. It never supplies Phase C Coder admission. Metadata-only, unapproved, missing, or audience-denied assets remain hidden.

## Maintenance order

Always use this reviewed sequence:

```text
discover candidates -> curator edits reviewed ledger -> capture immutable Source -> author/review generic Card -> validate -> generate views -> review diff -> commit
```

Capture never publishes a Card automatically. After authoring or editing a generic Card, run full validation and generation checks, inspect every generated query/catalog diff, and commit the reviewed Source/Card/generated set together as appropriate.

## Deferred phases

Phase C role-aware query admission and Phase D offline knowledge lift are implemented only by their separate approved plans. The standalone Core has no exact-profile Coder admission, no historical campaign lift, and no Phase E `kernel-opt-loop` adapter.

## Boundaries

KernelWiki does not edit `skills/kernel-opt-loop/`, campaigns, implementation profiles, base files, or harnesses. It does not create operator-named Track 2 Cards, fake AscendC authority, atomic claim records, or persisted KnowledgePackets.

The reviewed repository and Track 2 holdout boundary is stored in [`data/evaluation-holdouts.yaml`](data/evaluation-holdouts.yaml). Its scoring and no-tuning rules are documented in [`references/evaluation-protocol.md`](references/evaluation-protocol.md).

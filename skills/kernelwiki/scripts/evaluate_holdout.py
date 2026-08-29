from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import tempfile
from typing import Any

import yaml

from corpus import Corpus
from kernelwiki_common import KernelWikiError, canonical_json_bytes, run_cli, sha256_bytes
from role_context import load_role_context
from role_search import ROLE_GROUPS, parse_role_query_request, role_result_payload, role_search
from validate import validate_skill_root


ADVERSARIAL_CASES = {
    "adversarial-device-wall": "device-time improvement does not imply wall-time improvement",
    "adversarial-dot-scope": "generic tl.dot evidence does not satisfy dtype/shape-specific capability",
    "adversarial-output-reuse": "positive output reuse does not hide a conflicting counterexample",
    "adversarial-profiler-evidence": "raw torch profiler evidence does not become CANN device attribution",
    "adversarial-topk-transfer": "grouped-top-k evidence remains bounded when querying index-top-k",
}
ADVERSARIAL_CASE_NAMES = tuple(f"{case_id}.json" for case_id in sorted(ADVERSARIAL_CASES))
DEVELOPMENT_CONTEXT_IDS = ("index_topk", "sparse_attn")


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise KernelWikiError("holdout-invalid", f"{label} must be an object")
    return value


def _string_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise KernelWikiError("holdout-invalid", f"{label} must be a string list")
    return value


def _positive_int(value: Any, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise KernelWikiError("holdout-invalid", f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise KernelWikiError("holdout-invalid", f"{label} must be a nonnegative integer")
    return value


def _parse_yaml(data: bytes, label: str) -> Mapping[str, Any]:
    try:
        return _mapping(yaml.safe_load(data), label)
    except yaml.YAMLError as error:
        raise KernelWikiError("holdout-invalid", f"invalid {label}: {error}") from error


def load_evaluation_cases(root: Path) -> tuple[Mapping[str, Any], ...]:
    fixture_root = Path(root) / "tests" / "fixtures" / "track2"
    cases: list[Mapping[str, Any]] = []
    for name in ADVERSARIAL_CASE_NAMES:
        path = fixture_root / name
        try:
            document = _mapping(json.loads(path.read_text(encoding="utf-8")), name)
        except (OSError, json.JSONDecodeError) as error:
            raise KernelWikiError("holdout-invalid", f"cannot load {name}: {error}", path) from error
        case_id = document.get("case_id")
        if case_id not in ADVERSARIAL_CASES or document.get("contract") != ADVERSARIAL_CASES[case_id]:
            raise KernelWikiError("holdout-invalid", f"unexpected adversarial contract in {name}", path)
        cases.append(document)
    if {case["case_id"] for case in cases} != set(ADVERSARIAL_CASES):
        raise KernelWikiError("holdout-invalid", "the fixed five adversarial cases are required")
    return tuple(cases)


def _role_payload(corpus: Corpus, query_text: str, context: Mapping[str, Any], directory: Path, name: str) -> Mapping[str, Any]:
    path = directory / f"{name}.json"
    path.write_bytes(canonical_json_bytes(context))
    loaded = load_role_context(path)
    request = parse_role_query_request(query_text, group_limits={group: 50 for group in ROLE_GROUPS}, show_excluded=True)
    return role_result_payload(role_search(corpus, request, loaded, None))


def _group_ids(payload: Mapping[str, Any]) -> dict[str, list[str]]:
    groups = _mapping(payload.get("groups"), "role result groups")
    return {
        group: [str(item["id"]) for item in groups.get(group, ()) if isinstance(item, Mapping) and "id" in item]
        for group in ROLE_GROUPS
    }


def _unsafe_coder_admissions(payload: Mapping[str, Any]) -> int:
    count = 0
    groups = _mapping(payload.get("groups"), "Coder groups")
    for entries in groups.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            admission = entry.get("admission", {})
            if not isinstance(admission, Mapping):
                continue
            admitted_ids = sum(
                len(admission.get(name, ()))
                for name in ("admitted_guidance_ids", "admitted_example_ids", "admitted_asset_ids")
                if isinstance(admission.get(name, ()), list)
            )
            if admission.get("status") == "admitted" or admitted_ids:
                count += 1
    return count


def _unknown_promotions(payload: Mapping[str, Any]) -> int:
    count = 0
    groups = _mapping(payload.get("groups"), "role result groups")
    for entries in groups.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            admission = entry.get("admission", {})
            if not isinstance(admission, Mapping) or admission.get("status") != "admitted":
                continue
            gaps = entry.get("capability_gaps", ())
            if isinstance(gaps, list) and any(
                isinstance(gap, Mapping) and str(gap.get("capability_status", "")).casefold() == "unknown"
                for gap in gaps
            ):
                count += 1
    return count


def _cross_target_recipe_leaks(payload: Mapping[str, Any], forbidden_terms: Sequence[str]) -> int:
    groups = _mapping(payload.get("groups"), "Coder groups")
    leaks = 0
    for entries in groups.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            admission = entry.get("admission", {})
            if not isinstance(admission, Mapping):
                continue
            exposed = {
                name: admission.get(name, ())
                for name in ("admitted_guidance_ids", "admitted_example_ids", "admitted_asset_ids")
            }
            text = json.dumps(exposed, sort_keys=True).casefold()
            if any(term.casefold() in text for term in forbidden_terms):
                leaks += 1
    return leaks


def _evaluate_case(corpus: Corpus, case: Mapping[str, Any]) -> Mapping[str, Any]:
    case_id = str(case.get("case_id"))
    if ADVERSARIAL_CASES.get(case_id) != case.get("contract"):
        raise KernelWikiError("holdout-invalid", f"unexpected contract for {case_id}")
    query_text = case.get("query_text")
    if not isinstance(query_text, str) or not query_text:
        raise KernelWikiError("holdout-invalid", f"{case_id} query_text is required")
    with tempfile.TemporaryDirectory(prefix="kernelwiki-eval-") as temporary:
        directory = Path(temporary)
        designer = _role_payload(corpus, query_text, _mapping(case.get("designer_context"), "Designer context"), directory, "designer")
        coder = _role_payload(corpus, query_text, _mapping(case.get("coder_context"), "Coder context"), directory, "coder")
    designer_ids = _group_ids(designer)
    coder_ids = _group_ids(coder)
    required = _string_list(case.get("required_designer_card_ids", []), "required_designer_card_ids")
    present = {item for values in designer_ids.values() for item in values}
    unsafe = _unsafe_coder_admissions(coder)
    unknown = _unknown_promotions(coder)
    leaks = _cross_target_recipe_leaks(coder, _string_list(case.get("forbidden_coder_terms", []), "forbidden_coder_terms"))
    return {
        "case_id": case_id,
        "contract": case["contract"],
        "designer_group_card_ids": designer_ids,
        "coder_group_card_ids": coder_ids,
        "missing_required_designer_card_ids": sorted(set(required) - present),
        "unsafe_coder_admissions": unsafe,
        "unknown_promotions": unknown,
        "cross_target_recipe_leaks": leaks,
        "passed": unsafe == unknown == leaks == 0,
    }


def evaluate_queries(corpus: Corpus, cases: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if len(cases) != 5 or {case.get("case_id") for case in cases} != set(ADVERSARIAL_CASES):
        raise KernelWikiError("holdout-invalid", "the fixed five adversarial cases are required")
    evaluated = sorted((_evaluate_case(corpus, case) for case in cases), key=lambda item: item["case_id"])
    unsafe = sum(item["unsafe_coder_admissions"] for item in evaluated)
    unknown = sum(item["unknown_promotions"] for item in evaluated)
    leaks = sum(item["cross_target_recipe_leaks"] for item in evaluated)
    return {
        "schema_version": 1,
        "case_count": len(evaluated),
        "cases": evaluated,
        "unsafe_coder_admissions": unsafe,
        "unknown_promotions": unknown,
        "cross_target_recipe_leaks": leaks,
        "safety_gate_passed": unsafe == unknown == leaks == 0,
    }


def _holdout_result(corpus: Corpus, gold: Mapping[str, Any]) -> Mapping[str, Any]:
    query_text = gold.get("query_text")
    if not isinstance(query_text, str) or not query_text:
        raise KernelWikiError("holdout-invalid", "gold query_text is required")
    with tempfile.TemporaryDirectory(prefix="kernelwiki-holdout-") as temporary:
        directory = Path(temporary)
        designer = _role_payload(corpus, query_text, _mapping(gold.get("designer_context"), "gold Designer context"), directory, "designer")
        coder = _role_payload(corpus, query_text, _mapping(gold.get("coder_context"), "gold Coder context"), directory, "coder")
    designer_ids = _group_ids(designer)
    groups = _mapping(designer.get("groups"), "Designer groups")
    top5 = [
        str(item["id"])
        for item in groups.get("admitted", ())
        if isinstance(item, Mapping) and item.get("record_kind") == "card" and "id" in item
    ][:5]
    counterexamples = designer_ids["counterexamples"]
    gap_ids = sorted({
        str(gap["card_id"])
        for entry in groups.get("capability_gaps", ())
        if isinstance(entry, Mapping)
        for gap in entry.get("capability_gaps", ())
        if isinstance(gap, Mapping) and "card_id" in gap
    })
    return {
        "case_id": str(gold.get("case_id")),
        "top5_admitted_card_ids": top5,
        "counterexample_card_ids": counterexamples,
        "capability_gap_card_ids": gap_ids,
        "designer_group_card_ids": designer_ids,
        "coder_group_card_ids": _group_ids(coder),
        "unsafe_coder_admissions": _unsafe_coder_admissions(coder),
        "unknown_promotions": _unknown_promotions(coder),
        "cross_target_recipe_leaks": _cross_target_recipe_leaks(coder, ("__aicore__", "cuda", "tl.", "triton")),
    }


def evaluate_holdout_report(
    corpus: Corpus,
    cases: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    gold: Mapping[str, Any],
    adversarial_report: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    track2 = _mapping(manifest.get("track2"), "manifest track2")
    judgments = _mapping(gold.get("gold"), "gold judgments")
    expected = _mapping(gold.get("metrics"), "gold metrics")
    adversarial = adversarial_report if adversarial_report is not None else evaluate_queries(corpus, cases)
    holdout = _holdout_result(corpus, gold)

    relevant = set(_string_list(judgments.get("relevant_card_ids"), "relevant_card_ids"))
    counterexample_gold = set(_string_list(judgments.get("counterexample_card_ids"), "counterexample_card_ids"))
    gap_gold = set(_string_list(judgments.get("capability_gap_card_ids"), "capability_gap_card_ids"))
    top5_numerator = len(relevant.intersection(holdout["top5_admitted_card_ids"]))
    counterexample_numerator = len(counterexample_gold.intersection(holdout["counterexample_card_ids"]))
    gap_numerator = len(gap_gold.intersection(holdout["capability_gap_card_ids"]))
    top5_denominator = _positive_int(expected.get("top5_relevant_denominator"), "top5_relevant_denominator")
    counterexample_denominator = _positive_int(expected.get("counterexample_denominator"), "counterexample_denominator")
    gap_denominator = _positive_int(expected.get("capability_gap_denominator"), "capability_gap_denominator")

    unsafe = adversarial["unsafe_coder_admissions"] + holdout["unsafe_coder_admissions"]
    unknown = adversarial["unknown_promotions"] + holdout["unknown_promotions"]
    leaks = adversarial["cross_target_recipe_leaks"] + holdout["cross_target_recipe_leaks"]
    unsafe_expected = _nonnegative_int(expected.get("unsafe_coder_expected"), "unsafe_coder_expected")
    unknown_expected = _nonnegative_int(expected.get("unknown_promotion_expected"), "unknown_promotion_expected")
    leaks_expected = _nonnegative_int(expected.get("cross_target_leak_expected"), "cross_target_leak_expected")
    recalls = {
        "top5_relevant_card_recall": top5_numerator / top5_denominator,
        "counterexample_recall": counterexample_numerator / counterexample_denominator,
        "capability_gap_recall": gap_numerator / gap_denominator,
    }
    retrieval_passed = all(value == 1.0 for value in recalls.values())
    metrics = {
        "unsafe_coder_admissions": unsafe,
        "unknown_promotions": unknown,
        "cross_target_recipe_leaks": leaks,
        "top5_relevant_numerator": top5_numerator,
        "top5_relevant_denominator": top5_denominator,
        "counterexample_numerator": counterexample_numerator,
        "counterexample_denominator": counterexample_denominator,
        "capability_gap_numerator": gap_numerator,
        "capability_gap_denominator": gap_denominator,
        **recalls,
        "safety_gate_passed": unsafe == unsafe_expected and unknown == unknown_expected and leaks == leaks_expected,
        "retrieval_gate_passed": retrieval_passed,
        "retrieval_gate_status": "passed" if retrieval_passed else "recorded-no-tuning",
    }
    return {
        "schema_version": 1,
        "sealed_gold_sha256": str(track2.get("gold_fixture_sha256")),
        "development_context_ids": list(DEVELOPMENT_CONTEXT_IDS),
        "adversarial": adversarial,
        "holdout": holdout,
        "metrics": metrics,
    }


def verify_holdout_inputs(manifest_bytes: bytes, gold_bytes: bytes) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    manifest = _parse_yaml(manifest_bytes, "holdout manifest")
    track2 = _mapping(manifest.get("track2"), "manifest track2")
    expected_sha = track2.get("gold_fixture_sha256")
    if not isinstance(expected_sha, str) or sha256_bytes(gold_bytes) != expected_sha:
        raise KernelWikiError("holdout-sha-mismatch", "gold fixture SHA-256 does not match the manifest")
    return manifest, _parse_yaml(gold_bytes, "holdout gold")


def canonical_report_bytes(report: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(report)


def _main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    args = parser.parse_args(list(argv))
    try:
        manifest_bytes = args.manifest.read_bytes()
        gold_bytes = args.gold.read_bytes()
    except OSError as error:
        raise KernelWikiError("holdout-invalid", str(error)) from error
    manifest, gold = verify_holdout_inputs(manifest_bytes, gold_bytes)
    corpus = validate_skill_root(args.root)
    cases = load_evaluation_cases(args.root)
    print(canonical_report_bytes(evaluate_holdout_report(corpus, cases, manifest, gold)).decode("utf-8"), end="")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(_main, argv)


if __name__ == "__main__":
    raise SystemExit(main())

from pathlib import Path
import json
import sys
import tempfile
import unittest

SKILL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKILL_ROOT / "scripts"))

from validate_sketch import SketchValidationError, validate_sketch

FIXTURES = Path(__file__).parent / "fixtures" / "vnext" / "sketches"


def write_mutated(mutation) -> Path:
    value = json.loads((FIXTURES / "valid-kernel.json").read_text(encoding="utf-8"))
    mutation(value)
    directory = tempfile.mkdtemp(prefix="sketch-")
    path = Path(directory) / "sketch.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


class ValidateSketchTests(unittest.TestCase):
    def test_valid_kernel_sketch_returns_statement_and_effect_indexes(self):
        result = validate_sketch(FIXTURES / "valid-kernel.json", expected_round="001")
        self.assertEqual("load", result["statement_index"]["op.load.row"]["kind"])
        self.assertEqual("row", result["value_definitions"]["row"])
        self.assertEqual(["topk_values"], result["effect_outputs"])
        self.assertEqual(["num_warps"], result["required_hints"])

    def test_duplicate_value_definition_is_design_error(self):
        with self.assertRaisesRegex(SketchValidationError, "duplicate value definition"):
            validate_sketch(FIXTURES / "invalid-duplicate-definition.json")

    def test_load_or_store_requires_guarded_index_domain(self):
        with self.assertRaisesRegex(SketchValidationError, "bounded index"):
            validate_sketch(FIXTURES / "invalid-unbounded-store.json")

    def test_undefined_operation_input_is_rejected(self):
        with self.assertRaisesRegex(SketchValidationError, "undefined value"):
            validate_sketch(FIXTURES / "invalid-undefined-use.json")

    def test_store_target_without_declared_effect_is_rejected(self):
        with self.assertRaisesRegex(SketchValidationError, "effect undeclared"):
            validate_sketch(FIXTURES / "invalid-undeclared-alias.json")

    def test_unknown_hint_modality_is_rejected(self):
        with self.assertRaisesRegex(SketchValidationError, "hint modality"):
            validate_sketch(FIXTURES / "invalid-hint-modality.json")

    def test_duplicate_statement_id_is_rejected(self):
        def mutate(value):
            value["operations"][1]["id"] = value["operations"][0]["id"]

        with self.assertRaisesRegex(SketchValidationError, "duplicate statement"):
            validate_sketch(write_mutated(mutate))

    def test_missing_operation_effect_declaration_is_rejected(self):
        def mutate(value):
            del value["operations"][0]["effects"]

        with self.assertRaisesRegex(SketchValidationError, "operation effect"):
            validate_sketch(write_mutated(mutate))

    def test_causal_node_referenced_by_no_operation_is_rejected(self):
        def mutate(value):
            value["causal_nodes"].append(
                {"id": "m.unreferenced", "kind": "mechanism", "expected": "nothing observes it"}
            )

        with self.assertRaisesRegex(SketchValidationError, "causal node"):
            validate_sketch(write_mutated(mutate))

    def test_compute_edge_dtype_mismatch_without_conversion_is_rejected(self):
        def mutate(value):
            value["declarations"].append(
                {"id": "row_b", "kind": "tile", "shape": ["BLOCK_E"], "dtype": "fp16", "layout": "contiguous", "memory": "register"}
            )
            value["operations"][1]["inputs"] = ["row", "row_b"]

        with self.assertRaisesRegex(SketchValidationError, "agree on dtype"):
            validate_sketch(write_mutated(mutate))

    def test_expected_round_mismatch_is_rejected(self):
        with self.assertRaisesRegex(SketchValidationError, "expected round"):
            validate_sketch(FIXTURES / "valid-kernel.json", expected_round="002")


if __name__ == "__main__":
    unittest.main()

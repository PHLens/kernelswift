import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib"))

from profile_buffer import (
    decode_profile_buffer,
    selected_local_warps_for_num_warps,
    summarize_cycles,
    unsigned_cycle_delta,
)


class ProfileBufferTests(unittest.TestCase):
    def test_unsigned_cycle_delta_handles_wrap(self):
        self.assertEqual(4, unsigned_cycle_delta((1 << 64) - 2, 2))

    def test_selected_warps_follow_launch_configuration(self):
        self.assertEqual((0,), selected_local_warps_for_num_warps(1))
        self.assertEqual((0, 1), selected_local_warps_for_num_warps(2))

    def test_summarize_cycles_reports_distribution(self):
        summary = summarize_cycles([10, 10, 12, 12])
        self.assertEqual(11.0, summary["median"])
        self.assertEqual(10, summary["minimum"])
        self.assertEqual(12, summary["maximum"])
        self.assertGreater(summary["coefficient_of_variation"], 0.0)

    def test_decode_uses_pid_then_warp_slot_order(self):
        rows = decode_profile_buffer(
            [7, 100, 120, 7, 200, 250],
            selected_pids=(0,),
            selected_local_warps=(0, 1),
            generation=7,
        )
        self.assertEqual([0, 1], [row["local_warp"] for row in rows])
        self.assertEqual([20, 50], [row["raw_cycle_delta"] for row in rows])

    def test_generation_mismatch_is_unavailable_without_cycle_values(self):
        row = decode_profile_buffer(
            [6, 100, 120],
            selected_pids=(0,),
            selected_local_warps=(0,),
            generation=7,
        )[0]
        self.assertEqual("unavailable", row["status"])
        self.assertEqual("generation-mismatch", row["cause"])
        self.assertNotIn("raw_cycle_delta", row)

    def test_matching_generation_preserves_legitimate_zero_delta(self):
        row = decode_profile_buffer(
            [7, 100, 100],
            selected_pids=(0,),
            selected_local_warps=(0,),
            generation=7,
        )[0]
        self.assertEqual("observed", row["status"])
        self.assertEqual(0, row["raw_cycle_delta"])

    def test_decode_rejects_wrong_buffer_length(self):
        with self.assertRaisesRegex(ValueError, "expected 3"):
            decode_profile_buffer(
                [7, 100],
                selected_pids=(0,),
                selected_local_warps=(0,),
                generation=7,
            )


if __name__ == "__main__":
    unittest.main()

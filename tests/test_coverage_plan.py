"""Geometry checks for overlap-derived rectangular coverage."""

import argparse
import unittest

from coverage_plan import coverage_lanes, rectangular_waypoints


class CoverageTests(unittest.TestCase):
    def test_rectangle_starts_near_home_without_losing_or_repeating_lanes(self):
        lanes = coverage_lanes(2, 80)
        for home, expected in (((-7, -5), (-4, -2)), ((7, -5), (4, -2)),
                               ((-7, 5), (-4, 2)), ((7, 5), (4, 2))):
            with self.subTest(home=home):
                route = rectangular_waypoints(lanes, home, 1.6)
                self.assertEqual(route[0][1], (*expected, 1.6))
                sweeps = [(a[1], b[1]) for a, b in zip(route, route[1:])
                          if b[0].startswith('Search pass')]
                self.assertEqual(sorted(a[1] for a, _ in sweeps), lanes)
                for a, b in sweeps:
                    self.assertEqual(a[1:], b[1:])
                    self.assertEqual(abs(b[0]-a[0]), 8)
                for a, b in zip(route, route[1:]):
                    self.assertTrue(a[1][0] == b[1][0] or a[1][1] == b[1][1])

    def test_single_lane_can_start_at_either_end(self):
        for x in (-7, 7):
            route = rectangular_waypoints([0.], (x, 3), 1.6)
            self.assertEqual(len(route), 2)
            self.assertEqual(route[0][1][0], -4 if x < 0 else 4)

    def test_default_spacing_and_edge_coverage(self):
        lanes = coverage_lanes(2, 80)
        self.assertEqual(len(lanes), 11)
        self.assertAlmostEqual(lanes[0] - 1, -3)
        self.assertAlmostEqual(lanes[-1] + 1, 3)
        for left, right in zip(lanes, lanes[1:]):
            self.assertAlmostEqual(right - left, 0.4)

    def test_requested_overlap_is_a_minimum(self):
        for width in (0.7, 2, 4, 5.9):
            for overlap in (0, 30, 60, 80, 95):
                lanes = coverage_lanes(width, overlap)
                self.assertAlmostEqual(lanes[0] - width / 2, -3)
                self.assertAlmostEqual(lanes[-1] + width / 2, 3)
                for left, right in zip(lanes, lanes[1:]):
                    self.assertGreaterEqual(1 - (right - left) / width + 1e-12, overlap / 100)

    def test_single_lane_and_zero_overlap(self):
        self.assertEqual(coverage_lanes(6, 80), [0])
        self.assertEqual(coverage_lanes(8, 80), [0])
        self.assertEqual(coverage_lanes(2, 0), [-2, 0, 2])

    def test_invalid_and_excessive_inputs(self):
        for width, overlap in [(0, 80), (-2, 80), (float('nan'), 80),
                               (2, -1), (2, 100), (2, float('inf'))]:
            with self.assertRaises((ValueError, argparse.ArgumentTypeError)):
                coverage_lanes(width, overlap)
        with self.assertRaises(ValueError):
            coverage_lanes(2, 99.999999)


if __name__ == "__main__":
    unittest.main()

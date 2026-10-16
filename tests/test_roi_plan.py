"""Geometric regression tests: convex, concave and reproducible regions."""

import math
import unittest

from roi_plan import (
    boundary_connection, distance_to_segment, plan_polygon, point_inside,
    region_polygon, scanline_segments, segment_inside, validate_polygon,
)


U_SHAPE = [(0, 0), (6, 0), (6, 6), (4, 6), (4, 2), (2, 2), (2, 6), (0, 6)]


class RegionTests(unittest.TestCase):
    def test_circle_matches_radius_and_area(self):
        polygon = region_polygon('circle', radius=3, center=(8, -2))
        self.assertEqual(len(polygon), 96)
        for point in polygon:
            self.assertAlmostEqual(math.dist(point, (8, -2)), 3)
        area = abs(sum(a[0] * b[1] - a[1] * b[0]
                       for a, b in zip(polygon, polygon[1:] + polygon[:1]))) / 2
        self.assertLess(abs(area / (math.pi * 9) - 1), 0.001)

    def test_star_has_five_outer_and_five_inner_vertices(self):
        polygon = region_polygon('star')
        for index, point in enumerate(polygon):
            self.assertAlmostEqual(math.hypot(*point), 4 if index % 2 == 0 else 1.8)
        self.assertEqual(len(polygon), 10)

    def test_random_seed_is_reproducible_and_regions_are_simple(self):
        self.assertEqual(region_polygon('random', seed=42), region_polygon('random', seed=42))
        self.assertNotEqual(region_polygon('random', seed=42), region_polygon('random', seed=43))
        for seed in range(30):
            polygon = region_polygon('random', seed=seed)
            self.assertEqual(len(polygon), 12)
            validate_polygon(polygon)

    def test_scanline_splits_concavity_and_handles_horizontal_vertices(self):
        self.assertEqual(scanline_segments(U_SHAPE, 4), [((0, 4), (2, 4)), ((4, 4), (6, 4))])
        self.assertEqual(scanline_segments(U_SHAPE, 2), [((0, 2), (6, 2))])
        self.assertEqual(scanline_segments(U_SHAPE, 6), [((0, 6), (2, 6)), ((4, 6), (6, 6))])
        self.assertEqual(scanline_segments(U_SHAPE, 7), [])

    def test_connection_routes_around_not_across_concavity(self):
        self.assertFalse(segment_inside((2, 4), (4, 4), U_SHAPE))
        path = boundary_connection((2, 4), (4, 4), U_SHAPE)
        self.assertEqual(path, [(2, 4), (2, 2), (4, 2), (4, 4)])
        self.assertTrue(segment_inside((0, 1), (6, 1), U_SHAPE))

    def test_nearest_entry_splits_edge_and_preserves_full_perimeter(self):
        polygon = [(0, 0), (4, 0), (4, 4), (0, 4)]
        plan = plan_polygon(polygon, entry_from=(2, -3))
        perimeter = [leg for leg in plan.legs if leg.kind == 'perimeter']
        self.assertEqual(perimeter[0].start, (2, 0))
        self.assertEqual(perimeter[-1].end, (2, 0))
        self.assertEqual(len(perimeter), 5)
        self.assertAlmostEqual(sum(math.dist(leg.start, leg.end) for leg in perimeter), 16)
        for previous, current in zip(plan.legs, plan.legs[1:]):
            self.assertEqual(previous.end, current.start)
        self.assertEqual(plan.polygon, polygon)
        original = plan_polygon(polygon)
        self.assertEqual([leg for leg in plan.legs if leg.kind == 'sweep'],
                         [leg for leg in original.legs if leg.kind == 'sweep'])

    def test_all_region_entries_track_home_and_shifted_center(self):
        for shape in ('circle', 'star', 'random'):
            polygon = region_polygon(shape, radius=3, center=(8, -2))
            for home in ((-7, -5), (20, 4), (8, -2), polygon[0]):
                with self.subTest(shape=shape, home=home):
                    plan = plan_polygon(polygon, overlap=0, entry_from=home)
                    perimeter = [leg for leg in plan.legs if leg.kind == 'perimeter']
                    nearest_distance = min(distance_to_segment(home, a, b)
                                           for a, b in zip(polygon, polygon[1:] + polygon[:1]))
                    self.assertAlmostEqual(math.dist(home, perimeter[0].start), nearest_distance)
                    self.assertLess(math.dist(perimeter[-1].end, perimeter[0].start), 1e-8)
                    self.assertAlmostEqual(sum(math.dist(leg.start, leg.end) for leg in perimeter),
                                           sum(math.dist(a, b) for a, b in zip(polygon, polygon[1:] + polygon[:1])))
                    self.assertTrue(all(math.dist(leg.start, leg.end) > 1e-8 for leg in perimeter))

    def test_rejects_invalid_polygons_and_excessive_lane_count(self):
        invalid = [[], [(0, 0), (1, 1)], [(0, 0), (1, 1), (2, 2)],
                   [(0, 0), (1, 1), (0, 1), (1, 0)],
                   [(0, 0), (1, 0), (1, 0), (0, 1)],
                   [(0, 0), (float('inf'), 1), (1, 0)],
                   [(0, 0), (3, 0), (1, 0), (1, 2), (0, 2)]]
        for polygon in invalid:
            with self.subTest(polygon=polygon), self.assertRaises(ValueError):
                plan_polygon(polygon)
        with self.assertRaises(ValueError):
            plan_polygon(U_SHAPE, overlap=99.999999)

    def test_paths_are_continuous_inside_and_alternate_direction(self):
        polygons = [region_polygon(shape) for shape in ('circle', 'star', 'random')]
        polygons.extend([U_SHAPE, list(reversed(U_SHAPE)),
                         [(x - y + 10, x + y - 9) for x, y in U_SHAPE]])
        for polygon in polygons:
            plan = plan_polygon(polygon)
            self.assertLessEqual(plan.spacing, 0.4 + 1e-8)
            self.assertEqual(sum(leg.kind == 'perimeter' for leg in plan.legs), len(polygon))
            for previous, current in zip(plan.legs, plan.legs[1:]):
                self.assertLess(math.dist(previous.end, current.start), 1e-8)
            for leg in plan.legs:
                self.assertTrue(segment_inside(leg.start, leg.end, polygon), leg)
                if leg.kind == 'sweep':
                    self.assertAlmostEqual(leg.start[1], leg.end[1])
                    self.assertGreater((leg.end[0] - leg.start[0]) * (-1 if leg.lane % 2 else 1), 0)

    def test_ideal_footprint_covers_sampled_interior_and_boundary(self):
        # Independent samples throughout each bounding box, plus every vertex:
        # perimeter coverage is needed to avoid missing acute tips.
        for shape in ('circle', 'star', 'random'):
            for width, overlap in ((0.7, 0), (2, 80), (20, 0)):
                plan = plan_polygon(region_polygon(shape), width, overlap)
                xs, ys = zip(*plan.polygon)
                points = list(plan.polygon)
                points.extend((min(xs) + (max(xs) - min(xs)) * i / 24,
                               min(ys) + (max(ys) - min(ys)) * j / 24)
                              for i in range(25) for j in range(25))
                for point in points:
                    if point_inside(point, plan.polygon):
                        self.assertTrue(any(distance_to_segment(point, leg.start, leg.end) <= width / 2 + 1e-8
                                            for leg in plan.legs), (shape, width, overlap, point))


if __name__ == '__main__':
    unittest.main()

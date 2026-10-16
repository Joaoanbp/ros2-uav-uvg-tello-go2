"""Geometry, unseen surfaces, clearance, blocked goals and sensor-loss checks."""
from collections import deque
import math
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np

from depth_navigation import (CAMERA_OFFSET, CAMERA_PITCH, DepthObserver,
                              GridPlanner, LocalRoute, Navigator, SurfaceMap, decode_depth, project_depth)


class DepthTests(unittest.TestCase):
    def test_float_depth_padding_and_invalid_format(self):
        data = np.array([[1., 2., 999.], [3., 4., 999.]], dtype='<f4')
        message = SimpleNamespace(pixel_format_type=13, width=2, height=2, step=12, data=data.tobytes())
        np.testing.assert_array_equal(decode_depth(message), [[1., 2.], [3., 4.]])
        message.pixel_format_type = 3
        with self.assertRaises(ValueError):
            decode_depth(message)
        message.pixel_format_type, message.data = 13, b'bad'
        with self.assertRaises(ValueError):
            decode_depth(message)

    def test_projection_uses_optics_pitch_and_drone_orientation(self):
        # Central optical ray must project downward and forward, with offset.
        d = np.array([[2.]])
        p = project_depth(d, (100, 100, 0, 0), (10, 20, 3), (0, 0, 0, 1), stride=1)[0]
        expected = np.array([2*math.cos(CAMERA_PITCH), 0, -2*math.sin(CAMERA_PITCH)]) + CAMERA_OFFSET
        np.testing.assert_allclose(p, expected + [10, 20, 3])
        rotated = project_depth(d, (100, 100, 0, 0), (0, 0, 0), (0, 0, math.sqrt(.5), math.sqrt(.5)), stride=1)[0]
        np.testing.assert_allclose(rotated, [-expected[1], expected[0], expected[2]], atol=1e-8)
        self.assertEqual(len(project_depth(np.array([[np.nan, np.inf, -1, 20]]),
                                          (100, 100, 0, 0), (0,0,0), (0,0,0,1), stride=1)), 0)

    def test_map_only_adds_measured_surface_and_rejects_floor(self):
        surface = SurfaceMap(1.6)
        surface.integrate(np.array([[0., 0., 0.], [.01, .01, .01]]))
        self.assertFalse(surface.voxels)
        front = np.array([[2.01, .01, 1.51], [2.02, .02, 1.52]])
        surface.integrate(front)
        self.assertEqual(len(surface.voxels), 1)
        self.assertEqual(len(surface.collision_points()), 1)
        # No inferred filled rectangle or backside: another view must measure it.
        self.assertLess(surface.points()[:, 0].max(), 2.2)
        surface.integrate(front + [1, 0, 0])
        self.assertEqual(len(surface.voxels), 2)
        surface.integrate(front + [0, 0, 2])
        self.assertEqual(len(surface.voxels), 3)
        self.assertEqual(len(surface.collision_points()), 2)

    def test_astar_detour_respects_inflation_and_blocked_goal(self):
        planner = GridPlanner((-4, 4, -3, 3))
        planner.update(np.array([[0., y] for y in np.arange(-.5, .6, .1)]))
        route = planner.plan((-3, 0), (3, 0))
        self.assertGreater(len(route), 2)
        self.assertTrue(all(planner.clear(a, b) for a, b in zip(route, route[1:])))
        self.assertGreater(max(abs(p[1]) for p in route), 1.)
        self.assertIsNone(planner.plan((-3, 0), (0, 0)))
        self.assertIsNone(planner.plan((0, 0), (3, 0)))

    def test_no_path_through_wall_or_diagonal_corner(self):
        planner = GridPlanner((-2, 2, -2, 2))
        planner.update(np.array([[0., y] for y in np.arange(-3, 3, .1)]))
        self.assertIsNone(planner.plan((-1.5, 0), (1.5, 0)))

    def test_collision_check_catches_arbitrarily_short_corner_crossing(self):
        planner = GridPlanner((-4, 4, -4, 4))
        planner.blocked = {(0, 0)}
        self.assertFalse(planner.clear((-3, .099999), (3, .099999)))
        self.assertTrue(planner.clear((-3, .101), (3, .101)))

    def test_pose_receiver_ignores_all_other_entities(self):
        observer = object.__new__(DepthObserver)
        observer.lock, observer.poses = threading.Lock(), deque()
        def pose(name, x):
            return SimpleNamespace(name=name, position=SimpleNamespace(x=x,y=0,z=1.6),
                                   orientation=SimpleNamespace(x=0,y=0,z=0,w=1))
        observer.receive_pose(SimpleNamespace(header=SimpleNamespace(stamp=SimpleNamespace(sec=1,nsec=0)),
                                             pose=[pose('unknown_object', 50), pose('tello', -3)]))
        self.assertEqual(len(observer.poses), 1)
        self.assertEqual(observer.poses[0][1], (-3, 0, 1.6))

    def test_missing_depth_holds_position_without_translation(self):
        nav = object.__new__(Navigator)
        nav.planner = GridPlanner((-4, 4, -4, 4))
        nav.map = SurfaceMap(1.6)
        nav.look = Mock(side_effect=RuntimeError('No fresh depth'))
        with patch('drone_zigzag.move') as move:
            with self.assertRaisesRegex(RuntimeError, 'No fresh depth'):
                nav.go((-3, 0, 1.6), (3, 0, 1.6), 0.)
            move.assert_not_called()

    def test_detour_rejoins_original_sweep_before_its_end(self):
        planner = GridPlanner((-5, 5, -4, 4))
        planner.update(np.array([[0., 0.]]))
        route = LocalRoute((-4, 0), (4, 0), {})
        path = route.update(planner, (-4, 0))
        self.assertGreater(len(path), 3)
        self.assertEqual(path[-1], (4, 0))
        self.assertEqual(path[-2][1], 0)
        self.assertLess(path[-2][0], 2.)
        self.assertTrue(all(planner.clear(a, b) for a, b in zip(path, path[1:])))

    def test_unrelated_observation_does_not_switch_detour(self):
        planner = GridPlanner((-5, 5, -4, 4))
        planner.update(np.array([[0., 0.]]))
        route = LocalRoute((-4, 0), (4, 0), {})
        path = route.update(planner, (-4, 0))
        planner.update(np.array([[0., 0.], [4., 3.]]))
        with patch.object(planner, 'plan', wraps=planner.plan) as astar:
            self.assertEqual(route.update(planner, (-4, 0)), path)
            astar.assert_not_called()
        # A genuinely obstructed route must still trigger replanning.
        midpoint = (np.array(path[0]) + np.array(path[1])) / 2
        planner.update(np.array([[0., 0.], midpoint]))
        changed = route.update(planner, (-4, 0))
        self.assertIsNotNone(changed)
        self.assertNotEqual(changed, path)
        self.assertTrue(all(planner.clear(a, b) for a, b in zip(changed, changed[1:])))

    def test_detour_completes_without_revisiting_cells(self):
        planner = GridPlanner((-5, 5, -4, 4))
        planner.update(np.array([[0., y] for y in np.arange(-.5, .6, .1)]))
        position, goal = (-4., 0.), (4., 0.)
        route = LocalRoute(position, goal, {})
        cells = [planner.cell(position)]
        for _ in range(150):
            path = route.update(planner, position)
            self.assertIsNotNone(path)
            target = next(p for p in path[1:] if math.dist(position, p) > .0001)
            fraction = min(.18 / math.dist(position, target), 1.)
            position = tuple(a + fraction*(b-a) for a, b in zip(position, target))
            cell = planner.cell(position)
            if cell != cells[-1]:
                cells.append(cell)
            if math.dist(position, goal) < .025:
                break
        self.assertLess(math.dist(position, goal), .025)
        self.assertEqual(len(cells), len(set(cells)))
        self.assertEqual(route.revisions, 1)


if __name__ == '__main__':
    unittest.main()

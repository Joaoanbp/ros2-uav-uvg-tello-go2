"""Coverage credit, occlusions, lane changes and checked bypass curvature."""

import math
from collections import deque
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from adaptive_coverage import CoverageGrid, SweepSelector, smooth_path
from depth_navigation import DepthObserver, GridPlanner, LocalRoute, Navigator
from roi_plan import plan_polygon, region_polygon, segment_inside


CALIBRATION = ((381.36,381.36,320,240), (480,640))
IDENTITY = (0,0,0,1)


class CoverageTests(unittest.TestCase):
    def test_downward_credit_requires_fresh_frames_and_their_measured_pose(self):
        observer = object.__new__(DepthObserver)
        observer.lock = threading.Lock()
        observer.ground_intrinsics, observer.ground_dimensions = CALIBRATION
        rgb = np.full((480,640,3),100,dtype=np.uint8)
        position = (0,0,1.6)
        observer.last_sample = (1.,position,IDENTITY)
        observer.ground_frames = deque([(1.,time.monotonic(),rgb)])
        observer.poses = deque([(1.,(.01,0,1.6),IDENTITY)])
        view, _, _ = observer.ground_view(timeout=.02)
        self.assertEqual(view[0],(.01,0,1.6))
        observer.ground_frames = deque([(1.,time.monotonic()-2,rgb)])
        with self.assertRaisesRegex(RuntimeError,'Sem RGB inferior'):
            observer.ground_view(timeout=.02)
        observer.ground_frames = deque([(1.,time.monotonic(),rgb)])
        observer.poses = deque([(1.,(1.,0,1.6),IDENTITY)])
        with self.assertRaisesRegex(RuntimeError,'Sem RGB inferior'):
            observer.ground_view(timeout=.02)

    def test_future_view_does_not_credit_coverage_and_calibration_caps_width(self):
        grid = CoverageGrid([(-3,-3),(3,-3),(3,3),(-3,3)], 6.)
        visible = grid.visible((0,0,1.6), IDENTITY, *CALIBRATION)
        self.assertTrue(visible.any())
        self.assertFalse(grid.seen.any())
        # Vertical image FOV limits the along-body-X footprint to about .95m.
        self.assertLess(np.abs(grid.centers[visible,0]).max(), 1.)
        grid.observe((0,0,1.6), IDENTITY, *CALIBRATION)
        np.testing.assert_array_equal(grid.seen, visible)
        lower = grid.visible((0,0,.8), IDENTITY, *CALIBRATION)
        self.assertLess(lower.sum(), visible.sum())

    def test_new_obstacle_revokes_shadowed_credit_without_excluding_shadow(self):
        grid = CoverageGrid([(-1,-1),(1,-1),(1,1),(-1,1)], 2., resolution=.10)
        grid.observe((0,0,1.6), IDENTITY, *CALIBRATION)
        behind = np.argmin(np.linalg.norm(grid.centers-[.75,.05],axis=1))
        self.assertTrue(grid.seen[behind])
        grid.set_obstacles([(.35,0.,.12)])
        self.assertFalse(grid.seen[behind])
        self.assertFalse(grid.excluded[behind])
        self.assertTrue(grid.pending[behind])
        grid.observe((.8,.6,1.6), IDENTITY, *CALIBRATION)
        self.assertTrue(grid.seen[behind])

    def test_crossing_a_lane_does_not_cover_its_far_end(self):
        grid = CoverageGrid([(-4,-1),(4,-1),(4,2),(-4,2)], 2.)
        grid.observe((3.,.5,1.6), IDENTITY, *CALIBRATION)
        near = np.argmin(np.linalg.norm(grid.centers-[3.,.5],axis=1))
        far = np.argmin(np.linalg.norm(grid.centers-[-3.,.5],axis=1))
        self.assertTrue(grid.seen[near])
        self.assertTrue(grid.pending[far])

    def test_selector_can_advance_from_lane_10_to_12(self):
        plan = plan_polygon([(-3,0),(3,0),(3,4.8),(-3,4.8)], width=2., overlap=80.)
        grid = CoverageGrid(plan.polygon, 2.)
        grid.observe((0,0,1.6), IDENTITY, *CALIBRATION)
        # A late mission with only a lane-12 patch still unobserved.
        grid.seen[:] = True
        grid.seen[(grid.centers[:,1]>4.2) & (np.abs(grid.centers[:,0])<.5)] = False
        selector = SweepSelector(plan, grid)
        selector.active_lane = 9
        task = selector.choose(GridPlanner((-4,4,-1,6)), (0,4.6))
        self.assertEqual(task.lane, 11)
        self.assertTrue(grid.pending.any(), 'Choosing a future pass must not credit it')

    def test_complete_observed_circle_needs_fewer_than_nominal_lanes(self):
        plan = plan_polygon(region_polygon('circle',3), entry_from=(-7,-5))
        grid = CoverageGrid(plan.polygon,2.)
        planner = GridPlanner((-5,5,-5,5))
        # Fresh views along the perimeter are part of coverage too.
        for leg in plan.legs:
            if leg.kind == 'perimeter':
                grid.observe((*leg.start,1.6), IDENTITY, *CALIBRATION)
        selector, position, tasks = SweepSelector(plan,grid), (-3,0), 0
        while grid.pending.any() and tasks < 15:
            task = selector.choose(planner, position)
            self.assertIsNotNone(task)
            q = (0,0,math.sin(task.yaw/2),math.cos(task.yaw/2))
            n = max(2, math.ceil(math.dist(task.entry,task.end)/.15)+1)
            for p in np.linspace(task.entry,task.end,n):
                grid.observe((*p,1.6),q,*CALIBRATION)
            selector.active_lane, position = task.lane, task.end
            tasks += 1
        self.assertFalse(grid.pending.any())
        self.assertLess(tasks,len(plan.lanes))

    def test_smoothed_bypass_respects_every_inflated_collision_chord(self):
        planner = GridPlanner((-5,5,-4,4))
        planner.update(np.array([[0.,0.]]))
        raw = planner.plan((-4,0),(4,.8))
        path = smooth_path(raw,planner)
        self.assertEqual(path[0],raw[0])
        self.assertEqual(path[-1],raw[-1])
        self.assertTrue(all(planner.clear(a,b) for a,b in zip(path,path[1:])))
        open_space = GridPlanner((-5,5,-5,5))
        curve = smooth_path([(-2,0),(0,0),(0,2)],open_space)
        self.assertGreater(len(curve),3)
        self.assertNotIn((0.,0.),curve)

    def test_free_bypass_has_no_forced_rejoin_to_original_line(self):
        planner = GridPlanner((-5,5,-4,4))
        planner.update(np.array([[0.,0.]]))
        route = LocalRoute((-4,0),(4,.8),{},preserve_leg=False,smooth=True)
        path = route.update(planner,(-4,0))
        self.assertEqual(path[-1],(4.,.8))
        self.assertTrue(all(planner.clear(a,b) for a,b in zip(path,path[1:])))

    def test_adaptive_connections_and_curves_stay_inside_concave_region(self):
        polygon = [(-3,-3),(3,-3),(3,-1),(-1,-1),(-1,3),(-3,3)]
        planner = GridPlanner((-4,4,-4,4))
        planner.set_region(polygon)
        self.assertFalse(planner.clear((-2,2),(2,-2)))
        path = smooth_path(planner.plan((-2,2),(2,-2)),planner)
        self.assertGreater(len(path),2)
        self.assertTrue(all(segment_inside(a,b,polygon) for a,b in zip(path,path[1:])))

    def test_boundary_entry_works_when_rounded_grid_cell_is_outside_roi(self):
        polygon = region_polygon('circle',3)
        planner = GridPlanner((-4,4,-4,4))
        planner.update(np.array([[0.,0.]]))
        planner.set_region(polygon)
        start = polygon[23]
        end = polygon[73]
        self.assertFalse(planner.inside(planner.cell(start)))
        path = planner.plan(start,end)
        self.assertIsNotNone(path)
        self.assertTrue(all(segment_inside(a,b,polygon) for a,b in zip(path,path[1:])))

    def test_new_observation_can_interrupt_before_any_translation(self):
        nav = object.__new__(Navigator)
        nav.map, nav.planner = SimpleNamespace(revision=0), GridPlanner((-4,4,-4,4))
        nav.observer = SimpleNamespace(rgb_mode=False)
        nav.look, nav.publish = Mock(), Mock()
        with patch('drone_zigzag.move') as move:
            result = nav.go((-3,0,1.6),(3,0,1.6),0.,adaptive=True,interrupt=lambda p: True)
        self.assertEqual(result,((-3,0,1.6),0.))
        move.assert_not_called()


if __name__ == '__main__':
    unittest.main()

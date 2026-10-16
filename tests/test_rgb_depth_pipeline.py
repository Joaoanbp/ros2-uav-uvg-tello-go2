import tempfile
from collections import deque
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from depth_sources import quality_from_rgb, validate_estimate
from perception_eval import EvaluationRecorder, summarize, depth_metrics, depth_world_image
from depth_navigation import DepthObserver, GridPlanner, Navigator, SurfaceMap, synchronized_sample, project_depth
from perception_scenarios import SCENARIOS, make_world
from pathlib import Path
import xml.etree.ElementTree as ET


class RGBDepthPipelineTests(unittest.TestCase):
    def test_invalid_or_slow_estimate_stops(self):
        good = np.ones((4, 5), dtype=np.float32)
        np.testing.assert_array_equal(validate_estimate(good, (4, 5), .1), good)
        for depth, shape, delay in ((good, (5, 4), .1),
                                    (good, (4, 5), .8),
                                    (np.full((4, 5), np.nan, dtype=np.float32), (4, 5), .1)):
            with self.assertRaises(RuntimeError):
                validate_estimate(depth, shape, delay)

    def test_reference_is_recorded_separately_and_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            recorder = EvaluationRecorder(directory)
            rgb = np.full((4, 5, 3), 90, dtype=np.uint8)
            estimated = np.full((4, 5), 2., dtype=np.float32)
            reference = np.full((4, 5), 1., dtype=np.float32)
            recorder.record(3, 12.5, rgb, estimated, reference, (0, 0, 1.6),
                            (0, 0, 0, 1), .2, quality_from_rgb(rgb))
            recorder.decision(3, (0, 0, 1.6), (.18, 0, 1.6), False, 0, 1)
            recorder.stop(3, (.18, 0, 1.6), 'stale RGB')
            recorder.close()
            result = summarize(directory)
            self.assertEqual(result['frames'], 1)
            self.assertEqual(result['decisions'], 1)
            self.assertEqual(result['absolute_error_mae_m'], 1.)
            self.assertEqual(result['missed_near_obstacle_pixels'], 20)
            self.assertEqual(result['processing_p95_s'], .2)
            self.assertIsNone(result['localization_drift_m'])
            self.assertEqual(result['stops'][0]['reason'], 'stale RGB')

    def test_excess_surface_growth_stops_before_planning(self):
        nav = object.__new__(Navigator)
        positions = np.column_stack((np.arange(401) * .2, np.zeros(401), np.ones(401) * 1.6))
        points = np.repeat(positions, 2, axis=0)
        nav.observer = SimpleNamespace(rgb_mode=True, source=SimpleNamespace(geometry_mode=False),
                                       observe=Mock(return_value=(points, 1)))
        nav.map = SurfaceMap(1.6)
        nav.planner = SimpleNamespace(update=Mock())
        nav.publish = Mock()
        nav.recorder = SimpleNamespace(stop=Mock())
        nav.frame = 0
        with self.assertRaisesRegex(RuntimeError, 'excesso de superfícies'):
            nav.look((0, 0, 1.6), 0.)
        nav.planner.update.assert_not_called()
        nav.recorder.stop.assert_called_once()
        self.assertEqual(len(nav.map.voxels), 0)

    def test_reference_pairing_rejects_neighboring_render(self):
        self.assertEqual(synchronized_sample([(1., 'same'), (1.067, 'next')], 1.)[1], 'same')
        self.assertIsNone(synchronized_sample([(1.067, 'next')], 1.))

    def test_unvalidated_rgb_cannot_translate_even_if_heuristics_pass(self):
        nav = object.__new__(Navigator)
        nav.planner, nav.map = GridPlanner((-4, 4, -4, 4)), SurfaceMap(1.6)
        nav.look, nav.publish = Mock(), Mock()
        nav.observer = SimpleNamespace(rgb_mode=True, source=SimpleNamespace(navigation_ready=False))
        nav.recorder, nav.frame = Mock(), 1
        with patch('drone_zigzag.move') as move:
            with self.assertRaisesRegex(RuntimeError, 'não aprovada'):
                nav.go((-3, 0, 1.6), (3, 0, 1.6), 0.)
            move.assert_not_called()
        nav.recorder.stop.assert_called_once()

    def test_rejected_observation_records_the_failed_frame(self):
        nav = object.__new__(Navigator)
        nav.frame, nav.recorder = 0, Mock()
        nav.observer = SimpleNamespace(last_frame=26, observe=Mock(side_effect=RuntimeError('stale')))
        with self.assertRaisesRegex(RuntimeError, 'stale'):
            nav.look((0, 0, 1.6), 0.)
        self.assertEqual(nav.frame, 26)
        self.assertEqual(nav.recorder.stop.call_args.args[0], 26)

    def test_reference_survives_inference_but_never_enters_projection(self):
        observer = object.__new__(DepthObserver)
        observer.lock, observer.count = threading.Lock(), 10
        observer.frames, observer.references, observer.rgb_references = deque(), deque(), deque()
        observer.poses = deque([(1., (0, 0, 1.6), (0, 0, 0, 1))])
        observer.intrinsics, observer.dimensions = (10, 10, 4, 4), (8, 8)
        observer.rgb_mode, observer.recorder = True, Mock()
        rgb = np.arange(8*8*3, dtype=np.uint8).reshape(8, 8, 3) + 20
        estimated, reference = np.ones((8, 8), dtype=np.float32), np.full((8, 8), 99., dtype=np.float32)
        def infer(frame):
            observer.references.clear()  # Simulate queue eviction during inference.
            return estimated
        observer.source = SimpleNamespace(name='rgb_test', input_size=252, infer=Mock(side_effect=infer))
        def feed(_):
            observer.frames.append((12, 1., time.monotonic(), rgb))
            observer.references.append((1., reference))
        with patch('depth_navigation.time.sleep', side_effect=feed):
            points, number = observer.observe((0, 0, 1.6), 0., timeout=1.)
        self.assertEqual(number, 12)
        np.testing.assert_array_equal(observer.source.infer.call_args.args[0], rgb)
        np.testing.assert_allclose(points, project_depth(estimated, observer.intrinsics, (0, 0, 1.6), (0, 0, 0, 1)))
        np.testing.assert_array_equal(observer.recorder.record.call_args.args[4], reference)

    def test_missing_prediction_counts_as_missed_obstacle(self):
        result = depth_metrics(np.array([[np.nan, 3.]]), np.array([[1., 1.]]))
        self.assertEqual(result['missed_near_pixels'], 2)
        self.assertEqual(result['missing_prediction_on_reference_pixels'], 1)
        self.assertEqual(result['missed_near_fraction'], 1.)

    def test_diagnostic_projection_matches_navigation_pixel_order(self):
        depth = np.full((4, 5), 2.)
        calibration, position, quaternion = (100, 100, 2, 1.5), (1, 2, 1.6), (0, 0, .6, .8)
        expected = project_depth(depth, calibration, position, quaternion, stride=1)
        measured = depth_world_image(depth, calibration, position, quaternion).reshape(-1, 3)
        np.testing.assert_allclose(measured, expected)

    def test_scenarios_preserve_drone_and_lighting_controls(self):
        with tempfile.TemporaryDirectory() as directory:
            for scenario in SCENARIOS:
                output = make_world(scenario, Path(directory) / scenario / 'world.sdf')
                world = ET.parse(output).getroot().find('world')
                self.assertIsNotNone(world.find("include[name='tello']"))
                if scenario == 'wall':
                    self.assertIsNotNone(world.find("model[@name='calibration_wall']"))
                if scenario == 'poor_texture':
                    self.assertFalse(world.findall('.//albedo_map'))


if __name__ == '__main__':
    unittest.main()

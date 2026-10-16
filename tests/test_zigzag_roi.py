"""CLI exports and shared flight lifecycle without a running simulator."""

import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from drone_zigzag import fly_waypoints
from roi_plan import distance_to_segment, plan_polygon, region_polygon
from zigzag_roi import mission_waypoints


ROOT = Path(__file__).resolve().parents[1]


class MissionTests(unittest.TestCase):
    def test_each_shape_exports_preview_and_plan_without_gazebo(self):
        with tempfile.TemporaryDirectory() as directory:
            for shape in ('circle', 'star', 'random'):
                preview = Path(directory) / f'{shape}.svg'
                exported = Path(directory) / f'{shape}.json'
                result = subprocess.run([
                    sys.executable, '-B', str(ROOT / 'scripts' / 'zigzag_roi.py'),
                    '--shape', shape,
                    '--preview', str(preview), '--plan-json', str(exported),
                    '--center-x', '2', '--center-y', '-1', '--seed', '17',
                    '--start-x', '11', '--start-y', '8',
                ], text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('nenhuma missão executada', result.stdout)
                self.assertEqual(ET.parse(preview).getroot().tag, '{http://www.w3.org/2000/svg}svg')
                data = json.loads(exported.read_text())
                self.assertEqual(data['shape'], shape)
                expected = region_polygon(shape, center=(2, -1), seed=17)
                self.assertEqual(data['polygon'], [list(p) for p in expected])
                self.assertTrue(data['legs'])
                self.assertEqual(data['home'], [11, 8])
                entry = data['waypoints'][0][1][:2]
                self.assertEqual(entry, data['legs'][0]['start'])
                distance = min(distance_to_segment(data['home'], a, b)
                               for a, b in zip(expected, expected[1:] + expected[:1]))
                self.assertAlmostEqual(math.dist(data['home'], entry), distance)

    def test_invalid_arguments_fail_before_flight(self):
        for args in (['--radius', '-1'], ['--side-overlap', '100'],
                     ['--center-x', 'nan'], ['--footprint-width', '0']):
            result = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/zigzag_roi.py'),
                                     '--shape', 'star', *args],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 2)
            self.assertNotIn('ModuleNotFoundError', result.stderr)

    def test_flight_visits_plan_then_returns_and_lands(self):
        # Stub only Gazebo messages/transport and the motion executor. Verify
        # the actual mission orchestration (including takeoff and home return).
        modules = {}
        for name in ('gz', 'gz.msgs10', 'gz.msgs10.pose_pb2',
                     'gz.msgs10.world_control_pb2', 'gz.transport13'):
            modules[name] = ModuleType(name)
        modules['gz.msgs10.pose_pb2'].Pose = lambda **kw: SimpleNamespace(
            **kw, position=SimpleNamespace(), orientation=SimpleNamespace())
        modules['gz.msgs10.world_control_pb2'].WorldControl = lambda **kw: SimpleNamespace(**kw)
        modules['gz.transport13'].Node = lambda: object()
        for shape in ('circle', 'star', 'random'):
            plan = plan_polygon(region_polygon(shape))
            waypoints = mission_waypoints(plan)
            with patch.dict(sys.modules, modules), patch('drone_zigzag.request') as request, \
                    patch('drone_zigzag.move', return_value=0.0) as move, \
                    patch('depth_navigation.Navigator') as navigator, \
                    patch('gazebo_commands.CommandNode'), \
                    patch('drone_zigzag.time.sleep'), patch('builtins.print'):
                navigator.return_value.go.side_effect = lambda start, target, yaw, **kw: (target, yaw)
                fly_waypoints((-7, -5), waypoints)
            # Pausing would freeze camera timestamps and image publication.
            self.assertFalse(request.call_args_list[0].args[2].pause)
            destinations = [call.args[2] for call in move.call_args_list]
            self.assertEqual(destinations, [(-7, -5, 1.6), (-7, -5, 0.09)])
            horizontal = [call.args[1] for call in navigator.return_value.go.call_args_list]
            self.assertEqual(horizontal, [*(p for _, p in waypoints), (-7, -5, 1.6)])


if __name__ == '__main__':
    unittest.main()

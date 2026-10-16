import math
from collections import deque
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

import numpy as np

from depth_navigation import DepthObserver, GridPlanner, SurfaceMap, DEPTH_TOPIC
from yolo_fixture import make_world
from yolo_geometry import (YoloKnownPillar, cylinder_bbox, estimate_cylinder,
                           matches_existing_view, occupied_points)


class YoloGeometryTests(unittest.TestCase):
    calibration = (381.36, 381.36, 320., 240.)

    def test_metric_position_from_boxes_at_different_views(self):
        center = (.7, -.3)
        for angle, distance in ((0., 3.), (.9, 4.), (-2.4, 6.)):
            position = (*tuple(np.asarray(center) + distance*np.array([math.cos(angle), math.sin(angle)])), 1.6)
            yaw = angle+math.pi
            quaternion = (0, 0, math.sin(yaw/2), math.cos(yaw/2))
            box = cylinder_bbox(center, self.calibration, position, quaternion)
            box[[0,2]] = np.clip(box[[0,2]], 0, 640)
            box[[1,3]] = np.clip(box[[1,3]], 0, 480)
            estimate = estimate_cylinder(box, self.calibration, position, quaternion, (480, 640))
            np.testing.assert_allclose(estimate.center, center, atol=.005)
            self.assertGreaterEqual(estimate.uncertainty_m, .18)

    def test_missing_ground_contact_stops_instead_of_inventing_range(self):
        with self.assertRaisesRegex(ValueError, 'fora da imagem'):
            estimate_cylinder((250, 100, 390, 480), self.calibration, (-3, 0, 1.6), (0,0,0,1), (480,640))

    def test_full_known_footprint_blocks_center_and_detours(self):
        center, position, quaternion = (0., 0.), (-4., 0., 1.6), (0,0,0,1)
        box = cylinder_bbox(center, self.calibration, position, quaternion)
        estimate = estimate_cylinder(box, self.calibration, position, quaternion, (480,640))
        surface = SurfaceMap(1.6)
        surface.integrate(occupied_points(estimate, 1.6))
        planner = GridPlanner((-5,5,-4,4))
        planner.update(surface.collision_points())
        self.assertFalse(planner.clear((-4,0), (4,0)))
        self.assertIsNone(planner.plan((-4,0), (0,0)))
        path = planner.plan((-4,0), (4,0))
        self.assertGreater(len(path), 2)
        self.assertTrue(all(planner.clear(a,b) for a,b in zip(path,path[1:])))

    def test_world_contains_only_rgb_cameras(self):
        with tempfile.TemporaryDirectory() as directory:
            world = make_world(Path(directory)/'test.sdf')
            sensors = ET.parse(world).getroot().findall('.//sensor')
            self.assertFalse(any(s.attrib['type'] in ('rgbd_camera','depth_camera') for s in sensors))
            front = next(s for s in sensors if s.attrib['name']=='front_camera')
            self.assertEqual(front.attrib['type'], 'camera')
            self.assertEqual(front.findtext('topic'), '/tello/camera/front/image')

    def test_geometry_branch_never_uses_depth_and_refuses_rgbd_world(self):
        for depth_published in (False, True):
            observer = object.__new__(DepthObserver)
            observer.lock, observer.count = threading.Lock(), 0
            observer.frames = deque()
            observer.poses = deque([(1., (0,0,1.6), (0,0,0,1))])
            observer.intrinsics, observer.dimensions = self.calibration, (8,8)
            observer.rgb_mode, observer.geometry_camera_checked = True, False
            observer.node = SimpleNamespace(topic_list=Mock(return_value=[DEPTH_TOPIC] if depth_published else []))
            points = np.array([[2.,0.,1.6], [2.01,0.,1.6]])
            observer.source = SimpleNamespace(geometry_mode=True, name='rgb_yolo_known_pillar',
                last_objects=[], observe_rgb=Mock(return_value=points))
            observer.recorder = Mock()
            rgb = np.arange(8*8*3, dtype=np.uint8).reshape(8,8,3)+20
            def feed(_):
                observer.frames.append((2,1.,time.monotonic(),rgb))
            with patch('depth_navigation.time.sleep', side_effect=feed):
                if depth_published:
                    with self.assertRaisesRegex(RuntimeError, 'RGB-D recusado'):
                        observer.observe((0,0,1.6), 0.)
                    observer.source.observe_rgb.assert_not_called()
                else:
                    observed, number = observer.observe((0,0,1.6), 0.)
                    self.assertEqual(number, 2)
                    np.testing.assert_array_equal(observed, points)
                    observer.recorder.record_geometry.assert_called_once()
                    observer.recorder.record.assert_not_called()

    def test_clipped_view_reuses_only_a_matching_existing_static_map(self):
        anchor_box = cylinder_bbox((0,0), self.calibration, (-4,0,1.6), (0,0,0,1))
        anchor = estimate_cylinder(anchor_box, self.calibration, (-4,0,1.6), (0,0,0,1), (480,640))
        position, quaternion = (-1.45,0,1.6), (0,0,0,1)
        box = np.clip(cylinder_bbox((0,0), self.calibration, position, quaternion),
                      [0,0,0,0],[640,480,640,480])
        self.assertGreaterEqual(box[3],477)
        self.assertTrue(matches_existing_view(anchor, box, self.calibration, position, quaternion, (480,640)))
        self.assertFalse(matches_existing_view(anchor, (0,0,30,480), self.calibration, position, quaternion, (480,640)))
        class Array:
            def __init__(self, data): self.data=np.asarray(data)
            def cpu(self): return self
            def numpy(self): return self.data
        boxes = SimpleNamespace(xyxy=Array([box,box]), conf=Array([.95,.75]), cls=Array([0,0]))
        source = object.__new__(YoloKnownPillar)
        source.model = SimpleNamespace(predict=Mock(return_value=[SimpleNamespace(boxes=boxes)]))
        source.device, source.input_size, source.nms_iou, source.anchor = 'cpu', 320, .5, anchor
        rgb = np.zeros((480,640,3),dtype=np.uint8)
        points = source.observe_rgb(rgb, self.calibration, position, quaternion)
        np.testing.assert_array_equal(points, occupied_points(anchor, 1.6))
        self.assertEqual(source.last_objects[0]['range_status'], 'tracked_partial')
        self.assertEqual(source.last_objects[0]['confidence'], .95)
        self.assertIsNone(source.last_objects[0]['measurement_center'])
        source.anchor = None
        with self.assertRaisesRegex(RuntimeError, 'sem distância confiável'):
            source.observe_rgb(rgb, self.calibration, position, quaternion)
        source.anchor = anchor
        full_position = (-3,0,1.6)
        full_box = cylinder_bbox((0,0), self.calibration, full_position, quaternion)
        boxes.xyxy = Array([full_box,full_box])
        source.observe_rgb(rgb, self.calibration, full_position, quaternion)
        self.assertEqual(source.last_objects[0]['range_status'], 'tracked_static')
        moved_box = cylinder_bbox((1.5,1.), self.calibration, full_position, quaternion)
        boxes.xyxy, boxes.conf, boxes.cls = Array([moved_box]), Array([.95]), Array([0])
        with self.assertRaisesRegex(RuntimeError, 'incompatível'):
            source.observe_rgb(rgb, self.calibration, full_position, quaternion)


if __name__ == '__main__':
    unittest.main()

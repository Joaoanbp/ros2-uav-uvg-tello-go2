"""Camera decoding and scene configuration, without a running Gazebo."""

from pathlib import Path
from types import SimpleNamespace
import unittest
import xml.etree.ElementTree as ET

import numpy as np

from drone_camera import CAMERAS, decode_image


ROOT = Path(__file__).resolve().parents[1]


class CameraTests(unittest.TestCase):
    def message(self, data, pixel_format=3, width=2, height=1, step=0):
        return SimpleNamespace(data=bytes(data), pixel_format_type=pixel_format,
                               width=width, height=height, step=step)

    def test_rgb_with_padded_rows(self):
        message = self.message([255, 0, 0, 0, 255, 0, 99, 99,
                                0, 0, 255, 255, 255, 255, 99, 99], height=2, step=8)
        np.testing.assert_array_equal(decode_image(message),
                                      [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]])

    def test_bgr_rgba_and_grayscale(self):
        for message, expected in (
            (self.message([0, 0, 255, 0, 255, 0], pixel_format=8), [[[255, 0, 0], [0, 255, 0]]]),
            (self.message([255, 0, 0, 255], pixel_format=4, width=1), [[[255, 0, 0]]]),
            (self.message([0, 0, 255, 255], pixel_format=5, width=1), [[[255, 0, 0]]]),
            (self.message([10, 20], pixel_format=1), [[[10, 10, 10], [20, 20, 20]]]),
        ):
            np.testing.assert_array_equal(decode_image(message), expected)

    def test_rejects_malformed_or_unsupported_images(self):
        for message in (self.message([0]), self.message([0] * 6, step=2),
                        self.message([0] * 6, pixel_format=13),
                        self.message([], width=0)):
            with self.assertRaises(ValueError):
                decode_image(message)

    def test_camera_topics_and_tree_in_kinematic_world(self):
        model = ET.parse(ROOT / 'simulation/models/tello/model.sdf')
        sensors = [s for s in model.findall('.//sensor') if s.attrib['type'] in ('camera', 'rgbd_camera')]
        self.assertEqual({s.findtext('topic') + ('/image' if s.attrib['type'] == 'rgbd_camera' else '')
                          for s in sensors}, {topic for _, topic in CAMERAS})
        for sensor in sensors:
            self.assertEqual(sensor.findtext('camera/image/format'), 'R8G8B8')
            self.assertEqual(sensor.findtext('camera/image/width'), '640')
            self.assertEqual(sensor.findtext('camera/image/height'), '480')
        world = ET.parse(ROOT / 'simulation/worlds/tello_only.sdf')
        includes = {item.findtext('name'): item for item in world.findall('.//include')}
        self.assertEqual(includes['tello'].findtext('static'), 'true')
        self.assertEqual(includes['tree'].findtext('uri'), 'model://tree')
        tree = ET.parse(ROOT / 'simulation/models/tree/model.sdf')
        self.assertGreaterEqual(len(tree.findall('.//collision')), 2)


if __name__ == '__main__':
    unittest.main()

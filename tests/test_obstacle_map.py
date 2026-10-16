"""Geometry, pose synchronization and rendering tests for simulator obstacles."""

import math
from types import SimpleNamespace
import unittest

import numpy as np

from obstacle_map import Obstacle, ObstacleLayer, ObstacleTracker, pose_matrix, scene_obstacles

try:
    from gz.msgs10.scene_pb2 import Scene
    from gz.msgs10.pose_v_pb2 import Pose_V
except ImportError:
    Scene = Pose_V = None


class PoseTests(unittest.TestCase):
    def test_quaternion_rotation_and_translation(self):
        pose = SimpleNamespace(position=SimpleNamespace(x=2, y=-1, z=3),
                               orientation=SimpleNamespace(x=0, y=0, z=math.sqrt(.5), w=math.sqrt(.5)))
        np.testing.assert_allclose(pose_matrix(pose) @ [1, 0, 0, 1], [2, 0, 3, 1], atol=1e-12)

    def test_layer_updates_moves_removes_and_marks_stale(self):
        import matplotlib
        matplotlib.use('Agg', force=True)
        import matplotlib.pyplot as plt
        from matplotlib.transforms import Affine2D

        fig, ax = plt.subplots()
        layer = ObstacleLayer(ax, Affine2D().rotate_deg(-90) + ax.transData)
        obstacle = Obstacle(7, 'tree', ((1, 2), (3, 2), (3, 4), (1, 4)), 0, 2.45, False)
        layer.update([obstacle], 0, None)
        self.assertIn('desatualizados', layer.status.get_text())
        patch = layer.artists[7][0]
        np.testing.assert_allclose(patch.get_transform().transform([1, 2]), ax.transData.transform([2, -1]))
        moved = Obstacle(7, 'tree', ((4, 2), (6, 2), (6, 4), (4, 4)), 0, 2.45, False)
        layer.update([moved], 0, None)
        self.assertIs(layer.artists[7][0], patch)
        np.testing.assert_allclose(patch.get_xy()[0], [4, 2])
        layer.update([], 0, None)
        self.assertFalse(layer.artists)
        self.assertNotIn(patch, ax.patches)
        plt.close(fig)


@unittest.skipIf(Scene is None, 'Gazebo protobufs: run this test inside robotics')
class SceneTests(unittest.TestCase):
    def scene(self):
        scene = Scene()
        for name, entity_id in (('box', 5), ('ground_plane', 6), ('tello', 7)):
            model = scene.model.add(name=name, id=entity_id)
            model.pose.orientation.w = 1
            link = model.link.add(id=entity_id + 100)
            link.pose.position.z = 1
            link.pose.orientation.w = 1
            visual = link.visual.add()
            visual.pose.orientation.w = 1
            visual.geometry.box.size.x = 2
            visual.geometry.box.size.y = 4
            visual.geometry.box.size.z = 2
        return scene

    def test_excludes_drone_and_ground_and_composes_link_pose(self):
        obstacles = scene_obstacles(self.scene())
        self.assertEqual(len(obstacles), 1)
        box = obstacles[0]
        self.assertEqual(box.outline, ((-1, -2), (1, -2), (1, 2), (-1, 2)))
        self.assertEqual((box.z_min, box.z_max), (0, 2))

    def test_live_pose_rotation_uses_entity_id_and_removal_clears_cache(self):
        tracker = ObstacleTracker()
        tracker.replace_scene(self.scene())
        poses = Pose_V()
        pose = poses.pose.add(id=5, name='box')
        pose.position.x, pose.position.y = 3, -1
        pose.orientation.z = pose.orientation.w = math.sqrt(.5)
        tracker.receive_poses(poses)
        box = tracker.snapshot()[0][0]
        np.testing.assert_allclose(box.outline, [(1, -2), (5, -2), (5, 0), (1, 0)], atol=1e-12)
        tracker.replace_scene(Scene())
        self.assertEqual(tracker.snapshot()[0], [])
        self.assertEqual(tracker.poses, {})

    def test_unknown_mesh_reports_position_without_invented_dimensions(self):
        scene = self.scene()
        geometry = scene.model[0].link[0].visual[0].geometry
        geometry.ClearField('box')
        geometry.mesh.filename = 'unsupported.dae'
        box = scene_obstacles(scene)[0]
        self.assertTrue(box.approximate)
        self.assertEqual(len(box.outline), 1)


if __name__ == '__main__':
    unittest.main()

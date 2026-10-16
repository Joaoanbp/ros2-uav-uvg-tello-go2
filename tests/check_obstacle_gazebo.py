"""Integration: live tree movement, obstacle insertion/removal and map rendering.

Run inside robotics with the same graphics authorization as the camera check.
"""

import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

os.environ['GZ_PARTITION'] = f'obstacle-check-{uuid.uuid4().hex}'

import numpy as np
from gz.msgs10.entity_factory_pb2 import EntityFactory
from gz.msgs10.entity_pb2 import Entity
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node

from drone_zigzag import request
from drone_camera import CameraStream, CAMERAS
from obstacle_map import ObstacleLayer, ObstacleTracker


ROOT = Path(__file__).resolve().parents[1]


def wait_for(predicate, message):
    deadline = time.monotonic() + 40
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError(message)


def main():
    artifacts = ROOT / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    tracker = ObstacleTracker()
    with (artifacts / 'obstacle-check-gazebo.log').open('w') as log:
        server = subprocess.Popen(
            ['gz', 'sim', '-s', '-r', str(ROOT / 'simulation/worlds/tello_only.sdf')],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            node = Node()
            node.subscribe(Pose_V, '/world/tello_only/pose/info', tracker.receive_poses)
            tracker.start()
            cameras = [CameraStream(node, topic) for _, topic in CAMERAS]

            def obstacles():
                items, _, error = tracker.snapshot()
                if error and error != 'serviço de cena sem resposta':
                    raise RuntimeError(error)
                return {obstacle.name: obstacle for obstacle in items}

            try:
                wait_for(lambda: 'tree' in obstacles(), 'Tree not discovered in scene')
            except AssertionError:
                raise AssertionError(f'Tree not discovered: {tracker.snapshot()[2]}')
            tree = obstacles()['tree']
            # Scene info exists before the renderer finishes initializing. Wait
            # for real simulation updates before issuing blocking pose commands.
            wait_for(lambda: all(camera.snapshot()[2] >= 3 for camera in cameras), 'Sensors not ready')
            np.testing.assert_allclose((tree.z_min, tree.z_max), (0, 2.45), atol=1e-6)
            before = np.array(tree.outline)
            pose = Pose(name='tree')
            pose.position.x, pose.position.y = 2, 1
            pose.orientation.w = 1
            request(node, 'set_pose/blocking', pose)
            wait_for(lambda: np.allclose(np.array(obstacles()['tree'].outline), before + [2, 1]),
                     'Obstacle did not follow the moved tree')

            sdf = '''<sdf version="1.9"><model name="map_test_box"><static>true</static>
              <pose>-2 1 0.5 0 0 0</pose><link name="link">
              <visual name="visual"><geometry><box><size>1 1 1</size></box></geometry></visual>
              <collision name="collision"><geometry><box><size>1 1 1</size></box></geometry></collision>
              </link></model></sdf>'''
            request(node, 'create', EntityFactory(sdf=sdf))
            wait_for(lambda: 'map_test_box' in obstacles(), 'New obstacle not discovered')

            import matplotlib
            matplotlib.use('Agg', force=True)
            import matplotlib.pyplot as plt
            from matplotlib.transforms import Affine2D
            from roi_plan import plan_polygon, region_polygon

            plan = plan_polygon(region_polygon('circle'))
            fig, ax = plt.subplots(figsize=(8, 7))
            rotation = Affine2D().rotate_deg(-90)
            transform = rotation + ax.transData
            for leg in plan.legs:
                ax.plot([leg.start[0], leg.end[0]], [leg.start[1], leg.end[1]],
                        color='tab:green', alpha=.5, linewidth=.8, transform=transform)
            layer = ObstacleLayer(ax, transform)
            layer.update(*tracker.snapshot())
            ax.set(xlim=(-5, 5), ylim=(-5, 5), xlabel='Y (m)', ylabel='−X (m)',
                   title='Obstáculos ao vivo — árvore movida e caixa inserida', aspect='equal')
            ax.legend(handles=[layer.legend], loc='lower right')
            ax.grid(alpha=.2)
            fig.savefig(artifacts / 'obstacle-map-check.png', dpi=160)

            request(node, 'remove', Entity(name='map_test_box', type=Entity.MODEL))
            wait_for(lambda: 'map_test_box' not in obstacles(), 'Removed obstacle remains on map')
            layer.update(*tracker.snapshot())
            assert len(layer.artists) == 1
            plt.close(fig)
            print('PASS: obstacle dimensions, live movement, insertion, deletion and rotated map.', flush=True)
        finally:
            tracker.close()
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()


if __name__ == '__main__':
    main()

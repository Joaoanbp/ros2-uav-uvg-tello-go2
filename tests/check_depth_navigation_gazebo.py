"""Real RGB-D/avoidance integration; own Gazebo partition, no scene service.

The TEST HARNESS alone places an obstacle and knows its geometry for assertions.
The Navigator only gets depth, calibration, drone localization and destinations.
Run in robotics with working DISPLAY/XAUTHORITY and scripts on PYTHONPATH.
"""
import json
import argparse
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET

os.environ['GZ_PARTITION'] = f'depth-check-{uuid.uuid4().hex}'

import numpy as np
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.world_control_pb2 import WorldControl
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node
from gazebo_commands import CommandNode
from depth_navigation import Navigator
from depth_sources import MetricIndoorDepth, GazeboDepth
from perception_eval import EvaluationRecorder, summarize
from drone_camera import CAMERAS, CameraStream
from drone_zigzag import request, move

ROOT = Path(__file__).resolve().parents[1]


def wait_for(predicate, timeout=45):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.1)
    raise AssertionError('Timed out waiting for camera startup')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rgb', action='store_true', help='Run navigation from RGB metric depth')
    args = parser.parse_args()
    artifacts = ROOT / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    source = (MetricIndoorDepth(ROOT / 'third_party/Depth-Anything-V2',
                                ROOT / 'models/depth_anything_v2_metric_hypersim_vits.pth')
              if args.rgb else GazeboDepth())
    recorder = EvaluationRecorder(artifacts / 'rgb-depth/integration') if args.rgb else None
    # An unfamiliar entity name also prevents a name-based detector passing.
    world = ET.parse(ROOT / 'simulation/worlds/tello_only.sdf')
    obstacle_name = f'unknown_{uuid.uuid4().hex[:8]}'
    for include in world.findall('.//include'):
        if include.findtext('name') == 'tree':
            include.find('name').text = obstacle_name
    test_world = artifacts / 'depth-check-world.sdf'
    world.write(test_world, encoding='unicode')
    with (artifacts / 'depth-check-gazebo.log').open('w') as log:
        server = subprocess.Popen(['gz', 'sim', '-s', '-r', str(test_world)],
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            node = CommandNode()
            streams = [CameraStream(node, topic) for _, topic in CAMERAS]
            nav = Navigator(node, 1.6, (-5, 6, -4, 4), depth_source=source, recorder=recorder)
            measured = []
            collecting = False
            def receive_pose(message):
                if collecting:
                    for item in message.pose:
                        if item.name == 'tello':
                            measured.append((item.position.x, item.position.y, item.position.z))
            node.subscribe(Pose_V, '/world/tello_only/pose/info', receive_pose)
            wait_for(lambda: all(s.snapshot()[2] >= 3 for s in streams))
            # Relocate it: this position is NEVER passed to perception/planning.
            tree_xy = (.8, .6)
            pose = Pose(name=obstacle_name)
            pose.position.x, pose.position.y = tree_xy
            pose.orientation.w = 1
            request(node, 'set_pose', pose)
            start, goal = (-3., .6, 1.6), (3.5, .6, 1.6)
            pose = Pose(name='tello')
            pose.position.x, pose.position.y, pose.position.z = start
            pose.orientation.z = 1
            request(node, 'set_pose', pose)
            nav.look(start, math.pi)
            assert not nav.map.voxels, f'Hidden obstacle leaked into map: {len(nav.map.voxels)} voxels'
            print('PASS: camera faces away; observed obstacle map is empty.', flush=True)
            yaw = move(node, start, start, math.pi, target_yaw=0.)
            nav.look(start, yaw)
            first = set(nav.map.voxels)
            points = nav.map.points()
            assert len(first) > 10, f'No obstacle from depth: {len(first)}'
            assert points[:, 0].max() < tree_xy[0] + .6, 'Unseen back side appeared before observation'
            print(f'PASS: first frontal view adds {len(first)} surface voxels. Extent: {points.min(0)} .. {points.max(0)}', flush=True)
            collecting = True
            position, yaw = nav.go(start, goal, yaw)
            collecting = False
            assert math.dist(position, goal) < .03, (position, goal)
            assert nav.replans > 0, 'No avoidance occurred'
            assert measured, 'No measured flight poses received'
            path = np.asarray(measured)
            clearance = np.linalg.norm(path[:, :2] - np.asarray(tree_xy), axis=1).min()
            assert clearance > 1.08, f'Trajectory intersects canopy/body margin: {clearance}'
            yaw = move(node, position, position, yaw, target_yaw=math.pi)
            nav.look(position, yaw)
            final = set(nav.map.voxels)
            assert len(final - first) > 20, 'No progressive mapping from new viewpoints'
            assert nav.map.points()[:, 0].max() > tree_xy[0] + .7, 'Back surface not mapped after viewing it'
            print(f'PASS: reached goal with {nav.replans} A* detour(s), min center distance {clearance:.3f} m; '
                  f'map grew {len(first)} -> {len(final)} voxels.', flush=True)
            # The return also uses perception and accumulated occupancy.
            position, yaw = nav.go(position, start, yaw)
            assert math.dist(position, start) < .03
            print('PASS: return path avoids the measured obstacle.', flush=True)
            request(node, 'control', WorldControl(pause=True))
            try:
                nav.observer.observe(position, yaw, timeout=1.)
            except RuntimeError:
                print('PASS: stopped camera prevents another movement observation.', flush=True)
            else:
                raise AssertionError('Accepted stale depth while simulation paused')
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots(figsize=(8, 6))
            p = nav.map.points()
            ax.scatter(p[:, 0], p[:, 1], s=8, color='tab:red', label='Observed surface voxels')
            ax.plot(path[:, 0], path[:, 1], label='Measured flight at 1.6 m')
            ax.plot([start[0], goal[0]], [start[1], goal[1]], '--', label='Original obstructed route')
            ax.set_aspect('equal'); ax.grid(); ax.legend()
            fig.savefig(artifacts / 'depth-navigation-check.png', dpi=160, bbox_inches='tight')
            (artifacts / 'depth-navigation-check.json').write_text(json.dumps({
                'first_voxels': len(first), 'final_voxels': len(final),
                'min_tree_center_distance': float(clearance), 'replans': nav.replans,
                'tree_location_known_only_by_test': tree_xy, 'path': path.tolist(),
            }, indent=2))
        finally:
            if recorder is not None:
                recorder.close()
                print(json.dumps(summarize(artifacts / 'rgb-depth/integration'), indent=2), flush=True)
            node.close()
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()


if __name__ == '__main__':
    main()

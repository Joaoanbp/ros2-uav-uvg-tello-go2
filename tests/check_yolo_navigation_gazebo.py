"""RGB-only YOLO/geometry/A* acceptance check. No depth sensor exists in this world.

Run in robotics with scripts on PYTHONPATH. The fixture alone knows the pillar
pose; the navigator gets RGB, calibration, own pose and intended waypoints.
"""

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

os.environ['GZ_PARTITION'] = 'yolo-navigation-' + uuid.uuid4().hex

import numpy as np
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.pose_v_pb2 import Pose_V

from depth_navigation import Navigator, DEPTH_TOPIC
from drone_zigzag import move, request
from gazebo_commands import CommandNode
from perception_eval import EvaluationRecorder
from roi_plan import plan_polygon, distance_to_segment, region_polygon
from zigzag_roi import mission_waypoints
from yolo_fixture import make_world, ROOT
from yolo_geometry import YoloKnownPillar, PILLAR_RADIUS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adaptive', action='store_true', help='Validate observed coverage and flexible lanes')
    args = parser.parse_args()
    output = ROOT / ('artifacts/yolo-adaptive' if args.adaptive else 'artifacts/yolo-navigation')
    output.mkdir(parents=True, exist_ok=True)
    # Intentionally different from the launcher fixture, not passed to YOLO.
    fixture_center = (.75, -.25)
    world = make_world(output / 'world.sdf', center=fixture_center, name='unfamiliar_' + uuid.uuid4().hex[:8])
    source = YoloKnownPillar(ROOT / 'models/yolo_pillar.pt', ROOT / 'models/yolo_pillar.json')
    assert source.navigation_ready, 'Held-out detector/geometry validation must pass first'
    home, altitude, radius = (-7., -5.) if args.adaptive else (-5., -3.), 1.6, 3.
    polygon = region_polygon('circle',radius)
    plan = plan_polygon(polygon, width=2. if args.adaptive else 6.,
                        overlap=80. if args.adaptive else 0., entry_from=home)
    waypoints = mission_waypoints(plan, altitude)
    result = {'source': source.name, 'success': False, 'depth_sensor_present': False,
              'fixture_location_known_only_by_test': fixture_center}
    recorder = EvaluationRecorder(output / 'observations')
    node = None
    with (output / 'gazebo.log').open('w') as log:
        server = subprocess.Popen(['gz', 'sim', '-s', '-r', str(world)], stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        try:
            node = CommandNode()
            nav = Navigator(node, altitude, (-9, 7, -7, 6), depth_source=source, recorder=recorder)
            if args.adaptive:
                nav.enable_coverage(plan)
            measured = []
            collecting = False
            def receive_pose(message):
                if collecting:
                    for item in message.pose:
                        if item.name == 'tello' and item.position.z > altitude-.1:
                            measured.append((item.position.x, item.position.y, item.position.z))
            assert node.subscribe(Pose_V, '/world/tello_only/pose/info', receive_pose)
            assert DEPTH_TOPIC not in nav.observer.subscribed_topics
            assert not nav.map.voxels, 'Obstacle location leaked into initial map'
            deadline = time.monotonic()+45
            while nav.observer.count < 3:
                if time.monotonic()>deadline or server.poll() is not None:
                    raise RuntimeError('RGB-only cameras failed to start')
                time.sleep(.05)
            from gz.transport13 import Node
            topics = Node().topic_list()
            assert not any('depth_image' in topic for topic in topics), topics
            result['sensor_topics'] = topics
            pose = Pose(name='tello')
            pose.position.x, pose.position.y, pose.position.z = (*home, .09)
            pose.orientation.w = 1.
            request(node, 'set_pose', pose)
            time.sleep(.3)
            yaw = move(node, (*home, .09), (*home, altitude), 0.)
            position = (*home, altitude)
            collecting = True
            initial = waypoints[:1+sum(leg.kind=='perimeter' for leg in plan.legs)] if args.adaptive else waypoints
            for label, target in initial:
                print(f'YOLO mission {label}: {target}', flush=True)
                position, yaw = nav.go(position, target, yaw)
            if args.adaptive:
                position, yaw = nav.cover(position,yaw)
                coverage = nav.coverage.summary()
                transitions = nav.coverage_selector.transitions
                assert coverage['pending_cells']==0, coverage
                assert len(transitions)<len(plan.lanes), transitions
                assert any(abs(a['lane']-b['lane'])>=2 for a,b in zip(transitions,transitions[1:])), transitions
                result.update(coverage=coverage, adaptive_tasks=len(transitions),
                              nominal_lanes=len(plan.lanes), lane_transitions=transitions)
            position, yaw = nav.go(position, (*home,altitude), yaw, adaptive=args.adaptive)
            assert math.dist(position, (*home, altitude)) < .03
            assert nav.skipped == 0, f'{nav.skipped} waypoints skipped'
            assert nav.replans > 0, 'No A* detour observed'
            assert nav.map.voxels, 'No visual obstacle mapped'
            collecting = False
            assert measured, 'No measured drone poses collected'
            path = np.asarray(measured)
            clearance = min(distance_to_segment(fixture_center, a[:2], b[:2]) for a,b in zip(path,path[1:]))
            assert clearance >= PILLAR_RADIUS + .5, f'Insufficient physical clearance: {clearance}'
            yaw = move(node, position, (*home, .09), yaw)
            # Sensor loss must still prevent another motion observation.
            from gz.msgs10.world_control_pb2 import WorldControl
            request(node, 'control', WorldControl(pause=True))
            try:
                nav.observer.observe((*home, .09), yaw, timeout=1.)
            except RuntimeError:
                sensor_loss_stops = True
            else:
                raise AssertionError('Accepted stale RGB while world paused')
            if args.adaptive:
                try:
                    nav.observer.ground_view(timeout=.1)
                except RuntimeError:
                    result['ground_sensor_loss_stops'] = True
                else:
                    raise AssertionError('Accepted stale downward RGB for coverage')
            result.update(success=True, source_subscriptions=nav.observer.subscribed_topics,
                          replans=nav.replans, skipped=nav.skipped, mapped_voxels=len(nav.map.voxels),
                          minimum_center_distance_m=clearance, minimum_surface_distance_m=clearance-PILLAR_RADIUS,
                          sensor_loss_stops=sensor_loss_stops, path=path.tolist())
            plot_result(output, plan, nav, path, fixture_center)
            print(f'PASS: RGB-only perimeter + A* detour + return + landing; center clearance {clearance:.2f}m', flush=True)
        except Exception as error:
            result['error'] = str(error)
            raise
        finally:
            recorder.close()
            (output / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
            if node is not None:
                node.close()
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()


def plot_result(output, plan, nav, path, fixture_center):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    events = [json.loads(line) for line in (output / 'observations/events.jsonl').read_text().splitlines()]
    frame = next(e for e in events if e['type']=='frame' and e['objects'])
    with np.load(output / 'observations' / frame['file']) as data:
        rgb = data['rgb']
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].imshow(rgb)
    for obj in frame['objects']:
        x1,y1,x2,y2 = obj['bbox']
        axes[0].add_patch(Rectangle((x1,y1), x2-x1, y2-y1, fill=False, edgecolor='lime', linewidth=2))
    axes[0].set_title('YOLO on RGB; known-size metric geometry')
    axes[0].axis('off')
    for leg in plan.legs:
        axes[1].plot([leg.start[0],leg.end[0]],[leg.start[1],leg.end[1]], ':', color='seagreen', alpha=.5)
    if nav.coverage is not None:
        grid = nav.coverage
        axes[1].scatter(*grid.centers[grid.seen].T, s=18, marker='s', c='mediumseagreen', alpha=.25,
                        label='Observed ground cells')
        axes[1].scatter(*grid.centers[grid.pending].T, s=18, c='goldenrod', label='Pending ground cells')
        axes[1].set_title(f'Adaptive coverage: {len(nav.coverage_selector.transitions)} useful passes')
    occupied = nav.map.points()
    axes[1].scatter(occupied[:,0], occupied[:,1], s=8, color='tab:red', label='RGB geometry + uncertainty')
    axes[1].add_patch(Circle(fixture_center, PILLAR_RADIUS, fill=False, color='black', label='Fixture truth (scoring only)'))
    axes[1].plot(path[:,0],path[:,1],color='tab:blue',label='Executed path')
    axes[1].set_aspect('equal'); axes[1].grid(); axes[1].legend(fontsize=8)
    if nav.coverage is None:
        axes[1].set_title('Perimeter, A* detour, return')
    fig.tight_layout(); fig.savefig(output / 'result.png', dpi=150); plt.close(fig)


if __name__ == '__main__':
    main()

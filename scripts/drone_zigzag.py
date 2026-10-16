#!/usr/bin/env python3
"""Drone zigzag: busca cinemática com o Tello no mundo tello_only."""

import argparse
import math
import time

from coverage_plan import add_coverage_arguments, coverage_lanes, coverage_summary, finite_coordinate
from coverage_plan import add_navigation_arguments, rectangular_waypoints


WORLD = "tello_only"


def request(node, service, message):
    from gz.msgs10.boolean_pb2 import Boolean

    executed, response = node.request(
        # The first camera render can stall an update while shaders are loaded.
        f"/world/{WORLD}/{service}", message, type(message), Boolean, 10000
    )
    if not executed or not response.data:
        reason = "timeout" if not executed else "request rejected"
        raise RuntimeError(f"Gazebo request failed: {service} ({reason})")


def move(node, start, end, yaw, speed=1.5, target_yaw=None):
    from gz.msgs10.pose_pb2 import Pose

    dx, dy = end[0] - start[0], end[1] - start[1]
    if target_yaw is None:
        target_yaw = math.atan2(dy, dx) if math.hypot(dx, dy) > 0.001 else yaw
    turn = math.atan2(math.sin(target_yaw - yaw), math.cos(target_yaw - yaw))
    duration = max(math.dist(start, end) / speed, abs(turn), 0.1)
    steps = math.ceil(duration * 20)
    began = time.monotonic()
    for step in range(1, steps + 1):
        fraction = step / steps
        blend = fraction * fraction * (3 - 2 * fraction)
        pose = Pose(name="tello")
        pose.position.x, pose.position.y, pose.position.z = (
            a + (b - a) * blend for a, b in zip(start, end)
        )
        angle = yaw + turn * blend
        pose.orientation.z = math.sin(angle / 2)
        pose.orientation.w = math.cos(angle / 2)
        # Queue the pose; the image/pose observer confirms application from fresh
        # timestamped sensor data before the navigator permits another step.
        request(node, "set_pose", pose)
        time.sleep(max(0, began + duration * fraction - time.monotonic()))
    return target_yaw


def fly_waypoints(home, waypoints, altitude=1.6, *, depth_source=None, evaluation_dir=None,
                  coverage_plan=None, coverage_mode='fixed'):
    """Kinematic flight with image-driven avoidance during horizontal motion."""
    from gz.msgs10.pose_pb2 import Pose
    from gz.msgs10.world_control_pb2 import WorldControl
    from gazebo_commands import CommandNode
    from depth_navigation import Navigator
    from perception_eval import EvaluationRecorder

    node = CommandNode()
    extent = [home, *[p[:2] for _, p in waypoints]]
    bounds = (min(p[0] for p in extent)-2, max(p[0] for p in extent)+2,
              min(p[1] for p in extent)-2, max(p[1] for p in extent)+2)
    recorder = EvaluationRecorder(evaluation_dir) if evaluation_dir else None
    navigator = Navigator(node, altitude, bounds, depth_source=depth_source, recorder=recorder)
    adaptive = coverage_mode == 'adaptive' and coverage_plan is not None
    if adaptive:
        navigator.enable_coverage(coverage_plan)
    # tello_only makes the drone static for direct-pose flight. Keep simulation
    # time advancing so RGB sensors publish new frames throughout the mission.
    request(node, "control", WorldControl(pause=False))
    position, yaw = (*home, 0.09), 0.0
    pose = Pose(name="tello")
    pose.position.x, pose.position.y, pose.position.z = position
    pose.orientation.w = 1.0
    request(node, "set_pose", pose)
    print(f"Starting position: {position}", flush=True)
    time.sleep(0.5)
    try:
        print(f'Takeoff: {(*home, altitude)}', flush=True)
        yaw = move(node, position, (*home, altitude), yaw)
        position = (*home, altitude)
        initial = waypoints
        if adaptive:
            perimeter_count = sum(leg.kind == 'perimeter' for leg in coverage_plan.legs)
            initial = waypoints[:perimeter_count+1]
        for label, target in initial:
            print(f"{label}: {target}", flush=True)
            position, yaw = navigator.go(position, target, yaw)
        if adaptive:
            position, yaw = navigator.cover(position, yaw)
        print(f'Return home: {(*home, altitude)}', flush=True)
        position, yaw = navigator.go(position, (*home, altitude), yaw, adaptive=adaptive)
        if math.dist(position, (*home, altitude)) > .05:
            raise RuntimeError('Base inacessível; mantendo altitude, sem pouso automático')
        yaw = move(node, position, (*home, .09), yaw)
        navigator.state = f'Missão concluída · {navigator.skipped} waypoints inacessíveis'
        navigator.route = []
        navigator.publish()
    except (RuntimeError, KeyboardInterrupt) as error:
        navigator.state = f'PARADO: {error}'
        navigator.route = []
        navigator.publish()
        raise
    finally:
        if recorder is not None:
            recorder.close()
        node.close()
    print("Search complete. Tello landed; simulation remains running for the cameras.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-x", type=finite_coordinate, default=-7.0, help="Home X in meters (default: -7)")
    parser.add_argument("--start-y", type=finite_coordinate, default=-5.0, help="Home Y in meters (default: -5)")
    add_coverage_arguments(parser)
    add_navigation_arguments(parser)
    args = parser.parse_args()
    try:
        lanes = coverage_lanes(args.footprint_width, args.side_overlap)
    except ValueError as error:
        parser.error(str(error))
    print(coverage_summary(args.footprint_width, args.side_overlap, lanes), flush=True)
    waypoints = rectangular_waypoints(lanes, (args.start_x, args.start_y), args.altitude)
    source = make_depth_source(args)
    from adaptive_coverage import rectangular_plan
    plan = rectangular_plan(lanes, (args.start_x, args.start_y), args.altitude,
                            args.footprint_width, args.side_overlap)
    fly_waypoints((args.start_x, args.start_y), waypoints, args.altitude,
                  depth_source=source, evaluation_dir=args.evaluation_dir,
                  coverage_plan=plan, coverage_mode=args.coverage_mode)


def make_depth_source(args):
    if args.depth_source == 'yolo':
        from yolo_geometry import YoloKnownPillar
        return YoloKnownPillar(args.yolo_checkpoint, args.yolo_profile, device=args.depth_device)
    from depth_sources import GazeboDepth, MetricIndoorDepth
    if args.depth_source == 'gazebo':
        return GazeboDepth()
    if args.depth_input_size < 98 or args.depth_cpu_threads < 1:
        raise ValueError('depth-input-size deve ser >= 98 e depth-cpu-threads >= 1')
    return MetricIndoorDepth(args.depth_repository, args.depth_checkpoint, args.depth_device,
                             args.depth_input_size, args.depth_cpu_threads)


if __name__ == "__main__":
    main()

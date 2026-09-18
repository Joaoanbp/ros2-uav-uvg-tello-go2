#!/usr/bin/env python3
"""Deterministic kinematic smoke test for the combined Gazebo world."""

import time

from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.world_control_pb2 import WorldControl
from gz.transport13 import Node


WORLD = "uav_ugv"
POSE_SERVICE = f"/world/{WORLD}/set_pose/blocking"
CONTROL_SERVICE = f"/world/{WORLD}/control"


def request(node, service, message, request_type):
    executed, response = node.request(
        service, message, request_type, Boolean, 2000
    )
    if not executed or not response.data:
        raise RuntimeError(f"Gazebo service request failed: {service}")


def set_paused(node, paused):
    request(node, CONTROL_SERVICE, WorldControl(pause=paused), WorldControl)


def set_pose(node, name, x, y, z, qz=0.0, qw=1.0):
    pose = Pose(name=name)
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    pose.orientation.z = qz
    pose.orientation.w = qw
    request(node, POSE_SERVICE, pose, Pose)


def smoothstep(value):
    return value * value * (3.0 - 2.0 * value)


def main():
    node = Node()

    # Direct pose commands are intentionally run while physics is paused. This
    # prevents gravity and ArduPilot from fighting the kinematic smoke test.
    set_paused(node, True)
    set_pose(node, "tello", 0.0, 0.0, 0.09)
    set_pose(node, "go2_edu", 1.5, 0.0, 0.46, 1.0, 0.0)

    print("Tello: ascending to 5 meters...", flush=True)
    for step in range(1, 101):
        fraction = smoothstep(step / 100.0)
        altitude = 0.09 + (5.0 - 0.09) * fraction
        set_pose(node, "tello", 0.0, 0.0, altitude)
        time.sleep(0.05)

    print("Go2 EDU: moving forward 2 meters...", flush=True)
    for step in range(1, 81):
        fraction = smoothstep(step / 80.0)
        # Go2 starts at yaw=pi, so forward is negative world X.
        x_position = 1.5 - 2.0 * fraction
        set_pose(node, "go2_edu", x_position, 0.0, 0.46, 1.0, 0.0)
        time.sleep(0.05)

    set_pose(node, "tello", 0.0, 0.0, 5.0)
    set_pose(node, "go2_edu", -0.5, 0.0, 0.46, 1.0, 0.0)
    print("Mission complete: Tello altitude=5 m, Go2 displacement=2 m.")
    print("Gazebo remains paused. Press Play to resume physics.")


if __name__ == "__main__":
    main()

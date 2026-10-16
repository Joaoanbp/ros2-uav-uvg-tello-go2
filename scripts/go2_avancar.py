#!/usr/bin/env python3
"""Go2 avançar: deslocamento cinemático para frente no Gazebo."""

import argparse
import math
import time

from gz.transport13 import Node

from missao_de_teste import set_paused, set_pose, smoothstep


def positive_distance(value):
    distance = float(value)
    if not math.isfinite(distance) or distance <= 0:
        raise argparse.ArgumentTypeError("distance must be a positive finite number")
    return distance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distance", type=positive_distance, default=5.0,
                        help="distance in meters (default: 5)")
    args = parser.parse_args()

    node = Node()
    for attempt in range(5):
        try:
            set_paused(node, True)
            break
        except RuntimeError:
            if attempt == 4:
                raise
            time.sleep(1)

    # The model faces negative world X (yaw=pi).
    set_pose(node, "go2_edu", 1.5, 0.0, 0.46, 1.0, 0.0)
    print(f"Go2: moving forward {args.distance:g} m", flush=True)
    time.sleep(1)
    steps = max(1, math.ceil(args.distance * 60))
    start = time.monotonic()
    for step in range(1, steps + 1):
        x = 1.5 - args.distance * smoothstep(step / steps)
        set_pose(node, "go2_edu", x, 0.0, 0.46, 1.0, 0.0)
        time.sleep(max(0, start + step * 0.05 - time.monotonic()))
    print(f"Go2: finished at X={1.5 - args.distance:g} m; Gazebo remains paused.",
          flush=True)


if __name__ == "__main__":
    main()

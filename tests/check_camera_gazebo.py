"""Integration check: run inside robotics with DISPLAY/XAUTHORITY configured.

PYTHONPATH=/usr/lib/python3/dist-packages:scripts python3 tests/check_camera_gazebo.py
Creates its own Gazebo transport partition and stops only its own server.
"""

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

os.environ['GZ_PARTITION'] = f'camera-check-{uuid.uuid4().hex}'

import numpy as np
from gz.msgs10.pose_pb2 import Pose
from gz.transport13 import Node

from drone_camera import CAMERAS, CameraStream
from drone_zigzag import request


ROOT = Path(__file__).resolve().parents[1]


def wait_for(predicate, message, timeout=45):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise AssertionError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--flight', action='store_true', help='Also run a complete ROI mission with the real Qt dashboard')
    parser.add_argument('--shape', choices=('circle', 'star', 'random'), default='circle')
    args = parser.parse_args()
    artifacts = ROOT / 'artifacts'
    artifacts.mkdir(exist_ok=True)
    with (artifacts / f'{args.shape}-camera-check-gazebo.log').open('w') as log:
        server = subprocess.Popen(
            ['gz', 'sim', '-s', '-r', '-v', '3', str(ROOT / 'simulation/worlds/tello_only.sdf')],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
        )
        try:
            node = Node()
            streams = [CameraStream(node, topic) for _, topic in CAMERAS]
            wait_for(lambda: all(s.snapshot()[2] >= 3 for s in streams),
                     f'Both cameras must publish frames. See artifacts/{args.shape}-camera-check-gazebo.log')

            def capture(x):
                pose = Pose(name='tello')
                pose.position.x, pose.position.y, pose.position.z = x, 0, 3
                pose.orientation.w = 1
                print(f'Moving drone to ({x}, 0, 3); frames: {[s.snapshot()[2] for s in streams]}', flush=True)
                request(node, 'set_pose/blocking', pose)
                counts = [s.snapshot()[2] for s in streams]
                wait_for(lambda: all(s.snapshot()[2] >= count + 5 for s, count in zip(streams, counts)),
                         'Camera frames stopped after moving the drone', timeout=15)
                return [s.snapshot()[0] for s in streams]

            front_view = capture(-3)
            overhead_view = capture(0)
            for before, after in zip(front_view, overhead_view):
                assert before.shape == (480, 640, 3)
                assert np.mean(np.abs(before.astype(float) - after.astype(float))) > 1, 'Image did not change with drone pose'
            from matplotlib import image
            image.imsave(artifacts / f'{args.shape}-camera-front-check.png', front_view[0])
            image.imsave(artifacts / f'{args.shape}-camera-down-check.png', overhead_view[1])
            print('PASS: both cameras publish changing 640x480 RGB images while the static drone moves.', flush=True)
            print(f'Saved artifacts/{args.shape}-camera-front-check.png and {args.shape}-camera-down-check.png', flush=True)
            if args.flight:
                output = artifacts / f'{args.shape}-camera-flight-check.png'
                output.unlink(missing_ok=True)
                flight_log = artifacts / f'{args.shape}-camera-flight-check.log'
                with flight_log.open('w') as mission_log:
                    flight = subprocess.Popen([
                        sys.executable, str(ROOT / 'scripts/drone_trajetoria.py'),
                        '--shape', args.shape, '--run-search', '--depth-source', 'gazebo',
                        '--radius', '2' if args.shape == 'circle' else '4',
                        '--footprint-width', '2' if args.shape == 'circle' else '3', '--side-overlap', '0', '--altitude', '1.6',
                        '--start-x', '-3', '--start-y', '-2.5', '--output', str(output),
                    ], stdout=mission_log, stderr=subprocess.STDOUT, start_new_session=True)
                    try:
                        wait_for(lambda: 'Saved plot:' in flight_log.read_text() or flight.poll() is not None,
                                 f'Live {args.shape} mission did not finish within 600 s', timeout=600)
                        text = flight_log.read_text()
                        assert output.exists(), f'GUI exit={flight.poll()}; {text}'
                        assert 'Search complete. Tello landed' in text, text
                        assert 'Recorded ' in text, text
                        assert 'Obstacle observed: A* detour' in text, text
                        print(f'PASS: full {args.shape} flight, Qt camera dashboard, measured path and landing.', flush=True)
                    finally:
                        if flight.poll() is None:
                            os.killpg(flight.pid, signal.SIGTERM)
                            try:
                                flight.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                os.killpg(flight.pid, signal.SIGKILL)
                                flight.wait()
        finally:
            if server.poll() is None:
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()


if __name__ == '__main__':
    main()

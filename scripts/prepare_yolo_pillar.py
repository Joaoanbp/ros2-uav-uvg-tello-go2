#!/usr/bin/env python3
"""Capture RGB training fixtures, train YOLO, and validate metric ranging.

Fixture geometry supplies OFFLINE labels/scoring. Navigation loads only the
detector and known object dimensions, never these obstacle locations.
"""

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace
import uuid

import numpy as np

from yolo_fixture import ROOT, make_world
from yolo_geometry import cylinder_bbox, estimate_cylinder, PILLAR_HEIGHT, PILLAR_RADIUS


def capture(output, counts):
    import cv2
    os.environ['GZ_PARTITION'] = 'yolo-capture-' + uuid.uuid4().hex
    from depth_navigation import DepthObserver, synchronized_sample, rotation
    from gazebo_commands import CommandNode
    from drone_zigzag import request
    from gz.msgs10.pose_pb2 import Pose
    output.mkdir(parents=True, exist_ok=True)
    fixture_name = 'training_fixture_' + uuid.uuid4().hex[:8]
    world = make_world(output / 'world.sdf', name=fixture_name)
    records = []
    with (output / 'gazebo.log').open('w') as log:
        server = subprocess.Popen(['gz', 'sim', '-s', '-r', str(world)], stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True)
        node = None
        try:
            node = CommandNode()
            observer = DepthObserver(node, source=SimpleNamespace(name='rgb_yolo_capture', geometry_mode=True))
            deadline = time.monotonic()+45
            while observer.count < 3:
                if time.monotonic()>deadline or server.poll() is not None:
                    raise RuntimeError('RGB fixture did not start')
                time.sleep(.05)
            for split, count, seed in zip(('train', 'val', 'test'), counts, (404, 505, 606)):
                rng = np.random.default_rng(seed)
                (output / 'images' / split).mkdir(parents=True, exist_ok=True)
                (output / 'labels' / split).mkdir(parents=True, exist_ok=True)
                for index in range(count):
                    center = rng.uniform(-1.5, 1.5, 2)
                    angle = rng.uniform(-math.pi, math.pi)
                    distance = rng.uniform(2.1, 8.5)
                    position = (*tuple(center + distance*np.array([math.cos(angle), math.sin(angle)])), rng.uniform(1.4, 1.85))
                    negative = index % 8 == 0
                    yaw = angle + math.pi + (math.pi if negative else rng.uniform(-.40, .40))
                    yaw = math.atan2(math.sin(yaw), math.cos(yaw))
                    target = Pose(name=fixture_name)
                    target.position.x, target.position.y = center
                    target.orientation.w = 1.
                    request(node, 'set_pose', target)
                    drone = Pose(name='tello')
                    drone.position.x, drone.position.y, drone.position.z = position
                    drone.orientation.z, drone.orientation.w = math.sin(yaw/2), math.cos(yaw/2)
                    request(node, 'set_pose', drone)
                    after = observer.count+2
                    deadline = time.monotonic()+10
                    selected = None
                    while selected is None:
                        if time.monotonic()>deadline:
                            raise RuntimeError('No synchronized RGB/own pose for fixture')
                        with observer.lock:
                            frames, poses = list(observer.frames), list(observer.poses)
                            intrinsics = observer.intrinsics
                        for number, timestamp, _, rgb in reversed(frames):
                            own = synchronized_sample(poses, timestamp, .03)
                            if number < after or own is None or intrinsics is None or math.dist(own[1], position) > .015:
                                continue
                            observed_yaw = math.atan2(rotation(own[2])[1, 0], rotation(own[2])[0, 0])
                            if abs(math.atan2(math.sin(observed_yaw-yaw), math.cos(observed_yaw-yaw))) > .015:
                                continue
                            selected = timestamp, rgb, own, intrinsics
                            break
                        time.sleep(.01)
                    timestamp, rgb, own, intrinsics = selected
                    image_h, image_w = rgb.shape[:2]
                    bbox = None
                    if not negative:
                        raw = cylinder_bbox(center, intrinsics, own[1], own[2])
                        bbox = np.clip(raw, [0, 0, 0, 0], [image_w, image_h, image_w, image_h])
                    stem = f'{split}-{index:04d}'
                    image_path = output / 'images' / split / f'{stem}.jpg'
                    cv2.imwrite(str(image_path), rgb[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 96])
                    label = ''
                    if bbox is not None:
                        x1,y1,x2,y2 = bbox
                        label = f'0 {(x1+x2)/2/image_w:.8f} {(y1+y2)/2/image_h:.8f} {(x2-x1)/image_w:.8f} {(y2-y1)/image_h:.8f}\n'
                    (output / 'labels' / split / f'{stem}.txt').write_text(label)
                    records.append({'split': split, 'image': str(image_path.relative_to(output)),
                                    'sim_time': timestamp, 'pose_xyz': own[1], 'pose_xyzw': own[2],
                                    'intrinsics': intrinsics, 'center_truth_for_evaluation_only': center.tolist(),
                                    'bbox_truth': bbox.tolist() if bbox is not None else None})
                    if (index+1) % 20 == 0:
                        print(f'Captured {split}: {index+1}/{count}', flush=True)
        finally:
            if node is not None:
                node.close()
            if server.poll() is None:
                import signal
                os.killpg(server.pid, signal.SIGTERM)
                try:
                    server.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()
    (output / 'manifest.json').write_text(json.dumps({'mode': 'offline_rgb_training_fixture',
        'synthetic_depth_present': False, 'records': records}, indent=2)+'\n')
    (output / 'data.yaml').write_text(f'path: {output.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: calibration_pillar\n')


def train(args):
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    model = YOLO(str(ROOT / 'models/yolov8n.pt'))
    model.train(data=str(args.dataset / 'data.yaml'), epochs=args.epochs, imgsz=320, batch=16,
                device='cpu', workers=0, freeze=10, optimizer='AdamW', lr0=.002,
                patience=10, seed=711, deterministic=True, project=str(args.dataset / 'runs'),
                name='pillar', exist_ok=True, cache=True, plots=False,
                mosaic=.2, close_mosaic=5, translate=.1, scale=.25, fliplr=.5, hsv_v=.3)
    shutil.copy2(args.dataset / 'runs/pillar/weights/best.pt', ROOT / 'models/yolo_pillar.pt')


def validate(args):
    import cv2
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    checkpoint = ROOT / 'models/yolo_pillar.pt'
    model = YOLO(str(checkpoint))
    records = json.loads((args.dataset / 'manifest.json').read_text())['records']
    report = []
    for record in (r for r in records if r['split'] == 'test'):
        bgr = cv2.imread(str(args.dataset / record['image']))
        began = time.monotonic()
        prediction = model.predict(bgr, imgsz=320, conf=.30, iou=.5, device='cpu', verbose=False)[0]
        elapsed = time.monotonic()-began
        truth = record['bbox_truth']
        eligible = truth is not None and truth[0]>=3 and truth[2]<=bgr.shape[1]-3 and truth[3]<=bgr.shape[0]-3
        estimates = []
        for box, confidence in zip(prediction.boxes.xyxy.cpu().numpy(), prediction.boxes.conf.cpu().numpy()):
            try:
                estimates.append(estimate_cylinder(box, record['intrinsics'], record['pose_xyz'],
                                                   record['pose_xyzw'], bgr.shape[:2], confidence=confidence))
            except ValueError:
                pass
        best = min(estimates, key=lambda e: math.dist(e.center, record['center_truth_for_evaluation_only'])) if estimates else None
        error = math.dist(best.center, record['center_truth_for_evaluation_only']) if best is not None else None
        report.append({'image': record['image'], 'eligible': eligible, 'negative': truth is None,
                       'detections': len(prediction.boxes), 'processing_s': elapsed,
                       'estimate': asdict(best) if best is not None else None, 'position_error_m': error})
    eligible_rows = [r for r in report if r['eligible']]
    detected = [r for r in eligible_rows if r['estimate'] is not None]
    errors = [r['position_error_m'] for r in detected]
    misses = len(eligible_rows)-len(detected)
    false_positives = sum(r['negative'] and r['detections'] > 0 for r in report)
    approved = bool(eligible_rows and misses == 0 and false_positives == 0 and
                    max(errors, default=999) < .4 and
                    all(r['position_error_m'] <= r['estimate']['uncertainty_m'] for r in detected) and
                    np.percentile([r['processing_s'] for r in report[1:]], 95) < .75)
    result = {'eligible': len(eligible_rows), 'misses': misses, 'false_positives': false_positives,
              'position_mae_m': float(np.mean(errors)) if errors else None,
              'position_max_error_m': max(errors, default=None), 'approved': approved, 'records': report}
    (args.dataset / 'validation.json').write_text(json.dumps(result, indent=2)+'\n')
    profile = {'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
               'class_name': 'calibration_pillar', 'radius_m': PILLAR_RADIUS, 'height_m': PILLAR_HEIGHT,
               'input_size': 320, 'nms_iou': .5, 'approved_controlled_simulation': approved,
               'domain': 'static known-size pillar on flat z=0 floor; simulator own pose',
               'validation_summary': {k:v for k,v in result.items() if k != 'records'}}
    (ROOT / 'models/yolo_pillar.json').write_text(json.dumps(profile, indent=2)+'\n')
    print(json.dumps(profile, indent=2), flush=True)
    if not approved:
        raise RuntimeError('YOLO/geometry validation failed; see validation.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'artifacts/yolo-pillar')
    parser.add_argument('--capture', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--counts', type=int, nargs=3, default=(180, 40, 48))
    args = parser.parse_args()
    args.dataset = args.dataset.resolve()
    if args.capture:
        capture(args.dataset, args.counts)
    if args.train:
        train(args)
    if args.validate:
        validate(args)
    if not (args.capture or args.train or args.validate):
        parser.error('Choose --capture, --train and/or --validate')


if __name__ == '__main__':
    main()

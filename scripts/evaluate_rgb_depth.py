#!/usr/bin/env python3
"""Capture fixed Gazebo views, replay RGB inference, and build a diagnostic report.

Capture uses test-harness pose commands, not autonomous navigation. Reference
depth is saved for evaluation only. Replay never feeds reference into inference.
"""

import argparse
import hashlib
from html import escape
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

import numpy as np

from perception_scenarios import ROOT, SAMPLE_POSES, SCENARIOS, make_world
from perception_eval import depth_metrics, depth_world_image


def capture_scenario(scenario, output):
    os.environ['GZ_PARTITION'] = 'rgb-eval-' + uuid.uuid4().hex
    from depth_navigation import DepthObserver, synchronized_sample, rotation
    from gazebo_commands import CommandNode
    from drone_zigzag import request
    from gz.msgs10.pose_pb2 import Pose

    output.mkdir(parents=True, exist_ok=True)
    world = make_world(scenario, output / 'world.sdf')
    samples = []
    with (output / 'gazebo.log').open('w') as log:
        server = subprocess.Popen(['gz', 'sim', '-s', '-r', str(world)], stdout=log, stderr=subprocess.STDOUT)
        node = None
        try:
            node = CommandNode()
            # Evaluation-only observer: subscribe both streams; do not call observe/infer.
            observer = DepthObserver(node, recorder=True)
            deadline = time.monotonic() + 45
            while observer.count < 3:
                if time.monotonic() > deadline or server.poll() is not None:
                    raise RuntimeError(f'{scenario}: no camera stream; see {output / "gazebo.log"}')
                time.sleep(.05)
            for index, (x, y, z, yaw) in enumerate(SAMPLE_POSES):
                pose = Pose(name='tello')
                pose.position.x, pose.position.y, pose.position.z = x, y, z
                pose.orientation.z, pose.orientation.w = math.sin(yaw/2), math.cos(yaw/2)
                request(node, 'set_pose', pose)
                after = observer.count + 3
                deadline = time.monotonic() + 12
                matched = None
                while matched is None:
                    if time.monotonic() > deadline:
                        raise RuntimeError(f'{scenario}: no synchronized RGB/depth/pose')
                    with observer.lock:
                        frames, rgbs, poses = list(observer.frames), list(observer.rgb_references), list(observer.poses)
                        intrinsics = observer.intrinsics
                    for number, timestamp, _, reference in reversed(frames):
                        rgb = synchronized_sample(rgbs, timestamp)
                        own = synchronized_sample(poses, timestamp, .02)
                        if number < after or rgb is None or own is None or intrinsics is None:
                            continue
                        measured_yaw = math.atan2(rotation(own[2])[1, 0], rotation(own[2])[0, 0])
                        if math.dist(own[1], (x,y,z)) > .01 or abs(measured_yaw-yaw) > .01:
                            continue
                        matched = (timestamp, reference, rgb, own, intrinsics)
                        break
                    time.sleep(.02)
                timestamp, reference, rgb, own, intrinsics = matched
                filename = f'sample-{index:02d}.npz'
                np.savez_compressed(output / filename, rgb=rgb[1], reference=reference)
                samples.append({'file': filename, 'sim_time': timestamp, 'rgb_time': rgb[0],
                                'pose_time': own[0], 'pose_xyz': own[1], 'pose_xyzw': own[2],
                                'intrinsics': intrinsics, 'dimensions': list(reference.shape)})
                print(f'Captured {scenario} view {index+1}/{len(SAMPLE_POSES)}', flush=True)
        finally:
            if node is not None:
                node.close()
            if server.poll() is None:
                server.terminate()
                try:
                    server.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()
    (output / 'manifest.json').write_text(json.dumps({
        'scenario': scenario, 'seed': 2026, 'mode': 'stationary_test_fixture',
        'world_sha256': hashlib.sha256(world.read_bytes()).hexdigest(),
        'depth_convention': 'axial meters', 'samples': samples}, indent=2) + '\n')


def projection_checks(reference, sample, scenario):
    """Confirm Gazebo axial depth and extrinsics against fixture planes."""
    points = depth_world_image(reference, sample['intrinsics'], sample['pose_xyz'], sample['pose_xyzw'])
    valid = np.isfinite(reference) & (reference > .08) & (reference < 20)
    from depth_navigation import CAMERA_OFFSET, rotation
    origin = np.asarray(sample['pose_xyz']) + CAMERA_OFFSET @ rotation(sample['pose_xyzw']).T
    rays = depth_world_image(np.ones_like(reference), sample['intrinsics'], sample['pose_xyz'], sample['pose_xyzw']) - origin
    with np.errstate(divide='ignore', invalid='ignore'):
        ground_depth = -origin[2]/rays[..., 2]
    hits = origin + rays*ground_depth[..., None]
    v, _ = np.mgrid[:reference.shape[0], :reference.shape[1]]
    # Known visible fixture floor, selected analytically rather than by residual.
    floor = (valid & (v > reference.shape[0]*.8) & (ground_depth > 0) &
             (ground_depth < 20) & (hits[..., 0] < -.4) & (hits[..., 0] > -5.5) &
             (np.abs(hits[..., 1]) < 4.5))
    result = {'floor_pixels': int(floor.sum()),
              'floor_p95_height_error_m': float(np.percentile(np.abs(points[..., 2][floor]), 95)) if floor.any() else None}
    if scenario == 'wall':
        # The front face of the calibration wall is x=.5, y in [-1,1], z in [0,3].
        expected = (.5-origin[0])/rays[..., 0]
        hits = origin + rays*expected[..., None]
        mask = ((expected > .08) & (np.abs(hits[..., 1]) < .9) &
                (hits[..., 2] > .1) & (hits[..., 2] < 2.9) & valid)
        residual = np.abs(reference[mask]-expected[mask])
        result.update(wall_pixels=int(mask.sum()),
                      wall_p95_axial_error_m=float(np.percentile(residual, 95)) if residual.size else None)
    return result


def plot_sample(rgb, estimated, reference, path, title):
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(rgb); axes[0].set_title('Front RGB')
    truth = np.isfinite(reference) & (reference > .08) & (reference < 20)
    for ax, data, label in ((axes[1], estimated, 'Estimated axial depth (m)'),
                            (axes[2], np.where(truth, reference, np.nan), 'Gazebo reference (m)')):
        im = ax.imshow(data, vmin=0, vmax=10, cmap='viridis')
        ax.set_title(label); fig.colorbar(im, ax=ax, fraction=.045)
    im = axes[3].imshow(np.where(truth, estimated-reference, np.nan), vmin=-3, vmax=3, cmap='coolwarm')
    axes[3].set_title('Estimate − reference (m)'); fig.colorbar(im, ax=axes[3], fraction=.045)
    for ax in axes:
        ax.axis('off')
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=115); plt.close(fig)


def evaluate(args):
    from depth_sources import MetricIndoorDepth, quality_from_rgb, validate_estimate
    from depth_navigation import SurfaceMap, project_depth
    model = MetricIndoorDepth(args.repository, args.checkpoint, args.device,
                              input_size=args.input_sizes[0], cpu_threads=4)
    result_dir = args.output / 'results'
    result_dir.mkdir(exist_ok=True)
    frames = []
    checkpoint_hash = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    for size in args.input_sizes:
        model.input_size = size
        # Warmup is measured separately; no moving robot is involved in replay.
        warmup_start = time.monotonic()
        model.infer(np.full((480, 640, 3), 128, dtype=np.uint8))
        warmup = time.monotonic()-warmup_start
        for scenario in args.scenarios:
            directory = args.output / 'dataset' / scenario
            manifest = json.loads((directory / 'manifest.json').read_text())
            for index, sample in enumerate(manifest['samples']):
                with np.load(directory / sample['file']) as data:
                    rgb, reference = data['rgb'], data['reference']
                durations = []
                for _ in range(args.repeats):
                    began = time.monotonic(); estimated = model.infer(rgb)
                    durations.append(time.monotonic()-began)
                metrics = depth_metrics(estimated, reference, intrinsics=sample['intrinsics'],
                                        position=sample['pose_xyz'], quaternion=sample['pose_xyzw'])
                quality = quality_from_rgb(rgb)
                rejected = []
                try:
                    validate_estimate(estimated, reference.shape, max(durations))
                except RuntimeError as error:
                    rejected.append(str(error))
                if quality['luma_mean'] < 15 or quality['luma_mean'] > 240 or quality['saturated_fraction'] > .75 or quality['luma_std'] < 5:
                    rejected.append('Image quality gate')
                surface = SurfaceMap(sample['pose_xyz'][2])
                surface.integrate(project_depth(estimated, sample['intrinsics'], sample['pose_xyz'], sample['pose_xyzw']))
                if len(surface.voxels) > 400:
                    rejected.append('Surface-growth gate (>400 voxels)')
                png = f'{scenario}-{index:02d}-{size}.png'
                np.save(result_dir / f'{scenario}-{index:02d}-{size}.npy', estimated)
                plot_sample(rgb, estimated, reference, result_dir / png, f'{scenario}, view {index+1}, input {size}')
                row = {'scenario': scenario, 'view': index, 'input_size': size, 'panel': png,
                       'processing_seconds': durations, 'warmup_seconds': warmup,
                       'quality': quality, 'quality_gate_reasons': rejected,
                       'estimated_surface_voxels': len(surface.voxels), 'metrics': metrics,
                       'projection': projection_checks(reference, sample, scenario)}
                frames.append(row)
                print(f'{scenario} view {index+1} @{size}: MAE={metrics["mae_m"]:.2f}m, inference={max(durations):.2f}s, gates={len(rejected)}', flush=True)
    report = {'mode': 'stationary_evaluation_not_autonomous_flight', 'model': model.name,
              'rgb_navigation_approved': False,
              'checkpoint_sha256': checkpoint_hash, 'frames': frames,
              'device': args.device, 'cpu_threads': 4, 'repeats': args.repeats,
              'model_repository_commit': subprocess.check_output(
                  ['git', '-C', args.repository, 'rev-parse', 'HEAD'], text=True).strip(),
              'exploratory_limits': {'mae_m': .3, 'p95_error_m': .5, 'missed_near_fraction': .01, 'processing_s': .75},
              'localization': 'simulator truth; no camera-localization validation'}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    write_report(args.output, report)


def write_report(output, report):
    frames = report['frames']
    rows = []
    for scenario in dict.fromkeys(row['scenario'] for row in frames):
        for size in dict.fromkeys(row['input_size'] for row in frames):
            selected = [r for r in frames if r['scenario'] == scenario and r['input_size'] == size]
            if not selected:
                continue
            mae = np.mean([r['metrics']['mae_m'] for r in selected])
            p95 = max(r['metrics']['p95_absolute_error_m'] for r in selected)
            delay = np.percentile([t for r in selected for t in r['processing_seconds']], 95)
            missed = sum(r['metrics']['missed_near_pixels'] for r in selected)
            near = sum(r['metrics']['near_pixels_under_4m'] for r in selected)
            band_total = sum(r['metrics'].get('flight_band_pixels', 0) for r in selected)
            band_miss = sum(r['metrics'].get('flight_band_missed_pixels', 0) for r in selected)
            false_near = sum(r['metrics']['false_near_pixels'] for r in selected)
            far_pixels = sum(r['metrics'].get('far_pixels', 0) for r in selected)
            rate = missed/near if near else 0.
            gates = sum(bool(r['quality_gate_reasons']) for r in selected)
            band_rate = band_miss/band_total if band_total else None
            accuracy = mae <= .3 and p95 <= .5 and rate <= .01 and (band_rate is None or band_rate <= .01)
            passed = accuracy and delay <= .75 and gates == 0
            rows.append({'scenario': scenario, 'input_size': size, 'mean_frame_mae_m': float(mae),
                         'worst_frame_p95_m': float(p95), 'processing_p95_s': float(delay),
                         'missed_near_fraction': rate if near else None,
                         'false_near_fraction': false_near/far_pixels if far_pixels else None,
                         'flight_band_missed_fraction': band_rate,
                         'rejected_views': gates, 'views': len(selected),
                         'accuracy_pass': bool(accuracy), 'exploratory_pass': bool(passed)})
    report['summary'] = rows
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    def fmt(value, percent=False):
        return 'n/a' if value is None else (f'{100*value:.1f}%' if percent else f'{value:.2f}')
    table = []
    for r in rows:
        table.append('<tr>' + ''.join(f'<td>{v}</td>' for v in (
            escape(r['scenario']), r['input_size'], fmt(r['mean_frame_mae_m']), fmt(r['worst_frame_p95_m']),
            fmt(r['missed_near_fraction'], True), fmt(r['flight_band_missed_fraction'], True), fmt(r['false_near_fraction'], True),
            fmt(r['processing_p95_s']), f'{r["rejected_views"]}/{r["views"]}', 'PASS' if r['exploratory_pass'] else 'FAIL')) + '</tr>')
    panels = []
    for frame in frames:
        panels.append(f'<details><summary>{escape(frame["scenario"])} · view {frame["view"]+1} · input {frame["input_size"]}</summary>'
                      f'<img loading="lazy" src="results/{escape(frame["panel"])}" alt="RGB, estimated depth, Gazebo depth and error">'
                      f'<p>Gate reasons: {escape(", ".join(frame["quality_gate_reasons"]) or "none")}</p></details>')
    html = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Tello RGB depth evaluation</title><style>body{font:16px system-ui;max-width:1500px;margin:32px auto;padding:0 20px;background:#f5f7fb;color:#172033}table{border-collapse:collapse;width:100%;background:white}th,td{padding:10px;border-bottom:1px solid #ddd;text-align:left}img{width:100%}details{background:white;padding:12px;margin:10px 0}summary{cursor:pointer}p{max-width:1000px;line-height:1.5}</style>
<h1>Tello RGB depth evaluation</h1><p>Fixed test-fixture poses, simulator localization. Reference depth is used only for scoring. This is not an autonomous-flight acceptance report.</p>
<p>Exploratory limits: mean frame MAE ≤0.30 m, every frame p95 ≤0.50 m, missed pixels below 4 m ≤1%, inference p95 ≤0.75 s, and no image/map gate rejections. These limits are explicit experiment criteria, not a safety certification. Missing flight-band observations are shown as n/a.</p>
<p>RGB and Gazebo depth are paired within 5 ms; drone pose within 20 ms. The wall fixture independently checks axial depth and camera projection against a known plane. No reference-based rescaling or fitting is applied to model output.</p>
<table><thead><tr><th>Scenario</th><th>Input</th><th>MAE m</th><th>Worst p95 m</th><th>Overestimated near pixels</th><th>Missed near flight band</th><th>False near pixels</th><th>Infer p95 s</th><th>Rejected views</th><th>Criteria</th></tr></thead><tbody>'''
    wall_checks = [r['projection']['wall_p95_axial_error_m'] for r in frames
                   if r['scenario'] == 'wall' and r['projection'].get('wall_p95_axial_error_m') is not None]
    wall_text = f'<p>Worst wall-projection p95 residual: {1000*max(wall_checks):.1f} mm.</p>' if wall_checks else ''
    fov_text = ''
    if report.get('fov_diagnostic'):
        fov_text = '<h2>Field-of-view diagnostic</h2><p>The simulated camera has an 80° horizontal view. Hypersim training uses about 60°. These scores compare the same central pixels with full-image context versus a 60° RGB crop, without changing metric scale. Cropping leaves the image borders unobserved and is not enabled for navigation.</p><table><tr><th>Scenario</th><th>Full-context central MAE m</th><th>60° crop MAE m</th></tr>'
        for item in report['fov_diagnostic']:
            fov_text += f'<tr><td>{escape(item["scenario"])}</td><td>{item["full_context_metrics"]["mae_m"]:.2f}</td><td>{item["cropped_context_metrics"]["mae_m"]:.2f}</td></tr>'
        fov_text += '</table>'
    html += ''.join(table) + '</tbody></table>' + wall_text + fov_text + '<h2>Recorded views</h2>' + ''.join(panels) + '</html>'
    (output / 'report.html').write_text(html, encoding='utf-8')
    print(f'Report: {output / "report.html"}', flush=True)


def diagnose_fov(args):
    from depth_sources import MetricIndoorDepth
    report = json.loads((args.output / 'report.json').read_text())
    size = min(row['input_size'] for row in report['frames'])
    model = MetricIndoorDepth(args.repository, args.checkpoint, args.device, input_size=size)
    diagnostics = []
    for scenario in ('outdoor_clear', 'indoor_clear', 'wall', 'tree', 'poor_texture'):
        directory = args.output / 'dataset' / scenario
        if not (directory / 'manifest.json').exists():
            continue
        sample = json.loads((directory / 'manifest.json').read_text())['samples'][0]
        with np.load(directory / sample['file']) as data:
            rgb, reference = data['rgb'], data['reference']
        full = np.load(args.output / 'results' / f'{scenario}-00-{size}.npy')
        h, w = reference.shape
        # Training intrinsics from metric_depth/dataset/hypersim.py, not a fit to reference.
        crop_w = min(w, round(sample['intrinsics'][0] / (886.81/1024)))
        crop_h = min(h, round(crop_w*h/w))
        x, y = (w-crop_w)//2, (h-crop_h)//2
        selection = np.s_[y:y+crop_h, x:x+crop_w]
        estimated = model.infer(rgb[selection])
        diagnostic = {'scenario': scenario, 'input_size': size, 'crop_xywh': [x, y, crop_w, crop_h],
                      'full_context_metrics': depth_metrics(full[selection], reference[selection]),
                      'cropped_context_metrics': depth_metrics(estimated, reference[selection])}
        diagnostics.append(diagnostic)
        print(f'FOV {scenario}: full={diagnostic["full_context_metrics"]["mae_m"]:.2f}m, crop={diagnostic["cropped_context_metrics"]["mae_m"]:.2f}m', flush=True)
    report['fov_diagnostic'] = diagnostics
    refresh_report(args, report)


def refresh_report(args, report=None):
    from depth_sources import quality_from_rgb
    if report is None:
        report = json.loads((args.output / 'report.json').read_text())
    for row in report['frames']:
        directory = args.output / 'dataset' / row['scenario']
        sample = json.loads((directory / 'manifest.json').read_text())['samples'][row['view']]
        with np.load(directory / sample['file']) as data:
            row['quality'] = quality_from_rgb(data['rgb'])
            estimated = np.load(args.output / 'results' / f'{row["scenario"]}-{row["view"]:02d}-{row["input_size"]}.npy')
            row['metrics'] = depth_metrics(estimated, data['reference'], intrinsics=sample['intrinsics'],
                                           position=sample['pose_xyz'], quaternion=sample['pose_xyzw'])
            row['projection'] = projection_checks(data['reference'], sample, row['scenario'])
    report['rgb_navigation_approved'] = False
    write_report(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/perception-evaluation')
    parser.add_argument('--scenarios', nargs='+', choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument('--capture', action='store_true', help='Recapture dataset before evaluation')
    parser.add_argument('--capture-only', action='store_true')
    parser.add_argument('--capture-scenario', choices=SCENARIOS, help=argparse.SUPPRESS)
    parser.add_argument('--input-sizes', type=int, nargs='+', default=[252, 518])
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--repository', default=str(ROOT / 'third_party/Depth-Anything-V2'))
    parser.add_argument('--checkpoint', default=str(ROOT / 'models/depth_anything_v2_metric_hypersim_vits.pth'))
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--diagnose-fov', action='store_true', help='Add a camera-FOV diagnostic to an existing report')
    parser.add_argument('--refresh-report', action='store_true', help='Recompute metrics from saved estimates without inference')
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.repeats < 1 or min(args.input_sizes) < 98:
        parser.error('repeats must be positive and input sizes >=98')
    if args.capture_scenario:
        capture_scenario(args.capture_scenario, args.output / 'dataset' / args.capture_scenario)
        return
    if args.diagnose_fov:
        diagnose_fov(args)
        return
    if args.refresh_report:
        refresh_report(args)
        return
    if args.capture or args.capture_only:
        for scenario in args.scenarios:
            subprocess.run([sys.executable, __file__, '--capture-scenario', scenario, '--output', str(args.output)], check=True)
    if not args.capture_only:
        evaluate(args)


if __name__ == '__main__':
    main()

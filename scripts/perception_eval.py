"""Synchronized evaluation artifacts. Never import this into a depth estimator."""

import argparse
import json
from pathlib import Path

import numpy as np


def depth_metrics(estimated, reference, *, intrinsics=None, position=None, quaternion=None):
    """Pixel and flight-height errors; invalid predictions count as missed measurements."""
    if estimated.shape != reference.shape or estimated.ndim != 2:
        raise ValueError('Depth arrays must have matching HxW shapes')
    truth = np.isfinite(reference) & (reference > .08) & (reference < 20)
    prediction = np.isfinite(estimated) & (estimated > .08) & (estimated < 20)
    common = truth & prediction
    error = np.zeros_like(estimated, dtype=float)
    np.subtract(estimated, reference, out=error, where=common)
    absolute = np.abs(error[common])
    near = truth & (reference < 4.)
    missed = near & (~prediction | (error > .5))
    far = truth & (reference >= 2.5)
    false_near = far & prediction & (estimated < 2.)
    result = {
        'reference_pixels': int(truth.sum()), 'valid_prediction_fraction': float(prediction.mean()),
        'missing_prediction_on_reference_pixels': int((truth & ~prediction).sum()),
        'mae_m': float(absolute.mean()) if absolute.size else None,
        'p95_absolute_error_m': float(np.percentile(absolute, 95)) if absolute.size else None,
        'signed_error_m': float(error[common].mean()) if absolute.size else None,
        'near_pixels_under_4m': int(near.sum()), 'missed_near_pixels': int(missed.sum()),
        'missed_near_fraction': float(missed.sum()/near.sum()) if near.any() else None,
        'false_near_pixels': int(false_near.sum()), 'far_pixels': int(far.sum()),
        'false_near_fraction': float(false_near.sum()/far.sum()) if far.any() else None,
        'by_distance': {},
    }
    for low, high in ((.08, 1), (1, 2), (2, 4), (4, 8), (8, 20)):
        mask = common & (reference >= low) & (reference < high)
        result['by_distance'][f'{low:g}-{high:g}m'] = {
            'pixels': int(mask.sum()), 'mae_m': float(np.abs(error[mask]).mean()) if mask.any() else None}
    if intrinsics is not None and position is not None and quaternion is not None:
        points = depth_world_image(reference, intrinsics, position, quaternion)
        band = truth & (reference < 4.) & (np.abs(points[..., 2] - position[2]) <= .375)
        band_missed = band & (~prediction | (error > .5))
        result.update(flight_band_pixels=int(band.sum()), flight_band_missed_pixels=int(band_missed.sum()),
                      flight_band_missed_fraction=float(band_missed.sum()/band.sum()) if band.any() else None)
    return result


def depth_world_image(depth, intrinsics, position, quaternion):
    """Reference-only diagnostic projection; preserves pixel correspondence."""
    from depth_navigation import CAMERA_OFFSET, CAMERA_PITCH, rotation
    fx, fy, cx, cy = intrinsics
    v, u = np.mgrid[:depth.shape[0], :depth.shape[1]]
    rays = np.stack((np.ones_like(u), -(u-cx)/fx, -(v-cy)/fy), axis=-1)
    c, s = np.cos(CAMERA_PITCH), np.sin(CAMERA_PITCH)
    pitch = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    rot = rotation(quaternion)
    rays = rays @ pitch.T @ rot.T
    valid_depth = np.where(np.isfinite(depth), depth, np.nan)
    return valid_depth[..., None]*rays + CAMERA_OFFSET @ rot.T + np.asarray(position)


class EvaluationRecorder:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.events = (self.directory / 'events.jsonl').open('w', encoding='utf-8')

    def record(self, number, timestamp, rgb, estimated, reference, position, quaternion,
               processing_seconds, quality, **metadata):
        filename = f'frame-{number:06d}.npz'
        np.savez_compressed(self.directory / filename, rgb=rgb, estimated=estimated,
                            reference=reference if reference is not None else np.empty((0,), dtype=np.float32))
        self._write({'type': 'frame', 'number': number, 'sim_time': timestamp,
                     'file': filename, 'pose_xyz': position, 'pose_xyzw': quaternion,
                     'processing_seconds': processing_seconds, 'quality': quality,
                     'reference_synchronized': reference is not None, **metadata})

    def decision(self, number, position, target, detouring, replans, map_revision):
        self._write({'type': 'decision', 'frame': number, 'pose_xyz': position,
                     'target_xyz': target, 'detouring': detouring, 'replans': replans,
                     'map_revision': map_revision})

    def record_geometry(self, number, timestamp, rgb, points, objects, position, quaternion,
                        processing_seconds, quality, **metadata):
        filename = f'frame-{number:06d}.npz'
        np.savez_compressed(self.directory / filename, rgb=rgb, occupied_points=points)
        self._write({'type': 'frame', 'observation_kind': 'known_size_geometry', 'number': number,
                     'sim_time': timestamp, 'file': filename, 'objects': objects,
                     'pose_xyz': position, 'pose_xyzw': quaternion, 'quality': quality,
                     'processing_seconds': processing_seconds, 'reference_synchronized': False, **metadata})

    def stop(self, number, position, reason):
        self._write({'type': 'stop', 'frame': number, 'pose_xyz': position, 'reason': str(reason)})

    def coverage(self, action, *, rgb=None, **metadata):
        if rgb is not None:
            filename = f'ground-{metadata["frame"]:06d}.npz'
            np.savez_compressed(self.directory / filename, rgb=rgb)
            metadata['file'] = filename
        self._write({'type': 'coverage', 'action': action, **metadata})

    def _write(self, event):
        self.events.write(json.dumps(event) + '\n')
        self.events.flush()

    def close(self):
        self.events.close()


def summarize(directory, *, near_distance=2., miss_margin=.5):
    directory = Path(directory)
    frames, decisions, stops, coverage = [], [], [], []
    with (directory / 'events.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            event = json.loads(line)
            {'frame': frames, 'decision': decisions, 'stop': stops, 'coverage': coverage}[event['type']].append(event)
    errors, error_sum, error_count, misses, reference_near, false_near, reference_far, no_return_near, no_return_count, timings = [], 0., 0, 0, 0, 0, 0, 0, 0, []
    for event in frames:
        timings.append(event['processing_seconds'])
        if not event['reference_synchronized']:
            continue
        with np.load(directory / event['file']) as data:
            estimated, reference = data['estimated'], data['reference']
            if estimated.shape != reference.shape:
                continue
            valid = np.isfinite(reference) & (reference > .08) & (reference < 20)
            estimated_valid = np.isfinite(estimated) & (estimated > .08) & (estimated < 20)
            common = valid & estimated_valid
            frame_errors = np.abs(estimated[common] - reference[common]).ravel()
            error_sum += float(frame_errors.sum())
            error_count += frame_errors.size
            errors.extend(frame_errors[::16].tolist())  # bounded by frame count for p95
            near = valid & (reference < near_distance)
            reference_near += int(near.sum())
            misses += int((near & (~estimated_valid | (estimated > reference + miss_margin))).sum())
            far = valid & (reference >= near_distance + miss_margin)
            reference_far += int(far.sum())
            false_near += int((far & (estimated < near_distance)).sum())
            no_return = ~valid
            no_return_count += int(no_return.sum())
            no_return_near += int((no_return & (estimated < near_distance)).sum())
    return {'frames': len(frames), 'synchronized_reference_frames': sum(e['reference_synchronized'] for e in frames),
            'decisions': len(decisions), 'stops': stops,
            'absolute_error_mae_m': error_sum/error_count if error_count else None,
            'absolute_error_p95_m': float(np.percentile(errors, 95)) if errors else None,
            'near_obstacle_pixels': reference_near, 'missed_near_obstacle_pixels': misses,
            'missed_near_obstacle_fraction': misses/reference_near if reference_near else None,
            'far_pixels': reference_far, 'false_near_obstacle_pixels': false_near,
            'false_near_obstacle_fraction': false_near/reference_far if reference_far else None,
            'reference_no_return_pixels': no_return_count,
            'estimated_near_on_no_return_pixels': no_return_near,
            'processing_p95_s': float(np.percentile(timings, 95)) if timings else None,
            'localization_drift_m': None, 'localization_status': 'simulator pose only; camera localization not evaluated',
            'coverage': next((event for event in reversed(coverage) if event['action']=='complete'), None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.directory), indent=2))


if __name__ == '__main__':
    main()

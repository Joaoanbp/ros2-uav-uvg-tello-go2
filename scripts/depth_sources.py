"""Depth sources for the front camera. Reference depth is never an RGB input."""

from pathlib import Path

import numpy as np


class MetricIndoorDepth:
    """Depth Anything V2 Small, Hypersim metric checkpoint, run offboard."""

    name = 'rgb_depth_anything_v2_indoor_small'
    # The controlled fixture suite fails distance and missed-obstacle criteria.
    # Image quality and voxel-growth heuristics cannot certify model confidence.
    navigation_ready = False

    def __init__(self, repository, checkpoint, device='cpu', input_size=252, cpu_threads=4):
        import sys

        repository = Path(repository).expanduser().resolve()
        checkpoint = Path(checkpoint).expanduser().resolve()
        if not (repository / 'metric_depth/depth_anything_v2/dpt.py').is_file():
            raise FileNotFoundError(f'Depth Anything V2 metric_depth missing: {repository}')
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Indoor metric checkpoint missing: {checkpoint}')
        import torch
        if device == 'cpu':
            torch.set_num_threads(cpu_threads)
        sys.path.insert(0, str(repository / 'metric_depth'))
        from depth_anything_v2.dpt import DepthAnythingV2

        self.device = device
        self.input_size = input_size
        self.cpu_threads = cpu_threads
        self.checkpoint = str(checkpoint)
        self.model = DepthAnythingV2(encoder='vits', features=64,
                                    out_channels=[48, 96, 192, 384], max_depth=20)
        weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.model.load_state_dict(weights)
        self.model = self.model.to(device).eval()

    def infer(self, rgb):
        import torch
        import torch.nn.functional as functional

        with torch.inference_mode():
            # Same transforms as upstream image2tensor, with explicit device.
            # Upstream infer_image chooses a device globally, ignoring --depth-device.
            from depth_anything_v2.util.transform import Resize, NormalizeImage, PrepareForNet
            import cv2
            sample = {'image': rgb.astype(np.float32) / 255.}
            sample = Resize(self.input_size, self.input_size, resize_target=False,
                            keep_aspect_ratio=True, ensure_multiple_of=14,
                            resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC)(sample)
            sample = NormalizeImage(mean=[.485, .456, .406], std=[.229, .224, .225])(sample)
            sample = PrepareForNet()(sample)
            tensor = torch.from_numpy(sample['image']).unsqueeze(0).to(self.device)
            depth = self.model(tensor)
            depth = functional.interpolate(depth[:, None], rgb.shape[:2], mode='bilinear',
                                           align_corners=True)[0, 0].cpu().numpy()
        return np.asarray(depth, dtype=np.float32)


class GazeboDepth:
    name = 'gazebo_depth_baseline'
    navigation_ready = True

    def infer(self, depth):
        return depth


def validate_estimate(depth, shape, elapsed, *, max_delay=.75):
    """Fail closed on missing, stale, malformed or nonmetric-looking output."""
    depth = np.asarray(depth)
    if depth.shape != shape or depth.dtype.kind != 'f':
        raise RuntimeError('Profundidade estimada com tamanho ou tipo inválido; drone parado')
    valid = np.isfinite(depth) & (depth > .08) & (depth < 20)
    if valid.mean() < .90:
        raise RuntimeError('Profundidade estimada incompleta; drone parado')
    if elapsed > max_delay:
        raise RuntimeError('Estimativa de profundidade atrasada; drone parado')
    return depth


def quality_from_rgb(rgb):
    """Simple observable warning, not a calibrated model confidence score."""
    gray = np.mean(rgb.astype(np.float32), axis=2)
    horizontal = np.abs(np.diff(gray, axis=1)).mean() if gray.shape[1] > 1 else 0.
    vertical = np.abs(np.diff(gray, axis=0)).mean() if gray.shape[0] > 1 else 0.
    return {'luma_mean': float(gray.mean()), 'luma_std': float(gray.std()),
            'saturated_fraction': float(((gray < 8) | (gray > 247)).mean()),
            'mean_image_gradient': float((horizontal+vertical)/2)}

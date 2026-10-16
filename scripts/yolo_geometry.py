"""Metric ranging of a known upright cylinder from RGB detections and own pose.

The object dimensions are an explicit test prior. Runtime inputs are RGB boxes,
camera calibration and own pose; fixture labels use the projection helper offline.
"""

from dataclasses import dataclass, asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

PILLAR_RADIUS = .4
PILLAR_HEIGHT = 1.8


class TruncatedDetection(ValueError):
    pass


def camera_transform(position, quaternion):
    from depth_navigation import CAMERA_OFFSET, CAMERA_PITCH, rotation
    c, s = math.cos(CAMERA_PITCH), math.sin(CAMERA_PITCH)
    body = rotation(quaternion)
    basis = body @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.asarray(position) + body @ CAMERA_OFFSET, basis


def cylinder_bbox(center, intrinsics, position, quaternion,
                  radius=PILLAR_RADIUS, height=PILLAR_HEIGHT):
    """Project the known shape; also used OFFLINE by the training fixture."""
    origin, basis = camera_transform(position, quaternion)
    angles = np.linspace(0, 2*np.pi, 96, endpoint=False)
    ring = np.column_stack((np.cos(angles)*radius + center[0],
                            np.sin(angles)*radius + center[1], np.zeros(96)))
    points = np.vstack((ring, ring + [0, 0, height]))
    local = (points-origin) @ basis
    if np.min(local[:, 0]) <= .05:
        raise ValueError('Objeto cruza o plano da câmera')
    fx, fy, cx, cy = intrinsics
    u = cx - fx*local[:, 1]/local[:, 0]
    v = cy - fy*local[:, 2]/local[:, 0]
    return np.array([u.min(), v.min(), u.max(), v.max()])


@dataclass(frozen=True)
class ObjectEstimate:
    center: tuple
    uncertainty_m: float
    reprojection_error_px: float
    confidence: float
    bbox: tuple


def estimate_cylinder(box, intrinsics, position, quaternion, dimensions,
                      *, radius=PILLAR_RADIUS, height=PILLAR_HEIGHT, confidence=1.):
    """Fit image bounds to the known shape, grounded at z=0.

    Top-clipped detections retain three independent image constraints. Lateral
    and floor-contact truncation are rejected, never converted to free space.
    """
    from scipy.optimize import least_squares
    box = np.asarray(box, dtype=float)
    image_h, image_w = dimensions
    if not np.all(np.isfinite(box)) or np.any(box[2:] <= box[:2]):
        raise ValueError('Caixa YOLO inválida')
    if box[0] < 3 or box[2] > image_w-3 or box[3] > image_h-3:
        raise TruncatedDetection('Contato com o chão ou lateral do objeto fora da imagem')
    fx, fy, cx, cy = intrinsics
    origin, basis = camera_transform(position, quaternion)
    ray = basis @ np.array([1., -((box[0]+box[2])/2-cx)/fx, -(box[3]-cy)/fy])
    if ray[2] >= -.03 or origin[2] <= .2:
        raise ValueError('Contato com o chão não permite projeção métrica')
    ground = origin - origin[2]/ray[2]*ray
    direction = ground[:2]-origin[:2]
    initial = ground[:2] + radius*direction/np.linalg.norm(direction)
    active = [0, 2, 3] if box[1] < 3 else [0, 1, 2, 3]
    def residual(center):
        return (cylinder_bbox(center, intrinsics, position, quaternion, radius, height)-box)[active]
    fit = least_squares(residual, initial, bounds=(initial-2., initial+2.),
                        loss='soft_l1', f_scale=3., max_nfev=30)
    reprojection = float(np.max(np.abs(residual(fit.x))))
    if not fit.success or reprojection > 12:
        raise ValueError(f'Detecção inconsistente com o objeto conhecido ({reprojection:.1f} px)')
    covariance = np.linalg.pinv(fit.jac.T @ fit.jac)*4.**2
    uncertainty = max(.18, 2*math.sqrt(max(np.linalg.eigvalsh(covariance).max(), 0.)))
    distance = math.dist(fit.x, origin[:2])
    if not 1.1 < distance < 12 or uncertainty > 1.2:
        raise ValueError('Distância ou incerteza fora do domínio validado')
    return ObjectEstimate(tuple(float(v) for v in fit.x), uncertainty, reprojection,
                          float(confidence), tuple(float(v) for v in box))


def occupied_points(estimate, altitude, radius=PILLAR_RADIUS, height=PILLAR_HEIGHT):
    """Known-size occupancy prior, including uncertainty; not dense RGB depth."""
    if altitude > height+.3:
        return np.empty((0, 3))
    extent = radius + estimate.uncertainty_m
    axis = np.arange(-extent, extent+.06, .10)
    xx, yy = np.meshgrid(axis, axis)
    mask = xx*xx+yy*yy <= extent*extent
    points = np.column_stack((xx[mask]+estimate.center[0], yy[mask]+estimate.center[1],
                              np.full(int(mask.sum()), min(altitude, height))))
    # SurfaceMap's two-observation-per-voxel filter is for dense depth pixels.
    # These are explicit model occupancy samples, so retain every sample.
    return np.repeat(points, 2, axis=0)


def matches_existing_view(anchor, box, intrinsics, position, quaternion, dimensions):
    """Associate an RGB view with an already ranged static object, without new ranging."""
    try:
        projected = cylinder_bbox(anchor.center, intrinsics, position, quaternion)
    except ValueError:
        return False
    h, w = dimensions
    projected = np.clip(projected, [0,0,0,0], [w,h,w,h])
    box = np.asarray(box)
    overlap = np.maximum(0, np.minimum(projected[2:], box[2:])-np.maximum(projected[:2], box[:2]))
    intersection = float(np.prod(overlap))
    area_a = float(np.prod(np.maximum(0, projected[2:]-projected[:2])))
    area_b = float(np.prod(np.maximum(0, box[2:]-box[:2])))
    return intersection / max(area_a+area_b-intersection, 1e-6) >= .5


class YoloKnownPillar:
    name = 'rgb_yolo_known_pillar'
    geometry_mode = True

    def __init__(self, checkpoint, profile, *, device='cpu'):
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(4)
        self.checkpoint = Path(checkpoint).resolve()
        self.profile = json.loads(Path(profile).read_text())
        actual_hash = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        if self.profile['checkpoint_sha256'] != actual_hash:
            raise ValueError('Checkpoint YOLO não corresponde ao perfil validado')
        if self.profile['radius_m'] != PILLAR_RADIUS or self.profile['height_m'] != PILLAR_HEIGHT:
            raise ValueError('Dimensões do perfil não correspondem ao objeto conhecido')
        self.navigation_ready = bool(self.profile.get('approved_controlled_simulation', False))
        self.model = YOLO(str(self.checkpoint))
        if self.model.names != {0: 'calibration_pillar'}:
            raise ValueError('Detector não foi treinado para o pilar conhecido')
        self.device = device
        self.input_size = self.profile['input_size']
        self.nms_iou = self.profile.get('nms_iou', .5)
        self.last_objects = []
        self.anchor = None
        # Initialize/fuse the detector before takeoff, outside frame freshness checks.
        self.model.predict(np.full((480,640,3),128,dtype=np.uint8), imgsz=self.input_size,
                           conf=.30, iou=self.nms_iou, device=self.device, verbose=False)

    def observe_rgb(self, rgb, intrinsics, position, quaternion):
        result = self.model.predict(rgb[:, :, ::-1].copy(), imgsz=self.input_size,
                                    conf=.30, iou=self.nms_iou, device=self.device, verbose=False)[0]
        objects, rejected, tracked = [], [], []
        for box, confidence, label in zip(result.boxes.xyxy.cpu().numpy(),
                                          result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()):
            if int(label) != 0:
                continue
            # A ranged static object is a track. Associate image boxes before
            # requesting another metric fit or counting world obstacles.
            if self.anchor is not None and matches_existing_view(
                    self.anchor, box, intrinsics, position, quaternion, rgb.shape[:2]):
                tracked.append((box, confidence))
                continue
            try:
                objects.append(estimate_cylinder(box, intrinsics, position, quaternion, rgb.shape[:2],
                                                 confidence=confidence))
            except ValueError as error:
                if box[0] < rgb.shape[1]*.6 and box[2] > rgb.shape[1]*.4:
                    rejected.append(str(error))
        # A central detection with unresolved range cannot authorize a step.
        if rejected:
            raise RuntimeError('YOLO viu objeto sem distância confiável; drone parado: ' + rejected[0])
        if len(objects) > 1:
            raise RuntimeError('Protótipo validado para um pilar; múltiplas detecções ambíguas')
        if tracked and not objects:
            # All these boxes have already been associated with the same
            # previously mapped object. They are duplicate observations, not
            # multiple world obstacles; keep the strongest visible outline.
            box, confidence = max(tracked, key=lambda item: item[1])
            partial = box[0] < 3 or box[1] < 3 or box[2] > rgb.shape[1]-3 or box[3] > rgb.shape[0]-3
            self.last_objects = [{**asdict(self.anchor), 'bbox': tuple(float(v) for v in box),
                                  'confidence': float(confidence), 'measurement_center': None,
                                  'range_status': 'tracked_partial' if partial else 'tracked_static'}]
            return occupied_points(self.anchor, position[2])
        if objects:
            measurement = objects[0]
            if self.anchor is None:
                self.anchor = measurement
            elif math.dist(measurement.center, self.anchor.center) > self.anchor.uncertainty_m + measurement.uncertainty_m + .10:
                raise RuntimeError('Posição visual incompatível com o pilar estático; drone parado')
            # Keep the first conservative full footprint. Frame-to-frame box
            # jitter must not push the occupied boundary into an ongoing detour.
            self.last_objects = [{**asdict(self.anchor), 'bbox': measurement.bbox,
                                  'confidence': measurement.confidence,
                                  'measurement_center': measurement.center, 'range_status': 'measured'}]
            return occupied_points(self.anchor, position[2])
        self.last_objects = []
        return np.empty((0, 3))

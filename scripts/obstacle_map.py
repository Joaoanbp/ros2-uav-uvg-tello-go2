"""Live obstacle envelopes from Gazebo scene/poses (simulator truth, not vision)."""

from dataclasses import dataclass
import itertools
import threading
import time

import numpy as np


def pose_matrix(pose):
    q = pose.orientation
    x, y, z, w = q.x, q.y, q.z, q.w
    norm = x*x + y*y + z*z + w*w
    matrix = np.eye(4)
    if norm > 1e-16:
        s = 2 / norm
        matrix[:3, :3] = [
            [1-s*(y*y+z*z), s*(x*y-z*w), s*(x*z+y*w)],
            [s*(x*y+z*w), 1-s*(x*x+z*z), s*(y*z-x*w)],
            [s*(x*z-y*w), s*(y*z+x*w), 1-s*(x*x+y*y)],
        ]
    matrix[:3, 3] = (pose.position.x, pose.position.y, pose.position.z)
    return matrix


def geometry_corners(geometry):
    """Conservative local boxes around supported primitive geometries."""
    if geometry.HasField('box'):
        size = geometry.box.size
        half = np.array([size.x, size.y, size.z]) / 2
    elif geometry.HasField('sphere'):
        half = np.repeat(geometry.sphere.radius, 3)
    elif geometry.HasField('ellipsoid'):
        radii = geometry.ellipsoid.radii
        half = np.array([radii.x, radii.y, radii.z])
    else:
        kind = next((k for k in ('cylinder', 'cone', 'capsule') if geometry.HasField(k)), None)
        if kind is None:
            return None
        shape = getattr(geometry, kind)
        half = np.array([shape.radius, shape.radius,
                         shape.length / 2 + (shape.radius if kind == 'capsule' else 0)])
    if not np.all(np.isfinite(half)) or np.any(half <= 0):
        return None
    return np.array(list(itertools.product((-1, 1), repeat=3))) * half


def transform_points(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def model_points(model, parent=None, include_pose=True):
    """Walk nested model/link geometry; prefer collision data when provided."""
    matrix = np.eye(4) if parent is None else parent
    if include_pose:
        matrix = matrix @ pose_matrix(model.pose)
    points = []
    unknown = False
    for link in model.link:
        link_matrix = matrix @ pose_matrix(link.pose)
        for item in (link.collision if link.collision else link.visual):
            if item.geometry.HasField('plane'):
                continue
            corners = geometry_corners(item.geometry)
            if corners is None:
                unknown = True
                continue
            points.extend(transform_points(corners, link_matrix @ pose_matrix(item.pose)))
    for child in model.model:
        child_points, child_unknown = model_points(child, matrix)
        points.extend(child_points)
        unknown |= child_unknown
    return points, unknown


@dataclass(frozen=True)
class Obstacle:
    entity_id: int
    name: str
    outline: tuple
    z_min: float
    z_max: float
    approximate: bool


def scene_geometry(scene, excluded=('tello', 'ground_plane')):
    """Convert protobufs to plain geometry before crossing thread boundaries."""
    result = []
    for model in scene.model:
        if model.name in excluded or model.deleted:
            continue
        points, unknown = model_points(model, include_pose=False)
        if not points:
            if not unknown:
                continue  # Ground planes or models with no geometry.
            points = [(0, 0, 0)]
        result.append((model.id, model.name, np.array(points), pose_matrix(model.pose), unknown))
    return result


def geometry_obstacles(models, poses=None):
    result = []
    poses = poses or {}
    for entity_id, name, local_points, matrix, unknown in models:
        points = transform_points(local_points, poses.get(entity_id, matrix))
        if len(points) == 1:
            outline = ((points[0, 0], points[0, 1]),)
            low = high = points[0, 2]
        else:
            low_xyz, high_xyz = np.min(points, axis=0), np.max(points, axis=0)
            x0, y0, low = low_xyz
            x1, y1, high = high_xyz
            outline = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
        result.append(Obstacle(entity_id, name, outline, float(low), float(high), unknown))
    return result


def scene_obstacles(scene, poses=None):
    matrices = {key: pose_matrix(pose) for key, pose in (poses or {}).items()}
    return geometry_obstacles(scene_geometry(scene), matrices)


class ObstacleTracker:
    """Poll full scene for spawn/removal; consume live root-model pose updates."""

    def __init__(self, world='tello_only'):
        self.world = world
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.models = []
        self.poses = {}
        self.model_ids = set()
        self.last_scene = None
        self.error = None
        self.worker = None

    def replace_scene(self, scene, requested_at=None):
        models = scene_geometry(scene)
        with self.lock:
            self.models = models
            self.model_ids = {model[0] for model in models}
            self.poses = {key: value for key, value in self.poses.items()
                          if key in self.model_ids and requested_at is not None and value[1] >= requested_at}
            self.last_scene = time.monotonic()
            self.error = None

    def receive_poses(self, message):
        with self.lock:
            for pose in message.pose:
                if pose.id in self.model_ids:
                    self.poses[pose.id] = (pose_matrix(pose), time.monotonic())

    def snapshot(self):
        with self.lock:
            models = self.models
            poses = {key: value[0] for key, value in self.poses.items()}
            last_scene, error = self.last_scene, self.error
        return geometry_obstacles(models, poses), last_scene, error

    def start(self):
        self.worker = threading.Thread(target=self._poll, daemon=True)
        self.worker.start()

    def _poll(self):
        from gz.msgs10.empty_pb2 import Empty
        from gz.msgs10.scene_pb2 import Scene
        from gz.transport13 import Node
        from google.protobuf.message_factory import MessageFactory

        node = Node()
        # The Ubuntu protobuf C++ backend can lazily register nested types.
        # Keep a factory alive with the complete scene schema registered before
        # deserializing a real response (including models, links and visuals).
        factory = MessageFactory()
        pending = [Scene.DESCRIPTOR]
        registered = set()
        prototypes = []
        while pending:
            descriptor = pending.pop()
            if descriptor.full_name in registered:
                continue
            registered.add(descriptor.full_name)
            prototypes.append(factory.GetPrototype(descriptor))
            pending.extend(field.message_type for field in descriptor.fields if field.message_type)
        while not self.stop_event.is_set():
            requested_at = time.monotonic()
            try:
                executed, scene = node.request(
                    f'/world/{self.world}/scene/info', Empty(), Empty, Scene, 1000)
                if executed:
                    self.replace_scene(scene, requested_at)
                else:
                    with self.lock:
                        self.error = 'serviço de cena sem resposta'
            except Exception as error:
                with self.lock:
                    self.error = str(error)
            self.stop_event.wait(1.0)

    def close(self):
        self.stop_event.set()
        if self.worker:
            self.worker.join(timeout=2)


class ObstacleLayer:
    """Matplotlib overlay; all artists are modified only on the GUI thread."""

    def __init__(self, ax, transform):
        from matplotlib.patches import Patch

        self.ax = ax
        self.transform = transform
        self.artists = {}
        self.legend = Patch(facecolor='tab:red', edgecolor='firebrick', alpha=0.25,
                            label='Obstáculos — dados do Gazebo')
        self.status = ax.text(0.02, 0.98, 'Aguardando obstáculos do Gazebo…',
                              transform=ax.transAxes, va='top', fontsize=8,
                              bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'none'})

    def update(self, obstacles, last_scene, error):
        from matplotlib.patches import Polygon

        current = {obstacle.entity_id for obstacle in obstacles}
        for key in set(self.artists) - current:
            for artist in self.artists.pop(key):
                artist.remove()
        points = []
        for obstacle in obstacles:
            outline = np.array(obstacle.outline)
            if obstacle.entity_id not in self.artists:
                patch = Polygon(outline, closed=True, facecolor='tab:red', edgecolor='firebrick',
                                alpha=0.25, transform=self.transform, zorder=3)
                self.ax.add_patch(patch)
                marker, = self.ax.plot([], [], 'x', color='firebrick', transform=self.transform, zorder=4)
                text = self.ax.text(0, 0, '', color='firebrick', fontsize=8,
                                    transform=self.transform, zorder=5)
                self.artists[obstacle.entity_id] = patch, marker, text
            patch, marker, text = self.artists[obstacle.entity_id]
            patch.set_xy(outline)
            center = outline.mean(axis=0)
            marker.set_data([center[0]], [center[1]])
            text.set_position((center[0], outline[:, 1].max() + 0.15))
            details = 'geometria parcial/desconhecida' if obstacle.approximate else f'Z {obstacle.z_min:.2f}–{obstacle.z_max:.2f} m'
            text.set_text(f'{obstacle.name}\n{details}')
            points.extend(obstacle.outline)
        age = None if last_scene is None else time.monotonic() - last_scene
        if age is None:
            label = 'Aguardando cena do Gazebo — obstáculos indisponíveis'
        elif age > 5 or error:
            label = f'Obstáculos: última cena há {age:.1f} s (dados desatualizados)'
        else:
            label = f'{len(obstacles)} obstáculo(s) · fonte: Gazebo · envoltórias XY'
        self.status.set_text(label)
        return points

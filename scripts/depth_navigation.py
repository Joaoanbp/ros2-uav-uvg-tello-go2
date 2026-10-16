"""Image-based mapping and receding-horizon navigation. No scene/model queries.

Inputs: depth or known-object RGB geometry, calibration, and own pose only.
The map starts empty; known-object occupancy includes the explicit shape prior.
"""

from collections import deque
import heapq
import json
import math
import threading
import time

import numpy as np
from roi_plan import distance_to_segment, point_inside, segment_inside, cross, subtract
from depth_sources import GazeboDepth, quality_from_rgb, validate_estimate
from drone_camera import decode_image

DEPTH_TOPIC = '/tello/camera/front/depth_image'
RGB_TOPIC = '/tello/camera/front/image'
INFO_TOPIC = '/tello/camera/front/camera_info'
MAP_TOPIC = '/tello/perception/map'
# Sensor-to-body calibration, matching model.sdf, not obstacle geometry.
CAMERA_OFFSET = np.array([0.065, 0., -0.002])
CAMERA_PITCH = 0.436332


def decode_depth(message):
    if message.pixel_format_type != 13:  # gz::msgs::PixelFormatType::R_FLOAT32
        raise ValueError('Profundidade deve ser R_FLOAT32 em metros')
    w, h = message.width, message.height
    step = message.step or w * 4
    if w <= 0 or h <= 0 or step < w * 4 or step % 4 or len(message.data) < step * h:
        raise ValueError('Imagem de profundidade truncada ou stride inválido')
    return np.frombuffer(message.data, dtype='<f4', count=step * h // 4).reshape(h, step // 4)[:, :w].copy()


def rotation(quaternion):
    x, y, z, w = quaternion
    norm = math.sqrt(x*x + y*y + z*z + w*w)
    if norm < 1e-8:
        raise ValueError('Orientação do drone inválida')
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def project_depth(depth, intrinsics, position, quaternion, stride=4, max_range=8.):
    """Optical pixels -> Gazebo camera (+X forward) -> body -> world."""
    fx, fy, cx, cy = intrinsics
    if min(fx, fy) <= 0:
        raise ValueError('Calibração inválida')
    v, u = np.mgrid[0:depth.shape[0]:stride, 0:depth.shape[1]:stride]
    d = depth[::stride, ::stride]
    valid = np.isfinite(d) & (d > 0.08) & (d < max_range)
    d, u, v = d[valid], u[valid], v[valid]
    points = np.column_stack((d, -(u-cx)*d/fx, -(v-cy)*d/fy))
    c, s = math.cos(CAMERA_PITCH), math.sin(CAMERA_PITCH)
    pitch = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    body = points @ pitch.T + CAMERA_OFFSET
    return body @ rotation(quaternion).T + np.asarray(position)


class SurfaceMap:
    def __init__(self, altitude, resolution=0.15):
        self.altitude = altitude
        self.resolution = resolution
        self.voxels = set()
        self.revision = 0

    def integrate(self, points):
        # Reject the horizontal ground plane in the localized world frame.
        # No tree class, color, shape, name, location or dimensions are used.
        points = points[(points[:, 2] > 0.20) & (points[:, 2] < 10)]
        keys, counts = np.unique(np.floor(points / self.resolution).astype(int), axis=0, return_counts=True)
        before = len(self.voxels)
        self.voxels.update(tuple(p) for p in keys[counts >= 2])
        if len(self.voxels) != before:
            self.revision += 1

    def points(self):
        return (np.asarray(sorted(self.voxels), dtype=float).reshape(-1, 3) + .5) * self.resolution

    def collision_points(self):
        points = self.points()
        return points[np.abs(points[:, 2] - self.altitude) <= .30 + self.resolution / 2, :2]


class GridPlanner:
    """A* on measured occupied cells, inflated for the drone and margin.

    Unknown cells are candidates, not certified free space: the executor must
    look in the direction of travel and recheck after each short translation.
    """
    def __init__(self, bounds, resolution=.20, clearance=.50):
        self.bounds = bounds
        self.resolution = resolution
        self.clearance = clearance
        self.blocked = set()
        self.region = None
        self.region_cells = {}
        self.region_convex = True

    def set_region(self, polygon):
        self.region = polygon
        self.region_cells = {}
        if polygon is not None:
            turns = [cross(subtract(b,a),subtract(c,b)) for a,b,c in
                     zip(polygon, polygon[1:]+polygon[:1], polygon[2:]+polygon[:2])]
            self.region_convex = all(v>=-1e-8 for v in turns) or all(v<=1e-8 for v in turns)

    def position_inside(self, point):
        xmin, xmax, ymin, ymax = self.bounds
        return (xmin <= point[0] <= xmax and ymin <= point[1] <= ymax
                and (self.region is None or point_inside(point[:2],self.region)))

    def cell(self, point):
        return tuple(round(float(v) / self.resolution) for v in point[:2])

    def point(self, cell):
        return tuple(v * self.resolution for v in cell)

    def update(self, points):
        self.blocked.clear()
        radius = math.ceil((self.clearance + self.resolution) / self.resolution)
        for point in points:
            cx, cy = self.cell(point)
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    key = (cx + dx, cy + dy)
                    if math.dist(self.point(key), point) <= self.clearance + self.resolution * .71:
                        self.blocked.add(key)

    def inside(self, cell):
        x, y = self.point(cell)
        xmin, xmax, ymin, ymax = self.bounds
        if not xmin <= x <= xmax or not ymin <= y <= ymax:
            return False
        if self.region is not None and cell not in self.region_cells:
            self.region_cells[cell] = point_inside((x,y), self.region)
        return self.region is None or self.region_cells[cell]

    def clear(self, start, end):
        # Exact segment/square intersections. Sampling can miss a cell corner
        # on a long segment that a shorter execution step subsequently hits.
        if self.region is not None:
            if self.region_convex:
                if not point_inside(start[:2],self.region) or not point_inside(end[:2],self.region):
                    return False
            elif not segment_inside(start[:2],end[:2],self.region):
                return False
        half = self.resolution / 2 + 1e-9
        for cell in self.blocked:
            center = self.point(cell)
            enter, leave = 0., 1.
            for axis in (0, 1):
                low, high = center[axis]-half, center[axis]+half
                delta = end[axis]-start[axis]
                if abs(delta) < 1e-12:
                    if not low <= start[axis] <= high:
                        enter, leave = 1., 0.
                        break
                else:
                    a, b = (low-start[axis])/delta, (high-start[axis])/delta
                    enter, leave = max(enter, min(a, b)), min(leave, max(a, b))
            if enter <= leave:
                return False
        return True

    def plan(self, start, goal, *, preferred=(), visited=None):
        begin, finish = self.cell(start), self.cell(goal)
        if self.region is not None:
            if not self.position_inside(start) or not self.position_inside(goal):
                return None
            if self.clear(start,goal):
                return [tuple(start[:2]),tuple(goal[:2])]
            def connection(point):
                cx, cy = self.cell(point)
                nearby = sorted(((cx+dx,cy+dy) for dx in range(-2,3) for dy in range(-2,3)),
                                key=lambda cell: math.dist(point[:2],self.point(cell)))
                return next((cell for cell in nearby if cell not in self.blocked and self.inside(cell)
                             and self.clear(point,self.point(cell))), None)
            begin, finish = connection(start), connection(goal)
            if begin is None or finish is None:
                return None
        if begin in self.blocked or finish in self.blocked:
            return None
        if not self.inside(begin) or not self.inside(finish):
            return None
        if self.clear(start, goal):
            return [tuple(start[:2]), tuple(goal[:2])]
        queue = [(math.dist(begin, finish), 0., begin)]
        costs, parent = {begin: 0.}, {}
        penalties = {}
        while queue:
            _, cost, current = heapq.heappop(queue)
            if cost > costs[current]:
                continue
            if current == finish:
                cells = [current]
                while current != begin:
                    current = parent[current]
                    cells.append(current)
                raw = [tuple(start[:2]), *[self.point(c) for c in reversed(cells)], tuple(goal[:2])]
                # Greedy line-of-sight simplification, checked against inflation.
                result, index = [raw[0]], 0
                while index < len(raw) - 1:
                    next_index = len(raw) - 1
                    while next_index > index + 1 and not self.clear(raw[index], raw[next_index]):
                        next_index -= 1
                    result.append(raw[next_index])
                    index = next_index
                return result
            for dx, dy in ((1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)):
                neighbor = current[0]+dx, current[1]+dy
                if neighbor in self.blocked or not self.inside(neighbor):
                    continue
                if dx and dy and ((current[0]+dx, current[1]) in self.blocked or
                                  (current[0], current[1]+dy) in self.blocked):
                    continue
                if self.region is not None and not self.region_convex and not segment_inside(
                        self.point(current),self.point(neighbor),self.region):
                    continue
                if neighbor not in penalties:
                    point = self.point(neighbor)
                    continuity = min((distance_to_segment(point, a, b)
                                      for a, b in zip(preferred, preferred[1:])), default=0.)
                    penalties[neighbor] = (1.5 * min(continuity, 2.) +
                                           .8 * min((visited or {}).get(neighbor, 0), 3))
                candidate = cost + math.hypot(dx, dy) * (1 + penalties[neighbor])
                if candidate < costs.get(neighbor, math.inf):
                    costs[neighbor], parent[neighbor] = candidate, current
                    heapq.heappush(queue, (candidate + math.dist(neighbor, finish), candidate, neighbor))
        return None


class LocalRoute:
    """Keep a safe detour until reached, then rejoin the original leg ahead.

    New observations invalidate a route only when they actually obstruct it.
    Replanning prefers the previous corridor and penalizes already flown cells;
    these are soft costs, so a newly blocked side can still be abandoned.
    """
    def __init__(self, start, goal, visits, *, preserve_leg=True, smooth=False):
        self.start, self.goal = tuple(start[:2]), tuple(goal[:2])
        self.remaining = [self.goal]
        self.visits = visits
        self.revisions = 0
        self.preserve_leg, self.smooth = preserve_leg, smooth

    def update(self, planner, position):
        position = tuple(position[:2])
        if not planner.position_inside(position) or not planner.position_inside(self.goal):
            return None
        while len(self.remaining) > 1 and math.dist(position, self.remaining[0]) < .025:
            self.remaining.pop(0)
        previous = [position, *self.remaining]
        if all(planner.clear(a, b) for a, b in zip(previous, previous[1:])):
            return previous
        if not planner.clear(position, position) or not planner.clear(self.goal, self.goal):
            return None
        # Find the earliest clear suffix on the original planned leg. This
        # preserves the remaining sweep instead of taking a chord to its end.
        count = max(1, math.ceil(math.dist(self.start, self.goal) / (planner.resolution / 2)))
        rejoin = self.goal
        delta = (self.goal[0]-self.start[0], self.goal[1]-self.start[1])
        length2 = sum(v*v for v in delta)
        progress = sum((position[i]-self.start[i])*delta[i] for i in (0, 1)) / max(length2, 1e-12)
        for index in (range(count-1, max(-1, math.floor(progress*count)), -1) if self.preserve_leg else ()):
            point = tuple(self.start[i] + delta[i]*index/count for i in (0, 1))
            if not planner.clear(point, rejoin):
                break
            rejoin = point
        path = planner.plan(position, rejoin, preferred=previous if self.revisions else (), visited=self.visits)
        if path is None:
            return None
        if math.dist(path[-1], self.goal) > .001:
            path.append(self.goal)
        if self.smooth:
            from adaptive_coverage import smooth_path
            path = smooth_path(path, planner)
        self.remaining = path[1:]
        self.revisions += 1
        return path


def stamp(message):
    return message.header.stamp.sec + message.header.stamp.nsec * 1e-9


def synchronized_sample(samples, timestamp, tolerance=.005):
    """RGB and reference from one RGB-D render must share a sensor timestamp."""
    sample = min(samples, key=lambda item: abs(item[0]-timestamp)) if samples else None
    return sample if sample is not None and abs(sample[0]-timestamp) <= tolerance else None


def message_prototypes(*message_types):
    """Retain nested C++ protobuf classes used by transport callback threads."""
    from google.protobuf.message_factory import MessageFactory
    factory = MessageFactory()
    prototypes, registered = [], set()
    descriptors = [message.DESCRIPTOR for message in message_types]
    while descriptors:
        descriptor = descriptors.pop()
        if descriptor.full_name in registered:
            continue
        registered.add(descriptor.full_name)
        prototypes.append(factory.GetPrototype(descriptor))
        descriptors.extend(f.message_type for f in descriptor.fields if f.message_type)
    return factory, prototypes


class DepthObserver:
    """Synchronize a selected depth source with own pose.

    Reference depth is subscribed only when an evaluation recorder is supplied.
    It is never passed to the mapper or motion planner in RGB mode.
    """
    def __init__(self, node, source=None, recorder=None):
        from gz.msgs10.image_pb2 import Image
        from gz.msgs10.camera_info_pb2 import CameraInfo
        from gz.msgs10.pose_v_pb2 import Pose_V
        from gz.transport13 import SubscribeOptions
        # Gazebo's Python transport needs nested C++ protobuf prototypes retained
        # before callbacks run on its worker threads (Ubuntu protobuf 3.12).
        self.factory, self.prototypes = message_prototypes(Image, CameraInfo, Pose_V)
        self.node = node
        self.source = source or GazeboDepth()
        self.recorder = recorder
        self.rgb_mode = self.source.name != GazeboDepth.name
        self.lock = threading.Lock()
        self.poses = deque(maxlen=1000)
        self.frames = deque(maxlen=4)
        self.references = deque(maxlen=8)
        self.rgb_references = deque(maxlen=8)
        self.intrinsics = None
        self.dimensions = None
        self.count = 0
        self.last_frame = 0
        self.error = None
        self.geometry_camera_checked = False
        self.ground_frames = deque(maxlen=8)
        self.ground_intrinsics = self.ground_dimensions = None
        self.last_sample = None
        subscriptions = [(Image, RGB_TOPIC if self.rgb_mode else DEPTH_TOPIC,
                          self.receive_rgb if self.rgb_mode else self.receive_depth),
                (CameraInfo, INFO_TOPIC, self.receive_info),
                (Pose_V, '/world/tello_only/pose/info', self.receive_pose)]
        if self.rgb_mode and recorder is not None and not getattr(self.source, 'geometry_mode', False):
            subscriptions.append((Image, DEPTH_TOPIC, self.receive_reference))
        if not self.rgb_mode and recorder is not None:
            subscriptions.append((Image, RGB_TOPIC, self.receive_rgb_reference))
        self.subscribed_topics = tuple(topic for _, topic, _ in subscriptions)
        for message_type, topic, callback in subscriptions:
            options = SubscribeOptions()
            options.msgs_per_sec = 30
            if not node.subscribe(message_type, topic, callback, options):
                raise RuntimeError(f'Falha ao assinar {topic}')

    def enable_ground(self):
        from gz.msgs10.image_pb2 import Image
        from gz.msgs10.camera_info_pb2 import CameraInfo
        subscriptions = [(Image, '/tello/camera/down/image', self.receive_ground),
                         (CameraInfo, '/tello/camera/down/camera_info', self.receive_ground_info)]
        for message_type, topic, callback in subscriptions:
            if not self.node.subscribe(message_type, topic, callback):
                raise RuntimeError(f'Falha ao assinar {topic}')
        self.subscribed_topics += tuple(topic for _, topic, _ in subscriptions)

    def receive_ground(self, message):
        try:
            frame = decode_image(message)
        except ValueError:
            return
        with self.lock:
            self.ground_frames.append((stamp(message), time.monotonic(), frame))

    def receive_ground_info(self, message):
        k = list(message.intrinsics.k)
        if len(k) == 9 and min(k[0], k[4]) > 0:
            with self.lock:
                self.ground_intrinsics = (k[0], k[4], k[2], k[5])
                self.ground_dimensions = (message.height, message.width)

    def ground_view(self, timeout=1.):
        """Use an actual downward frame paired with the last measured front pose."""
        ts, position, quaternion = self.last_sample
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            with self.lock:
                samples = list(self.ground_frames)
                poses = list(self.poses)
                intrinsics, dimensions = self.ground_intrinsics, self.ground_dimensions
            sample = min(samples, key=lambda s: abs(s[0]-ts)) if samples else None
            pose = min(poses, key=lambda p: abs(p[0]-sample[0])) if poses and sample else None
            if (sample is not None and intrinsics is not None and abs(sample[0]-ts) <= .08
                    and time.monotonic()-sample[1] <= .75 and sample[2].shape[:2] == dimensions
                    and pose is not None and abs(pose[0]-sample[0]) <= .08
                    and math.dist(pose[1], position) <= .035
                    and abs(float(np.dot(pose[2],quaternion))) >= math.cos(.035/2)):
                quality = quality_from_rgb(sample[2])
                if (quality['luma_mean'] < 15 or quality['luma_mean'] > 240 or
                        quality['saturated_fraction'] > .75):
                    raise RuntimeError('RGB inferior inválido para cobertura; drone parado')
                return (pose[1], pose[2], intrinsics, dimensions), sample[0], sample[2]
            time.sleep(.01)
        raise RuntimeError('Sem RGB inferior/calibração sincronizados; cobertura não confirmada, drone parado')

    def receive_info(self, message):
        k = list(message.intrinsics.k)
        if len(k) == 9 and k[0] > 0 and k[4] > 0:
            with self.lock:
                self.intrinsics = (k[0], k[4], k[2], k[5])
                self.dimensions = (message.height, message.width)

    def receive_pose(self, message):
        for pose in message.pose:
            if pose.name != 'tello':
                continue
            p, q = pose.position, pose.orientation
            with self.lock:
                self.poses.append((stamp(message), (p.x, p.y, p.z), (q.x, q.y, q.z, q.w)))

    def receive_depth(self, message):
        try:
            frame = decode_depth(message)
        except ValueError as error:
            with self.lock:
                self.error = str(error)
            return
        with self.lock:
            self.count += 1
            self.frames.append((self.count, stamp(message), time.monotonic(), frame))
            self.error = None

    def receive_rgb(self, message):
        try:
            frame = decode_image(message)
        except ValueError as error:
            with self.lock:
                self.error = str(error)
            return
        with self.lock:
            self.count += 1
            self.frames.append((self.count, stamp(message), time.monotonic(), frame))
            self.error = None

    def receive_reference(self, message):
        try:
            frame = decode_depth(message)
        except ValueError:
            return
        with self.lock:
            self.references.append((stamp(message), frame))

    def receive_rgb_reference(self, message):
        try:
            frame = decode_image(message)
        except ValueError:
            return
        with self.lock:
            self.rgb_references.append((stamp(message), frame))

    def observe(self, position, yaw, timeout=8):
        # Two new frames after arrival avoid integrating a pre-command render.
        with self.lock:
            after = self.count + 2
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                frames, poses = list(self.frames), list(self.poses)
                intrinsics, dimensions = self.intrinsics, self.dimensions
            for number, ts, received, frame in reversed(frames):
                if number < after or time.monotonic() - received > .75 or not poses or intrinsics is None:
                    continue
                ps, pos, quat = min(poses, key=lambda sample: abs(sample[0] - ts))
                measured_yaw = math.atan2(rotation(quat)[1, 0], rotation(quat)[0, 0])
                angle_error = math.atan2(math.sin(measured_yaw-yaw), math.cos(measured_yaw-yaw))
                if abs(ps-ts) > .08 or math.dist(pos, position) > .035 or abs(angle_error) > .035:
                    continue
                if frame.shape[:2] != dimensions:
                    raise RuntimeError('Imagem e calibração incompatíveis; drone parado')
                self.last_frame = number
                self.last_sample = (ts, pos, quat)
                began = time.monotonic()
                if getattr(self.source, 'geometry_mode', False):
                    if not self.geometry_camera_checked:
                        if DEPTH_TOPIC in self.node.topic_list():
                            raise RuntimeError('Modo YOLO exige o mundo RGB-only do lançador; mundo RGB-D recusado')
                        self.geometry_camera_checked = True
                    try:
                        points = self.source.observe_rgb(frame, intrinsics, pos, quat)
                    except (ValueError, RuntimeError) as error:
                        if self.recorder is not None:
                            self.recorder.record_geometry(number, ts, frame, np.empty((0,3)), [],
                                pos, quat, time.monotonic()-began, quality_from_rgb(frame),
                                intrinsics=list(intrinsics), source=self.source.name, rejected=str(error))
                        raise RuntimeError(f'Observação YOLO sem geometria confiável; drone parado: {error}') from error
                    elapsed = time.monotonic()-began
                    quality = quality_from_rgb(frame)
                    if self.recorder is not None:
                        self.recorder.record_geometry(number, ts, frame, points, self.source.last_objects,
                                                      pos, quat, elapsed, quality, intrinsics=list(intrinsics),
                                                      source=self.source.name, pose_time=ps,
                                                      capture_to_inference_seconds=time.monotonic()-received)
                    if (quality['luma_mean'] < 15 or quality['luma_mean'] > 240 or
                            quality['saturated_fraction'] > .75 or quality['luma_std'] < 5):
                        raise RuntimeError('Imagem YOLO escura, saturada ou sem contraste; drone parado')
                    if time.monotonic()-received > .75:
                        raise RuntimeError('Observação YOLO atrasada; drone parado')
                    return points, number
                # Retain the matching reference before inference can evict it
                # from the transport queue. It is used only by the recorder.
                with self.lock:
                    reference = synchronized_sample(list(self.references), ts)
                    rgb_reference = synchronized_sample(list(self.rgb_references), ts)
                try:
                    depth = self.source.infer(frame)
                except Exception as error:
                    raise RuntimeError(f'Falha na fonte de profundidade; drone parado: {error}') from error
                elapsed = time.monotonic() - began
                if self.rgb_mode:
                    quality = quality_from_rgb(frame)
                    if self.recorder is not None:
                        self.recorder.record(number, ts, frame, depth,
                                             reference[1] if reference else None, pos, quat, elapsed, quality,
                                             intrinsics=list(intrinsics), pose_time=ps,
                                             reference_time=reference[0] if reference else None,
                                             capture_to_inference_seconds=time.monotonic()-received,
                                             source=self.source.name, input_size=self.source.input_size)
                    if (quality['luma_mean'] < 15 or quality['luma_mean'] > 240 or
                            quality['saturated_fraction'] > .75 or quality['luma_std'] < 5):
                        raise RuntimeError('Imagem escura, saturada ou sem textura; drone parado')
                    depth = validate_estimate(depth, dimensions, time.monotonic()-received)
                elif np.count_nonzero(np.isfinite(depth) & (depth > .08)) < depth.size * .01:
                    raise RuntimeError('Profundidade sem medidas válidas suficientes; drone parado')
                elif self.recorder is not None:
                    if rgb_reference:
                        self.recorder.record(number, ts, rgb_reference[1], depth, depth, pos, quat,
                                             elapsed, quality_from_rgb(rgb_reference[1]),
                                             intrinsics=list(intrinsics), pose_time=ps, reference_time=ts,
                                             source=self.source.name)
                return project_depth(depth, intrinsics, pos, quat), number
            time.sleep(.02)
        raise RuntimeError(f'Sem imagem/pose sincronizadas recentes; drone parado. {self.error or ""}')


class Navigator:
    def __init__(self, node, altitude, bounds, *, depth_source=None, recorder=None):
        from gz.msgs10.stringmsg_pb2 import StringMsg
        from gz.transport13 import Node
        if depth_source is None:
            from pathlib import Path
            from yolo_geometry import YoloKnownPillar
            models = Path(__file__).resolve().parents[1] / 'models'
            depth_source = YoloKnownPillar(models / 'yolo_pillar.pt', models / 'yolo_pillar.json')
        self.node = node
        self.observer = DepthObserver(Node(), source=depth_source, recorder=recorder)
        self.recorder = recorder
        self.map = SurfaceMap(altitude)
        self.planner = GridPlanner(bounds)
        self.publisher = node.advertise(MAP_TOPIC, StringMsg)
        self.replans = 0
        self.skipped = 0
        self.frame = 0
        self.state = 'Aguardando percepção ' + self.observer.source.name
        self.route = []
        self.history = []
        self.visits = {}
        self.coverage = None
        self.coverage_selector = None
        self.publish()

    def enable_coverage(self, plan):
        from adaptive_coverage import CoverageGrid, SweepSelector
        self.coverage = CoverageGrid(plan.polygon, plan.footprint_width)
        self.coverage_selector = SweepSelector(plan, self.coverage)
        self.observer.enable_ground()
        self.publish()

    def publish(self):
        from gz.msgs10.stringmsg_pb2 import StringMsg
        payload = {'source': self.observer.source.name, 'voxels': self.map.points().round(3).tolist(),
                   'resolution': self.map.resolution, 'altitude': self.map.altitude,
                   'route': self.route, 'state': self.state, 'frames': self.frame,
                   'replans': self.replans, 'skipped': self.skipped,
                   'objects': getattr(self.observer.source, 'last_objects', [])}
        if self.coverage is not None:
            payload['coverage'] = self.coverage.payload()
        self.publisher.publish(StringMsg(data=json.dumps(payload)))

    def look(self, position, yaw):
        try:
            points, self.frame = self.observer.observe(position, yaw)
        except RuntimeError as error:
            self.frame = getattr(self.observer, 'last_frame', self.frame)
            if self.recorder is not None:
                self.recorder.stop(self.frame, position, error)
            raise
        revision = self.map.revision
        if self.observer.rgb_mode and not getattr(self.observer.source, 'geometry_mode', False):
            candidate = SurfaceMap(self.map.altitude, self.map.resolution)
            candidate.integrate(points)
            if len(candidate.voxels - self.map.voxels) > 400:
                self.state = 'PARADO: excesso de superfícies estimadas em um quadro'
                self.publish()
                if self.recorder is not None:
                    self.recorder.stop(self.frame, position, self.state)
                raise RuntimeError('Profundidade RGB inconsistente: excesso de superfícies; drone parado')
        self.map.integrate(points)
        if revision != self.map.revision:
            self.planner.update(self.map.collision_points())
        if getattr(self, 'coverage', None) is not None:
            # Only the ranged object envelope is excluded. Its flight safety
            # margin and any shadowed ground remain coverage obligations.
            anchor = getattr(self.observer.source, 'anchor', None)
            if anchor is not None:
                from yolo_geometry import PILLAR_RADIUS
                self.coverage.set_obstacles([(*anchor.center, PILLAR_RADIUS+anchor.uncertainty_m)])
            elif len(self.map.voxels):
                # Dense sources provide surface columns, not a known full shape.
                self.coverage.set_obstacles([(*p, self.map.resolution/math.sqrt(2))
                                              for p in self.map.collision_points()])
            try:
                view, timestamp, rgb = self.observer.ground_view()
            except RuntimeError as error:
                if self.recorder is not None:
                    self.recorder.stop(self.frame, position, error)
                raise
            gained = self.coverage.observe(*view)
            if self.recorder is not None:
                self.recorder.coverage('observation', frame=self.frame, sim_time=timestamp,
                                       pose_xyz=view[0], pose_xyzw=view[1], intrinsics=view[2],
                                       gained_cells=gained, summary=self.coverage.summary(), rgb=rgb)
        self.publish()


    def go(self, start, goal, yaw, *, adaptive=False, interrupt=None):
        from drone_zigzag import move
        position = tuple(start)
        deadline = time.monotonic() + 180
        detouring = False
        local = LocalRoute(start, goal, getattr(self, 'visits', {}),
                           preserve_leg=not adaptive, smooth=adaptive)
        while math.dist(position, goal) > .025:
            if time.monotonic() > deadline:
                raise RuntimeError('Tempo de navegação esgotado; mantendo posição')
            path = local.update(self.planner, position)
            if path is None:
                if self.planner.cell(position) in self.planner.blocked:
                    raise RuntimeError('Margem de segurança invadida; mantendo posição')
                self.skipped += 1
                self.state = 'Waypoint inacessível nas observações; pulando'
                self.route = []
                self.publish()
                print(f'Waypoint blocked; skipped: {goal}', flush=True)
                return position, yaw
            next_point = next((p for p in path[1:] if math.dist(position[:2], p) > .0001), goal[:2])
            heading = math.atan2(next_point[1]-position[1], next_point[0]-position[0])
            turn = math.atan2(math.sin(heading-yaw), math.cos(heading-yaw))
            if abs(turn) > .02:
                yaw = move(self.node, position, position, yaw, target_yaw=heading)
            # No translation until a fresh view along the intended motion arrives.
            revision = self.map.revision
            self.look(position, yaw)
            if self.observer.rgb_mode and not self.observer.source.navigation_ready:
                self.state = 'PARADO: fonte RGB não aprovada na validação de distâncias'
                self.route = []
                self.publish()
                if self.recorder is not None:
                    self.recorder.stop(self.frame, position, self.state)
                raise RuntimeError(self.state)
            if interrupt is not None and interrupt(position):
                self.route = []
                self.publish()
                return position, yaw
            if revision != self.map.revision:
                # New pixels may invalidate even the next step. Recompute first.
                updated = local.update(self.planner, position)
                if updated != path:
                    continue
            blocked = len(path) > 2
            if blocked and not detouring:
                self.replans += 1
                print(f'Obstacle observed: A* detour #{self.replans}, {len(self.map.voxels)} surface voxels', flush=True)
            detouring = blocked
            self.route = [list(p) for p in path]
            self.state = ('Contornando — retomada da faixa adiante' if blocked else
                          'Zigzag — YOLO + geometria RGB' if getattr(self.observer.source, 'geometry_mode', False)
                          else 'Zigzag — observando profundidade')
            distance = math.dist(position[:2], next_point)
            fraction = min(.18 / max(distance, 1e-9), 1.)
            target = (position[0] + (next_point[0]-position[0])*fraction,
                      position[1] + (next_point[1]-position[1])*fraction, goal[2])
            if not self.planner.clear(position, target):
                raise RuntimeError('Próximo passo obstruído; mantendo posição')
            if self.recorder is not None:
                self.recorder.decision(self.frame, position, target, blocked,
                                       self.replans, self.map.revision)
            self.publish()
            yaw = move(self.node, position, target, yaw, speed=.7)
            position = target
            self.history.append(position)
            cell = self.planner.cell(position)
            previous_cell = self.planner.cell(self.history[-2]) if len(self.history) > 1 else None
            if cell != previous_cell:
                local.visits[cell] = local.visits.get(cell, 0) + 1
        return position, yaw

    def cover(self, position, yaw):
        previous_region = self.planner.region
        self.planner.set_region(self.coverage_selector.plan.polygon)
        try:
            return self._cover(position,yaw)
        finally:
            self.planner.set_region(previous_region)

    def _cover(self, position, yaw):
        """Sweep only pending ground, allowing a bypass to finish on later lanes."""
        from drone_zigzag import move
        selector, grid = self.coverage_selector, self.coverage
        self.look(position, yaw)
        tasks, no_progress = 0, 0
        while grid.pending.any():
            before = int(grid.seen.sum())
            task = selector.choose(self.planner, position)
            if task is None:
                raise RuntimeError(f'Cobertura incompleta: {int(grid.pending.sum())} células sem vista acessível; mantendo posição')
            previous_lane = selector.active_lane
            selector.active_lane = task.lane
            tasks += 1
            event = {'from_lane': previous_lane+1 if previous_lane >= 0 else None,
                     'lane': task.lane+1, 'entry': task.entry, 'end': task.end,
                     'pending_cells': int(grid.pending.sum())}
            selector.transitions.append(event)
            print(f'Adaptive sweep {tasks}: lane {task.lane+1}, {event["pending_cells"]} ground cells pending', flush=True)
            if self.recorder is not None:
                self.recorder.coverage('task', **event)
            revision = self.map.revision

            def obsolete(current):
                # Never force a planned pass after the camera has covered it.
                # A new obstacle also returns control to the lane selector.
                return self.map.revision != revision or not (task.mask & grid.pending).any()

            position, yaw = self.go(position, (*task.entry, position[2]), yaw,
                                    adaptive=True, interrupt=obsolete)
            if not obsolete(position):
                yaw = move(self.node, position, position, yaw, target_yaw=task.yaw)
                self.look(position, yaw)
                position, yaw = self.go(position, (*task.end, position[2]), yaw,
                                        adaptive=True, interrupt=obsolete)
            self.look(position, yaw)
            no_progress = no_progress+1 if int(grid.seen.sum()) <= before and self.map.revision == revision else 0
            if no_progress >= 3 or tasks > len(grid.centers)*2:
                raise RuntimeError('Cobertura sem progresso observado; mantendo posição')
        self.route = []
        self.state = 'Cobertura observada concluída'
        self.publish()
        if self.recorder is not None:
            self.recorder.coverage('complete', summary=grid.summary(), tasks=tasks,
                                   transitions=selector.transitions)
        print(f'Adaptive coverage complete: {tasks} useful passes; {grid.summary()}', flush=True)
        return position, yaw


class PerceptionLayer:
    """Dashboard reads the mission's observed map, never Gazebo scene geometry."""
    def __init__(self, ax, transform, node):
        from gz.msgs10.stringmsg_pb2 import StringMsg
        self.lock = threading.Lock()
        self.latest = None
        self.received = None
        self.artist = ax.scatter([], [], s=12, marker='s', c='tab:red', alpha=.65,
                                 transform=transform, label='Obstáculos no mapa')
        self.path, = ax.plot([], [], color='tab:purple', linewidth=2, transform=transform,
                             label='Rota local recalculada')
        self.covered = ax.scatter([], [], s=15, marker='s', c='mediumseagreen', alpha=.25,
                                  transform=transform, label='Solo observado')
        self.pending = ax.scatter([], [], s=9, marker='.', c='goldenrod', alpha=.45,
                                  transform=transform, label='Solo ainda pendente')
        self.excluded = ax.scatter([], [], s=12, marker='x', c='dimgray', alpha=.5,
                                   transform=transform, label='Envoltória excluída')
        self.status = ax.text(.02, .96, 'Mapa vazio — aguardando percepção', va='top',
                              transform=ax.transAxes, fontsize=8)
        if not node.subscribe(StringMsg, MAP_TOPIC, self.receive):
            raise RuntimeError('Não foi possível assinar mapa da percepção')

    def receive(self, message):
        try:
            data = json.loads(message.data)
            if data.get('source') not in (GazeboDepth.name, 'rgb_depth_anything_v2_indoor_small', 'rgb_yolo_known_pillar'):
                return
        except (ValueError, TypeError):
            return
        with self.lock:
            self.latest, self.received = data, time.monotonic()

    def update(self):
        with self.lock:
            data, received = self.latest, self.received
        if data is None:
            return []
        points = np.asarray(data['voxels']).reshape(-1, 3)
        self.artist.set_offsets(points[:, :2])
        route = np.asarray(data['route']).reshape(-1, 2)
        self.path.set_data(route[:, 0], route[:, 1])
        coverage = data.get('coverage')
        for artist, key in ((self.covered, 'observed'), (self.pending, 'pending'), (self.excluded, 'excluded')):
            artist.set_offsets(np.asarray(coverage[key] if coverage else []).reshape(-1,2))
        age = time.monotonic() - received
        source = ('YOLO + geometria RGB — pilar conhecido' if data['source'] == 'rgb_yolo_known_pillar'
                  else 'RGB estimado' if data['source'].startswith('rgb_') else 'Gazebo depth — linha de base')
        self.status.set_text(f"{source}\n{data['state']}\n{len(points)} voxels observados · {data['replans']} desvios"
                             + (f"\nSolo observado {coverage['observed_required_fraction']:.0%} · {coverage['pending_cells']} células pendentes"
                                if coverage else '')
                             + (f"\nZ medido {points[:, 2].min():.1f}–{points[:, 2].max():.1f} m" if len(points) else '')
                             + (f'\nAtualização há {age:.0f}s' if age > 3 else ''))
        return points[:, :2].tolist()

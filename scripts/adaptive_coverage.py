"""Observed ground coverage and flexible sweep selection, without simulator truth.

The grid is credited only by synchronized, calibrated downward RGB views.
Future footprints score routes; they never mark cells as observed.
"""

from dataclasses import dataclass
import math

import numpy as np

from roi_plan import point_inside


DOWN_OFFSET = np.array([0., 0., -.025])
DOWN_PITCH = math.pi / 2


def rectangular_plan(lanes, home, altitude, width, overlap):
    """Use the same closest-entry rectangle route as preview and execution."""
    from coverage_plan import rectangular_waypoints
    from roi_plan import Plan, Leg
    route = rectangular_waypoints(lanes, home, altitude)
    legs = []
    for (label_a, a), (label_b, b) in zip(route, route[1:]):
        sweep = label_b.startswith('Search pass')
        lane = min(range(len(lanes)), key=lambda i: abs(lanes[i]-b[1])) if sweep else -1
        legs.append(Leg(a[:2], b[:2], 'sweep' if sweep else 'transit', lane))
    return Plan([(-4,-3),(4,-3),(4,3),(-4,3)], legs, list(lanes),
                abs(lanes[1]-lanes[0]) if len(lanes)>1 else 0., width, overlap)


class CoverageGrid:
    def __init__(self, polygon, footprint_width, resolution=.20):
        self.resolution = resolution
        self.footprint_width = footprint_width
        xmin, ymin = np.min(polygon, axis=0)
        xmax, ymax = np.max(polygon, axis=0)
        self.centers = np.array([
            (x, y) for y in np.arange(ymin+resolution/2, ymax, resolution)
            for x in np.arange(xmin+resolution/2, xmax, resolution)
            if point_inside((x, y), polygon)], dtype=float).reshape(-1, 2)
        if not len(self.centers):
            raise ValueError('Região menor que a resolução de cobertura de 20 cm')
        half = resolution / 2
        offsets = np.array([[-half,-half], [-half,half], [half,-half], [half,half]])
        self.corners = self.centers[:, None, :] + offsets
        self.seen = np.zeros(len(self.centers), dtype=bool)
        self.excluded = np.zeros(len(self.centers), dtype=bool)
        self.obstacles = []
        self.views = []
        self.calibration = None
        self.revision = 0

    @property
    def pending(self):
        return ~(self.seen | self.excluded)

    def set_obstacles(self, obstacles):
        """Conservative known-object envelopes, including ranging uncertainty.

        Exclusion is reported separately from observed coverage. Flight safety
        inflation is NOT an exclusion: its visible ground still needs coverage.
        """
        obstacles = [tuple(float(v) for v in obstacle) for obstacle in obstacles]
        if obstacles == self.obstacles:
            return
        self.obstacles = obstacles
        self.excluded[:] = False
        for x, y, radius in obstacles:
            self.excluded |= np.linalg.norm(self.centers-[x,y], axis=1) <= radius + self.resolution/math.sqrt(2)
        # A newly detected object can invalidate a previously credited view.
        # Replay actual views with its shadow; do not retain that false credit.
        self.seen[:] = False
        for view in self.views:
            self.seen |= self.visible(*view)
        self.seen &= ~self.excluded
        self.revision += 1

    def visible(self, position, quaternion, intrinsics, dimensions):
        from depth_navigation import rotation
        body = rotation(quaternion)
        c, s = math.cos(DOWN_PITCH), math.sin(DOWN_PITCH)
        basis = body @ np.array([[c,0,s], [0,1,0], [-s,0,c]])
        origin = np.asarray(position) + body @ DOWN_OFFSET
        xyz = np.concatenate((self.corners, np.zeros((*self.corners.shape[:2], 1))), axis=2)
        local = (xyz-origin) @ basis
        fx, fy, cx, cy = intrinsics
        h, w = dimensions
        forward = np.maximum(local[..., 0], 1e-9)
        u, v = cx-fx*local[..., 1]/forward, cy-fy*local[..., 2]/forward
        # All corners must be inside the calibrated image and the user footprint.
        visible = np.all((local[..., 0] > .03) & (u > w*.02) & (u < w*.98) &
                         (v > h*.02) & (v < h*.98), axis=1)
        visible &= np.all(np.linalg.norm(self.corners-origin[:2], axis=2) <= self.footprint_width/2, axis=1)
        for x, y, radius in self.obstacles:
            # A vertical opaque envelope shadows any ground ray crossing it.
            delta = self.corners-origin[:2]
            length2 = np.sum(delta*delta, axis=2)
            t = np.clip(np.sum((np.array([x,y])-origin[:2])*delta, axis=2)/np.maximum(length2, 1e-12), 0, 1)
            closest = origin[:2] + t[..., None]*delta
            visible &= np.all(np.linalg.norm(closest-[x,y], axis=2) > radius, axis=1)
        return visible & ~self.excluded

    def observe(self, position, quaternion, intrinsics, dimensions):
        view = (tuple(position), tuple(quaternion), tuple(intrinsics), tuple(dimensions))
        self.calibration = (tuple(intrinsics), tuple(dimensions), float(position[2]))
        self.views.append(view)
        before = int(self.seen.sum())
        self.seen |= self.visible(*view)
        return int(self.seen.sum())-before

    def summary(self):
        required = int((~self.excluded).sum())
        return {'resolution_m': self.resolution, 'total_cells': len(self.centers),
                'observed_cells': int(self.seen.sum()), 'excluded_envelope_cells': int(self.excluded.sum()),
                'pending_cells': int(self.pending.sum()),
                'observed_required_fraction': float(self.seen.sum()/required) if required else 1.,
                'observation_count': len(self.views)}

    def payload(self):
        return {**self.summary(), 'observed': self.centers[self.seen].round(3).tolist(),
                'pending': self.centers[self.pending].round(3).tolist(),
                'excluded': self.centers[self.excluded].round(3).tolist()}


@dataclass
class SweepTask:
    lane: int
    entry: tuple
    end: tuple
    yaw: float
    mask: np.ndarray
    score: float


class SweepSelector:
    """Choose useful free sweep intervals; observed lanes need no repeat pass."""
    def __init__(self, plan, grid):
        self.plan, self.grid = plan, grid
        self.cache_key = None
        self.runs = []
        self.active_lane = -1
        self.transitions = []

    def _runs(self, planner):
        key = (self.grid.revision, self.grid.calibration, frozenset(planner.blocked),
               tuple(planner.region) if planner.region is not None else None)
        if key == self.cache_key:
            return
        self.cache_key, self.runs = key, []
        intrinsics, dimensions, altitude = self.grid.calibration
        for leg in self.plan.legs:
            if leg.kind != 'sweep':
                continue
            count = max(1, math.ceil(math.dist(leg.start, leg.end)/.20))
            points = np.linspace(leg.start, leg.end, count+1)
            yaw = math.atan2(leg.end[1]-leg.start[1], leg.end[0]-leg.start[0])
            quaternion = (0,0,math.sin(yaw/2),math.cos(yaw/2))
            run, masks = [], []
            for point in [*points, None]:
                free = point is not None and planner.clear(point, point)
                if free and run:
                    free = planner.clear(run[-1], point)
                if not free:
                    if run:
                        self.runs.append((leg.lane, np.array(run), np.array(masks), yaw))
                    run, masks = [], []
                if point is not None and planner.clear(point, point):
                    run.append(tuple(point))
                    masks.append(self.grid.visible((*point, altitude), quaternion, intrinsics, dimensions))

    def choose(self, planner, position):
        self._runs(planner)
        pending = self.grid.pending
        candidates = []
        for lane, points, masks, yaw in self.runs:
            useful = np.flatnonzero(np.any(masks & pending, axis=1))
            if not len(useful):
                continue
            left, right = useful[0], useful[-1]
            mask = np.any(masks[left:right+1], axis=0)
            gain = int((mask & pending).sum())
            for reverse in (False, True):
                entry, end = (points[right], points[left]) if reverse else (points[left], points[right])
                # Progress through lane numbers is a soft preference. A missed
                # patch behind the drone remains a valid later cleanup task.
                penalty = 1 + .04*max(0, self.active_lane-lane)
                cost = (math.dist(position[:2], entry)+math.dist(entry, end)+.40)*penalty/gain
                candidates.append(SweepTask(lane, tuple(entry), tuple(end),
                                            yaw+math.pi if reverse else yaw, mask, cost))
        candidates.sort(key=lambda task: task.score)
        best = None
        for task in candidates:
            if best is not None and task.score >= best.score:
                break
            path = planner.plan(position, task.entry)
            if path is None:
                continue
            path_length = sum(math.dist(a,b) for a,b in zip(path,path[1:]))
            detour = max(0., path_length-math.dist(position[:2], task.entry))
            task.score += detour/max(int((task.mask & pending).sum()), 1)
            if best is None or task.score < best.score:
                best = task
        return best


def smooth_path(path, planner, radius=.45):
    """Round corners with quadratic arcs, validating EVERY executed chord.

    Reduce rounding when clearance is tight. Retain a corner when no checked
    curve fits, rather than cutting inside the inflated obstacle map.
    """
    if len(path) < 3:
        return path
    result = [tuple(path[0])]
    for a, b, c in zip(path, path[1:], path[2:]):
        a, b, c = map(np.asarray, (a,b,c))
        trim = min(radius, math.dist(a,b)*.35, math.dist(b,c)*.35)
        accepted = False
        for _ in range(5):
            if trim < .015:
                break
            left = b+(a-b)*trim/math.dist(a,b)
            right = b+(c-b)*trim/math.dist(c,b)
            count = max(3, math.ceil(2*trim/.10))
            arc = [(1-t)**2*left+2*(1-t)*t*b+t*t*right for t in np.linspace(0,1,count+1)]
            trial = [result[-1], *arc, c]
            if all(planner.clear(p,q) for p,q in zip(trial, trial[1:])):
                result.extend(tuple(p) for p in arc)
                accepted = True
                break
            trim /= 2
        if not accepted:
            result.append(tuple(b))
    result.append(tuple(path[-1]))
    return result

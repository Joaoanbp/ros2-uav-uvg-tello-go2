"""Planejamento cinemático em polígonos simples, sem dependências do Gazebo.

Faixas horizontais são recortadas pelo polígono. Conexões que cruzariam uma
concavidade seguem o contorno. Uma volta inicial pelo perímetro cobre as pontas
sob o modelo ideal de footprint circular (diâmetro = footprint_width).
"""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random

from coverage_plan import finite_coordinate, overlap_percent, positive_width


EPS = 1e-8


@dataclass(frozen=True)
class Leg:
    start: tuple
    end: tuple
    kind: str
    lane: int = -1


@dataclass
class Plan:
    polygon: list
    legs: list
    lanes: list
    spacing: float
    footprint_width: float
    overlap: float


def add_roi_arguments(parser, *, include_shape=False):
    if include_shape:
        parser.add_argument('--shape', choices=('rectangle', 'circle', 'star', 'random', 'drawn'),
                            default='rectangle', help='Formato da região de interesse')
    parser.add_argument('--polygon-json', type=Path,
                        help='Missão poligonal escolhida no navegador')
    parser.add_argument('--radius', type=positive_width, default=4.0,
                        help='Raio externo, em metros (padrão: 4)')
    parser.add_argument('--center-x', type=finite_coordinate, default=0.0)
    parser.add_argument('--center-y', type=finite_coordinate, default=0.0)
    parser.add_argument('--seed', type=int, default=42,
                        help='Semente reproduzível do polígono aleatório (padrão: 42)')


def region_polygon(shape, radius=4.0, center=(0.0, 0.0), seed=42):
    positive_width(radius)
    for value in center:
        finite_coordinate(value)
    if shape == 'circle':
        radii = [radius] * 96
    elif shape == 'star':
        radii = [radius if index % 2 == 0 else radius * 0.45 for index in range(10)]
    elif shape == 'random':
        # Ordered polar angles and positive radii give a simple, star-shaped
        # polygon. A local RNG makes it reproducible without global state.
        rng = random.Random(seed)
        radii = [radius * rng.uniform(0.4, 1.0) for _ in range(12)]
    else:
        raise ValueError(f'Formato desconhecido: {shape}')
    points = []
    for index, length in enumerate(radii):
        angle = math.pi / 2 + 2 * math.pi * index / len(radii)
        points.append((center[0] + length * math.cos(angle),
                       center[1] + length * math.sin(angle)))
    return points


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def subtract(a, b):
    return (a[0] - b[0], a[1] - b[1])


def interpolate(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def distance_to_segment(point, a, b):
    delta = subtract(b, a)
    length2 = delta[0] ** 2 + delta[1] ** 2
    if length2 == 0:
        return math.dist(point, a)
    relative = subtract(point, a)
    t = max(0.0, min(1.0, (relative[0] * delta[0] + relative[1] * delta[1]) / length2))
    return math.dist(point, interpolate(a, b, t))


def edges(polygon):
    return zip(polygon, polygon[1:] + polygon[:1])


def point_inside(point, polygon):
    """Even/odd containment, including the perimeter."""
    x, y = point
    inside = False
    for a, b in edges(polygon):
        if distance_to_segment(point, a, b) <= EPS:
            return True
        if (a[1] > y) != (b[1] > y):
            hit = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if x < hit:
                inside = not inside
    return inside


def intersection_parameters(a, b, c, d):
    """Parameters on AB where it meets CD, including collinear overlap."""
    r, s = subtract(b, a), subtract(d, c)
    denominator = cross(r, s)
    offset = subtract(c, a)
    if abs(denominator) > EPS:
        t, u = cross(offset, s) / denominator, cross(offset, r) / denominator
        return [max(0.0, min(1.0, t))] if -EPS <= t <= 1 + EPS and -EPS <= u <= 1 + EPS else []
    if abs(cross(offset, r)) > EPS:
        return []
    length2 = r[0] ** 2 + r[1] ** 2
    if length2 <= EPS ** 2:
        return [0.0] if distance_to_segment(a, c, d) <= EPS else []
    ts = [sum(v * w for v, w in zip(subtract(p, a), r)) / length2 for p in (c, d)]
    low, high = max(0.0, min(ts)), min(1.0, max(ts))
    return [low, high] if low <= high + EPS else []


def validate_polygon(polygon):
    if len(polygon) < 3:
        raise ValueError('A região precisa de pelo menos três vértices')
    for point in polygon:
        if len(point) != 2 or not all(math.isfinite(v) for v in point):
            raise ValueError('Vértices devem ter duas coordenadas finitas')
    boundary = list(edges(polygon))
    if any(math.dist(a, b) <= EPS for a, b in boundary):
        raise ValueError('A região contém vértices consecutivos repetidos')
    for i, (a, b) in enumerate(boundary):
        c = polygon[(i + 2) % len(polygon)]
        if abs(cross(subtract(b, a), subtract(c, b))) <= EPS:
            if sum(v * w for v, w in zip(subtract(b, a), subtract(c, b))) < 0:
                raise ValueError('O perímetro contém arestas sobrepostas')
        for j in range(i + 1, len(boundary)):
            if j == i + 1 or (i == 0 and j == len(boundary) - 1):
                continue
            if intersection_parameters(a, b, *boundary[j]):
                raise ValueError('O perímetro não pode se cruzar ou se tocar')
    area2 = sum(cross(a, b) for a, b in boundary)
    if abs(area2) <= EPS:
        raise ValueError('A região precisa ter área positiva')


def parse_drawn_mission(data):
    """Validate browser/JSON input before planning or starting a mission."""
    if not isinstance(data, dict):
        raise ValueError('A missão deve ser um objeto JSON')

    def point(value, label):
        if (not isinstance(value, (list, tuple)) or len(value) != 2 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                    not math.isfinite(v) or abs(v) > 100 for v in value)):
            raise ValueError(f'{label} deve ter X e Y finitos entre -100 e 100 m')
        return [float(v) for v in value]

    raw = data.get('polygon')
    if not isinstance(raw, list) or not 3 <= len(raw) <= 128:
        raise ValueError('Marque entre 3 e 128 vértices para o perímetro')
    polygon = [point(value, f'Vértice {index + 1}') for index, value in enumerate(raw)]
    validate_polygon(polygon)
    home = point(data.get('home'), 'Base de decolagem')

    def number(name, default, minimum=None, maximum=None):
        value = data.get(name, default)
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or
                (minimum is not None and value < minimum) or
                (maximum is not None and value > maximum)):
            raise ValueError(f'Valor inválido para {name}')
        return float(value)

    altitude = number('altitude', 1.6, .4, 20)
    width = number('footprint_width', 2.0, .01, 100)
    overlap = number('side_overlap', 80.0, 0, 99.9)
    return {'polygon': polygon, 'home': home, 'altitude': altitude,
            'footprint_width': width, 'side_overlap': overlap}


def apply_drawn_mission(args):
    if args.polygon_json is None:
        raise ValueError('Informe --polygon-json com o arquivo salvo no navegador')
    try:
        data = json.loads(args.polygon_json.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f'Não foi possível ler o polígono: {error}') from error
    mission = parse_drawn_mission(data)
    args.drawn_polygon = mission['polygon']
    args.start_x, args.start_y = mission['home']
    args.altitude = mission['altitude']
    args.footprint_width = mission['footprint_width']
    args.side_overlap = mission['side_overlap']
    return mission


def segment_inside(a, b, polygon):
    if not point_inside(a, polygon) or not point_inside(b, polygon):
        return False
    cuts = sorted({0.0, 1.0, *(t for c, d in edges(polygon)
                              for t in intersection_parameters(a, b, c, d))})
    return all(point_inside(interpolate(a, b, (left + right) / 2), polygon)
               for left, right in zip(cuts, cuts[1:]) if right - left > EPS)


def scanline_segments(polygon, y):
    """Clip a horizontal line; keep separate intervals at concavities.

    Include vertices/horizontal edges, then classify interval midpoints. This
    also handles scanlines that pass exactly through a polygon vertex.
    """
    hits = []
    for a, b in edges(polygon):
        if abs(a[1] - b[1]) <= EPS:
            if abs(y - a[1]) <= EPS:
                hits.extend((a[0], b[0]))
        elif min(a[1], b[1]) - EPS <= y <= max(a[1], b[1]) + EPS:
            hits.append(a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1]))
    unique = []
    for x in sorted(hits):
        if not unique or x - unique[-1] > EPS:
            unique.append(x)
    result = []
    for left, right in zip(unique, unique[1:]):
        if point_inside(((left + right) / 2, y), polygon):
            if result and abs(result[-1][1][0] - left) <= EPS:
                result[-1] = (result[-1][0], (right, y))
            else:
                result.append(((left, y), (right, y)))
    return result


def boundary_connection(start, end, polygon):
    """Direct chord when contained; otherwise the shorter perimeter arc."""
    if segment_inside(start, end, polygon):
        return [start, end]
    lengths = [math.dist(a, b) for a, b in edges(polygon)]
    offsets = [0.0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    perimeter = offsets[-1]

    def location(point):
        for index, (a, b) in enumerate(edges(polygon)):
            if distance_to_segment(point, a, b) <= EPS:
                return (offsets[index] + math.dist(a, point)) % perimeter
        raise ValueError('Conexão esperava um ponto no perímetro')

    s, e = location(start), location(end)

    def arc(origin, destination, origin_point, destination_point):
        distance = (destination - origin) % perimeter
        vertices = sorted(((offset - origin) % perimeter, vertex)
                          for offset, vertex in zip(offsets, polygon))
        return [origin_point, *(p for delta, p in vertices if EPS < delta < distance - EPS), destination_point]

    forward = arc(s, e, start, end)
    backward = list(reversed(arc(e, s, end, start)))
    return min((forward, backward), key=lambda path: sum(math.dist(a, b) for a, b in zip(path, path[1:])))


def perimeter_legs(polygon, entry_from=None):
    """One complete boundary loop, optionally from its closest point to home.

    Project onto edges, not just vertices: the entry can split an edge without
    changing the region or the horizontal coverage lanes.
    """
    if entry_from is None:
        return [Leg(a, b, 'perimeter') for a, b in edges(polygon)]
    for value in entry_from:
        finite_coordinate(value)
    candidates = []
    for index, (a, b) in enumerate(edges(polygon)):
        delta = subtract(b, a)
        relative = subtract(entry_from, a)
        t = max(0., min(1., sum(v*w for v, w in zip(relative, delta)) /
                       sum(v*v for v in delta)))
        point = interpolate(a, b, t)
        candidates.append((math.dist(entry_from, point), index, point))
    _, index, entry = min(candidates)
    vertices = polygon[index + 1:] + polygon[:index + 1]
    path = [entry, *vertices, entry]
    return [Leg(a, b, 'perimeter') for a, b in zip(path, path[1:])
            if math.dist(a, b) > EPS]


def plan_polygon(polygon, width=2.0, overlap=80.0, *, entry_from=None):
    polygon = [tuple(p) for p in polygon]
    validate_polygon(polygon)
    positive_width(width)
    overlap_percent(overlap)
    low, high = min(y for _, y in polygon), max(y for _, y in polygon)
    maximum_spacing = width * (1 - overlap / 100)
    if maximum_spacing <= 0 or (high - low) / maximum_spacing > 10000:
        raise ValueError('A cobertura exige mais de 10000 faixas; aumente o footprint ou reduza a sobreposição')
    count = max(1, math.ceil((high - low) / maximum_spacing - 1e-12))
    spacing = (high - low) / count
    lanes = [low + (index + 0.5) * spacing for index in range(count)]
    legs = perimeter_legs(polygon, entry_from)
    current = legs[0].start
    for index, y in enumerate(lanes):
        intervals = scanline_segments(polygon, y)
        if index % 2:
            intervals = [(b, a) for a, b in reversed(intervals)]
        for start, end in intervals:
            connection = boundary_connection(current, start, polygon)
            legs.extend(Leg(a, b, 'transit') for a, b in zip(connection, connection[1:])
                        if math.dist(a, b) > EPS)
            legs.append(Leg(start, end, 'sweep', index))
            current = end
    return Plan(polygon, legs, lanes, spacing, width, overlap)


def plan_from_args(args, shape):
    if shape == 'drawn':
        polygon = getattr(args, 'drawn_polygon', None)
        if polygon is None:
            polygon = apply_drawn_mission(args)['polygon']
    else:
        polygon = region_polygon(shape, args.radius, (args.center_x, args.center_y), args.seed)
    return plan_polygon(polygon,
                        args.footprint_width, args.side_overlap,
                        entry_from=(args.start_x, args.start_y))


def plan_summary(plan):
    area = abs(sum(cross(a, b) for a, b in edges(plan.polygon))) / 2
    perimeter = sum(math.dist(a, b) for a, b in edges(plan.polygon))
    length = sum(math.dist(leg.start, leg.end) for leg in plan.legs)
    return (f'Área: {area:.2f} m² | perímetro: {perimeter:.2f} m | '
            f'{len(plan.lanes)} faixas | espaçamento: {plan.spacing:.3f} m | '
            f'trajeto na região: {length:.2f} m')

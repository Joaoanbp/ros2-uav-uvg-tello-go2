"""Ideal rectangular ground coverage; no camera or flight dynamics required."""

import argparse
import math


def finite_coordinate(value):
    number = float(value)
    if not math.isfinite(number):
        raise argparse.ArgumentTypeError("Coordinates must be finite numbers")
    return number


def overlap_percent(value):
    number = finite_coordinate(value)
    if not 0 <= number < 100:
        raise argparse.ArgumentTypeError("Side overlap must be at least 0 and less than 100 percent")
    return number


def positive_width(value):
    number = finite_coordinate(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Footprint width must be positive")
    return number


def add_coverage_arguments(parser):
    parser.add_argument('--coverage-mode', choices=('adaptive', 'fixed'), default='adaptive',
                        help='adaptive acompanha o solo observado e muda de faixa; fixed repete o zigzag nominal')
    parser.add_argument("--side-overlap", type=overlap_percent, default=80.0,
                        help="Minimum modeled side overlap in percent (default: 80)")
    parser.add_argument("--footprint-width", type=positive_width, default=2.0,
                        help="Assumed cross-track ground footprint in meters, independent of altitude (default: 2)")


def add_navigation_arguments(parser):
    def altitude(value):
        number = finite_coordinate(value)
        if number < .4:
            raise argparse.ArgumentTypeError('Altitude de cruzeiro deve ser pelo menos 0.4 m')
        return number
    parser.add_argument('--altitude', type=altitude, default=1.6,
                        help='Altitude de cruzeiro em metros (padrão: 1.6)')
    parser.add_argument('--depth-source', choices=('yolo', 'rgb', 'gazebo'), default='yolo',
                        help='Percepção: yolo usa RGB e pilar conhecido; rgb é profundidade experimental; gazebo é referência')
    parser.add_argument('--yolo-checkpoint', default='models/yolo_pillar.pt')
    parser.add_argument('--yolo-profile', default='models/yolo_pillar.json')
    parser.add_argument('--depth-repository', default='third_party/Depth-Anything-V2',
                        help='Checkout do Depth Anything V2')
    parser.add_argument('--depth-checkpoint', default='models/depth_anything_v2_metric_hypersim_vits.pth',
                        help='Checkpoint métrico indoor Small')
    parser.add_argument('--depth-device', default='cpu', help='Dispositivo PyTorch, ex.: cpu ou cuda')
    parser.add_argument('--depth-input-size', type=int, default=252,
                        help='Lado de entrada do modelo em pixels (padrão CPU: 252)')
    parser.add_argument('--depth-cpu-threads', type=int, default=4,
                        help='Threads CPU do PyTorch (padrão: 4)')
    parser.add_argument('--evaluation-dir', default=None,
                        help='Diretório de observações e decisões; yolo registra somente RGB/geometria')


def coverage_lanes(width, overlap):
    """Cover Y=[-3,3], inset edge lanes by half the footprint when possible.

    Evenly distribute lanes so spacing never exceeds the requested maximum.
    The assumed footprint extends beyond the centerline on both sides.
    """
    positive_width(width)
    overlap_percent(overlap)
    if width >= 6:
        return [0.0]
    maximum_spacing = width * (1 - overlap / 100)
    span = 6 - width
    if maximum_spacing <= 0 or span / maximum_spacing > 9999:
        raise ValueError("Requested coverage requires more than 10000 lanes; increase footprint width or reduce overlap")
    intervals = max(1, math.ceil(span / maximum_spacing - 1e-12))
    first = -3 + width / 2
    return [first + span * index / intervals for index in range(intervals + 1)]


def coverage_summary(width, overlap, lanes):
    if len(lanes) == 1:
        return "1 lane; footprint covers area width; adjacent-lane overlap N/A"
    spacing = lanes[1] - lanes[0]
    actual = 100 * (1 - spacing / width)
    return (f"{len(lanes)} lanes | spacing {spacing:.3f} m | "
            f"side overlap {actual:.1f}% (requested {overlap:g}%)")


def rectangular_waypoints(lanes, home, altitude):
    """Choose the closest admissible start of a complete serpentine sweep.

    Both lane orders and both initial horizontal directions are equivalent
    coverage plans. Select by approach distance without adding repeated passes.
    """
    candidates = []
    for ordered in (list(lanes), list(reversed(lanes))):
        for entry_x in (-4., 4.):
            waypoints = [('Search entry', (entry_x, ordered[0], altitude))]
            for index, y in enumerate(ordered):
                start_x = entry_x if index % 2 == 0 else -entry_x
                if index:
                    waypoints.append((f'Turn into lane {index + 1}', (start_x, y, altitude)))
                waypoints.append((f'Search pass {index + 1}/{len(ordered)}', (-start_x, y, altitude)))
            candidates.append(waypoints)
    return min(candidates, key=lambda route: math.dist(home, route[0][1][:2]))

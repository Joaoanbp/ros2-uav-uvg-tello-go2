"""CLI compartilhada pelas missões zigzag de regiões poligonais."""

import argparse
from dataclasses import asdict
from html import escape
import json
from pathlib import Path

from coverage_plan import add_coverage_arguments, finite_coordinate
from roi_plan import add_roi_arguments, apply_drawn_mission, plan_from_args, plan_summary
from coverage_plan import add_navigation_arguments


def mission_waypoints(plan, altitude=1.6):
    return [('ROI entry', (*plan.legs[0].start, altitude)),
            *((f'{leg.kind} {index + 1}/{len(plan.legs)}', (*leg.end, altitude))
              for index, leg in enumerate(plan.legs))]


def save_preview(path, plan, shape, home):
    """Standalone SVG preview, requiring only the Python standard library."""
    points = [*plan.polygon, home]
    xmin, xmax = min(p[0] for p in points), max(p[0] for p in points)
    ymin, ymax = min(p[1] for p in points), max(p[1] for p in points)
    scale = min(780 / (xmax - xmin), 620 / (ymax - ymin))

    def xy(point):
        return (60 + (point[0] - xmin) * scale, 710 - (point[1] - ymin) * scale)

    def line(a, b, color, width=1.6, dashed=False):
        x1, y1 = xy(a)
        x2, y2 = xy(b)
        dash = ' stroke-dasharray="6 5"' if dashed else ''
        return (f'<line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" '
                f'stroke="{color}" stroke-width="{width}"{dash}/>')

    coordinates = ' '.join(f'{x:.3f},{y:.3f}' for x, y in map(xy, plan.polygon))
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="800" viewBox="0 0 900 800">',
           '<rect width="900" height="800" fill="#f8fafc"/>',
           '<g font-family="sans-serif" fill="#172033">',
           f'<text x="30" y="32" font-size="23">Zigzag {escape(shape)} — plano da missão</text>',
           f'<text x="30" y="58" font-size="13">{escape(plan_summary(plan))}</text>',
           f'<polygon points="{coordinates}" fill="#e2f4e8" stroke="#15803d" stroke-width="3"/>']
    for leg in plan.legs:
        if leg.kind != 'perimeter':
            svg.append(line(leg.start, leg.end, '#2563eb' if leg.kind == 'sweep' else '#d97706',
                            dashed=leg.kind == 'transit'))
    svg.extend((line(home, plan.legs[0].start, '#64748b', dashed=True),
                line(plan.legs[-1].end, home, '#64748b', dashed=True)))
    hx, hy = xy(home)
    svg.append(f'<circle cx="{hx:.3f}" cy="{hy:.3f}" r="5" fill="#172033"/>')
    svg.append(f'<text x="{hx + 8:.3f}" y="{hy - 8:.3f}" font-size="14">Base</text>')
    sx, sy = xy(plan.legs[0].start)
    svg.append(f'<circle cx="{sx:.3f}" cy="{sy:.3f}" r="5" fill="#15803d"/>')
    svg.extend(['<text x="30" y="750" font-size="14">Verde: perímetro voado primeiro · Azul: zigzag · Laranja: conexões</text>',
                '<text x="30" y="775" font-size="13">Cinza: ida/retorno à base · X → / Y ↑ · Plano ideal, não trajetória medida</text>',
                '</g></svg>'])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(svg), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Zigzag: perímetro e cobertura no mundo tello_only.')
    parser.add_argument('--shape', choices=('circle', 'star', 'random', 'drawn'), required=True,
                        help='Formato da região da missão')
    add_roi_arguments(parser)
    add_coverage_arguments(parser)
    add_navigation_arguments(parser)
    parser.add_argument('--start-x', type=finite_coordinate, default=-7.0)
    parser.add_argument('--start-y', type=finite_coordinate, default=-5.0)
    parser.add_argument('--dry-run', action='store_true', help='Planeja sem executar no Gazebo')
    parser.add_argument('--preview', type=Path, help='Salva prévia SVG e não executa no Gazebo')
    parser.add_argument('--plan-json', type=Path, help='Exporta geometria, trechos e waypoints em JSON')
    args = parser.parse_args()
    shape = args.shape
    try:
        if shape == 'drawn':
            apply_drawn_mission(args)
        plan = plan_from_args(args, shape)
    except ValueError as error:
        parser.error(str(error))
    home = (args.start_x, args.start_y)
    print(f'Zigzag {shape} — {plan_summary(plan)}', flush=True)
    if args.coverage_mode == 'adaptive':
        print('Faixas são guias nominais; execução cobre solo observado e pode avançar para outra faixa.', flush=True)
    if shape == 'random':
        print(f'Semente: {args.seed}', flush=True)
    if args.plan_json:
        args.plan_json.parent.mkdir(parents=True, exist_ok=True)
        args.plan_json.write_text(json.dumps({
            'shape': shape, 'seed': args.seed if shape == 'random' else None,
            'home': home, 'altitude': args.altitude,
            'footprint_model': 'ideal circular footprint; diameter in meters',
            'coverage_mode': args.coverage_mode,
            **asdict(plan), 'waypoints': mission_waypoints(plan, args.altitude),
        }, indent=2) + '\n', encoding='utf-8')
    if args.preview:
        save_preview(args.preview, plan, shape, home)
        print(f'Prévia: {args.preview}', flush=True)
    if args.dry_run or args.preview:
        print('Somente planejamento; nenhuma missão executada.', flush=True)
        return
    from drone_zigzag import fly_waypoints, make_depth_source
    source = make_depth_source(args)
    fly_waypoints(home, mission_waypoints(plan, args.altitude), args.altitude,
                  depth_source=source, evaluation_dir=args.evaluation_dir,
                  coverage_plan=plan, coverage_mode=args.coverage_mode)


if __name__ == '__main__':
    main()

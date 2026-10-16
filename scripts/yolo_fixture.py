"""Controlled known-size obstacle world, with an actual RGB-only front sensor."""

import argparse
import copy
import math
from pathlib import Path
import random
import xml.etree.ElementTree as ET

from roi_plan import (add_roi_arguments, apply_drawn_mission, distance_to_segment,
                      edges, finite_coordinate, point_inside, region_polygon,
                      validate_polygon)
from yolo_geometry import PILLAR_HEIGHT, PILLAR_RADIUS

ROOT = Path(__file__).resolve().parents[1]
TARGET_SIZE = (.20, .20, .15)


def target_position(polygon, pillar_center, home=None, *, rng=None):
    """Sample ground truth for spawning only; never feed it to perception."""
    validate_polygon(polygon)
    rng = rng or random.SystemRandom()
    half_diagonal = math.hypot(*TARGET_SIZE[:2])/2
    boundary_margin = half_diagonal + .05
    xmin, xmax = min(p[0] for p in polygon), max(p[0] for p in polygon)
    ymin, ymax = min(p[1] for p in polygon), max(p[1] for p in polygon)
    for _ in range(10000):
        point = (rng.uniform(xmin,xmax), rng.uniform(ymin,ymax))
        if not point_inside(point,polygon):
            continue
        if min(distance_to_segment(point,a,b) for a,b in edges(polygon)) < boundary_margin:
            continue
        if math.dist(point,pillar_center) < PILLAR_RADIUS+half_diagonal+.15:
            continue
        if home is not None and math.dist(point,home) < half_diagonal+.5:
            continue
        return point
    raise ValueError('A ROI não tem espaço livre suficiente para o pequeno objeto de busca')


def target_model(position):
    model = ET.Element('model', name='search_target')
    ET.SubElement(model, 'static').text = 'true'
    ET.SubElement(model, 'pose').text = f'{position[0]} {position[1]} {TARGET_SIZE[2]/2} 0 0 0'
    link = ET.SubElement(model, 'link', name='target_link')
    for kind in ('collision', 'visual'):
        item = ET.SubElement(link, kind, name='blue_box')
        box = ET.SubElement(ET.SubElement(item, 'geometry'), 'box')
        ET.SubElement(box, 'size').text = ' '.join(str(v) for v in TARGET_SIZE)
        if kind == 'visual':
            material = ET.SubElement(item, 'material')
            ET.SubElement(material, 'ambient').text = '.02 .12 .95 1'
            ET.SubElement(material, 'diffuse').text = '.02 .12 .95 1'
    return model


def make_world(output, center=(.4, .15), name='test_pillar', *, target_polygon=None, home=None):
    tree = ET.parse(ROOT / 'simulation/worlds/tello_only.sdf')
    world = tree.getroot().find('world')
    for include in list(world.findall('include')):
        world.remove(include)
    drone = copy.deepcopy(ET.parse(ROOT / 'simulation/models/tello/model.sdf').getroot().find('model'))
    static = drone.find('static')
    if static is None:
        static = ET.SubElement(drone, 'static')
    static.text = 'true'
    ET.SubElement(drone, 'pose').text = '-5 -3 .09 0 0 0'
    front = drone.find(".//sensor[@name='front_camera']")
    front.set('type', 'camera')
    front.find('topic').text = '/tello/camera/front/image'
    world.append(drone)
    pillar = ET.SubElement(world, 'model', name=name)
    ET.SubElement(pillar, 'static').text = 'true'
    ET.SubElement(pillar, 'pose').text = f'{center[0]} {center[1]} 0 0 0 0'
    link = ET.SubElement(pillar, 'link', name='pillar_link')
    collision = ET.SubElement(link, 'collision', name='body')
    ET.SubElement(collision, 'pose').text = f'0 0 {PILLAR_HEIGHT/2} 0 0 0'
    cylinder = ET.SubElement(ET.SubElement(collision, 'geometry'), 'cylinder')
    ET.SubElement(cylinder, 'radius').text = str(PILLAR_RADIUS)
    ET.SubElement(cylinder, 'length').text = str(PILLAR_HEIGHT)
    # Black/orange bands define the appearance of the controlled test object.
    for index in range(9):
        visual = ET.SubElement(link, 'visual', name=f'band_{index}')
        ET.SubElement(visual, 'pose').text = f'0 0 {(index+.5)*PILLAR_HEIGHT/9} 0 0 0'
        geometry = ET.SubElement(ET.SubElement(visual, 'geometry'), 'cylinder')
        ET.SubElement(geometry, 'radius').text = str(PILLAR_RADIUS)
        ET.SubElement(geometry, 'length').text = str(PILLAR_HEIGHT/9)
        material = ET.SubElement(visual, 'material')
        color = '.95 .26 .025 1' if index % 2 == 0 else '.025 .025 .025 1'
        ET.SubElement(material, 'ambient').text = color
        ET.SubElement(material, 'diffuse').text = color
    if target_polygon is not None:
        world.append(target_model(target_position(target_polygon, center, home)))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree)
    tree.write(output, encoding='unicode', xml_declaration=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/tello-yolo-rgb.sdf')
    parser.add_argument('--shape', choices=('circle','star','random','drawn'), default='circle')
    parser.add_argument('--start-x', type=finite_coordinate, default=-7.)
    parser.add_argument('--start-y', type=finite_coordinate, default=-5.)
    add_roi_arguments(parser)
    # The launcher forwards its mission arguments so ROI geometry has one
    # parser. Navigation-only options are consumed by the mission itself.
    args, _ = parser.parse_known_args()
    try:
        if args.shape == 'drawn':
            apply_drawn_mission(args)
            polygon = args.drawn_polygon
        else:
            polygon = region_polygon(args.shape, args.radius, (args.center_x,args.center_y), args.seed)
        print(make_world(args.output, target_polygon=polygon, home=(args.start_x,args.start_y)))
    except ValueError as error:
        parser.error(str(error))

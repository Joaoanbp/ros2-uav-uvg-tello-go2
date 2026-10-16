"""Deterministic Gazebo fixtures for perception evaluation, never navigation input."""

from pathlib import Path
import struct
import xml.etree.ElementTree as ET
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = ('outdoor_clear', 'indoor_clear', 'wall', 'tree', 'thin_poles',
             'poor_texture', 'dim_light', 'bright_light')
SAMPLE_POSES = ((-3., -.4, 1.6, 0.), (-2.5, .4, 1.6, .12), (-1.8, -.2, 1.6, -.10))


def write_png(path, rgb):
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data))
    height, width = rgb.shape[:2]
    raw = b''.join(b'\0' + row.tobytes() for row in rgb)
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', width, height, 8, 2, 0, 0, 0))
                     + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


def box(world, name, pose, size, color, texture=None):
    model = ET.SubElement(world, 'model', name=name)
    ET.SubElement(model, 'static').text = 'true'
    ET.SubElement(model, 'pose').text = ' '.join(map(str, (*pose, 0, 0, 0)))
    link = ET.SubElement(model, 'link', name='link')
    for tag in ('collision', 'visual'):
        element = ET.SubElement(link, tag, name=tag)
        geometry = ET.SubElement(ET.SubElement(element, 'geometry'), 'box')
        ET.SubElement(geometry, 'size').text = ' '.join(map(str, size))
        if tag == 'visual':
            material = ET.SubElement(element, 'material')
            for kind in ('ambient', 'diffuse'):
                ET.SubElement(material, kind).text = ' '.join(map(str, (*color, 1)))
            if texture:
                workflow = ET.SubElement(ET.SubElement(material, 'pbr'), 'metal')
                ET.SubElement(workflow, 'albedo_map').text = str(texture)
                ET.SubElement(workflow, 'roughness').text = '.9'
    return model


def make_world(scenario, output):
    if scenario not in SCENARIOS:
        raise ValueError(f'Unknown scenario: {scenario}')
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    tree = ET.parse(ROOT / 'simulation/worlds/tello_only.sdf')
    world = tree.getroot().find('world')
    for include in list(world.findall('include')):
        if include.findtext('name') == 'tree':
            world.remove(include)
    if scenario != 'outdoor_clear':
        world.remove(world.find("model[@name='ground_plane']"))
        texture = None
        if scenario != 'poor_texture':
            y, x = np.mgrid[:512, :512]
            pattern = ((x // 32 + y // 32) % 2)[..., None]
            noise = np.random.default_rng(2026).integers(-9, 10, (512, 512, 1))
            rgb = np.clip(np.array([145, 126, 105]) + pattern * 55 + noise, 0, 255).astype(np.uint8)
            texture = output.parent / 'floor-checker.png'
            write_png(texture, rgb)
        box(world, 'floor', (0, 0, -.05), (12, 10, .1), (.65, .60, .53), texture)
        box(world, 'ceiling', (0, 0, 3.5), (12, 10, .15), (.85, .85, .85))
        for name, pose, size, color in (
            ('east_wall', (6, 0, 1.75), (.2, 10, 3.5), (.80, .75, .66)),
            ('west_wall', (-6, 0, 1.75), (.2, 10, 3.5), (.75, .80, .83)),
            ('north_wall', (0, 5, 1.75), (12, .2, 3.5), (.80, .85, .75)),
            ('south_wall', (0, -5, 1.75), (12, .2, 3.5), (.82, .77, .72))):
            box(world, name, pose, size, color)
        if scenario != 'poor_texture':
            for index in range(8):
                box(world, f'wall_panel_{index}', (5.88, -3.5 + index, 1.7),
                    (.02, .55, .9), (.15 + .07*index, .2, .6 - .05*index))
        if scenario == 'wall':
            box(world, 'calibration_wall', (.6, 0, 1.5), (.2, 2, 3), (.8, .08, .06))
        if scenario == 'tree':
            include = ET.SubElement(world, 'include')
            ET.SubElement(include, 'uri').text = 'model://tree'
            ET.SubElement(include, 'name').text = 'fixture_tree'
            ET.SubElement(include, 'pose').text = '.6 0 0 0 0 0'
        if scenario == 'thin_poles':
            for index in range(5):
                # 8 cm branches test small obstacles at multiple lateral positions.
                box(world, f'pole_{index}', (.6, -.9 + index*.45, 1.3),
                    (.08, .08, 2.6), (.16, .10, .05))
        if scenario in ('dim_light', 'bright_light'):
            ambient, diffuse = ('.015 .015 .015 1', '.04 .04 .04 1') if scenario == 'dim_light' else ('1 1 1 1', '1 1 1 1')
            world.find('scene/ambient').text = ambient
            world.find('light/diffuse').text = diffuse
    ET.indent(tree)
    tree.write(output, encoding='unicode', xml_declaration=True)
    return output

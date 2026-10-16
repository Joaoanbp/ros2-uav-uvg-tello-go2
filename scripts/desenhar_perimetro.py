#!/usr/bin/env python3
"""Editor local de missão poligonal: prévia e salvamento para o Gazebo."""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import tempfile
import webbrowser

from roi_plan import parse_drawn_mission, plan_polygon, plan_summary


ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / 'web' / 'desenhar_perimetro.html'
DEFAULT_OUTPUT = ROOT / 'artifacts' / 'zigzag-desenhado.json'


def make_preview(raw):
    mission = parse_drawn_mission(raw)
    height = max(y for _, y in mission['polygon']) - min(y for _, y in mission['polygon'])
    spacing = mission['footprint_width'] * (1 - mission['side_overlap'] / 100)
    if height / spacing > 500:
        raise ValueError('A prévia teria mais de 500 faixas; aumente o footprint ou reduza a sobreposição')
    plan = plan_polygon(mission['polygon'], mission['footprint_width'],
                        mission['side_overlap'], entry_from=mission['home'])
    return mission, {'summary': plan_summary(plan),
                     'entry': list(plan.legs[0].start),
                     'legs': [{'start': list(leg.start), 'end': list(leg.end),
                               'kind': leg.kind} for leg in plan.legs]}


def save_mission(destination, mission):
    destination.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile('w', encoding='utf-8', dir=destination.parent,
                                         prefix='.zigzag-desenhado-', suffix='.tmp',
                                         delete=False) as file:
            name = file.name
            json.dump(mission, file, ensure_ascii=False, indent=2)
            file.write('\n')
        os.replace(name, destination)
    finally:
        if name is not None and os.path.exists(name):
            os.unlink(name)


def handler_for(destination, server):
    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, status, body, content_type):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status, data):
            self.send_bytes(status, json.dumps(data, ensure_ascii=False).encode('utf-8'),
                            'application/json; charset=utf-8')

        def do_GET(self):
            if self.path == '/':
                self.send_bytes(200, PAGE.read_bytes(), 'text/html; charset=utf-8')
            elif self.path == '/favicon.ico':
                self.send_bytes(204, b'', 'image/x-icon')
            else:
                self.send_json(404, {'error': 'Página não encontrada'})

        def do_POST(self):
            if self.path not in ('/api/preview', '/api/start'):
                self.send_json(404, {'error': 'Endpoint desconhecido'})
                return
            if self.headers.get('Content-Type', '').split(';', 1)[0].strip().lower() != 'application/json':
                self.send_json(415, {'error': 'Envie a missão em JSON'})
                return
            try:
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 65536:
                    raise ValueError('Missão vazia ou grande demais')
                raw = json.loads(self.rfile.read(size))
                mission, preview = make_preview(raw)
                if self.path == '/api/start':
                    save_mission(destination, mission)
                    server.started = True
                self.send_json(200, preview)
            except (ValueError, TypeError, json.JSONDecodeError) as error:
                self.send_json(400, {'error': str(error)})
            except OSError as error:
                self.send_json(500, {'error': f'Não foi possível salvar a missão: {error}'})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1',
                        help='Endereço da interface (use 0.0.0.0 para acessar pelo celular na mesma rede)')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    server = HTTPServer((args.host, args.port), BaseHTTPRequestHandler)
    server.started = False
    server.RequestHandlerClass = handler_for(args.output, server)
    server.timeout = .5
    address = f'http://127.0.0.1:{server.server_port}/'
    print(f'Editor do perímetro: {address}', flush=True)
    if args.host == '0.0.0.0':
        print(f'No celular, abra http://IP_DESTE_COMPUTADOR:{server.server_port}/', flush=True)
    print('Clique ou toque nos vértices; conclua pelo botão Iniciar no Gazebo.', flush=True)
    if not args.no_browser:
        webbrowser.open(address)
    try:
        while not server.started:
            server.handle_request()
    except KeyboardInterrupt:
        return 130
    finally:
        server.server_close()
    print(f'Missão salva em {args.output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

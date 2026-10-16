"""The browser mission uses the same validated polygon as the flight CLI."""

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from desenhar_perimetro import handler_for, make_preview, save_mission


ROOT = Path(__file__).resolve().parents[1]
VALID = {'polygon': [[-1, -1], [1, -1], [1, 1], [-1, 1]],
         'home': [-2, 0], 'altitude': 1.6,
         'footprint_width': 2, 'side_overlap': 80}


class DrawnMissionTests(unittest.TestCase):
    def test_preview_starts_at_closest_point_and_rejects_crossing(self):
        mission, preview = make_preview(VALID)
        self.assertEqual(mission['home'], [-2., 0.])
        self.assertEqual(preview['entry'], [-1., 0.])
        self.assertEqual(preview['legs'][0]['kind'], 'perimeter')
        self.assertTrue(any(leg['kind'] == 'sweep' for leg in preview['legs']))
        crossed = {**VALID, 'polygon': [[-1,-1], [1,1], [-1,1], [1,-1]]}
        with self.assertRaisesRegex(ValueError, 'cruzar|área'):
            make_preview(crossed)

    def test_http_preview_and_start_save_for_flight_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / 'mission.json'
            mission, _ = make_preview(VALID)
            save_mission(destination, mission)
            preview = Path(directory) / 'preview.svg'
            result = subprocess.run([
                sys.executable, '-B', str(ROOT / 'scripts' / 'zigzag_roi.py'),
                '--shape', 'drawn',
                '--polygon-json', str(destination), '--preview', str(preview),
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('<svg', preview.read_text())
            destination.unlink()
            try:
                server = HTTPServer(('127.0.0.1', 0), BaseHTTPRequestHandler)
            except PermissionError:
                self.skipTest('A sandbox não permite abrir portas locais')
            server.started = False
            server.RequestHandlerClass = handler_for(destination, server)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f'http://127.0.0.1:{server.server_port}'

            def post(path, body):
                request = Request(base + path, data=json.dumps(body).encode(),
                                  headers={'Content-Type': 'application/json'}, method='POST')
                return urlopen(request, timeout=3)

            try:
                with urlopen(base + '/', timeout=3) as response:
                    self.assertIn(b'Marcar v', response.read())
                with post('/api/preview', VALID) as response:
                    self.assertEqual(json.load(response)['entry'], [-1., 0.])
                self.assertFalse(destination.exists())
                with self.assertRaises(HTTPError) as error:
                    post('/api/start', {**VALID, 'polygon': [[0,0], [1,1], [0,0]]})
                self.assertEqual(error.exception.code, 400)
                error.exception.close()
                self.assertFalse(destination.exists())
                with post('/api/start', VALID) as response:
                    self.assertIn('faixas', json.load(response)['summary'])
                self.assertTrue(server.started)
                self.assertEqual(json.loads(destination.read_text())['polygon'], VALID['polygon'])
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=3)


if __name__ == '__main__':
    unittest.main()

"""Isolate synchronous Gazebo RPCs from Python sensor callbacks.

The Harmonic Python binding can hold the GIL during request_raw while a
transport callback waits for it. A small command-only process prevents that
deadlock without dropping sensor frames or shelling out for every pose.
"""
import atexit
import base64
import json
from pathlib import Path
import select
import subprocess
import sys


class CommandNode:
    def __init__(self):
        from gz.transport13 import Node
        self.transport = Node()
        self.worker = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker'],
                                       stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        atexit.register(self.close)

    def __getattr__(self, name):
        return getattr(self.transport, name)

    def request(self, service, message, request_type, response_type, timeout):
        payload = {'service': service, 'request_type': request_type.DESCRIPTOR.full_name,
                   'response_type': response_type.DESCRIPTOR.full_name, 'timeout': timeout,
                   'data': base64.b64encode(message.SerializeToString()).decode('ascii')}
        self.worker.stdin.write(json.dumps(payload) + '\n')
        self.worker.stdin.flush()
        if not select.select([self.worker.stdout], [], [], timeout/1000 + 3)[0]:
            self.close()
            raise RuntimeError('Processo de comandos Gazebo sem resposta; missão interrompida')
        line = self.worker.stdout.readline()
        if not line:
            raise RuntimeError('Processo de comandos Gazebo encerrou inesperadamente')
        reply = json.loads(line)
        if 'error' in reply:
            raise RuntimeError(reply['error'])
        response = response_type()
        response.ParseFromString(base64.b64decode(reply['data']))
        return reply['executed'], response

    def close(self):
        if self.worker.poll() is None:
            self.worker.stdin.close()
            try:
                self.worker.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.worker.terminate()
                self.worker.wait(timeout=3)


def worker():
    from gz.transport13 import Node
    node = Node()
    for line in sys.stdin:
        try:
            payload = json.loads(line)
            executed, data = node.request_raw(payload['service'], base64.b64decode(payload['data']),
                                             payload['request_type'], payload['response_type'], payload['timeout'])
            reply = {'executed': executed, 'data': base64.b64encode(data).decode('ascii')}
        except Exception as error:
            reply = {'error': str(error)}
        print(json.dumps(reply), flush=True)


if __name__ == '__main__':
    worker()

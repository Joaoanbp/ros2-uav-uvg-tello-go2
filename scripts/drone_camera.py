#!/usr/bin/env python3
"""Câmeras RGB do drone via Gazebo Transport, com visualização Qt ao vivo."""

import argparse
import threading
import time

import numpy as np

from check_graphics import configure_live_plot


CAMERAS = (
    ('Frontal RGB — inclinada 25°', '/tello/camera/front/image'),
    ('Inferior — vista do solo', '/tello/camera/down/image'),
)


def decode_image(message):
    """Decode gz.msgs.Image, respecting row stride and byte-channel order.

    Gazebo pixel formats: L_INT8=1, RGB_INT8=3, RGBA_INT8=4, BGRA_INT8=5,
    BGR_INT8=8. Reject unsupported depths instead of silently corrupting colors.
    """
    channels = {1: 1, 3: 3, 4: 4, 5: 4, 8: 3}.get(message.pixel_format_type)
    if channels is None:
        raise ValueError(f'Formato de imagem não suportado: {message.pixel_format_type}')
    width, height = message.width, message.height
    if width <= 0 or height <= 0:
        raise ValueError('Imagem sem dimensões válidas')
    row_bytes = width * channels
    step = message.step or row_bytes
    if step < row_bytes or len(message.data) < step * height:
        raise ValueError('Imagem truncada ou stride inválido')
    rows = np.frombuffer(message.data, dtype=np.uint8, count=step * height).reshape(height, step)
    pixels = rows[:, :row_bytes].reshape(height, width, channels)
    if message.pixel_format_type in (5, 8):
        pixels = pixels[:, :, [2, 1, 0]]
    elif channels == 1:
        pixels = np.repeat(pixels, 3, axis=2)
    else:
        pixels = pixels[:, :, :3]
    return pixels.copy()


class CameraStream:
    """Keep just the latest frame: transport callbacks never touch Qt."""

    def __init__(self, node, topic):
        from gz.msgs10.image_pb2 import Image

        self.topic = topic
        self.lock = threading.Lock()
        self.frame = None
        self.received_at = None
        self.count = 0
        self.error = None
        if not node.subscribe(Image, topic, self.receive):
            raise RuntimeError(f'Não foi possível assinar a câmera: {topic}')

    def receive(self, message):
        try:
            frame = decode_image(message)
        except ValueError as error:
            with self.lock:
                self.error = str(error)
            return
        with self.lock:
            self.frame = frame
            self.received_at = time.monotonic()
            self.count += 1
            self.error = None

    def snapshot(self):
        with self.lock:
            return self.frame, self.received_at, self.count, self.error


class CameraPanel:
    """Two camera views, usable in the flight map or a standalone window."""

    def __init__(self, fig, axes, node):
        self.fig = fig
        self.front_axis = axes[0]
        self.object_artists = []
        self.object_key = None
        self.streams = []
        self.artists = []
        self.labels = []
        self.shown_counts = [0, 0]
        for ax, (name, topic) in zip(axes, CAMERAS):
            stream = CameraStream(node, topic)
            self.streams.append(stream)
            self.artists.append(ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8)))
            ax.set_title(name, fontsize=11)
            ax.set_axis_off()
            self.labels.append(ax.text(
                0.02, 0.04, 'Aguardando imagem do Gazebo…', transform=ax.transAxes,
                color='white', fontsize=9,
                bbox={'facecolor': 'black', 'alpha': 0.65, 'edgecolor': 'none'},
            ))
        self.timer = fig.canvas.new_timer(interval=100)
        self.timer.add_callback(self.update)
        self.timer.start()
        fig.canvas.mpl_connect('close_event', lambda event: self.timer.stop())

    def set_objects(self, objects):
        """Overlay the last YOLO observation, not a new detector on the GUI thread."""
        from matplotlib.patches import Rectangle
        key = repr(objects)
        if key == self.object_key:
            return
        self.object_key = key
        for artist in self.object_artists:
            artist.remove()
        self.object_artists = []
        for obj in objects:
            x1, y1, x2, y2 = obj['bbox']
            rectangle = Rectangle((x1,y1), x2-x1, y2-y1, fill=False, edgecolor='lime', linewidth=2)
            self.front_axis.add_patch(rectangle)
            label = self.front_axis.text(x1, max(10, y1-5),
                f"YOLO {obj['confidence']:.0%} · última observação", color='lime', fontsize=8,
                bbox={'facecolor':'black','alpha':.6,'edgecolor':'none'})
            self.object_artists.extend((rectangle, label))

    @property
    def ready(self):
        now = time.monotonic()
        return all(received is not None and now - received < 3 and not error
                   for _, received, _, error in (s.snapshot() for s in self.streams))

    def update(self):
        now = time.monotonic()
        for index, stream in enumerate(self.streams):
            frame, received, count, error = stream.snapshot()
            if frame is not None and count != self.shown_counts[index]:
                self.artists[index].set_data(frame)
                self.shown_counts[index] = count
            if error:
                label = error
            elif received is None:
                label = 'Aguardando imagem do Gazebo…'
            elif now - received > 2:
                label = f'Sem novas imagens há {now - received:.1f} s — verifique a simulação'
            else:
                label = f'Ao vivo · {frame.shape[1]} × {frame.shape[0]} · quadro {count}'
            self.labels[index].set_text(label)
        self.fig.canvas.draw_idle()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    try:
        plt = configure_live_plot()
    except (ImportError, RuntimeError, OSError) as error:
        parser.error(str(error))
    from gz.transport13 import Node

    node = Node()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.canvas.manager.set_window_title('Drone — câmeras ao vivo')
    panel = CameraPanel(fig, axes, node)
    fig.tight_layout()
    plt.show()
    panel.timer.stop()


if __name__ == '__main__':
    main()

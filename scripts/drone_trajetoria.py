#!/usr/bin/env python3
"""Trajetória do drone: exibe posições do Tello e pode executar o zigzag."""

import argparse
import os
from collections import deque
from pathlib import Path
import subprocess
import sys
import threading
import time

from matplotlib.patches import Polygon, Rectangle
from matplotlib.transforms import Affine2D
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.transport13 import Node
from coverage_plan import add_coverage_arguments, coverage_lanes, coverage_summary, finite_coordinate
from roi_plan import add_roi_arguments, apply_drawn_mission, plan_from_args, plan_summary
from check_graphics import configure_live_plot
from drone_camera import CameraPanel
from depth_navigation import PerceptionLayer, message_prototypes
from coverage_plan import add_navigation_arguments, rectangular_waypoints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-search", action="store_true")
    parser.add_argument("--no-camera", action="store_true", help="Exibe somente o mapa da trajetória")
    parser.add_argument("--output", type=Path, help="Save the plot when the search ends")
    parser.add_argument("--start-x", type=finite_coordinate, default=-7.0, help="Home X in meters (default: -7)")
    parser.add_argument("--start-y", type=finite_coordinate, default=-5.0, help="Home Y in meters (default: -5)")
    parser.add_argument("--map-rotation", type=int, choices=(0, 90, 180, 270), default=90,
                        help="Clockwise map rotation in degrees (default: 90)")
    add_coverage_arguments(parser)
    add_navigation_arguments(parser)
    add_roi_arguments(parser, include_shape=True)
    args = parser.parse_args()
    plan = None
    try:
        if args.shape == 'drawn':
            apply_drawn_mission(args)
        if args.shape == 'rectangle':
            lanes = coverage_lanes(args.footprint_width, args.side_overlap)
            boundary = [(-4, -3), (-4, 3), (4, 3), (4, -3)]
            summary = coverage_summary(args.footprint_width, args.side_overlap, lanes)
        else:
            plan = plan_from_args(args, args.shape)
            boundary = plan.polygon
            summary = plan_summary(plan)
    except ValueError as error:
        parser.error(str(error))
    try:
        plt = configure_live_plot()
    except (ImportError, RuntimeError, OSError) as error:
        parser.error(str(error))
    pending = deque(maxlen=10000)
    lock = threading.Lock()

    def receive(message):
        for pose in message.pose:
            if pose.name == "tello":
                with lock:
                    pending.append((pose.position.x, pose.position.y, pose.position.z))

    node = Node()
    prototypes = message_prototypes(Pose_V)
    node.subscribe(Pose_V, "/world/tello_only/pose/info", receive)
    cameras = None
    if args.no_camera:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = plt.figure(figsize=(14, 8))
        grid = fig.add_gridspec(2, 2, width_ratios=(1.15, 1), hspace=0.20)
        ax = fig.add_subplot(grid[:, 0])
        cameras = CameraPanel(fig, [fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, 1])], node)
    fig.canvas.manager.set_window_title("Tello — trajetória e câmeras ao vivo")
    labels = {0: ("X (m)", "Y (m)"), 90: ("Y (m)", "−X (m)"),
              180: ("−X (m)", "−Y (m)"), 270: ("−Y (m)", "X (m)")}
    xlabel, ylabel = labels[args.map_rotation]
    ax.set(title=f"Tello zigzag {args.shape} — live position ({args.map_rotation}° clockwise)",
           xlabel=xlabel, ylabel=ylabel)
    ax.set_aspect("equal", adjustable="box")
    rotation = Affine2D().rotate_deg(-args.map_rotation)
    map_transform = rotation + ax.transData
    obstacle_layer = PerceptionLayer(ax, map_transform, node)

    def fit_map(points):
        displayed = rotation.transform([*boundary,
                                        (args.start_x, args.start_y), *points])
        ax.set_xlim(displayed[:, 0].min() - 1, displayed[:, 0].max() + 1)
        ax.set_ylim(displayed[:, 1].min() - 1, displayed[:, 1].max() + 1)

    fit_map([])
    ax.grid(True, alpha=0.3)
    if plan is None:
        for index, y in enumerate(lanes):
            ax.add_patch(Rectangle((-4, y - args.footprint_width / 2), 8, args.footprint_width,
                                   facecolor="tab:green", alpha=0.10, transform=map_transform,
                                   label="Planned footprint strips" if index == 0 else None))
            ax.plot([-4, 4], [y, y], ":", color="tab:green", linewidth=0.8, transform=map_transform)
    else:
        ax.add_patch(Polygon(boundary, facecolor='tab:green', alpha=0.10, transform=map_transform))
        seen = set()
        styles = {'perimeter': ('tab:green', '-', 'Perimeter pass'),
                  'sweep': ('tab:green', ':', 'Planned zigzag'),
                  'transit': ('tab:orange', '--', 'Connections inside ROI')}
        for leg in plan.legs:
            color, style, label = styles[leg.kind]
            ax.plot([leg.start[0], leg.end[0]], [leg.start[1], leg.end[1]],
                    color=color, linestyle=style, linewidth=0.8, transform=map_transform,
                    label=label if leg.kind not in seen else None)
            seen.add(leg.kind)
    fig.suptitle(summary + ('\nGuias nominais · cobertura adaptativa pelo solo observado'
                           if args.coverage_mode == 'adaptive' else ''), fontsize=9)
    trail, = ax.plot([], [], color="tab:blue", linewidth=2, label="Traveled path", transform=map_transform)
    marker, = ax.plot([], [], "o", color="tab:orange", markersize=9, label="Tello", transform=map_transform)
    ax.plot(args.start_x, args.start_y, "+", color="black", markersize=12, label="Home", transform=map_transform)
    entry = (rectangular_waypoints(lanes, (args.start_x, args.start_y), args.altitude)[0][1]
             if plan is None else plan.legs[0].start)
    ax.plot(entry[0], entry[1], 'D', color='seagreen', markersize=6,
            label='Início da missão', transform=map_transform)
    closed_boundary = boundary + boundary[:1]
    ax.plot([p[0] for p in closed_boundary], [p[1] for p in closed_boundary],
            "--", color="gray", alpha=0.6, label="Search area", transform=map_transform)
    ax.legend(loc="upper right", fontsize=8)
    status = ax.text(0.02, 0.02, "Waiting for Gazebo position…", transform=ax.transAxes)
    xs, ys = [], []
    last = None
    mission = None
    completed = False
    started = time.monotonic()
    finished_at = None
    collecting = not args.run_search
    startup_error = None

    def update():
        nonlocal last, mission, completed, finished_at, collecting, startup_error
        obstacle_points = obstacle_layer.update()
        if cameras is not None:
            with obstacle_layer.lock:
                observation = obstacle_layer.latest
            cameras.set_objects(observation.get('objects', []) if observation is not None and
                                observation.get('source') == 'rgb_yolo_known_pillar' else [])
        with lock:
            samples = list(pending)
            pending.clear()
        for position in samples:
            if not collecting:
                # Ignore the old pose and the initial repositioning jump.
                if mission is None or any(abs(a - b) > 0.01 for a, b in zip(position, (args.start_x, args.start_y, 0.09))):
                    continue
                collecting = True
            if position != last:
                xs.append(position[0])
                ys.append(position[1])
                last = position
        if last is not None:
            trail.set_data(xs, ys)
            marker.set_data([last[0]], [last[1]])
            state = "Search complete" if completed else "Live"
            if completed and mission is not None and mission.returncode:
                state = f'Search failed (exit {mission.returncode})'
            status.set_text(f"{state}  |  X {last[0]:.2f} m  Y {last[1]:.2f} m  Altitude {last[2]:.2f} m")
        fit_map([*zip(xs, ys), *obstacle_points])
        if args.run_search and mission is None and time.monotonic() - started >= 2:
            if cameras is not None and not cameras.ready:
                status.set_text('Aguardando imagens das duas câmeras para iniciar…')
                if time.monotonic() - started > 30:
                    startup_error = ('As câmeras não publicaram imagens em 30 s. '
                                     'Verifique se o mundo está rodando e o log do Gazebo.')
                    timer.stop()
                    plt.close(fig)
                return
            script = 'drone_zigzag.py' if plan is None else 'zigzag_roi.py'
            if args.shape == 'drawn':
                region_args = ['--polygon-json', str(args.polygon_json)]
            elif plan is None:
                region_args = []
            else:
                region_args = ['--radius', str(args.radius), '--center-x', str(args.center_x),
                               '--center-y', str(args.center_y), '--seed', str(args.seed)]
            mission_env = os.environ.copy()
            if args.depth_source in ('rgb', 'yolo') and mission_env.pop('TELLO_GUI_SYSTEM_PYTHON', None):
                # The launcher isolates the Qt dashboard from pip packages.
                # The RGB inference process needs the user's installed torch.
                mission_env.pop('PYTHONNOUSERSITE', None)
                mission_env['PYTHONPATH'] = os.pathsep.join(
                    p for p in mission_env.get('PYTHONPATH', '').split(os.pathsep)
                    if p and p != '/usr/lib/python3/dist-packages')
            mission = subprocess.Popen([
                sys.executable, str(Path(__file__).with_name(script)),
                *([] if plan is None else ['--shape', args.shape]),
                "--start-x", str(args.start_x), "--start-y", str(args.start_y),
                "--side-overlap", str(args.side_overlap),
                "--footprint-width", str(args.footprint_width),
                "--coverage-mode", args.coverage_mode,
                "--altitude", str(args.altitude),
                "--depth-source", args.depth_source,
                "--yolo-checkpoint", args.yolo_checkpoint,
                "--yolo-profile", args.yolo_profile,
                "--depth-repository", args.depth_repository,
                "--depth-checkpoint", args.depth_checkpoint,
                "--depth-device", args.depth_device,
                "--depth-input-size", str(args.depth_input_size),
                "--depth-cpu-threads", str(args.depth_cpu_threads),
                *(['--evaluation-dir', args.evaluation_dir] if args.evaluation_dir else []),
                *region_args,
            ], env=mission_env)
        if mission is not None and mission.poll() is not None and not completed:
            if finished_at is None:
                finished_at = time.monotonic()
            # Allow the last measured pose to arrive before saving.
            if time.monotonic() - finished_at >= 1:
                completed = True
                if mission.returncode:
                    status.set_text(f"Search failed (exit {mission.returncode})")
                elif last is None:
                    status.set_text("Search ended, but no Gazebo positions were received")
                else:
                    status.set_text(f"Search complete  |  Altitude {last[2]:.2f} m")
                    print(f"Recorded {len(xs)} positions; final position: {last}", flush=True)
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(args.output, dpi=160)
                    print(f"Saved plot: {args.output}", flush=True)
                # Keep the obstacle overlay updating after the drone lands.
        fig.canvas.draw_idle()

    timer = fig.canvas.new_timer(interval=100)
    timer.add_callback(update)
    timer.start()
    print("Live plot subscribed to /world/tello_only/pose/info", flush=True)
    try:
        plt.show()
    finally:
        timer.stop()
    if startup_error:
        raise SystemExit(startup_error)
    # Closing the graph does not interrupt a running flight demo.
    if mission is not None:
        mission.wait()


if __name__ == "__main__":
    main()

# ROS 2 UAV/UGV: Tello + Unitree Go2 EDU

Gazebo Harmonic research environment containing:

- a lightweight DJI Tello approximation connected to ArduPilot SITL over UDP
  port `9002`;
- an articulated 12-DOF Unitree Go2 EDU approximation with IMU, front camera,
  and lidar sensors.

The models are intentionally self-contained (primitive geometry, no downloaded
assets). Their geometry and inertial parameters are useful for integration and
algorithm prototyping, but must be identified/calibrated before results are
claimed to represent the physical robots.

## Run

Create the local Docker configuration. The group IDs must match the host so the
container can access the GPU:

```bash
cp docker/.env.example docker/.env
id -u
id -g
getent group video render
```

Update `docker/.env` with the values printed above. This local file is ignored
by Git.

Build and start the development container:

```bash
docker compose -f docker/compose.yaml up -d --build
```

Restore the pinned third-party dependencies and build the ArduPilot Gazebo
plugin inside the container:

```bash
docker compose -f docker/compose.yaml exec robotics bash
cd /workspace
vcs import . < dependencies.repos
git -C code/ardupilot submodule update --init --recursive
cmake -S code/ardupilot_gazebo -B code/ardupilot_gazebo/build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build code/ardupilot_gazebo/build --parallel
exit
```

The dependency source trees under `code/` are intentionally ignored by Git;
`dependencies.repos` records their upstream URLs and exact revisions.

Start the combined Gazebo world:

```bash
./scripts/run_simulation.sh
```

For server-only operation:

```bash
./scripts/run_simulation.sh --headless
```

In a second terminal, verify both models and the ArduPilot socket:

```bash
./scripts/check_simulation.sh
```

Start ArduPilot SITL for the Tello approximation in another container shell:

```bash
docker compose -f docker/compose.yaml exec robotics bash
cd /workspace/code/ardupilot/ArduCopter
../Tools/autotest/sim_vehicle.py -v ArduCopter -f gazebo-iris --model JSON --console --map
```

The current `gazebo-iris` ArduPilot parameter set is a starting point. Tune the
mass, rotor aerodynamics, and flight-controller parameters before flight-control
experiments.

## Inspect UDP port 9002

Port 9002 is UDP, not TCP, and both endpoints use loopback because the container
has host networking. Run this on the host:

```bash
sudo tcpdump -ni lo 'udp port 9002'
```

To check for the bound Gazebo socket without capturing packets:

```bash
docker compose -f docker/compose.yaml exec robotics ss -lunp | grep ':9002'
```

## Simple motion demo

With the combined world running, execute:

```bash
./scripts/simple_mission.sh
```

This pauses physics, resets the models, raises the Tello to an altitude of 5 m,
moves the Go2 forward 2 m, and leaves Gazebo paused at the final positions. This
is a kinematic integration smoke test: poses are commanded directly, so it does
not validate ArduPilot flight control or a Unitree walking controller.

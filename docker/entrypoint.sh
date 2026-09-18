#!/usr/bin/env bash

set -e


# ------------------------------------------------------------
# ROS 2 Humble
# ------------------------------------------------------------

source /opt/ros/humble/setup.bash


# ------------------------------------------------------------
# Workspace ROS 2 do projeto
# ------------------------------------------------------------

if [ -f /workspace/ros2_ws/install/setup.bash ]; then
    source /workspace/ros2_ws/install/setup.bash
fi


# ------------------------------------------------------------
# Unitree ROS 2
# ------------------------------------------------------------

if [ -f /workspace/code/unitree_ros2/cyclonedds_ws/install/setup.bash ]; then
    source /workspace/code/unitree_ros2/cyclonedds_ws/install/setup.bash
fi


# ------------------------------------------------------------
# ArduPilot
# ------------------------------------------------------------

if [ -f "$HOME/.ardupilot_env" ]; then
    source "$HOME/.ardupilot_env"
fi

export PATH="$HOME/.local/bin:$PATH"


# ------------------------------------------------------------
# Middleware ROS 2 / Unitree
# ------------------------------------------------------------

export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"


# ------------------------------------------------------------
# Gazebo
# ------------------------------------------------------------

export GZ_VERSION="${GZ_VERSION:-harmonic}"


# Plugin ArduPilot
if [ -d /workspace/code/ardupilot_gazebo/build ]; then
    export GZ_SIM_SYSTEM_PLUGIN_PATH="/workspace/code/ardupilot_gazebo/build:${GZ_SIM_SYSTEM_PLUGIN_PATH}"
fi


# Modelos e mundos
export GZ_SIM_RESOURCE_PATH="\
/workspace/code/ardupilot_gazebo/models:\
/workspace/code/ardupilot_gazebo/worlds:\
/workspace/simulation/models:\
/workspace/simulation/worlds:\
${GZ_SIM_RESOURCE_PATH}"


# ------------------------------------------------------------
# Qt / Gazebo no Wayland via XWayland
# ------------------------------------------------------------

export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"


# Qt / DDS gostam de ter um runtime directory
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/runtime-${USER}}"

mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"


exec "$@"

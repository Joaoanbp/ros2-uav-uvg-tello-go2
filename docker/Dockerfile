FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive

ARG USERNAME=robot
ARG USER_UID=1000
ARG USER_GID=1000

ENV TZ=Etc/UTC
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

ENV ROS_DISTRO=humble
ENV GZ_VERSION=harmonic
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

SHELL ["/bin/bash", "-c"]


# ============================================================
# Pacotes básicos
# ============================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    locales \
    sudo \
    curl \
    wget \
    ca-certificates \
    gnupg \
    lsb-release \
    software-properties-common \
    git \
    git-lfs \
    build-essential \
    gcc \
    g++ \
    make \
    cmake \
    ninja-build \
    pkg-config \
    ccache \
    gawk \
    unzip \
    zip \
    tar \
    bzip2 \
    nano \
    vim \
    less \
    htop \
    tmux \
    iproute2 \
    iputils-ping \
    net-tools \
    dnsutils \
    usbutils \
    pciutils \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    python3-setuptools \
    python3-wheel \
    libyaml-cpp-dev \
    libeigen3-dev \
    mesa-utils \
    x11-apps \
    libgl1-mesa-dri \
    libglx-mesa0 \
    python3-pexpect \
    rsync \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*


# ============================================================
# Repositório ROS 2
# ============================================================
RUN python3 -m pip install --no-cache-dir --upgrade \
    PyYAML \
    pymavlink \
    MAVProxy \
    future

RUN mkdir -p /usr/share/keyrings \
    && curl -sSL \
       https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
       -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo \
       "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" \
       > /etc/apt/sources.list.d/ros2.list


# ============================================================
# Repositório Gazebo
# ============================================================

RUN curl -fsSL \
    https://packages.osrfoundation.org/gazebo.gpg \
    -o /usr/share/keyrings/pkgs-osrf-archive-keyring.gpg \
    && echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/pkgs-osrf-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable jammy main" \
    > /etc/apt/sources.list.d/gazebo-stable.list


# ============================================================
# ROS 2 Humble + CycloneDDS + ferramentas de desenvolvimento
# ============================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-desktop \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-rosidl-generator-dds-idl \
    ros-humble-xacro \
    ros-humble-robot-state-publisher \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-publisher-gui \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-teleop-twist-keyboard \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    && rm -rf /var/lib/apt/lists/*


# ============================================================
# Gazebo Harmonic + integração ROS 2
# ============================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    gz-harmonic \
    libgz-cmake3-dev \
    libgz-sim8-dev \
    libgz-transport13-dev \
    libgz-msgs10-dev \
    ros-humble-ros-gzharmonic \
    rapidjson-dev \
    libopencv-dev \
    python3-opencv \
    python3-matplotlib \
    python3-pyqt5 \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    gstreamer1.0-gl \
    && rm -rf /var/lib/apt/lists/*


# ============================================================
# Usuário com mesmo UID/GID do host
# ============================================================

RUN set -eux; \
    existing_group="$(getent group "${USER_GID}" | cut -d: -f1 || true)"; \
    if [ -n "${existing_group}" ]; then \
        group_name="${existing_group}"; \
    else \
        groupadd --gid "${USER_GID}" "${USERNAME}"; \
        group_name="${USERNAME}"; \
    fi; \
    existing_user="$(getent passwd "${USER_UID}" | cut -d: -f1 || true)"; \
    if [ -n "${existing_user}" ]; then \
        usermod -l "${USERNAME}" \
                -d "/home/${USERNAME}" \
                -m "${existing_user}"; \
    else \
        useradd \
            --uid "${USER_UID}" \
            --gid "${group_name}" \
            --create-home \
            --shell /bin/bash \
            "${USERNAME}"; \
    fi; \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" \
        > "/etc/sudoers.d/${USERNAME}"; \
    chmod 0440 "/etc/sudoers.d/${USERNAME}"


# ============================================================
# rosdep
# ============================================================

RUN rosdep init || true


# ============================================================
# Diretório de trabalho
# ============================================================

WORKDIR /workspace


# ============================================================
# Entrypoint
# ============================================================

COPY entrypoint.sh /usr/local/bin/robotics-entrypoint

RUN chmod +x /usr/local/bin/robotics-entrypoint

# ROS 2 disponível automaticamente em shells interativos
# ============================================================
# Shell ROS 2
# ============================================================

RUN echo 'source /opt/ros/humble/setup.bash' >> /etc/bash.bashrc && \
    echo 'if [ -f /workspace/ros2_ws/install/setup.bash ]; then source /workspace/ros2_ws/install/setup.bash; fi' \
    >> /etc/bash.bashrc


USER ${USERNAME}


# Atualiza cache rosdep como usuário normal
RUN rosdep update


ENV PATH="/home/${USERNAME}/.local/bin:${PATH}"


ENTRYPOINT ["/usr/local/bin/robotics-entrypoint"]

CMD ["bash"]

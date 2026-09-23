# Ubuntu 26.04 (resolute) + ROS 2 Lyrical Luth userspace for the Jetson, as a container.
# The host stays on JetPack 7.2.1 / Ubuntu 24.04 / Jazzy (see docs/ubuntu26_native_plan.md
# for why the native upgrade is not done in place). Built ON the Jetson (arm64):
#
#   bash hardware/scripts/lyrical_container.sh build     # needs internet or PROXY=...
#   bash hardware/scripts/lyrical_container.sh test      # colcon build + unit tests inside
#   bash hardware/scripts/lyrical_container.sh shell     # devices + GPU runtime, host network
FROM ubuntu:26.04
ARG http_proxy
ARG https_proxy
# Check-Date off: the Jetson has no RTC battery/NTP and boots with a stale clock, which makes
# apt reject fresh Release files as "not valid yet".
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8 ROS_DISTRO=lyrical
RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg lsb-release locales \
        python3 python3-pip python3-venv python3-numpy python3-scipy python3-yaml \
        python3-opencv python3-pil python3-pytest python3-serial python3-transforms3d \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
        > /etc/apt/sources.list.d/ros2.list
RUN apt-get -o Acquire::Check-Valid-Until=false -o Acquire::Check-Date=false update && apt-get install -y --no-install-recommends \
        ros-lyrical-ros-base ros-lyrical-ackermann-msgs ros-lyrical-sensor-msgs \
        ros-lyrical-nav-msgs ros-lyrical-geometry-msgs ros-lyrical-tf2-ros \
        ros-lyrical-ament-index-python python3-colcon-common-extensions python3-rosdep \
    && rm -rf /var/lib/apt/lists/*
# onnxruntime for policy_bridge (CPU; the student is ~1 M params, 1-2 ms on the Orin CPU)
RUN pip3 install --break-system-packages --no-cache-dir onnxruntime osqp || echo "a wheel is unavailable for this Python"
RUN echo 'source /opt/ros/lyrical/setup.bash' >> /root/.bashrc
WORKDIR /ws

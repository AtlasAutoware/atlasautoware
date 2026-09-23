#!/usr/bin/env bash
# ROS 2 Lyrical on Ubuntu 26.04 in a container on the Jetson (host stays JetPack 7.2.1 / Jazzy).
#   build   build atlas/lyrical:26.04 from docker/lyrical/Dockerfile (PROXY=http://127.0.0.1:3128
#           when the car has no internet: open `ssh -R 3128:127.0.0.1:3128 atlas@car` from a laptop
#           running a proxy)
#   test    copy the package into a scratch workspace, colcon build it under Lyrical, run the
#           ROS-free unit tests, and import every node module
#   shell   interactive shell with /dev (lidar, VESC, camera), the NVIDIA runtime and host network
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
IMG="${IMG:-atlas/lyrical:26.04}"
PX=()
if [ -n "${PROXY:-}" ]; then PX=(--build-arg "http_proxy=$PROXY" --build-arg "https_proxy=$PROXY"); fi
case "${1:-test}" in
  build)
    docker build --network host "${PX[@]}" -t "$IMG" "$REPO/docker/lyrical" ;;
  test)
    docker run --rm --network none -v "$REPO:/src:ro" "$IMG" bash -lc '
      set -e; source /opt/ros/lyrical/setup.bash
      echo "== $(. /etc/os-release; echo $PRETTY_NAME) | ROS_DISTRO=$ROS_DISTRO | $(python3 --version)"
      mkdir -p /ws/src && cp -r /src /ws/src/atlasautoware && cd /ws
      rm -rf src/atlasautoware/build src/atlasautoware/install src/atlasautoware/log
      colcon build --packages-select f1tenth_gym_ros 2>&1 | tail -3
      source install/setup.bash
      python3 - <<EOF
import importlib, pkgutil, f1tenth_gym_ros as p
ok, bad = [], []
for m in pkgutil.iter_modules(p.__path__):
    try: importlib.import_module("f1tenth_gym_ros." + m.name); ok.append(m.name)
    except Exception as e: bad.append((m.name, type(e).__name__, str(e)[:80]))
print(f"imported {len(ok)} modules under Lyrical; failed {len(bad)}")
for b in bad: print("  FAIL", *b)
EOF
      cd /ws/src/atlasautoware && python3 -m pytest -q -rfE -p no:cacheprovider tests 2>&1 | grep -E "^(FAILED|ERROR)|passed|failed" | cut -c1-160' ;;
  shell)
    docker run --rm -it --network host --ipc host --runtime nvidia --privileged -v /dev:/dev \
      -v "$REPO:/ws/src/atlasautoware" "$IMG" bash ;;
  *) echo "usage: $0 build|test|shell" >&2; exit 2 ;;
esac

#!/usr/bin/env bash
# deploy_to_car.sh — push the repo to the Jetson and rebuild everything that changed.
# Run from the laptop clone:   hardware/scripts/deploy_to_car.sh [host]
# Default host is the USB-C link; pass 10.42.0.1 when on the AtlasCar hotspot.
#
# Idempotent: safe to run repeatedly. It does NOT start driving anything; it leaves the
# car in remote-pilot mode so you can check the page yourself.
set -e
HOST="${1:-192.168.55.1}"
SRC="$(cd "$(dirname "$0")/../.." && pwd)"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=6 atlas@$HOST"

echo "== reachability =="
$SSH "echo ok, on \$(hostname), IPs: \$(hostname -I)" || {
    echo "cannot reach $HOST. Power the car, plug the USB-C data cable, or join AtlasCar."; exit 1; }

echo "== sync repo -> ~/atlas_ws/src/atlasautoware =="
rsync -a --delete \
      --exclude build/ --exclude install/ --exclude log/ --exclude __pycache__/ \
      --exclude '*.egg-info/' --exclude .git/ \
      -e "ssh -o BatchMode=yes" "$SRC/" "atlas@$HOST:~/atlas_ws/src/atlasautoware/"

echo "== launcher scripts -> ~ =="
$SSH "cp ~/atlas_ws/src/atlasautoware/hardware/scripts/{run_remote,restart_remote,carnet}.sh ~/ 2>/dev/null; chmod +x ~/*.sh"

echo "== f1tenth_stack overrides (joy profile, vesc.yaml, bringup) =="
$SSH "cp ~/atlas_ws/src/atlasautoware/hardware/f1tenth_stack/config/*.yaml ~/f1tenth_ws/src/f1tenth_system/f1tenth_stack/config/ && \
      cp ~/atlas_ws/src/atlasautoware/hardware/f1tenth_stack/launch/bringup_launch.py ~/f1tenth_ws/src/f1tenth_system/f1tenth_stack/launch/ && \
      cd ~/f1tenth_ws && source /opt/ros/humble/setup.bash && colcon build --packages-select f1tenth_stack 2>&1 | grep -E 'Finished|failed'"

echo "== atlasautoware package =="
$SSH "cd ~/atlas_ws && source /opt/ros/humble/setup.bash && colcon build --symlink-install --packages-select f1tenth_gym_ros 2>&1 | grep -E 'Finished|failed'"

echo "== python deps the web UI needs for track pictures =="
$SSH "python3 -c 'import numpy, scipy, PIL' 2>/dev/null && echo '  numpy/scipy/PIL present' || \
      echo '  MISSING: sudo apt install -y python3-scipy python3-pil   (track conversion needs them)'"

echo "== restart remote pilot mode =="
$SSH "bash ~/restart_remote.sh"
sleep 22
$SSH "curl -s -o /dev/null -m 5 -w 'pilot page: http %{http_code}\n' http://127.0.0.1:8080/ ; \
      curl -s -m 5 http://127.0.0.1:8080/auto | head -c 400; echo"

cat <<'EOF'

Deployed. Open the pilot page (http://10.42.0.1:8080/ on the hotspot, or
http://192.168.55.1:8080/ over USB) and check, in this order:

  1. manual drive still works, and releasing the key brakes immediately
     (the joy profile changed: web_pilot now sends the brake-to-zero itself)
  2. the panel's preflight: lidar and VESC should be green; the pose topic will be
     red until localization runs, which is expected
  3. wheels OFF the ground, speed cap 30%, ENGAGE, then test Space, the STOP button,
     and closing the tab. Each must stop the car.
  4. only then, on the floor

TensorRT: build the engine on the car (it is device-specific) once CUDA/TensorRT are
installed, then camera_perception picks it up automatically:
  /usr/src/tensorrt/bin/trtexec --onnx=models/car_yolov8_640.onnx \
      --saveEngine=models/car_yolov8_640.engine --fp16
EOF

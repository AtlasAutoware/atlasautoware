# AtlasAutoware (`atlasautoware`) — High-Level Codebase Overview

## What this repository is

This repository is a **ROS 2 Humble autonomous racing stack** for a 1/10-scale self-driving car (F1TENTH style).  
It supports both:

- **Simulation** via `f1tenth_gym` + ROS bridge
- **Real hardware** (Jetson + OAK-D + RPLidar + VESC)

Core pipeline:

1. Build/obtain a map
2. Optimize a raceline
3. Profile speeds along that raceline
4. Run control (MPC/MAP, safety braking, traction limiting)
5. Add opponent awareness (lidar + optional camera)

---

## Main files to read first

- `/home/runner/work/atlasautoware/atlasautoware/README.md`  
  Best project-level overview of architecture and usage.
- `/home/runner/work/atlasautoware/atlasautoware/package.xml`  
  ROS package metadata + ROS dependencies.
- `/home/runner/work/atlasautoware/atlasautoware/setup.py`  
  Python package setup and all ROS executable entrypoints.
- `/home/runner/work/atlasautoware/atlasautoware/launch/*.py`  
  How components are composed at runtime (sim, mapping, real car bringup).
- `/home/runner/work/atlasautoware/atlasautoware/config/*.yaml`  
  Runtime parameters for sim, hardware, and SLAM mapping.

---

## Main folders and what each is for

- `/home/runner/work/atlasautoware/atlasautoware/f1tenth_gym_ros`  
  **Core source code** (controllers, bridge, perception, hardware drivers, optimization).
- `/home/runner/work/atlasautoware/atlasautoware/launch`  
  ROS 2 launch files for simulation, mapping, and hardware bringup.
- `/home/runner/work/atlasautoware/atlasautoware/config`  
  YAML parameter sets for nodes.
- `/home/runner/work/atlasautoware/atlasautoware/maps`  
  Track map images + ROS map YAML metadata.
- `/home/runner/work/atlasautoware/atlasautoware/racelines`  
  Generated raceline CSVs and overlay images.
- `/home/runner/work/atlasautoware/atlasautoware/tools`  
  Offline utilities (optimize, tune, benchmark, annotate, train detector).
- `/home/runner/work/atlasautoware/atlasautoware/tests`  
  Main Python test suite (logic/controller/hardware math).
- `/home/runner/work/atlasautoware/atlasautoware/test`  
  ROS ament lint/test wrappers (flake8, pep257, copyright).
- `/home/runner/work/atlasautoware/atlasautoware/docs`  
  Technical docs (MPC, hardware, mapping, dashboard, camera perception).
- `/home/runner/work/atlasautoware/atlasautoware/data`  
  Dataset assets and instructions for car detection training.
- `/home/runner/work/atlasautoware/atlasautoware/ui`  
  Lightweight dashboard server + frontend.
- `/home/runner/work/atlasautoware/atlasautoware/.github`  
  CI workflows and issue templates.
- `/home/runner/work/atlasautoware/atlasautoware/resource`  
  ROS package resource marker.

---

## Core code modules (inside `f1tenth_gym_ros`)

- **Simulation bridge**: `gym_bridge.py`
- **Race control nodes**: `raceline_mpc.py`, `race_agent.py`, `pursuit_agent.py`, `opponent_driver.py`
- **Control algorithms**: `mpc_controller.py`, `map_controller.py`, `mpcc_controller.py`
- **Racing intelligence**: `race_brain.py`, `spliner.py`
- **Map/raceline generation**: `raceline_optimizer.py`, `track_learner.py`, `raceline_refiner.py`, `velocity_profiler.py`
- **Hardware I/O**: `drive_node.py`, `rplidar_node.py`, `oakd_camera.py`, `vesc_protocol.py`, `pca9685.py`
- **Perception**: `camera_perception.py`
- **State estimation and sensor correction**: `velocity_ekf.py`, `scan_deskew.py`
- **Mapping helper**: `mapping_driver.py`

---

## Main libraries and frameworks used

### ROS / robotics stack
- `rclpy`, `launch`, `launch_ros`, `tf2_ros`
- ROS message packages: `sensor_msgs`, `nav_msgs`, `geometry_msgs`, `ackermann_msgs`, `visualization_msgs`, `std_msgs`
- `nav2_map_server`, `nav2_lifecycle_manager`, `slam_toolbox`, `rviz2`

### Simulation + math/control
- `f1tenth_gym` (`gym` backend)
- `numpy`, `scipy`, `osqp`, `transforms3d`, `Pillow`, `PyYAML`

### Hardware + perception
- `depthai` (OAK-D)
- `opencv-python` (`cv2.dnn`, optional CUDA)
- Optional TensorRT path (`tensorrt`, `pycuda`)
- `rplidar-roboticia`
- `pyserial`, `smbus2`

### Tooling/testing
- `pytest`
- `ultralytics` (offline YOLO training)
- Docker / docker-compose for containerized runtime

---

## How everything fits together (high level)

- **Sim mode**: `gym_bridge` publishes synthetic scan/odom → controller node publishes `/drive`.
- **Real car mode**: `rplidar_node` + `oakd_camera` + localization feed controller → `drive_node` actuates VESC/servo.
- **Raceline workflow**: map → optimize line → speed profile → controller tracks it.
- **Safety layers**: AEB from lidar, traction governor from IMU, watchdog/arming in drive backend.
- **Opponent racing**: `race_brain` fuses lidar and optional camera opponents, then chooses CRUISE/ATTACK/DEFEND/EVADE.
- **Developer workflow**: tools for benchmarking/tuning and docs that describe each subsystem.

---
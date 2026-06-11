# Camera perception (trained car detector) — real-car add-on

The f1tenth_gym **sim has no camera**, so this runs on the physical car. It
detects the other cars with a trained YOLOv8 model and feeds the *same* race
brain the lidar does — so all the existing attack/defend/evade logic applies.

## Why this design
- **ONNX + `cv2.dnn` for inference** → the car needs only OpenCV, not PyTorch.
- **Training is offline** (`tools/train_car_detector.py`, GPU machine) — this
  laptop has no GPU and the car shouldn't train anyway.
- **Sensor-agnostic output:** `camera_perception.py` emits `race_brain.Opponent`
  objects — identical to the lidar detector. So you can run camera-only, or
  **fuse**: camera answers *"is that a car, and at what bearing?"*, lidar answers
  *"exactly how far?"*. Fusing kills the lidar's wall/car ambiguity and the
  camera's depth error at once.

## End-to-end workflow
1. **Collect** frames while running other cars on track:
   `python3 tools/collect_camera_data.py --topic /camera/color/image_raw --out data/car_images`
2. **Label** the cars (single class `car`) in YOLO format (Roboflow / Label Studio).
3. **Train + export** on a GPU box / Colab:
   `pip install ultralytics && python3 tools/train_car_detector.py --data data.yaml`
   → `best.onnx`
4. **Deploy:** copy to `models/car_yolov8.onnx`, measure camera intrinsics, run:
   `ros2 run ... camera_perception --ros-args -p fx:=<fx> -p cx:=<cx> -p image_topic:=<topic>`

## How a detection becomes a decision
`box_to_relative()` back-projects each YOLO box to a position relative to the car
(pinhole + known car width ≈ 0.30 m → depth; pixel offset → bearing). That feeds
the smoothing tracker (`OpponentDetector`, the alpha-beta filter reused) and then
`RaceStrategist`, which already turns an opponent list into CRUISE / ATTACK /
DEFEND / EVADE + a target line.

## Validated already (offline, no model needed)
- Geometry: a 0.30 m car at 100 px with fx=600 → 1.80 m depth; right-of-centre →
  negative (right) lateral; world transform correct.
- YOLOv8 ONNX output parsing + NMS + image-scale.

## How it's wired into the race (done)
`race_agent` already fuses `/camera_opponents_poses` with the lidar tracker
(`fuse_opponents`: camera bearing/class + lidar range), so camera-confirmed cars
show in the live RViz "thinking" and drive the strategist.

The camera is **also a backup obstacle sensor for safety**: `_nearest_opp_ahead`
checks the fused opponent set in the forward travel cone, and a confirmed car
within `opp_brake_dist` (1.2 m) forces a limp back-off even if the lidar AEB
didn't trip — covering a car in the single-plane lidar's blind spot (below/above
the scan). Tunables: `opp_brake_dist`, `opp_brake_cone` in `race_agent.py`.

On hardware set the odom topic (sim default is the gym's namespaced one):
`ros2 run f1tenth_gym_ros camera_perception --ros-args -p odom_topic:=/odom -p image_topic:=<cam> -p fx:=<fx> -p cx:=<cx>`

## Boundaries: lidar stays primary
Walls/track edges stay on lidar — continuous metric geometry beats per-frame
detection, and `track_learner` builds its map from lidar+SLAM. Using the camera
for *non-car* boundaries the lidar plane misses (curbs, painted lines) needs a
drivable-area **segmentation** model + ground-plane homography — the same
offline-train → ONNX → deploy workflow as the car detector, and a deliberate
later add-on, not part of the map-building path.

# Camera perception (trained car detector) — real-car add-on

The f1tenth_gym **sim has no camera**, so this runs on the physical car. It
detects the other cars with a trained YOLOv8 model and feeds the *same* race
brain the lidar does — so all the existing attack/defend/evade logic applies.

## Why this design
- **TensorRT + CUDA for live inference:** the Jetson process opens a serialized
  engine and calls TensorRT directly. It has no PyTorch, PyCUDA, or ONNX Runtime
  dependency.
- **Training stays offline** (`tools/train_car_detector.py`, GPU machine). The
  ONNX file is an interchange artifact used once when TensorRT builds the
  device-specific engine.
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
4. **Deploy:** copy the 640-pixel export to `models/car_yolov8_640.onnx` and,
   on the Jetson, run `hardware/scripts/install_tensorrt.sh`.
5. **Run:** `ros2 launch f1tenth_gym_ros car_bringup_launch.py use_perception:=true`.

## How a detection becomes a decision
`box_to_relative()` back-projects each YOLO box to a position relative to the car
(pinhole + known car width ≈ 0.30 m → depth; pixel offset → bearing). That feeds
the smoothing tracker (`OpponentDetector`, the alpha-beta filter reused) and then
`RaceStrategist`, which already turns an opponent list into CRUISE / ATTACK /
DEFEND / EVADE + a target line.

## Validation
- Geometry: a 0.30 m car at 100 px with fx=600 → 1.80 m depth; right-of-centre →
  negative (right) lateral; world transform correct.
- YOLOv8 ONNX output parsing + NMS + image-scale.
- TensorRT named-I/O buffer allocation is covered without GPU hardware. The
  complete FP16 path was also built with TensorRT 10.16 and run on an RTX 5070:
  2.30 ms mean end-to-end over 500 blank 640-pixel frames. Across 40 held-out
  car images, detection counts matched ONNX Runtime exactly; the largest box
  coordinate difference was 0.30 pixel and the largest confidence difference
  was 0.0009. Of 185 real Orbbec hard negatives, both backends produced the
  same one raw false positive, and the live node's size and range guards reject
  it. This validates the runtime but is not a Jetson benchmark.

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


## Trained model on the car (2026-09-02)

`models/car_yolov8.onnx` (416 px, retained for portable CPU testing) and
`models/car_yolov8_640.onnx` (TensorRT build input) are a **YOLOv8n
single-class "car" detector** fine-tuned from COCO weights.

**Data** — four public Roboflow Universe sets (all CC BY 4.0), merged to one class with
`tools/merge_car_datasets.py` (drops non-car classes, keeps each source's own train/val split):
[F1 Tenth by F1Tenth Cars](https://universe.roboflow.com/f1tenth-cars/f1-tenth) (167),
[F1Tenth Car Detection by CURC](https://universe.roboflow.com/curc-autonomous-vehicle-project/f1tenth-car-detection) (v6, 4 771 incl. augmentations),
[f110 by pepperpeople](https://universe.roboflow.com/pepperpeople/f110) (483),
[Detect F1Tenth by CU Robotics](https://universe.roboflow.com/cu-robotics-qtbgu/detect-f1tenth) (39)
→ 4 907 train / 547 val, 5 389 boxes. Plus **185 hard negatives captured from this car's Orbbec**
(hands, faces, hallway, bench — no cars; kept off GitHub because they contain people) so the model
stops calling a hand at 0.3 m an opponent. Datasets are not committed (180 MB); download the zips
(YOLOv8 format) and re-run the merge script to rebuild.

**Training** — `tools/train_car_detector_gpu.py` (RTX 5070, 50 epochs, 17 min) then a 20-epoch
fine-tune with the negatives. Held-out val (547 images): **mAP50 0.97, mAP50-95 0.75,
precision 0.99, recall 0.97** before the negatives; see `finetune.log` metrics in the commit for after.

**Runtime on the Jetson** — `hardware/scripts/install_tensorrt.sh` installs the
JetPack-native TensorRT packages and calls `build_tensorrt_engine.sh`. The
latter builds FP16 into `~/.cache/atlasautoware/car_yolov8_640.engine`, checks
the serialized engine with `trtexec`, then measures the actual detector path.
The ROS hardware configuration requires that engine. ONNX Runtime remains an
explicit development backend and is no longer part of the car launch path.

The old CPU baseline was 58 ms/frame at 416 pixels and 124 ms/frame at 640
pixels with four ONNX Runtime threads. Keep those figures for comparison when
recording the Jetson TensorRT result.

**Launch** — `ros2 launch f1tenth_gym_ros car_bringup_launch.py use_perception:=true` (off by
default; only `race_agent` consumes `/camera_opponents_poses`). Params in `config/hardware.yaml`:
Orbbec intrinsics fx 460.5 / cx 326.4 (640×480), `car_width 0.30`, `conf 0.5`, and two plausibility
guards — `min_range 0.8` m and `max_box_frac 0.6` — because a box implying a car closer than that
or filling the frame is a hand or a face, and inside 0.8 m the lidar owns the problem anyway.

# F1TENTH car-detector dataset — quickstart

Turnkey path from raw camera frames to a deployable ONNX model.

```
data/
  car_images/                 # raw frames dumped by collect_camera_data.py
  car_dataset/
    data.yaml                 # YOLOv8 config (1 class: car)
    images/{train,val}/       # labeled images go here
    labels/{train,val}/       # matching YOLO .txt labels (same basename)
```

## 1. Collect (on the car)
```
python3 tools/collect_camera_data.py --topic /camera/color/image_raw \
    --out data/car_images --every 5
```
Drive several laps with the other car(s) on track, varied lighting/angles. A few
hundred frames is plenty for one transfer-learned class.

## 2. Label
Label the other cars as class `car` in **YOLO format**. Fastest options:
- **Roboflow** (web) — draw boxes, export "YOLOv8", it gives images+labels+data.yaml.
- **Label Studio** or **labelImg** (offline) — export YOLO txt.

Each label file is one row per car: `0 cx cy w h` (normalized 0–1). Put images in
`images/train` (≈80%) and `images/val` (≈20%), labels alongside in `labels/`.

## 3. Train + export (GPU machine / Colab)
```
pip install ultralytics
python3 tools/train_car_detector.py --data data/car_dataset/data.yaml --epochs 100
```
→ `runs/detect/train/weights/best.onnx`

## 4. Deploy (on the car)
```
mkdir -p models && cp <best.onnx> models/car_yolov8.onnx
# measure camera intrinsics once (fx, cx); then:
ros2 run f1tenth_gym_ros camera_perception --ros-args \
    -p image_topic:=/camera/color/image_raw -p fx:=<fx> -p cx:=<cx>
```
The race agent auto-fuses `/camera_opponents_poses` with lidar when the camera is
live (see `docs/camera_perception.md`); with no camera it runs lidar-only.

## Tips
- Start from `yolov8n.pt` (nano) — fast on a Jetson, ample for one class.
- More varied data >> more epochs. Include partial / distant / blurred cars.
- Re-measure intrinsics if you change the camera or resolution.

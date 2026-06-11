"""
Train + export the F1TENTH car detector (YOLOv8) — run on a GPU machine.
========================================================================

This laptop has no GPU, and the car runs inference in OpenCV (ONNX), so training
is a separate offline step.  Easiest path: Google Colab (free GPU) or any CUDA box.

Workflow
--------
1. Collect images on the car:  tools/collect_camera_data.py
2. Label the cars (single class `car`) in YOLO format. Roboflow is fastest and
   exports a ready `data.yaml`. Aim for a few hundred frames across the lighting
   / angles you'll race in — small, transfer-learned models need surprisingly few.
3. Train + export ONNX (this script):
       pip install ultralytics
       python3 tools/train_car_detector.py --data path/to/data.yaml --epochs 100
   -> writes runs/.../weights/best.onnx
4. Copy best.onnx to the car as  models/car_yolov8.onnx  and run
   f1tenth_gym_ros/camera_perception.py (it loads ONNX via cv2.dnn — no torch needed).

Notes
-----
- Start from `yolov8n.pt` (nano): fastest on a Jetson and plenty for one class.
- Measure the camera intrinsics (fx, cx) once and pass them to the node — they
  set the distance/bearing scale for the box->position projection.
- A COCO-pretrained model will NOT detect F1TENTH RC cars out of the box; that is
  exactly why we fine-tune on your own frames.
"""

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='YOLO data.yaml (one class: car)')
    ap.add_argument('--model', default='yolov8n.pt', help='pretrained start weights')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16)
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        raise SystemExit('pip install ultralytics  (run this on a GPU machine)')

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)
    onnx_path = model.export(format='onnx', imgsz=args.imgsz, opset=12)
    print(f'\nExported ONNX: {onnx_path}')
    print('Copy it to the car as models/car_yolov8.onnx')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Train the single-class F1TENTH car detector and export ONNX for the Jetson.

Mirrors tools/train_car_detector.py in the atlasautoware repo (yolov8n start weights,
imgsz 640, ONNX opset 12) with a few extras: patience, cosine LR, and a final
val summary. Output: runs/car/weights/best.pt + car_yolov8.onnx.
NOTE: the __main__ guard matters — Python 3.14 spawns dataloader workers with
forkserver, which re-imports this file.
"""
import argparse, shutil, pathlib, multiprocessing as mp
mp.set_start_method("fork", force=True)   # Python 3.14 defaults to forkserver; torch DataLoader workers want fork


def main():
    from ultralytics import YOLO
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(pathlib.Path.home() / "f1tenth_train/dataset/data.yaml"))
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--name", default="car")
    a = ap.parse_args()
    runs = pathlib.Path.home() / "f1tenth_train/runs"

    model = YOLO(a.model)
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, name=a.name, exist_ok=True,
                project=str(runs), patience=15, cos_lr=True, workers=6, plots=True, verbose=False)
    best = runs / a.name / "weights" / "best.pt"
    m = YOLO(str(best))
    metrics = m.val(data=a.data, imgsz=a.imgsz, batch=a.batch, plots=False, verbose=False)
    print(f"\nVAL  mAP50={metrics.box.map50:.3f}  mAP50-95={metrics.box.map:.3f}  "
          f"precision={metrics.box.mp:.3f}  recall={metrics.box.mr:.3f}")
    onnx = m.export(format="onnx", imgsz=a.imgsz, opset=12, simplify=True, dynamic=False)
    out = pathlib.Path.home() / "f1tenth_train" / "car_yolov8.onnx"
    shutil.copy(onnx, out); print("ONNX ->", out)


if __name__ == "__main__":
    main()

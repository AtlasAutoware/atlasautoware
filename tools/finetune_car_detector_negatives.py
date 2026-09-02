#!/usr/bin/env python3
"""Add Orbbec hard negatives (empty labels) to the train split and fine-tune from best.pt."""
import multiprocessing as mp; mp.set_start_method("fork", force=True)
import pathlib, shutil, glob


def main():
    from ultralytics import YOLO
    root = pathlib.Path.home() / "f1tenth_train"; ds = root / "dataset"
    negs = sorted(glob.glob(str(root / "negatives/*.jpg")))
    n_val = max(10, len(negs) // 6)                       # keep a few negatives in val too
    for i, f in enumerate(negs):
        split = "val" if i % 6 == 0 else "train"
        dst = ds / "images" / split / ("orbbec_" + pathlib.Path(f).name)
        if not dst.exists():
            shutil.copy(f, dst); (ds / "labels" / split / (dst.stem + ".txt")).write_text("")
    print(f"negatives added: {len(negs)} (val every 6th)")
    m = YOLO(str(root / "runs/car/weights/best.pt"))
    m.train(data=str(ds / "data.yaml"), epochs=20, imgsz=640, batch=32, name="car_ft", exist_ok=True,
            project=str(root / "runs"), lr0=0.002, lrf=0.05, cos_lr=True, warmup_epochs=1, patience=10,
            workers=6, plots=True, verbose=False)
    best = root / "runs/car_ft/weights/best.pt"; m = YOLO(str(best))
    r = m.val(data=str(ds / "data.yaml"), imgsz=640, batch=32, plots=False, verbose=False)
    print(f"\nVAL(ft)  mAP50={r.box.map50:.3f}  mAP50-95={r.box.map:.3f}  precision={r.box.mp:.3f}  recall={r.box.mr:.3f}")
    for sz, name in ((416, "car_yolov8_416.onnx"), (640, "car_yolov8.onnx")):
        p = m.export(format="onnx", imgsz=sz, opset=12, simplify=True, dynamic=False, verbose=False)
        shutil.copy(p, root / name); print("ONNX ->", root / name)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge Roboflow YOLOv8 exports into one single-class 'car' dataset.

Every source class that means "an F1TENTH / RC car" -> class 0. Classes that are not
cars (kiwi-cart, traffic signs, ...) are dropped from the label files; images whose
labels end up empty are kept as negatives only if they had no non-car objects.
"""
import os, re, sys, shutil, zipfile, yaml, collections, pathlib

ROOT = pathlib.Path.home() / "f1tenth_train"
RAW, OUT = ROOT / "raw", ROOT / "dataset"
ZIPS = sorted(pathlib.Path.home().joinpath("Downloads").glob("*.zip"))
ZIPS = [z for z in ZIPS if re.search(r"tenth|f110", z.name, re.I)]
CAR_WORDS = {"car", "cars", "f1tenth", "racecar", "car from the front", "f1 tenth", "f110", "vehicle"}
DROP_WORDS = {"kiwi-cart", "kiwi cart"}

def is_car(name):
    n = name.strip().lower()
    return n in CAR_WORDS or n.isdigit() and n == "0"   # CURC labels one car class literally "0"

stats = collections.Counter(); per_src = {}
if OUT.exists(): shutil.rmtree(OUT)
for split in ("train", "val"):
    (OUT / "images" / split).mkdir(parents=True); (OUT / "labels" / split).mkdir(parents=True)

for z in ZIPS:
    src = re.sub(r"[^a-z0-9]+", "_", z.stem.lower()).strip("_")[:24]
    d = RAW / src
    if not d.exists():
        d.mkdir(parents=True); zipfile.ZipFile(z).extractall(d)
    cfg = yaml.safe_load(open(next(d.rglob("data.yaml"))))
    names = cfg["names"]; names = names if isinstance(names, list) else [names[k] for k in sorted(names)]
    car_ids = {i for i, n in enumerate(names) if is_car(n)}
    drop_ids = {i for i, n in enumerate(names) if n.strip().lower() in DROP_WORDS}
    other_ids = set(range(len(names))) - car_ids - drop_ids
    print(f"{src}: classes={names} -> car={sorted(car_ids)} drop={sorted(drop_ids)} other={sorted(other_ids)}")
    if other_ids: print("   WARNING unmapped classes:", [names[i] for i in other_ids], "(treated as drop)")
    per_src[src] = collections.Counter()
    for rf_split, split in (("train", "train"), ("valid", "val"), ("test", "val")):
        img_dir = d / rf_split / "images"
        if not img_dir.exists(): continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in (".jpg", ".jpeg", ".png"): continue
            lbl = d / rf_split / "labels" / (img.stem + ".txt")
            rows, dropped = [], 0
            if lbl.exists():
                for line in open(lbl):
                    p = line.split()
                    if len(p) < 5: continue
                    cid = int(float(p[0]))
                    if cid in car_ids: rows.append("0 " + " ".join(p[1:5]))
                    else: dropped += 1
            if not rows and dropped: continue          # image only had non-car objects: skip
            new = f"{src}__{img.name}"
            shutil.copy(img, OUT / "images" / split / new)
            open(OUT / "labels" / split / (pathlib.Path(new).stem + ".txt"), "w").write("\n".join(rows) + ("\n" if rows else ""))
            per_src[src][split] += 1; per_src[src]["boxes"] += len(rows); stats[split] += 1
            if not rows: stats["negatives"] += 1

yaml.safe_dump({"path": str(OUT), "train": "images/train", "val": "images/val", "names": {0: "car"}},
               open(OUT / "data.yaml", "w"))
print("\nper source:", dict(per_src))
print("total:", dict(stats)); print("data.yaml ->", OUT / "data.yaml")

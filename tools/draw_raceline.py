"""
Draw an optimized raceline CSV over an original track image.

Uses the map YAML (resolution + origin) to map the raceline's world coordinates
back to pixels, so the line lands exactly on the source drawing.  Line colour =
speed (red = slow / braking, green = fast).
"""

import argparse

import numpy as np
import yaml
from PIL import Image, ImageDraw


def speed_color(v, vmin, vmax):
    t = 0.0 if vmax <= vmin else (v - vmin) / (vmax - vmin)
    t = min(1.0, max(0.0, t))
    if t < 0.5:
        return (255, int(510 * t), 0)
    return (int(255 * (2 - 2 * t)), 255, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--yaml', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--width', type=int, default=4)
    args = ap.parse_args()

    meta = yaml.safe_load(open(args.yaml))
    res = float(meta['resolution'])
    ox, oy = meta['origin'][0], meta['origin'][1]

    base = Image.open(args.image).convert('RGB')
    H = base.height
    draw = ImageDraw.Draw(base)

    import csv as _csv
    xs, ys, sp = [], [], []
    for row in _csv.DictReader(open(args.csv)):
        xs.append(float(row['x'])); ys.append(float(row['y']))
        sp.append(float(row['speed']))
    xs, ys, sp = np.array(xs), np.array(ys), np.array(sp)

    def to_px(x, y):
        return (x / res - ox / res, (H - 1) - (y / res - oy / res))

    pts = [to_px(x, y) for x, y in zip(xs, ys)]
    vmin, vmax = float(sp.min()), float(sp.max())
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        draw.line([a, b], fill=speed_color(sp[i], vmin, vmax), width=args.width)
    base.save(args.out)
    print(f'wrote {args.out}')


if __name__ == '__main__':
    main()

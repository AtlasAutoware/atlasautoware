"""
Annotate a raceline over the track image: number the corners, label apex speeds,
and mark the natural overtaking zones.

Strategy logic
--------------
- A **corner** is a contiguous stretch where the car is off top speed (braking /
  cornering).  Its **apex** is the slowest point; that speed is labelled.
- A **straight** is the fast stretch between corners.  The best place to pass is
  under braking at the END of a long straight (slipstream down the straight, out-
  brake into the next corner) — especially a straight that *follows* a slow corner
  (great exit = bigger run).  The longest such straights are marked "OVERTAKE".
"""

import argparse
import csv as _csv

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


def speed_color(v, vmin, vmax):
    t = 0.0 if vmax <= vmin else (v - vmin) / (vmax - vmin)
    t = min(1.0, max(0.0, t))
    return (255, int(510 * t), 0) if t < 0.5 else (int(255 * (2 - 2 * t)), 255, 0)


def font(sz):
    for p in ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(p, sz)
        except Exception:
            pass
    return ImageFont.load_default()


def runs(mask):
    """Contiguous True runs on a circular array -> list of (start, end) inclusive."""
    n = len(mask)
    if mask.all():
        return [(0, n - 1)]
    if not mask.any():
        return []
    start = 0
    while mask[start]:                      # rotate to a False so runs don't wrap
        start += 1
    out, i = [], start
    for _ in range(n):
        j = (i) % n
        if mask[j] and not mask[(j - 1) % n]:
            s = j
        if mask[j] and not mask[(j + 1) % n]:
            out.append((s, j))
        i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--yaml', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--corner-frac', type=float, default=0.90,
                    help='speed below this fraction of vmax = corner')
    ap.add_argument('--n-overtake', type=int, default=2, help='# overtaking zones to mark')
    args = ap.parse_args()

    meta = yaml.safe_load(open(args.yaml))
    res = float(meta['resolution']); ox, oy = meta['origin'][0], meta['origin'][1]

    xs, ys, sp = [], [], []
    for row in _csv.DictReader(open(args.csv)):
        xs.append(float(row['x'])); ys.append(float(row['y'])); sp.append(float(row['speed']))
    xs, ys, sp = np.array(xs), np.array(ys), np.array(sp)
    n = len(xs)
    ds = np.array([np.hypot(xs[(i+1) % n]-xs[i], ys[(i+1) % n]-ys[i]) for i in range(n)])

    base = Image.open(args.image).convert('RGB')
    H = base.height
    draw = ImageDraw.Draw(base)

    def px(i):
        return (xs[i] / res - ox / res, (H - 1) - (ys[i] / res - oy / res))

    # Racing line, speed-coloured.
    vmin, vmax = float(sp.min()), float(sp.max())
    for i in range(n):
        draw.line([px(i), px((i + 1) % n)], fill=speed_color(sp[i], vmin, vmax), width=4)

    # Corners = slow runs; straights = the gaps between them.
    corner_mask = sp < args.corner_frac * vmax
    corners = runs(corner_mask)
    # order corners by start index (driving order)
    corners.sort()

    f_corner = font(15); f_ot = font(16)
    apex_info = []
    for k, (s, e) in enumerate(corners, 1):
        idxs = [(s + t) % n for t in range((e - s) % n + 1)]
        apex = min(idxs, key=lambda i: sp[i])
        apex_info.append((k, apex, sp[apex]))
        cx, cy = px(apex)
        # marker + label offset up-left so it clears the line
        draw.ellipse([cx-4, cy-4, cx+4, cy+4], fill=(0, 0, 255))
        label = f'T{k}  {sp[apex]:.1f}'
        draw.rectangle([cx+6, cy-20, cx+6+10+8*len(label), cy-4], fill=(255, 255, 255))
        draw.text((cx+9, cy-19), label, fill=(0, 0, 160), font=f_corner)

    # Straights between consecutive corners; rank by length for overtaking.
    straights = []
    for k in range(len(corners)):
        e = corners[k][1]
        s_next = corners[(k + 1) % len(corners)][0]
        seg = [(e + 1 + t) % n for t in range((s_next - e - 1) % n)]
        if not seg:
            continue
        length = float(sum(ds[i] for i in seg))
        straights.append((length, seg, (k % len(corners)) + 1, (k + 1) % len(corners) + 1))
    straights.sort(reverse=True)

    for length, seg, from_t, into_t in straights[:args.n_overtake]:
        # mark the braking zone (end of straight) — where the pass completes
        brake_i = seg[int(len(seg) * 0.82)]
        bx, by = px(brake_i)
        draw.ellipse([bx-7, by-7, bx+7, by+7], outline=(200, 0, 0), width=3)
        draw.text((bx+9, by+2), f'OVERTAKE', fill=(200, 0, 0), font=f_ot)
        draw.text((bx+9, by+20), f'into T{into_t}', fill=(200, 0, 0), font=f_corner)

    # Legend.
    lf = font(14)
    draw.rectangle([6, 6, 250, 78], fill=(255, 255, 255), outline=(0, 0, 0))
    draw.text((12, 10), 'green=fast  red=slow', fill=(0, 0, 0), font=lf)
    draw.text((12, 28), 'T#  = corner + apex speed (m/s)', fill=(0, 0, 160), font=lf)
    draw.text((12, 46), 'red ring = overtake under braking', fill=(200, 0, 0), font=lf)
    draw.text((12, 60), 'late-apex line = strong exit/run', fill=(0, 0, 0), font=lf)

    base.save(args.out)
    print(f'wrote {args.out} | {len(corners)} corners, '
          f'{min(args.n_overtake,len(straights))} overtaking zones')
    for k, apex, v in apex_info:
        print(f'  T{k}: apex {v:.1f} m/s')


if __name__ == '__main__':
    main()

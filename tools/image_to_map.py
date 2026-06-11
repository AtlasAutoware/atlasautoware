"""
Rendered track drawing  ->  clean occupancy map for the raceline optimizer.

A hand/CAD-rendered track picture (white background, two thin boundary lines
forming the racing corridor, plus obstacle dots and a start marker) is not a ROS
occupancy grid.  This tool extracts just the **racing corridor** and writes a
clean map the optimizer can flood-fill:

  - threshold dark pixels (lines + dots + marker) and seal anti-aliased gaps;
  - the corridor is the white region that is NOT the outside (it doesn't touch
    the image border) and that ENCLOSES the largest hole (the infield).  This
    one rule isolates the racing surface and ignores dots, the start box, and
    pockets automatically;
  - output an occupancy PNG (corridor = white/free, everything else =
    black/occupied) + a YAML whose `resolution` is auto-scaled so the corridor
    is ~2.2 m wide (a realistic F1TENTH width, so the optimizer's default car /
    margin sizes let the line use the whole track).

Usage:
  python3 tools/image_to_map.py --image racetrackForComp.png \
      --out-map maps/comp_track.png --out-yaml maps/comp_track.yaml
"""

import argparse
import os

import numpy as np
from PIL import Image
from scipy import ndimage


def extract_corridor(gray, wall_thresh, dilate):
    walls = ndimage.binary_dilation(gray < wall_thresh, iterations=dilate)
    free = ~walls
    lbl, n = ndimage.label(free)
    border = set(lbl[0, :]) | set(lbl[-1, :]) | set(lbl[:, 0]) | set(lbl[:, -1])
    border.discard(0)
    best_hole, best_lbl = -1, None
    for i in range(1, n + 1):
        if i in border:
            continue
        comp = lbl == i
        area = int(comp.sum())
        if area < 300:
            continue
        hole = int(ndimage.binary_fill_holes(comp).sum() - area)
        if hole > best_hole:
            best_hole, best_lbl = hole, i
    if best_lbl is None:
        raise RuntimeError('No enclosed corridor found — try a higher --wall-thresh '
                           'or more --dilate (the boundary lines may have gaps).')
    return lbl == best_lbl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--out-map', required=True)
    ap.add_argument('--out-yaml', required=True)
    ap.add_argument('--wall-thresh', type=int, default=200,
                    help='pixels darker than this are walls/obstacles')
    ap.add_argument('--dilate', type=int, default=1, help='px to seal line gaps')
    ap.add_argument('--track-width', type=float, default=2.2,
                    help='target real track width (m) used to auto-scale resolution')
    args = ap.parse_args()

    gray = np.array(Image.open(args.image).convert('L'))
    corridor = extract_corridor(gray, args.wall_thresh, args.dilate)

    # Auto-scale: make the median corridor width == track-width metres.
    edt = ndimage.distance_transform_edt(corridor)
    mx = ndimage.maximum_filter(edt, 3)
    ridge = corridor & (edt >= mx - 1e-6) & (edt > 1)
    half_px = float(np.median(edt[ridge]))
    resolution = (args.track_width / 2.0) / half_px

    # Clean occupancy map: corridor white (free), everything else black (occ).
    out = np.where(corridor, 255, 0).astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_map)), exist_ok=True)
    Image.fromarray(out, mode='L').save(args.out_map)

    with open(args.out_yaml, 'w') as f:
        f.write(f'image: {os.path.basename(args.out_map)}\n')
        f.write(f'resolution: {resolution:.5f}\n')
        f.write('origin: [0.0, 0.0, 0.0]\n')
        f.write('negate: 0\n')
        f.write('occupied_thresh: 0.65\n')
        f.write('free_thresh: 0.196\n')

    # A guaranteed on-corridor seed (world coords), for the optimizer.
    H = gray.shape[0]
    rs, cs = np.where(corridor)
    k = len(rs) // 2
    seed_x = cs[k] * resolution
    seed_y = ((H - 1) - rs[k]) * resolution
    print(f'corridor px area={int(corridor.sum())} half-width~{half_px:.1f}px')
    print(f'resolution={resolution:.5f} m/px  (track ~{gray.shape[1]*resolution:.0f}'
          f'x{gray.shape[0]*resolution:.0f} m)')
    print(f'wrote {args.out_map} and {args.out_yaml}')
    print(f'SEED {seed_x:.3f} {seed_y:.3f}')


if __name__ == '__main__':
    main()

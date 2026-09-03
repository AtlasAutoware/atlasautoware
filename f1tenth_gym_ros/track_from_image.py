#!/usr/bin/env python3
"""Turn a picture of a track into a ROS occupancy map that build_raceline.py can use.

Two kinds of picture are accepted.

  mode="map"    the image is already an occupancy grid (a SLAM .pgm/.png: white free,
                black occupied, grey unknown). It is copied and given a .yaml with the
                resolution and origin you specify.

  mode="photo"  a photo of a drawn track, a whiteboard sketch, or a screenshot: dark
                ink is wall, light paper is drivable. The image is greyscaled,
                flattened for uneven lighting, thresholded (Otsu), de-specked, and the
                drivable area is taken as the free region that the track loop encloses.
                Everything outside that region becomes unknown, so the raceline
                optimizer cannot wander off the page.

Output: <maps_dir>/<name>.pgm + <name>.yaml, and a preview PNG showing what the car
will actually treat as drivable. Scale comes from `track_width_m`: you say how wide
the driving lane is in real life, the script measures it in pixels and derives
resolution, which is the one number people always get wrong by hand.

CLI:
    python3 track_from_image.py photo.jpg --name my_track --mode photo --track-width 1.0
"""
import argparse, os
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from scipy import ndimage

FREE, UNKNOWN, OCC = 254, 205, 0


def _otsu(gray):
    hist = np.bincount(gray.ravel(), minlength=256).astype(float)
    w = np.cumsum(hist); m = np.cumsum(hist * np.arange(256))
    tot, msum = w[-1], m[-1]
    if tot == 0: return 128
    wb = w[:-1]; wf = tot - wb
    ok = (wb > 0) & (wf > 0)
    if not ok.any(): return 128
    mb = np.where(ok, m[:-1] / np.where(wb == 0, 1, wb), 0)
    mf = np.where(ok, (msum - m[:-1]) / np.where(wf == 0, 1, wf), 0)
    var = wb * wf * (mb - mf) ** 2
    var[~ok] = -1
    return int(np.argmax(var))


def photo_to_grid(img, close_px=3, min_blob_frac=2e-4):
    """RGB/greyscale photo of a drawn track -> uint8 occupancy grid (FREE/OCC/UNKNOWN)."""
    g = np.asarray(ImageOps.grayscale(img), dtype=np.uint8)
    # flatten uneven lighting: divide by a heavily blurred copy (a cheap background model)
    bg = np.asarray(Image.fromarray(g).filter(ImageFilter.GaussianBlur(max(g.shape) / 25.0)), float)
    flat = np.clip(g.astype(float) / np.maximum(bg, 1e-3) * 128.0, 0, 255).astype(np.uint8)
    ink = flat < _otsu(flat)                                  # True = dark = wall
    # close small gaps in hand-drawn lines so the loop is watertight
    if close_px > 0:
        st = np.ones((close_px, close_px), bool)
        ink = ndimage.binary_closing(ink, st)
    # drop specks (dust, pen dots) below a fraction of the image
    lab, n = ndimage.label(ink)
    if n:
        sizes = ndimage.sum(ink, lab, range(1, n + 1))
        small = np.isin(lab, 1 + np.flatnonzero(sizes < min_blob_frac * ink.size))
        ink &= ~small
    free = ~ink
    # the drivable lane is a free component that does NOT touch the border (the border
    # component is the paper around the track)
    lab, n = ndimage.label(free)
    if n == 0:
        raise ValueError('no free space found; is the photo mostly dark?')
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    sizes = {i: int((lab == i).sum()) for i in range(1, n + 1)}
    inner = [i for i in sizes if i not in border]
    if inner:
        keep = max(inner, key=lambda i: sizes[i])             # biggest enclosed region
    else:
        keep = max(sizes, key=lambda i: sizes[i])             # fallback: biggest free area
        print('[warn] no enclosed free region; the track loop may be open in the photo')
    # Everything that is not the lane becomes wall, not "unknown". map_server's trinary
    # thresholds classify the usual 205-grey as *free*, which would let the raceline
    # optimizer drive off the paper and report zero wall clearance.
    grid = np.full(g.shape, OCC, np.uint8)
    grid[lab == keep] = FREE
    return grid


def map_image_to_grid(img):
    """An existing occupancy image (SLAM output) -> the same three-value convention."""
    g = np.asarray(ImageOps.grayscale(img), dtype=np.uint8)
    grid = np.full(g.shape, OCC, np.uint8)          # unknown counts as wall (see photo_to_grid)
    grid[g >= 250] = FREE
    return grid


def lane_width_px(grid):
    """Typical drivable width in px = 2x the median distance-to-wall along the lane ridge.

    The ridge (local maxima of the distance transform) is the lane centreline, where the
    distance to the nearest wall is exactly the half-width. Averaging over all free
    pixels instead would underestimate, since most of them sit off-centre.
    """
    free = grid == FREE
    if not free.any(): return 0.0
    dist = ndimage.distance_transform_edt(free)
    ridge = free & (dist >= ndimage.grey_dilation(dist, size=5) - 1e-6) & (dist > 1)
    d = dist[ridge] if ridge.sum() >= 20 else dist[free]
    return float(2.0 * np.median(d))


def build(image_path, maps_dir, name, mode='photo', track_width_m=1.0, resolution=None,
          origin=None, close_px=3):
    img = Image.open(image_path)
    img = ImageOps.exif_transpose(img)
    grid = photo_to_grid(img, close_px=close_px) if mode == 'photo' else map_image_to_grid(img)
    if resolution is None:
        w_px = lane_width_px(grid)
        if w_px < 3:
            raise ValueError('could not measure the lane width; pass --resolution explicitly')
        resolution = float(track_width_m) / w_px
    H, W = grid.shape
    if origin is None:                                        # put (0,0) at the map centre
        origin = [-W * resolution / 2.0, -H * resolution / 2.0, 0.0]
    os.makedirs(maps_dir, exist_ok=True)
    pgm = os.path.join(maps_dir, f'{name}.pgm')
    Image.fromarray(grid, mode='L').save(pgm)
    yaml_path = os.path.join(maps_dir, f'{name}.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"image: {name}.pgm\nmode: trinary\nresolution: {resolution:.6f}\n"
                f"origin: [{origin[0]:.4f}, {origin[1]:.4f}, {origin[2]:.4f}]\n"
                f"negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n")
    # preview: green drivable, dark wall, grey unknown
    prev = np.zeros((H, W, 3), np.uint8); prev[:] = (60, 60, 60)
    prev[grid == FREE] = (70, 200, 120); prev[grid == OCC] = (25, 25, 25)
    preview = os.path.join(maps_dir, f'{name}_preview.png')
    Image.fromarray(prev).save(preview)
    free_frac = float((grid == FREE).mean())
    return {'map_yaml': yaml_path, 'pgm': pgm, 'preview': preview, 'resolution': resolution,
            'size_px': [W, H], 'origin': origin, 'free_fraction': round(free_frac, 4),
            'lane_width_px': round(lane_width_px(grid), 1),
            'track_m': [round(W * resolution, 2), round(H * resolution, 2)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('image')
    ap.add_argument('--name', required=True)
    ap.add_argument('--maps-dir', default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'maps'))
    ap.add_argument('--mode', choices=['photo', 'map'], default='photo')
    ap.add_argument('--track-width', type=float, default=1.0, help='real lane width, m')
    ap.add_argument('--resolution', type=float, default=None, help='m/px (overrides --track-width)')
    ap.add_argument('--close-px', type=int, default=3)
    a = ap.parse_args()
    info = build(a.image, a.maps_dir, a.name, a.mode, a.track_width, a.resolution, close_px=a.close_px)
    for k, v in info.items():
        print(f'{k:>15}: {v}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""depth_fusion: fold the Gemini 335 depth image into the lidar scan.

The RPLIDAR C1 sees one plane about 11 cm off the floor. A cone, a low box, a ramp lip
or an overhanging shelf edge above or below that plane is invisible to it, and the
emergency brake and planner only know what /scan tells them. The depth camera sees
the volume in front of the car, so:

    depth image ──back-project──► points in base_link ──keep z in [z_min, z_max]──►
    per-bearing minimum range ("virtual scan") ──min() with the lidar per bearing──► /scan_fused

/scan_fused has exactly the lidar's angle layout, so anything that consumes /scan can
consume it unchanged (raceline_mpc scan_topic:=/scan_fused). /scan_depth is the
depth-only virtual scan, for the pilot page and for debugging the geometry. Outside
the camera's field of view the fused scan is just the lidar.

Geometry: camera optical frame (x right, y down, z forward) → base_link (x forward,
y left, z up): forward = Z, left = -X, up = -Y, then the mount offset and pitch.
Floor points fall below z_min and are dropped; that is the whole floor filter, so the
mount height and pitch parameters need to be right (an error of a few degrees in
pitch makes the floor 'rise' into the band at range). `floor_margin` widens z_min at
range to absorb that.
"""
import math, time
import numpy as np


def depth_to_scan(depth_m, fx, fy, cx, cy, n_bins, angle_min, angle_inc, cam_x=0.30,
                  cam_z=0.14, pitch=0.0, z_min=0.03, z_max=0.35, r_min=0.2, r_max=4.0,
                  stride=2, floor_margin=0.02):
    """depth_m: (H,W) float32 metres, 0/NaN = invalid. Returns (ranges[n_bins], n_points).

    ranges is +inf where the camera saw nothing in the band for that bearing.
    """
    H, W = depth_m.shape
    d = depth_m[::stride, ::stride]
    v, u = np.mgrid[0:H:stride, 0:W:stride]
    ok = np.isfinite(d) & (d > 0.05) & (d < r_max * 1.5)
    if not ok.any():
        return np.full(n_bins, np.inf, np.float32), 0
    Z = d[ok].astype(np.float32); U = u[ok].astype(np.float32); V = v[ok].astype(np.float32)
    X = (U - cx) * Z / fx                     # right
    Y = (V - cy) * Z / fy                     # down
    fwd, left, up = Z, -X, -Y                 # optical -> body axes (camera frame)
    if pitch:                                  # positive pitch = camera looking down
        c, s = math.cos(pitch), math.sin(pitch)
        fwd, up = c * fwd + s * up, -s * fwd + c * up
    fwd = fwd + cam_x; up = up + cam_z        # into base_link
    rng = np.hypot(fwd, left)
    band = (up > z_min + floor_margin * rng) & (up < z_max) & (rng > r_min) & (rng < r_max)
    if not band.any():
        return np.full(n_bins, np.inf, np.float32), 0
    ang = np.arctan2(left[band], fwd[band]); rng = rng[band]
    b = np.floor((ang - angle_min) / angle_inc).astype(np.int64)
    inb = (b >= 0) & (b < n_bins)
    out = np.full(n_bins, np.inf, np.float32)
    np.minimum.at(out, b[inb], rng[inb])
    return out, int(inb.sum())


def fuse(lidar, depth_scan, min_hits=1):
    """Per-bearing minimum. lidar 0/inf/nan = no return; depth inf = no return."""
    L = np.asarray(lidar, np.float32).copy()
    Lbad = ~np.isfinite(L) | (L <= 0.0)
    D = np.asarray(depth_scan, np.float32)
    Dok = np.isfinite(D)
    out = L.copy()
    take = Dok & (Lbad | (D < L))
    out[take] = D[take]
    return out, take


# ── ROS node ────────────────────────────────────────────────────────────────────
def main(args=None):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CameraInfo, Image, LaserScan

    class DepthFusion(Node):
        def __init__(self):
            super().__init__('depth_fusion')
            P = (('depth_topic', '/camera/depth/image_raw'), ('info_topic', '/camera/depth/camera_info'),
                 ('scan_topic', '/scan'), ('fused_topic', '/scan_fused'), ('depth_scan_topic', '/scan_depth'),
                 ('cam_x', 0.30), ('cam_z', 0.14), ('pitch_deg', 0.0), ('z_min', 0.03), ('z_max', 0.35),
                 ('r_min', 0.2), ('r_max', 4.0), ('stride', 2), ('floor_margin', 0.02), ('max_age', 0.3))
            for k, v in P: self.declare_parameter(k, v)
            g = lambda n: self.get_parameter(n).value
            self.p = {k: g(k) for k, _ in P}
            self.K = None; self.depth = None; self.depth_t = 0.0; self.last_layout = None
            self.create_subscription(CameraInfo, self.p['info_topic'], self._info, 10)
            self.create_subscription(Image, self.p['depth_topic'], self._depth, qos_profile_sensor_data)
            self.create_subscription(LaserScan, self.p['scan_topic'], self._scan, qos_profile_sensor_data)
            self.pub_f = self.create_publisher(LaserScan, self.p['fused_topic'], 10)
            self.pub_d = self.create_publisher(LaserScan, self.p['depth_scan_topic'], 10)
            self.n_depth = 0; self.n_scan = 0; self.n_take = 0
            self.create_timer(5.0, self._report)
            self.get_logger().info(f"depth fusion: {self.p['depth_topic']} + {self.p['scan_topic']} -> "
                                   f"{self.p['fused_topic']} (band z {self.p['z_min']}..{self.p['z_max']} m)")

        def _info(self, m):
            self.K = (float(m.k[0]), float(m.k[4]), float(m.k[2]), float(m.k[5]))

        def _depth(self, m):
            if m.encoding in ('16UC1', 'mono16'):
                d = np.frombuffer(m.data, np.uint16).reshape(m.height, m.width).astype(np.float32) * 1e-3
            elif m.encoding == '32FC1':
                d = np.frombuffer(m.data, np.float32).reshape(m.height, m.width)
            else:
                return
            self.depth, self.depth_t = d, time.time(); self.n_depth += 1

        def _scan(self, s):
            self.n_scan += 1
            n = len(s.ranges)
            fused = LaserScan(); fused.header = s.header
            for k in ('angle_min', 'angle_max', 'angle_increment', 'time_increment', 'scan_time', 'range_min', 'range_max'):
                setattr(fused, k, getattr(s, k))
            fused.intensities = []
            if self.K is None or self.depth is None or time.time() - self.depth_t > self.p['max_age']:
                fused.ranges = list(s.ranges); self.pub_f.publish(fused); return
            fx, fy, cx, cy = self.K
            dscan, npts = depth_to_scan(self.depth, fx, fy, cx, cy, n, s.angle_min, s.angle_increment,
                                        self.p['cam_x'], self.p['cam_z'], math.radians(self.p['pitch_deg']),
                                        self.p['z_min'], self.p['z_max'], self.p['r_min'], self.p['r_max'],
                                        int(self.p['stride']), self.p['floor_margin'])
            out, take = fuse(np.asarray(s.ranges, np.float32), dscan)
            self.n_take += int(take.sum())
            fused.ranges = [float(x) for x in out]; self.pub_f.publish(fused)
            d = LaserScan(); d.header = s.header
            for k in ('angle_min', 'angle_max', 'angle_increment', 'range_min', 'range_max'):
                setattr(d, k, getattr(s, k))
            d.ranges = [float(x) if np.isfinite(x) else 0.0 for x in dscan]
            self.pub_d.publish(d)

        def _report(self):
            if self.n_scan:
                self.get_logger().info(f'{self.n_depth/5:.0f} depth fps, {self.n_scan/5:.0f} scan Hz, '
                                       f'{self.n_take/max(1,self.n_scan):.0f} bins/scan taken from depth'
                                       + ('' if self.K else '  [no camera_info yet]'))
            self.n_depth = self.n_scan = self.n_take = 0

    rclpy.init(args=args); n = DepthFusion()
    try: rclpy.spin(n)
    except KeyboardInterrupt: pass
    n.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()

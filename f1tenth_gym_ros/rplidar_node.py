"""
RPLidar driver — Slamtec RPLidar -> fixed-grid sensor_msgs/LaserScan on /scan.
===============================================================================

Publishes a fixed-grid 360 deg LaserScan on /scan — the same message the sim
bridge produces, so every racing node (raceline_mpc AEB + opponent detector,
race_brain gap detection, slam_toolbox mapping) runs unmodified on hardware.

Two sources, selected by the `source` parameter:

  source: 'topic'   (RPLIDAR C1 / S-series — the car)
      The official Slamtec SDK driver (`rplidar_ros rplidar_node`) owns the
      serial port and publishes a raw, irregular-grid scan on `raw_scan_topic`
      (default /scan_raw).  This node re-bins it onto `num_bins` even slots,
      applies the mounting offset and the EKF de-skew, and republishes /scan.
      The pip `rplidar` library does NOT speak the C1 protocol (it fails with
      "Descriptor length mismatch"), which is why the SDK driver does the I/O.

  source: 'serial'  (A1 / A2 / A3 — legacy path, unchanged)
      Reads the unit directly via the `rplidar` pip package.

Why a fixed grid: measurements arrive at irregular angles that drift
scan-to-scan; the racing code indexes beams by `angle_min + i*increment`, so
each revolution is binned into `num_bins` even slots (nearest-return wins,
empty slots = inf).

Run:
    ros2 run f1tenth_gym_ros rplidar_node --ros-args --params-file config/hardware.yaml
    # C1 (source: topic) needs the SDK driver alongside:
    ros2 run rplidar_ros rplidar_node --ros-args -p serial_port:=/dev/sensors/rplidar \
        -p serial_baudrate:=460800 -p scan_mode:=Standard -r scan:=/scan_raw
"""

import math
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Pure scan geometry (unit-tested without hardware)
# ─────────────────────────────────────────────────────────────────────────────

def bin_scan(measurements, num_bins, range_min=0.10, range_max=25.0,
             angle_offset=0.0):
    """One revolution of (quality, angle_deg, dist_mm) -> fixed-grid ranges.

    Grid covers [-pi, pi) CCW (ROS convention); RPLidar's clockwise angles are
    negated.  `angle_offset` (rad, CCW) corrects how the unit is mounted.
    Returns float32 ranges, inf where no valid return landed in a bin.
    """
    num_bins = int(num_bins)
    ranges = np.full(num_bins, np.inf, np.float32)
    if not len(measurements):
        return ranges
    m = np.asarray(measurements, np.float64)            # (M, 3) q/angle/dist
    d = m[:, 2] / 1000.0
    keep = (m[:, 0] > 0) & (d >= range_min) & (d <= range_max)
    theta = -np.radians(m[keep, 1]) + angle_offset
    theta = (theta + math.pi) % (2.0 * math.pi) - math.pi
    inc = 2.0 * math.pi / num_bins
    idx = ((theta + math.pi) / inc).astype(np.intp) % num_bins
    np.minimum.at(ranges, idx, d[keep].astype(np.float32))  # nearest return wins
    return ranges


def regrid_scan(ranges, angle_min, angle_increment, num_bins, range_min=0.10,
                range_max=25.0, angle_offset=0.0, time_increment=0.0):
    """Irregular ROS LaserScan -> fixed [-pi, pi) grid of `num_bins` slots.

    Input beams are already in ROS convention (CCW, angle_min + i*inc); this
    only rotates by the mounting `angle_offset` (rad, CCW) and re-bins with
    nearest-return-wins, empty slots inf — identical output contract to
    `bin_scan`, so downstream code cannot tell the two sources apart.

    Returns (ranges, ages).  `ages` is each output bin's measurement time
    relative to the END of the sweep (<= 0 s) for the de-skew step, derived
    from the driver's `time_increment` (beam i fired at i*time_increment after
    the header stamp).  With time_increment == 0 ages are all 0 (no de-skew).
    """
    num_bins = int(num_bins)
    out = np.full(num_bins, np.inf, np.float32)
    ages = np.zeros(num_bins, np.float64)
    r = np.asarray(ranges, np.float64)
    n = r.size
    if n == 0:
        return out, ages
    keep = np.isfinite(r) & (r >= range_min) & (r <= range_max)
    if not keep.any():
        return out, ages
    i_in = np.nonzero(keep)[0]
    theta = float(angle_min) + i_in * float(angle_increment) + angle_offset
    theta = (theta + math.pi) % (2.0 * math.pi) - math.pi
    inc = 2.0 * math.pi / num_bins
    idx = ((theta + math.pi) / inc).astype(np.intp) % num_bins
    d = r[i_in].astype(np.float32)
    np.minimum.at(out, idx, d)                          # nearest return wins
    if time_increment and math.isfinite(time_increment) and time_increment > 0.0:
        # age of the winning return in each bin, relative to sweep end
        t_in = i_in * float(time_increment)
        sweep = (n - 1) * float(time_increment)
        winner = out[idx] == d                           # this beam set the bin
        ages_in = np.full(num_bins, 0.0)
        # later beams overwrite earlier — for ties the newest wins, fine
        ages_in[idx[winner]] = t_in[winner] - sweep
        ages = ages_in
    return out, ages


def _make_node():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
    from sensor_msgs.msg import LaserScan
    from nav_msgs.msg import Odometry
    from scan_deskew import deskew_ranges

    class RPLidarNode(Node):
        def __init__(self):
            super().__init__('rplidar_node')
            self.declare_parameter('source', 'serial')      # serial | topic
            self.declare_parameter('raw_scan_topic', '/scan_raw')
            self.declare_parameter('port', '/dev/ttyUSB0')
            self.declare_parameter('baudrate', 115200)
            self.declare_parameter('scan_topic', '/scan')
            self.declare_parameter('frame_id', 'laser')
            self.declare_parameter('num_bins', 720)
            self.declare_parameter('range_min', 0.10)
            self.declare_parameter('range_max', 25.0)
            self.declare_parameter('angle_offset', 0.0)   # rad, mounting yaw
            self.declare_parameter('deskew_odom_topic', '')  # e.g. /ekf/odom
            p = lambda n: self.get_parameter(n).value     # noqa: E731
            self.source = str(p('source')).lower()
            self.port = p('port')
            self.baud = int(p('baudrate'))
            self.frame = p('frame_id')
            self.num_bins = int(p('num_bins'))
            self.range_min = float(p('range_min'))
            self.range_max = float(p('range_max'))
            self.angle_offset = float(p('angle_offset'))

            qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            self.pub = self.create_publisher(LaserScan, p('scan_topic'), qos)
            self._twist = None
            self._deskew = bool(p('deskew_odom_topic'))
            if self._deskew:
                self.create_subscription(Odometry, p('deskew_odom_topic'),
                                         self._odom_cb, 10)
                # serial path: beam age per ROS-grid bin — the unit sweeps its
                # OWN angle 0->2pi clockwise over scan_time; ROS bin theta
                # corresponds to rplidar angle -theta, hence this time fraction
                theta = -math.pi + np.arange(self.num_bins) \
                    * (2.0 * math.pi / self.num_bins)
                frac = ((-theta - self.angle_offset) % (2.0 * math.pi)) \
                    / (2.0 * math.pi)
                self._ages_frac = frac - 1.0          # x scan_time at publish
                self.get_logger().info(
                    f"de-skew on (twist from {p('deskew_odom_topic')})")
            self._stop = False
            self._last_pub = time.monotonic()
            self._raw_count = 0
            self.thread = None
            if self.source == 'topic':
                self.create_subscription(LaserScan, p('raw_scan_topic'),
                                         self._raw_cb, qos_profile_sensor_data)
                self.get_logger().info(
                    f'rplidar_node ready — regrid {p("raw_scan_topic")} '
                    f'(SDK driver) -> {self.num_bins} bins -> {p("scan_topic")}')
            else:
                self.thread = threading.Thread(target=self._read_loop, daemon=True)
                self.thread.start()
                self.get_logger().info(
                    f'rplidar_node ready — {self.port}@{self.baud} '
                    f'{self.num_bins} bins -> {p("scan_topic")}')

        # ── common publish ────────────────────────────────────────────────────
        def _emit(self, ranges, ages, stamp=None, scan_time=None):
            now = time.monotonic()
            scan = LaserScan()
            scan.header.stamp = stamp if stamp is not None \
                else self.get_clock().now().to_msg()
            scan.header.frame_id = self.frame
            scan.angle_min = -math.pi
            scan.angle_max = math.pi - 2.0 * math.pi / self.num_bins
            scan.angle_increment = 2.0 * math.pi / self.num_bins
            # sweep duration: trust the driver's value when it has one,
            # otherwise the inter-publish interval
            if scan_time is not None and math.isfinite(scan_time) and scan_time > 1e-3:
                scan.scan_time = float(scan_time)
            else:
                scan.scan_time = max(now - self._last_pub, 1e-3)
            scan.time_increment = scan.scan_time / self.num_bins
            scan.range_min = self.range_min
            scan.range_max = self.range_max
            if self._twist is not None and ages is not None:
                vx, vy, w = self._twist
                ranges = deskew_ranges(
                    ranges, scan.angle_min, scan.angle_increment,
                    scan.scan_time, vx, vy, w, ages=ages)
            scan.ranges = np.asarray(ranges, np.float32).tolist()
            self.pub.publish(scan)
            self._last_pub = now

        # ── topic source (SDK driver -> regrid) ───────────────────────────────
        def _raw_cb(self, m):
            ranges, ages = regrid_scan(
                m.ranges, m.angle_min, m.angle_increment, self.num_bins,
                self.range_min, self.range_max, self.angle_offset,
                time_increment=m.time_increment)
            # the SDK stamps the START of the sweep; our contract is sweep END
            stamp = m.header.stamp
            if m.time_increment > 0.0 and len(m.ranges):
                from rclpy.time import Time
                from rclpy.duration import Duration
                end = Time.from_msg(stamp) + Duration(
                    seconds=m.time_increment * (len(m.ranges) - 1))
                stamp = end.to_msg()
            self._emit(ranges, ages if self._deskew else None, stamp,
                       scan_time=m.scan_time)
            self._raw_count += 1
            if self._raw_count == 1:
                self.get_logger().info(
                    f'first raw scan: {len(m.ranges)} beams, '
                    f'{math.degrees(m.angle_max - m.angle_min):.0f} deg, '
                    f'scan_time {m.scan_time:.3f}s')

        # ── serial source (pip rplidar, A-series) ─────────────────────────────
        def _publish(self, measurements):
            scan_time = max(time.monotonic() - self._last_pub, 1e-3)
            ranges = bin_scan(measurements, self.num_bins, self.range_min,
                              self.range_max, self.angle_offset)
            ages = self._ages_frac * scan_time if self._deskew else None
            self._emit(ranges, ages)

        def _odom_cb(self, m):
            self._twist = (m.twist.twist.linear.x, m.twist.twist.linear.y,
                           m.twist.twist.angular.z)

        def _read_loop(self):
            try:
                from rplidar import RPLidar
            except ImportError:
                self.get_logger().error(
                    'rplidar package missing — pip3 install rplidar-roboticia')
                return
            while not self._stop:
                lidar = None
                try:
                    lidar = RPLidar(self.port, baudrate=self.baud)
                    self.get_logger().info(f'connected: {lidar.get_info()}')
                    for measurements in lidar.iter_scans(max_buf_meas=5000):
                        if self._stop:
                            break
                        self._publish(measurements)
                except Exception as e:
                    if not self._stop:
                        self.get_logger().warning(
                            f'lidar error ({e}) — reconnecting.  If this is an '
                            f'RPLIDAR C1/S-series, set source: topic and run '
                            f'the rplidar_ros SDK driver instead.')
                        time.sleep(2.0)
                finally:
                    if lidar is not None:
                        try:
                            lidar.stop()
                            lidar.stop_motor()
                            lidar.disconnect()
                        except Exception:
                            pass

        def shutdown(self):
            self._stop = True
            if self.thread is not None:
                self.thread.join(timeout=3.0)

    return rclpy, RPLidarNode


def main(args=None):
    rclpy, NodeCls = _make_node()
    rclpy.init(args=args)
    node = None
    try:
        node = NodeCls()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.shutdown()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

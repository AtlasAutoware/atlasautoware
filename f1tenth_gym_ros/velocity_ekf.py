"""
Velocity EKF — IMU + wheel-odometry fusion with slip-aware gating.
==================================================================

The state-estimation layer the winning stacks consider decisive: localization
quality is dominated by the *velocity* estimate that feeds it, and raw wheel
odometry lies exactly when racing matters (wheel slip under hard accel/brake
and at the grip limit).  This filter fuses the IMU (200 Hz, prediction) with
VESC wheel speed (update), and *gates* the wheel measurement by its
Mahalanobis innovation distance: when the wheel disagrees with the inertial
estimate beyond the gate, the measurement noise is inflated and a slip flag
raised, so the filter coasts on the IMU through the slip and locks back on
when grip returns (AMZ/ForzaETH recipe).

  state    x = [vx, vy, omega]                       (body frame)
  predict  vx += (ax + omega*vy) dt                  (IMU specific force,
           vy += (ay - omega*vx) dt                   planar mechanization)
  update   z = gyro_z          h = omega
           z = v_wheel         h = vx                (slip-gated, see below)
           z = 0               h = vy - omega*l_r    (nonholonomic pseudo-
                                                      measurement: rear axle
                                                      doesn't slide sideways)

Slip detection compares the wheel against a pure-IMU velocity integral
anchored at the last accepted wheel update — NOT against the filter's own
innovation, which a slip *ramp* defeats by dragging the estimate along.
The anchored residual sees the full wheel offset at once; hysteresis
(re-enter at 60% of the trip threshold) avoids chatter, and a rejection
timeout forcibly re-anchors so a biased IMU can't latch the gate forever.

Outputs feed the controller's speed state, the latency forward-predictor, a
particle-filter motion prior, and the lidar de-skew (scan_deskew.py).  Pure
numpy core (unit-tested, microseconds per tick) + a thin ROS node.

References: ForzaETH race stack state_estimation (JFR 2024, arXiv:2403.11784);
AMZ "End-to-End Velocity Estimation for Autonomous Racing" (arXiv:2003.06917).
"""

import numpy as np


class VelocityEKF:
    def __init__(self, l_r=0.17145,
                 q_v=0.1, q_w=2.0,
                 r_gyro=1e-4, r_wheel=4e-4, r_vy=2.5e-3,
                 slip_thresh=0.25, reject_timeout=3.0):
        self.x = np.zeros(3)                      # [vx, vy, omega]
        self.P = np.eye(3)
        self.l_r = float(l_r)
        self.Q = np.diag([q_v, q_v, q_w])
        self.r_gyro = float(r_gyro)
        self.r_wheel = float(r_wheel)
        self.r_vy = float(r_vy)
        self.slip_thresh = float(slip_thresh)     # m/s wheel-vs-IMU residual
        self.reject_timeout = float(reject_timeout)
        self.slip = False
        self.accel_bias = np.zeros(2)             # set from a standstill window
        self._v_imu = None                        # IMU-only integral (anchor)
        self._reject_t = 0.0

    # ── prediction at IMU rate ──────────────────────────────────────────────
    def predict(self, ax, ay, dt):
        ax -= self.accel_bias[0]
        ay -= self.accel_bias[1]
        vx, vy, w = self.x
        self.x = np.array([vx + (ax + w * vy) * dt,
                           vy + (ay - w * vx) * dt,
                           w])
        F = np.array([[1.0,  w * dt,  vy * dt],
                      [-w * dt, 1.0, -vx * dt],
                      [0.0, 0.0, 1.0]])
        self.P = F @ self.P @ F.T + self.Q * dt
        if self._v_imu is not None:
            self._v_imu += (ax + w * vy) * dt
            if self.slip:
                self._reject_t += dt

    # ── scalar measurement update (Joseph form) ─────────────────────────────
    def _update(self, z, h, H, r):
        H = np.asarray(H, float)
        S = float(H @ self.P @ H + r)
        innov = float(z - h)
        d2 = innov * innov / S
        K = (self.P @ H) / S
        self.x = self.x + K * innov
        IKH = np.eye(3) - np.outer(K, H)
        self.P = IKH @ self.P @ IKH.T + np.outer(K, K) * r
        return d2

    def update_gyro(self, gyro_z):
        self._update(gyro_z, self.x[2], [0.0, 0.0, 1.0], self.r_gyro)

    def update_wheel_speed(self, v_wheel):
        """Slip-gated against the IMU-anchored velocity (see module docs):
        residual beyond `slip_thresh` rejects the wheel and coasts on the
        IMU; hysteresis re-accepts at 60% of the threshold; a rejection
        timeout forcibly re-anchors so the gate can never latch forever."""
        if self._v_imu is None:                   # first contact: trust wheel
            self._v_imu = float(v_wheel)
        resid = abs(float(v_wheel) - self._v_imu)
        trip = self.slip_thresh * (0.6 if self.slip else 1.0)
        self.slip = resid > trip
        if self.slip and self._reject_t > self.reject_timeout:
            self.slip = False                     # safety valve: re-anchor
        if not self.slip:
            self._update(v_wheel, self.x[0], [1.0, 0.0, 0.0], self.r_wheel)
            self._v_imu = float(self.x[0])        # re-anchor on accepted data
            self._reject_t = 0.0

    def update_nonholonomic(self):
        """Rear axle can't slide sideways: vy - omega*l_r ~ 0 (makes vy observable)."""
        h = self.x[1] - self.x[2] * self.l_r
        self._update(0.0, h, [0.0, 1.0, -self.l_r], self.r_vy)

    @property
    def twist(self):
        """(vx, vy, omega) — body-frame velocity estimate."""
        return float(self.x[0]), float(self.x[1]), float(self.x[2])


# ─────────────────────────────────────────────────────────────────────────────
# ROS node (only imported/used on the car)
# ─────────────────────────────────────────────────────────────────────────────

def _make_node():
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile
    from sensor_msgs.msg import Imu
    from nav_msgs.msg import Odometry

    class VelocityEKFNode(Node):
        def __init__(self):
            super().__init__('velocity_ekf')
            self.declare_parameter('imu_topic', '/oakd/imu')
            self.declare_parameter('wheel_odom_topic', '/vesc/odom')
            self.declare_parameter('out_topic', '/ekf/odom')
            self.declare_parameter('base_frame', 'base_link')
            self.declare_parameter('l_r', 0.17145)
            p = lambda n: self.get_parameter(n).value   # noqa: E731

            self.ekf = VelocityEKF(l_r=float(p('l_r')))
            self.pub = self.create_publisher(Odometry, p('out_topic'),
                                             QoSProfile(depth=10))
            self.base_frame = p('base_frame')
            self._last_t = None
            self._slip_logged = False
            self.create_subscription(Imu, p('imu_topic'), self._imu_cb, 50)
            self.create_subscription(Odometry, p('wheel_odom_topic'),
                                     self._wheel_cb, 10)
            self.get_logger().info(
                f"velocity_ekf ready — imu={p('imu_topic')} "
                f"wheel={p('wheel_odom_topic')} -> {p('out_topic')}")

        def _imu_cb(self, m):
            t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
            if self._last_t is not None:
                dt = t - self._last_t
                if 0.0 < dt < 0.1:
                    self.ekf.predict(m.linear_acceleration.x,
                                     m.linear_acceleration.y, dt)
                    self.ekf.update_gyro(m.angular_velocity.z)
                    self.ekf.update_nonholonomic()
                    self._publish(m.header.stamp)
            self._last_t = t

        def _wheel_cb(self, m):
            self.ekf.update_wheel_speed(m.twist.twist.linear.x)
            if self.ekf.slip and not self._slip_logged:
                self.get_logger().warning('wheel slip detected — gating odometry')
            self._slip_logged = self.ekf.slip

        def _publish(self, stamp):
            vx, vy, w = self.ekf.twist
            out = Odometry()
            out.header.stamp = stamp
            out.header.frame_id = self.base_frame
            out.child_frame_id = self.base_frame
            out.twist.twist.linear.x = vx
            out.twist.twist.linear.y = vy
            out.twist.twist.angular.z = w
            out.twist.covariance[0] = float(self.ekf.P[0, 0])
            out.twist.covariance[7] = float(self.ekf.P[1, 1])
            out.twist.covariance[35] = float(self.ekf.P[2, 2])
            self.pub.publish(out)

    return rclpy, VelocityEKFNode


def main(args=None):
    rclpy, NodeCls = _make_node()
    rclpy.init(args=args)
    try:
        rclpy.spin(NodeCls())
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()

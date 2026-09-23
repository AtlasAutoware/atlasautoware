"""
drive_mux — priority arbiter in front of drive_node.
==================================================

    /teleop     (web_pilot / gamepad, PRIORITY)  ─┐
                                                  ├─► /drive ─► drive_node ─► VESC
    /nav_drive  (raceline_mpc autonomy)         ─┘

Why this exists: web_pilot publishes /teleop and raceline_mpc publishes the
autonomous command.  Pointing both at /drive gives two publishers on one topic
and the actuator sees whichever arrived last — the car twitches between the
human and the planner.  This node makes the precedence explicit and auditable:

  * A /teleop message wins for `teleop_hold` seconds after it arrives.  The
    human always outranks the planner, which is the rule the competition
    marshals (and common sense) expect.
  * Outside that window /nav_drive passes through untouched.
  * If BOTH sources go quiet for `timeout` seconds the mux publishes a single
    explicit zero-speed command and then stays silent, so drive_node hits its
    own cmd_timeout and neutrals the ESC.  Silence is never interpreted as
    "hold the last throttle".

The mux does no shaping, limiting or smoothing — raceline_mpc keeps its AEB and
traction governor upstream, drive_node keeps its arming and timeout downstream.
This only decides WHO is talking.
"""

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped


class DriveMux(Node):
    def __init__(self):
        super().__init__("drive_mux")
        self.declare_parameter("teleop_topic", "/teleop")
        self.declare_parameter("nav_topic", "/nav_drive")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("teleop_hold", 0.5)   # s teleop outranks autonomy
        self.declare_parameter("timeout", 0.5)       # s both quiet -> zero once
        self.declare_parameter("publish_hz", 50.0)

        g = lambda k: self.get_parameter(k).value
        self.teleop_hold = float(g("teleop_hold"))
        self.timeout = float(g("timeout"))

        self.pub = self.create_publisher(AckermannDriveStamped, g("drive_topic"), 10)
        self.create_subscription(AckermannDriveStamped, g("teleop_topic"), self._teleop, 10)
        self.create_subscription(AckermannDriveStamped, g("nav_topic"), self._nav, 10)

        self.teleop_msg = None
        self.nav_msg = None
        self.t_teleop = -1e9
        self.t_nav = -1e9
        self.source = "none"
        self.zeroed = True
        self.create_timer(1.0 / float(g("publish_hz")), self._tick)
        self.get_logger().info(
            f"drive_mux: {g('teleop_topic')} (priority) + "
            f"{g('nav_topic')} -> {g('drive_topic')}; teleop holds "
            f"{self.teleop_hold:.2f}s")

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _teleop(self, m):
        self.teleop_msg, self.t_teleop = m, self._now()

    def _nav(self, m):
        self.nav_msg, self.t_nav = m, self._now()

    def _tick(self):
        now = self._now()
        msg, src = None, "none"
        if self.teleop_msg is not None and now - self.t_teleop < self.teleop_hold:
            msg, src = self.teleop_msg, "teleop"
        elif self.nav_msg is not None and now - self.t_nav < self.timeout:
            msg, src = self.nav_msg, "nav"

        if msg is None:
            if not self.zeroed:                      # both sources quiet: zero once
                stop = AckermannDriveStamped()
                stop.header.stamp = self.get_clock().now().to_msg()
                self.pub.publish(stop)
                self.zeroed = True
                self.get_logger().warn("drive_mux: no source — published zero, releasing")
            self.source = "none"
            return

        self.zeroed = False
        if src != self.source:
            self.get_logger().info(f"drive_mux: source -> {src}")
            self.source = src
        msg.header.stamp = self.get_clock().now().to_msg()
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = DriveMux()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

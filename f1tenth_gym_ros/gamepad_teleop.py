"""
Gamepad teleop — translate sensor_msgs/Joy to AckermannDriveStamped on /drive.
===============================================================================

Works with any XInput-compatible gamepad (Xbox, PS4/PS5 in XInput mode,
generic USB) via the standard ROS 2 joy node.

Control layout (default axis/button indices match Xbox / DS4):
  Left stick Y   (axis 1)  — throttle forward/reverse
  Left stick X   (axis 0)  — steering (left = positive)
  Right trigger  (axis 5)  — brake (overrides throttle when held)
  LB  (button 4)           — deadman switch: MUST be held for any motion
  RB  (button 5)           — turbo: raises speed limit from safe_speed to max_speed
  A/Cross (button 0)       — emergency stop (sends zero and latches until released)

Safety:
  - Releasing the deadman immediately sends a zero-speed command.
  - The drive_node's own cmd_timeout acts as a second watchdog.
  - E-stop latches until the button is released, then requires deadman re-press.

Parameters (all under gamepad_teleop node namespace in hardware.yaml):
  joy_topic        : /joy
  drive_topic      : /drive
  deadman_button   : 4      # LB
  turbo_button     : 5      # RB
  estop_button     : 0      # A / Cross
  throttle_axis    : 1      # left stick Y
  steer_axis       : 0      # left stick X
  brake_axis       : 5      # right trigger
  brake_invert     : false  # true for DualSense/PS — triggers rest at +1, not -1
  safe_speed       : 2.0    # m/s without turbo
  max_speed        : 5.0    # m/s with turbo
  max_steer        : 0.41   # rad (match drive_node max_steer)
  steer_gain       : 1.0    # scale stick deflection → steer angle
  publish_hz       : 20.0   # rate at which zero commands are repeated when idle

Run:
    ros2 run joy joy_node
    ros2 run f1tenth_gym_ros gamepad_teleop
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Joy
from ackermann_msgs.msg import AckermannDriveStamped


class GamepadTeleop(Node):

    def __init__(self):
        super().__init__('gamepad_teleop')

        def p(name, val):
            self.declare_parameter(name, val)
            return self.get_parameter(name).value

        self._joy_topic    = p('joy_topic',       '/joy')
        self._drive_topic  = p('drive_topic',     '/drive')
        self._deadman_btn  = p('deadman_button',   4)
        self._turbo_btn    = p('turbo_button',     5)
        self._estop_btn    = p('estop_button',     0)
        self._throttle_ax  = p('throttle_axis',    1)
        self._steer_ax     = p('steer_axis',       0)
        self._brake_ax     = p('brake_axis',       5)
        self._safe_speed   = p('safe_speed',       2.0)
        self._max_speed    = p('max_speed',        5.0)
        self._max_steer    = p('max_steer',        0.41)
        self._steer_gain   = p('steer_gain',       1.0)
        self._brake_invert = p('brake_invert',     False)
        hz                 = p('publish_hz',       20.0)

        self._estop    = False
        self._estop_held = False
        self._last_joy: Joy | None = None

        self._pub = self.create_publisher(AckermannDriveStamped, self._drive_topic, 10)
        self._sub = self.create_subscription(Joy, self._joy_topic, self._joy_cb, 10)
        self._timer = self.create_timer(1.0 / hz, self._tick)

        self.get_logger().info(
            f'Gamepad teleop ready — deadman=btn{self._deadman_btn}, '
            f'turbo=btn{self._turbo_btn}, estop=btn{self._estop_btn}, '
            f'safe={self._safe_speed} m/s, max={self._max_speed} m/s'
        )

    def _joy_cb(self, msg: Joy):
        # E-stop toggle: press latches, release clears latch
        estop_pressed = self._btn(msg, self._estop_btn)
        if estop_pressed and not self._estop_held:
            self._estop = True
            self._estop_held = True
            self.get_logger().warn('E-STOP engaged — release button to clear')
        if not estop_pressed:
            if self._estop_held:
                self._estop = False
                self._estop_held = False
                self.get_logger().info('E-stop cleared')

        self._last_joy = msg

    def _tick(self):
        msg = self._last_joy
        cmd = AckermannDriveStamped()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.header.frame_id = 'base_link'

        if msg is None or self._estop or not self._btn(msg, self._deadman_btn):
            # Deadman released, e-stop, or no signal — send zero
            self._pub.publish(cmd)
            return

        # Throttle: axis rests at 0, forward push = positive
        raw_throttle = self._axis(msg, self._throttle_ax)

        # Brake trigger: Xbox rests at -1 (full pull = +1); DualSense rests at +1
        brake_raw = self._axis(msg, self._brake_ax)
        if self._brake_invert:
            brake = max(0.0, (1.0 - brake_raw) / 2.0)
        else:
            brake = max(0.0, (brake_raw + 1.0) / 2.0)

        speed_limit = self._max_speed if self._btn(msg, self._turbo_btn) else self._safe_speed
        speed = raw_throttle * speed_limit * (1.0 - brake)

        # Steering: axis left = +1 → positive steer angle = left turn
        steer_raw = self._axis(msg, self._steer_ax)
        steer = steer_raw * self._max_steer * self._steer_gain
        steer = max(-self._max_steer, min(self._max_steer, steer))

        cmd.drive.speed = float(speed)
        cmd.drive.steering_angle = float(steer)
        self._pub.publish(cmd)

    @staticmethod
    def _btn(msg: Joy, idx: int) -> bool:
        return bool(idx < len(msg.buttons) and msg.buttons[idx])

    @staticmethod
    def _axis(msg: Joy, idx: int) -> float:
        return float(msg.axes[idx]) if idx < len(msg.axes) else 0.0


def main(args=None):
    rclpy.init(args=args)
    node = GamepadTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

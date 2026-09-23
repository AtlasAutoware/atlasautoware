# Real-car bringup — sensors + actuation + the competition racing node.
#
#   rplidar_sdk (Slamtec SDK) ──► /scan_raw ──► rplidar_node (regrid) ──► /scan ─┐
#   camera (Orbbec Gemini 335 or OAK-D)                                          │
#       ──► /oakd/rgb, /oakd/camera_info, /oakd/imu ──► raceline_mpc ──► /drive ──► drive_node
#   (localization, e.g. particle filter, provides /pf/pose/odom separately)
#
# LiDAR (lidar_driver:=sdk, default): the car's RPLIDAR C1 speaks a protocol the
# pip `rplidar` library does not ("Descriptor length mismatch"), so the official
# rplidar_ros SDK driver owns the serial port and publishes the raw scan on
# /scan_raw; our rplidar_node re-bins it onto the fixed 720-slot grid every racing
# node indexes into (+ mounting offset, + EKF de-skew) and republishes /scan.
# lidar_driver:=pip keeps the old direct-serial path for A1/A2/A3 units.
#
# Camera backends (camera_backend:=orbbec|oakd, default orbbec — that is what is
# on the car). The Orbbec path runs the official orbbec_camera driver
# (gemini_330_series.launch.py) and remaps its topics onto the /oakd/* names the
# rest of the stack was written against, so velocity_ekf / raceline_mpc /
# camera_perception need no changes:
#   /camera/color/image_raw    -> /oakd/rgb          (bgr8)
#   /camera/color/camera_info  -> /oakd/camera_info
#   /camera/gyro_accel/sample  -> /oakd/imu          (accel+gyro in one Imu msg, rotated
#                                                   optical->body by imu_optical_to_body)
# Depth/pointcloud are off by default (use_depth:=true to enable) — nothing in
# the racing stack consumes them yet and they cost USB bandwidth + CPU.
#
# drive_node auto-detects its actuation path (PCA9685 over I2C, or VESC over
# UART) at startup.  Toggle individual pieces with the launch args, e.g. to
# bring up sensors only while testing on a bench:
#   ros2 launch f1tenth_gym_ros car_bringup_launch.py use_racing:=false
# Multi-car (head-to-head) avoidance in raceline_mpc is configured in
# config/hardware.yaml (raceline_mpc.avoid_opponents) and can be overridden:
#   ros2 launch f1tenth_gym_ros car_bringup_launch.py avoid_opponents:=false
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node, SetRemap
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def _both(flag, backend):
    """IfCondition: use_camera is true AND camera_backend == backend."""
    return IfCondition(PythonExpression([
        "'", LaunchConfiguration(flag), "' == 'true' and '",
        LaunchConfiguration('camera_backend'), "' == '", backend, "'"]))


def _lidar(driver):
    """IfCondition: use_lidar is true AND lidar_driver == driver."""
    return IfCondition(PythonExpression([
        "'", LaunchConfiguration('use_lidar'), "' == 'true' and '",
        LaunchConfiguration('lidar_driver'), "' == '", driver, "'"]))


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('f1tenth_gym_ros'),
        'config',
        'hardware.yaml',
    )
    ld = LaunchDescription()
    for arg, default in (('use_lidar', 'true'), ('use_camera', 'true'),
                         ('use_drive', 'true'), ('use_racing', 'true'),
                         ('use_teleop', 'false'),
                         ('lidar_driver', 'sdk'),        # sdk (C1/S-series) | pip (A1/A2/A3)
                         ('lidar_port', '/dev/sensors/rplidar'),
                         ('lidar_baud', '460800'),        # C1 = 460800
                         ('camera_backend', 'orbbec'),   # orbbec | oakd
                         ('use_depth', 'false'),         # orbbec depth + pointcloud
                         ('use_perception', 'false'),    # YOLO car detector -> /camera_opponents_poses
                         ('avoid_opponents', 'true')):   # multi-car avoidance in raceline_mpc
        ld.add_action(DeclareLaunchArgument(arg, default_value=default))

    # ── lidar: Slamtec SDK driver -> /scan_raw, then our fixed-grid regrid -> /scan ──
    ld.add_action(Node(
        package='rplidar_ros',
        executable='rplidar_node',
        name='rplidar_sdk',
        parameters=[{'channel_type': 'serial',
                     'serial_port': LaunchConfiguration('lidar_port'),
                     'serial_baudrate': ParameterValue(LaunchConfiguration('lidar_baud'),
                                                       value_type=int),
                     'frame_id': 'laser',
                     'inverted': False,
                     'angle_compensate': True,
                     'scan_mode': 'Standard'}],
        remappings=[('scan', '/scan_raw')],
        condition=_lidar('sdk'),
    ))
    ld.add_action(Node(
        package='f1tenth_gym_ros',
        executable='rplidar_node',
        name='rplidar_node',
        parameters=[config, {'source': 'topic', 'raw_scan_topic': '/scan_raw'}],
        condition=_lidar('sdk'),
    ))
    # legacy direct-serial path (pip rplidar; A-series units only)
    ld.add_action(Node(
        package='f1tenth_gym_ros',
        executable='rplidar_node',
        name='rplidar_node',
        parameters=[config, {'source': 'serial',
                             'port': LaunchConfiguration('lidar_port')}],
        condition=_lidar('pip'),
    ))

    # ── camera: Orbbec Gemini 335 via the official driver, remapped to /oakd/* ──
    ld.add_action(GroupAction(
        actions=[
            SetRemap(src='/camera/color/image_raw', dst='/oakd/rgb'),
            SetRemap(src='/camera/color/camera_info', dst='/oakd/camera_info'),
            # IMU: the driver reports accel/gyro in the camera OPTICAL frame (x right,
            # y down, z forward); velocity_ekf / raceline_mpc expect ROS body axes
            # (x forward, z up), so rotate instead of a plain remap.
            Node(
                package='f1tenth_gym_ros',
                executable='imu_optical_to_body',
                name='imu_optical_to_body',
                parameters=[{'in_topic': '/camera/gyro_accel/sample',
                             'out_topic': '/oakd/imu',
                             'frame_id': 'camera_link'}],
            ),
            IncludeLaunchDescription(
                # Resolve this optional package only when its camera group runs.
                # Eager lookup prevents even use_camera:=false / oakd bringup
                # on Jetsons that do not have the Orbbec driver installed.
                PythonLaunchDescriptionSource(PathJoinSubstitution([
                    FindPackageShare('orbbec_camera'),
                    'launch', 'gemini_330_series.launch.py'])),
                launch_arguments={
                    'camera_name': 'camera',
                    'enable_color': 'true',
                    'color_width': '640', 'color_height': '480', 'color_fps': '30',
                    'enable_depth': LaunchConfiguration('use_depth'),
                    'enable_point_cloud': LaunchConfiguration('use_depth'),
                    'enable_ir': 'false',
                    'enable_accel': 'true', 'enable_gyro': 'true',
                    'enable_sync_output_accel_gyro': 'true',
                    'accel_rate': '200hz', 'gyro_rate': '200hz',
                    'log_level': 'warn',
                }.items(),
            ),
        ],
        condition=_both('use_camera', 'orbbec'),
    ))
    # ── camera: OAK-D Pro (DepthAI) — the original backend, kept selectable ──
    ld.add_action(Node(
        package='f1tenth_gym_ros',
        executable='oakd_camera',
        name='oakd_camera',
        parameters=[config],
        condition=_both('use_camera', 'oakd'),
    ))

    # YOLOv8 on /oakd/rgb -> /camera_opponents_poses. race_agent consumes it;
    # raceline_mpc does not. use_perception:=true loads the device-built
    # TensorRT engine from hardware.yaml.
    ld.add_action(Node(
        package='f1tenth_gym_ros',
        executable='camera_perception',
        name='camera_perception',
        parameters=[config],
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration('use_camera'), "' == 'true' and '",
            LaunchConfiguration('use_perception'), "' == 'true'"])),
    ))
    # Depth -> /scan_fused: obstacles above/below the lidar plane (cones, ramps, low
    # boxes) reach the brake and planner. Point raceline_mpc at scan_topic:=/scan_fused.
    ld.add_action(Node(
        package='f1tenth_gym_ros',
        executable='depth_fusion',
        name='depth_fusion',
        parameters=[config],
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration('use_camera'), "' == 'true' and '",
            LaunchConfiguration('use_depth'), "' == 'true'"])),
    ))
    ld.add_action(Node(
        package='f1tenth_gym_ros',
        executable='drive_node',
        name='drive_node',
        parameters=[config],
        condition=IfCondition(LaunchConfiguration('use_drive')),
    ))
    ld.add_action(Node(
        package='f1tenth_gym_ros',
        executable='velocity_ekf',
        name='velocity_ekf',
        parameters=[config],
        condition=IfCondition(LaunchConfiguration('use_camera')),
    ))
    ld.add_action(Node(
        package='f1tenth_gym_ros',
        executable='raceline_mpc',
        name='raceline_mpc',
        parameters=[config, {'avoid_opponents': ParameterValue(
            LaunchConfiguration('avoid_opponents'), value_type=bool)},
                    {'drive_topic': '/nav_drive'}],
        condition=IfCondition(LaunchConfiguration('use_racing')),
    ))

    # ── web_teleop: web_pilot's virtual gamepad (/joy) -> AckermannDrive on /teleop ──
    # web_pilot publishes a sensor_msgs/Joy in the F310 layout (throttle axis 1,
    # steer axis 3, triggers rest at +1, deadman = button 4) and NOT a drive
    # command, so without this translator the browser controls do nothing.
    # It is a second instance of gamepad_teleop with the web layout; the
    # gamepad_teleop block in hardware.yaml stays tuned for the physical DualSense.
    ld.add_action(Node(
        package='f1tenth_gym_ros',
        executable='gamepad_teleop',
        name='web_teleop',
        parameters=[{'joy_topic': '/joy',
                     'drive_topic': '/teleop',
                     'deadman_button': 4, 'turbo_button': 5, 'estop_button': 0,
                     'throttle_axis': 1, 'steer_axis': 3,
                     'brake_axis': 5, 'brake_invert': True,
                     'safe_speed': 2.0, 'max_speed': 4.0,
                     'max_steer': 0.41, 'steer_gain': 1.0,
                     'publish_hz': 20.0}],
        condition=IfCondition(LaunchConfiguration('use_drive')),
    ))

    # ── drive_mux: /teleop (human, priority) + /nav_drive (autonomy) -> /drive ──
    # Without this, web_pilot and raceline_mpc are two publishers on /drive and
    # the actuator obeys whichever packet landed last. The mux makes the human
    # outrank the planner explicitly, and zeroes when BOTH go quiet.
    ld.add_action(Node(
        package='f1tenth_gym_ros',
        executable='drive_mux',
        name='drive_mux',
        parameters=[{'teleop_topic': '/teleop',
                     'nav_topic': '/nav_drive',
                     'drive_topic': '/drive',
                     'teleop_hold': 0.5}],
        condition=IfCondition(LaunchConfiguration('use_drive')),
    ))

    ld.add_action(Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        parameters=[{'device_id': 0, 'autorepeat_rate': 20.0}],
        condition=IfCondition(LaunchConfiguration('use_teleop')),
    ))
    ld.add_action(Node(
        package='f1tenth_gym_ros',
        executable='gamepad_teleop',
        name='gamepad_teleop',
        parameters=[config],
        condition=IfCondition(LaunchConfiguration('use_teleop')),
    ))

    # static mounting transforms — measure on the actual car and adjust
    # args: x y z yaw pitch roll parent child
    ld.add_action(Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='base_to_laser',
        arguments=['0.27', '0', '0.11', '0', '0', '0', 'base_link', 'laser'],
    ))
    # Orbbec driver publishes camera_link -> camera_*_optical_frame itself; we only
    # place camera_link on the car. (OAK-D backend uses the oakd_rgb frame.)
    ld.add_action(Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='base_to_camera',
        arguments=['0.30', '0', '0.14', '0', '0', '0', 'base_link', 'camera_link'],
    ))
    ld.add_action(Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='base_to_oakd',
        arguments=['0.30', '0', '0.14', '0', '0', '0', 'base_link', 'oakd_rgb'],
        condition=_both('use_camera', 'oakd'),
    ))
    return ld

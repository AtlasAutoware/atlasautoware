# Real-car bringup — sensors + actuation + the competition racing node.
#
#   rplidar_node ──► /scan ────────────────────────┐
#   camera (Orbbec Gemini 335 or OAK-D)            │
#       ──► /oakd/rgb, /oakd/camera_info, /oakd/imu ──► raceline_mpc ──► /drive ──► drive_node
#   (localization, e.g. particle filter, provides /pf/pose/odom separately)
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
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node, SetRemap
from ament_index_python.packages import get_package_share_directory
import os


def _both(flag, backend):
    """IfCondition: use_camera is true AND camera_backend == backend."""
    return IfCondition(PythonExpression([
        "'", LaunchConfiguration(flag), "' == 'true' and '",
        LaunchConfiguration('camera_backend'), "' == '", backend, "'"]))


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
                         ('camera_backend', 'orbbec'),   # orbbec | oakd
                         ('use_depth', 'false'),         # orbbec depth + pointcloud
                         ('use_perception', 'false')):   # YOLO car detector -> /camera_opponents_poses
        ld.add_action(DeclareLaunchArgument(arg, default_value=default))

    ld.add_action(Node(
        package='f1tenth_gym_ros',
        executable='rplidar_node',
        name='rplidar_node',
        parameters=[config],
        condition=IfCondition(LaunchConfiguration('use_lidar')),
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
                PythonLaunchDescriptionSource(os.path.join(
                    get_package_share_directory('orbbec_camera'),
                    'launch', 'gemini_330_series.launch.py')),
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

    # ── camera_perception: YOLOv8 car detector on /oakd/rgb -> /camera_opponents_poses ──
    # (consumed by race_agent for head-to-head; raceline_mpc ignores it). Off by default:
    # onnxruntime on the CPU costs ~60 ms/frame at 416 px. use_perception:=true to enable.
    ld.add_action(Node(
        package='f1tenth_gym_ros',
        executable='camera_perception',
        name='camera_perception',
        parameters=[config],
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration('use_camera'), "' == 'true' and '",
            LaunchConfiguration('use_perception'), "' == 'true'"])),
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
        parameters=[config],
        condition=IfCondition(LaunchConfiguration('use_racing')),
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

"""SLAM for the car: slam_toolbox (async) building a map live, + pose_relay -> /pf/pose/odom.

    ros2 launch f1tenth_gym_ros slam_online.launch.py           # map an unknown space
    ros2 launch f1tenth_gym_ros slam_online.launch.py scan_topic:=/scan_fused

Drive the car around (remote mode) to build the map; save it with
    ros2 run nav2_map_server map_saver_cli -f ~/atlas_ws/src/atlasautoware/maps/<name>
then localize against it with particle_filter (run_localize.sh) next time.
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    scan = LaunchConfiguration('scan_topic')
    cfg = os.path.join(os.path.expanduser('~'), 'atlas_ws', 'src', 'atlasautoware', 'config', 'slam_toolbox.yaml')
    if not os.path.isfile(cfg):
        cfg = os.path.join(os.path.dirname(__file__), '..', 'config', 'slam_toolbox.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('scan_topic', default_value='/scan'),
        Node(
            package='slam_toolbox', executable='async_slam_toolbox_node', name='slam_toolbox',
            output='screen', parameters=[cfg, {'scan_topic': scan}],
        ),
        Node(
            package='f1tenth_gym_ros', executable='pose_relay', name='pose_relay',
            parameters=[{'map_frame': 'map', 'base_frame': 'base_link', 'pose_topic': '/pf/pose/odom'}],
        ),
    ])

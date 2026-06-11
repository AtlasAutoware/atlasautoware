"""
SLAM mapping launch (real car) — build a map of a new track during practice.

    # sim
    ros2 launch f1tenth_gym_ros slam_mapping_launch.py drive:=true learn:=true
    # real car (base frame differs from the sim's namespaced frame)
    ros2 launch f1tenth_gym_ros slam_mapping_launch.py drive:=true learn:=true \
         base_frame:=base_link

Requires slam_toolbox:  sudo apt install ros-$ROS_DISTRO-slam-toolbox
While this runs, slam_toolbox fuses /scan + VESC odom into /map. Drive a few
clean laps (auto with drive:=true, or hand-teleop).  With learn:=true the
track_learner re-optimizes the racing line live (racelines/learned_raceline.csv
+ overlay) so you watch the optimal line build up.  When happy, finish with:
    tools/finish_mapping.sh <track_name>
which saves the map and (re)generates the final raceline.

`base_frame` is the only sim/hardware difference here: the gym names it
`ego_racecar/base_link`; the real f1tenth_system uses `base_link`.  It flows to
both slam_toolbox and track_learner so they always agree.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('f1tenth_gym_ros')
    params = os.path.join(pkg, 'config', 'slam_mapping.yaml')
    drive = LaunchConfiguration('drive')
    learn = LaunchConfiguration('learn')
    base_frame = LaunchConfiguration('base_frame')

    return LaunchDescription([
        DeclareLaunchArgument('drive', default_value='false',
                              description='also run the autonomous mapping driver'),
        DeclareLaunchArgument('learn', default_value='false',
                              description='also run track_learner (live raceline)'),
        DeclareLaunchArgument('base_frame', default_value='ego_racecar/base_link',
                              description='robot base frame (hardware: base_link)'),
        Node(
            package='slam_toolbox',
            executable='async_slam_toolbox_node',
            name='slam_toolbox',
            output='screen',
            # file params first, then override base_frame from the launch arg so
            # one knob keeps slam_toolbox and track_learner on the same frame.
            parameters=[params, {'base_frame': base_frame}],
        ),
        Node(
            package='f1tenth_gym_ros',
            executable='mapping_driver',
            name='mapping_driver',
            output='screen',
            condition=IfCondition(drive),
        ),
        Node(
            package='f1tenth_gym_ros',
            executable='track_learner',
            name='track_learner',
            output='screen',
            parameters=[{'base_frame': base_frame}],
            condition=IfCondition(learn),
        ),
    ])

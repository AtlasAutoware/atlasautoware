# Headless launch — starts the sim bridge WITHOUT rviz2.
# rviz2 crashes with a GLSL shader error on the noVNC virtual display and
# ros2 launch then SIGTERMs the whole group including the bridge.
# This launch omits rviz2 so the bridge runs stably for benchmarking / auto-tuning.
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command
from ament_index_python.packages import get_package_share_directory
import os
import yaml


def generate_launch_description():
    ld = LaunchDescription()
    config = os.path.join(
        get_package_share_directory('f1tenth_gym_ros'),
        'config',
        'sim.yaml',
    )
    config_dict = yaml.safe_load(open(config, 'r'))
    has_opp = config_dict['bridge']['ros__parameters']['num_agent'] > 1

    bridge_node = Node(
        package='f1tenth_gym_ros',
        executable='gym_bridge',
        name='bridge',
        parameters=[config],
    )
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        parameters=[
            {'yaml_filename': config_dict['bridge']['ros__parameters']['map_path'] + '.yaml'},
            {'topic': 'map'},
            {'frame_id': 'map'},
            {'output': 'screen'},
            {'use_sim_time': True},
        ],
    )
    nav_lifecycle_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'autostart': True},
            {'node_names': ['map_server']},
        ],
    )
    ego_robot_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='ego_robot_state_publisher',
        parameters=[{'robot_description': Command(
            ['xacro ', os.path.join(
                get_package_share_directory('f1tenth_gym_ros'),
                'launch', 'ego_racecar.xacro',
            )]
        )}],
        remappings=[('/robot_description', 'ego_robot_description')],
    )
    opp_robot_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='opp_robot_state_publisher',
        parameters=[{'robot_description': Command(
            ['xacro ', os.path.join(
                get_package_share_directory('f1tenth_gym_ros'),
                'launch', 'opp_racecar.xacro',
            )]
        )}],
        remappings=[('/robot_description', 'opp_robot_description')],
    )

    # No rviz2 — avoids GLSL crash on noVNC virtual display killing the bridge
    ld.add_action(bridge_node)
    ld.add_action(nav_lifecycle_node)
    ld.add_action(map_server_node)
    ld.add_action(ego_robot_publisher)
    if has_opp:
        ld.add_action(opp_robot_publisher)

    return ld

# Copyright 2022 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    #  # --- Packages with assets/launches we reuse ---
    # pkg_bringup      = get_package_share_directory('ros_gz_example_bringup')
    # pkg_gazebo       = get_package_share_directory('ros_gz_example_gazebo')
    # pkg_description  = get_package_share_directory('ros_gz_example_description')
    # pkg_ros_gz_sim   = get_package_share_directory('ros_gz_sim')

    # --- Args ---
    rviz_arg = DeclareLaunchArgument('rviz', default_value='true', description='Open RViz')
    cmd_vel_arg = DeclareLaunchArgument(
        'cmd_vel_topic',
        default_value='/cmd_vel',
        description='ROS Twist topic Gazebo listens to'
    )
    ns_arg = DeclareLaunchArgument(
        'ns',
        default_value='',
        description='Optional ROS namespace for all nodes'
    )

    rviz = LaunchConfiguration('rviz')
    cmd_vel_topic = LaunchConfiguration('cmd_vel_topic')
    ns = LaunchConfiguration('ns')
    rviz_arg = DeclareLaunchArgument('rviz', default_value='true', description='Open RViz')

    # # --- Load robot SDF for robot_state_publisher ---
    # sdf_path = os.path.join(pkg_description, 'models', 'diff_drive', 'model.sdf')
    # with open(sdf_path, 'r') as f:
    #     robot_desc = f.read()

    # # --- Gazebo sim world ---
    # gz_sim = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
    #     launch_arguments={
    #         'gz_args': PathJoinSubstitution([pkg_gazebo, 'worlds', 'diff_drive.sdf'])
    #     }.items(),
    # )

    # # --- Robot state publisher  ---
    # robot_state_publisher = Node(
    #     package='robot_state_publisher',
    #     executable='robot_state_publisher',
    #     name='robot_state_publisher',
    #     output='both',
    #     parameters=[
    #         {'use_sim_time': True},
    #         {'robot_description': robot_desc},
    #     ],
    # )

    # # --- RViz (kept, using bringup’s config) ---
    # rviz_node = Node(
    #     package='rviz2',
    #     executable='rviz2',
    #     arguments=['-d', os.path.join(pkg_bringup, 'config', 'diff_drive.rviz')],
    #     condition=IfCondition(rviz),
    #     output='screen',
    # )

    # # --- ROS↔GZ bridge (kept, using bringup’s YAML) ---
    # bridge = Node(
    #     package='ros_gz_bridge',
    #     executable='parameter_bridge',
    #     parameters=[{
    #         'config_file': os.path.join(pkg_bringup, 'config', 'ros_gz_example_bridge.yaml'),
    #         'qos_overrides./tf_static.publisher.durability': 'transient_local',
    #     }],
    #     output='screen',
    # )

    # ---- Roomba controller node ---
    roomba = Node(
        package='lab4',
        executable='roomba',
        name='roomba',
        output='screen',
        parameters=[{'use_sim_time': True}]
        
    )
    # ---- Clearance calculator node ---
    clearance = Node(
        package='lab4',
        executable='clearance_calculator',
        name='clearance_calculator',
        output='screen',
        parameters=[{'use_sim_time': True}]
    )


    return LaunchDescription([
        rviz_arg,
        cmd_vel_arg,
        ns_arg,
        gz_sim,
        bridge,
        robot_state_publisher,
        rviz_node,
        cmd_vel_arg,
        s
    ])

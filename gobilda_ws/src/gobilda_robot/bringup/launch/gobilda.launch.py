from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import (
    Command,
    FindExecutable,
    PathJoinSubstitution,
    LaunchConfiguration,
)

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare arguments
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            'use_mock_hardware',
            default_value='false',
            description='Start robot with mock hardware mirroring command to its states.',
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'frame_id',
            default_value='rplidar_link',
            description='Frame id for the lidar link. Name found in the URDF.',
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'ekf_params_file',
            default_value=PathJoinSubstitution([
                    FindPackageShare('gobilda_robot'),
                    'config',
                    'ekf.yaml',
                ])
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'oakd_params_file',
            default_value=PathJoinSubstitution([
                    FindPackageShare('gobilda_robot'),
                    'config',
                    'oakd.yaml',
                ])
        )
    )

    declared_arguments.append(
        DeclareLaunchArgument(
            'rectify_rgb',
            default_value='True',
            description='Whether to rectify RGB images'
        )
    )


    # Initialize Arguments
    use_mock_hardware = LaunchConfiguration('use_mock_hardware')
    laser_frame_id = LaunchConfiguration('frame_id')
    
    oakd_params_file = LaunchConfiguration('oakd_params_file')
    ekf_params_file = LaunchConfiguration('ekf_params_file')

    rectify_rgb = LaunchConfiguration('rectify_rgb')

    # Grab the RPLidar launch file
    rplidar = IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('rplidar_ros'),
                    'launch',
                    'rplidar_a1_launch.py',
                ])
            ]),

           launch_arguments={
                'frame_id': laser_frame_id,
                'serial_port': '/dev/rplidar',
                'serial_baudrate': '115200',
            }.items(),
    )

    # Add the KISS-ICP Node for LiDAR based odometry
    # Before KISS-ICP node we need to convert from Laser --> PointCloud message
    laser_to_pointcloud = Node(
        package='pointcloud_to_laserscan',
        executable='laserscan_to_pointcloud_node',
        remappings=[
            ('scan_in', 'scan'),
            # Did not like it when I remapped the 'cloud' topic
        ],
    )
    
    kiss_icp = IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('kiss_icp'),
                    'launch',
                    'odometry.launch.py',
                ])
            ]),

           launch_arguments={
                'topic': '/cloud',
                'visualize': 'false',
                'base_frame': 'base_link',
                'lidar_odom_frame': 'odom',
                'invert_odom_tf': 'False',
                'orientation_covariance': '0.3',
                'publish_odom_tf': 'False',
            }.items(),
    )

    # Grab the Camera launch file
    # Path to local camera configure file
    oakd_camera= IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('depthai_ros_driver'),
                    'launch',
                    'camera_as_part_of_a_robot.launch.py',
                ])
            ]),

            launch_arguments={
                'name': 'oakd',
                'parent_frame': 'base_link',
                'publish_tf_from_calibration': 'false',
                'imu_from_descr': 'false',
                'params_file': oakd_params_file,
            }.items()
    )

    rtabmap_odom = Node(
        package='rtabmap_odom',
        executable='rgbd_odometry',
        name='rgbd_odometry',
        output='screen',
        parameters=[
            # Synchronization parameters
            {'approx_sync': True},
            {'publish_tf': False},

            # RANSAC params
            {'Vis/MinInliers': '15'},
            {'publish_tf': False},
        ],
        remappings=[
            ("rgb/image", "oakd/rgb/image_rect"),
            ("rgb/camera_info", "oakd/rgb/camera_info"),
            ("depth/image", "oakd/stereo/image_raw"),
            ('odom', 'rtabmap/odom'),
        ],
    )

    robot_localization = Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_odom',
            output='screen',
            parameters=[ekf_params_file],
    )

    # Get URDF via xacro
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name='xacro')]),
            ' ',
            PathJoinSubstitution(
                [FindPackageShare('gobilda_robot'), 'urdf', 'gobilda.urdf.xacro']
            ),
            ' ',
            'use_mock_hardware:=',
            use_mock_hardware,
        ]
    )
    robot_description = {'robot_description': robot_description_content}

    robot_controllers = PathJoinSubstitution(
        [
            FindPackageShare('gobilda_robot'),
            'config',
            'gobilda_controllers.yaml',
        ]
    )

    control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        parameters=[robot_controllers],
        output='both',
        remappings=[
            ('~/robot_description', '/robot_description'),
            ('/gobilda_base_controller/cmd_vel', '/gobilda/cmd_vel'),
        ],
        # uncomment for debugging
        # arguments=[ '--ros-args', '--log-level', 'debug', ],
    )
    robot_state_pub_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='both',
        parameters=[robot_description],
    )

    joint_state_broadcaster_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', 
                   '--controller-manager',
                   '/controller_manager'],
    )

    robot_controller_spawner = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['gobilda_base_controller', 
                   '--controller-manager',
                   '/controller_manager',],
    )

    nodes = [
        control_node,
        robot_state_pub_node,
        joint_state_broadcaster_spawner,
        robot_controller_spawner,
        laser_to_pointcloud,
        rtabmap_odom,
        robot_localization,
    ]

    launch_files = [
        rplidar,
        oakd_camera,
        kiss_icp,
    ]
    
    # Ordering here has some effects on the startup timing of the nodes
    return LaunchDescription(declared_arguments + launch_files + nodes)
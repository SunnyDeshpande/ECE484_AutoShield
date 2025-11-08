from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, EnvironmentVariable, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.conditions import IfCondition

import os
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    vehicle_name_definition = DeclareLaunchArgument(
        'vehicle_name',
        default_value=EnvironmentVariable('VEHICLE_NAME'),
        description='Name of the vehicle'
    )
    vehicle_name=LaunchConfiguration('vehicle_name')

    # robot_description_content = Command([
    #     PathJoinSubstitution([FindExecutable(name='xacro')]),
    #     ' ',
    #     PathJoinSubstitution([FindPackageShare('gem_e2_description'), 'urdf', 'gem_e2.urdf.xacro'])
    # ])

    # robot_description = {'robot_description': robot_description_content}

    

    zed_camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('basic_launch'),
                'launch',
                'perception',
                'zed_camera.launch.py'
            ])
        ),
        launch_arguments={
            'camera_model' : 'zed2'}.items(),
        condition=IfCondition(
            PythonExpression(["'", vehicle_name, "' == 'e2'"])
        )
    )

    front_camera_launch = IncludeLaunchDescription(        
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('depthai_ros_driver'), 'launch'),
            '/rgbd_pcl.launch.py']),
        condition=IfCondition(
            PythonExpression(["'", vehicle_name, "' == 'e4'"])
        )
    )

    # Top Lidar
    ouster_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('basic_launch'),
                    'launch',
                    'perception',
                    'ouster_driver.launch.py'
                ])
            ),
            launch_arguments={
                'param_file' : 'ouster_config'}.items(),
        )
    
    lucid_cam_launch = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                PathJoinSubstitution([
                    FindPackageShare('basic_launch'),
                    'launch',
                    'perception',
                    'corner_cameras.launch.py'
                ])
            )
    )
    
    rviz_display_launch = IncludeLaunchDescription(        
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('basic_launch'), 'launch'),
            '/rviz_display.launch.py'])
    )

    return LaunchDescription([
        vehicle_name_definition,
        front_camera_launch,
        zed_camera_launch,
        ouster_launch,
        lucid_cam_launch,
        rviz_display_launch
    ])

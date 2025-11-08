import os
import launch
from launch_ros.actions import Node
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command, TextSubstitution, PythonExpression, EnvironmentVariable
from ament_index_python.packages import get_package_share_directory

os.environ['RCUTILS_CONSOLE_OUTPUT_FORMAT'] = '{time}: [{name}] [{severity}]\t{message}'

def generate_launch_description():
    vehicle_name_arg = DeclareLaunchArgument(
        'vehicle_name',
        default_value=EnvironmentVariable('VEHICLE_NAME'),
        description='Name of the vehicle used to locate the parameter file '
    )
    vehicle_name = LaunchConfiguration('vehicle_name')
    car = PythonExpression(["'", vehicle_name, "' == 'e2' ? 'e2' : 'e4'"])

    tf_imu = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments = "0 0 0 0 0 0 base_link imu".split(' ')
    )

    tf_gnss = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments = "0 0 0 0 0 0 imu gnss".split(' ')
    )

    tf_vsm = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments = "0 0 0 0 0 0 imu vsm".split(' ')
    )

    tf_aux1 = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        arguments = "0 0 0 0 0 0 imu aux1".split(' ')
    )
    # tf of front radar sensor
    tf2_umrr_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='front_radar_link_to_umrr',
        arguments=['0', '0', '0', '0', '0', '0', 'front_radar_link', 'umrr']
    )

    # tf of front camera sensor
    tf2_oak_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='front_camera_link_to_oak_d_base_frame',
        arguments=['0', '0', '0', '0', '0', '0', 'front_camera_link', 'oak-d-base-frame']
    )

    # tf of top lidar sensor
    tf2_ouster_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='top_lidar_link_to_os_sensor',
        arguments=['0', '0', '0.04', '0', '0', '0', 'top_lidar_link', 'os_sensor']
    )


    base_link_to_lidar1_publisher = Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='base_link_to_lidar1',
            arguments=['-0.12', '0', '1.6', '0', '0', '0', 'base_link', 'os_sensor']
    )

    return launch.LaunchDescription([
        tf_imu,
        tf_gnss, 
        tf_vsm, 
        tf_aux1,
        tf2_umrr_node,
        tf2_oak_node,
        tf2_ouster_node
        ])

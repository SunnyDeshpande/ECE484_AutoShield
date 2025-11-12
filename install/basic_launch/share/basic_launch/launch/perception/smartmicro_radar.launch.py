import os
import launch_ros

from ament_index_python import get_package_share_directory
from launch_ros.actions import Node

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

vehicle_name=os.environ.get('VEHICLE_NAME','NO_VEHICLE_NAME')


def generate_launch_description():
       
    radar_sensor_params = os.path.join(
        get_package_share_directory('basic_launch'), 'config', vehicle_name, 'smartmicro_radar/radar.sensor.yaml')
    
    radar_adapter_params = os.path.join(
        get_package_share_directory('basic_launch'), 'config', vehicle_name, 'smartmicro_radar/radar.adapter.yaml')

    radar_node = Node(
        package='umrr_ros2_driver',
        executable='smartmicro_radar_node_exe',
        name='smart_radar',
        parameters=[radar_sensor_params, radar_adapter_params]
    )
    
    return LaunchDescription([
        radar_node
    ])

    

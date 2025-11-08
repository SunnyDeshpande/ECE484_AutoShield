from launch import LaunchDescription
from launch.actions import LogInfo
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the vehicle environment variable
    vehicle_env = os.environ.get('VEHICLE_NAME', 'e4')
    config_file = vehicle_env + '_pp.yaml'
    config_path = os.path.join(
        get_package_share_directory('gem_gnss_control'),
        'config',
        config_file
    )

    return LaunchDescription([
        # LogInfo(msg=f'Using Vehicle config: {vehicle_env}'),
        
        Node(
            package='gem_gnss_control',
            executable='pure_pursuit',  # or 'pure_pursuit_node' if you updated setup.py
            name='pure_pursuit',
            output='screen',
            parameters=[config_path]
        )
    ])

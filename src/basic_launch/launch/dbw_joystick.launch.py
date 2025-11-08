from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    pkg_dir = get_package_share_directory('pacmod2_game_control')
    launch_path = os.path.join(pkg_dir, 'launch', 'pacmod2_game_control.launch.xml')

    return LaunchDescription([
        IncludeLaunchDescription(
            AnyLaunchDescriptionSource(launch_path)
        )
    ])
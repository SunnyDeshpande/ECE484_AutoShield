import os
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import GroupAction
from launch.substitutions import Command
from launch_ros.actions import PushRosNamespace 
from launch.actions import IncludeLaunchDescription 
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    # launch gnss_ins sensor
    gnss_launch = IncludeLaunchDescription(        
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('septentrio_gnss_driver'), 'launch'),
            '/rover.launch.py'])
    )

    gnss_with_ns_launch = GroupAction(
        actions=[
            PushRosNamespace('septentrio'),  # Set the namespace
            gnss_launch                      # Add the GNSS launch
        ]
    )

    gem_gnss_image_node = Node(
        package='gem_gnss_image',
        executable='gem_gnss_image',
        output='screen',
        name='gem_gnss_image_node')

    rviz_print_node = Node(
        package='gem_rviz_display',
        executable='gem_rviz_display',
        output='screen',
        name='gem_rviz_display')


    # create and return launch description object
    return LaunchDescription(
        [            
            gem_gnss_image_node,
            gnss_with_ns_launch,
            rviz_print_node,
        ]
        
    )
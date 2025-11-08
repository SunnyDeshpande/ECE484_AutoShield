import os
import launch
from launch_ros.actions import Node
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command, TextSubstitution
from ament_index_python.packages import get_package_share_directory
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            RegisterEventHandler, EmitEvent, LogInfo)
from launch.launch_description_sources import PythonLaunchDescriptionSource

os.environ['RCUTILS_CONSOLE_OUTPUT_FORMAT'] = '{time}: [{name}] [{severity}]\t{message}'
# Verbose log:
#os.environ['RCUTILS_CONSOLE_OUTPUT_FORMAT'] = '{time}: [{name}] [{severity}]\t{message} ({function_name}() at {file_name}:{line_number})'

# Start as component:

def generate_launch_description():
    
    rviz_display_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('basic_launch'), 'launch'),
            '/rviz_display.launch.py'])
    )

    gem_gnss_image_node = Node(
        package='gem_gnss_image',
        executable='gem_gnss_image',
        output='screen',
        name='gem_gnss_image_node',
    )


    default_file_name = 'ins.yaml'
    name_arg_file_name = "file_name"
    arg_file_name = DeclareLaunchArgument(name_arg_file_name,
                                          default_value=TextSubstitution(text=str(default_file_name)))
    name_arg_file_path = 'path_to_config'
    vehicle_name = os.environ.get('VEHICLE_NAME','e4')
    default_config=os.path.join(
        get_package_share_directory('basic_launch'),
        'config',
        vehicle_name,
        'septentrio_gnss/',
        )
    arg_file_path = DeclareLaunchArgument(name_arg_file_path,
                                          default_value=[default_config, LaunchConfiguration(name_arg_file_name)])

    composable_node = ComposableNode(
        name='septentrio_gnss_driver',
        package='septentrio_gnss_driver', 
        plugin='rosaic_node::ROSaicNode',
        #emulate_tty=True,
        parameters=[LaunchConfiguration(name_arg_file_path)])

    container = ComposableNodeContainer(
        name='septentrio_gnss_driver_container',
        namespace='septentrio_gnss_driver',
        package='rclcpp_components',
        executable='component_container',
        emulate_tty=True,
        sigterm_timeout = '20',
        composable_node_descriptions=[composable_node],
        output='screen'
    )

    return launch.LaunchDescription([arg_file_name, arg_file_path, gem_gnss_image_node, rviz_display_launch, container])

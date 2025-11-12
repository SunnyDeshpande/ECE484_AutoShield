import os
from launch_ros.actions import Node
from launch import LaunchDescription
from launch.actions import GroupAction
from launch.substitutions import LaunchConfiguration, EnvironmentVariable, PathJoinSubstitution, PythonExpression
from launch_ros.actions import PushRosNamespace, ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.substitutions import FindPackageShare
from launch.actions import IncludeLaunchDescription, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext
from launch.conditions import IfCondition

import yaml

def generate_launch_description():
    # ---------------------------------------------------------------------
    launch_arguments = []
    
    vehicle_env = os.environ.get('VEHICLE_NAME', 'e4') # make sure "export VEHICLE_NAME=e2" is in your .bashrc
    
    context = LaunchContext()
    camera_param_path = os.path.join(
        get_package_share_directory("basic_launch"),
        "config",vehicle_env,"lucid/lucid_cameras.yaml"
    )
    camera_yaml_param_rl = None
    camera_yaml_param_rr = None

    with open(camera_param_path, "r") as f:
        camera_params = yaml.safe_load(f)
        camera_yaml_param_fl = camera_params["arena_camera_node_fl"]["ros__parameters"]
        camera_yaml_param_fr = camera_params["arena_camera_node_fr"]["ros__parameters"]
        if (vehicle_env=='e4'):
            camera_yaml_param_rl = camera_params["arena_camera_node_rl"]["ros__parameters"]
            camera_yaml_param_rr = camera_params["arena_camera_node_rr"]["ros__parameters"]


    fr_camera_container = ComposableNodeContainer(
        name="arena_camera_container_right",
        namespace="lucid",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[
            ComposableNode(
                package="lucid_vision_driver",
                plugin="ArenaCameraNode",
                name="arena_camera_node_right",
                parameters=[{"camera_name": camera_yaml_param_fr['camera_name'],
                             "frame_id": camera_yaml_param_fr['frame_id'],
                             "pixel_format": camera_yaml_param_fr['pixel_format'],
                             "serial_no": camera_yaml_param_fr['serial_no'],
                             "camera_info_url": camera_yaml_param_fr['camera_info_url'],
                             "fps": camera_yaml_param_fr['fps'],
                             "horizontal_binning": camera_yaml_param_fr['horizontal_binning'],
                             "vertical_binning": camera_yaml_param_fr['vertical_binning'],
                             "use_default_device_settings": camera_yaml_param_fr['use_default_device_settings'],
                             "exposure_auto": camera_yaml_param_fr['exposure_auto'],
                             "exposure_target": camera_yaml_param_fr['exposure_target'],
                             "gain_auto": camera_yaml_param_fr['gain_auto'],
                             "gain_target": camera_yaml_param_fr['gain_target'],
                             "gamma_target": camera_yaml_param_fr['gamma_target'],
                             "enable_compressing": camera_yaml_param_fr['enable_compressing'],
                             "enable_rectifying": camera_yaml_param_fr['enable_rectifying'],
                             "image_horizontal_flip": camera_yaml_param_fr['image_horizontal_flip'],
                             "image_vertical_flip": camera_yaml_param_fr['image_vertical_flip'] 
                             }],
                remappings=[
                ],
                extra_arguments=[
                    {"use_intra_process_comms": True}
                ],
            ),
        ],
        output="both",
    )
    fl_camera_container = ComposableNodeContainer(
        name="arena_camera_container_left",
        namespace="lucid",
        package="rclcpp_components",
        executable="component_container",
        composable_node_descriptions=[
            ComposableNode(
                package="lucid_vision_driver",
                plugin="ArenaCameraNode",
                name="arena_camera_node_left",
                parameters=[{"camera_name": camera_yaml_param_fl['camera_name'],
                             "frame_id": camera_yaml_param_fl['frame_id'],      
                             "pixel_format": camera_yaml_param_fl['pixel_format'],
                             "serial_no": camera_yaml_param_fl['serial_no'],
                             "camera_info_url": camera_yaml_param_fl['camera_info_url'],
                             "fps": camera_yaml_param_fl['fps'],
                             "horizontal_binning": camera_yaml_param_fl['horizontal_binning'],
                             "vertical_binning": camera_yaml_param_fl['vertical_binning'],
                             "use_default_device_settings": camera_yaml_param_fl['use_default_device_settings'],
                             "exposure_auto": camera_yaml_param_fl['exposure_auto'],
                             "exposure_target": camera_yaml_param_fl['exposure_target'],
                             "gain_auto": camera_yaml_param_fl['gain_auto'],
                             "gain_target": camera_yaml_param_fl['gain_target'],
                             "gamma_target": camera_yaml_param_fl['gamma_target'],
                             "enable_compressing": camera_yaml_param_fl['enable_compressing'],
                             "enable_rectifying": camera_yaml_param_fl['enable_rectifying'],
                             "image_horizontal_flip": camera_yaml_param_fl['image_horizontal_flip'],
                             "image_vertical_flip": camera_yaml_param_fl['image_vertical_flip'] 
                             }],
                remappings=[
                ],
                extra_arguments=[
                    {"use_intra_process_comms": True}
                ],
            ),
            
        ],
        output="both",
    )
    if (vehicle_env=='e4'):
        rl_camera_container = ComposableNodeContainer(
            name="camera_node_rl",
            namespace="/perception/object_detection",
            package="rclcpp_components",
            executable="component_container",
            composable_node_descriptions=[
                ComposableNode(
                    package="lucid_vision_driver",
                    plugin="ArenaCameraNode",
                    name="arena_camera_node_rl",
                    parameters=[{"camera_name": camera_yaml_param_rl['camera_name'],
                                "frame_id": camera_yaml_param_rl['frame_id'],
                                "pixel_format": camera_yaml_param_rl['pixel_format'],
                                "serial_no": camera_yaml_param_rl['serial_no'],
                                "camera_info_url": camera_yaml_param_rl['camera_info_url'],
                                "fps": camera_yaml_param_rl['fps'],
                                "horizontal_binning": camera_yaml_param_rl['horizontal_binning'],
                                "vertical_binning": camera_yaml_param_rl['vertical_binning'],
                                "use_default_device_settings": camera_yaml_param_rl['use_default_device_settings'],
                                "exposure_auto": camera_yaml_param_rl['exposure_auto'],
                                "exposure_target": camera_yaml_param_rl['exposure_target'],
                                "gain_auto": camera_yaml_param_rl['gain_auto'],
                                "gain_target": camera_yaml_param_rl['gain_target'],
                                "gamma_target": camera_yaml_param_rl['gamma_target'],
                                "image_horizontal_flip": camera_yaml_param_rl['image_horizontal_flip'],
                                "image_vertical_flip": camera_yaml_param_rl['image_vertical_flip'],                             
                                "enable_compressing": camera_yaml_param_rl['enable_compressing'],
                                "enable_rectifying": camera_yaml_param_rl['enable_rectifying'],
                                }],
                    remappings=[
                    ],
                    extra_arguments=[
                        {"use_intra_process_comms": True}
                    ],
                ),
            ],
            output="both",
        )
        rr_camera_container = ComposableNodeContainer(
            name="camera_node_rr",
            namespace="/perception/object_detection",
            package="rclcpp_components",
            executable="component_container",
            composable_node_descriptions=[
                ComposableNode(
                    package="lucid_vision_driver",
                    plugin="ArenaCameraNode",
                    name="arena_camera_node_rr",
                    parameters=[{"camera_name": camera_yaml_param_rr['camera_name'],
                                "frame_id": camera_yaml_param_rr['frame_id'],
                                "pixel_format": camera_yaml_param_rr['pixel_format'],
                                "serial_no": camera_yaml_param_rr['serial_no'],
                                "camera_info_url": camera_yaml_param_rr['camera_info_url'],
                                "fps": camera_yaml_param_rr['fps'],
                                "horizontal_binning": camera_yaml_param_rr['horizontal_binning'],
                                "vertical_binning": camera_yaml_param_rr['vertical_binning'],
                                "use_default_device_settings": camera_yaml_param_rr['use_default_device_settings'],
                                "exposure_auto": camera_yaml_param_rr['exposure_auto'],
                                "exposure_target": camera_yaml_param_rr['exposure_target'],
                                "gain_auto": camera_yaml_param_rr['gain_auto'],
                                "gain_target": camera_yaml_param_rr['gain_target'],
                                "gamma_target": camera_yaml_param_rr['gamma_target'],
                                "image_horizontal_flip": camera_yaml_param_rr['image_horizontal_flip'],
                                "image_vertical_flip": camera_yaml_param_rr['image_vertical_flip'],                             
                                "enable_compressing": camera_yaml_param_rr['enable_compressing'],
                                "enable_rectifying": camera_yaml_param_rr['enable_rectifying'],
                                }],
                    remappings=[
                    ],
                    extra_arguments=[
                        {"use_intra_process_comms": True}
                    ],
                    # condition=IfCondition(
                    #     PythonExpression(["'", vehicle_env, "' == 'e4'"])
                    # )
                ),
            ],
            output="both",
        )

    # create and return launch description object
    ld = LaunchDescription(launch_arguments)
    ld.add_action(fl_camera_container)
    ld.add_action(fr_camera_container)

    if vehicle_env == 'e4':
        ld.add_action(rl_camera_container)
        ld.add_action(rr_camera_container)

    return ld
        
    
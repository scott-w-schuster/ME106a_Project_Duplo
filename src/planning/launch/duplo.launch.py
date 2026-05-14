from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription, RegisterEventHandler, ExecuteProcess
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    rviz_config = os.path.join(
        get_package_share_directory('planning'),
        'planning',
        'duplo_config.rviz'
    )

    camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch',
                'rs_launch.py'
            )
        ),
        launch_arguments={
            'pointcloud.enable':        'true',
            'align_depth.enable':       'true',
            'rgb_camera.color_profile': '1920x1080x30',
            'publish_tf':               'false',
        }.items(),
    )

    camera_transform_node = Node(
        package='perception',
        executable='camera_transform',
    )

    brick_detection_node = Node(
        package='perception',
        executable='brick_detector',
        name='brick_detection_node',
        output='screen',
    )

    planning_main_node = Node(
        package='planning',
        executable='main',
        arguments=['--controller', 'pid'],
    )

    planning_node_node = Node(
        package='planning',
        executable='planning_node',
        output='screen',
    )

    tuck   = ExecuteProcess(cmd=['ros2', 'run', 'ur7e_utils', 'tuck'])
    enable = ExecuteProcess(cmd=['ros2', 'run', 'ur7e_utils', 'enable_comms'])

    moveit = ExecuteProcess(cmd=[
        'ros2', 'launch', 'ur_moveit_config', 'ur_moveit.launch.py',
        'ur_type:=ur7e', 'launch_rviz:=false'
    ])

    rviz_on = ExecuteProcess(
        cmd=['rviz2', '-d', rviz_config],
        output={'stdout': 'log', 'stderr': 'log'},
    )

    return LaunchDescription([
        # Camera stack starts immediately — not blocked by robot sequence
        camera,
        camera_transform_node,
        rviz_on,

        # Robot enable → MoveIt → tuck → planning
        enable,

        RegisterEventHandler(
            event_handler=OnProcessStart(
                target_action=enable,
                on_start=[moveit],
            )
        ),

        RegisterEventHandler(
            event_handler=OnProcessStart(
                target_action=moveit,
                on_start=[tuck],
            )
        ),

        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=tuck,
                on_exit=[planning_main_node, brick_detection_node, planning_node_node],
            )
        ),
    ])

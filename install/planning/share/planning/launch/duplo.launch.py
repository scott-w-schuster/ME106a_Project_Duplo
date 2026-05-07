from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription,
    RegisterEventHandler, EmitEvent, ExecuteProcess, TimerAction
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.events import Shutdown
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


    ur_type = LaunchConfiguration("ur_type", default="ur7e")
    launch_rviz = LaunchConfiguration("launch_rviz", default="false")


    # --- Define all processes/nodes ---


    realsense_launch = ExecuteProcess(
        cmd=['ros2', 'launch', 'realsense2_camera', 'rs_launch.py',
             'pointcloud.enable:=true', 'rgb_camera.color_profile:=1920x1080x30'],
        output='screen'
    )


    enable_comms = ExecuteProcess(
        cmd=['ros2', 'run', 'ur7e_utils', 'enable_comms'],
        output='screen'
    )


    tuck = ExecuteProcess(
        cmd=['ros2', 'run', 'ur7e_utils', 'tuck'],
        output='screen'
    )


    ur_moveit_launch = ExecuteProcess(
        cmd=['ros2', 'launch', 'ur_moveit_config', 'ur_moveit.launch.py',
             'ur_type:=ur7e', 'launch_rviz:=false'],
        output='screen'
    )


    camera_transform_node = Node(
        package='perception',
        executable='camera_transform',
        output='screen'
    )


    planning_main_node = Node(
        package='planning',
        executable='main',
        output='screen',
    )


    brick_detection_node = Node(
        package='perception',
        executable='brick_detector',
        output='screen',
        name='brick_detection_node',
    )


    planning_node_node = Node(
        package='planning',
        executable='planning_node',
        output='screen',
    )


    ik_planner_node = Node(
        package='planning',
        executable='ik',
        name='ik',
        output='screen'
    )


    rviz_process = ExecuteProcess(
        cmd=['rviz2', '-d', rviz_config],
        output='screen',
    )


    return LaunchDescription([
        # Step 1: Start realsense
        realsense_launch,


        RegisterEventHandler(
            OnProcessExit(
                target_action=realsense_launch,
                on_exit=[enable_comms]
            )
        ),


        RegisterEventHandler(
            OnProcessExit(
                target_action=enable_comms,
                on_exit=[tuck]
            )
        ),


        RegisterEventHandler(
            OnProcessExit(
                target_action=tuck,
                on_exit=[ur_moveit_launch]
            )
        ),



        RegisterEventHandler(
            OnProcessExit(
                target_action=ur_moveit_launch,
                on_exit=[camera_transform_node]
            )
        ),

        RegisterEventHandler(
            OnProcessExit(
                target_action=camera_transform_node,
                on_exit=[planning_main_node]
            )
        ),

        RegisterEventHandler(
            OnProcessExit(
                target_action=planning_main_node,
                on_exit=[brick_detection_node]
            )
        ),

        RegisterEventHandler(
            OnProcessExit(
                target_action=brick_detection_node,
                on_exit=[planning_node_node]
            )
        ),

        RegisterEventHandler(
            OnProcessExit(
                target_action=planning_node_node,
                on_exit=[rviz_process]
            )
        ),
    ])

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler, EmitEvent, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import time
import os

def generate_launch_description():
    rviz_config = os.path.join(
        get_package_share_directory('planning'),
        'planning',
        'duplo_config.rviz'
    )

    ur_type = LaunchConfiguration("ur_type", default="ur7e")
    launch_rviz = LaunchConfiguration("launch_rviz", default="false")

    ik_planner_node = Node(
        package='planning',
        executable='ik',
        name='ik',
        output='screen'
    )

    camera_transform_node = Node(
        package='perception',
        executable='camera_transform',
    )

    brick_detection_node = Node(
        package='perception',
        executable='brick_detector',
        output='screen',
        name='brick_detection_node',
    )

    planning_main_node = Node(
        package='planning',
        executable='main',
        output='screen',
    )

    planning_node_node = Node(
        package='planning',
        executable='planning_node',
        output='screen',
    )

    return LaunchDescription([
        ExecuteProcess( cmd=[ 'ros2', 'launch', 'realsense2_camera', 'rs_launch.py', 'pointcloud.enable:=true', 'rgb_camera.color_profile:=1920x1080x30']),
        ExecuteProcess( cmd=['ros2','run','ur7e_utils','enable_comms']),
        ExecuteProcess(cmd=['ros2','run','ur7e_utils','tuck']),
        ExecuteProcess(cmd=['ros2','launch','ur_moveit_config','ur_moveit.launch.py','ur_type:=ur7e','launch_rviz:=false']),
        camera_transform_node,
        planning_main_node,
        brick_detection_node,
        planning_node_node,
        ExecuteProcess(
            cmd=['rviz2', '-d', rviz_config],
            output='screen',
        ),
    ])

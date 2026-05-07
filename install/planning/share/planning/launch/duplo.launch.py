from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler, EmitEvent
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
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
    launch_rviz = LaunchConfiguration("launch_rviz", default="false") # make false if you don't want rviz to launch when launching moveit

   
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch',
                'rs_launch.py'
            )
        ),
        launch_arguments={
            'pointcloud.enable': 'true',
            'rgb_camera.color_profile': '1920x1080x30',
        }.items(),
    )

    # MoveIt include
    moveit_launch_file = os.path.join(
        get_package_share_directory("ur_moveit_config"),
        "launch",
        "ur_moveit.launch.py"
    )
    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(moveit_launch_file),
        launch_arguments={
            "ur_type": ur_type,
            "launch_rviz": launch_rviz
        }.items(),
    )

    ik_planner_node = Node(
        package='planning',
        executable='ik',
        name='ik',
        output='screen'
    )

    camera_transformn_node=Node(
            package='perception',
            executable='camera_transform',
        ),
    brick_detection_node=Node(
        package='perception',
        executable='brick_detector',
        output='screen',
        name='brick_detection_node',
    ),
    planning_main_node=Node(
        package='planning',
        executable='main',
        output='screen',
    ),
    planning_node_node=Node(
        package='planning',
        executable='planning_node',
        output='screen',
    ),
    

    shutdown_on_any_exit = RegisterEventHandler(
        OnProcessExit(
            on_exit=[EmitEvent(event=Shutdown(reason='A launched process exited'))]
        )
    )

    return LaunchDescription([

        # Actions
        realsense_launch,
        moveit_launch,
        ik_planner_node,
        camera_transformn_node,
        planning_main_node,
        brick_detection_node,
        planning_node_node,

        ExecuteProcess(
        cmd=['rviz2', '-d', rviz_config],
        output='screen',
        ),

        # Global handler (keep at end)
        shutdown_on_any_exit,
    ])
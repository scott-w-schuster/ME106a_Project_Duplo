from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler, EmitEvent, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit, OnProcessStart
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from launch.actions import LogInfo
import os


def generate_launch_description():
  rviz_config = os.path.join(
      get_package_share_directory('planning'),
      'planning',
      'duplo_config.rviz'
  )


  ur_type = LaunchConfiguration("ur_type", default="ur7e")
  launch_rviz = LaunchConfiguration("launch_rviz", default="false")

  camera_transform_node = Node(
      package='perception',
      executable='camera_transform',
  )


  brick_detection_node = Node(
      package='perception',
      executable='brick_detector',
      name='brick_detection_node',
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


  tuck = ExecuteProcess(cmd=['ros2', 'run', 'ur7e_utils', 'tuck'])


  enable = ExecuteProcess(cmd=['ros2', 'run', 'ur7e_utils', 'enable_comms'])


  camera = ExecuteProcess(cmd=[
      'ros2', 'launch', 'realsense2_camera', 'rs_launch.py',
      'pointcloud.enable:=true', 'rgb_camera.color_profile:=1920x1080x30'
  ])

  moveit = ExecuteProcess(cmd=[
      'ros2', 'launch', 'ur_moveit_config', 'ur_moveit.launch.py',
      'ur_type:=ur7e', 'launch_rviz:=false'
  ])


  rviz_on = ExecuteProcess(
      cmd=['rviz2', '-d', rviz_config],
      output={'stdout': 'log', 'stderr':'log'},
      )


  return LaunchDescription([
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
              on_exit=[camera],
          )
      ),


      RegisterEventHandler(
          event_handler=OnProcessStart(
              target_action=camera,
              on_start=[camera_transform_node],
          )
      ),


      RegisterEventHandler(
          event_handler=OnProcessStart(
              target_action=camera_transform_node,
              on_start=[rviz_on],
          )
      ),


      RegisterEventHandler(
          event_handler=OnProcessStart(
              target_action=rviz_on,
              on_start=[planning_main_node],
          )
      ),


      RegisterEventHandler(
          event_handler=OnProcessStart(
              target_action=planning_main_node,
              on_start=[brick_detection_node],
          )
      ),


      RegisterEventHandler(
          event_handler=OnProcessStart(
              target_action=brick_detection_node,
              on_start=[planning_node_node],
          )
      ),
  ])

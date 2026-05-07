import argparse
import os
from pathlib import Path  # noqa: E402
import sys

# Hack to get relative import of .camera_config file working
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)

from camera_config import CameraConfig, USB_CAM_DIR  # noqa: E402

from launch import LaunchDescription  # noqa: E402
from launch.actions import GroupAction  # noqa: E402
from launch_ros.actions import Node  # noqa: E402

CAMERAS = []
CAMERAS.append(
    # CameraConfig(
    #     name='camera1',
    #     param_path=Path(USB_CAM_DIR, 'config', 'params_1.yaml')
    # )
    # Add more Camera's here and they will automatically be launched below
    CameraConfig(
        name='camera1',
        param_path=Path(USB_CAM_DIR, 'config', 'params.yaml')
    )
)


def generate_launch_description():
    rviz_config_dir=os.path.join(
    get_package_share_Directory('planning'),
    'rviz',
    'duplo_config.rviz')
    return LaunchDescription([
    ExecuteProcess( cmd=[ 'ros2', 'launch', 'realsense2_camera', 'rs_launch.py', 'pointcloud.enable:=true', 
            'rgb_camera.color_profile:=1920x1080x30' ])

    ExecuteProcess( cmd=[ 'ros2','run', 'ur7e_utils', 'enable_comms'])

    ExecuteProcess( cmd=['ros2', 'run', 'ur7e_utils', 'tuck'])

    static_tf_broadcaster_node=[
        Node(
            package='perception', executable='static_tf_transform.py'
        )
    ]

    brick_detector_node = [
        Node(
            package='perception',executable='brick_detector',output='screen',name='Brick_detection_node'
        )
    ]
    
    main_node=[
        Node(package='planning', executable='main',output='screen')]
        
    ik_node =[
        Node(package='planning',executable='ik')]
        
    planning_node_node=[
        Node(package='planning',executable='planning_node')]    

    ExecuteProcess(
        cmd=['rviz2','-d','rviz_config_dir'],
        output='screen'
    )

    ])



import os
from ament_index_python.packages import get_package_share_directory, get_package_prefix
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    config = 'src/aqua_station_keeping/config/params.yaml'

    node=Node(
        package = 'aqua_station_keeping',
        name = 'underwater_adversary_command_publisher',
        executable = 'underwater_adversary_command_publisher',
        parameters = [config], 
        output='screen',
        emulate_tty=True,
    )
    ld.add_action(node)
    return ld

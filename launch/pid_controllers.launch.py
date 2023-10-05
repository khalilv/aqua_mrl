from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    config = 'src/aqua_station_keeping/config/params.yaml'

    node=Node(
        package = 'aqua_station_keeping',
        name = 'pid_controllers',
        executable = 'pid_controllers',
        parameters = [config], 
        output='screen',
        emulate_tty=True,
    )
    ld.add_action(node)
    return ld

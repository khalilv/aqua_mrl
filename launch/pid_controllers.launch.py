from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    config = 'src/aqua_rl/config/params.yaml'

    node=Node(
        package = 'aqua_rl',
        name = 'pid_controllers',
        executable = 'pid_controllers',
        parameters = [config], 
        output='screen',
        emulate_tty=True,
    )
    ld.add_action(node)
    return ld

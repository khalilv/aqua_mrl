from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    config = 'src/aqua_mrl/config/params.yaml'

    node=Node(
        package = 'aqua_mrl',
        name = 'underwater_adversary_command_publisher',
        executable = 'underwater_adversary_command_publisher',
        parameters = [config], 
        output='screen',
        emulate_tty=True,
    )
    ld.add_action(node)
    return ld

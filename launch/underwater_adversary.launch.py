from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    config = 'src/aqua_pipeline_inspection/config/params.yaml'

    node=Node(
        package = 'aqua_pipeline_inspection',
        name = 'underwater_adversary_command_publisher',
        executable = 'underwater_adversary_command_publisher',
        parameters = [config], 
        output='screen',
        emulate_tty=True,
    )
    ld.add_action(node)
    return ld

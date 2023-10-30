from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    ld = LaunchDescription()

    # config = 'src/aqua_pipeline_inspection/config/params.yaml'

    node1=Node(
        package = 'aqua_pipeline_inspection',
        name = 'manual_controller',
        executable = 'manual_controller',
        output='screen',
        emulate_tty=True,
    )
    node2=Node(
        package = 'aqua_pipeline_inspection',
        name = 'yolo_detector',
        executable = 'yolo_detector',
        output='screen',
        emulate_tty=True,
    )
    ld.add_action(node1)
    ld.add_action(node2)
    return ld
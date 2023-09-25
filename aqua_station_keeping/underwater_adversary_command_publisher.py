import rclpy
from rclpy.node import Node
import numpy as np
from ir_msgs.msg import UnderwaterAdversaryCommand

class underwater_adversary_command_publisher(Node):

    def __init__(self):
        super().__init__('underwater_adversary_command_publisher')
        self.publisher = self.create_publisher(UnderwaterAdversaryCommand, 'simulator/adversary_command', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.publish_command)
        self.cmd = UnderwaterAdversaryCommand()


    def publish_command(self):
        self.cmd.current_x = np.random.uniform()
        self.cmd.current_y = np.random.uniform()
        self.cmd.current_z = np.random.uniform()
        self.publisher.publish(self.cmd)
        self.get_logger().info('Publishing adversary command')

def main(args=None):
    rclpy.init(args=args)

    publisher = underwater_adversary_command_publisher()

    rclpy.spin(publisher)

    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
import rclpy
from rclpy.node import Node
import numpy as np
from aqua2_interfaces.msg import UnderwaterAdversaryCommand

class underwater_adversary_command_publisher(Node):

    def __init__(self):
        super().__init__('underwater_adversary_command_publisher')
        self.declare_parameters(namespace='',
                                parameters=[
                                    ('current_magnitude', 0.0)
                                ])
        self.publisher = self.create_publisher(UnderwaterAdversaryCommand, 'simulator/adversary_command', 10)
        timer_period = 20  # seconds
        self.timer = self.create_timer(timer_period, self.publish_command)
        self.cmd = UnderwaterAdversaryCommand()
        self.current_magnitude = self.get_parameter('current_magnitude').get_parameter_value().double_value
        print('Initialized: underwater adversary')

    def publish_command(self):
        x = np.random.uniform(low=-1.0, high=1.0)
        z = np.random.uniform(low=-1.0, high=1.0)
        
        #scale up vector to current magnitude
        n = np.sqrt(np.square(x) + np.square(z))
        self.cmd.current_x = (self.current_magnitude/n) * x
        self.cmd.current_z = (self.current_magnitude/n) * z
        self.cmd.current_y = 0.0
        
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
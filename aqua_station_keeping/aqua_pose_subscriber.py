import rclpy
from rclpy.node import Node
import numpy as np
from ir_msgs.msg import AquaPose

class aqua_pose_subscriber(Node):

    def __init__(self):
        super().__init__('aqua_pose_subscriber')
        self.subscriber = self.create_subscription(AquaPose, 'simulator/aqua_pose', self.pose_callback, 10)


    def pose_callback(self, msg):
        self.get_logger().info('Recieved aqua pose: "%s"' % msg)

def main(args=None):
    rclpy.init(args=args)

    subscriber = aqua_pose_subscriber()

    rclpy.spin(subscriber)

    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
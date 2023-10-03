import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import Imu

class imu_subscriber(Node):
    def __init__(self):
        super().__init__('imu_subscriber')
        self.imu_subscriber = self.create_subscription(Imu, '/imu/filtered_data', self.imu_callback, 10)
        self.measured_roll_angle = 0.0

    def imu_callback(self, imu):
        self.calculate_roll(imu)

    def calculate_roll(self, imu):
        ay = imu.linear_acceleration.y
        az = imu.linear_acceleration.z
        self.measured_roll_angle = np.arctan2(ay,az) * 180/np.pi
        print(self.measured_roll_angle)

def main(args=None):
    rclpy.init(args=args)

    subscriber = imu_subscriber()

    rclpy.spin(subscriber)

    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

    

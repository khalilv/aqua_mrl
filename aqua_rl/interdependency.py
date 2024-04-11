import rclpy
from rclpy.node import Node
import time
from aqua2_interfaces.msg import AquaPose, Command
from aqua_rl import hyperparams
import numpy as np

class interdependency(Node):
    def __init__(self):
        super().__init__('interdependency')
        self.queue_size = hyperparams.queue_size_
        self.timer = self.create_timer(10.0, self.switch_rate)
        self.imu_subscriber = self.create_subscription(AquaPose, '/aqua/pose', self.imu_callback, self.queue_size)
        self.command_publisher = self.create_publisher(Command, '/a13/command', self.queue_size)
        
        self.command = Command()
        self.command.pitch = 0.0
        self.command.yaw = 0.0
        self.command.roll = 0.0
        self.command.heave = 0.0
        self.command.speed = hyperparams.speed_

        self.yaw_angles = []
        self.pitch_angles = []
        self.times = []

        self.yaw = 0.0

        self.mode = 0
        

    def switch_rate(self):
        if self.mode == 0:
            print('switching rate')
            self.yaw = 0.25
        elif self.mode == 1:
            print('saving data')
            with open('rolln20.npy', 'wb') as f:
                np.save(f, np.array(self.yaw_angles))
                np.save(f, np.array(self.pitch_angles))
                np.save(f, np.array(self.times))
        self.mode += 1

    def imu_callback(self, imu):
        self.yaw_angles.append(imu.yaw)
        self.pitch_angles.append(imu.pitch)
        self.times.append(time.time())
        self.command.pitch = 0.0
        self.command.yaw = self.yaw
        self.command_publisher.publish(self.command)
        return
    
def main(args=None):
    rclpy.init(args=args)
    node = interdependency()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
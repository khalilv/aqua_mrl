import rclpy
import numpy as np
from rclpy.node import Node
from aqua_rl.control.PID import AnglePID
from aqua2_interfaces.msg import AquaPose, Command
from aqua_rl import hyperparams
from std_msgs.msg import Float32
from aqua_rl.helpers import action_mapping

class autopilot(Node):
    def __init__(self):
        super().__init__('autopilot')
        self.queue_size = hyperparams.queue_size_

        self.imu_subscriber = self.create_subscription(AquaPose, '/aqua/pose', self.imu_callback, self.queue_size)
        self.command_publisher = self.create_publisher(Command, '/a13/command', self.queue_size)
        self.action_subscriber = self.create_subscription(
            Float32,
            hyperparams.autopilot_command_,
            self.action_callback,
            self.queue_size)
        
        self.measured_roll_angle = None
        self.measured_pitch_angle = None
        self.measured_yaw_angle = None

        self.speed = hyperparams.speed_
        self.roll_gains = hyperparams.roll_gains_
        self.pitch_gains = hyperparams.autopilot_pitch_gains_
        self.yaw_gains = hyperparams.autopilot_yaw_gains_
        
        self.pitch_actions = np.linspace(-hyperparams.pitch_angle_limit_, hyperparams.pitch_angle_limit_, hyperparams.pitch_action_space_)
        self.yaw_actions = np.linspace(-hyperparams.yaw_angle_limit_, hyperparams.yaw_angle_limit_, hyperparams.yaw_action_space_)

        self.roll_target = 0.0
        self.pitch_target = 0.0
        self.yaw_target = 90.0

        self.roll_pid = AnglePID(target = self.roll_target, gains = self.roll_gains, reverse=True)
        self.pitch_pid = AnglePID(target = self.pitch_target, gains = self.pitch_gains)
        self.yaw_pid = AnglePID(target = self.yaw_target, gains = self.yaw_gains)

        self.command = Command()
        self.command.pitch = 0.0
        self.command.yaw = 0.0
        self.command.roll = 0.0
        self.command.heave = 0.0
        self.command.speed = self.speed

        print('Initialized: autopilot')

    def imu_callback(self, imu):
        self.measured_roll_angle = self.calculate_roll(imu)
        self.measured_pitch_angle = self.calculate_pitch(imu)
        self.measured_yaw_angle = self.calculate_yaw(imu)

        self.command.pitch = self.pitch_pid.control(self.measured_pitch_angle)
        self.command.roll = self.roll_pid.control(self.measured_roll_angle)
        self.command.yaw = self.yaw_pid.control(self.measured_yaw_angle)
        self.command_publisher.publish(self.command)
        return
    
    def action_callback(self, a):
        action = a.data
        pitch_idx, yaw_idx = action_mapping(action, hyperparams.yaw_action_space_)
        pitch_offset = self.pitch_actions[int(pitch_idx)]
        yaw_offset = self.yaw_actions[int(yaw_idx)]
        self.pitch_pid.target += pitch_offset
        self.yaw_pid.target += yaw_offset
        return

    def calculate_roll(self, imu):
        return imu.roll
    
    def calculate_pitch(self, imu):
        return imu.pitch
    
    def calculate_yaw(self, imu):
        return imu.yaw

def main(args=None):
    rclpy.init(args=args)

    node = autopilot()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

    

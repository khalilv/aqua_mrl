import rclpy
from rclpy.node import Node
from aqua_pipeline_inspection.control.PID import AnglePID
from aqua2_interfaces.msg import AquaPose, Command
from time import time 

class pid_controllers(Node):
    def __init__(self):
        super().__init__('pid_controllers')
        self.declare_parameters(namespace='',
                                parameters=[
                                    ('roll_gains', [0.0,0.0,0.0]),
                                    ('pitch_gains', [0.0,0.0,0.0]),
                                    ('pitch_target', 0.0),
                                    ('roll_target', 0.0),
                                    ('speed', 0.0),
                                    
                                ])
        self.imu_subscriber = self.create_subscription(AquaPose, '/aqua/pose', self.imu_callback, 10)
        self.command_publisher = self.create_publisher(Command, '/a13/command', 10)
        self.measured_roll_angle = 0.0
        self.measured_pitch_angle = 0.0
        self.speed = self.get_parameter('speed').get_parameter_value().double_value
        self.roll_gains = self.get_parameter('roll_gains').get_parameter_value().double_array_value
        self.pitch_gains = self.get_parameter('pitch_gains').get_parameter_value().double_array_value
        self.roll_target = self.get_parameter('roll_target').get_parameter_value().double_value
        self.pitch_target = self.get_parameter('pitch_target').get_parameter_value().double_value
        self.roll_pid = AnglePID(target = self.roll_target, gains = self.roll_gains, reverse=True)
        self.pitch_pid = AnglePID(target = self.pitch_target, gains = self.pitch_gains)
        print('Initialized: PID controllers')

    def imu_callback(self, imu):
        self.measured_roll_angle = self.calculate_roll(imu)
        self.measured_pitch_angle = self.calculate_pitch(imu)
        command = Command();
        command.speed =self.speed
        command.pitch = self.pitch_pid.control(self.measured_pitch_angle)
        command.roll = self.roll_pid.control(self.measured_roll_angle)
        command.heave = 0.0
        command.yaw = 0.0
        self.command_publisher.publish(command)
        print(command)
        return command

    def calculate_roll(self, imu):
        return imu.roll
    
    def calculate_pitch(self, imu):
        return imu.pitch

def main(args=None):
    rclpy.init(args=args)

    node = pid_controllers()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

    

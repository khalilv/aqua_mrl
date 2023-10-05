import rclpy
from rclpy.node import Node
from aqua_station_keeping.control.PID import AnglePID, PID
from aqua2_interfaces.msg import AquaPose, Command

class pid_controllers(Node):
    def __init__(self):
        super().__init__('pid_controllers')
        self.declare_parameters(namespace='',
                                parameters=[
                                    ('roll_gains', [0.0,0.0,0.0]),
                                    ('pitch_gains', [0.0,0.0,0.0]),
                                    ('depth_gains', [0.0,0.0,0.0]),
                                    ('depth_target', 0.0),
                                    ('pitch_target', 0.0),
                                    ('roll_target', 0.0),
                                ])
        self.imu_subscriber = self.create_subscription(AquaPose, '/aqua/pose', self.imu_callback, 10)
        self.command_publisher = self.create_publisher(Command, '/a13/command', 10)
        self.measured_roll_angle = 0.0
        self.measured_pitch_angle = 0.0
        self.measured_yaw_angle = 0.0
        self.measured_depth = 0.0
        self.roll_gains = self.get_parameter('roll_gains').get_parameter_value().double_array_value
        self.pitch_gains = self.get_parameter('pitch_gains').get_parameter_value().double_array_value
        self.depth_gains = self.get_parameter('depth_gains').get_parameter_value().double_array_value
        self.depth_target = self.get_parameter('depth_target').get_parameter_value().double_value
        self.roll_target = self.get_parameter('roll_target').get_parameter_value().double_value
        self.pitch_target = self.get_parameter('pitch_target').get_parameter_value().double_value
        self.roll_pid = AnglePID(target = self.roll_target, gains = self.roll_gains, reverse=True)
        self.pitch_pid = AnglePID(target = self.pitch_target, gains = self.pitch_gains)
        self.heave_pid = PID(target = self.depth_target, gains = self.depth_gains)
        print('Initialized: PID controllers')

    def imu_callback(self, imu):
        self.measured_roll_angle = self.calculate_roll(imu)
        self.measured_pitch_angle = self.calculate_pitch(imu)
        self.measured_yaw_angle = self.calculate_yaw(imu)
        self.measured_depth = self.get_depth(imu)
        command = Command();
        command.speed = 0.0
        command.yaw = 0.0
        command.pitch = self.pitch_pid.control(self.measured_pitch_angle)
        command.roll = self.roll_pid.control(self.measured_roll_angle)
        command.heave = self.heave_pid.control(self.measured_depth)
        self.command_publisher.publish(command)
        print(command)

    def calculate_roll(self, imu):
        # ay = imu.linear_acceleration.y
        # az = imu.linear_acceleration.z
        return imu.roll
    
    def calculate_pitch(self, imu):
        # ax = imu.linear_acceleration.x
        # az = imu.linear_acceleration.z
        return imu.pitch
    
    def calculate_yaw(self, imu):
        # ax = imu.linear_acceleration.x
        # az = imu.linear_acceleration.z
        return imu.yaw
    
    def get_depth(self, imu):
        return imu.y

def main(args=None):
    rclpy.init(args=args)

    node = pid_controllers()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

    

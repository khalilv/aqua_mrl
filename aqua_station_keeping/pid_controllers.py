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
        self.pitch_leeway = 10
        self.roll_leeway = 10
        self.depth_leeway = 3
        print('Initialized: PID controllers')

    def imu_callback(self, imu):
        self.measured_roll_angle = self.calculate_roll(imu)
        self.measured_pitch_angle = self.calculate_pitch(imu)
        self.measured_depth = self.get_depth(imu)
        if abs(self.pitch_pid.calculate_angle_error(self.pitch_pid.convert_to_0_360(self.measured_pitch_angle))) > self.pitch_leeway:
            print("STABILIZING PITCH")
            command = self.stabilize_pitch()
            self.command_publisher.publish(command)
        elif abs(self.roll_pid.calculate_angle_error(self.roll_pid.convert_to_0_360(self.measured_roll_angle))) > self.roll_leeway:
            print("STABILIZING ROLL")
            command = self.stabilize_roll()
            self.command_publisher.publish(command)
        elif abs(self.measured_depth - self.depth_target) > self.depth_leeway:  
            print("STABILIZING DEPTH")
            command = self.stabilize_depth()
            self.command_publisher.publish(command)
        else:
            print("STABLE")
            command = self.stable()
            self.command_publisher.publish(command)

    def calculate_roll(self, imu):
        return imu.roll
    
    def calculate_pitch(self, imu):
        return imu.pitch
      
    def get_depth(self, imu):
        return imu.y
    
    def stabilize_pitch(self):
        command = Command();
        command.pitch = self.pitch_pid.control(self.measured_pitch_angle)
        command.roll = self.roll_pid.control(self.measured_roll_angle)
        command.heave = self.heave_pid.control(self.measured_depth)
        command.speed = 10 * command.pitch
        command.yaw = 0.1
        return command
    
    def stabilize_roll(self):
        command = Command();
        command.pitch = 0.0
        command.roll = self.roll_pid.control(self.measured_roll_angle)
        command.heave = self.heave_pid.control(self.measured_depth)
        command.speed = 0.0
        command.yaw = 0.0
        return command
    
    def stabilize_depth(self):
        command = Command();
        command.pitch = 0.0
        command.roll = 0.0
        command.heave = self.heave_pid.control(self.measured_depth)
        command.speed = 0.0
        command.yaw = 0.0
        return command

    def stable(self):
        command = Command();
        command.pitch = 0.0
        command.roll = 0.0
        command.heave = 0.0
        command.speed = 0.0
        command.yaw = 0.0
        return command

def main(args=None):
    rclpy.init(args=args)

    node = pid_controllers()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

    

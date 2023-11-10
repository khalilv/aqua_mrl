import rclpy
from rclpy.node import Node
from aqua_pipeline_inspection.control.PID import AnglePID, PID
from aqua2_interfaces.msg import AquaPose, Command
from std_msgs.msg import Float32

class pid_controllers(Node):
    def __init__(self):
        super().__init__('pid_controllers')
        self.declare_parameters(namespace='',
                                parameters=[
                                    ('roll_gains', [0.0,0.0,0.0]),
                                    ('pitch_gains', [0.0,0.0,0.0]),
                                    ('depth_target', 0.0),
                                    ('roll_target', 0.0),
                                    ('speed', 0.0),
                                    
                                ])
        self.imu_subscriber = self.create_subscription(AquaPose, '/aqua/pose', self.imu_callback, 10)
        self.depth_subscriber = self.create_subscription(Float32, '/aqua/depth', self.depth_sensor_callback, 10)
        self.command_publisher = self.create_publisher(Command, '/a13/command', 10)
        self.measured_roll_angle = 0.0
        self.measured_depth = 0.0
        self.speed = self.get_parameter('speed').get_parameter_value().double_value
        self.roll_gains = self.get_parameter('roll_gains').get_parameter_value().double_array_value
        self.pitch_gains = self.get_parameter('pitch_gains').get_parameter_value().double_array_value
        self.roll_target = self.get_parameter('roll_target').get_parameter_value().double_value
        self.depth_target = self.get_parameter('depth_target').get_parameter_value().double_value
        self.roll_pid = AnglePID(target = self.roll_target, gains = self.roll_gains, reverse=True)
        self.pitch_pid = PID(target = self.depth_target, gains = self.pitch_gains, command_range=[-0.02,0.02], normalization_factor=5)

        #initialize command
        self.command = Command()
        self.command.speed = self.speed
        self.command.roll = 0.0
        self.command.pitch = 0.0
        self.command.yaw = 0.0
        self.command.heave = 0.0

        print('Initialized: PID controllers')

    def imu_callback(self, imu):
        self.measured_roll_angle = self.calculate_roll(imu)
        self.command.pitch = self.pitch_pid.control(self.measured_depth)
        self.command.roll = self.roll_pid.control(self.measured_roll_angle)
        self.command_publisher.publish(self.command)
        print(self.command)
        return

    def depth_sensor_callback(self, depth):
        self.measured_depth = depth.data
        return

    def calculate_roll(self, imu):
        return imu.roll
    

def main(args=None):
    rclpy.init(args=args)

    node = pid_controllers()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

    

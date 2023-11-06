import rclpy
from rclpy.node import Node
from aqua2_interfaces.msg import Command, AquaPose
from aqua_pipeline_inspection.control.PID import AnglePID, PID
from std_msgs.msg import Float32

class pid_pipeline_inspection(Node):
    def __init__(self):
        super().__init__('pid_pipeline_inspection')
        self.command_publisher = self.create_publisher(Command, '/a13/command', 10)
        self.imu_subscriber = self.create_subscription(AquaPose, '/aqua/pose', self.imu_callback, 10)
        self.pipeline_error_subscriber = self.create_subscription(
            Float32, 
            '/pipeline/error', 
            self.pipeline_error_callback, 
            10)
        self.roll_pid = AnglePID(target = 0.0, gains = [2.75, 0.0, 3.75], reverse=True)
        self.pitch_pid = AnglePID(target = 0.0, gains = [0.5181, 0.0, 0.9])
        self.yaw_pid = PID(target = 0.0, gains = [0.6, 0.0, 1.1], reverse=True, normalization_factor=700)
        self.measured_roll_angle = 0.0
        self.measured_pitch_angle = 0.0
        self.command = Command()
        self.command.speed = 0.0 
        self.command.roll = 0.0
        self.command.pitch = 0.0
        self.command.yaw = 0.0
        self.img_size = (300, 400)
        print('Initialized: pid pipeline inspection ')
  
    def imu_callback(self, imu):
        self.measured_roll_angle = self.calculate_roll(imu)
        self.measured_pitch_angle = self.calculate_pitch(imu)
        return

    def calculate_roll(self, imu):
        return imu.roll
    
    def calculate_pitch(self, imu):
        return imu.pitch
    
    def pipeline_error_callback(self, error):
        if error.data >= self.img_size[0] + self.img_size[1] + 1: #stop command = w + h + 1
            print('Recieved stop command')
            self.command.speed = 0.0 
            self.command.roll = 0.0
            self.command.pitch = 0.0
            self.command.yaw = 0.0
        else:
            self.command.speed = 0.25 #fixed speed
            self.command.yaw = self.yaw_pid.control(error.data)
            self.command.roll = self.roll_pid.control(self.measured_roll_angle)
            self.command.pitch = self.pitch_pid.control(self.measured_pitch_angle)
        
        self.command_publisher.publish(self.command)
        print(self.command)


def main(args=None):
    rclpy.init(args=args)

    node = pid_pipeline_inspection()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

    



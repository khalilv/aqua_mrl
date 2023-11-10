import rclpy
from rclpy.node import Node
from aqua2_interfaces.msg import Command, AquaPose
from aqua_pipeline_inspection.control.PID import AnglePID, PID
from std_msgs.msg import Float32
import numpy as np 
import os

class pid_pipeline_inspection(Node):
    def __init__(self):
        super().__init__('pid_pipeline_inspection')

        #subscribers and publishers
        self.command_publisher = self.create_publisher(Command, '/a13/command', 10)
        self.imu_subscriber = self.create_subscription(AquaPose, '/aqua/pose', self.imu_callback, 10)
        self.pipeline_error_subscriber = self.create_subscription(
            Float32, 
            '/pipeline/error', 
            self.pipeline_error_callback, 
            10)
        self.depth_subscriber = self.create_subscription(Float32, '/aqua/depth', self.depth_sensor_callback, 10)
        
        #initialize pid controllers
        self.target_depth = 11.0
        self.roll_pid = AnglePID(target = 0.0, gains = [0.1, 0.0, 2.75], reverse=True)
        self.pitch_pid = PID(target = self.target_depth, gains = [0.01, 0.0, 0.25], command_range=[-0.02,0.02], normalization_factor=5)
        self.yaw_pid = PID(target = 0.0, gains = [0.6, 0.0, 1.1], reverse=True, normalization_factor=700)
        self.measured_roll_angle = 0.0
        self.measured_depth = 0.0
        
        #initialize command
        self.command = Command()
        self.command.speed = 0.0 
        self.command.roll = 0.0
        self.command.pitch = 0.0
        self.command.yaw = 0.0
        self.command.heave = 0.0
        
        self.img_size = (300, 400)
        
        #trajectory recording
        self.record_trajectory = True
        self.trajectory = []
        self.save_path = 'src/aqua_pipeline_inspection/aqua_pipeline_inspection/trajectories/'
        self.num_trajectories = len(os.listdir(self.save_path))
        
        #target trajectory
        self.offset_x = 47.14558
        self.offset_z = -19.43558
        self.target_trajectory = '/usr/local/data/kvirji/AQUA/aqua_pipeline_inspection/aqua_pipeline_inspection/trajectories/pipeline_center.npy'
        with open(self.target_trajectory, 'rb') as f:
            self.pipeline_x = np.load(f) + self.offset_x
            self.pipeline_z = np.load(f) + self.offset_z
        
        #define max errors to target trajectory
        self.max_z_error_to_target = 3.0 #if z distance to pipeline > max => halt
        self.max_y_error_to_target = 3.0 #if y distance to pipeline > max => halt

        #end of pipe
        self.finish_line_x = 25 + self.offset_x #25 + offset

        print('Initialized: pid pipeline inspection')
  
    def imu_callback(self, imu):
        self.measured_roll_angle = self.calculate_roll(imu)
        if imu.x > self.finish_line_x:
            self.finish(True)
        elif np.abs(imu.z - self.get_target_z(imu.x)) > self.max_z_error_to_target:
            print('Drifted far from target trajectory in z direction')
            self.finish(False)
        else:
            self.trajectory.append([imu.x, imu.y, imu.z])
        return
    
    def depth_sensor_callback(self, depth):
        self.measured_depth = depth.data
        if np.abs(self.measured_depth - self.target_depth) > self.max_y_error_to_target:
            print('Drifted far from target trajectory in y direction')
            self.finish(False)
        return

    def calculate_roll(self, imu):
        return imu.roll
    
    def get_target_z(self, x):
        ind = np.argwhere(self.pipeline_x >= x)[0][0]
        x1 = self.pipeline_x[ind - 1]
        z1 = self.pipeline_z[ind - 1]
        x2 = self.pipeline_x[ind]
        z2 = self.pipeline_z[ind]
        m = (z2 - z1) / (x2 - x1)
        b = z2 - m * x2
        return m*x + b
    
    def pipeline_error_callback(self, error):
        if error.data >= self.img_size[0] + self.img_size[1] + 1: #stop command = w + h + 1
            print('Recieved stop command from vision module')
            self.finish(False)
        else:
            self.command.speed = 0.25 #fixed speed
            self.command.yaw = self.yaw_pid.control(error.data)
            self.command.roll = self.roll_pid.control(self.measured_roll_angle)
            self.command.pitch = self.pitch_pid.control(self.measured_depth)
        
        self.command_publisher.publish(self.command)
        return 
    
    def finish(self, complete):
        if complete:
            print('Goal reached')
        else:
            print('Goal not reached')

        if self.record_trajectory:
            print('Saving trajectory')
            with open(self.save_path + 'pid_trajectory_with_current_{}.npy'.format(str(self.num_trajectories)), 'wb') as f:
                np.save(f, np.array(self.trajectory))
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)

    node = pid_pipeline_inspection()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

    



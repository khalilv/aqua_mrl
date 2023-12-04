import rclpy
from rclpy.node import Node
from aqua2_interfaces.msg import Command, AquaPose
from ir_aquasim_interfaces.srv import EmptyWorkaround
from aqua_rl.control.PID import AnglePID, PID
from std_msgs.msg import Float32MultiArray, Float32
import numpy as np 
import os
from time import sleep

class pid_pipeline_inspection(Node):
    def __init__(self):
        super().__init__('pid_pipeline_inspection')

        #subscribers and publishers
        self.command_publisher = self.create_publisher(Command, '/a13/command', 10)
        self.imu_subscriber = self.create_subscription(AquaPose, '/aqua/pose', self.imu_callback, 10)
        self.pipeline_parameter_subscriber = self.create_subscription(
            Float32MultiArray, 
            '/pipeline/parameters', 
            self.pipeline_parameters_callback, 
            10)
        self.depth_subscriber = self.create_subscription(Float32, '/aqua/depth', self.depth_callback, 10)

        #initialize pid controllers
        self.target_depth = 10.0
        self.roll_pid = AnglePID(target = 0.0, gains = [2.75, 0.0, 3.75], reverse=True)
        self.pitch_pid = AnglePID(target = 0.0, gains = [0.5181, 0.0, 0.9])
        self.heave_pid = PID(target= self.target_depth, gains=[0.1, 0.0, 0.2], reverse=True)
        self.yaw_pid = PID(target = 0.0, gains = [0.6, 0.0, 1.1], reverse=True, normalization_factor=700)
        self.measured_roll_angle = 0.0
        self.measured_pitch_angle = 0.0
        self.measured_depth = self.target_depth #to avoid any sporadic behaviour at the start
        
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
        self.save_path = 'src/aqua_rl/aqua_rl/trajectories/pid/'
        self.num_trajectories = len(os.listdir(self.save_path))
        
        #target trajectory
        self.offset_x = 47.14558
        self.offset_z = -19.43558
        self.target_trajectory = 'src/aqua_rl/aqua_rl/trajectories/pipeline_center.npy'
        with open(self.target_trajectory, 'rb') as f:
            self.pipeline_x = np.load(f) + self.offset_x
            self.pipeline_z = np.load(f) + self.offset_z
        
        #define max errors to target trajectory
        self.max_z_error_to_target = 3.0 #if z distance to pipeline > max => halt
        self.max_y_error_to_target = 3.0 #if y distance to pipeline > max => halt

        #end of pipe
        self.finish_line_x = 25 + self.offset_x #25 + offset

        #reset command
        self.reset_client = self.create_client(EmptyWorkaround, '/simulator/reset_robot')
        self.reset_req = EmptyWorkaround.Request()
        print('Initialized: pid pipeline inspection')
  
    def imu_callback(self, imu):
        self.measured_roll_angle = self.calculate_roll(imu)
        self.measured_pitch_angle = self.calculate_pitch(imu)
        if imu.x > self.finish_line_x:
            self.finish(True)
        elif np.abs(imu.z - self.get_target_z(imu.x)) > self.max_z_error_to_target:
            print('Drifted far from target trajectory in z direction')
            self.finish(False)
        else:
            self.trajectory.append([imu.x, imu.y, imu.z])
        return
    
    def depth_callback(self, depth):
        self.measured_depth = depth.data
        if np.abs(self.measured_depth - self.target_depth) > self.max_y_error_to_target:
            print('Drifted far from target trajectory in y direction')
            self.finish(False)
        return

    def calculate_roll(self, imu):
        return imu.roll
    
    def calculate_pitch(self, imu):
        return imu.pitch
    
    def get_target_z(self, x):
        ind = np.argwhere(self.pipeline_x >= x)[0][0]
        x1 = self.pipeline_x[ind - 1]
        z1 = self.pipeline_z[ind - 1]
        x2 = self.pipeline_x[ind]
        z2 = self.pipeline_z[ind]
        m = (z2 - z1) / (x2 - x1)
        b = z2 - m * x2
        return m*x + b
    
    def pipeline_parameters_callback(self, error):
        if error.data.count(-1) == len(error.data): #stop command are all -1
            print('Recieved stop command from vision module')
            self.finish(False)
        else:
            self.command.speed = 0.25 #fixed speed
            self.command.yaw = self.yaw_pid.control(error.data[0])
            self.command.roll = self.roll_pid.control(self.measured_roll_angle)
            self.command.pitch = self.pitch_pid.control(self.measured_pitch_angle)
            self.command.heave = self.heave_pid.control(self.measured_depth)
        self.command_publisher.publish(self.command)
        return 
    
    def finish(self, complete):
        if complete:
            print('Goal reached')
        else:
            print('Goal not reached')

        if self.record_trajectory:
            print('Saving trajectory')
            with open(self.save_path + 'trajectory_{}.npy'.format(str(self.num_trajectories)), 'wb') as f:
                np.save(f, np.array(self.trajectory))
            self.num_trajectories += 1
        
        self.reset()
        return
    
    def reset(self):
        print('-------------- Resetting simulation --------------')

        self.reset_client.call_async(self.reset_req)
        
        #reset errors
        self.measured_roll_angle = 0.0
        self.measured_pitch_angle = 0.0
        self.measured_depth = self.target_depth

        #reset trajectory
        self.trajectory = []

        sleep(0.5)
        return
    
def main(args=None):
    rclpy.init(args=args)

    node = pid_pipeline_inspection()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

    



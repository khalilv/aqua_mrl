import rclpy
from rclpy.node import Node
from aqua2_interfaces.msg import Command, AquaPose
from ir_aquasim_interfaces.srv import EmptyWorkaround
from aqua_pipeline_inspection.control.PID import AnglePID, PID
from aqua_pipeline_inspection.control.DQN import DQN
from std_msgs.msg import Float32MultiArray, Float32
import numpy as np 
import os
from time import sleep
import torch


class dqn_controller(Node):
    def __init__(self):
        super().__init__('dqn_controller')

        #subscribers and publishers
        self.command_publisher = self.create_publisher(Command, '/a13/command', 10)
        self.imu_subscriber = self.create_subscription(AquaPose, '/aqua/pose', self.imu_callback, 10)
        self.pipeline_error_subscriber = self.create_subscription(
            Float32MultiArray, 
            '/pipeline/error', 
            self.pipeline_error_callback, 
            10)
        self.depth_subscriber = self.create_subscription(Float32, '/aqua/depth', self.depth_callback, 10)

        #initialize pid controllers
        self.target_depth = 10.0
        self.roll_pid = AnglePID(target = 0.0, gains = [2.75, 0.0, 3.75], reverse=True)
        self.pitch_pid = AnglePID(target = 0.0, gains = [0.5181, 0.0, 0.9])
        self.heave_pid = PID(target= self.target_depth, gains=[0.1, 0.0, 0.2], reverse=True)
        self.measured_roll_angle = 0.0
        self.measured_pitch_angle = 0.0
        self.measured_depth = self.target_depth #to avoid any sporadic behaviour at the start
        
        #dqn controller for yaw 
        self.dqn = DQN(5,3)
        self.num_episodes = 10
        self.episode = 0
        self.state = None
        self.action = None
        self.next_state = None
        self.reward = None

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
        self.save_path = 'src/aqua_pipeline_inspection/aqua_pipeline_inspection/trajectories/dqn/'
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
        self.finish_line_x = -50# 25 + self.offset_x #25 + offset

        #reset command
        self.reset_client = self.create_client(EmptyWorkaround, '/simulator/reset_robot')
        self.reset_req = EmptyWorkaround.Request()
        print('Initialized: dqn controller')
  
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
    
    def pipeline_error_callback(self, error):
        pid_error = error.data[0]
        theta = error.data[2]
        centroid_x = error.data[3]
        centroid_y = error.data[4]
        if pid_error >= self.img_size[0] + self.img_size[1] + 1: #stop command = w + h + 1
            print('Recieved stop command from vision module')
            self.finish(False)
        else:
            self.next_state = torch.tensor([theta, centroid_x, centroid_y], dtype=torch.float32, device=self.dqn.device).unsqueeze(0)
            reward = self.reward_calculation(theta, centroid_x, centroid_y)
            self.reward = torch.tensor([reward], device=self.dqn.device)
            self.action = self.dqn.select_action(self.next_state)
            if self.state is not None:
                self.dqn.memory.push(self.state, self.action, self.next_state, self.reward)
            
            self.state = self.next_state
            # Perform one step of the optimization (on the policy network)
            self.dqn.optimize()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = self.dqn.target_net.state_dict()
            policy_net_state_dict = self.dqn.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*self.dqn.TAU + target_net_state_dict[key]*(1-self.dqn.TAU)
            self.dqn.target_net.load_state_dict(target_net_state_dict)
            
            yaw_action = self.action.detach().cpu().numpy()[0][0]
            if yaw_action == 0:
                self.command.yaw = -0.5
            if yaw_action == 1:
                self.command.yaw = -0.25
            if yaw_action == 2:
                self.command.yaw = 0.0
            if yaw_action == 3:
                self.command.yaw = 0.25
            if yaw_action == 4:
                self.command.yaw = 0.5
            self.command.speed = 0.25 #fixed speed
            self.command.roll = self.roll_pid.control(self.measured_roll_angle)
            self.command.pitch = self.pitch_pid.control(self.measured_pitch_angle)
            self.command.heave = self.heave_pid.control(self.measured_depth)
            self.command_publisher.publish(self.command)
            return 
    
    def reward_calculation(self, theta, centroid_x, centroid_y):
        x_bounds = [25 - self.img_size[1]/2, 25 + self.img_size[1]/2]
        y_bounds = [25 - self.img_size[0]/2, 25 + self.img_size[0]/2]
        if np.abs(theta) < 15*np.pi/180 and centroid_x > x_bounds[0] and centroid_x < x_bounds[1] and centroid_y > y_bounds[0] and centroid_y < y_bounds[1]:
            return 0
        else:
            return -1
        

    def finish(self, complete):
        if complete:
            print('Goal reached')
            self.dqn.memory.push(self.state, self.action, None, torch.tensor([10], device=self.dqn.device))
        else:
            print('Goal not reached')
            self.dqn.memory.push(self.state, self.action, None, torch.tensor([-10], device=self.dqn.device))
        
        if self.record_trajectory:
            print('Saving trajectory')
            with open(self.save_path + 'trajectory_{}.npy'.format(str(self.num_trajectories)), 'wb') as f:
                np.save(f, np.array(self.trajectory))
            self.num_trajectories += 1
        
        self.reset()
        return
    
    def reset(self):
        print('Resetting simulation')

        self.reset_client.call_async(self.reset_req)
        
        #reset errors
        self.measured_roll_angle = 0.0
        self.measured_pitch_angle = 0.0
        self.measured_depth = self.target_depth

        #reset trajectory
        self.trajectory = []
        self.episode += 1
        self.state = None
        sleep(0.5)
        return
    
def main(args=None):
    rclpy.init(args=args)

    node = dqn_controller()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

    



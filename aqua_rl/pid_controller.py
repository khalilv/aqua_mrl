import rclpy
import torch
import numpy as np 
import os
import subprocess
from rclpy.node import Node
from aqua2_interfaces.msg import Command, AquaPose, DiverCommand
from ir_aquasim_interfaces.srv import SetPosition
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32MultiArray
from time import sleep, time
from aqua_rl.control.PID import AnglePID, PID
from aqua_rl.control.DQN import ReplayMemory
from aqua_rl.helpers import reward_calculation, action_mapping, inverse_mapping
from aqua_rl import hyperparams
from torch.utils.tensorboard import SummaryWriter 

class pid_controller(Node):
    def __init__(self):
        super().__init__('pid_controller')

        #hyperparams
        self.queue_size = hyperparams.queue_size_
        self.roll_gains = hyperparams.roll_gains_
        self.pitch_gains = hyperparams.pitch_gains_
        self.yaw_gains = hyperparams.yaw_gains_
        self.history_size = hyperparams.history_size_
        self.pitch_limit = hyperparams.pitch_limit_
        self.yaw_limit = hyperparams.yaw_limit_
        self.yaw_action_space = hyperparams.yaw_action_space_
        self.pitch_action_space = hyperparams.pitch_action_space_
        self.img_size = hyperparams.img_size_
        self.depth_range = hyperparams.depth_range_
        self.train_duration = hyperparams.train_duration_
        self.diver_max_speed = hyperparams.diver_max_speed_
        self.frame_skip = hyperparams.frame_skip_

        #subscribers and publishers
        self.command_publisher = self.create_publisher(Command, '/a13/command', self.queue_size)
        self.imu_subscriber = self.create_subscription(AquaPose, '/aqua/pose', self.imu_callback, self.queue_size)
        self.detection_subscriber = self.create_subscription(
            Float32MultiArray, 
            '/diver/coordinates', 
            self.detection_callback, 
            self.queue_size)
        self.diver_publisher = self.create_publisher(DiverCommand, 'diver_control', self.queue_size)
        self.diver_pose_subscriber = self.create_subscription(
            AquaPose,
            hyperparams.diver_topic_name_,
            self.diver_pose_callback,
            self.queue_size)
        
        #initialize pid controllers
        self.roll_pid = AnglePID(target = 0.0, gains = self.roll_gains, reverse=True)
        self.pitch_pid = PID(target = self.img_size/2, gains = self.pitch_gains, reverse=True, normalization_factor=self.img_size/2)
        self.yaw_pid = PID(target = self.img_size/2, gains = self.yaw_gains, reverse=True, normalization_factor=self.img_size/2)
        self.measured_roll_angle = None

        self.yaw_actions = np.linspace(-self.yaw_limit, self.yaw_limit, self.yaw_action_space)
        self.pitch_actions = np.linspace(-self.pitch_limit, self.pitch_limit, self.pitch_action_space)

        #initialize command
        self.command = Command()
        self.command.speed = 0.0 
        self.command.roll = 0.0
        self.command.pitch = 0.0
        self.command.yaw = 0.0
        self.command.heave = 0.0
        
        #command
        self.diver_cmd = DiverCommand()
        self.diver_pose = None
        
        #flush queues
        self.flush_steps = self.queue_size + 30
        self.flush_commands = self.flush_steps
        self.zero_command_steps = int(self.flush_commands / 5)
        self.zero_commands = 0
        self.flush_imu = 0
        self.flush_diver = 0
        self.flush_detection = 0

        #state and depth history
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = []
        self.episode_rewards = []
        self.starting_action = inverse_mapping(self.pitch_action_space//2,self.yaw_action_space//2,self.yaw_action_space)
        self.action = torch.tensor([[self.starting_action]], device=self.device, dtype=torch.long)
        self.action_history = [self.starting_action]
        self.erm = ReplayMemory(10000)
        self.state = None
        self.next_state = None
        self.reward = None

        self.aqua_trajectory = []
        self.diver_trajectory = []

        self.duration = 0
        self.finished = False
        self.complete = False

        self.timer = self.create_timer(5, self.publish_diver_command)

        print('Initialized: PID controller')
  
    def imu_callback(self, imu):
        
        #finished flag
        if self.finished:
            return
        
        #flush queue
        if self.flush_imu < self.flush_steps:
            self.flush_imu += 1
            return
        
        self.measured_roll_angle = self.calculate_roll(imu)

        self.aqua_trajectory.append([imu.x, imu.y, imu.z])
        return
            
    def publish_diver_command(self):     
        if self.diver_pose:

            #scale vector to current magnitude
            self.diver_cmd.vx = np.random.uniform(hyperparams.speed_, hyperparams.speed_+0.1)
            self.diver_cmd.vy = np.random.uniform(-1,1)
            self.diver_cmd.vz = np.random.uniform(-1,1)

            speed = np.sqrt(np.square(self.diver_cmd.vy) + np.square(self.diver_cmd.vz))
            if speed > self.diver_max_speed:
                self.diver_cmd.vy = self.diver_cmd.vy * self.diver_max_speed/speed
                self.diver_cmd.vz = self.diver_cmd.vz * self.diver_max_speed/speed
                
            if self.diver_pose[1] > self.depth_range[0] and self.diver_cmd.vy > 0:
                self.diver_cmd.vy *= -1
            elif self.diver_pose[1] < self.depth_range[1] and self.diver_cmd.vy < 0:
                self.diver_cmd.vy *= -1

            #publish
            self.diver_publisher.publish(self.diver_cmd)
            print('Publishing diver command')

            return 
    

    def diver_pose_callback(self, pose):    
        
        #finished flag
        if self.finished:
            return
        
        #flush queue
        if self.flush_diver < self.flush_steps:
            self.flush_diver += 1
            return 
           
        self.diver_pose = [pose.x, pose.y, pose.z]
        self.diver_trajectory.append(self.diver_pose)

        return 
    
    def calculate_roll(self, imu):
        return imu.roll
        
    def detection_callback(self, coords):
        
        #flush detections queue
        if self.flush_detection< self.flush_steps:
            self.flush_detection += 1
            return

        #flush out command queue
        if self.flush_commands < self.flush_steps:
            if self.zero_commands < self.zero_command_steps:
                self.command.speed = hyperparams.speed_ 
                self.command.roll = 0.0
                self.command.pitch = 0.0
                self.command.yaw = 0.0
                self.command.heave = 0.0
                self.command_publisher.publish(self.command)
                # #reset adv
                # self.adv_command.current_x = 0.0
                # self.adv_command.current_z = 0.0
                # self.adv_command.current_y = 0.0
                # self.adv_command_publisher.publish(self.adv_command)
                self.zero_commands += 1
            self.flush_commands += 1
            return
        
        #if finished, reset simulation
        if self.finished:
            self.finish()
            return
        
        coords = np.array(coords.data)
        #check for null input from detection module
        if coords.sum() < 0:
            print("Recieved null input from vision module. Terminating.")
            self.flush_commands = 0
            self.finished = True
            self.complete = False
            return
        else:
            detected_center = [(coords[2] + coords[0])/2, (coords[3] + coords[1])/2]
       
        if self.duration > self.train_duration:
            print("Duration Reached")
            self.flush_commands = 0
            self.finished = True
            self.complete = True
            return
        self.duration += 1
        
        self.history.append(detected_center)
        if len(self.history) == self.history_size and len(self.action_history) == self.history_size:
            ns = np.concatenate((np.array(self.history).flatten(), np.array(self.action_history).flatten()))
            self.next_state = torch.tensor(ns, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            reward = reward_calculation(detected_center, self.img_size, self.img_size)
            self.episode_rewards.append(reward)
            self.reward = torch.tensor([reward], dtype=torch.float32, device=self.device)

            if self.state is not None:
                self.erm.push(self.state, self.action, self.next_state, self.reward)
            
            pitch_idx = self.discretize(self.pitch_pid.control(detected_center[1]), self.pitch_actions)
            yaw_idx = self.discretize(self.yaw_pid.control(detected_center[0]), self.yaw_actions)
            action_idx = inverse_mapping(pitch_idx, yaw_idx, self.yaw_action_space)
            self.action = torch.tensor([[action_idx]], device=self.device, dtype=torch.long)
            self.state = self.next_state
            
            self.history = self.history[self.frame_skip:]
            self.action_history = self.action_history[self.frame_skip:]
        
        #adversary action
        # x,y,z = adv_mapping(self.adv_action.detach().cpu().numpy()[0][0])
        # self.adv_command.current_x = self.adv_madnitude_x * x
        # self.adv_command.current_z = self.adv_madnitude_z * z
        # self.adv_command.current_y = self.adv_madnitude_y * y 
        # self.adv_command_publisher.publish(self.adv_command)
        
        #protagonist action
        action_idx = self.action.detach().cpu().numpy()[0][0]
        self.action_history.append(action_idx)
        pitch_idx, yaw_idx = action_mapping(action_idx, self.yaw_action_space)
        self.command.pitch = self.pitch_actions[pitch_idx]
        self.command.yaw = self.yaw_actions[yaw_idx]            
        self.command.speed = hyperparams.speed_ #fixed speed
        self.command.roll = self.roll_pid.control(self.measured_roll_angle)
        self.command_publisher.publish(self.command)
        return 
    
    def discretize(self, v, l):
        index = np.argmin(np.abs(np.subtract(l,v)))
        return index
    
    def finish(self):
         
        self.episode_rewards = np.array(self.episode_rewards)
        print('Episode rewards. Average: ', np.mean(self.episode_rewards), ' Sum: ', np.sum(self.episode_rewards))
        
       
        if self.state is not None and not self.complete:
            self.erm.push(self.state, self.action, None, self.reward)

        
        torch.save({
            'memory': self.erm
        }, 'pid_expert.pt')
        
        with open('pid_traj.npy', 'wb') as f:
            np.save(f, self.episode_rewards)
            np.save(f, np.array(self.aqua_trajectory))
            np.save(f, np.array(self.diver_trajectory))

        print('Saved data. Kill node now.')
        input()

        return

       
def main(args=None):
    rclpy.init(args=args)

    node = pid_controller()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

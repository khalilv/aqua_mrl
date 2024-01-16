import rclpy
import torch
import numpy as np 
import os
import subprocess
from pynput import keyboard
from rclpy.node import Node
from aqua2_interfaces.msg import Command, AquaPose
from ir_aquasim_interfaces.srv import SetPosition
from geometry_msgs.msg import Pose
from std_msgs.msg import UInt8MultiArray, Float32
from time import sleep, time
from aqua_rl.control.PID import AnglePID, PID
from aqua_rl.control.DQN import ReplayMemory
from aqua_rl.helpers import define_template, reward_calculation, random_starting_position
from aqua_rl import hyperparams
from torch.utils.tensorboard import SummaryWriter 


class manual_controller(Node):
    def __init__(self):
        super().__init__('manual_controller')
        #hyperparams
        self.queue_size = hyperparams.queue_size_
        self.roll_gains = hyperparams.roll_gains_
        self.history_size = hyperparams.history_size_
        self.pitch_limit = hyperparams.pitch_limit_
        self.yaw_limit = hyperparams.yaw_limit_
        self.yaw_action_space = hyperparams.yaw_action_space_
        self.pitch_action_space = hyperparams.pitch_action_space_
        self.img_size = hyperparams.img_size_
        self.depth_range = hyperparams.depth_range_
        self.target_depth = hyperparams.target_depth_
        
        #subscribers and publishers
        self.command_publisher = self.create_publisher(Command, '/a13/command', self.queue_size)
        self.imu_subscriber = self.create_subscription(AquaPose, '/aqua/pose', self.imu_callback, self.queue_size)
        self.segmentation_subscriber = self.create_subscription(
            UInt8MultiArray, 
            '/segmentation', 
            self.segmentation_callback, 
            self.queue_size)
        self.depth_subscriber = self.create_subscription(Float32, '/aqua/depth', self.depth_callback, self.queue_size)


        self.roll_pid = AnglePID(target = 0.0, gains = self.roll_gains, reverse=True)
        self.pitch_pid = PID(target = 0.0, gains = [0.005, 0.0, 0.175])

        self.measured_roll_angle = 0.0
        self.relative_depth = None
        
        #dqn controller for yaw and pitch 
        self.yaw_actions = np.linspace(-self.yaw_limit, self.yaw_limit, self.yaw_action_space)
        self.pitch_actions = np.linspace(-self.pitch_limit, self.pitch_limit, self.pitch_action_space)
        self.state = None
        self.next_state = None
        self.state_depths = None
        self.next_state_depths = None
        self.action = None
        self.reward = None
        self.image_history = []
        self.depth_history = []
        self.episode_rewards = []
        self.erm = ReplayMemory(60000)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #trajectory recording
        self.trajectory = []
        self.evaluate = False 

        #target for reward
        self.template = define_template(self.img_size)

        #stopping conditions
        self.empty_state_counter = 0

        #initialize command
        self.command = Command()
        self.command.speed = 0.0 
        self.command.roll = 0.0
        self.command.pitch = 0.0
        self.command.yaw = 0.0
        self.command.heave = 0.0

        self.yaw_action_idx = 1
        self.pitch_action_idx = 1

        self.listener = keyboard.Listener(on_press=self.get_command)
        self.listener.start()
        self.recieved_command = False
        print('Initialized: manual controller ')
  
    def get_command(self, key):
        try:
            if key.char == 'a': # yaw left
                self.yaw_action_idx = 2
            elif key.char == 'd': # yaw right
                self.yaw_action_idx = 0
            else:
                self.yaw_action_idx = 1 #straight
        except AttributeError:
            if key == keyboard.Key.space: # save image
                #save erm
                torch.save({
                    'memory': self.erm
                }, './expert_erm.pt')
                print('Saved ERM.')
                input()
                return
        return 

        

    def imu_callback(self, imu):
        self.measured_roll_angle = self.calculate_roll(imu)
        return

    def calculate_roll(self, imu):
        return imu.roll
    
    def depth_callback(self, depth):
        self.relative_depth = self.target_depth + depth.data
        return

    def segmentation_callback(self, seg_map):

        #exit if depth has not been measured
        if self.relative_depth is None:
            return 
        
        seg_map = np.array(seg_map.data).reshape(self.img_size)
               
        self.depth_history.append(self.relative_depth)
        self.image_history.append(seg_map)
        if len(self.image_history) == self.history_size:
            ns = np.array(self.image_history)
            nsd = np.array(self.depth_history)
                                  
            self.next_state = torch.tensor(ns, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.next_state_depths = torch.tensor(nsd, dtype=torch.float32, device=self.device).unsqueeze(0)
            reward = reward_calculation((np.sum(ns, axis=0) > 2).astype(int), np.mean(self.relative_depth), self.template)

            self.episode_rewards.append(reward)
            self.reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
            print(np.sum(self.episode_rewards))
            if self.state is not None and self.action is not None and self.state_depths is not None:
                self.erm.push(self.state, self.state_depths, self.action, self.next_state, self.next_state_depths, self.reward)
            
            self.pitch_action_idx = self.discretize(self.pitch_pid.control(self.relative_depth), self.pitch_actions)
            a = int(self.pitch_action_idx*self.yaw_action_space) + self.yaw_action_idx
            a = torch.tensor([[a]], device=self.device, dtype=torch.long)
            self.action = a    
            self.state = self.next_state
            self.state_depths = self.next_state_depths

            self.image_history = []
            self.depth_history = []
        
        if self.action is not None:
            action_idx = self.action.detach().cpu().numpy()[0][0]
            self.command.pitch = self.pitch_actions[int(action_idx/self.yaw_action_space)]
            self.command.yaw = self.yaw_actions[action_idx % self.yaw_action_space]            
            self.command.speed = hyperparams.speed_ #fixed speed
            self.command.roll = self.roll_pid.control(self.measured_roll_angle)
            self.command_publisher.publish(self.command)
        return 
    
    def discretize(self, v, l):
        index = np.argmin(np.abs(np.subtract(l,v)))
        return index


def main(args=None):
    rclpy.init(args=args)

    node = manual_controller()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

    



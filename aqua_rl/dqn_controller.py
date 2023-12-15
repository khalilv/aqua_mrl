import rclpy
import torch
import numpy as np 
import os
from rclpy.node import Node
from aqua2_interfaces.msg import Command, AquaPose
from ir_aquasim_interfaces.srv import SetPosition
from geometry_msgs.msg import Pose
from std_msgs.msg import UInt8MultiArray
from time import sleep
from aqua_rl.control.PID import AnglePID
from aqua_rl.control.DQN import DQN
from aqua_rl.helpers import define_template, reward_calculation
from aqua_rl import hyperparams
from pynput.keyboard import Key, Controller
import subprocess
import time
from multiprocessing import Process


class dqn_controller(Node):
    def __init__(self):
        super().__init__('dqn_controller')

        #hyperparams
        self.queue_size = hyperparams.queue_size_
        self.roll_gains = hyperparams.roll_gains_
        self.history_size = hyperparams.history_size_
        self.pitch_limit = hyperparams.pitch_limit_
        self.yaw_limit = hyperparams.yaw_limit_
        self.yaw_action_space = hyperparams.yaw_action_space_
        self.pitch_action_space = hyperparams.pitch_action_space_
        self.img_size = hyperparams.img_size_
        self.empty_state_max = hyperparams.empty_state_max_
        self.checkpoint_experiment = hyperparams.checkpoint_experiment_
        self.depth_range = hyperparams.depth_range_
        self.target_depth = hyperparams.target_depth_
        self.template_width = hyperparams.template_width_
        self.finish_line_x = hyperparams.finish_line_
        self.start_line_x = hyperparams.starting_line_
        self.alpha = hyperparams.alpha_
        self.beta = hyperparams.beta_

        #number of training episodes 
        self.train_for = 2
        
        #subscribers and publishers
        self.command_publisher = self.create_publisher(Command, '/a13/command', self.queue_size)
        self.imu_subscriber = self.create_subscription(AquaPose, '/aqua/pose', self.imu_callback, self.queue_size)
        self.segmentation_subscriber = self.create_subscription(
            UInt8MultiArray, 
            '/segmentation', 
            self.segmentation_callback, 
            self.queue_size)
        
        #flush queues
        self.flush_steps = self.queue_size + 30
        self.flush_commands = self.flush_steps
        self.zero_command_steps = int(self.flush_commands / 5)
        self.zero_commands = 0
        self.flush_imu = 0
        self.flush_segmentation = 0

        #finished is episode has ended. complete is if aqua has reached the goal 
        self.finished = False
        self.complete = False

        self.roll_pid = AnglePID(target = 0.0, gains = self.roll_gains, reverse=True)
        self.measured_roll_angle = 0.0

        self.relative_depth = None

        #dqn controller for yaw and pitch 
        self.yaw_actions = np.linspace(-self.yaw_limit, self.yaw_limit, self.yaw_action_space)
        self.pitch_actions = np.linspace(-self.pitch_limit, self.pitch_limit, self.pitch_action_space)
        self.dqn = DQN(int(self.yaw_action_space * self.pitch_action_space), self.history_size) 
        self.state = None
        self.next_state = None
        self.state_info = None
        self.next_state_info = None
        self.action = None
        self.reward = None
        self.image_history = []
        self.action_history = []
        self.depth_history = []
        self.episode_rewards = []

        #trajectory recording
        self.trajectory = []
        self.save_every = 3
        self.evaluate = False 

        #target for reward
        self.template = define_template(self.img_size, self.template_width)
        
        #stopping condition for empty vision input
        self.empty_state_counter = 0

        self.root_path = 'src/aqua_rl/checkpoints/dqn/'
        try:
            self.save_path = os.path.join(self.root_path, str(self.checkpoint_experiment))
            self.save_memory_path = os.path.join(self.save_path, 'erm')
            self.traj_save_path = os.path.join(self.root_path.replace('checkpoints', 'trajectories'), str(self.checkpoint_experiment))
            eps = len(os.listdir(self.traj_save_path)) - 1
            last_checkpoint_ep = (eps // self.save_every) * self.save_every
            checkpoint_path = self.save_path + '/episode_' + str(last_checkpoint_ep).zfill(5) + '.pt'
            checkpoint = torch.load(checkpoint_path, map_location=self.dqn.device)
            self.dqn.policy_net.load_state_dict(checkpoint['model_state_dict_policy'], strict=True)
            self.dqn.target_net.load_state_dict(checkpoint['model_state_dict_target'], strict=True)
            self.dqn.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            memory = torch.load(self.save_memory_path + '/memory.pt', map_location=self.dqn.device)
            self.dqn.memory.memory = memory['memory']
            print('ERM size: ', self.dqn.memory.__len__())
            self.dqn.steps_done = checkpoint['training_steps']
            self.episode = last_checkpoint_ep + 1
            print('Weights loaded. starting from episode: ', self.episode, ', training steps completed: ', self.dqn.steps_done)
        except:
            print('No checkpoint found. Starting from episode 0')
            self.new_checkpoint_experiment = len(os.listdir(self.root_path))
            os.mkdir(os.path.join(self.root_path, str(self.new_checkpoint_experiment)))
            os.mkdir(os.path.join(self.root_path.replace('checkpoints', 'trajectories'), str(self.new_checkpoint_experiment)))
            self.traj_save_path = os.path.join(self.root_path.replace('checkpoints', 'trajectories'), str(self.new_checkpoint_experiment))
            self.save_path = os.path.join(self.root_path, str(self.new_checkpoint_experiment))
            os.mkdir(os.path.join(self.save_path, 'erm'))
            self.save_memory_path = os.path.join(self.save_path, 'erm')
            self.episode = 0
        
        #set stopping point
        self.stop_episode = self.episode + self.train_for

        #initialize command
        self.command = Command()
        self.command.speed = 0.0 
        self.command.roll = 0.0
        self.command.pitch = 0.0
        self.command.yaw = 0.0
        self.command.heave = 0.0

        #reset command
        self.reset_client = self.create_client(SetPosition, '/simulator/set_position')
        self.reset_req = SetPosition.Request()
        print('Initialized: dqn controller')

    def imu_callback(self, imu):
        
        #finished flag
        if self.finished:
            return
        
        #flush queue
        if self.flush_imu < self.flush_steps:
            self.flush_imu += 1
            return
        
        self.measured_roll_angle = self.calculate_roll(imu)
        self.relative_depth = self.calculate_relative_depth(imu)

        if imu.x > self.finish_line_x:
            self.flush_commands = 0
            self.finished = True
            self.complete = True
        if imu.x < self.start_line_x:
            print('Drifted behind starting position')
            self.flush_commands = 0
            self.finished = True
            self.complete = False
        elif imu.y < self.depth_range[1]:
            print('Drifted close to seabed')
            self.flush_commands = 0
            self.finished = True
            self.complete = False
        elif imu.y > self.depth_range[0]:
            print('Drifted far above target')
            self.flush_commands = 0
            self.finished = True
            self.complete = False
        else:
            self.trajectory.append([imu.x, imu.y, imu.z])
        return
    
    def calculate_roll(self, imu):
        return imu.roll
    
    def calculate_relative_depth(self, imu):
        return imu.y - self.target_depth

    def segmentation_callback(self, seg_map):

        #exit if depth has not been measured
        if self.relative_depth is None:
            return 
        
        #flush image queue
        if self.flush_segmentation < self.flush_steps:
            self.flush_segmentation += 1
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
                self.zero_commands += 1
            self.flush_commands += 1
            return
        
        #if finished, reset simulation
        if self.finished:
            self.finish()
            return

        seg_map = np.array(seg_map.data).reshape(self.img_size)
        if len(self.image_history) < self.history_size:
            self.depth_history.append(self.relative_depth)
            self.image_history.append(seg_map)
            self.action_history.append([0.0,0.0])
        else:
            self.image_history.pop(0)
            self.image_history.append(seg_map)
            self.depth_history.pop(0)
            self.depth_history.append(self.relative_depth)
            ns = np.array(self.image_history)
            nsi = np.hstack((np.array(self.action_history), np.expand_dims(np.array(self.depth_history), axis= 1))).flatten()

            #check for empty input from vision module
            if ns.sum() == 0:
                self.empty_state_counter += 1
            else:
                self.empty_state_counter = 0
            
            #if nothing has been detected in empty_state_max frames then reset
            if self.empty_state_counter >= self.empty_state_max:
                print("Nothing detected in state space for {} states".format(str(self.empty_state_max)))
                self.flush_commands = 0
                self.finished = True
                self.complete = False
                return
            
            self.next_state = torch.tensor(ns, dtype=torch.float32, device=self.dqn.device).unsqueeze(0)
            self.next_state_info = torch.tensor(nsi, dtype=torch.float32, device=self.dqn.device).unsqueeze(0)
            reward = reward_calculation(seg_map, self.relative_depth, self.template, self.alpha, self.beta)
            self.episode_rewards.append(reward)
            self.reward = torch.tensor([reward], device=self.dqn.device)

            if self.evaluate:
                #select greedy action, dont optimize model or append to replay buffer
                self.action = self.dqn.select_eval_action(self.next_state, self.next_state_info)
            else:
                if self.state is not None and self.action is not None and self.state_info is not None:
                    self.dqn.memory.push(self.state, self.state_info, self.action, self.next_state, self.next_state_info, self.reward)

                self.action = self.dqn.select_action(self.next_state, self.next_state_info)       
                self.state = self.next_state
                self.state_info = self.next_state_info

                # Perform one step of the optimization (on the policy network)
                self.dqn.optimize()
                
                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.dqn.target_net.state_dict()
                policy_net_state_dict = self.dqn.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.dqn.TAU + target_net_state_dict[key]*(1-self.dqn.TAU)
                self.dqn.target_net.load_state_dict(target_net_state_dict)
            
            action_idx = self.action.detach().cpu().numpy()[0][0]
            self.command.pitch = self.pitch_actions[int(action_idx/self.yaw_action_space)]
            self.command.yaw = self.yaw_actions[action_idx % self.yaw_action_space]
            self.action_history.pop(0)
            self.action_history.append([self.command.pitch, self.command.yaw])
            
            self.command.speed = hyperparams.speed_ #fixed speed
            self.command.roll = self.roll_pid.control(self.measured_roll_angle)
            self.command_publisher.publish(self.command)
        return 
        
    def finish(self):
        if self.complete:
            print('Goal reached')
            reward = hyperparams.goal_reached_reward_
            self.episode_rewards.append(reward)
        else:
            reward = hyperparams.goal_not_reached_reward_
            self.episode_rewards.append(reward)

        self.episode_rewards = np.array(self.episode_rewards)
        print('Episode rewards. Average: ', np.mean(self.episode_rewards), ' Sum: ', np.sum(self.episode_rewards))

        if self.state is not None and self.state_info is not None and not self.evaluate:
            self.dqn.memory.push(self.state, self.state_info, self.action, None, None, torch.tensor([reward], device=self.dqn.device))
        
        if self.episode % self.save_every == 0:
            print('Saving checkpoint')
            torch.save({
                'training_steps': self.dqn.steps_done,
                'model_state_dict_policy': self.dqn.policy_net.state_dict(),
                'model_state_dict_target': self.dqn.target_net.state_dict(),
                'optimizer_state_dict': self.dqn.optimizer.state_dict(),
            }, self.save_path +  '/episode_{}.pt'.format(str(self.episode).zfill(5)))
            torch.save({
                'memory': self.dqn.memory.memory
            }, self.save_memory_path +  '/memory.pt')
        
        with open(self.traj_save_path + '/episode_{}.npy'.format(str(self.episode).zfill(5)), 'wb') as f:
            np.save(f, self.episode_rewards)
            np.save(f, np.array(self.trajectory))
        if self.episode < self.stop_episode:
            self.reset()
        else:
            subprocess.run('python3 ./src/aqua_rl/aqua_rl/resetter.py', shell=True)
        return

    def reset(self):
        print('-------------- Resetting simulation --------------')
        
        starting_pose = Pose()

        #starting position
        starting_pose.position.x = 70.0
        starting_pose.position.z = -0.3                                             
        starting_pose.position.y = -12.0
        
        #starting orientation
        starting_pose.orientation.x = 0.0
        starting_pose.orientation.y = -0.7071068
        starting_pose.orientation.z = 0.0
        starting_pose.orientation.w = 0.7071068
        self.reset_req.pose = starting_pose
        self.reset_client.call_async(self.reset_req)
        sleep(0.5)

        #reset pid controllers
        self.measured_roll_angle = 0.0
        self.relative_depth = None
        self.roll_pid = AnglePID(target = 0.0, gains = self.roll_gains, reverse=True)

        #increment episode and reset rewards
        self.episode_rewards = []
        self.episode += 1
        self.evaluate = self.episode % self.save_every == 0

        if self.evaluate:
            print('Starting evaluation')

        #reset trajectory
        self.trajectory = []
        
        #reset state and history queue
        self.state = None
        self.state_info = None
        self.action = None
        self.image_history = []
        self.action_history = []
        self.depth_history = []

        #reset flush queues 
        self.flush_commands = self.flush_steps
        self.zero_commands = 0
        self.flush_imu = 0
        self.flush_segmentation = 0

        #reset end conditions 
        self.finished = False
        self.complete = False
        
        return
    
def main(args=None):
    rclpy.init(args=args)

    node = dqn_controller()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
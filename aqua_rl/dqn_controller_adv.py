import rclpy
import torch
import numpy as np 
import os
import subprocess
from rclpy.node import Node
from aqua2_interfaces.msg import Command, AquaPose, UnderwaterAdversaryCommand
from ir_aquasim_interfaces.srv import SetPosition
from geometry_msgs.msg import Pose
from std_msgs.msg import UInt8MultiArray, Float32
from time import sleep, time
from aqua_rl.control.PID import AnglePID
from aqua_rl.control.DQN import DQN, ReplayMemory
from aqua_rl.helpers import reward_calculation, random_starting_position, adv_mapping
from aqua_rl import hyperparams
from torch.utils.tensorboard import SummaryWriter 

class dqn_controller_adv(Node):
    def __init__(self):
        super().__init__('dqn_controller_adv')

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
        self.depth_range = hyperparams.depth_range_
        self.target_depth = hyperparams.target_depth_
        self.finish_line_x = hyperparams.finish_line_
        self.start_line_x = hyperparams.starting_line_
        self.load_erm = hyperparams.load_erm_ 
        self.experiment_number = hyperparams.experiment_number_
        self.train_for = hyperparams.train_for_
        self.detection_threshold = hyperparams.detection_threshold_
        self.dirl_weights = hyperparams.dirl_weights_
        self.max_duration = hyperparams.max_duration_
        self.frames_to_skip = hyperparams.frames_to_skip_
        self.roi_detection_threshold = hyperparams.roi_detection_threshold_
        self.mean_importance = hyperparams.mean_importance_
        self.adv_action_space = hyperparams.adv_action_space_
        self.adv_madnitude_x = hyperparams.adv_magnitude_x_
        self.adv_madnitude_y = hyperparams.adv_magnitude_y_
        self.adv_madnitude_z = hyperparams.adv_magnitude_z_

        #subscribers and publishers
        self.command_publisher = self.create_publisher(Command, '/a13/command', self.queue_size)
        self.imu_subscriber = self.create_subscription(AquaPose, '/aqua/pose', self.imu_callback, self.queue_size)
        self.segmentation_subscriber = self.create_subscription(
            UInt8MultiArray, 
            '/segmentation', 
            self.segmentation_callback, 
            self.queue_size)
        self.depth_subscriber = self.create_subscription(Float32, '/aqua/depth', self.depth_callback, self.queue_size)
        self.adv_command_publisher = self.create_publisher(UnderwaterAdversaryCommand, 'adv_command', self.queue_size)

        #flush queues
        self.flush_steps = self.queue_size + 30
        self.flush_commands = self.flush_steps
        self.zero_command_steps = int(self.flush_commands / 5)
        self.zero_commands = 0
        self.flush_imu = 0
        self.flush_segmentation = 0
        self.flush_depth = 0

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
        self.state_depths = None
        self.next_state_depths = None
        self.state_actions = None
        self.next_state_actions = None
        self.action = torch.tensor([[4]], device=self.dqn.device, dtype=torch.long)
        self.reward = None
        self.image_history = []
        self.depth_history = []
        self.action_history = [4]
        self.episode_rewards = []

        #adversary
        self.dqn_adv = DQN(self.adv_action_space, self.history_size)
        self.adv_action = torch.tensor([[0]], device=self.dqn_adv.device, dtype=torch.long)
        self.erm = ReplayMemory(self.dqn_adv.MEMORY_SIZE)

        #trajectory recording
        self.trajectory = []
        self.evaluate = False 

        #stopping conditions
        self.empty_state_counter = 0

        self.root_path = 'src/aqua_rl/experiments/{}'.format(str(self.experiment_number))
        if os.path.exists(self.root_path):
            self.save_path = os.path.join(self.root_path, 'weights/adv')
            self.save_memory_path = os.path.join(self.root_path, 'erm/adv')
            self.save_traj_path = os.path.join(self.root_path, 'trajectories/adv')
            self.writer = SummaryWriter(os.path.join(self.root_path, 'logs/adv'))
            last_checkpoint = max(sorted(os.listdir(self.save_path)))
            checkpoint = torch.load(os.path.join(self.save_path, last_checkpoint), map_location=self.dqn_adv.device)
            self.dqn_adv.policy_net.load_state_dict(checkpoint['model_state_dict_policy'], strict=True)
            self.dqn_adv.target_net.load_state_dict(checkpoint['model_state_dict_target'], strict=True)
            self.dqn_adv.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.dqn_adv.steps_done = checkpoint['training_steps']
            
            self.pro_save_path = os.path.join(self.root_path, 'weights')
            pro_last_checkpoint = max(sorted(os.listdir(self.pro_save_path)))
            pro_checkpoint = torch.load(os.path.join(self.pro_save_path, pro_last_checkpoint), map_location=self.dqn.device)
            self.dqn.policy_net.load_state_dict(pro_checkpoint['model_state_dict_policy'], strict=True)
            self.dqn.target_net.load_state_dict(pro_checkpoint['model_state_dict_target'], strict=True)
            self.dqn.optimizer.load_state_dict(pro_checkpoint['optimizer_state_dict'])
            self.dqn.steps_done = pro_checkpoint['training_steps']
            
            if self.load_erm:
                print('Loading ERM from previous experience. Note this may take time')
                t0 = time()
                for file_path in sorted(os.listdir(self.save_memory_path), reverse=True):
                    if os.path.isfile(os.path.join(self.save_memory_path, file_path)):
                        if self.dqn_adv.memory.__len__() < self.dqn_adv.MEMORY_SIZE:
                            memory = torch.load(os.path.join(self.save_memory_path, file_path), map_location=self.dqn_adv.device)
                            erm = memory['memory']
                            self.dqn_adv.memory.memory += erm.memory
                t1 = time()
                print('ERM size: ', self.dqn_adv.memory.__len__(), '. Time taken to load: ', t1 - t0)
            else:
                print('WARNING: weights loaded but starting from a fresh replay memory')
            
            self.episode = int(last_checkpoint[8:13]) + 1
            self.stop_episode = self.episode + self.train_for - 1
            print('Weights loaded. starting from episode: ', self.episode, ', training steps completed: ', self.dqn_adv.steps_done)
        else:
            raise Exception('ERROR: Experiment {} does not exist. Please train protagonist agent first.'.format(str(self.experiment_number)))
        
        #initialize command
        self.command = Command()
        self.command.speed = 0.0 
        self.command.roll = 0.0
        self.command.pitch = 0.0
        self.command.yaw = 0.0
        self.command.heave = 0.0
        self.adv_command = UnderwaterAdversaryCommand()

        #duration counting
        self.duration = 0

        #popen_called
        self.popen_called = False

        #reset command
        self.reset_client = self.create_client(SetPosition, '/simulator/set_position')
        self.reset_req = SetPosition.Request()
        print('Initialized: dqn controller adv')

    def imu_callback(self, imu):
        
        #finished flag
        if self.finished:
            return
        
        #flush queue
        if self.flush_imu < self.flush_steps:
            self.flush_imu += 1
            return
        
        self.measured_roll_angle = self.calculate_roll(imu)

        if imu.x > self.finish_line_x:
            print('Reached finish line')
            self.flush_commands = 0
            self.finished = True
            self.complete = True
        if imu.x < self.start_line_x:
            print('Tracked backwards to starting line')
            self.flush_commands = 0
            self.finished = True
            self.complete = True
        else:
            self.trajectory.append([imu.x, imu.y, imu.z])
        return
    
    def calculate_roll(self, imu):
        return imu.roll
    
    def depth_callback(self, depth):
        
        #finished flag
        if self.finished:
            return
        
        #flush queue
        if self.flush_depth < self.flush_steps:
            self.flush_depth += 1
            return
        
        if -depth.data < self.depth_range[1]:
            print('Drifted close to seabed')
            self.flush_commands = 0
            self.finished = True
            self.complete = False
        elif -depth.data > self.depth_range[0]:
            print('Drifted far above target')
            self.flush_commands = 0
            self.finished = True
            self.complete = False
        else:
            self.relative_depth = self.target_depth + depth.data
        return

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
                #reset adv
                self.adv_command.current_x = 0.0
                self.adv_command.current_z = 0.0
                self.adv_command.current_y = 0.0
                self.adv_command_publisher.publish(self.adv_command)
                self.zero_commands += 1
            self.flush_commands += 1
            return
        
        #if finished, reset simulation
        if self.finished:
            self.finish()
            return

        seg_map = np.array(seg_map.data).reshape(self.img_size)
        #check for empty input from vision module
        if seg_map.sum() < self.detection_threshold:
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
        
        if self.duration > self.max_duration:
            print("Duration Reached")
            self.flush_commands = 0
            self.finished = True
            self.complete = True
            return
        self.duration += 1
        
        self.depth_history.append(self.relative_depth)
        self.image_history.append(seg_map)
        if len(self.image_history) == self.history_size and len(self.depth_history) == self.history_size and len(self.action_history) == self.history_size:
            ns = np.array(self.image_history)
            nsd = np.array(self.depth_history)
            nsa = np.array(self.action_history)
  
            self.next_state = torch.tensor(ns, dtype=torch.float32, device=self.dqn_adv.device).unsqueeze(0)
            self.next_state_depths = torch.tensor(nsd, dtype=torch.float32, device=self.dqn_adv.device).unsqueeze(0)
            self.next_state_actions = torch.tensor(nsa, dtype=torch.float32, device=self.dqn_adv.device).unsqueeze(0)

            reward = reward_calculation(seg_map, self.relative_depth, self.roi_detection_threshold, self.mean_importance)

            self.episode_rewards.append(reward)
            self.reward = torch.tensor([reward], dtype=torch.float32, device=self.dqn_adv.device)

            if self.evaluate:
                #select greedy action, dont optimize model or append to replay buffer
                self.adv_action = self.dqn_adv.select_eval_action(self.next_state, self.next_state_depths, self.next_state_actions)
            else:
                if self.state is not None and self.state_depths is not None and self.state_actions is not None:
                    self.dqn_adv.memory.push(self.state, self.state_depths, self.state_actions, self.adv_action, self.next_state, self.next_state_depths, self.next_state_actions, -1 * self.reward)
                    self.erm.push(self.state, self.state_depths, self.state_actions, self.adv_action, self.next_state, self.next_state_depths, self.next_state_actions, -1 * self.reward)

                self.adv_action = self.dqn_adv.select_action(self.next_state, self.next_state_depths, self.next_state_actions)       
                self.state = self.next_state
                self.state_depths = self.next_state_depths
                self.state_actions = self.next_state_actions

            self.action = self.dqn.select_eval_action(self.next_state, self.next_state_depths, self.next_state_actions)
            self.image_history = self.image_history[self.frames_to_skip:]
            self.depth_history = self.depth_history[self.frames_to_skip:]
            self.action_history = self.action_history[self.frames_to_skip:]

        if not self.evaluate:
            # Perform one step of the optimization (on the policy network)
            if self.dqn_adv.steps_done % 1 == 0:
                loss = self.dqn_adv.optimize()
                if loss is not None:        
                    self.writer.add_scalar('Loss', loss, self.dqn_adv.steps_done)
                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.dqn_adv.target_net.state_dict()
                policy_net_state_dict = self.dqn_adv.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.dqn_adv.TAU + target_net_state_dict[key]*(1-self.dqn_adv.TAU)
                self.dqn_adv.target_net.load_state_dict(target_net_state_dict)

        #adversary action
        x,y,z = adv_mapping(self.adv_action.detach().cpu().numpy()[0][0])
        self.adv_command.current_x = self.adv_madnitude_x * x
        self.adv_command.current_z = self.adv_madnitude_z * z
        self.adv_command.current_y = self.adv_madnitude_y * y 
        self.adv_command_publisher.publish(self.adv_command)
        
        #protagonist action
        action_idx = self.action.detach().cpu().numpy()[0][0]
        self.action_history.append(action_idx)
        self.command.pitch = self.pitch_actions[int(action_idx/self.yaw_action_space)]
        self.command.yaw = self.yaw_actions[action_idx % self.yaw_action_space]            
        self.command.speed = hyperparams.speed_ #fixed speed
        self.command.roll = self.roll_pid.control(self.measured_roll_angle)
        self.command_publisher.publish(self.command)
        return 
        
    def finish(self):

        if self.popen_called:
            return 
          
        if self.complete:
            reward = hyperparams.goal_reached_reward_
            self.episode_rewards.append(reward)
        else:
            reward = hyperparams.goal_not_reached_reward_
            self.episode_rewards.append(reward)
        
        self.episode_rewards = np.array(self.episode_rewards)
        print('Episode rewards. Average: ', np.mean(self.episode_rewards), ' Sum: ', np.sum(self.episode_rewards))
        self.reward = torch.tensor([reward], dtype=torch.float32, device=self.dqn_adv.device)
        
        if self.evaluate:
            self.writer.add_scalar('Episode Rewards (Eval)', np.sum(self.episode_rewards), self.episode)
        else:
            self.writer.add_scalar('Episode Rewards (Train)', np.sum(self.episode_rewards), self.episode)

        self.writer.add_scalar('Duration', self.duration, self.episode)

        if self.state is not None and self.state_depths is not None and self.state_actions is not None and not self.evaluate and not self.complete:
            self.dqn_adv.memory.push(self.state, self.state_depths, self.state_actions, self.adv_action, None, None, None, -1 * self.reward)
            self.erm.push(self.state, self.state_depths, self.state_actions, self.adv_action, None, None, None, -1 * self.reward)

        if self.episode == self.stop_episode:
            print('Saving checkpoint')
            torch.save({
                'training_steps': self.dqn_adv.steps_done,
                'model_state_dict_policy': self.dqn_adv.policy_net.state_dict(),
                'model_state_dict_target': self.dqn_adv.target_net.state_dict(),
                'optimizer_state_dict': self.dqn_adv.optimizer.state_dict(),
            }, self.save_path +  '/episode_{}.pt'.format(str(self.episode).zfill(5)))
            torch.save({
                'memory': self.erm
            }, self.save_memory_path +  '/episode_{}.pt'.format(str(self.episode).zfill(5)))
        
        with open(self.save_traj_path + '/episode_{}.npy'.format(str(self.episode).zfill(5)), 'wb') as f:
            np.save(f, self.episode_rewards)
            np.save(f, np.array(self.trajectory))
        
        if self.episode < self.stop_episode:
            self.reset()
        else:
            subprocess.Popen('python3 ./src/aqua_rl/aqua_rl/resetter_adv.py', shell=True)
            self.popen_called = True
        return

    def reset(self):
        print('-------------- Resetting simulation --------------')
        
        #increment episode and reset rewards
        self.episode_rewards = []
        self.episode += 1
        self.evaluate = self.episode == self.stop_episode

        if self.evaluate:
            print('Starting evaluation')

        starting_pose = Pose()

        #starting position
        if not self.evaluate:
            random_position = random_starting_position() 
            starting_pose.position.x = random_position[0]
            starting_pose.position.z = random_position[1]                               
            starting_pose.position.y = np.random.uniform(self.target_depth-1, self.target_depth+1)
        else:
            starting_pose.position.x = 70.0
            starting_pose.position.z = -0.3                               
            starting_pose.position.y = self.target_depth
        
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

        #reset trajectory
        self.trajectory = []
        
        #reset state and history queues
        self.state = None
        self.next_state = None
        self.state_depths = None
        self.next_state_depths = None
        self.state_actions = None
        self.next_state_actions = None
        self.action = torch.tensor([[4]], device=self.dqn.device, dtype=torch.long)
        self.adv_action = torch.tensor([[0]], device=self.dqn_adv.device, dtype=torch.long)
        self.reward = None
        self.image_history = []
        self.depth_history = []
        self.action_history = [4]

        #reset flush queues 
        self.flush_commands = self.flush_steps
        self.zero_commands = 0
        self.flush_imu = 0
        self.flush_segmentation = 0
        self.flush_depth = 0

        #reset counters
        self.empty_state_counter = 0
        
        #reset end conditions 
        self.finished = False
        self.complete = False

        self.duration = 0
        
        return
    
def main(args=None):
    rclpy.init(args=args)

    node = dqn_controller_adv()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
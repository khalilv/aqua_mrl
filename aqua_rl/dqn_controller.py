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
from aqua_rl.control.PID import AnglePID
from aqua_rl.control.DQN import DQN, ReplayMemory
from aqua_rl.helpers import reward_calculation
from aqua_rl import hyperparams
from torch.utils.tensorboard import SummaryWriter 

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
        self.depth_range = hyperparams.depth_range_
        self.load_erm = hyperparams.load_erm_ 
        self.experiment_number = hyperparams.experiment_number_
        self.train_for = hyperparams.train_for_
        self.train_duration = hyperparams.train_duration_
        self.eval_duration = hyperparams.eval_duration_
        self.depth_range = hyperparams.depth_range_
        self.diver_max_speed = hyperparams.diver_max_speed_

        # self.switch_every = hyperparams.switch_every_
        # self.adv_action_space = hyperparams.adv_action_space_
        # self.adv_madnitude_x = hyperparams.adv_magnitude_x_
        # self.adv_madnitude_y = hyperparams.adv_magnitude_y_
        # self.adv_madnitude_z = hyperparams.adv_magnitude_z_

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
        # self.adv_command_publisher = self.create_publisher(UnderwaterAdversaryCommand, 'adv_command', self.queue_size)

        #flush queues
        self.flush_steps = self.queue_size + 30
        self.flush_commands = self.flush_steps
        self.zero_command_steps = int(self.flush_commands / 5)
        self.zero_commands = 0
        self.flush_imu = 0
        self.flush_diver = 0
        self.flush_detection = 0

        #finished is episode has ended. complete is if aqua has reached the goal 
        self.finished = False
        self.complete = False

        self.roll_pid = AnglePID(target = 0.0, gains = self.roll_gains, reverse=True)
        self.measured_roll_angle = 0.0

        #dqn controller for yaw and pitch 
        self.yaw_actions = np.linspace(-self.yaw_limit, self.yaw_limit, self.yaw_action_space)
        self.pitch_actions = np.linspace(-self.pitch_limit, self.pitch_limit, self.pitch_action_space)
        self.dqn = DQN(self.pitch_action_space, self.yaw_action_space, self.history_size) 
        self.state = None
        self.next_state = None
        self.pitch_action = torch.tensor([[2]], device=self.dqn.device, dtype=torch.long)
        self.yaw_action = torch.tensor([[2]], device=self.dqn.device, dtype=torch.long)
        self.pitch_reward = None
        self.yaw_reward = None
        self.history = []
        self.episode_rewards = []
        self.erm = ReplayMemory(self.dqn.MEMORY_SIZE)

        #adversary
        # self.dqn_adv = DQN(self.adv_action_space, self.history_size)
        # self.adv_action = torch.tensor([[0]], device=self.dqn_adv.device, dtype=torch.long)

        #trajectory recording
        self.aqua_trajectory = []
        self.diver_trajectory = []
        self.evaluate = False 

        #stopping conditions
        self.empty_state_counter = 0

        self.root_path = 'src/aqua_rl/experiments/{}'.format(str(self.experiment_number))
        if os.path.exists(self.root_path):
            self.save_path = os.path.join(self.root_path, 'weights')
            self.save_memory_path = os.path.join(self.root_path, 'erm')
            self.save_traj_path = os.path.join(self.root_path, 'trajectories')
            self.writer = SummaryWriter(os.path.join(self.root_path, 'logs'))
            last_checkpoint = max(sorted(os.listdir(self.save_path)))
            checkpoint = torch.load(os.path.join(self.save_path, last_checkpoint), map_location=self.dqn.device)
            self.dqn.policy_net.load_state_dict(checkpoint['model_state_dict_policy'], strict=True)
            self.dqn.target_net.load_state_dict(checkpoint['model_state_dict_target'], strict=True)
            self.dqn.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.dqn.steps_done = checkpoint['training_steps']
            
            # self.adv_save_path = os.path.join(self.root_path, 'weights/adv')
            # adv_last_checkpoint = max(sorted(os.listdir(self.adv_save_path)))
            # adv_checkpoint = torch.load(os.path.join(self.adv_save_path, adv_last_checkpoint), map_location=self.dqn_adv.device)
            # self.dqn_adv.policy_net.load_state_dict(adv_checkpoint['model_state_dict_policy'], strict=True)
            # self.dqn_adv.target_net.load_state_dict(adv_checkpoint['model_state_dict_target'], strict=True)
            # self.dqn_adv.optimizer.load_state_dict(adv_checkpoint['optimizer_state_dict'])
            # self.dqn_adv.steps_done = adv_checkpoint['training_steps']
            # print('Adversary loaded from: ', adv_last_checkpoint)

            if self.load_erm:
                print('Loading ERM from previous experience. Note this may take time')
                t0 = time()
                for file_path in sorted(os.listdir(self.save_memory_path), reverse=True):
                    if os.path.isfile(os.path.join(self.save_memory_path, file_path)):
                        if self.dqn.memory.__len__() < self.dqn.MEMORY_SIZE:
                            memory = torch.load(os.path.join(self.save_memory_path, file_path), map_location=self.dqn.device)
                            erm = memory['memory']
                            self.dqn.memory.memory += erm.memory
                t1 = time()
                print('ERM size: ', self.dqn.memory.__len__(), '. Time taken to load: ', t1 - t0)
            else:
                print('WARNING: weights loaded but starting from a fresh replay memory')
            
            self.episode = int(last_checkpoint[8:13]) + 1
            self.stop_episode = self.episode + self.train_for - 1
            print('Weights loaded. starting from episode: ', self.episode, ', training steps completed: ', self.dqn.steps_done)
        else:
            print('WARNING: starting a new experiment as experiment {} does not exist'.format(str(self.experiment_number)))
            os.mkdir(self.root_path)
            os.mkdir(os.path.join(self.root_path, 'weights'))
            os.mkdir(os.path.join(self.root_path, 'erm'))
            os.mkdir(os.path.join(self.root_path, 'trajectories'))
            os.mkdir(os.path.join(self.root_path, 'logs'))
            # os.mkdir(os.path.join(self.root_path, 'weights/adv'))
            # os.mkdir(os.path.join(self.root_path, 'erm/adv'))
            # os.mkdir(os.path.join(self.root_path, 'trajectories/adv'))
            # os.mkdir(os.path.join(self.root_path, 'logs/adv'))
            self.save_path = os.path.join(self.root_path, 'weights')
            self.save_memory_path = os.path.join(self.root_path, 'erm')
            self.save_traj_path = os.path.join(self.root_path, 'trajectories')
            self.writer = SummaryWriter(os.path.join(self.root_path, 'logs'))
            self.episode = 0
            self.stop_episode = self.episode + self.train_for
            # torch.save({
            #     'training_steps': self.dqn_adv.steps_done,
            #     'model_state_dict_policy': self.dqn_adv.policy_net.state_dict(),
            #     'model_state_dict_target': self.dqn_adv.target_net.state_dict(),
            #     'optimizer_state_dict': self.dqn_adv.optimizer.state_dict(),
            # }, self.save_path +  '/adv/episode_{}.pt'.format(str(self.episode).zfill(5)))
            print('New experiment {} started. Starting from episode 0'.format(str(self.experiment_number)))
        
        #initialize command
        self.command = Command()
        self.command.speed = 0.0 
        self.command.roll = 0.0
        self.command.pitch = 0.0
        self.command.yaw = 0.0
        self.command.heave = 0.0
        # self.adv_command = UnderwaterAdversaryCommand()

        #command
        self.diver_cmd = DiverCommand()
        self.diver_pose = None

        #duration counting
        self.duration = 0

        #popen_called
        self.popen_called = False

        #reset command
        self.reset_client = self.create_client(SetPosition, '/simulator/set_position')
        self.reset_req = SetPosition.Request()

        self.reset_diver_client = self.create_client(SetPosition, '/diver/set_position')
        self.reset_diver_req = SetPosition.Request()

        self.timer = self.create_timer(5, self.publish_diver_command)
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

        self.aqua_trajectory.append([imu.x, imu.y, imu.z])
        return
    
    def publish_diver_command(self):     
        if self.diver_pose:

            #scale vector to current magnitude
            self.diver_cmd.vx = 0.35
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
            self.empty_state_counter += 1
            center = [-1, -1]
        else:
            self.empty_state_counter = 0
            center = [(coords[2] + coords[0])/2, (coords[3] + coords[1])/2]

        #if nothing has been detected in empty_state_max frames then reset
        if self.empty_state_counter >= self.empty_state_max:
            print("Nothing detected in state space for {} states".format(str(self.empty_state_max)))
            self.flush_commands = 0
            self.finished = True
            self.complete = False
            return
        
        if self.duration > (self.eval_duration if self.evaluate else self.train_duration):
            print("Duration Reached")
            self.flush_commands = 0
            self.finished = True
            self.complete = True
            return
        self.duration += 1
        
        self.history.append(center)
        if len(self.history) == self.history_size:
            ns = np.array(self.history).flatten()
  
            self.next_state = torch.tensor(ns, dtype=torch.float32, device=self.dqn.device).unsqueeze(0)
           
            pitch_reward, yaw_reward = reward_calculation(center, self.img_size, self.img_size, 0.2, 0.2)
            self.episode_rewards.append([pitch_reward, yaw_reward])
            self.pitch_reward = torch.tensor([pitch_reward], dtype=torch.float32, device=self.dqn.device)
            self.yaw_reward = torch.tensor([yaw_reward], dtype=torch.float32, device=self.dqn.device)

            if self.evaluate:
                #select greedy action, dont optimize model or append to replay buffer
                self.pitch_action, self.yaw_action = self.dqn.select_eval_action(self.next_state)
            else:
                if self.state is not None:
                    self.dqn.memory.push(self.state, self.pitch_action, self.yaw_action, self.next_state, self.pitch_reward, self.yaw_reward)
                    self.erm.push(self.state, self.pitch_action, self.yaw_action, self.next_state, self.pitch_reward, self.yaw_reward)

                self.pitch_action, self.yaw_action = self.dqn.select_action(self.next_state)       
                self.state = self.next_state
               
                #select adversary action
                # self.adv_action = self.dqn_adv.select_eval_action(self.next_state, self.next_state_depths, self.next_state_actions)
                # Perform one step of the optimization (on the policy network)
                if self.dqn.steps_done % 1 == 0:
                    loss = self.dqn.optimize()
                    if loss is not None:        
                        self.writer.add_scalar('Loss', loss, self.dqn.steps_done)
                    # Soft update of the target network's weights
                    # θ′ ← τ θ + (1 −τ )θ′
                    target_net_state_dict = self.dqn.target_net.state_dict()
                    policy_net_state_dict = self.dqn.policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*self.dqn.TAU + target_net_state_dict[key]*(1-self.dqn.TAU)
                    self.dqn.target_net.load_state_dict(target_net_state_dict)
            
            self.history.pop(0)

        #adversary action
        # x,y,z = adv_mapping(self.adv_action.detach().cpu().numpy()[0][0])
        # self.adv_command.current_x = self.adv_madnitude_x * x
        # self.adv_command.current_z = self.adv_madnitude_z * z
        # self.adv_command.current_y = self.adv_madnitude_y * y 
        # self.adv_command_publisher.publish(self.adv_command)
        
        #protagonist action
        pitch_action_idx = self.pitch_action.detach().cpu().numpy()[0][0]
        yaw_action_idx = self.yaw_action.detach().cpu().numpy()[0][0]
        self.command.pitch = self.pitch_actions[pitch_action_idx]
        self.command.yaw = self.yaw_actions[yaw_action_idx]            
        self.command.speed = hyperparams.speed_ #fixed speed
        self.command.roll = self.roll_pid.control(self.measured_roll_angle)
        self.command_publisher.publish(self.command)
        return 
    def finish(self):

        if self.popen_called:
            return 
          
        self.episode_rewards = np.array(self.episode_rewards)
        print('Episode rewards. Average: ', np.mean(self.episode_rewards, axis=0), ' Sum: ', np.sum(self.episode_rewards, axis=0))
        
        if self.evaluate:
            self.writer.add_scalar('Episode Rewards (Eval)', np.sum(self.episode_rewards), self.episode)
        else:
            self.writer.add_scalar('Episode Rewards (Train)', np.sum(self.episode_rewards), self.episode)

        self.writer.add_scalar('Duration', self.duration, self.episode)

        if self.state is not None and not self.evaluate and not self.complete:
            self.dqn.memory.push(self.state, self.pitch_action, self.yaw_action, None, self.pitch_reward, self.yaw_reward)
            self.erm.push(self.state, self.pitch_action, self.yaw_action, None, self.pitch_reward, self.yaw_reward)

        if self.episode == self.stop_episode:
            print('Saving checkpoint')
            torch.save({
                'training_steps': self.dqn.steps_done,
                'model_state_dict_policy': self.dqn.policy_net.state_dict(),
                'model_state_dict_target': self.dqn.target_net.state_dict(),
                'optimizer_state_dict': self.dqn.optimizer.state_dict(),
            }, self.save_path +  '/episode_{}.pt'.format(str(self.episode).zfill(5)))
            torch.save({
                'memory': self.erm
            }, self.save_memory_path +  '/episode_{}.pt'.format(str(self.episode).zfill(5)))
        
        with open(self.save_traj_path + '/episode_{}.npy'.format(str(self.episode).zfill(5)), 'wb') as f:
            np.save(f, self.episode_rewards)
            np.save(f, np.array(self.aqua_trajectory))
            np.save(f, np.array(self.diver_trajectory))

        if self.episode < self.stop_episode:
            self.reset()
        else:
            subprocess.Popen('python3 ./src/aqua_rl/aqua_rl/resetter.py', shell=True)
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
        starting_pose.position.x = 70.0
        starting_pose.position.z = -0.3                               
        starting_pose.position.y = -10.0
        #starting orientation
        starting_pose.orientation.x = 0.0
        starting_pose.orientation.y = -0.7071068
        starting_pose.orientation.z = 0.0
        starting_pose.orientation.w = 0.7071068
        self.reset_req.pose = starting_pose
        self.reset_client.call_async(self.reset_req)

        starting_diver_pose = Pose()
        #starting position
        starting_diver_pose.position.x = 62.5
        starting_diver_pose.position.z = 0.3                               
        starting_diver_pose.position.y = -10.0
        #starting orientation
        starting_diver_pose.orientation.x = 0.4976952
        starting_diver_pose.orientation.y = -0.5022942
        starting_diver_pose.orientation.z = 0.4976952
        starting_diver_pose.orientation.w = 0.5022942
        self.reset_diver_req.pose = starting_diver_pose
        self.reset_diver_client.call_async(self.reset_diver_req)
        sleep(1.0)

        #reset pid controllers
        self.measured_roll_angle = 0.0
        self.roll_pid = AnglePID(target = 0.0, gains = self.roll_gains, reverse=True)

        #reset trajectory
        self.aqua_trajectory = []
        self.diver_trajectory = []

        #reset state and history queues
        self.state = None
        self.next_state = None
        
        self.pitch_action = torch.tensor([[2]], device=self.dqn.device, dtype=torch.long)
        self.yaw_action = torch.tensor([[2]], device=self.dqn.device, dtype=torch.long)
        # self.adv_action = torch.tensor([[0]], device=self.dqn.device, dtype=torch.long)
        self.pitch_reward = None
        self.yaw_reward = None
        self.history = []


        #reset flush queues 
        self.flush_commands = self.flush_steps
        self.zero_commands = 0
        self.flush_imu = 0
        self.flush_detection = 0
        self.flush_diver = 0

        #reset counters
        self.empty_state_counter = 0
        self.duration = 0

        #reset end conditions 
        self.finished = False
        self.complete = False

        #reset diver pose
        self.diver_pose = None

        
        return
    
def main(args=None):
    rclpy.init(args=args)

    node = dqn_controller()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
import rclpy
import torch
import numpy as np 
import os
import subprocess
import shutil
from rclpy.node import Node
from aqua2_interfaces.msg import AquaPose, DiverCommand
from ir_aquasim_interfaces.srv import SetPosition
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32MultiArray, UInt8MultiArray, Bool
from time import sleep, time
from aqua_rl.control.DQN import DQN, ReplayMemory
from aqua_rl.control.PID import PID
from aqua_rl.helpers import reward_calculation, map_missing_detection, normalize_coords, safe_region
from aqua_rl import hyperparams
from torch.utils.tensorboard import SummaryWriter 

class dqn_controller(Node):
    def __init__(self):
        super().__init__('dqn_controller')

        #hyperparams
        self.queue_size = hyperparams.queue_size_
        self.history_size = hyperparams.history_size_
        self.yaw_action_space = hyperparams.yaw_action_space_
        self.pitch_action_space = hyperparams.pitch_action_space_
        self.img_size = hyperparams.img_size_
        self.depth_range = hyperparams.depth_range_
        self.load_erm = hyperparams.load_erm_ 
        self.experiment_number = hyperparams.experiment_number_
        self.train_for = hyperparams.train_for_
        self.train_duration = hyperparams.train_duration_
        self.eval_duration = hyperparams.eval_duration_
        self.diver_max_speed = hyperparams.diver_max_speed_
        self.reward_sharpness = hyperparams.sharpness_
        self.frame_skip = hyperparams.frame_skip_
        self.empty_state_max = hyperparams.empty_state_max_
        self.yaw_gains = hyperparams.yaw_gains_
        self.pitch_gains = hyperparams.pitch_gains_
        self.pid_decay_start = hyperparams.pid_decay_start_
        self.pid_decay_end = hyperparams.pid_decay_end_

        
        #subscribers and publishers
        self.imu_subscriber = self.create_subscription(AquaPose, '/aqua/pose', self.imu_callback, self.queue_size)
        self.command_publisher = self.create_publisher(UInt8MultiArray, hyperparams.autopilot_command_, self.queue_size)
        self.autopilot_start_stop_publisher = self.create_publisher(Bool, hyperparams.autopilot_start_stop_, self.queue_size)
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
        self.pitch_pid = PID(target = 0, gains = self.pitch_gains, command_range=[-(self.pitch_action_space//2), self.pitch_action_space//2], reverse=True)
        self.yaw_pid = PID(target = 0, gains = self.yaw_gains, command_range=[-(self.yaw_action_space//2), self.yaw_action_space//2], reverse=True)

        self.yaw_actions = np.linspace(-(self.yaw_action_space//2), self.yaw_action_space//2, self.yaw_action_space)
        self.pitch_actions = np.linspace(-(self.pitch_action_space//2), self.pitch_action_space//2, self.pitch_action_space)

        #flush queues
        self.flush_steps = self.queue_size + 35
        self.flush_imu = 0
        self.flush_diver = 0
        self.flush_detection = 0

        #finished is episode has ended. complete is if aqua has reached the goal 
        self.finished = False
        self.complete = False

        #dqn controller for yaw and pitch 
        self.dqn = DQN(self.pitch_action_space, self.yaw_action_space, self.history_size) 
        self.state = None
        self.next_state = None
        self.pitch_action = None        
        self.yaw_action = None
        self.pitch_reward = None
        self.yaw_reward = None
        self.history = []
        self.action_history = []
        self.episode_rewards = []
        self.erm = ReplayMemory(self.dqn.MEMORY_SIZE)

        #trajectory recording
        self.aqua_trajectory = []
        self.diver_trajectory = []
        self.evaluate = False 

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
            self.save_path = os.path.join(self.root_path, 'weights')
            self.save_memory_path = os.path.join(self.root_path, 'erm')
            self.save_traj_path = os.path.join(self.root_path, 'trajectories')
            self.writer = SummaryWriter(os.path.join(self.root_path, 'logs'))
            self.episode = 0
            self.stop_episode = self.episode + self.train_for
            
            self.expert_path = 'src/aqua_rl/expert_data/pid/{}_{}'.format(self.history_size, self.frame_skip)
            if os.path.exists(self.expert_path):
                for file_path in os.listdir(self.expert_path):
                    shutil.copyfile(os.path.join(self.expert_path, file_path), os.path.join(self.save_memory_path, file_path))
                print('Loading ERM from expert experience. Note this may take time')
                t0 = time()
                for file_path in sorted(os.listdir(self.save_memory_path), reverse=True):
                    if os.path.isfile(os.path.join(self.save_memory_path, file_path)):
                        if self.dqn.memory.__len__() < self.dqn.MEMORY_SIZE:
                            memory = torch.load(os.path.join(self.save_memory_path, file_path), map_location=self.dqn.device)
                            erm = memory['memory']
                            self.dqn.memory.memory += erm.memory
                t1 = time()
                print('ERM size: ', self.dqn.memory.__len__(), '. Time taken to load: ', t1 - t0)
            
            print('New experiment {} started. Starting from episode 0'.format(str(self.experiment_number)))
        
        #autopilot commands
        self.command = UInt8MultiArray()
        self.autopilot_flag = Bool()
        self.autopilot_flag.data = False

        #diver command
        self.diver_cmd = DiverCommand()
        self.diver_pose = None

        #duration counting
        self.duration = 0
        self.empty_state_counter = 0

        #popen_called
        self.popen_called = False

        #reset commands
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
        
        self.aqua_trajectory.append([imu.x, imu.y, imu.z])
        return
    
    def publish_diver_command(self):     
        if self.diver_pose:

            #scale vector to current magnitude
            self.diver_cmd.vx = np.random.uniform(hyperparams.speed_+0.025, hyperparams.speed_+0.15)
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
    

    def detection_callback(self, coords):
        
        #flush detections queue
        if self.flush_detection< self.flush_steps:
            self.flush_detection += 1
            return
        
        if not self.autopilot_flag.data:
            print('starting autopilot')
            self.autopilot_flag.data = True
            self.autopilot_start_stop_publisher.publish(self.autopilot_flag)

        #if finished, reset simulation
        if self.finished:
            self.finish()
            return
        
        coords = np.array(coords.data)
        
        #check for null input from detection module
        if coords[0] == -1 and coords[1] == -1 and coords[2] == -1 and coords[3] == -1:
            self.empty_state_counter += 1
            detected_center = map_missing_detection(self.history[-1][0],self.history[-1][1])
        else:
            self.empty_state_counter = 0
            yc = (coords[1] + coords[3])/2
            xc = (coords[0] + coords[2])/2
            detected_center = normalize_coords(yc, xc, self.img_size, self.img_size)
        
        if self.empty_state_counter > self.empty_state_max:
            print("Lost target. Resetting")
            self.finished = True
            self.complete = False
            return

        if self.duration > self.train_duration:
            print("Duration Reached")
            self.finished = True
            self.complete = True
            return
        self.duration += 1
        
        self.history.append(detected_center)
        if len(self.history) == self.history_size:
            ns = np.array(self.history).flatten()
            self.next_state = torch.tensor(ns, dtype=torch.float32, device=self.dqn.device).unsqueeze(0)
            
            pitch_reward, yaw_reward = reward_calculation(detected_center[0], detected_center[1], self.reward_sharpness)
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
                if np.abs(detected_center[0]) >= safe_region(self.dqn.steps_done, self.pid_decay_start, self.pid_decay_end):
                    pa = self.discretize(self.pitch_pid.control(detected_center[0]), self.pitch_actions)
                    self.pitch_action = torch.tensor([[int(pa)]], device=self.dqn.device)
                if np.abs(detected_center[1]) >= safe_region(self.dqn.steps_done, self.pid_decay_start, self.pid_decay_end):
                    ya = self.discretize(self.yaw_pid.control(detected_center[1]), self.yaw_actions)
                    self.yaw_action = torch.tensor([[int(ya)]], device=self.dqn.device) 
                
                self.state = self.next_state
               
                # Perform one step of the optimization (on the policy network)
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

                if self.dqn.steps_done < self.pid_decay_start:
                    if self.dqn.steps_done % 25 == 0:
                        self.pitch_pid.target = np.random.uniform(-0.75, 0.75)
                        self.yaw_pid.target = np.random.uniform(-0.75, 0.75)
                else:
                    self.pitch_pid.target = 0.0
                    self.yaw_pid.target = 0.0

            #publish action
            pitch_action_idx = self.pitch_action.detach().cpu().numpy()[0][0]
            yaw_action_idx = self.yaw_action.detach().cpu().numpy()[0][0]
            self.command.data = [int(pitch_action_idx), int(yaw_action_idx)]
            self.command_publisher.publish(self.command)
            self.history = self.history[self.frame_skip:]
        return 
    
    def finish(self):

        #stop autopilot
        print('stopping autopilot')
        self.autopilot_flag.data = False
        self.autopilot_start_stop_publisher.publish(self.autopilot_flag)
        sleep(3.5)

        if self.popen_called:
            return 
          
        self.episode_rewards = np.array(self.episode_rewards)
        print('Episode rewards [pitch, yaw]. Average: ', np.mean(self.episode_rewards, axis=0), ' Sum: ', np.sum(self.episode_rewards, axis=0))
        
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
        sleep(3.5)

        #reset trajectory
        self.aqua_trajectory = []
        self.diver_trajectory = []

        #reset state and history queues
        self.state = None
        self.next_state = None
        self.pitch_reward = None
        self.yaw_reward = None
        self.history = []
        self.pitch_action = None
        self.yaw_action = None

        #reset flush queues 
        self.flush_imu = 0
        self.flush_detection = 0
        self.flush_diver = 0

        #reset counters
        self.duration = 0
        self.empty_state_counter = 0

        #reset end conditions 
        self.finished = False
        self.complete = False

        #reset diver pose
        self.diver_pose = None
        self.empty_state_counter = 0

        return
    
    def discretize(self, v, l):
        index = np.argmin(np.abs(np.subtract(l,v)))
        return index
    
def main(args=None):
    rclpy.init(args=args)

    node = dqn_controller()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
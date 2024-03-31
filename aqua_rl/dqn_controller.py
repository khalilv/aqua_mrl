import rclpy
import torch
import numpy as np 
import os
import subprocess
import shutil
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, UInt8MultiArray
from std_srvs.srv import SetBool
from time import sleep, time
from aqua_rl.control.DQN import DQN, ReplayMemory
from aqua_rl.control.PID import PID
from aqua_rl.helpers import reward_calculation, normalize_coords, safe_region
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
        self.load_erm = hyperparams.load_erm_ 
        self.experiment_number = hyperparams.experiment_number_
        self.train_for = hyperparams.train_for_
        self.train_duration = hyperparams.train_duration_
        self.eval_duration = hyperparams.eval_duration_
        self.reward_sharpness = hyperparams.sharpness_
        self.frame_skip = hyperparams.frame_skip_
        self.empty_state_max = hyperparams.empty_state_max_
        self.yaw_gains = hyperparams.yaw_gains_
        self.pitch_gains = hyperparams.pitch_gains_
        self.pid_decay_start = hyperparams.pid_decay_start_
        self.pid_decay_end = hyperparams.pid_decay_end_
        self.pitch_limit = hyperparams.pitch_limit_
        self.yaw_limit = hyperparams.yaw_limit_

        #subscribers and publishers
        self.command_publisher = self.create_publisher(UInt8MultiArray, hyperparams.autopilot_command_, self.queue_size)
        self.autopilot_start_stop_client = self.create_client(SetBool, hyperparams.autopilot_start_stop_)
        self.diver_start_stop_client = self.create_client(SetBool, hyperparams.diver_start_stop_)
        self.detection_subscriber = self.create_subscription(
            Float32MultiArray, 
            hyperparams.detection_topic_name_, 
            self.detection_callback, 
            self.queue_size)
        
        #initialize pid controllers
        self.pitch_pid = PID(target = 0, gains = self.pitch_gains, reverse=True, command_range=[-self.pitch_limit, self.pitch_limit])
        self.yaw_pid = PID(target = 0, gains = self.yaw_gains, reverse=True, command_range=[-self.yaw_limit, self.yaw_limit])
        self.pitch_actions = np.linspace(-self.pitch_limit, self.pitch_limit, self.pitch_action_space)
        self.yaw_actions = np.linspace(-self.yaw_limit, self.yaw_limit, self.yaw_action_space)

        #flush queues
        self.flush_steps = self.queue_size + 35
        self.flush_detection = 0

        #finished is episode has ended. complete is if aqua has reached the goal 
        self.finished = False
        self.complete = False
        self.evaluate = False 

        #dqn controller for yaw and pitch 
        self.dqn = DQN(self.pitch_action_space, self.yaw_action_space, self.history_size) 
        self.state = None
        self.next_state = None
        self.pitch_action = None        
        self.yaw_action = None
        self.pitch_reward = None
        self.yaw_reward = None
        self.history = []
        self.episode_rewards = []
        self.erm = ReplayMemory(self.dqn.MEMORY_SIZE)

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

        #autopilot start stop service data
        self.autopilot_start_stop_req = SetBool.Request()
        self.autopilot_start_stop_req.data = False

        #diver start stop service data
        self.diver_start_stop_req = SetBool.Request()
        self.diver_start_stop_req.data = False

        #duration counting
        self.duration = 0
        self.empty_state_counter = 0

        #popen_called
        self.popen_called = False

        print('Initialized: dqn controller')
    

    def detection_callback(self, coords):
        
        #flush detections queue
        if self.flush_detection < self.flush_steps:
            self.flush_detection += 1
            return
        
        if not self.autopilot_start_stop_req.data:
            print('Starting autopilot')
            self.autopilot_start_stop_req.data = True
            self.autopilot_start_stop_client.call_async(self.autopilot_start_stop_req)
        
        if not self.diver_start_stop_req.data:
            print('Starting diver controller')
            self.diver_start_stop_req.data = True
            self.diver_start_stop_client.call_async(self.diver_start_stop_req)

        #if finished, reset simulation
        if self.finished:
            self.finish()
            return
        
        coords = np.array(coords.data)
        
        #check for null input from detection module
        if coords[0] == -1 and coords[1] == -1 and coords[2] == -1 and coords[3] == -1:
            self.empty_state_counter += 1
            last_location = self.history[-1]
            yc, xc = last_location[0], last_location[1]
            dqn_state = [yc, xc, 0.0]
        else:
            self.empty_state_counter = 0
            yc = (coords[1] + coords[3])/2
            xc = (coords[0] + coords[2])/2
            yc, xc = normalize_coords(yc, xc, self.img_size, self.img_size)
            dqn_state = [yc, xc, 1.0]   
   
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
        
        self.history.append(dqn_state)
        if len(self.history) == self.history_size:
            ns = np.array(self.history).flatten()
            self.next_state = torch.tensor(ns, dtype=torch.float32, device=self.dqn.device).unsqueeze(0)
            
            pitch_reward, yaw_reward = reward_calculation(dqn_state[0], dqn_state[1], dqn_state[2], self.reward_sharpness)
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
                
                #override dqn actions if they are outside the safe region
                region = safe_region(self.dqn.steps_done, self.pid_decay_start, self.pid_decay_end)
                if np.abs(dqn_state[0]) >= region:
                    pa = self.discretize(self.pitch_pid.control(dqn_state[0]), self.pitch_actions)
                    self.pitch_action = torch.tensor([[int(pa)]], device=self.dqn.device)
                if np.abs(dqn_state[1]) >= region:
                    ya = self.discretize(self.yaw_pid.control(dqn_state[1]), self.yaw_actions)
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

                #in initial stages, set random pid target to explore state space
                if self.dqn.steps_done < self.pid_decay_start:
                    if self.dqn.steps_done % 25 == 0 and self.dqn.steps_done > 0 :
                        self.pitch_pid.target = np.random.uniform(-0.8, 0.8)
                        self.yaw_pid.target = np.random.uniform(-0.8, 0.8)
                #otherwise pid should always try to keep the target at 0
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

        if self.popen_called:
            return 
          
        self.episode_rewards = np.array(self.episode_rewards)
        mean_rewards = np.mean(self.episode_rewards, axis=0)
        sum_rewards = np.sum(self.episode_rewards, axis=0)
        print('Episode rewards [pitch, yaw]. Average: ', mean_rewards, ' Sum: ', sum_rewards)
        
        if self.evaluate:
            self.writer.add_scalar('Episode Rewards Pitch (Eval)', sum_rewards[0], self.episode)
            self.writer.add_scalar('Episode Rewards Yaw (Eval)', sum_rewards[1], self.episode)
        else:
            self.writer.add_scalar('Episode Rewards Pitch (Train)', sum_rewards[0], self.episode)
            self.writer.add_scalar('Episode Rewards Yaw (Train)', sum_rewards[1], self.episode)

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
        
        print('Stopping autopilot')
        self.autopilot_start_stop_req.data = False
        self.autopilot_start_stop_client.call_async(self.autopilot_start_stop_req)
        print('Stopping diver controller')
        self.diver_start_stop_req.data = False
        self.diver_start_stop_client.call_async(self.diver_start_stop_req)
        sleep(5)

        #reset state and history queues
        self.state = None
        self.next_state = None
        self.pitch_reward = None
        self.yaw_reward = None
        self.history = []
        self.pitch_action = None
        self.yaw_action = None

        #reset flush queues 
        self.flush_detection = 0

        #reset counters
        self.duration = 0
        self.empty_state_counter = 0

        #reset end conditions 
        self.finished = False
        self.complete = False
        print('-------------- Completed Reset --------------')
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
import rclpy
import torch
import numpy as np 
import os
import copy
import subprocess
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, UInt8MultiArray
from std_srvs.srv import SetBool
from time import sleep, time
from aqua_rl.control.DQN import DQN
from aqua_rl.control.TD3 import TD3, ReplayBuffer
from aqua_rl.helpers import reward_calculation, normalize_coords
from aqua_rl import hyperparams
from torch.utils.tensorboard import SummaryWriter 

class td3_adversary(Node):
    def __init__(self):
        super().__init__('td3_adversary')

        #hyperparams
        self.queue_size = hyperparams.queue_size_
        self.yaw_action_space = hyperparams.yaw_action_space_
        self.pitch_action_space = hyperparams.pitch_action_space_
        self.history_size = hyperparams.history_size_
        self.img_size = hyperparams.img_size_
        self.load_erm = hyperparams.load_erm_ 
        self.experiment_number = hyperparams.experiment_number_
        self.train_for = hyperparams.train_for_
        self.train_duration = hyperparams.train_duration_
        self.eval_duration = hyperparams.eval_duration_
        self.reward_sharpness = hyperparams.sharpness_
        self.frame_skip = hyperparams.frame_skip_
        self.empty_state_max = hyperparams.empty_state_max_
        self.adversary_limit = hyperparams.adv_limit_
        self.adversary_action_space = hyperparams.adv_action_space_

        #subscribers and publishers
        self.command_publisher = self.create_publisher(UInt8MultiArray, hyperparams.autopilot_command_, self.queue_size)
        self.current_publisher = self.create_publisher(Float32MultiArray, hyperparams.adv_command_topic_name_, self.queue_size)
        self.autopilot_start_stop_client = self.create_client(SetBool, hyperparams.autopilot_start_stop_)
        self.current_start_stop_client = self.create_client(SetBool, hyperparams.adv_start_stop_)
        self.diver_start_stop_client = self.create_client(SetBool, hyperparams.diver_start_stop_)
        self.detection_subscriber = self.create_subscription(
            Float32MultiArray, 
            hyperparams.detection_topic_name_, 
            self.detection_callback, 
            self.queue_size)
        
        #flush queues
        self.flush_steps = self.queue_size + 35
        self.flush_detection = 0

        #finished is episode has ended. complete is if aqua has reached the goal 
        self.finished = False
        self.complete = False
        self.evaluate = False 

        self.td3 = TD3(self.history_size, self.adversary_action_space, self.adversary_limit)
        
        #dqn controller for yaw and pitch 
        self.dqn = DQN(self.pitch_action_space, self.yaw_action_space, self.history_size) 
        self.state = None
        self.next_state = None
        self.adversary_action = None  
        self.reward = None      
        self.history = []
        self.episode_rewards = []
        self.adversary_erm = ReplayBuffer(int(self.history_size * 3), self.adversary_action_space, self.td3.memory_capacity)

        self.root_path = 'src/aqua_rl/experiments/{}'.format(str(self.experiment_number))
        self.save_path = os.path.join(self.root_path, 'weights/adv')
        self.save_memory_path = os.path.join(self.root_path, 'erm/adv')
        self.save_traj_path = os.path.join(self.root_path, 'trajectories/adv')
        self.writer = SummaryWriter(os.path.join(self.root_path, 'logs/adv'))
        last_checkpoint = max(sorted(os.listdir(self.save_path)))
        checkpoint = torch.load(os.path.join(self.save_path, last_checkpoint), map_location=self.td3.device)
        self.td3.critic.load_state_dict(checkpoint['critic_state_dict'], strict=True)
        self.td3.critic_optimizer.load_state_dict(checkpoint['critic_optim_state_dict'])
        self.critic_target = copy.deepcopy(self.td3.critic)
        self.td3.actor.load_state_dict(checkpoint['actor_state_dict'], strict=True)
        self.td3.actor_optimizer.load_state_dict(checkpoint['actor_optim_state_dict'])
        self.actor_target = copy.deepcopy(self.td3.actor)
        self.td3.total_it = checkpoint['total_it']

        self.dqn_save_path = os.path.join(self.root_path, 'weights')
        last_dqn_checkpoint = max(sorted(os.listdir(self.dqn_save_path)))
        dqn_checkpoint = torch.load(os.path.join(self.dqn_save_path, last_dqn_checkpoint), map_location=self.dqn.device)
        self.dqn.policy_net.load_state_dict(dqn_checkpoint['model_state_dict_policy'], strict=True)
        self.dqn.target_net.load_state_dict(dqn_checkpoint['model_state_dict_target'], strict=True)
        self.dqn.optimizer.load_state_dict(dqn_checkpoint['optimizer_state_dict'])
        self.dqn.steps_done = dqn_checkpoint['training_steps']
        
        if self.load_erm:
            print('Loading ERM from previous experience. Note this may take time')
            t0 = time()
            for file_path in sorted(os.listdir(self.save_memory_path), reverse=True):
                if os.path.isfile(os.path.join(self.save_memory_path, file_path)):
                    if self.td3.memory.size < self.td3.memory_capacity:
                        with open(os.path.join(self.save_memory_path, file_path), 'rb') as f:
                            states = np.load(f)
                            actions = np.load(f)
                            next_states = np.load(f)
                            rewards = np.load(f)
                            not_dones = np.load(f)
                            self.td3.memory.add_batch(states,actions,next_states,rewards,not_dones, len(states))
            t1 = time()
            print('ERM size: ', self.td3.memory.size, '. Time taken to load: ', t1 - t0)
            print(self.td3.memory.sample(5))
        else:
            print('WARNING: weights loaded but starting from a fresh replay memory')
        
        self.episode = int(last_checkpoint[8:13]) + 1
        self.stop_episode = self.episode + self.train_for - 1
        print('Weights loaded. starting from episode: ', self.episode, ', training steps completed: ', self.td3.total_it)
    
        #autopilot commands
        self.command = UInt8MultiArray()
        self.adversary_command = Float32MultiArray()

        #autopilot start stop service data
        self.autopilot_start_stop_req = SetBool.Request()
        self.autopilot_start_stop_req.data = False

        #diver start stop service data
        self.diver_start_stop_req = SetBool.Request()
        self.diver_start_stop_req.data = False

        #current start stop service data
        self.current_start_stop_req = SetBool.Request()
        self.current_start_stop_req.data = False

        #duration counting
        self.duration = 0
        self.empty_state_counter = 0

        #popen_called
        self.popen_called = False

        print('Initialized: td3 adversary')
    

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
        
        if not self.current_start_stop_req.data:
            print('Starting current controller')
            self.current_start_stop_req.data = True
            self.current_start_stop_client.call_async(self.current_start_stop_req)

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
            self.episode_rewards.append(pitch_reward + yaw_reward)
            self.reward = -0.5*(pitch_reward + yaw_reward) #sum rewards together and divide by negative two to keep rewards in range [-1,1]
            
            if self.evaluate:
                #select greedy action, dont optimize model or append to replay buffer
                self.adversary_action = self.td3.select_action(self.next_state)
            else:
                if self.state is not None:
                    self.td3.memory.add(self.state.detach().cpu().numpy(), self.adversary_action.detach().cpu().numpy(), self.next_state.detach().cpu().numpy(), self.reward, False)
                    self.adversary_erm.add(self.state.detach().cpu().numpy(), self.adversary_action.detach().cpu().numpy(), self.next_state.detach().cpu().numpy(), self.reward, False)
                
                #add noise to action
                self.adversary_action = (self.td3.select_action(self.next_state) + torch.tensor(np.random.normal(0, self.td3.limit * self.td3.expl_noise, size=self.adversary_action_space), device=self.td3.device)).clip(-self.td3.limit, self.td3.limit)       
                self.state = self.next_state
               
                # Perform one step of the optimization (on the policy network)
                loss = self.td3.train()
                if loss is not None:        
                    self.writer.add_scalar('Loss', loss, self.td3.total_it)

            #publish actions
            pitch_action, yaw_action = self.dqn.select_eval_action(self.next_state)
            pitch_action_idx = pitch_action.detach().cpu().numpy()[0][0]
            yaw_action_idx = yaw_action.detach().cpu().numpy()[0][0]
            self.command.data = [int(pitch_action_idx), int(yaw_action_idx)]
            self.command_publisher.publish(self.command)

            current = self.adversary_action.detach().cpu().numpy()[0]
            self.adversary_command.data = [float(current[0]), float(current[1]), float(current[2])]
            self.current_publisher.publish(self.adversary_command)

            self.history = self.history[self.frame_skip:]
        return 
    
    def finish(self):

        if self.popen_called:
            return 
          
        self.episode_rewards = np.array(self.episode_rewards)
        mean_rewards = np.mean(self.episode_rewards)
        sum_rewards = np.sum(self.episode_rewards)
        print('Episode rewards. Average: ', mean_rewards, ' Sum: ', sum_rewards)
        
        if self.evaluate:
            self.writer.add_scalar('Episode Rewards (Eval)', sum_rewards, self.td3.total_it)
        else:
            self.writer.add_scalar('Episode Rewards (Train)', sum_rewards, self.td3.total_it)

        self.writer.add_scalar('Duration', self.duration, self.td3.total_it)

        if self.state is not None and not self.evaluate and not self.complete:
            self.td3.memory.add(self.state.detach().cpu().numpy(), self.adversary_action.detach().cpu().numpy(), self.next_state.detach().cpu().numpy(), self.reward, True)
            self.adversary_erm.add(self.state.detach().cpu().numpy(), self.adversary_action.detach().cpu().numpy(), self.next_state.detach().cpu().numpy(), self.reward, True)
                
        if self.episode == self.stop_episode:
            print('Saving checkpoint')
            torch.save({
                'critic_state_dict': self.td3.critic.state_dict(),
                'critic_optim_state_dict': self.td3.critic_optimizer.state_dict(),
                'actor_state_dict': self.td3.actor.state_dict(),
                'actor_optim_state_dict': self.td3.actor_optimizer.state_dict(),
                'total_it': self.td3.total_it
            }, self.save_path +  '/episode_{}.pt'.format(str(self.episode).zfill(5)))
            
            with open(self.save_memory_path +  '/episode_{}.pt'.format(str(self.episode).zfill(5)), 'wb') as f:
                np.save(f, self.adversary_erm.state[:self.adversary_erm.ptr,:])
                np.save(f, self.adversary_erm.action[:self.adversary_erm.ptr,:])
                np.save(f, self.adversary_erm.next_state[:self.adversary_erm.ptr,:])
                np.save(f, self.adversary_erm.reward[:self.adversary_erm.ptr,:])
                np.save(f, self.adversary_erm.not_done[:self.adversary_erm.ptr,:])

        with open(self.save_traj_path + '/episode_{}.npy'.format(str(self.episode).zfill(5)), 'wb') as f:
            np.save(f, self.episode_rewards)

        if self.episode < self.stop_episode:
            self.reset()
        else:
            subprocess.Popen('python3 ./src/aqua_rl/aqua_rl/adversary_resetter.py', shell=True)
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
        print('Stopping current controller')
        self.current_start_stop_req.data = False
        self.current_start_stop_client.call_async(self.current_start_stop_req)
        sleep(5)

        #reset state and history queues
        self.state = None
        self.next_state = None
        self.reward = None
        self.history = []
        self.adversary_action = None

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

    
def main(args=None):
    rclpy.init(args=args)

    node = td3_adversary()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
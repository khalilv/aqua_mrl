import rclpy
import torch
import numpy as np 
import os
import subprocess
import shutil
import copy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, UInt8MultiArray
from std_srvs.srv import SetBool
from time import sleep, time
from aqua_rl.control.DQN import DQN, ReplayMemory
from aqua_rl.control.PID import PID
from aqua_rl.helpers import reward_calculation, normalize_coords, safe_region, get_command, get_current
from aqua_rl import hyperparams
from torch.utils.tensorboard import SummaryWriter 

class evaluation(Node):
    def __init__(self):
        super().__init__('evaluation')

        #hyperparams
        self.queue_size = hyperparams.queue_size_
        self.history_size = hyperparams.history_size_
        self.yaw_action_space = hyperparams.yaw_action_space_
        self.pitch_action_space = hyperparams.pitch_action_space_
        self.img_size = hyperparams.img_size_
        self.experiment_number = hyperparams.experiment_number_
        self.eval_for = hyperparams.eval_for_
        self.eval_duration = hyperparams.eval_duration_
        self.reward_sharpness = hyperparams.sharpness_
        self.frame_skip = hyperparams.frame_skip_
        self.empty_state_max = hyperparams.empty_state_max_
        self.adversary_action_space = hyperparams.adv_action_space_
        self.switch_every = hyperparams.switch_every_
        self.eval_episode = hyperparams.eval_episode_

        #subscribers and publishers
        self.command_publisher = self.create_publisher(UInt8MultiArray, hyperparams.autopilot_command_, self.queue_size)
        self.current_publisher = self.create_publisher(UInt8MultiArray, hyperparams.adv_command_topic_name_, self.queue_size)
        self.autopilot_start_stop_client = self.create_client(SetBool, hyperparams.autopilot_start_stop_)
        self.diver_start_stop_client = self.create_client(SetBool, hyperparams.diver_start_stop_)
        self.current_start_stop_client = self.create_client(SetBool, hyperparams.adv_start_stop_)
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

        #dqn controller for yaw and pitch 
        self.dqn = DQN(int(self.pitch_action_space * self.yaw_action_space), self.history_size) 
        self.history = []
        self.episode_rewards = []
        self.erm = ReplayMemory(self.dqn.MEMORY_SIZE)
        self.episode = 0
        self.stop_episode = self.eval_for - 1

        # self.adv = DQN(int(np.power(3, self.adversary_action_space)), self.history_size)

        self.weight_path = 'src/aqua_rl/experiments/{}/weights/episode_{}.pt'.format(str(self.experiment_number), str(self.eval_episode).zfill(5))
        checkpoint = torch.load(self.weight_path, map_location=self.dqn.device)
        self.dqn.policy_net.load_state_dict(checkpoint['model_state_dict_policy'], strict=True)
        self.dqn.target_net.load_state_dict(checkpoint['model_state_dict_target'], strict=True)
        self.dqn.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.dqn.steps_done = checkpoint['training_steps']
        
        # self.adv = DQN(int(np.power(3, self.adversary_action_space)), self.history_size)
        # self.adv_weight_path = 'src/aqua_rl/experiments/{}/weights/adv/episode_{}.pt'.format(str(self.experiment_number), str(ADV_EPISODE).zfill(5))
        # adv_checkpoint = torch.load(self.adv_weight_path, map_location=self.adv.device)
        # self.adv.policy_net.load_state_dict(adv_checkpoint['model_state_dict_policy'], strict=True)
        # self.adv.target_net.load_state_dict(adv_checkpoint['model_state_dict_target'], strict=True)
        # self.adv.optimizer.load_state_dict(adv_checkpoint['optimizer_state_dict'])
        # self.adv.steps_done = adv_checkpoint['training_steps']
        # print('DQN adversary loaded. Steps completed: ', self.adv.steps_done)
        
        self.save_path = 'src/aqua_rl/evaluations/{}/'.format(str(self.experiment_number))
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        #autopilot commands
        self.command = UInt8MultiArray()
        self.adversary_command = UInt8MultiArray()

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

        if self.duration > self.eval_duration:
            print("Duration Reached")
            self.finished = True
            self.complete = True
            return
        self.duration += 1
        
        self.history.append(dqn_state)
        if len(self.history) == self.history_size:
            s = np.array(self.history).flatten()
            state = torch.tensor(s, dtype=torch.float32, device=self.dqn.device).unsqueeze(0)
            reward = reward_calculation(dqn_state[0], dqn_state[1], dqn_state[2], self.reward_sharpness)
            self.episode_rewards.append(reward)
            action = self.dqn.select_eval_action(state)

            #publish adversary action
            # adversary_action = self.adv.select_eval_action(self.next_state).detach().cpu().numpy()[0][0]
            # current = get_current(adversary_action, self.adversary_action_space)
            # self.adversary_command.data = [int(current[0]), int(current[1]), int(current[2])]
            # self.current_publisher.publish(self.adversary_command)
            
            #publish actions
            action_idx = action.detach().cpu().numpy()[0][0]
            pitch_action_idx, yaw_action_idx = get_command(action_idx, self.pitch_action_space, self.yaw_action_space)
            self.command.data = [int(pitch_action_idx), int(yaw_action_idx)]
            self.command_publisher.publish(self.command)
            self.history = self.history[self.frame_skip:]

        return 
    
    def finish(self):
         
        self.episode_rewards = np.array(self.episode_rewards)
        mean_rewards = np.mean(self.episode_rewards)
        sum_rewards = np.sum(self.episode_rewards)
        print('Episode rewards. Average: ', mean_rewards, ' Sum: ', sum_rewards)
               
        with open(self.save_path + '/episode_{}.npy'.format(str(self.episode).zfill(5)), 'wb') as f:
            np.save(f, self.episode_rewards)

        if self.episode < self.stop_episode:
            self.reset()
        else:
            rclpy.shutdown()
        return

    def reset(self):
        print('-------------- Resetting simulation --------------')
        
        #increment episode and reset rewards
        self.episode_rewards = []
        self.episode += 1
       
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
        self.history = []

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
    
    # def discretize(self, v, l):
    #     index = np.argmin(np.abs(np.subtract(l,v)))
    #     return index
    
def main(args=None):
    rclpy.init(args=args)

    node = evaluation()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
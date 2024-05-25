import rclpy
import torch
import numpy as np 
import os
import subprocess
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import SetBool
from aqua2_interfaces.srv import SetFloat
from time import sleep, time
from aqua_rl.control.ThreeHeadDQN import ThreeHeadDQN, ThreeHeadReplayMemory
from aqua_rl.helpers import reward_calculation, normalize_coords
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
        self.speed_action_space = hyperparams.speed_action_space_
        self.img_size = hyperparams.img_size_
        self.eval_duration = hyperparams.eval_duration_
        self.location_sigma = hyperparams.location_sigma_
        self.area_sigma = hyperparams.area_sigma_
        self.frame_skip = hyperparams.frame_skip_
        self.empty_state_max = hyperparams.empty_state_max_
        self.target_area = hyperparams.target_area_
        self.initialize_debris_after = hyperparams.initialize_debris_after_
        self.experiment_name = 'baseline'
        self.weight_path = '/usr/local/data/kvirji/AQUA/aqua_rl/experiments/13/weights/episode_03300.pt'

        #subscribers and publishers
        self.command_publisher = self.create_publisher(Float32MultiArray, hyperparams.autopilot_command_, self.queue_size)
        self.autopilot_start_stop_client = self.create_client(SetBool, hyperparams.autopilot_start_stop_)
        self.diver_start_stop_client = self.create_client(SetBool, hyperparams.diver_start_stop_)
        self.debris_client = self.create_client(SetBool, hyperparams.debris_srv_name_)

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
        self.dqn = ThreeHeadDQN(self.pitch_action_space, self.yaw_action_space, self.speed_action_space, self.history_size) 
        self.state = None
        self.reward = None
        self.pitch_action =  torch.tensor([[3]], device=self.dqn.device, dtype=torch.long)   
        self.yaw_action = torch.tensor([[3]], device=self.dqn.device, dtype=torch.long)    
        self.speed_action = torch.tensor([[0]], device=self.dqn.device, dtype=torch.long)    

        self.action_history = []
        self.history = []
        self.episode_rewards = []
        self.target_locations = []
        self.erm = ThreeHeadReplayMemory(self.dqn.MEMORY_SIZE) 

        self.save_path = os.path.join('src/aqua_rl/dqn_evaluations', self.experiment_name)
        checkpoint = torch.load(self.weight_path, map_location=self.dqn.device)
        self.dqn.policy_net.load_state_dict(checkpoint['model_state_dict_policy'], strict=True)
        self.dqn.target_net.load_state_dict(checkpoint['model_state_dict_target'], strict=True)
        self.dqn.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.dqn.steps_done = checkpoint['training_steps']
        
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.episode = len(os.listdir(self.save_path))
        print('Weights loaded. starting from episode: ', self.episode, ', training steps completed: ', self.dqn.steps_done)
        
        #autopilot commands
        self.command = Float32MultiArray()

        #autopilot start stop service data
        self.autopilot_start_stop_req = SetBool.Request()
        self.autopilot_start_stop_req.data = False

        #debris service data
        self.debris_req = SetBool.Request()
        self.debris_req.data = False

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
            if len(self.history) > 0:
                last_location = self.history[-1]
            else:
                return #diver has not been located yet
            yc, xc, a = last_location[0], last_location[1], last_location[2]
            dqn_state = [yc, xc, a, 0.0]
        else:
            self.empty_state_counter = 0
            yc = (coords[1] + coords[3])/2
            xc = (coords[0] + coords[2])/2
            a = (np.abs(coords[1] - coords[3]) * np.abs(coords[0] - coords[2]))/(self.img_size*self.img_size)
            yc, xc = normalize_coords(yc, xc, self.img_size, self.img_size)
            dqn_state = [yc, xc, a, 1.0]   
   
        if self.empty_state_counter >= self.empty_state_max:
            print("Lost target. Resetting")
            self.finished = True
            self.complete = False
            return

        if self.duration >= self.eval_duration:
            print("Duration Reached")
            self.finished = True
            self.complete = True
            return
        self.duration += 1

        if self.duration == self.initialize_debris_after and not self.debris_req.data:            
            print('Initializing debris')
            self.debris_req.data = True
            self.debris_client.call_async(self.debris_req)
                
        self.history.append(dqn_state)
        if len(self.history) == self.history_size and len(self.action_history) == self.history_size - 1:
            ns = np.concatenate((np.array(self.history).flatten(), np.array(self.action_history).flatten()))
            self.state = torch.tensor(ns, dtype=torch.float32, device=self.dqn.device).unsqueeze(0)
            
            reward = reward_calculation(dqn_state[0], dqn_state[1], dqn_state[2], dqn_state[3], self.location_sigma, self.area_sigma, self.target_area)
            self.episode_rewards.append(reward)
            self.target_locations.append(dqn_state)
            self.reward = torch.tensor([reward], dtype=torch.float32, device=self.dqn.device)
            self.pitch_action, self.yaw_action, self.speed_action = self.dqn.select_eval_action(self.state)
            
            self.history = self.history[self.frame_skip:]
            self.action_history = self.action_history[self.frame_skip:]
                    
        #publish actions
        pitch_action_idx = self.pitch_action.detach().cpu().numpy()[0][0]
        yaw_action_idx = self.yaw_action.detach().cpu().numpy()[0][0]
        speed_action_idx = self.speed_action.detach().cpu().numpy()[0][0]
        self.command.data = [float(pitch_action_idx), float(yaw_action_idx), float(speed_action_idx)]
        self.command_publisher.publish(self.command)
        self.action_history.append([pitch_action_idx, yaw_action_idx, speed_action_idx])
        return 
    
    def finish(self):

        if self.popen_called:
            return 
          
        self.episode_rewards = np.array(self.episode_rewards)
        self.target_locations = np.array(self.target_locations)
        mean_rewards = np.mean(self.episode_rewards)
        sum_rewards = np.sum(self.episode_rewards)
        print('Episode rewards. Average: ', mean_rewards, ' Sum: ', sum_rewards)
        with open(self.save_path + '/episode_{}.npy'.format(str(self.episode).zfill(5)), 'wb') as f:
            np.save(f, self.episode_rewards)
            np.save(f, self.target_locations)
        self.reset()
        return

    def reset(self):
        print('-------------- Resetting simulation --------------')
        
        #increment episode and reset rewards
        self.episode_rewards = []
        self.target_locations = []
        self.episode += 1
        
        print('Stopping autopilot')
        self.autopilot_start_stop_req.data = False
        self.autopilot_start_stop_client.call_async(self.autopilot_start_stop_req)
        print('Stopping diver controller')
        self.diver_start_stop_req.data = False
        self.diver_start_stop_client.call_async(self.diver_start_stop_req)
        print('Removing debris')
        self.debris_req.data = False
        self.debris_client.call_async(self.debris_req)
        sleep(5)

        #reset state and history queues
        self.state = None
        self.reward = None
        self.history = []
        self.action_history = []
        self.pitch_action =  torch.tensor([[3]], device=self.dqn.device, dtype=torch.long)   
        self.yaw_action = torch.tensor([[3]], device=self.dqn.device, dtype=torch.long)    
        self.speed_action = torch.tensor([[0]], device=self.dqn.device, dtype=torch.long)    

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

    node = evaluation()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
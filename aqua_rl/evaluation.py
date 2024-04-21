import rclpy
import torch
import numpy as np 
import os
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, UInt8MultiArray
from std_srvs.srv import SetBool
from time import sleep
from aqua_rl.control.TwoHeadDQN import TwoHeadDQN
from aqua_rl.helpers import reward_calculation, normalize_coords
from aqua_rl import hyperparams

class evaluation(Node):
    def __init__(self):
        super().__init__('evaluation')

        #hyperparams
        self.queue_size = hyperparams.queue_size_
        self.history_size = hyperparams.history_size_
        self.yaw_action_space = hyperparams.yaw_action_space_
        self.pitch_action_space = hyperparams.pitch_action_space_
        self.img_size = hyperparams.img_size_
        self.experiment_number = hyperparams.eval_experiment_number_
        self.eval_for = hyperparams.eval_for_
        self.eval_duration = hyperparams.eval_duration_
        self.reward_sigma = hyperparams.sigma_
        self.frame_skip = hyperparams.frame_skip_
        self.empty_state_max = hyperparams.empty_state_max_
        self.switch_every = hyperparams.switch_every_
        self.eval_episode = hyperparams.eval_episode_
        self.eval_name = hyperparams.eval_name_

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
        self.dqn = TwoHeadDQN(self.pitch_action_space, self.yaw_action_space, self.history_size) 
        self.history = []
        self.action_history = []
        self.episode_rewards = []
        self.episode = 0
        self.pitch_action =  torch.tensor([[2]], device=self.dqn.device, dtype=torch.long)   
        self.yaw_action = torch.tensor([[2]], device=self.dqn.device, dtype=torch.long)  
        self.stop_episode = self.eval_for - 1

        self.weight_path = 'src/aqua_rl/experiments/{}/episode_{}.pt'.format(str(self.experiment_number), str(self.eval_episode).zfill(5))
        checkpoint = torch.load(self.weight_path, map_location=self.dqn.device)
        self.dqn.policy_net.load_state_dict(checkpoint['model_state_dict_policy'], strict=True)
        self.dqn.target_net.load_state_dict(checkpoint['model_state_dict_target'], strict=True)
        self.dqn.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.dqn.steps_done = checkpoint['training_steps']
        
        self.save_path = 'src/aqua_rl/evaluations/{}/'.format(str(self.experiment_number))
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

        self.save_path = os.path.join(self.save_path, self.eval_name)
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
        
        self.history.append(dqn_state)
        if len(self.history) == self.history_size and len(self.action_history) == self.history_size - 1:
            s = np.concatenate((np.array(self.history).flatten(), np.array(self.action_history).flatten()))
            state = torch.tensor(s, dtype=torch.float32, device=self.dqn.device).unsqueeze(0)
            reward = reward_calculation(dqn_state[0], dqn_state[1], dqn_state[2], self.reward_sigma)
            self.episode_rewards.append(reward)
            self.pitch_action, self.yaw_action = self.dqn.select_eval_action(state)
            self.history = self.history[self.frame_skip:]
            self.action_history = self.action_history[self.frame_skip:]

            #publish adversary action
            # adversary_action = self.adv.select_eval_action(self.next_state).detach().cpu().numpy()[0][0]
            # current = get_current(adversary_action, self.adversary_action_space)
            # self.adversary_command.data = [int(current[0]), int(current[1]), int(current[2])]
            # self.current_publisher.publish(self.adversary_command)
            
        #publish actions
        pitch_action_idx = self.pitch_action.detach().cpu().numpy()[0][0]
        yaw_action_idx = self.yaw_action.detach().cpu().numpy()[0][0]
        self.command.data = [int(pitch_action_idx), int(yaw_action_idx)]
        self.command_publisher.publish(self.command)
        self.action_history.append([pitch_action_idx, yaw_action_idx])
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
            self.reset()
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
        self.action_history = []
        self.pitch_action =  torch.tensor([[2]], device=self.dqn.device, dtype=torch.long)   
        self.yaw_action = torch.tensor([[2]], device=self.dqn.device, dtype=torch.long)    
        
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
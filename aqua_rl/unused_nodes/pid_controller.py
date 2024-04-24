import rclpy
import torch
import numpy as np 
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, UInt8MultiArray
from std_srvs.srv import SetBool
from time import sleep
from aqua_rl.control.DQN import ReplayMemory
from aqua_rl.control.PID import PID
from aqua_rl.helpers import reward_calculation, normalize_coords
from aqua_rl import hyperparams

class pid_controller(Node):
    def __init__(self):
        super().__init__('pid_controller')

        #hyperparams
        self.queue_size = hyperparams.queue_size_
        self.history_size = hyperparams.history_size_
        self.yaw_action_space = hyperparams.yaw_action_space_
        self.pitch_action_space = hyperparams.pitch_action_space_
        self.img_size = hyperparams.img_size_
        self.load_erm = hyperparams.load_erm_ 
        self.train_for = hyperparams.train_for_
        self.train_duration = hyperparams.train_duration_
        self.reward_sigma = hyperparams.sigma_
        self.frame_skip = hyperparams.frame_skip_
        self.empty_state_max = hyperparams.empty_state_max_
        self.yaw_gains = hyperparams.yaw_gains_
        self.pitch_gains = hyperparams.pitch_gains_
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

        #dqn controller for yaw and pitch 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state = None
        self.next_state = None
        self.pitch_action = None        
        self.yaw_action = None
        self.pitch_reward = None
        self.yaw_reward = None
        self.history = []
        self.episode_rewards = []
        self.erm = ReplayMemory(10000)

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

        print('Initialized: pid controller')
    

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
            yc, xc = last_location[0],last_location[1]
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
            self.next_state = torch.tensor(ns, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            pitch_reward, yaw_reward = reward_calculation(dqn_state[0], dqn_state[1], dqn_state[2], self.reward_sigma)
            self.episode_rewards.append([pitch_reward, yaw_reward])
            self.pitch_reward = torch.tensor([pitch_reward], dtype=torch.float32, device=self.device)
            self.yaw_reward = torch.tensor([yaw_reward], dtype=torch.float32, device=self.device)
            
           
            if self.state is not None:
                self.erm.push(self.state, self.pitch_action, self.yaw_action, self.next_state, self.pitch_reward, self.yaw_reward)
            
            
            pa = self.discretize(self.pitch_pid.control(dqn_state[0]), self.pitch_actions)
            ya = self.discretize(self.yaw_pid.control(dqn_state[1]), self.yaw_actions)

            self.pitch_action = torch.tensor([[int(pa)]], device=self.device)    
            self.yaw_action = torch.tensor([[int(ya)]], device=self.device) 
                
            self.state = self.next_state

            #publish action
            pitch_action_idx = self.pitch_action.detach().cpu().numpy()[0][0]
            yaw_action_idx = self.yaw_action.detach().cpu().numpy()[0][0]
            self.command.data = [int(pitch_action_idx), int(yaw_action_idx)]
            self.command_publisher.publish(self.command)
            self.history = self.history[self.frame_skip:]
        return 
    
    def finish(self):
         
        self.episode_rewards = np.array(self.episode_rewards)
        mean_rewards = np.mean(self.episode_rewards, axis=0)
        sum_rewards = np.sum(self.episode_rewards, axis=0)
        print('Episode rewards [pitch, yaw]. Average: ', mean_rewards, ' Sum: ', sum_rewards)
        
        if self.state is not None and not self.complete:
            self.erm.push(self.state, self.pitch_action, self.yaw_action, None, self.pitch_reward, self.yaw_reward)

        # torch.save({
        #     'memory': self.erm
        # }, 'pid_expert.pt')
            
        self.reset()
        return

    def reset(self):
        print('-------------- Resetting simulation --------------')
        
        #increment episode and reset rewards
        self.episode_rewards = []
        
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
        print('----------------------------')

        return
    
    def discretize(self, v, l):
        index = np.argmin(np.abs(np.subtract(l,v)))
        return index
    
def main(args=None):
    rclpy.init(args=args)

    node = pid_controller()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
import rclpy
import torch
import numpy as np 
from pynput import keyboard
from rclpy.node import Node
from aqua2_interfaces.msg import Command, AquaPose
from std_msgs.msg import UInt8MultiArray, Float32
from aqua_rl.control.PID import AnglePID, PID
from aqua_rl.control.DQN import ReplayMemory
from aqua_rl.helpers import reward_calculation, euler_from_quaternion
from aqua_rl import hyperparams
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry

class manual_controller(Node):
    def __init__(self):
        super().__init__('manual_controller')
        #hyperparams
        self.queue_size = hyperparams.queue_size_
        self.roll_gains = hyperparams.roll_gains_
        self.pitch_gains = hyperparams.pitch_gains_
        self.history_size = hyperparams.history_size_
        self.pitch_limit = hyperparams.pitch_limit_
        self.yaw_limit = hyperparams.yaw_limit_
        self.yaw_action_space = hyperparams.yaw_action_space_
        self.pitch_action_space = hyperparams.pitch_action_space_
        self.img_size = hyperparams.img_size_
        self.depth_range = hyperparams.depth_range_
        self.target_depth = hyperparams.target_depth_
        self.frames_to_skip = hyperparams.frames_to_skip_
        self.roi_detection_threshold = hyperparams.roi_detection_threshold_
        self.mean_importance = hyperparams.mean_importance_
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.using_hardware_topics = hyperparams.using_hardware_topics_

        #subscribers and publishers
        self.command_publisher = self.create_publisher(Command, hyperparams.command_topic_name_, self.queue_size)
        self.segmentation_subscriber = self.create_subscription(
            UInt8MultiArray, 
            '/segmentation', 
            self.segmentation_callback, 
            self.queue_size)
        if self.using_hardware_topics:
            self.depth_subscriber = self.create_subscription(Odometry, hyperparams.depth_topic_name_, self.depth_callback, self.queue_size)
            self.imu_subscriber = self.create_subscription(Imu, hyperparams.imu_topic_name_, self.imu_callback, self.queue_size)
            self.roll_pid = AnglePID(target = 0.0, gains = self.roll_gains)
        else:
            self.depth_subscriber = self.create_subscription(Float32, hyperparams.depth_topic_name_, self.depth_callback, self.queue_size)
            self.imu_subscriber = self.create_subscription(AquaPose, hyperparams.imu_topic_name_, self.imu_callback, self.queue_size)
            self.roll_pid = AnglePID(target = 0.0, gains = self.roll_gains, reverse=True)
            
        self.measured_roll_angle = 0.0
        self.relative_depth = None
        
        #dqn controller for yaw and pitch 
        self.yaw_actions = np.linspace(-self.yaw_limit, self.yaw_limit, self.yaw_action_space)
        self.pitch_actions = np.linspace(-self.pitch_limit, self.pitch_limit, self.pitch_action_space)
        self.state = None
        self.next_state = None
        self.state_depths = None
        self.next_state_depths = None
        self.state_actions = None
        self.next_state_actions = None
        self.action = torch.tensor([[4]], device=self.device, dtype=torch.long)
        self.reward = None
        self.image_history = []
        self.depth_history = []
        self.action_history = [4]
        self.episode_rewards = []
        self.erm = ReplayMemory(5000)

        #initialize command
        self.command = Command()
        self.command.speed = 0.0 
        self.command.roll = 0.0
        self.command.pitch = 0.0
        self.command.yaw = 0.0
        self.command.heave = 0.0

        self.yaw_action_idx = 1
        self.pitch_action_idx = 1

        self.listener = keyboard.Listener(on_press=self.get_command)
        self.listener.start()
        self.recieved_command = False

        self.file_name = 'expert_erm.pt'
        self.samples_collected = 0
        self.finished = False
        print('Initialized: manual controller ')
  
    def get_command(self, key):
        try:
            if key.char == 'a': # yaw right
                self.yaw_action_idx = 0
            elif key.char == 'd': # yaw left
                self.yaw_action_idx = 2
            elif key.char == 's':  #straight
                self.yaw_action_idx = 1 
            elif key.char == 'i': #pitch up
                self.pitch_action_idx = 0
            elif key.char == 'm': #pitch down
                self.pitch_action_idx = 2
            elif key.char == 'k': #no pitch
                self.pitch_action_idx = 1
        except AttributeError:
            if key == keyboard.Key.space: # save erm
                torch.save({
                    'memory': self.erm
                }, './src/aqua_rl/' + self.file_name)
                print('Saved ERM. Please kill node')
                self.finished = True
        return 

    def imu_callback(self, imu):
        self.measured_roll_angle = self.calculate_roll(imu)
        return

    def calculate_roll(self, imu):
        if self.using_hardware_topics:
            roll, _, _ = euler_from_quaternion(imu.orientation)
            return roll * 180/np.pi
        else:
            return imu.roll
    
    def depth_callback(self, depth):
        if self.using_hardware_topics:
            self.relative_depth = self.target_depth + depth.pose.pose.position.z
        else:
            self.relative_depth = self.target_depth + depth.data
        return

    def segmentation_callback(self, seg_map):

        if self.finished:
            self.command.speed = 0.0 
            self.command.roll = 0.0
            self.command.pitch = 0.0
            self.command.yaw = 0.0
            self.command.heave = 0.0
            self.command_publisher.publish(self.command)
            return
        
        #exit if depth has not been measured
        if self.relative_depth is None:
            return 
        
        seg_map = np.array(seg_map.data).reshape(self.img_size)
               
        self.depth_history.append(self.relative_depth)
        self.image_history.append(seg_map)
        if len(self.image_history) == self.history_size and len(self.depth_history) == self.history_size and len(self.action_history) == self.history_size:
            ns = np.array(self.image_history)
            nsd = np.array(self.depth_history)
            nsa = np.array(self.action_history)
                                  
            self.next_state = torch.tensor(ns, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.next_state_depths = torch.tensor(nsd, dtype=torch.float32, device=self.device).unsqueeze(0)
            self.next_state_actions = torch.tensor(nsa, dtype=torch.float32, device=self.device).unsqueeze(0)

            reward = reward_calculation(seg_map, self.relative_depth, self.roi_detection_threshold, self.mean_importance)

            self.episode_rewards.append(reward)
            self.reward = torch.tensor([reward], dtype=torch.float32, device=self.device)

            if self.state is not None and self.state_depths is not None and self.state_actions is not None:
                self.erm.push(self.state, self.state_depths, self.state_actions, self.action, self.next_state, self.next_state_depths, self.next_state_actions, self.reward)
                self.samples_collected += 1
                print('Expert samples collected: ', self.samples_collected)

            a = int(self.pitch_action_idx*self.yaw_action_space) + self.yaw_action_idx
            a = torch.tensor([[a]], device=self.device, dtype=torch.long)
            self.action = a    
            self.state = self.next_state
            self.state_depths = self.next_state_depths
            self.state_actions = self.next_state_actions

            self.image_history = self.image_history[self.frames_to_skip:]
            self.depth_history = self.depth_history[self.frames_to_skip:]
            self.action_history = self.action_history[self.frames_to_skip:]
  
        action_idx = self.action.detach().cpu().numpy()[0][0]
        self.action_history.append(action_idx)
        self.command.pitch = self.pitch_actions[int(action_idx/self.yaw_action_space)]
        self.command.yaw = self.yaw_actions[action_idx % self.yaw_action_space]            
        self.command.speed = hyperparams.speed_ #fixed speed
        self.command.roll = self.roll_pid.control(self.measured_roll_angle)
        self.command_publisher.publish(self.command)
        return 
    
    def discretize(self, v, l):
        index = np.argmin(np.abs(np.subtract(l,v)))
        return index


def main(args=None):
    rclpy.init(args=args)

    node = manual_controller()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

    



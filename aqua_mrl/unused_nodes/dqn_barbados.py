import rclpy
import torch
import numpy as np 
import os
from rclpy.node import Node
from aqua2_interfaces.msg import Command
from aqua2_interfaces.srv import RecordTopics
from std_srvs.srv import Empty
from std_msgs.msg import UInt8MultiArray 
from aqua_mrl.control.PID import AnglePID
from aqua_mrl.control.DQN import DQN, ReplayMemory
from aqua_mrl.helpers import reward_calculation, euler_from_quaternion
from aqua_mrl import hyperparams
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from time import sleep

class dqn_barbados(Node):
    def __init__(self):
        super().__init__('dqn_controller_barbados')

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
        self.detection_threshold = hyperparams.detection_threshold_
        self.frames_to_skip = hyperparams.frames_to_skip_
        self.roi_detection_threshold = hyperparams.roi_detection_threshold_
        self.mean_importance = hyperparams.mean_importance_
        self.eval_duration = hyperparams.eval_duration_
        self.checkpoint_path = 'src/aqua_mrl/experiments/0/best.pt'
        self.save_path = 'src/aqua_mrl/evaluations/test/'
        self.bagfile_prefix = 'test'
        
        #subscribers and publishers
        self.command_publisher = self.create_publisher(Command, hyperparams.command_topic_name_, self.queue_size)
        self.depth_subscriber = self.create_subscription(Odometry, hyperparams.depth_topic_name_, self.depth_callback, self.queue_size)
        self.imu_subscriber = self.create_subscription(Imu, hyperparams.imu_topic_name_, self.imu_callback, self.queue_size)
        self.segmentation_subscriber = self.create_subscription(
            UInt8MultiArray, 
            '/segmentation', 
            self.segmentation_callback, 
            self.queue_size)

        #finished is episode has ended. complete is if aqua has reached the goal 
        self.finished = False

        self.roll_pid = AnglePID(target = 0.0, gains = self.roll_gains)
        self.measured_roll_angle = None
        self.relative_depth = None
        
        #dqn controller for yaw and pitch
        self.yaw_actions = np.linspace(-self.yaw_limit, self.yaw_limit, self.yaw_action_space)
        self.pitch_actions = np.linspace(-self.pitch_limit, self.pitch_limit, self.pitch_action_space)
        self.dqn = DQN(int(self.yaw_action_space * self.pitch_action_space), self.history_size) 
        self.action = torch.tensor([[4]], device=self.dqn.device, dtype=torch.long)
        self.image_history = []
        self.depth_history = []
        self.action_history = [4]
        self.episode_rewards = []
        self.state = None
        self.next_state = None
        self.state_depths = None
        self.next_state_depths = None
        self.state_actions = None
        self.next_state_actions = None
        self.reward = None
        self.erm = ReplayMemory(5000)

        #stopping condition for empty vision input
        self.empty_state_counter = 0
        self.duration = 0

        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.episode = int(len(os.listdir(self.save_path))/2)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.dqn.device)
        self.dqn.policy_net.load_state_dict(checkpoint['model_state_dict_policy'], strict=True)
        self.dqn.target_net.load_state_dict(checkpoint['model_state_dict_target'], strict=True)
        self.dqn.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.dqn.steps_done = checkpoint['training_steps']
        self.get_logger().info(f'Weights loaded successfully')

        # print('Weights loaded from {}, training steps completed: {}'.format(self.checkpoint_path, self.dqn.steps_done ))

        #initialize command
        self.command = Command()
        self.command.speed = 0.0 
        self.command.roll = 0.0
        self.command.pitch = 0.0
        self.command.yaw = 0.0
        self.command.heave = 0.0

        self.srv_start = self.create_service(Empty, '/mrl/controller/start', self.tag_start_callback)
        self.srv_stop = self.create_service(Empty, '/mrl/controller/stop', self.tag_stop_callback)
        self.start_flag = False

        self.record_bags_from_node = False
        self.srv_record_start = self.create_client(RecordTopics, '/ramius/recorder/start_recording')
        self.srv_record_stop = self.create_client(Empty, '/ramius/recorder/stop_recording')
        self.req_record_start = RecordTopics.Request()
        self.req_record_stop = Empty.Request()
        self.get_logger().info(f'Initialized: dqn barbados')
  

    def tag_start_callback(self, request, response):
        if not self.start_flag:
            self.get_logger().info(f'Recieved start tag')
            self.start_flag = True
            
            if self.record_bags_from_node:
                self.req_record_start.bag_name = self.bagfile_prefix + '_episode_{}_'.format(str(self.episode))
                self.req_record_start.bag_type = 'mcap'
                self.req_record_start.topic_names = ["camera/back/image_raw/compressed", "imu/filtered_data", "depth"]
                self.srv_record_start.call_async(self.req_record_start)
                sleep(2)
                self.get_logger().info(f'Starting ros bag record')
        return response

    def tag_stop_callback(self, request, response):
        if self.start_flag:
            self.get_logger().info(f'Recieved stop tag')
            self.finished = True
            return response
    
    def imu_callback(self, imu):
        if self.start_flag and not self.finished:
            self.measured_roll_angle = self.calculate_roll(imu)
        return

    def calculate_roll(self, imu):
        roll, _, _ = euler_from_quaternion(imu.orientation)
        return roll * 180/np.pi
    
    def depth_callback(self, depth):
        if self.start_flag and not self.finished:
            self.relative_depth = self.target_depth + depth.pose.pose.position.z
            if np.abs(self.relative_depth) > 2: 
                self.get_logger().info(f'Drifted outside depth range')
                self.finished = True
        return
      
    def segmentation_callback(self, seg_map):
        if self.start_flag:

            #if episode is not finished
            if not self.finished:

                #exit if depth has not been measured
                if self.relative_depth is None or self.measured_roll_angle is None:
                    return 
            
                seg_map = np.array(seg_map.data).reshape(self.img_size)
                #check for empty input from vision module
                if seg_map.sum() < self.detection_threshold:
                    self.empty_state_counter += 1
                else:
                    self.empty_state_counter = 0
            
                #if nothing has been detected in empty_state_max frames then reset
                if self.empty_state_counter >= self.empty_state_max:
                    self.get_logger().info(f'Nothing detected in state space for x states')
                    self.finished = True
                    return
            
                if self.duration > self.eval_duration:
                    self.get_logger().info(f'Duration Reached')
                    self.finished = True
                    return
                self.duration += 1

                self.depth_history.append(self.relative_depth)
                self.image_history.append(seg_map)
                if len(self.image_history) == self.history_size and len(self.depth_history) == self.history_size and len(self.action_history) == self.history_size:
                    ns = np.array(self.image_history)
                    nsd = np.array(self.depth_history)
                    nsa = np.array(self.action_history)

                    self.next_state = torch.tensor(ns, dtype=torch.float32, device=self.dqn.device).unsqueeze(0)
                    self.next_state_depths = torch.tensor(nsd, dtype=torch.float32, device=self.dqn.device).unsqueeze(0)
                    self.next_state_actions = torch.tensor(nsa, dtype=torch.float32, device=self.dqn.device).unsqueeze(0)

                    reward = reward_calculation(seg_map, self.relative_depth, self.roi_detection_threshold, self.mean_importance)
                    self.episode_rewards.append(reward)
                    self.reward = torch.tensor([reward], dtype=torch.float32, device=self.dqn.device)
                    if self.state is not None and self.state_depths is not None and self.state_actions is not None:
                        self.erm.push(self.state, self.state_depths, self.state_actions, self.action, self.next_state, self.next_state_depths, self.next_state_actions, self.reward)
                
                    self.action = self.dqn.select_eval_action(self.next_state, self.next_state_depths, self.next_state_actions)
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
            else:
                self.finish() 

    def finish(self):
        for _ in range(10):
            self.command.speed = 0.0
            self.command.roll = 0.0
            self.command.pitch = 0.0
            self.command.yaw = 0.0
            self.command.heave = 0.0
            self.command_publisher.publish(self.command)

        if self.record_bags_from_node:
            self.srv_record_stop.call_async(self.req_record_stop)
            self.get_logger().info(f'Stopping ros bag record')

        self.episode_rewards = np.array(self.episode_rewards)
        # print('Episode rewards. Average: ', np.mean(self.episode_rewards), ' Sum: ', np.sum(self.episode_rewards))

        with open(self.save_path + '/episode_{}.npy'.format(str(self.episode)), 'wb') as f:
            np.save(f, self.episode_rewards)
        torch.save({
                    'memory': self.erm
                }, self.save_path + '/episode_{}.pt'.format(str(self.episode)))
        
        self.get_logger().info(f'Episode ended gracefully and saved data. Resetting.')

        self.measured_roll_angle = None
        self.relative_depth = None
        self.roll_pid = AnglePID(target = 0.0, gains = self.roll_gains)
        
        self.episode_rewards = []
        self.episode += 1
        
        #reset history queue
        self.image_history = []
        self.depth_history = []
        self.action_history = [4]
        self.action = torch.tensor([[4]], device=self.dqn.device, dtype=torch.long)
        self.state = None
        self.next_state = None
        self.state_depths = None
        self.next_state_depths = None
        self.state_actions = None
        self.next_state_actions = None
        self.reward = None
        
        #reset counters
        self.empty_state_counter = 0
        self.duration = 0

        #reset end conditions 
        self.finished = False
        self.start_flag = False

        return


    
def main(args=None):
    rclpy.init(args=args)

    node = dqn_barbados()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

    



import rclpy
import torch
import numpy as np 
import os
from rclpy.node import Node
from aqua2_interfaces.msg import Command, AquaPose
from ir_aquasim_interfaces.srv import SetPosition
from geometry_msgs.msg import Pose
from std_msgs.msg import UInt8MultiArray, Float32
from time import sleep
from aqua_rl.control.PID import AnglePID
from aqua_rl.control.DQN import DQN
from aqua_rl.helpers import reward_calculation
from aqua_rl import hyperparams

class dqn_controller_eval(Node):
    def __init__(self):
        super().__init__('dqn_controller_eval')

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
        self.finish_line_x = hyperparams.finish_line_
        self.start_line_x = hyperparams.starting_line_
        self.experiment_number = hyperparams.experiment_number_
        self.eval_episode = hyperparams.eval_episode_
        self.eval_for = hyperparams.eval_for_
        self.detection_threshold = hyperparams.detection_threshold_
        self.frames_to_skip = hyperparams.frames_to_skip_
        self.roi_detection_threshold = hyperparams.roi_detection_threshold_
        self.mean_importance = hyperparams.mean_importance_

        #subscribers and publishers
        self.command_publisher = self.create_publisher(Command, '/a13/command', self.queue_size)
        self.imu_subscriber = self.create_subscription(AquaPose, '/aqua/pose', self.imu_callback, self.queue_size)
        self.segmentation_subscriber = self.create_subscription(
            UInt8MultiArray, 
            '/segmentation', 
            self.segmentation_callback, 
            self.queue_size)
        self.depth_subscriber = self.create_subscription(Float32, '/aqua/depth', self.depth_callback, self.queue_size)

        #flush queues
        self.flush_steps = self.queue_size + 30
        self.flush_commands = self.flush_steps
        self.zero_command_steps = int(self.flush_commands / 5)
        self.zero_commands = 0
        self.flush_imu = 0
        self.flush_segmentation = 0
        self.flush_depth = 0

        #finished is episode has ended. complete is if aqua has reached the goal 
        self.finished = False
        self.complete = False

        self.roll_pid = AnglePID(target = 0.0, gains = self.roll_gains, reverse=True)
        self.measured_roll_angle = 0.0
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

        #stopping condition for empty vision input
        self.empty_state_counter = 0

        if self.eval_episode == -1:
            self.checkpoint_path = 'src/aqua_rl/experiments/{}/best.pt'.format(str(self.experiment_number))
            self.save_path = 'src/aqua_rl/evaluations/{}_episode_best/'.format(str(self.experiment_number))
        else:
            self.checkpoint_path = 'src/aqua_rl/experiments/{}/weights/episode_{}.pt'.format(str(self.experiment_number), str(self.eval_episode).zfill(5))
            self.save_path = 'src/aqua_rl/evaluations/{}_episode_{}/'.format(str(self.experiment_number), str(self.eval_episode).zfill(5))
        
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.dqn.device)
        print(checkpoint)
        self.dqn.policy_net.load_state_dict(checkpoint['model_state_dict_policy'], strict=True)
        self.dqn.target_net.load_state_dict(checkpoint['model_state_dict_target'], strict=True)
        self.dqn.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.dqn.steps_done = checkpoint['training_steps']
        self.episode = 0
        
        if self.eval_episode == -1:
            print('Weights loaded from best episode, training steps completed: ', self.dqn.steps_done)
        else:
            print('Weights loaded from episode: ', self.eval_episode, ', training steps completed: ', self.dqn.steps_done)

        #initialize command
        self.command = Command()
        self.command.speed = 0.0 
        self.command.roll = 0.0
        self.command.pitch = 0.0
        self.command.yaw = 0.0
        self.command.heave = 0.0
               
        #trajectory recording
        self.trajectory = []

        #reset command
        self.reset_client = self.create_client(SetPosition, '/simulator/set_position')
        self.reset_req = SetPosition.Request()
        print('Initialized: dqn controller eval')
  
    def imu_callback(self, imu):
        
        #finished flag
        if self.finished:
            return
        
        #flush queue
        if self.flush_imu < self.flush_steps:
            self.flush_imu += 1
            return
        
        self.measured_roll_angle = self.calculate_roll(imu)

        if imu.x > self.finish_line_x:
            print('Reached finish line')
            self.flush_commands = 0
            self.finished = True
            self.complete = True
        if imu.x < self.start_line_x:
            print('Tracked backwards to starting line')
            self.flush_commands = 0
            self.finished = True
            self.complete = True
        else:
            self.trajectory.append([imu.x, imu.y, imu.z])
        return
    
    def depth_callback(self, depth):
        
        #finished flag
        if self.finished:
            return
        
        #flush queue
        if self.flush_depth < self.flush_steps:
            self.flush_depth += 1
            return
        
        if -depth.data < self.depth_range[1]:
            print('Drifted close to seabed')
            self.flush_commands = 0
            self.finished = True
            self.complete = False
        elif -depth.data > self.depth_range[0]:
            print('Drifted far above target')
            self.flush_commands = 0
            self.finished = True
            self.complete = False
        else:
            self.relative_depth = self.target_depth + depth.data
        return
    
    def calculate_roll(self, imu):
        return imu.roll
       
    def segmentation_callback(self, seg_map):

        #exit if depth has not been measured
        if self.relative_depth is None:
            return 
        
        #flush image queue
        if self.flush_segmentation < self.flush_steps:
            self.flush_segmentation += 1
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
                self.zero_commands += 1
            self.flush_commands += 1
            return
        
        #if finished, reset simulation
        if self.finished:
            self.finish()
            return

        seg_map = np.array(seg_map.data).reshape(self.img_size)
        #check for empty input from vision module
        if seg_map.sum() < self.detection_threshold:
            self.empty_state_counter += 1
        else:
            self.empty_state_counter = 0
        
        #if nothing has been detected in empty_state_max frames then reset
        if self.empty_state_counter >= self.empty_state_max:
            print("Nothing detected in state space for {} states".format(str(self.empty_state_max)))
            self.flush_commands = 0
            self.finished = True
            self.complete = False
            return
        
        self.depth_history.append(self.relative_depth)
        self.image_history.append(seg_map)
        if len(self.image_history) == self.history_size and len(self.depth_history) == self.history_size and len(self.action_history) == self.history_size:
            s = np.array(self.image_history)
            sd = np.array(self.depth_history)
            sa = np.array(self.action_history)
            
            state = torch.tensor(s, dtype=torch.float32, device=self.dqn.device).unsqueeze(0)
            state_depths = torch.tensor(sd, dtype=torch.float32, device=self.dqn.device).unsqueeze(0)
            state_actions = torch.tensor(sa, dtype=torch.float32, device=self.dqn.device).unsqueeze(0)

            reward = reward_calculation(seg_map, self.relative_depth, self.roi_detection_threshold, self.mean_importance)
            self.episode_rewards.append(reward)
            self.action = self.dqn.select_eval_action(state, state_depths, state_actions)

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

    def finish(self):
        
        if self.complete:
            reward = hyperparams.goal_reached_reward_
            self.episode_rewards.append(reward)
        else:
            reward = hyperparams.goal_not_reached_reward_
            self.episode_rewards.append(reward)

        self.episode_rewards = np.array(self.episode_rewards)
        print('Episode rewards. Average: ', np.mean(self.episode_rewards), ' Sum: ', np.sum(self.episode_rewards))

        with open(self.save_path + '/episode_{}.npy'.format(str(self.episode).zfill(5)), 'wb') as f:
            np.save(f, self.episode_rewards)
            np.save(f, np.array(self.trajectory))
        
        if self.episode < self.eval_for:
            self.reset()
        else:
            rclpy.shutdown()
        return
    
    def reset(self):
        print('-------------- Resetting simulation --------------')

        starting_pose = Pose()

        #starting position
        starting_pose.position.x = 70.0
        starting_pose.position.z = -0.3
        starting_pose.position.y = self.target_depth

        #starting orientation
        starting_pose.orientation.x = 0.0
        starting_pose.orientation.y = -0.7071068
        starting_pose.orientation.z = 0.0
        starting_pose.orientation.w = 0.7071068
        self.reset_req.pose = starting_pose
        self.reset_client.call_async(self.reset_req)
        sleep(0.5)

        #reset pid controllers
        self.measured_roll_angle = 0.0
        self.relative_depth = None
        self.roll_pid = AnglePID(target = 0.0, gains = self.roll_gains, reverse=True)

        #increment episode and reset rewards
        self.episode_rewards = []
        self.episode += 1
        
        #reset trajectory
        self.trajectory = []
        
        #reset history queue
        self.image_history = []
        self.depth_history = []
        self.action_history = [4]
        self.action = torch.tensor([[4]], device=self.dqn.device, dtype=torch.long)

        #reset flush queues 
        self.flush_commands = self.flush_steps
        self.zero_commands = 0
        self.flush_imu = 0
        self.flush_segmentation = 0
        self.flush_depth = 0

        #reset counters
        self.empty_state_counter = 0

        #reset end conditions 
        self.finished = False
        self.complete = False
        
        return
    
def main(args=None):
    rclpy.init(args=args)

    node = dqn_controller_eval()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

    



import rclpy
from rclpy.node import Node
from aqua2_interfaces.msg import Command, AquaPose
from ir_aquasim_interfaces.srv import EmptyWorkaround
from aqua_rl.control.PID import AnglePID
from aqua_rl.control.DQN import DQN
from std_msgs.msg import UInt8MultiArray
import numpy as np 
import os
from time import sleep
import torch


class dqn_controller_eval(Node):
    def __init__(self):
        super().__init__('dqn_controller_eval')

        #subscribers and publishers
        self.queue_size = 5
        self.command_publisher = self.create_publisher(Command, '/a13/command', self.queue_size)
        self.imu_subscriber = self.create_subscription(AquaPose, '/aqua/pose', self.imu_callback, self.queue_size)
        self.pipeline_segmentation_subscriber = self.create_subscription(
            UInt8MultiArray, 
            '/pipeline/segmentation', 
            self.pipeline_segmentation_callback, 
            self.queue_size)
        
        #flush queues
        self.flush_steps = self.queue_size + 30
        self.flush_commands = self.flush_steps
        self.flush_imu = 0
        self.flush_segmentation = 0

        #finished is episode has ended. complete is if aqua has reached the goal 
        self.finished = False
        self.complete = False

        self.roll_gains = [0.25, 0.0, 0.75]
        self.roll_pid = AnglePID(target = 0.0, gains = self.roll_gains, reverse=True)
        self.measured_roll_angle = 0.0
        
        #dqn controller for yaw and pitch 
        self.yaw_action_space = 3
        self.pitch_action_space = 3
        self.history_size = 10
        self.yaw_actions = np.linspace(-0.25, 0.25, self.yaw_action_space)
        self.pitch_actions = np.linspace(-0.005, 0.005, self.pitch_action_space)
        self.dqn = DQN(int(self.yaw_action_space * self.pitch_action_space), self.history_size) 
        self.num_eval_episodes = 10
        self.reward = None
        self.history_queue = []
        self.episode_rewards = []

        #target for reward
        self.img_size = (32, 32)
        self.template = np.zeros(self.img_size)
        half = int(self.img_size[0]/2)
        self.template[:,half-2:half+2] = 1
        self.template = self.template.astype(np.uint8)

        self.checkpoint_experiment = 0
        self.checkpoint_episode = 180
        self.checkpoint_path = 'src/aqua_rl/aqua_rl/checkpoints/dqn/{}/episode_{}.pt'.format(str(self.checkpoint_experiment), str(self.checkpoint_episode).zfill(5))
        self.save_path = 'src/aqua_rl/aqua_rl/evaluations/{}_episode_{}/'.format(str(self.checkpoint_experiment), str(self.checkpoint_episode).zfill(5))
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.dqn.device)
        self.dqn.policy_net.load_state_dict(checkpoint['model_state_dict_policy'], strict=True)
        self.dqn.target_net.load_state_dict(checkpoint['model_state_dict_target'], strict=True)
        self.dqn.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.dqn.memory.memory = checkpoint['memory']
        self.dqn.steps_done = checkpoint['training_steps']
        self.eval_episode = 0
        print('Weights loaded from episode: ', self.checkpoint_episode, ', training steps completed: ', self.dqn.steps_done)

        #initialize command
        self.command = Command()
        self.command.speed = 0.0 
        self.command.roll = 0.0
        self.command.pitch = 0.0
        self.command.yaw = 0.0
        self.command.heave = 0.0
               
        #trajectory recording
        self.trajectory = []

        #target trajectory
        self.target_trajectory = 'src/aqua_rl/aqua_rl/trajectories/targets/rope_center.npy'
        with open(self.target_trajectory, 'rb') as f:
            self.rope_x = np.load(f) 
            self.rope_y = np.load(f)
            self.rope_z = np.load(f)
        
        #define max error to target trajectory
        self.max_error_line = (1.225, 18.55)
        self.depth_range = [-6, -14.5]

        #end of trajectory
        self.finish_line_x = 25

        #reset command
        self.reset_client = self.create_client(EmptyWorkaround, '/simulator/reset_robot')
        self.reset_req = EmptyWorkaround.Request()

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
        m, b = self.get_target_line(imu.x)
        dist_to_line = (m*imu.x - imu.z + b) / np.sqrt(np.square(m) + 1)
        if imu.x > self.finish_line_x:
            self.flush_commands = 0
            self.complete = True
            self.finished = True
        elif imu.y < self.depth_range[1]:
            print('Drifted close to seabed')
            self.flush_commands = 0
            self.finished = True
            self.complete = False
        elif imu.y > self.depth_range[0]:
            print('Drifted far above target')
            self.flush_commands = 0
            self.finished = True
            self.complete = False
        elif np.abs(dist_to_line) > self.get_max_error_from_depth(imu.y):
            print('Drifted far from target z trajectory')
            self.flush_commands = 0
            self.finished = True
            self.complete = False
        else:
            self.trajectory.append([imu.x, imu.y, imu.z])
        return
    
    def calculate_roll(self, imu):
        return imu.roll
    
    def get_max_error_from_depth(self, depth):
        return self.max_error_line[0]*depth + self.max_error_line[1]
       
       
    def get_target_line(self, x):
        ind = np.argwhere(self.rope_x >= x)[0][0]
        x1 = self.rope_x[ind - 1]
        z1 = self.rope_z[ind - 1]
        x2 = self.rope_x[ind]
        z2 = self.rope_z[ind]
        m = (z2 - z1) / (x2 - x1)
        b = z2 - m * x2
        return m, b
    
    def pipeline_segmentation_callback(self, seg_map):
        
        #flush image queue
        if self.flush_segmentation < self.flush_steps:
            self.flush_segmentation += 1
            return

        #flush out command queue
        if self.flush_commands < self.flush_steps:
            self.flush_commands += 1
            return
        
        #if finished, reset simulation
        if self.finished:
            self.finish()
            return

        seg_map = np.array(seg_map.data).reshape(self.img_size)
        if len(self.history_queue) < self.history_size:
            self.history_queue.append(seg_map)
        else:
            self.history_queue.pop()
            self.history_queue.append(seg_map)
            s = np.array(self.history_queue)
            state = torch.tensor(s, dtype=torch.float32, device=self.dqn.device).unsqueeze(0)
            reward = self.reward_calculation(seg_map)
            self.episode_rewards.append(reward)
            action = self.dqn.select_eval_action(state)

            action_idx = action.detach().cpu().numpy()[0][0]
            # print(action_idx, '->', (int(action_idx/3), action_idx % 3))
            self.command.pitch = self.pitch_actions[int(action_idx/3)]
            self.command.yaw = self.yaw_actions[action_idx % 3]
            
            self.command.speed = 0.25 #fixed speed
            self.command.roll = self.roll_pid.control(self.measured_roll_angle)
            self.command_publisher.publish(self.command)
        return 
    
    def reward_calculation(self, seg_map):
        # Calculate intersection and union
        intersection = np.logical_and(seg_map, self.template)
        union = np.logical_or(seg_map, self.template)
        iou = np.sum(intersection) / np.sum(union)
        return iou - 0.025
        

    def finish(self):
        if self.complete:
            print('Goal reached')
            reward = 10
            self.episode_rewards.append(reward)
        else:
            print('Goal not reached')
            reward = -10
            self.episode_rewards.append(reward)

        self.episode_rewards = np.array(self.episode_rewards)
        print('Episode rewards. Average: ', np.mean(self.episode_rewards), ' Sum: ', np.sum(self.episode_rewards))

        print('Saving trajectory')
        with open(self.save_path + '/episode_{}.npy'.format(str(self.eval_episode).zfill(5)), 'wb') as f:
            np.save(f, self.episode_rewards)
            np.save(f, np.array(self.trajectory))
        
        if self.eval_episode < self.num_eval_episodes:
            self.reset()
        else:
            rclpy.shutdown()
        return
    
    def reset(self):
        print('-------------- Resetting simulation --------------')

        self.reset_client.call_async(self.reset_req)
        sleep(0.5)

        #reset pid controllers
        self.measured_roll_angle = 0.0
        self.roll_pid = AnglePID(target = 0.0, gains = self.roll_gains, reverse=True)

        #increment episode and reset rewards
        self.episode_rewards = []
        self.eval_episode += 1
        
        #reset trajectory
        self.trajectory = []
        
        #reset history queue
        self.history_queue = []

        #reset flush queues 
        self.flush_commands = self.flush_steps
        self.flush_imu = 0
        self.flush_segmentation = 0

        #reset end conditions 
        self.finished = False
        self.complete = False
        
        print('-------------- Finished reset --------------')

        return
    
def main(args=None):
    rclpy.init(args=args)

    node = dqn_controller_eval()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

    



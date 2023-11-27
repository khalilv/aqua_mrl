import rclpy
from rclpy.node import Node
from aqua2_interfaces.msg import Command, AquaPose
from ir_aquasim_interfaces.srv import EmptyWorkaround
from aqua_pipeline_inspection.control.PID import AnglePID, PID
from aqua_pipeline_inspection.control.DQN import DQN
from std_msgs.msg import UInt8MultiArray
import numpy as np 
import os
from time import sleep
import torch


class dqn_controller(Node):
    def __init__(self):
        super().__init__('dqn_controller')

        #subscribers and publishers
        self.command_publisher = self.create_publisher(Command, '/a13/command', 5)
        self.imu_subscriber = self.create_subscription(AquaPose, '/aqua/pose', self.imu_callback, 5)
        self.pipeline_segmentation_subscriber = self.create_subscription(
            UInt8MultiArray, 
            '/pipeline/segmentation', 
            self.pipeline_segmentation_callback, 
            5)
        # self.depth_subscriber = self.create_subscription(Float32, '/aqua/depth', self.depth_callback, 10)

        #initialize pid controllers
        # self.target_depth = 10.0
        self.roll_pid = AnglePID(target = 0.0, gains = [0.25, 0.0, 1.75], reverse=True)
        self.measured_roll_angle = 0.0
        
        #dqn controller for yaw and pitch 
        self.yaw_action_space = 5
        self.pitch_action_space = 3
        self.history_size = 5
        self.yaw_actions = np.linspace(-0.4, 0.4, self.yaw_action_space)
        self.pitch_actions = np.linspace(-0.005, 0.005, self.pitch_action_space)
        self.dqn = DQN(int(self.yaw_action_space * self.pitch_action_space), self.history_size) 
        self.num_episodes = 600
        self.state = None
        self.next_state = None
        self.action = None
        self.reward = None
        self.history_queue = []
        self.episode_return = 0

        #target for reward
        self.template = np.zeros((84,84))
        self.template[:,38:45] = 1
        self.template = self.template.astype(np.uint8)

        self.root_path = 'src/aqua_pipeline_inspection/aqua_pipeline_inspection/trajectories/dqn/'
        self.checkpoint_experiment = 1
        try:
            self.save_path = os.path.join(self.root_path, str(self.checkpoint_experiment))
            eps = len(os.listdir(self.save_path)) - 1
            checkpoint_path = self.save_path + '/episode_' + str(eps) + '.pt'
            checkpoint = torch.load(checkpoint_path, map_location=self.dqn.device)
            self.dqn.policy_net.load_state_dict(checkpoint['model_state_dict_policy'], strict=True)
            self.dqn.target_net.load_state_dict(checkpoint['model_state_dict_target'], strict=True)
            self.dqn.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.dqn.memory.memory = checkpoint['memory']
            self.dqn.steps_done = checkpoint['training_steps']
            self.episode = eps + 1
            print('Weights loaded: starting from episode ', self.episode)
        except:
            print('No checkpoint found: starting from episode 0')
            self.new_checkpoint_experiment = len(os.listdir(self.root_path))
            os.mkdir(os.path.join(self.root_path, str(self.new_checkpoint_experiment)))
            self.save_path = os.path.join(self.root_path, str(self.new_checkpoint_experiment))
            self.episode = 0


        #initialize command
        self.command = Command()
        self.command.speed = 0.0 
        self.command.roll = 0.0
        self.command.pitch = 0.0
        self.command.yaw = 0.0
        self.command.heave = 0.0
        
        self.img_size = (300, 400)
        
        #trajectory recording
        self.trajectory = []

        #target trajectory
        self.target_trajectory = 'src/aqua_pipeline_inspection/aqua_pipeline_inspection/trajectories/targets/rope_center.npy'
        with open(self.target_trajectory, 'rb') as f:
            self.rope_x = np.load(f) 
            self.rope_y = np.load(f)
            self.rope_z = np.load(f)
        
        #define max error to target trajectory
        self.max_error_to_target = 4.0 #if distance to pipeline > max => halt

        #end of pipe
        self.finish_line_x = 25

        #reset command
        self.reset_client = self.create_client(EmptyWorkaround, '/simulator/reset_robot')
        self.reset_req = EmptyWorkaround.Request()
        print('Initialized: dqn controller')
  
    def imu_callback(self, imu):
        self.measured_roll_angle = self.calculate_roll(imu)
        zt, yt = self.get_target(imu.x)
        dist = np.sqrt(np.square(imu.z - zt) + np.square(imu.y - yt))
        if imu.x > self.finish_line_x:
            self.finish(True)
        elif dist > self.max_error_to_target:
            print('Drifted far from target trajectory')
            self.finish(False)
        elif imu.y < - 14.5:
            print('Hit seabed')
            self.finish(False)
        else:
            self.trajectory.append([imu.x, imu.y, imu.z])
        return
    
    # def depth_callback(self, depth):
    #     self.measured_depth = depth.data
    #     if np.abs(self.measured_depth - self.target_depth) > self.max_y_error_to_target:
    #         print('Drifted far from target trajectory in y direction')
    #         self.finish(False)
    #     return

    def calculate_roll(self, imu):
        return imu.roll
    
    def calculate_pitch(self, imu):
        return imu.pitch
    
    def get_target(self, x):
        ind = np.argwhere(self.rope_x >= x)[0][0]
        x1 = self.rope_x[ind - 1]
        z1 = self.rope_z[ind - 1]
        x2 = self.rope_x[ind]
        z2 = self.rope_z[ind]
        m = (z2 - z1) / (x2 - x1)
        b = z2 - m * x2
        return m*x + b, self.rope_y[ind]
    
    def pipeline_segmentation_callback(self, seg_map):
        seg_map = np.array(seg_map.data).reshape((84,84))
        if len(self.history_queue) < self.history_size:
            self.history_queue.append(seg_map)
        else:
            self.history_queue.pop()
            self.history_queue.append(seg_map)
            s = np.array(self.history_queue)
            self.next_state = torch.tensor(s, dtype=torch.float32, device=self.dqn.device).unsqueeze(0)
            reward = self.reward_calculation(seg_map)
            print(reward)
            self.episode_return += reward
            self.reward = torch.tensor([reward], device=self.dqn.device)
            self.action = self.dqn.select_action(self.next_state)
            if self.state is not None:
                self.dqn.memory.push(self.state, self.action, self.next_state, self.reward)
        
            self.state = self.next_state
            # Perform one step of the optimization (on the policy network)
            self.dqn.optimize()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = self.dqn.target_net.state_dict()
            policy_net_state_dict = self.dqn.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key]*self.dqn.TAU + target_net_state_dict[key]*(1-self.dqn.TAU)
            self.dqn.target_net.load_state_dict(target_net_state_dict)
            
            action_idx = self.action.detach().cpu().numpy()[0][0]
            # print(action_idx, '->', (int(action_idx/5), action_idx % 5))
            self.command.pitch = self.pitch_actions[int(action_idx/5)]
            self.command.yaw = self.yaw_actions[action_idx % 5]
            
            self.command.speed = 0.25 #fixed speed
            self.command.roll = self.roll_pid.control(self.measured_roll_angle)
            self.command_publisher.publish(self.command)
        return 
    
    def reward_calculation(self, seg_map):
        bit_and = np.bitwise_and(self.template, seg_map)
        score = np.sum(bit_and) / np.sum(self.template)
        return score - 0.1
        

    def finish(self, complete):
        if complete:
            print('Goal reached')
            reward = 10
            self.episode_return += reward
        else:
            print('Goal not reached')
            reward = 0 #-10
            self.episode_return += reward
        print('Episode return: ', self.episode_return)
        if self.state is not None:
            self.dqn.memory.push(self.state, self.action, None, torch.tensor([reward], device=self.dqn.device))
        
        print('Saving trajectory')       
        torch.save({
            'training_steps': self.dqn.steps_done,
            'episode_returns': self.episode_return,
            'model_state_dict_policy': self.dqn.policy_net.state_dict(),
            'model_state_dict_target': self.dqn.target_net.state_dict(),
            'optimizer_state_dict': self.dqn.optimizer.state_dict(),
            'memory': self.dqn.memory.memory,
            'trajectory': np.array(self.trajectory)
            }, self.save_path +  '/episode_{}.pt'.format(str(self.episode)))

        if self.episode < self.num_episodes:
            self.reset()
        else:
            rclpy.shutdown()
        return
    
    def reset(self):
        print('-------------- Resetting simulation --------------')

        self.reset_client.call_async(self.reset_req)
        sleep(0.5)

        #reset errors
        self.measured_roll_angle = 0.0
        self.roll_pid = AnglePID(target = 0.0, gains = [0.75, 0.0, 3.75], reverse=True)

        #increment episode and reset returns
        self.episode_return = 0
        self.episode += 1
        
        #reset trajectory
        self.trajectory = []
        self.state = None
        self.history_queue = []

        #reinitialize subscribers and publishers
        self.command_publisher = self.create_publisher(Command, '/a13/command', 5)
        self.imu_subscriber = self.create_subscription(AquaPose, '/aqua/pose', self.imu_callback, 5)
        self.pipeline_segmentation_subscriber = self.create_subscription(
            UInt8MultiArray, 
            '/pipeline/segmentation', 
            self.pipeline_segmentation_callback, 
            5)
        
        return
    
def main(args=None):
    rclpy.init(args=args)

    node = dqn_controller()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

    



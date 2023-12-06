import rclpy
from rclpy.node import Node
from aqua2_interfaces.msg import Command, AquaPose
from ir_aquasim_interfaces.srv import SetPosition
from geometry_msgs.msg import Pose
from aqua_rl.control.PID import AnglePID
from aqua_rl.control.DQN import DQN
from std_msgs.msg import UInt8MultiArray
import numpy as np 
import os
from time import sleep
import torch


class dqn_controller(Node):
    def __init__(self):
        super().__init__('dqn_controller')

        #hyperparams
        self.queue_size = 5
        self.roll_gains = [0.25, 0.0, 0.75]
        self.history_size = 20
        self.pitch_limit = 0.005
        self.yaw_limit = 0.25
        self.yaw_action_space = 3
        self.pitch_action_space = 3
        self.num_episodes = 600

        #subscribers and publishers
        self.command_publisher = self.create_publisher(Command, '/a13/command', self.queue_size)
        self.imu_subscriber = self.create_subscription(AquaPose, '/aqua/pose', self.imu_callback, self.queue_size)
        self.segmentation_subscriber = self.create_subscription(
            UInt8MultiArray, 
            '/segmentation', 
            self.segmentation_callback, 
            self.queue_size)
        
        #flush queues
        self.flush_steps = self.queue_size + 30
        self.flush_commands = self.flush_steps
        self.flush_imu = 0
        self.flush_segmentation = 0

        #finished is episode has ended. complete is if aqua has reached the goal 
        self.finished = False
        self.complete = False

        self.roll_pid = AnglePID(target = 0.0, gains = self.roll_gains, reverse=True)
        self.measured_roll_angle = 0.0
        
        #dqn controller for yaw and pitch 
        self.yaw_actions = np.linspace(-self.yaw_limit, self.yaw_limit, self.yaw_action_space)
        self.pitch_actions = np.linspace(-self.pitch_limit, self.pitch_limit, self.pitch_action_space)
        self.dqn = DQN(int(self.yaw_action_space * self.pitch_action_space), self.history_size) 
        self.state = None
        self.next_state = None
        self.action = None
        self.reward = None
        self.history_queue = []
        self.episode_rewards = []

        #trajectory recording
        self.trajectory = []
        self.save_every = 10
        self.evaluate = False 

        #target for reward
        self.img_size = (32, 32)
        self.template = np.zeros(self.img_size)
        half = int(self.img_size[0]/2)
        self.template[:,half-2:half+2] = 1
        self.template = self.template.astype(np.uint8)

        #stopping condition for empty vision input
        self.empty_state_counter = 0
        self.empty_state_max = 10

        self.root_path = 'src/aqua_rl/checkpoints/dqn/'
        self.checkpoint_experiment = 0
        try:
            self.save_path = os.path.join(self.root_path, str(self.checkpoint_experiment))
            self.traj_save_path = os.path.join(self.root_path.replace('checkpoints', 'trajectories'), str(self.checkpoint_experiment))
            eps = len(os.listdir(self.traj_save_path)) - 1
            last_checkpoint_ep = (eps // self.save_every) * self.save_every
            checkpoint_path = self.save_path + '/episode_' + str(last_checkpoint_ep).zfill(5) + '.pt'
            checkpoint = torch.load(checkpoint_path, map_location=self.dqn.device)
            self.dqn.policy_net.load_state_dict(checkpoint['model_state_dict_policy'], strict=True)
            self.dqn.target_net.load_state_dict(checkpoint['model_state_dict_target'], strict=True)
            self.dqn.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.dqn.memory.memory = checkpoint['memory']
            self.dqn.steps_done = checkpoint['training_steps']
            self.episode = eps + 1
            print('Weights loaded. starting from episode: ', self.episode, ', training steps completed: ', self.dqn.steps_done)
        except:
            print('No checkpoint found. Starting from episode 0')
            self.new_checkpoint_experiment = len(os.listdir(self.root_path))
            os.mkdir(os.path.join(self.root_path, str(self.new_checkpoint_experiment)))
            os.mkdir(os.path.join(self.root_path.replace('checkpoints', 'trajectories'), str(self.new_checkpoint_experiment)))
            self.traj_save_path = os.path.join(self.root_path.replace('checkpoints', 'trajectories'), str(self.new_checkpoint_experiment))
            self.save_path = os.path.join(self.root_path, str(self.new_checkpoint_experiment))
            self.episode = 0

        #initialize command
        self.command = Command()
        self.command.speed = 0.0 
        self.command.roll = 0.0
        self.command.pitch = 0.0
        self.command.yaw = 0.0
        self.command.heave = 0.0
               
        #target trajectory
        self.target_trajectory = 'src/aqua_rl/trajectories/targets/rope_center.npy'
        with open(self.target_trajectory, 'rb') as f:
            self.rope_x = np.load(f) 
            self.rope_y = np.load(f)
            self.rope_z = np.load(f)
        
        #define max error to target trajectory
        self.max_error_line = (1.225, 18.55)
        self.depth_range = [-6, -14.5]
        self.safety_buffer_on_restart = 2

        #end of trajectory
        self.finish_line_x = 25
        self.start_line_x = -70

        #reset command
        self.reset_client = self.create_client(SetPosition, '/simulator/set_position')
        self.reset_req = SetPosition.Request()
        print('Initialized: dqn controller')

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
            self.finished = True
            self.complete = True
        if imu.x < self.start_line_x:
            print('Drifted behind starting position')
            self.flush_commands = 0
            self.finished = True
            self.complete = False
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
        # elif np.abs(dist_to_line) > self.get_max_error_from_depth(imu.y):
        #     print('Drifted far from target z trajectory')
        #     self.flush_commands = 0
        #     self.finished = True
        #     self.complete = False
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
    
    def segmentation_callback(self, seg_map):

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
            self.history_queue.pop(0)
            self.history_queue.append(seg_map)
            s = np.array(self.history_queue)
            
            #check for empty input from vision module
            if s.sum() == 0:
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
            
            self.next_state = torch.tensor(s, dtype=torch.float32, device=self.dqn.device).unsqueeze(0)
            reward = self.reward_calculation(seg_map)
            self.episode_rewards.append(reward)
            self.reward = torch.tensor([reward], device=self.dqn.device)

            if self.evaluate:
                #select greedy action, dont optimize model or append to replay buffer
                self.action = self.dqn.select_eval_action(self.next_state)
            else:
                if self.state is not None and self.action is not None:
                    self.dqn.memory.push(self.state, self.action, self.next_state, self.reward)

                self.action = self.dqn.select_action(self.next_state)       
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

        if self.state is not None:
            self.dqn.memory.push(self.state, self.action, None, torch.tensor([reward], device=self.dqn.device))
        
        if self.episode % self.save_every == 0:
            print('Saving checkpoint')
            torch.save({
                'training_steps': self.dqn.steps_done,
                'model_state_dict_policy': self.dqn.policy_net.state_dict(),
                'model_state_dict_target': self.dqn.target_net.state_dict(),
                'optimizer_state_dict': self.dqn.optimizer.state_dict(),
                'memory': self.dqn.memory.memory
            }, self.save_path +  '/episode_{}.pt'.format(str(self.episode).zfill(5)))
        
        with open(self.traj_save_path + '/episode_{}.npy'.format(str(self.episode).zfill(5)), 'wb') as f:
            np.save(f, self.episode_rewards)
            np.save(f, np.array(self.trajectory))
        if self.episode < self.num_episodes:
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
        # starting_pose.position.y = np.random.uniform(self.depth_range[1]+self.safety_buffer_on_restart, 
        #                                              self.depth_range[0]-self.safety_buffer_on_restart)
        starting_pose.position.y = -12.0
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
        self.roll_pid = AnglePID(target = 0.0, gains = self.roll_gains, reverse=True)

        #increment episode and reset rewards
        self.episode_rewards = []
        self.episode += 1
        self.evaluate = self.episode % self.save_every == 0

        if self.evaluate:
            print('Starting evaluation')

        #reset trajectory
        self.trajectory = []
        
        #reset state and history queue
        self.state = None
        self.action = None
        self.history_queue = []

        #reset flush queues 
        self.flush_commands = self.flush_steps
        self.flush_imu = 0
        self.flush_segmentation = 0

        #reset end conditions 
        self.finished = False
        self.complete = False
        
        return
    
def main(args=None):
    rclpy.init(args=args)

    node = dqn_controller()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
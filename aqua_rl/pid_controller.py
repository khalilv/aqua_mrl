import rclpy
import torch
import numpy as np 
from rclpy.node import Node
from aqua2_interfaces.msg import AquaPose, DiverCommand
from ir_aquasim_interfaces.srv import SetPosition
from geometry_msgs.msg import Pose
from std_msgs.msg import Float32MultiArray, UInt8MultiArray, Bool
from time import sleep
from aqua_rl.control.DQN import ReplayMemory
from aqua_rl.control.PID import PID
from aqua_rl.helpers import reward_calculation, map_missing_detection, normalize_coords
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
        self.depth_range = hyperparams.depth_range_
        self.load_erm = hyperparams.load_erm_ 
        self.experiment_number = hyperparams.experiment_number_
        self.train_for = hyperparams.train_for_
        self.train_duration = hyperparams.train_duration_
        self.eval_duration = hyperparams.eval_duration_
        self.diver_max_speed = hyperparams.diver_max_speed_
        self.reward_sharpness = hyperparams.sharpness_
        self.frame_skip = hyperparams.frame_skip_
        self.empty_state_max = hyperparams.empty_state_max_
        self.yaw_gains = hyperparams.yaw_gains_
        self.pitch_gains = hyperparams.pitch_gains_
        
        #subscribers and publishers
        self.imu_subscriber = self.create_subscription(AquaPose, '/aqua/pose', self.imu_callback, self.queue_size)
        self.command_publisher = self.create_publisher(UInt8MultiArray, hyperparams.autopilot_command_, self.queue_size)
        self.autopilot_start_stop_publisher = self.create_publisher(Bool, hyperparams.autopilot_start_stop_, self.queue_size)
        self.detection_subscriber = self.create_subscription(
            Float32MultiArray, 
            '/diver/coordinates', 
            self.detection_callback, 
            self.queue_size)
        self.diver_publisher = self.create_publisher(DiverCommand, 'diver_control', self.queue_size)
        self.diver_pose_subscriber = self.create_subscription(
            AquaPose,
            hyperparams.diver_topic_name_,
            self.diver_pose_callback,
            self.queue_size)
        
        #initialize pid controllers
        self.pitch_pid = PID(target = 0, gains = self.pitch_gains, command_range=[-(self.pitch_action_space//2), self.pitch_action_space//2], reverse=True)
        self.yaw_pid = PID(target = 0, gains = self.yaw_gains, command_range=[-(self.yaw_action_space//2), self.yaw_action_space//2], reverse=True)

        self.yaw_actions = np.linspace(-(self.yaw_action_space//2), self.yaw_action_space//2, self.yaw_action_space)
        self.pitch_actions = np.linspace(-(self.pitch_action_space//2), self.pitch_action_space//2, self.pitch_action_space)
        
        #autopilot commands
        self.command = UInt8MultiArray()
        self.autopilot_flag = Bool()
        self.autopilot_flag.data = False
        
        #command
        self.diver_cmd = DiverCommand()
        self.diver_pose = None
        
        #flush queues
        self.flush_steps = self.queue_size + 35
        self.flush_imu = 0
        self.flush_diver = 0
        self.flush_detection = 0

        #state and depth history
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

        self.aqua_trajectory = []
        self.diver_trajectory = []

        self.duration = 0
        self.empty_state_counter = 0

        self.finished = False
        self.complete = False

        #reset commands
        self.reset_client = self.create_client(SetPosition, '/simulator/set_position')
        self.reset_req = SetPosition.Request()
        self.reset_diver_client = self.create_client(SetPosition, '/diver/set_position')
        self.reset_diver_req = SetPosition.Request()

        self.timer = self.create_timer(5, self.publish_diver_command)

        print('Initialized: PID controller')
  
    def imu_callback(self, imu):
        
        #finished flag
        if self.finished:
            return
        
        #flush queue
        if self.flush_imu < self.flush_steps:
            self.flush_imu += 1
            return
        
        self.aqua_trajectory.append([imu.x, imu.y, imu.z])
        return
            
    def publish_diver_command(self):     
        if self.diver_pose:

            #scale vector to current magnitude
            self.diver_cmd.vx = np.random.uniform(hyperparams.speed_, hyperparams.speed_+0.15)
            self.diver_cmd.vy = np.random.uniform(-1,1)
            self.diver_cmd.vz = np.random.uniform(-1,1)

            speed = np.sqrt(np.square(self.diver_cmd.vy) + np.square(self.diver_cmd.vz))
            if speed > self.diver_max_speed:
                self.diver_cmd.vy = self.diver_cmd.vy * self.diver_max_speed/speed
                self.diver_cmd.vz = self.diver_cmd.vz * self.diver_max_speed/speed
                
            if self.diver_pose[1] > self.depth_range[0] and self.diver_cmd.vy > 0:
                self.diver_cmd.vy *= -1
            elif self.diver_pose[1] < self.depth_range[1] and self.diver_cmd.vy < 0:
                self.diver_cmd.vy *= -1

            #publish
            self.diver_publisher.publish(self.diver_cmd)
            print('Publishing diver command')

            return 
    

    def diver_pose_callback(self, pose):    
        
        #finished flag
        if self.finished:
            return
        
        #flush queue
        if self.flush_diver < self.flush_steps:
            self.flush_diver += 1
            return 
           
        self.diver_pose = [pose.x, pose.y, pose.z]
        self.diver_trajectory.append(self.diver_pose)

        return 
        
    def detection_callback(self, coords):
        
        #flush detections queue
        if self.flush_detection< self.flush_steps:
            self.flush_detection += 1
            return
        
        if not self.autopilot_flag.data:
            print('starting autopilot')
            self.autopilot_flag.data = True
            self.autopilot_start_stop_publisher.publish(self.autopilot_flag)

        #if finished, reset simulation
        if self.finished:
            self.finish()
            return
        
        coords = np.array(coords.data)
        
        #check for null input from detection module
        if coords[0] == -1 and coords[1] == -1 and coords[2] == -1 and coords[3] == -1:
            self.empty_state_counter += 1
            detected_center = map_missing_detection(self.history[-1][0],self.history[-1][1])
        else:
            self.empty_state_counter = 0
            yc = (coords[1] + coords[3])/2
            xc = (coords[0] + coords[2])/2
            detected_center = normalize_coords(yc, xc, self.img_size, self.img_size)
        
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
        
        self.history.append(detected_center)
        if len(self.history) == self.history_size:
            ns = np.array(self.history).flatten()
            self.next_state = torch.tensor(ns, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            pitch_reward, yaw_reward = reward_calculation(detected_center[0], detected_center[1], self.reward_sharpness)
            self.episode_rewards.append([pitch_reward, yaw_reward])
            self.pitch_reward = torch.tensor([pitch_reward], dtype=torch.float32, device=self.device)
            self.yaw_reward = torch.tensor([yaw_reward], dtype=torch.float32, device=self.device)
            
            if self.state is not None:
                self.erm.push(self.state, self.pitch_action, self.yaw_action, self.next_state, self.pitch_reward, self.yaw_reward)
            
            self.pitch_action = self.discretize(self.pitch_pid.control(detected_center[0]), self.pitch_actions)
            self.yaw_action = self.discretize(self.yaw_pid.control(detected_center[1]), self.yaw_actions)  
            self.state = self.next_state
               
            #publish action
            self.command.data = [int(self.pitch_action), int(self.yaw_action)]
            self.command_publisher.publish(self.command)
            self.history = self.history[self.frame_skip:]
        return 
    
    def discretize(self, v, l):
        index = np.argmin(np.abs(np.subtract(l,v)))
        return index
    
    def finish(self):

        #stop autopilot
        print('stopping autopilot')
        self.autopilot_flag.data = False
        self.autopilot_start_stop_publisher.publish(self.autopilot_flag)
        sleep(3.5)
         
        self.episode_rewards = np.array(self.episode_rewards)
        print('Episode rewards (pitch, yaw). Average: ', np.mean(self.episode_rewards, axis=0), ' Sum: ', np.sum(self.episode_rewards, axis=0))
        
        if self.state is not None and not self.complete:
            self.erm.push(self.state, self.pitch_action, self.yaw_action, None, self.pitch_reward, self.yaw_reward)

        torch.save({
            'memory': self.erm
        }, 'pid_expert.pt')

        with open('pid_traj.npy', 'wb') as f:
            np.save(f, self.episode_rewards)
            np.save(f, np.array(self.aqua_trajectory))
            np.save(f, np.array(self.diver_trajectory))

        self.reset()
        return

    def reset(self):
        print('-------------- Resetting simulation --------------')
        
        #increment episode and reset rewards
        self.episode_rewards = []

        starting_pose = Pose()
        #starting position
        starting_pose.position.x = 70.0
        starting_pose.position.z = -0.3                               
        starting_pose.position.y = -10.0
        #starting orientation
        starting_pose.orientation.x = 0.0
        starting_pose.orientation.y = -0.7071068
        starting_pose.orientation.z = 0.0
        starting_pose.orientation.w = 0.7071068
        self.reset_req.pose = starting_pose
        self.reset_client.call_async(self.reset_req)

        starting_diver_pose = Pose()
        #starting position
        starting_diver_pose.position.x = 62.5
        starting_diver_pose.position.z = 0.3                               
        starting_diver_pose.position.y = -10.0
        #starting orientation
        starting_diver_pose.orientation.x = 0.4976952
        starting_diver_pose.orientation.y = -0.5022942
        starting_diver_pose.orientation.z = 0.4976952
        starting_diver_pose.orientation.w = 0.5022942
        self.reset_diver_req.pose = starting_diver_pose
        self.reset_diver_client.call_async(self.reset_diver_req)
        sleep(3.5)

        #reset trajectory
        self.aqua_trajectory = []
        self.diver_trajectory = []

        #reset state and history queues
        self.history = []
        self.state = None
        self.next_state = None
        self.pitch_reward = None
        self.yaw_reward = None
        self.pitch_action = None
        self.yaw_action = None

        #reset flush queues 
        self.flush_imu = 0
        self.flush_detection = 0
        self.flush_diver = 0

        #reset counters
        self.duration = 0
        self.empty_state_counter = 0

        #reset end conditions 
        self.finished = False
        self.complete = False

        #reset diver pose
        self.diver_pose = None

        return

def main(args=None):
    rclpy.init(args=args)

    node = pid_controller()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

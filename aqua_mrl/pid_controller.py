import rclpy
import os
import numpy as np 
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import SetBool
from aqua2_interfaces.srv import SetFloat
from time import sleep
from aqua_mrl.control.PID import PID
from aqua_mrl.helpers import reward_calculation, normalize_coords
from aqua_mrl import hyperparams

class pid_controller(Node):
    def __init__(self):
        super().__init__('pid_controller')

        #hyperparams
        self.queue_size = hyperparams.queue_size_
        self.history_size = hyperparams.history_size_
        self.yaw_gains = hyperparams.yaw_gains_
        self.pitch_gains = hyperparams.pitch_gains_
        self.thrust_gains = hyperparams.thrust_gains_
        self.img_size = hyperparams.img_size_
        self.train_for = hyperparams.train_for_
        self.eval_duration = hyperparams.eval_duration_
        self.location_sigma = hyperparams.location_sigma_
        self.area_sigma = hyperparams.area_sigma_
        self.frame_skip = hyperparams.frame_skip_
        self.empty_state_max = hyperparams.empty_state_max_
        self.target_area = hyperparams.target_area_
        self.initialize_debris_after = hyperparams.initialize_debris_after_
        self.experiment_name = 'baseline'

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
        
        #last location
        self.last_location = None
        
        #initialize pid controllers
        self.pitch_pid = PID(target = 0, gains = self.pitch_gains, reverse=True, command_range=[-hyperparams.pitch_limit_,hyperparams.pitch_limit_])
        self.yaw_pid = PID(target = 0, gains = self.yaw_gains, reverse=True, command_range=[-hyperparams.yaw_limit_,hyperparams.yaw_limit_])
        self.thrust_pid = PID(target = self.target_area, gains = self.thrust_gains, command_range=[hyperparams.min_speed_,hyperparams.max_speed_])

        #flush queues
        self.flush_steps = self.queue_size + 35
        self.flush_detection = 0

        #finished is episode has ended. complete is if aqua has reached the goal 
        self.finished = False
        self.complete = False

        #episode rewards
        self.episode_rewards = []
        self.target_locations = []

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

        self.save_path = os.path.join('src/aqua_mrl/pid_evaluations', self.experiment_name)
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        self.episode = len(os.listdir(self.save_path))

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
            if self.last_location is not None:
                yc, xc, a = self.last_location[0], self.last_location[1], self.last_location[2]
            else:
                return #diver has not been located yet
            yc, xc, a = self.last_location[0], self.last_location[1], self.last_location[2]
            dqn_state = [yc, xc, a, 0.0]
        else:
            self.empty_state_counter = 0
            yc = (coords[1] + coords[3])/2
            xc = (coords[0] + coords[2])/2
            a = (np.abs(coords[1] - coords[3]) * np.abs(coords[0] - coords[2]))/(self.img_size*self.img_size)
            yc, xc = normalize_coords(yc, xc, self.img_size, self.img_size)
            dqn_state = [yc, xc, a, 1.0]   
        
        self.last_location = dqn_state

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
                    
        reward = reward_calculation(dqn_state[0], dqn_state[1], dqn_state[2], dqn_state[3], self.location_sigma, self.area_sigma, self.target_area)
        self.episode_rewards.append(reward)
        self.target_locations.append(dqn_state)

        #publish actions
        pitch_action = self.pitch_pid.control(dqn_state[0])
        yaw_action = self.yaw_pid.control(dqn_state[1])
        speed_action =  self.thrust_pid.control(dqn_state[2])
        self.command.data = [pitch_action, yaw_action, speed_action]
        self.command_publisher.publish(self.command)
        return 
    
    def finish(self):
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
        
        #reset rewards
        self.episode_rewards = []
        self.target_locations = []

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

        #reset flush queues 
        self.flush_detection = 0

        #reset counters
        self.duration = 0
        self.empty_state_counter = 0

        #last location
        self.last_location = None

        #reset end conditions 
        self.finished = False
        self.complete = False

        #increment episode 
        self.episode += 1

        print('-------------- Completed Reset --------------')
        print(self.episode)

        return
    
def main(args=None):
    rclpy.init(args=args)

    node = pid_controller()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
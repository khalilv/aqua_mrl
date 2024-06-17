import rclpy
import torch
import numpy as np 
import cv2
import os
from pynput import keyboard
from rclpy.node import Node
from aqua2_interfaces.msg import Command, AquaPose
from std_msgs.msg import UInt8MultiArray, Float32
from aqua_mrl.control.PID import AnglePID, PID
from aqua_mrl.control.DQN import ReplayMemory
from aqua_mrl.helpers import reward_calculation, euler_from_quaternion
from aqua_mrl import hyperparams
from sensor_msgs.msg import Imu, CompressedImage
from nav_msgs.msg import Odometry

class manual_controller(Node):
    def __init__(self):
        super().__init__('manual_controller')
        #hyperparams
        self.queue_size = hyperparams.queue_size_
        self.roll_gains = hyperparams.roll_gains_
        self.pitch_limit = hyperparams.pitch_limit_
        self.yaw_limit = hyperparams.yaw_limit_
        self.yaw_action_space = hyperparams.yaw_action_space_
        self.pitch_action_space = hyperparams.pitch_action_space_
        self.roll_pid = AnglePID(target = 0.0, gains = self.roll_gains, reverse=True)

        #subscribers and publishers
        self.command_publisher = self.create_publisher(Command, hyperparams.command_topic_name_, self.queue_size)
        self.camera_subscriber = self.create_subscription(
            CompressedImage,
            hyperparams.camera_topic_name_,
            self.camera_callback,
            self.queue_size)
        self.imu_subscriber = self.create_subscription(AquaPose, hyperparams.imu_topic_name_, self.imu_callback, self.queue_size)        
        self.measured_roll_angle = 0.0
        
        #dqn controller for yaw and pitch 
        self.yaw_actions = np.linspace(-self.yaw_limit, self.yaw_limit, self.yaw_action_space)
        self.pitch_actions = np.linspace(-self.pitch_limit, self.pitch_limit, self.pitch_action_space)

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

        #online dataset collection
        self.dataset_path = 'src/aqua_mrl/diver_dataset/'
        self.dataset_size = len(os.listdir(self.dataset_path))
        self.save_probability = 0.1

        cv2.namedWindow("Original", cv2.WINDOW_AUTOSIZE)

        print('Initialized: manual controller ')
  
    def get_command(self, key):
        try:
            if key.char in ['a', 'd', 'i', 'm']:
                self.recieved_command = True

            if key.char == 'a': # yaw right
                self.yaw_action_idx = 0
            elif key.char == 'd': # yaw left
                self.yaw_action_idx = 2
            elif key.char == 'i': #pitch up
                self.pitch_action_idx = 0
            elif key.char == 'm': #pitch down
                self.pitch_action_idx = 2
        except AttributeError:
            return
        return 

    def imu_callback(self, imu):
        self.measured_roll_angle = self.calculate_roll(imu)
        return

    def calculate_roll(self, imu):
        return imu.roll
    
    def camera_callback(self, msg):

        img = np.fromstring(msg.data.tobytes(), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        
        #save image with probability
        if np.random.rand() < self.save_probability:
            cv2.imwrite(self.dataset_path + str(self.dataset_size) + '.jpg', img)
            self.dataset_size += 1

        cv2.imshow('Original', img)
        cv2.waitKey(1)
        if not self.recieved_command:
            self.pitch_action_idx = 1
            self.yaw_action_idx = 1
        self.command.pitch = self.pitch_actions[self.pitch_action_idx]
        self.command.yaw = self.yaw_actions[self.yaw_action_idx]            
        self.command.speed = hyperparams.speed_ #fixed speed
        self.command.roll = self.roll_pid.control(self.measured_roll_angle)
        self.command_publisher.publish(self.command)
        self.recieved_command = False
        return 


def main(args=None):
    rclpy.init(args=args)

    node = manual_controller()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

    



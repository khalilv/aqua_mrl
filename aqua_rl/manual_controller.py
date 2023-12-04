import rclpy
from rclpy.node import Node
from aqua2_interfaces.msg import Command, AquaPose
from aqua_rl.control.PID import AnglePID
from sensor_msgs.msg import CompressedImage
import os 
from pynput import keyboard
import numpy as np 
import cv2 

class manual_controller(Node):
    def __init__(self):
        super().__init__('manual_controller')
        self.command_publisher = self.create_publisher(Command, '/a13/command', 10)
        self.imu_subscriber = self.create_subscription(AquaPose, '/aqua/pose', self.imu_callback, 10)
        self.camera_subscriber = self.create_subscription(
            CompressedImage, 
            '/camera/back/image_raw/compressed', 
            self.camera_callback, 
            10)
        self.roll_pid = AnglePID(target = 0.0, gains = [2.75, 0.0, 3.75], reverse=True)
        self.pitch_pid = AnglePID(target = 0.0, gains = [0.5181, 0.0, 0.9])
        self.measured_roll_angle = 0.0
        self.measured_pitch_angle = 0.0
        self.dataset_path = 'src/aqua_rl/rope_dataset/'
        self.dataset_size = len(os.listdir(self.dataset_path))
        self.command = Command()
        self.command.speed = 0.25 #fixed speed
        self.command.roll = 0.0
        self.command.pitch = 0.0
        self.command.yaw = 0.0
        self.command.heave = 0.0
        self.listener = keyboard.Listener(on_press=self.get_command)
        self.listener.start()
        self.recieved_command = False
        self.save_img = False
        cv2.namedWindow("Downward Camera", cv2.WINDOW_AUTOSIZE)
        print('Initialized: manual controller ')
  
    def get_command(self, key):
        try:
            if key.char == 'a': # hard yaw left
                self.command.yaw = 0.75
                self.recieved_command = True
            elif key.char == 's': # soft yaw left
                self.command.yaw = 0.25
                self.recieved_command = True
            elif key.char == 'd': # soft yaw right
                self.command.yaw = -0.25
                self.recieved_command = True
            elif key.char == 'f': # hard yaw right
                self.command.yaw = -0.75
                self.recieved_command = True
            elif key.char == 'k':
                self.command.speed = np.clip(self.command.speed - 0.25, 0.25, 1.0)
            elif key.char == 'l':
                self.command.speed = np.clip(self.command.speed + 0.25, 0.25, 1.0)
        except AttributeError:
            if key == keyboard.Key.up: # heave up
                self.command.heave = 0.25
                self.recieved_command = True
            elif key == keyboard.Key.down: # heave down
                self.command.heave = -0.25
                self.recieved_command = True
            elif key == keyboard.Key.space: # save image
                self.save_img = True
        return 

        
    def imu_callback(self, imu):
        self.measured_roll_angle = self.calculate_roll(imu)
        self.measured_pitch_angle = self.calculate_pitch(imu)
        self.command.roll = self.roll_pid.control(self.measured_roll_angle)
        self.command.pitch = self.pitch_pid.control(self.measured_pitch_angle)
        if self.recieved_command:
            self.recieved_command = False
        else:
            self.command.yaw = 0.0
            self.command.heave = 0.0
        self.command_publisher.publish(self.command)
        return

    def calculate_roll(self, imu):
        return imu.roll
    
    def calculate_pitch(self, imu):
        return imu.pitch
    
    def camera_callback(self, msg):
        img = self.load_img(msg)
        cv2.imshow("Downward Camera", img)
        key_ = cv2.waitKey(1)
        if self.save_img:
            cv2.imwrite(self.dataset_path + str(self.dataset_size) + '.jpg', img)
            print('Saved Image. Dataset Size: ' + str(self.dataset_size))
            self.dataset_size += 1
            self.save_img = False

    def load_img(self, compressed_img):
        img = np.fromstring(compressed_img.data.tobytes(), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR) 
        return img

def main(args=None):
    rclpy.init(args=args)

    node = manual_controller()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

    



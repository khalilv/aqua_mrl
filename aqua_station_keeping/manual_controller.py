import rclpy
from rclpy.node import Node
from aqua2_interfaces.msg import Command
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np 

class manual_controller(Node):
    def __init__(self):
        super().__init__('manual_controller')
        self.command_publisher = self.create_publisher(Command, '/a13/command', 10)
        self.subscriber = self.create_subscription(
            CompressedImage, 
            '/camera/back/image_raw/compressed', 
            self.camera_callback, 
            10)
        cv2.namedWindow("Downward Camera", cv2.WINDOW_AUTOSIZE)
        print('Initialized: manual controller ')

    def camera_callback(self, msg):
        img = self.load_img(msg)
        cv2.imshow("Downward Camera", img)
        key_ = cv2.waitKey(1)
        command = self.get_command_from_keypress(key_)
        self.command_publisher.publish(command)
        print(command)

    def get_command_from_keypress(self, key):
        command = Command()
        command.speed = 0.25
        command.roll = 0.0
        command.yaw = 0.0
        command.heave = 0.0
        command.pitch = 0.0
        if key == ord('a'): # left
            command.yaw = 0.5
        if key == ord('d'): # right
            command.yaw = -0.5
        elif key == ord('s'): # down
            command.pitch = 0.2
        elif key == ord('w'): # up
            command.pitch = -0.2
        return command

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

    

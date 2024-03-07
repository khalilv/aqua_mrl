import rclpy
import cv2
import cv_bridge
from sensor_msgs.msg import CompressedImage
from rclpy.node import Node
from aqua_rl.DeepLabv3.deeplabv3 import DeepLabv3
from aqua_rl import hyperparams 
import numpy as np
import time
from std_msgs.msg import UInt8MultiArray, Float32
import os
from aqua_rl import hyperparams
from aqua_rl.helpers import reward_calculation

class detect(Node):

    def __init__(self):
        super().__init__('detect')
        self.queue_size = hyperparams.queue_size_

        self.camera_subscriber = self.create_subscription(
            CompressedImage,
            hyperparams.camera_topic_name_,
            self.camera_callback,
            self.queue_size)
        self.coords_publisher = self.create_publisher(UInt8MultiArray, '/detected_coordinates', self.queue_size)
        self.coords = UInt8MultiArray()
        self.cv_bridge = cv_bridge.CvBridge()

        #online dataset collection
        self.dataset_path = 'src/aqua_rl/diver_dataset/'
        self.dataset_size = len(os.listdir(self.dataset_path))
        self.save_probability = 0.0

        #measuring publish frequency
        self.t0 = 0

        self.coord = None

        cv2.namedWindow("Original", cv2.WINDOW_AUTOSIZE)
        
        print('Initialized: detection module')

    def camera_callback(self, msg):

        img = np.fromstring(msg.data.tobytes(), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        
        #save image with probability
        if np.random.rand() < self.save_probability:
            cv2.imwrite(self.dataset_path + str(self.dataset_size) + '.jpg', img)
            self.dataset_size += 1

        cv2.imshow('Original', img)
        cv2.waitKey(1)

        t1 = time.time()
        print('Publishing Frequency: ', (t1 - self.t0))
        self.t0 = t1
        return
        
def main(args=None):
    rclpy.init(args=args)
    subscriber = detect()

    rclpy.spin(subscriber)

    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':   
    main()

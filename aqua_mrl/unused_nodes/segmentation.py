import rclpy
import cv2
import cv_bridge
from sensor_msgs.msg import CompressedImage
from rclpy.node import Node
from aqua_mrl.DeepLabv3.deeplabv3 import DeepLabv3
from aqua_mrl import hyperparams 
import numpy as np
import time
from std_msgs.msg import UInt8MultiArray, Float32
import os
from aqua_mrl import hyperparams
from aqua_mrl.helpers import reward_calculation

class segmentation(Node):

    def __init__(self):
        super().__init__('segmentation')
        self.queue_size = hyperparams.queue_size_
        self.img_size = hyperparams.img_size_
        self.display_original = hyperparams.display_original_

        self.camera_subscriber = self.create_subscription(
            CompressedImage,
            hyperparams.camera_topic_name_,
            self.camera_callback,
            self.queue_size)
        self.segmentation_publisher = self.create_publisher(UInt8MultiArray, '/segmentation', self.queue_size)
        self.seg_map = UInt8MultiArray()
        self.cv_bridge = cv_bridge.CvBridge()
        self.model = DeepLabv3('src/aqua_mrl/segmentation_module/models/deeplabv3_mobilenetv3_ropev2/best.pt')
        
        #online dataset collection
        self.dataset_path = 'src/aqua_mrl/rope_dataset/'
        self.dataset_size = len(os.listdir(self.dataset_path))
        self.save_probability = 0.0

        #measuring publish frequency
        self.t0 = 0

        self.reward_publisher = self.create_publisher(Float32, '/reward', self.queue_size)
        self.reward = Float32()


        cv2.namedWindow("Segmentation Mask", cv2.WINDOW_AUTOSIZE)
        
        if self.display_original:
            cv2.namedWindow("Original", cv2.WINDOW_AUTOSIZE)

        print('Initialized: segmentation module')

    def camera_callback(self, msg):

        img = np.fromstring(msg.data.tobytes(), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        
        #save image with probability
        if np.random.rand() < self.save_probability:
            cv2.imwrite(self.dataset_path + str(self.dataset_size) + '.jpg', img)
            self.dataset_size += 1

        #segment image
        pred = self.model.segment(img)
        pred = pred.astype(np.uint8)
        pred = cv2.resize(pred, self.img_size)

        #self.reward.data = reward_calculation(pred, 0.0, hyperparams.roi_detection_threshold_, hyperparams.mean_importance_)
        #self.reward_publisher.publish(self.reward)

        #publish state
        self.seg_map.data = pred.flatten().tolist()
        self.segmentation_publisher.publish(self.seg_map)

        cv2.imshow('Segmentation Mask', pred * 255)
        cv2.waitKey(1)

        if self.display_original:
            cv2.imshow('Original', img)
            cv2.waitKey(1)
            
        t1 = time.time()
        print('Publishing Frequency: ', (t1 - self.t0))
        self.t0 = t1
        return
        
def main(args=None):
    rclpy.init(args=args)
    subscriber = segmentation()

    rclpy.spin(subscriber)

    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':   
    main()

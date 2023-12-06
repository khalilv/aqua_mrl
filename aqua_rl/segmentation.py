import rclpy
import cv2
import cv_bridge
from sensor_msgs.msg import CompressedImage
from rclpy.node import Node
import numpy as np
from aqua_rl.DeepLabv3.deeplabv3 import DeepLabv3
import time
from std_msgs.msg import UInt8MultiArray, Float32
import os

class segmentation(Node):

    def __init__(self):
        super().__init__('segmentation')
        self.queue_size = 5
        self.camera_subscriber = self.create_subscription(
            CompressedImage,
            '/camera/back/image_raw/compressed',
            self.camera_callback,
            self.queue_size)
        self.segmentation_publisher = self.create_publisher(UInt8MultiArray, '/segmentation', self.queue_size)
        self.reward_publisher = self.create_publisher(Float32, '/a13/reward', self.queue_size)

        self.seg_map = UInt8MultiArray()
        self.cv_bridge = cv_bridge.CvBridge()
        self.model = DeepLabv3('src/aqua_rl/segmentation_module/models/deeplabv3_mobilenetv3_ropev2/best.pt')
        self.img_size = (32, 32)
        
        #online dataset collection
        self.dataset_path = 'src/aqua_rl/rope_dataset/'
        self.dataset_size = len(os.listdir(self.dataset_path))
        self.save_probability = 0.0

        #measuring publish frequency
        self.t0 = 0

        cv2.namedWindow("Segmentation Mask", cv2.WINDOW_AUTOSIZE)
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

        # self.publish_reward(pred)

        #publish state
        self.seg_map.data = pred.flatten().tolist()
        self.segmentation_publisher.publish(self.seg_map)

        cv2.imshow('Segmentation Mask', pred * 255)
        cv2.waitKey(1)
        
        t1 = time.time()
        print('Publishing Frequency: ', (t1 - self.t0))
        self.t0 = t1
        return
    
    def publish_reward(self, seg_map):
        #target for reward
        template = np.zeros(self.img_size)
        half = int(self.img_size[0]/2)
        template[:,half-2:half+2] = 1
        template = self.template.astype(np.uint8)

        # Calculate intersection and union
        intersection = np.logical_and(seg_map, template)
        union = np.logical_or(seg_map, template)
        iou = np.sum(intersection) / np.sum(union)
        
        reward = Float32()
        reward.data = iou - 0.025
        self.reward_publisher.publish(reward)
        return
        
def main(args=None):
    rclpy.init(args=args)
    subscriber = segmentation()

    rclpy.spin(subscriber)

    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':   
    main()

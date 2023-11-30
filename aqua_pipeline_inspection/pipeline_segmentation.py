import rclpy
import cv2
import cv_bridge
from sensor_msgs.msg import CompressedImage
from rclpy.node import Node
import numpy as np
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from aqua_pipeline_inspection.DeepLabv3.deeplabv3 import DeepLabv3
import time
from std_msgs.msg import UInt8MultiArray
import os

class pipeline_segmentation(Node):

    def __init__(self):
        super().__init__('pipeline_segmentation')
        self.queue_size = 5
        self.camera_subscriber = self.create_subscription(
            CompressedImage,
            '/camera/back/image_raw/compressed',
            self.camera_callback,
            self.queue_size)
        self.segmentation_publisher = self.create_publisher(UInt8MultiArray, '/pipeline/segmentation', self.queue_size)
        self.seg_map = UInt8MultiArray()
        self.cv_bridge = cv_bridge.CvBridge()
        self.model = DeepLabv3('src/aqua_pipeline_inspection/pipeline_segmentation/models/deeplabv3_mobilenetv3_rope/best.pt')
        self.img_size = (32, 32)
        
        #online dataset collection
        self.dataset_path = 'src/aqua_pipeline_inspection/rope_dataset/'
        self.dataset_size = len(os.listdir(self.dataset_path))
        self.save_probability = 0.0

        cv2.namedWindow("Pipeline Detection", cv2.WINDOW_AUTOSIZE)
        print('Initialized: pipeline_segmentation')

    def camera_callback(self, msg):
        t0 = time.time()

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

        #publish state
        self.seg_map.data = pred.flatten().tolist()
        self.segmentation_publisher.publish(self.seg_map)

        cv2.imshow('Pipeline Detection', pred * 255)
        cv2.waitKey(1)
        
        t1 = time.time()
        print('Processing time: ', (t1 - t0))
        return
   
    def load_model(self, n_classes):
        model = models.segmentation.deeplabv3_resnet101(
            pretrained=True, progress=True)
        # freeze weights
        for param in model.parameters():
            param.requires_grad = False
        # replace classifier
        model.classifier = DeepLabHead(2048, num_classes=n_classes)
        return model

def main(args=None):
    rclpy.init(args=args)
    subscriber = pipeline_segmentation()

    rclpy.spin(subscriber)

    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':   
    main()

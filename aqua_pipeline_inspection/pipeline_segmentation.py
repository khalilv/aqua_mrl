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

class pipeline_segmentation(Node):

    def __init__(self):
        super().__init__('pipeline_segmentation')
        self.camera_subscriber = self.create_subscription(
            CompressedImage,
            '/camera/back/image_raw/compressed',
            self.camera_callback,
            10)
        self.cv_bridge = cv_bridge.CvBridge()
        self.model = DeepLabv3('src/aqua_pipeline_inspection/pipeline_segmentation/models/deeplabv3_mobilenetv3/best.pt')
        cv2.namedWindow("Pipeline", cv2.WINDOW_AUTOSIZE)
        print('Initialized: pipeline_segmentation')

    def camera_callback(self, msg):
        t0 = time.time()
        img = np.fromstring(msg.data.tobytes(), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        pred = self.model.segment(img)
        t1 = time.time()
        print('Processing time: ', (t1 - t0))

        #display mask and original image
        pred = np.stack((pred * 255,)*3, axis=-1).astype(np.uint8)
        concat = np.concatenate((pred, img), axis=1)
        cv2.imshow('Pipeline', concat)
        cv2.waitKey(1)
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

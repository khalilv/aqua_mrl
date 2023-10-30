import rclpy
import cv2
import cv_bridge
from aqua_pipeline_inspection.YOLOv7.yolov7 import YoloV7
from sensor_msgs.msg import CompressedImage
from rclpy.node import Node
from argparse import Namespace
import numpy as np

class yolo_detector(Node):

    def __init__(self):
        super().__init__('yolo_detector')
        self.subscriber = self.create_subscription(
            CompressedImage, 
            '/camera/back/image_raw/compressed', 
            self.camera_callback, 
            10)
        self.cv_bridge = cv_bridge.CvBridge()
        image_size = 416
        self.model = YoloV7(Namespace(half=False,
                                        confidence_threshold=0.25, 
                                        iou_threshold = 0.45,
                                        weights = 'src/aqua_pipeline_inspection/aqua_pipeline_inspection/YOLOv7/weights/treasure.pt',
                                        image_size = image_size, 
                                        trace = True, 
                                        verbose = False))
        
        cv2.namedWindow("Downward Camera", cv2.WINDOW_AUTOSIZE)
        print('Initialized: yolo_detector')

    def camera_callback(self, msg):
        img = np.fromstring(msg.data.tobytes(), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        _, image_with_detections = self.model.detect(img)
        cv2.imshow("Downward Camera", image_with_detections)
        key_ = cv2.waitKey(1)
        return

def main(args=None):
    rclpy.init(args=args)

    subscriber = yolo_detector()

    rclpy.spin(subscriber)

    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

   
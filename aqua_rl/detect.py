import rclpy
import cv2
import cv_bridge
from sensor_msgs.msg import CompressedImage
from rclpy.node import Node
from aqua_rl import hyperparams 
import numpy as np
import time
from std_msgs.msg import Float32MultiArray
import os
from aqua_rl import hyperparams
from aqua_rl.YOLOv7.yolov7 import YoloV7
from argparse import Namespace
from std_srvs.srv import SetBool


class detect(Node):

    def __init__(self):
        super().__init__('detect')
        self.queue_size = hyperparams.queue_size_
        self.img_size = hyperparams.img_size_
        self.debris_range = hyperparams.debris_range_

        self.camera_subscriber = self.create_subscription(
            CompressedImage,
            hyperparams.camera_topic_name_,
            self.camera_callback,
            self.queue_size)
        self.debris_service = self.create_service(
            SetBool,
            hyperparams.debris_srv_name_,
            self.debris_callback)
        self.coords_publisher = self.create_publisher(Float32MultiArray, hyperparams.detection_topic_name_, self.queue_size)
        self.coords = Float32MultiArray()
        self.cv_bridge = cv_bridge.CvBridge()
        self.model = YoloV7(Namespace(half=False,
                            confidence_threshold=0.7, 
                            iou_threshold = 0.1,
                            weights = 'src/aqua_rl/aqua_rl/YOLOv7/weights/diver_v2.pt',
                            image_size = self.img_size, 
                            trace = True, 
                            verbose = False,
                            track=True,
                            min_hits=3,
                            max_age=2))
        
        #online dataset collection
        self.dataset_path = 'src/aqua_rl/diver_dataset/'
        self.dataset_size = len(os.listdir(self.dataset_path))
        self.save_probability = 0.0

        #measuring publish frequency
        self.t0 = 0

        self.coord = None
        self.debris_exists = False
        cv2.namedWindow("yolov7", cv2.WINDOW_AUTOSIZE)
        
        print('Initialized: detection module')

    def camera_callback(self, msg):

        img = np.fromstring(msg.data.tobytes(), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        #save image with probability
        if np.random.rand() < self.save_probability:
            cv2.imwrite(self.dataset_path + str(self.dataset_size) + '.jpg', img)
            self.dataset_size += 1

        outputs, image_with_detections = self.model.detect(img)
        
        cv2.imshow("yolov7", image_with_detections)
        cv2.waitKey(1)

        if len(outputs) > 0:
            self.coord = max(outputs, key=lambda x: x[4])
            xc = (self.coord[0]+self.coord[2])/2
            yc = (self.coord[1]+self.coord[3])/2
            #remove detections in debris range
            if self.debris_exists and xc > self.debris_range[0] and xc < self.debris_range[1] and yc > self.debris_range[0] and yc < self.debris_range[1]:
                self.coord =  [-1, -1, -1, -1]
        else:
            self.coord = [-1, -1, -1, -1]

        self.coords.data = [float(self.coord[0]), float(self.coord[1]), float(self.coord[2]), float(self.coord[3])]
        self.coords_publisher.publish(self.coords)

        t1 = time.time()
        print('Publishing Frequency: ', (t1 - self.t0))
        self.t0 = t1
        return
    
    def debris_callback(self, request, response):
        if request.data:
            self.debris_exists = True
            response.message = 'Debris initialized'
        else:
            self.debris_exists = False
            response.message = 'Debris removed'
        response.success = True
        return response

        
        
def main(args=None):
    rclpy.init(args=args)
    subscriber = detect()

    rclpy.spin(subscriber)

    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':   
    main()

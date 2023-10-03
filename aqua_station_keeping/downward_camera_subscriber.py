import rclpy
import numpy as np
import cv2
import cv_bridge
import time
import torch
from aqua_station_keeping.core.raft import RAFT
from aqua_station_keeping.core.utils import flow_viz
from aqua_station_keeping.core.utils.utils import InputPadder
from sensor_msgs.msg import CompressedImage
from rclpy.node import Node
from argparse import Namespace


class downward_camera_subscriber(Node):

    def __init__(self):
        super().__init__('downward_camera_subscriber')
        self.subscriber = self.create_subscription(
            CompressedImage, 
            '/camera/back/image_raw/compressed', 
            self.camera_callback, 
            10)
        self.cv_bridge = cv_bridge.CvBridge()
        self.last_img = None
        self.last_time =time.time()
        self.device = 'cuda'
        self.checkpoint = 'src/aqua_station_keeping/aqua_station_keeping/models/raft-small.pth'
        self.small = True
        self.mixed_precision = False
        self.model = torch.nn.DataParallel(RAFT(Namespace(small=self.small,
                                                          mixed_precision=self.mixed_precision
                                                          )))
        self.model.load_state_dict(torch.load(self.checkpoint))
        self.model = self.model.module
        self.model.to(self.device)
        self.model.eval()

        self.img_h, self.img_w = 304,400
        self.ystep, self.xstep = 50,50
        self.yseg, self.xseg = int(np.floor(self.img_h/self.ystep)), int(np.floor(self.img_w/self.xstep))

        cv2.namedWindow("Downward Camera", cv2.WINDOW_AUTOSIZE)
        print('Initialized: downward camera subscriber ')

    def camera_callback(self, msg):
        img = self.load_img(msg)
        if self.last_img is not None:
            avg_flow = self.get_optical_flow(img, self.last_img)
            self.display(img,avg_flow) 
        self.last_img = img
        print("Processing time: ", time.time() - self.last_time)
        self.last_time = time.time()
        
    def load_img(self, compressed_img):
        img = np.fromstring(compressed_img.data.tobytes(), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = torch.from_numpy(img).permute(2, 0, 1).float()    
        img = img[None].to(self.device)  
        return img

    def get_optical_flow(self, current_img, last_img):
        with torch.no_grad():
            padder = InputPadder(current_img.shape)
            img1, img2 = padder.pad(current_img, last_img)
            flow_low,flow_up = self.model(img1, img2, iters=20, test_mode=True)
            flow = flow_up[0].permute(1,2,0).cpu().numpy()
            avg_flo = np.zeros((self.yseg,self.xseg,2))
            for y in range(self.yseg):
                for x in range(self.xseg):
                    avg_flo[y,x,:] = np.mean(flow[y*self.ystep:(y+1)*self.ystep,
                                                  x*self.xstep:(x+1)*self.xstep], axis=(0,1))
            return avg_flo
        
    def display(self, img, avg_flow):
        scale_factor = 20
        img = img[0].permute(1,2,0).cpu().numpy()
        for y in range(self.yseg):
            for x in range(self.xseg):
                x_start = (2*x + 1)*self.xstep/2
                y_start = (2*y + 1)*self.ystep/2
                x_end = x_start + avg_flow[y,x,0]*scale_factor
                y_end = y_start + avg_flow[y,x,1]*scale_factor
                cv2.arrowedLine(img, (int(x_start), int(y_start)), (int(x_end), int(y_end)),
                                            (0,0,0), 2)
        
        cv2.imshow("Downward Camera", img / 255.0)
        key_ = cv2.waitKey(1)

        return



def main(args=None):
    rclpy.init(args=args)

    subscriber = downward_camera_subscriber()

    rclpy.spin(subscriber)

    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

   
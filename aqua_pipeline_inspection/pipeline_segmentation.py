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
from std_msgs.msg import Float32
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise


class pipeline_segmentation(Node):

    def __init__(self):
        super().__init__('pipeline_segmentation')
        self.camera_subscriber = self.create_subscription(
            CompressedImage,
            '/camera/back/image_raw/compressed',
            self.camera_callback,
            10)
        self.command_publisher = self.create_publisher(Float32, '/pipeline/error', 10)
        self.command = Float32()
        self.cv_bridge = cv_bridge.CvBridge()
        self.img_size = (300,400)
        self.model = DeepLabv3('src/aqua_pipeline_inspection/pipeline_segmentation/models/deeplabv3_mobilenetv3/best.pt')
        self.current_error = 0

        #init kalman filter for tracking
        self.kalman = KalmanFilter(dim_x=2, dim_z=1)
        self.kalman.x = np.zeros(2) #position and velocity. init both to 0
        self.kalman.F = np.array([[1.,1.],[0.,1.]]) #state transition matrix
        self.kalman.H = np.array([[1.,0.]]) #measurement function. only measure position
        self.kalman.P *= np.array([[100.,0.],[0., 1000.]]) #covariance matrix give high uncertainty to unobservable initial velocities
        self.kalman.R = 20 #measurement noise in px
        self.kalman.Q = Q_discrete_white_noise(dim=2, dt=0.02, var=0.13)
        
        cv2.namedWindow("Pipeline Detection", cv2.WINDOW_AUTOSIZE)
        print('Initialized: pipeline_segmentation')

    def camera_callback(self, msg):
        t0 = time.time()

        img = np.fromstring(msg.data.tobytes(), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        #segment image
        pred = self.model.segment(img)
        pred = pred.astype(np.uint8) * 255
        
        #ransac on mask
        try:
            r,theta = self.ransac(pred, tau = 30, iters = 20)
        except AssertionError:
            print('No pipeline detected')
            self.command.data = float(self.img_size[0] + self.img_size[1] + 1) #send stop command as w + h + 1 
            self.command_publisher.publish(self.command)
            return

        #calculate error
        errors = self.line_to_error(r, theta)

        #kalman filter predict
        self.kalman.predict()

        #kalman filter update with measurement 
        if np.abs((errors[0] - self.current_error)) < np.abs((errors[1] - self.current_error)):
            self.kalman.update(errors[0])
        else:
            self.kalman.update(errors[1])

        #publish error/command
        self.current_error = self.kalman.x[0] 
        self.command.data = self.current_error
        self.command_publisher.publish(self.command)

        #display waypoint
        waypoint = self.error_to_boundary_point(self.current_error)       
        cv2.circle(pred, (waypoint[0], waypoint[1]), 5, 0, 2)
        cv2.imshow('Pipeline Detection', pred)
        # cv2.imwrite('pipe.png', pred)
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

    def line_to_error(self, r, theta):
        w = self.img_size[1]
        h = self.img_size[0]
        errors = []
        if theta != 0:
            #right border of image
            y = (r - w*np.cos(theta))/np.sin(theta)
            if y > 0 and y < h:
                errors.append(int(w/2 + y))

            #left border of image
            y = r/np.sin(theta)
            if y > 0 and y < h:
                errors.append(int(-w/2 - y))

        #bottom border of image
        x = (r - h*np.sin(theta))/np.cos(theta)
        if x > 0 and x < w:
            if x < w/2:
                errors.append(int(-w/2 - h - x))
            else:
                errors.append(int(w/2 + h + (w - x)))
        
        #top border of image
        x = r/np.cos(theta)
        if x > 0 and x < w:
            errors.append(int(x - w/2))

        return errors
    
    def error_to_boundary_point(self, error):
        w = self.img_size[1]
        h = self.img_size[0]
        if error > -w/2 and error < w/2: #top
            return [int(error + w/2),0]
        elif error > w/2 and error < w/2 + h: #right
            return [w, int(error - w/2)]
        elif error < -w/2 and error > -w/2 - h: #left
            return [0, int(-error - w/2)]
        elif error > 0: #bottom
            return [int(h + w/2 + w - error), h]
        else:
            return [int(-w/2 - h - error), h]
        
    def ransac(self, mask, tau, iters):
        argmask = np.argwhere(mask)
        assert len(argmask) > 2 # not enough points to fit a line
        max_vote = 0
        rho = None
        theta = None
        for _ in range(iters):
            point_indicies = np.random.choice(len(argmask), 2, False)
            p1 = argmask[point_indicies[0]]
            p2 = argmask[point_indicies[1]]
            t = np.arctan2((p1[1]-p2[1]), (p2[0]-p1[0]))
            r = p1[1]*np.cos(t) + p1[0]*np.sin(t)
            dist_to_line = np.abs(argmask[:,1]*np.cos(t) + argmask[:,0]*np.sin(t) - r)
            len_cset = (dist_to_line < tau).sum()
            if len_cset > max_vote and len_cset > self.min_cset():
                max_vote = len_cset
                rho, theta = r , t
        assert rho is not None
        assert theta is not None
        return rho, theta
    
    #minimum size for consensus set 
    def min_cset(self):
        w = self.img_size[1]
        h = self.img_size[0]
        return (0.4*w)*(0.4*h)/2

def main(args=None):
    rclpy.init(args=args)
    subscriber = pipeline_segmentation()

    rclpy.spin(subscriber)

    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    # l1 = [0,0,0,5]
    # l2 = [0,0,1,-1]
    # t1 = np.arctan(np.inf)
    # t2 = np.arctan(np.divide(l2[3] - l2[1], l2[2] - l2[0]))
    # diff = np.abs(t1-t2)
    # print(np.min([diff, np.abs(np.pi - diff)]))    
    main()

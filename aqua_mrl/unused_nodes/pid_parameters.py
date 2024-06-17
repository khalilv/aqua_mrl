import rclpy
import cv2
import cv_bridge
import time
import numpy as np
from sensor_msgs.msg import CompressedImage
from rclpy.node import Node
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from aqua_mrl.DeepLabv3.deeplabv3 import DeepLabv3
from std_msgs.msg import Float32MultiArray
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from aqua_mrl import hyperparams


class pid_parameters(Node):

    def __init__(self):
        super().__init__('pid_parameters')
        
        self.queue_size = hyperparams.queue_size_
        self.img_size = hyperparams.img_size_

        self.camera_subscriber = self.create_subscription(
            CompressedImage,
            '/camera/back/image_raw/compressed',
            self.camera_callback,
            self.queue_size)
        self.parameters_publisher = self.create_publisher(Float32MultiArray, '/segmentation/parameters', self.queue_size)
        self.parameters = Float32MultiArray()
        self.cv_bridge = cv_bridge.CvBridge()
        self.model = DeepLabv3('src/aqua_mrl/segmentation_module/models/deeplabv3_mobilenetv3_ropev2/best.pt')
        self.current_error = 0.0

        #measuring publish frequency
        self.t0 = 0

        #init kalman filter for tracking waypoint
        self.use_kalman = True
        self.kalman = KalmanFilter(dim_x=2, dim_z=1)
        self.kalman.x = np.zeros(2) #position and velocity. init both to 0
        self.kalman.F = np.array([[1.,1.],[0.,1.]]) #state transition matrix
        self.kalman.H = np.array([[1.,0.]]) #measurement function. only measure position
        self.kalman.P *= np.array([[100.,0.],[0., 1000.]]) #covariance matrix give high uncertainty to unobservable initial velocities
        self.kalman.R = 1 #measurement noise in px
        self.kalman.Q = Q_discrete_white_noise(dim=2, dt=0.02, var=0.13)

        cv2.namedWindow("Segmentation Mask", cv2.WINDOW_AUTOSIZE)
        print('Initialized: PID parameters')

    def camera_callback(self, msg):

        img = np.fromstring(msg.data.tobytes(), np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        #segment image
        pred = self.model.segment(img)
        pred = pred.astype(np.uint8)
        pred = cv2.resize(pred, self.img_size)

        #ransac on mask
        try:
            r, theta, _ = self.ransac(pred, tau = 15, iters = 50)
        except AssertionError:
            print('Nothing detected')
            self.parameters.data = [-1., -1., -1., -1., -1.] #send stop command as all -1 
            self.parameters_publisher.publish(self.parameters)
            return
        
        #calculate error
        errors = self.line_to_error(r, theta)

        p1 = self.error_to_boundary_point(errors[0])
        p2 = self.error_to_boundary_point(errors[1])
        cx = (p1[0] + p2[0])/2
        cy = (p1[1] + p2[1])/2
        # length = np.sqrt(np.square(p1[0]-p2[0]) + np.square(p1[1]-p2[1]))
        # width = cset_size / length

        #kalman filter predict
        self.kalman.predict()

        #select waypoint from candidates. select higher point. 
        #in horizontal pipe case this may cause switching
        if np.abs(errors[0]) < np.abs(errors[1]):
            er = errors[0]
        else:
            er = errors[1]
        
        if self.use_kalman:
            self.kalman.update(er) #update kalman filter with measurement
            self.current_error = self.kalman.x[0] #read updated state   
        else:
            self.current_error = er
      
        #publish parameter information
        self.parameters.data = [self.current_error, r, theta, cx, cy]
        self.parameters_publisher.publish(self.parameters)

        #display waypoint and center
        waypoint = self.error_to_boundary_point(self.current_error)   
        pred = np.stack((pred * 255,)*3, axis=-1)    
        cv2.circle(pred, (waypoint[0], waypoint[1]), 2, (255,0,0), 2)
        cv2.circle(pred, (int(cx), int(cy)), 2, (0,0,255), 2)
        cv2.imshow('Segmentation Mask', pred)
        cv2.waitKey(1)
        
        t1 = time.time()
        print('Publishing Frequency: ', (t1 - self.t0))
        self.t0 = t1

        return
   
    def line_to_error(self, r, theta):
        w = self.img_size[1]
        h = self.img_size[0]
        errors = []
        if theta != 0:
            #right border of image
            y = (r - w*np.cos(theta))/np.sin(theta)
            if y > 0 and y <= h:
                errors.append(w/2 + y)

            #left border of image
            y = r/np.sin(theta)
            if y > 0 and y <= h:
                errors.append(-w/2 - y)

        #bottom border of image
        x = (r - h*np.sin(theta))/np.cos(theta)
        if x >= 0 and x <= w:
            if x < w/2:
                errors.append(-w/2 - h - x)
            else:
                errors.append(w/2 + h + (w - x))
        
        #top border of image
        x = r/np.cos(theta)
        if x >= 0 and x <= w:
            errors.append(x - w/2)

        return errors
    
    def error_to_boundary_point(self, error):
        w = self.img_size[1]
        h = self.img_size[0]
        if error >= -w/2 and error <= w/2: #top
            return [int(error + w/2),0]
        elif error > w/2 and error <= w/2 + h: #right
            return [w, int(error - w/2)]
        elif error < -w/2 and error >= -w/2 - h: #left
            return [0, int(-error - w/2)]
        elif error > 0: #bottom
            return [int(h + w/2 + w - error), h]
        else:
            return [int(-w/2 - h - error), h]
        
    def ransac(self, mask, tau, iters):
        argmask = np.argwhere(mask)
        assert len(argmask) > 2 # not enough points to fit a line
        max_cset = 0
        rho = None
        theta = None
        for _ in range(iters):
            point_indicies = np.random.choice(len(argmask), 2, False)
            p1 = argmask[point_indicies[0]]
            p2 = argmask[point_indicies[1]]
            r, t = self.fit_polar(p1, p2)
            dist_to_line = np.abs(argmask[:,1]*np.cos(t) + argmask[:,0]*np.sin(t) - r)
            len_cset = (dist_to_line < tau).sum()
            if len_cset > max_cset and len_cset > self.min_cset():
                max_cset = len_cset
                rho, theta = r , t
        assert rho is not None
        assert theta is not None
        return rho, theta, max_cset
    
    #minimum size for consensus set 
    def min_cset(self):
        #TODO
        return 0
    
    #fit a polar line through two cartesian points. theta in [-pi/2, pi/2]
    def fit_polar(self, p1, p2):
        delta_x = (p1[1]-p2[1])
        delta_y = (p2[0]-p1[0])
        if delta_y == 0:
            t = np.pi/2 * np.sign(delta_x)
        else:
            t = np.arctan((p1[1]-p2[1])/(p2[0]-p1[0]))
        r = p1[1]*np.cos(t) + p1[0]*np.sin(t)
        return r, t

def main(args=None):
    rclpy.init(args=args)
    subscriber = pid_parameters()

    rclpy.spin(subscriber)

    subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__': 
    main()
import rclpy
import numpy as np 
import torch
import os
from rclpy.node import Node
from aqua2_interfaces.msg import Command, AquaPose
from aqua_rl.control.PID import AnglePID, PID
from std_msgs.msg import UInt8MultiArray, Float32
from aqua_rl import hyperparams
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

class pid_controller(Node):
    def __init__(self):
        super().__init__('pid_controller')

        #hyperparams
        self.queue_size = hyperparams.queue_size_
        self.roll_gains = hyperparams.roll_gains_
        self.img_size = hyperparams.img_size_
        self.yaw_limit = hyperparams.yaw_limit_
        self.yaw_action_space = hyperparams.yaw_action_space_
        self.pitch_limit = hyperparams.pitch_limit_
        self.pitch_action_space = hyperparams.pitch_action_space_
        self.target_depth = hyperparams.target_depth_
        self.history_size = hyperparams.history_size_

        #subscribers and publishers
        self.command_publisher = self.create_publisher(Command, '/a13/command', self.queue_size)
        self.imu_subscriber = self.create_subscription(AquaPose, '/aqua/pose', self.imu_callback, self.queue_size)
        self.segmentation_subscriber = self.create_subscription(
            UInt8MultiArray, 
            '/segmentation', 
            self.segmentation_callback, 
            self.queue_size)
        self.depth_subscriber = self.create_subscription(Float32, '/aqua/depth', self.depth_callback, self.queue_size)

        #initialize pid controllers
        self.roll_pid = AnglePID(target = 0.0, gains = self.roll_gains, reverse=True)
        self.pitch_pid = PID(target = 0.0, gains = [0.005, 0.0, 0.175])
        self.yaw_pid = PID(target = 0.0, gains = [0.6, 0.0, 1.1], reverse=True, normalization_factor=64)
        self.measured_roll_angle = None
        self.relative_depth = None

        self.yaw_actions = np.linspace(-self.yaw_limit, self.yaw_limit, self.yaw_action_space)
        self.pitch_actions = np.linspace(-self.pitch_limit, self.pitch_limit, self.pitch_action_space)

        #initialize command
        self.command = Command()
        self.command.speed = 0.0 
        self.command.roll = 0.0
        self.command.pitch = 0.0
        self.command.yaw = 0.0
        self.command.heave = 0.0

        #state and depth history
        self.image_history = []
        self.depth_history = []
        self.state = None
        self.state_depths = None
        self.action = None

        self.save_expert_data = True
        self.expert_dataset_path = 'src/aqua_rl/pid_expert/'
        if not os.path.exists(self.expert_dataset_path):
            os.mkdir(self.expert_dataset_path)
            os.mkdir(os.path.join(self.expert_dataset_path, 'states'))
            os.mkdir(os.path.join(self.expert_dataset_path, 'depths'))
            os.mkdir(os.path.join(self.expert_dataset_path, 'actions'))
            self.num_samples = 0
        else:
            self.num_samples = len(os.listdir(os.path.join(self.expert_dataset_path, 'states/')))
            print('Samples collected: ', self.num_samples)

        #init kalman filter for tracking waypoint
        self.use_kalman = True
        self.kalman = KalmanFilter(dim_x=2, dim_z=1)
        self.kalman.x = np.zeros(2) #position and velocity. init both to 0
        self.kalman.F = np.array([[1.,1.],[0.,1.]]) #state transition matrix
        self.kalman.H = np.array([[1.,0.]]) #measurement function. only measure position
        self.kalman.P *= np.array([[100.,0.],[0., 1000.]]) #covariance matrix give high uncertainty to unobservable initial velocities
        self.kalman.R = 1 #measurement noise in px
        self.kalman.Q = Q_discrete_white_noise(dim=2, dt=0.02, var=0.13)
        self.current_error = 0.0

        self.saved = False
                             
        print('Initialized: PID controller')
  
    def imu_callback(self, imu):
        self.measured_roll_angle = self.calculate_roll(imu)
        return
            
    def depth_callback(self, depth):
        self.relative_depth = self.target_depth + depth.data
        return

    def calculate_roll(self, imu):
        return imu.roll
    
    
    def segmentation_callback(self, seg_map):
        
        #return if depth or roll angle has not been measured
        if self.relative_depth is None or self.measured_roll_angle is None:
            return
        
        seg_map = np.array(seg_map.data).reshape(self.img_size)
        if len(self.image_history) < self.history_size:
            self.image_history.append(seg_map)
            self.depth_history.append(self.relative_depth)
        else:
            self.image_history.pop(0)
            self.image_history.append(seg_map)
            self.depth_history.pop(0)
            self.depth_history.append(self.relative_depth)

            #ransac on mask
            try:
                r, theta, _ = self.ransac(seg_map, tau = 15, iters = 50)
            except AssertionError:
                print('Nothing detected')
                return
            
            #calculate error
            errors = self.line_to_error(r, theta)

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
            
            self.state = np.array(self.image_history)
            self.state_depths = np.array(self.depth_history)
            self.yaw_action_idx = self.discretize(self.yaw_pid.control(self.current_error), self.yaw_actions)
            self.pitch_action_idx = self.discretize(self.pitch_pid.control(self.relative_depth), self.pitch_actions)
            self.action = int(self.pitch_action_idx*self.yaw_action_space) + self.yaw_action_idx
            if self.save_expert_data:
                with open(os.path.join(self.expert_dataset_path, 'states') + '/{}.npy'.format(str(self.num_samples).zfill(5)), 'wb') as s:
                    np.save(s, self.state)
                with open(os.path.join(self.expert_dataset_path, 'depths') + '/{}.npy'.format(str(self.num_samples).zfill(5)), 'wb') as d:
                    np.save(d, self.state_depths)
                with open(os.path.join(self.expert_dataset_path, 'actions') + '/{}.npy'.format(str(self.num_samples).zfill(5)), 'wb') as a:
                    np.save(a, self.action)
                self.num_samples += 1
                
            self.command.speed = 0.25 #fixed speed
            self.command.yaw = self.yaw_actions[self.yaw_action_idx]
            self.command.pitch = self.pitch_actions[self.pitch_action_idx]
            self.command.roll = self.roll_pid.control(self.measured_roll_angle)
            self.command.heave = 0.0
            self.command_publisher.publish(self.command)

        return

    def create_grid(self, seg_map, n, thresh):
        block_size = (seg_map.shape[0] // n, seg_map.shape[1] // n)
        # reshape into a 5x5 grid and sum pizels
        grid = seg_map[:n * block_size[0], :n * block_size[1]].reshape(n, block_size[0], n, block_size[1]).sum(axis=(1, 3))
        #define threshold number of pixels
        n_thresh = thresh * block_size[0] * block_size[1]

        # populate the grid based on the threshold
        grid = (grid > n_thresh).astype(int)

        #error = (5 - np.sum(grid[:,2])) * (0.1*((np.sum(grid[:,0:2])) - (np.sum(grid[:,3:5]))))
        return grid
    
    def discretize(self, v, l):
        index = np.argmin(np.abs(np.subtract(l,v)))
        return index
    
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

    node = pid_controller()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

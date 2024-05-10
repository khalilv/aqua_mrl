import rclpy
import numpy as np
from rclpy.node import Node
from aqua2_interfaces.msg import DiverCommand, AquaPose
from aqua2_interfaces.srv import SetInt, SetFloat
from aqua_rl import hyperparams
from std_srvs.srv import SetBool
from ir_aquasim_interfaces.srv import SetPosition
from geometry_msgs.msg import Pose

class diver_controller(Node):

    def __init__(self):
        super().__init__('diver_controller')

        #hyperparams
        self.queue_size = hyperparams.queue_size_
        self.depth_range = hyperparams.depth_range_
        self.max_speed = hyperparams.diver_max_speed_

        #command publisher
        self.publisher = self.create_publisher(DiverCommand, hyperparams.diver_command_topic_name_, self.queue_size)
        self.pose_subscriber = self.create_subscription(
            AquaPose,
            hyperparams.diver_pose_topic_name_,
            self.diver_pose_callback,
            self.queue_size)
        self.start_stop_service = self.create_service(
            SetBool,
            hyperparams.diver_start_stop_,
            self.start_stop_callback)
        self.seed_service = self.create_service(
            SetInt,
            hyperparams.diver_seed_srv_name_,
            self.seed_callback)
        self.speed_service = self.create_service(
            SetFloat,
            hyperparams.diver_speed_srv_name_,
            self.speed_callback)
        self.reset_diver_client = self.create_client(SetPosition, hyperparams.diver_reset_srv_name_)

        #command
        self.diver_cmd = DiverCommand()
        self.diver_pose = None
        
        #reset diver request
        self.reset_diver_req = SetPosition.Request()
        
        #starting diver position and orientation
        self.starting_pose = Pose()
        self.starting_pose.position.x = 62.5
        self.starting_pose.position.z = 0.3                               
        self.starting_pose.position.y = -10.0
        self.starting_pose.orientation.x = 0.4976952
        self.starting_pose.orientation.y = -0.5022942
        self.starting_pose.orientation.z = 0.4976952
        self.starting_pose.orientation.w = 0.5022942

        #flag to move diver
        self.move_diver = False
        timer_period = 5  # seconds
        self.timer = self.create_timer(timer_period, self.publish_diver_command)       
        print('Initialized: diver controller')

    def publish_diver_command(self):  
        #generate velocity
        self.diver_cmd.vx = np.random.uniform(0.5, 0.775)
        self.diver_cmd.vy = np.random.uniform(-self.max_speed, self.max_speed)
        self.diver_cmd.vz = np.random.uniform(-self.max_speed, self.max_speed)
        return
    
    def diver_pose_callback(self, pose):        
        self.diver_pose = [pose.x, pose.y, pose.z]
        if self.diver_pose is not None and self.move_diver:
            
            #change velocity if outside depth range           
            if self.diver_pose[1] > self.depth_range[0] and self.diver_cmd.vy > 0:
                self.diver_cmd.vy *= -1
            elif self.diver_pose[1] < self.depth_range[1] and self.diver_cmd.vy < 0:
                self.diver_cmd.vy *= -1
            #publish
            self.publisher.publish(self.diver_cmd)
        return 
    
    def start_stop_callback(self, request, response):
        if request.data:
            self.move_diver = True
            response.message = 'Diver controller started'
        else:
            self.move_diver = False
            self.reset_diver_req.pose = self.starting_pose
            self.reset_diver_client.call_async(self.reset_diver_req)
            response.message = 'Diver controller stopped and reset'
        response.success = True
        return response
    
    def seed_callback(self, request, response):
        print('Setting seed to {}'.format(int(request.value)))
        np.random.seed(int(request.value))
        response.msg = 'Set seed successfully'
        return response
    
    def speed_callback(self, request, response):
        print('Setting max speed to {}'.format(request.value))
        self.max_speed = request.value
        response.msg = 'Set speed successfully'
        return response
    
def main(args=None):
    rclpy.init(args=args)

    publisher = diver_controller()

    rclpy.spin(publisher)

    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()